import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts_realtime import gen_video_kpts as hrnet_pose
from lib.hrnet.gen_kpts_realtime import set_model as set_model
from lib.hrnet.gen_kpts_realtime import get_keypoints as get_keypoints
from lib.yolov3 import preprocess
from lib.yolov3.util import write_results
from lib.hrnet.lib.utils.utilitys import PreProcess
from lib.sort.sort import Sort

import os 
import numpy as np
import torch
import glob, time, copy
from tqdm import tqdm
from IPython import embed

sys.path.append(os.getcwd())
from model.mhformer import Model
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show2Dpose(kps, img):
    # print("L34 kps" , kps)
    # print("L34 img" , img.shape )

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('equal') # works fine in matplotlib==2.2.2 or 3.7.1

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    with torch.no_grad():
        # the first frame of the video should be detected a person
        keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('Generating 2D pose successfully!')

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 351
    args.frames = 27
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/pretrained/27'
    args.n_joints, args.out_joints = 17, 17

    ## Reload 
    model = Model(args).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model of MHFormer in 'checkpoint/pretrained/351'
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)

    model.eval()

    ## input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ## 3D
    print('\nGenerating 3D pose...')
    output_3d_all = []
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        img_size = img.shape

        ## input frames
        start = max(0, i - args.pad)
        end =  min(i + args.pad, len(keypoints[0])-1)

        input_2D_no = keypoints[0][start:end+1]
        
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        print("L202 input_2D " , input_2D.shape )

        N = input_2D.size(0)

        ## estimation
        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip     = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D = output_3D[0:, args.pad].unsqueeze(1) 
        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()

        output_3d_all.append(post_out)

        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])

        input_2D_no = input_2D_no[args.pad]

        # ## 2D
        image = show2Dpose(input_2D_no, copy.deepcopy(img))
        print("L221 image" , image.shape )
        # イメージを表示
        cv2.imshow('Real-time Image', image)
        # 33ミリ秒待つ
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # output_dir_2D = output_dir +'pose2D/'
        # os.makedirs(output_dir_2D, exist_ok=True)
        # cv2.imwrite(output_dir_2D + str(('%04d'% i)) + '_2D.png', image)

        # ## 3D
        # fig = plt.figure( figsize=(9.6, 5.4))
        # gs = gridspec.GridSpec(1, 1)
        # gs.update(wspace=-0.00, hspace=0.05) 
        # ax = plt.subplot(gs[0], projection='3d')
        # show3Dpose( post_out, ax)

        # output_dir_3D = output_dir +'pose3D/'
        # os.makedirs(output_dir_3D, exist_ok=True)
        # plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
        
    ## save 3D keypoints
    output_3d_all = np.stack(output_3d_all, axis = 0)
    os.makedirs(output_dir + 'output_3D/', exist_ok=True)
    output_npz = output_dir + 'output_3D/' + 'output_keypoints_3d.npz'
    np.savez_compressed(output_npz, reconstruction=output_3d_all)

    print('Generating 3D pose successfully!')

    ## all
    image_dir = 'results/' 
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 102
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')


def calculate_angle_between_vectors(v1, v2):
    """2つのベクトルの間の角度（度単位）を計算します。"""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(dot_product) * (180.0 / np.pi)
    angle = round(angle, 1)
    return angle


if __name__ == "__main__":

    CUDA = torch.cuda.is_available()

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    people_sort = Sort(min_hits=0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'

    yolo_det , pose_model = set_model( )

    num_peroson=1

    #  get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 351
    args.frames = 81
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/pretrained/81' 
    args.n_joints, args.out_joints = 17, 17

    ## Reload 
    model = Model(args).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model of MHFormer in 'checkpoint/pretrained/351'
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)

    model.eval()

    first_dummy_pose = np.zeros([17,3])
    fig = plt.figure( figsize=(9.6, 5.4))
    # ax = fig.add_subplot(111, projection='3d')
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05) 
    ax = plt.subplot(gs[0], projection='3d')
    show3Dpose( first_dummy_pose, ax)

    # titles = ["1", "2," "3"]
    # values = [1, 2, 3]
    # plt.barh(titles, values, color='skyblue')

    # 開始時刻を記録
    start_time = time.time()
    # ループ回数をカウントする変数
    count_hz = 0

    # ポーズデータのプロット
    # x, y, z = first_dummy_pose[:, 0], first_dummy_pose[:, 1], first_dummy_pose[:, 2]
    # ax.scatter(x, y, z)
    # plt.ion()  # 対話モードをオンにする


    # 軸ラベルの設定titles = ["1", "2," "3"]
    # values = [1, 2, 3]
    # plt.barh(titles, values, color='skyblue')
    # ax.set_zlabel('Z Label')

    plt.show(block=False)
    # plt.pause(10)  # 画面の更新とプログラムの一時停止を許可
    print("L351 plt.show() ------------ ")

    # ウェブカメラのキャプチャを開始
    cap = cv2.VideoCapture( 2 )

    # カメラが正常にオープンしたか確認
    if not cap.isOpened():
        print("ウェブカメラを開けませんでした。")
    else:
        WIDTH = 640  # 横サイズ
        HEIGHT = 480  # 縦サイズ
        # キャプチャの解像度を設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    ii = 0

    K=np.array([[353.16616863887845, 0.0, 319.90224850370953], [0.0, 353.779176512706, 261.4935085374984], [0.0, 0.0, 1.0]])
    D=np.array([[-0.0050702857421404375], [-0.011451685736428985], [-0.13253789251202447], [0.314218818601909]])
    pre_L_Shoulder_H_angle = 0.0
    pre_L_Shoulder_V_angle = 0.0
    pre_L_Elbow_angle = 0.0 
    pre_speed_time = time.time()

    while True:
        cap_times = []
        prep_image_times = []
        yolo_times = []
        e2D_times = []
        e3D_times =[]
        cv2_times =[]
        plt_times =[]

        cap_start_time = time.time()
        # 1枚の写真をキャプチャ
        ret, frame = cap.read()
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (WIDTH, HEIGHT), cv2.CV_16SC2)
        undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        frame = undistorted_img
        img_size = frame.shape

        cap_times.append( time.time() - cap_start_time)

        prep_image_start_time = time.time()
        inp_dim = 416 
        img, ori_img, img_dim = preprocess.prep_image(frame, inp_dim)
        img_dim = torch.FloatTensor(img_dim).repeat(1, 2)
        prep_image_times.append( time.time() - prep_image_start_time)

        yolo_start_time = time.time()
        with torch.no_grad():
            if CUDA:
                img_dim = img_dim.cuda()
                img = img.cuda()
            output = yolo_det(img, CUDA)
            # print("L327 output" , output )
            confidence=0.70
            num_classes = 80
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=0.4, det_hm=True)
            if len(output) == 0:
                bboxs = None
                scores = None

            img_dim = img_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim / img_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * img_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * img_dim[:, 1].view(-1, 1)) / 2
            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dim[i, 1])

            bboxs = []
            scores = []
            for i in range(len(output)):
                item = output[i]
                bbox = item[1:5].cpu().numpy()
                # conver float32 to .2f data
                bbox = [round(i, 2) for i in list(bbox)]
                score = item[5].cpu().numpy()
                bboxs.append(bbox)
                scores.append(score)
            scores = np.expand_dims(np.array(scores), 1)
            bboxs = np.array(bboxs)

            if bboxs is None or not bboxs.any():
                print('No person detected!')
                bboxs = bboxs_pre
                scores = scores_pre
            else:
                bboxs_pre = copy.deepcopy(bboxs) 
                scores_pre = copy.deepcopy(scores) 

            # Using Sort to track people
            people_track = people_sort.update(bboxs)

            # Track the first two people in the video and remove the ID
            if people_track.shape[0] == 1:
                people_track_ = people_track[-1, :-1].reshape(1, 4)
            elif people_track.shape[0] >= 2:
                people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
                people_track_ = people_track_[::-1]
            else:
                continue

            track_bboxs = []
            for bbox in people_track_:
                bbox = [round(i, 2) for i in list(bbox)]
                track_bboxs.append(bbox)

            yolo_times.append( time.time() - yolo_start_time)

            e2D_start_time = time.time()
            keypoints, scores = get_keypoints(frame,  track_bboxs, num_peroson, pose_model) 
            keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
            # 形状 (1, 1, 17, 2) のテンソルから形状 (1, 17, 2) に変更
            keypoints = keypoints.squeeze(1)

            # print("L394 keypoints" , ii, " ", keypoints.shape )
            cal_flame_num = int(args.frames / 2) - 2
            # cal_flame_num = 0
            if ii == 0 :
                keypoints_temp = np.zeros([args.frames , 17 , 2])
                for count in range( args.frames ):
                    if count >= cal_flame_num :
                        keypoints_temp[count] = keypoints.copy()
            else:
                keypoints_temp[cal_flame_num+1:-1] = keypoints_temp[cal_flame_num:-2].copy()
                keypoints_temp[cal_flame_num] = keypoints.copy()

            joints_left =  [4, 5, 6, 11, 12, 13]
            joints_right = [1, 2, 3, 14, 15, 16]

            input_2D = normalize_screen_coordinates(keypoints_temp, w=img_size[1], h=img_size[0])  

            input_2D_aug = copy.deepcopy(input_2D)
            input_2D_aug[ :, :, 0] *= -1
            input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
            input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
            
            input_2D = input_2D[np.newaxis, :, :, :, :]

            input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

            # print("L202 input_2D " , input_2D.shape )

            N = input_2D.size(0)
            e2D_times.append( time.time() - e2D_start_time)

            e3D_start_time = time.time()
            ## estimation
            output_3D_non_flip = model(input_2D[:, 0])
            output_3D_flip     = model(input_2D[:, 1])

            output_3D_flip[:, :, :, 0] *= -1
            output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

            output_3D = (output_3D_non_flip + output_3D_flip) / 2

            output_3D = output_3D[0:, args.pad].unsqueeze(1) 
            output_3D[:, :, 0, :] = 0
            post_out = output_3D[0, 0].cpu().detach().numpy()


            rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])

            e3D_times.append( time.time() - e3D_start_time)

            cv2_start_time = time.time()
            # input_2D_no = keypoints_temp[args.pad]
            input_2D_no = keypoints_temp[args.pad]
            # print("L465 post_out" , ii, " ", post_out.shape , post_out )

            # ## 2D
            image = show2Dpose(input_2D_no, copy.deepcopy(frame))
            # print("L221 image" , image.shape )
            # イメージを表示
            cv2.imshow('Real-time Image', image)
            # 33ミリ秒待つ
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2_times.append( time.time() - cv2_start_time)

            plt_start_time = time.time()

            x, y, z = post_out[:, 0], post_out[:, 1], post_out[:, 2]
        
            names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'] # 各ポイントに付ける名前

            ax.cla()  # 軸をクリア
            show3Dpose( post_out, ax)
            # 各ポイントに名前を付ける
            for i, name in enumerate(names):
                ax.text(x[i], y[i], z[i], name, color='red')
            # ax.scatter(x, y, z)  # 新しいデータでプロットを更新
            
            # 軸ラベルの再設定（クリア後に必要）
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            # 視点の設定
            # ax.view_init(elev=0, azim=90)
            
            plt.draw()  # プロットを再描画
            plt.pause(0.01)  # 短い一時停止

            plt_times.append( time.time() - plt_start_time)

            # 現在の時刻を取得
            current_time = time.time()
            # ループ回数をインクリメント
            count_hz += 1
            if current_time - start_time > 5:
                print(" Runinng Hz ",   count_hz / (current_time - start_time) ) 
                start_time = current_time
                np_cap_times = np.array( cap_times )
                np_prep_image_times = np.array( prep_image_times )
                np_yolo_times = np.array( yolo_times  )
                np_e2D_times = np.array( e2D_times  )
                np_e3D_times = np.array( e3D_times )
                np_cv2_times = np.array( cv2_times )
                np_plt_times = np.array( plt_times )
                np_all_times = np_cap_times + np_prep_image_times + np_yolo_times + np_e2D_times \
                                + np_e3D_times + np_cv2_times + np_plt_times
                print(" all_times ",    np.mean(np_all_times) ) 
                print(" cap_times ",    np.mean(np_cap_times) ) 
                print(" prep_image_times ",    np.mean(np_prep_image_times) ) 
                print(" yolo_times ",    np.mean(np_yolo_times) ) 
                print(" e2D_times ",    np.mean(np_e2D_times) ) 
                print(" e3D_times ",    np.mean(np_e3D_times) ) 
                print(" cv2_times ",    np.mean(np_cv2_times) ) 
                print(" plt_times ",    np.mean(np_plt_times) ) 
                cap_times.clear()
                prep_image_times.clear()
                yolo_times.clear()
                e2D_times.clear()
                e3D_times.clear()
                cv2_times.clear()
                plt_times.clear()
                count_hz = 0
            ii += 1

            H_vector_L_Shoulder_to_R_Shoulder = post_out[14][:2] - post_out[11][:2]  # Z成分を無視
            H_vector_L_Shoulder_to_L_Elbow = post_out[12][:2] - post_out[11][:2]  # Z成分を無視
            V_vector_L_Shoulder_to_R_Shoulder = post_out[14][[0, 2]] - post_out[11][[0, 2]]  # Z成分を無視
            V_vector_L_Shoulder_to_L_Elbow = post_out[12][[0, 2]] - post_out[11][[0, 2]]  # Z成分を無視

            vector_L_Elbow_to_L_Shoulder = post_out[11] - post_out[12] 
            vector_L_Elbow_to_L_Hand= post_out[13] - post_out[12] 

            # 2つのベクトル間の水平方向のなす角を計算
            L_Shoulder_H_angle = calculate_angle_between_vectors(H_vector_L_Shoulder_to_R_Shoulder, H_vector_L_Shoulder_to_L_Elbow)
            L_Shoulder_V_angle = calculate_angle_between_vectors(V_vector_L_Shoulder_to_R_Shoulder, V_vector_L_Shoulder_to_L_Elbow)
            L_Elbow_angle = calculate_angle_between_vectors(vector_L_Elbow_to_L_Shoulder, vector_L_Elbow_to_L_Hand)
            # print( "L_Shoulder_H_angle:" , L_Shoulder_H_angle, "  L_Shoulder_V_angle:", L_Shoulder_V_angle, "  L_Elbow Angle:" , L_Elbow_angle)
            ii += 1

            H_vector_L_Shoulder_to_R_Shoulder = post_out[14][:2] - post_out[11][:2]  # Z成分を無視
            H_vector_L_Shoulder_to_L_Elbow = post_out[12][:2] - post_out[11][:2]  # Z成分を無視
            V_vector_L_Shoulder_to_R_Shoulder = post_out[14][[0, 2]] - post_out[11][[0, 2]]  # Z成分を無視
            V_vector_L_Shoulder_to_L_Elbow = post_out[12][[0, 2]] - post_out[11][[0, 2]]  # Z成分を無視

            vector_L_Elbow_to_L_Shoulder = post_out[11] - post_out[12] 
            vector_L_Elbow_to_L_Hand= post_out[13] - post_out[12] 

            # 2つのベクトル間の水平方向のなす角を計算
            L_Shoulder_H_angle = calculate_angle_between_vectors(H_vector_L_Shoulder_to_R_Shoulder, H_vector_L_Shoulder_to_L_Elbow)
            L_Shoulder_V_angle = calculate_angle_between_vectors(V_vector_L_Shoulder_to_R_Shoulder, V_vector_L_Shoulder_to_L_Elbow)
            L_Elbow_angle = calculate_angle_between_vectors(vector_L_Elbow_to_L_Shoulder, vector_L_Elbow_to_L_Hand)

            L_Elbow_angle = 0.7* pre_L_Elbow_angle + 0.3 * L_Elbow_angle
            L_Shoulder_H_angle = 0.7* pre_L_Shoulder_H_angle + 0.3 * L_Shoulder_H_angle
            L_Shoulder_V_angle = 0.7* pre_L_Shoulder_V_angle + 0.3 * L_Shoulder_V_angle

            speed_time =  time.time() - pre_speed_time
            pre_speed_time = time.time()

            L_Elbow_angle_peed = ( L_Elbow_angle - pre_L_Elbow_angle) / speed_time
            L_Shoulder_H_angle_speed = ( L_Shoulder_H_angle - pre_L_Shoulder_H_angle ) / speed_time
            L_Shoulder_V_angle_speed = ( L_Shoulder_V_angle - pre_L_Shoulder_V_angle ) / speed_time

            pre_L_Elbow_angle = L_Elbow_angle
            pre_L_Shoulder_H_angle = L_Shoulder_H_angle
            pre_L_Shoulder_V_angle = L_Shoulder_V_angle
            print(f"L_Elbow_angle: {L_Elbow_angle:.2f}  L_Shoulder_H_angle: {L_Shoulder_H_angle:.2f}  L_Shoulder_V_angle: {L_Shoulder_V_angle:.2f}")
            print(f"L_Elbow_angle_peed: {L_Elbow_angle_peed:.2f}  L_Shoulder_H_angle_speed: {L_Shoulder_H_angle_speed:.2f}  L_Elbow Angle: {L_Shoulder_V_angle_speed:.2f}")

            none_path = "/home/knishizawa/MHFormer_realime/image/none.JPG"
            Elbow_path = "/home/knishizawa/MHFormer_realime/image/l_elbow.JPG"
            Shoulder_path = "/home/knishizawa/MHFormer_realime/image/l_sh.JPG"
            if L_Elbow_angle_peed < -5 and  abs(L_Shoulder_H_angle_speed) < 10 :
                image2 = cv2.imread(Elbow_path)
            elif  L_Shoulder_H_angle_speed < -5 and abs(L_Elbow_angle_peed) < 10 :
                image2 = cv2.imread(Shoulder_path)
            else:
                image2 = cv2.imread(none_path)

            # print(image2.shape) (720, 960, 3)
            # width,height = image2.shape[0]
            width = int( 960/2 )
            height = int( 720/2 )
            # 画像をリサイズする
            image2 = cv2.resize(image2, (width, height))
            cv2.imshow("Image 2", image2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # print( "L_Shoulder_H_angle:" , round(L_Shoulder_H_angle, 2), "  L_Shoulder_V_angle:", round(L_Shoulder_V_angle, 2), "  L_Elbow Angle:" ,round(L_Elbow_Angle, 2))
        # print("L298 video_length" , video_length )


    print('Generating demo successful!')

""""
0:Bottom torso
1: L Hip
2:l Knee
3:l Foot
4:R Hip
5:R Knee
6:R Foot
7:Center torso
8:Upper torso
9: Neck Base
10: Center head
11:R Shoulder
12:R Elbow
13:R Hand
14:L Shoulder
15:L Elbow
16:L Hand
"""