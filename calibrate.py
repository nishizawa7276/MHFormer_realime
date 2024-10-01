# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import v4l2capture
import select
import datetime
import time
from time import sleep

SQUARE_SIZE=34.07     #チェスボードの１つの四角の大きさ(mm)


# チェスボードのサイズ
CHECKERBOARD = (10,7) #チェスボードの四角の数
BOARD_SIZE=(10,7)    #チェスボードの四角の数

WIDTH = 640  # 横サイズ
HEIGHT = 480  # 縦サイズ
DEVICE="/dev/video0" #カメラデバイス

CAMERA_FILE="camera.csv" #出力ファイル
DIST_FILE="dist.csv"     #出力ファイル名

MAX_COUNT=15            #画像取得数

# メイン関数
def main():
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    #チェスボードを設定
    pattern_points = np.zeros( (np.prod(BOARD_SIZE), 3), np.float32 ) 
    pattern_points[:,:2] = np.indices(BOARD_SIZE).T.reshape(-1, 2)
    pattern_points *= SQUARE_SIZE
    obj_points = []
    img_points = []
    
    # チェスボードの3D座標を作成
    # objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    # objp[:,:2] = np.mgrid[0:CHECKERBOARD_SIZE[0],0:CHECKERBOARD_SIZE[1]].T.reshape(-1,2)

    # 3D座標と2D座標を格納するリスト
    obj_points = [] # 3D座標
    img_points = [] # 2D座標

    # ビデオを開く
    video = v4l2capture.Video_device(DEVICE)
    sizeX, sizeY = video.set_format(WIDTH, HEIGHT, fourcc='MJPG')    
    print("解像度={0}x{1}".format(sizeX, sizeY))

    #バッファを設定
    video.create_buffers(3)
    video.queue_all_buffers()

    #取り込みスタート
    video.start()

    #カメラが落ち着くまで待つ
    time.sleep(5)
    print("{0}枚の画像を撮影します。cキーで撮影。qキーで終了。".format(MAX_COUNT))
    cnt=0

    while(cnt<MAX_COUNT):
        # カメラから画像を取得
        select.select((video,), (), ())
        image_data = video.read_and_queue()
        img = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        #画面をリサイズして表示
        resized_img = cv2.resize(img,(800, 600))
        cv2.imshow('frame', resized_img)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            #グレースケールに変換
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corner = cv2.findChessboardCorners(img_gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            #チェスボードが写っているか判定
            # found, corner = cv2.findChessboardCorners(img_gray, BOARD_SIZE)
            if found:
                cnt=cnt+1
                print("チェスボードを検出={0}/{1}".format(cnt,MAX_COUNT))
                #チェスボードを計算
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                corner = cv2.cornerSubPix(img_gray, corner,(3,3),(-1,-1),subpix_criteria)
                img2 = cv2.drawChessboardCorners(img_gray, BOARD_SIZE, corner, found)
                #チェスボード情報を保存
                objpoints.append(objp)
                # obj_points.append(pattern_points.astype(np.float32))  # ここで3D座標をfloat32型に変換する
                # obj_points.append(np.expand_dims(np.asarray(objp), -2) )  # ここで3D座標をfloat32型に変換する
                # obj_points_expanded = np.expand_dims(np.asarray(objp), -2)
                # img_points.append(corner.reshape(-1, 2))
                imgpoints.append(corner)
                img_ok=img_gray

                # ファイル名
                output_filename = f"drawn_chessboard_corners_{cnt}.jpg"
                
                # 画像を保存
                cv2.imwrite(output_filename, img2)
            else:
                print("チェスボードを非検出" )
                
        elif key == ord('q'): # qで終了                                                               
            break

    # カメラの歪みを補正する
    print('img end')
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            img_gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("rms " + str(rms) )
    print("DIM=" + str(img_gray[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")

    # prams = {'DIM':DIM,
    #         'K':K,
    #         'D':D }

    # カメラの歪みパラメータを計算
    # rms, k, d, r, t = cv2.calibrateCamera(obj_points, img_points, (img_ok.shape[1],img_ok.shape[0]), None, None)

    # # 計算結果を表示
    # print ("　---normal----　")
    # print ("RMS = ", rms)
    # print ("K = \n", k)
    # print ("d = ", d )
    # print ("d = ", d.ravel())

    # #値をファイルに保存
    # np.savetxt(CAMERA_FILE, k, delimiter =',',fmt="%0.14f")
    # np.savetxt(DIST_FILE, d, delimiter =',',fmt="%0.14f")

    # # カメラの歪みを補正する
    # rms, mtx, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
    #     obj_points, img_points, (img_ok.shape[1],img_ok.shape[0]), None, None
    # )
    # # 計算結果を表示
    # print ("　---fisheye----　")
    # print ("RMS = ", rms)
    # print ("K = \n", mtx)
    # print ("dist_coeffs = ", dist_coeffs )

    # #値をファイルに保存
    # np.savetxt(CAMERA_FILE, mtx, delimiter =',',fmt="%0.14f")
    # np.savetxt(DIST_FILE, dist_coeffs, delimiter =',',fmt="%0.14f")
            
    #カメラとウィンドウを閉じる
    video.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


""""
Found 30 valid images for calibration
rms 0.4946494497638323
K=np.array([[804.5635732638252, 0.0, 956.7186956564307], [0.0, 804.9555134338189, 592.2364273725946], [0.0, 0.0, 1.0]])
D=np.array([[-0.017804139726756717], [-0.004034739990270839], [-0.0023401196032088144], [0.0005942396667011329]])



"""