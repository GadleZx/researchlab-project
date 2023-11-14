# http://www.paul-reed.co.uk/programming.html
# https://paulbourke.net/panorama/icosahedral/
# https://github.com/rdbch/icosahedral_sampler

import os
import random
import copy
import sys
sys.path.append('.')
import math
import argparse

import numpy as np
import open3d as o3d
from PIL import Image

import numpy as np
import cv2
from dictionary import AttrDict

from equirectangular2topview_V2 import ground2image, image2ground_base
from common import camera2xy

def process_video(frame_step, render_step):
    # 入力動画ファイルのパス
    input_video_path = "data/data_002.mp4"

    # 出力動画ファイルのパス
    output_video_path = "topview_track_output_video.mp4"

    # テキストファイルが格納されているディレクトリのパス
    text_files_directory = "data/tracking_data_002/"

    # OpenCVを使用して入力動画を読み込み
    input_video = cv2.VideoCapture(input_video_path)

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力フォーマットを指定
    fps = int(input_video.get(cv2.CAP_PROP_FPS))  # 元動画のフレームレートを取得

    config = dict()
    config['x'] = 12#11
    config['y'] = 0#-6
    config['z'] = -20
    config['sensor_height'] = 1.7
    config['Wmin'] = -10.0
    config['Wmax'] = 10.0
    config['Hmin'] = -10.0
    config['Hmax'] = 10.0
    config['img_w'] = 128 * 2
    config['img_h'] = 128 * 2
    config['dome_radius'] = 15
    config['save'] = False
    config = AttrDict(config)
    # NOTE: Rotation is in the order ZXY (Z+ up, X+ right, Y+ front)
    #       Pan, Tilt, Roll
    sensor_orientation_deg = np.array([config.x, config.y, config.z])
    # Source image
    image_size_source = None
    # sensor camera position
    sensor_position = np.array([0, 0, config.sensor_height]) # meters

    frame_width = config.img_w
    frame_height =  config.img_h

    # 出力動画の準備
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


    frame_counter = 0
    id_colors = {}  # IDごとの色を保持する辞書

    while True:
        # 動画からフレームを読み込む
        ret, frame = input_video.read()
        if not ret:
            break  # 動画の最後に達したらループを終了
        if image_size_source is None:
            image_size_source = [frame.shape[1], frame.shape[0]]
        #if frame_counter > 250:
        #    cv2.imwrite("frame.jpg", frame)
        #    exit(0)

        frame_counter += 1
        if frame_counter % frame_step != 0:
            continue

        pc, img = ground2image(frame, render_step, 'uv', config)

        # バウンディングボックスの情報をリセット
        bbox_file_path = os.path.join(text_files_directory, f'data_{frame_counter}.txt')

        # テキストファイルが存在し、空でない場合に処理を行う
        if os.path.exists(bbox_file_path) and os.path.getsize(bbox_file_path) > 0:
            # バウンディングボックスの情報を読み取ります
            with open(bbox_file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    data = line.split()

                    id = int(data[0])
                    if id not in id_colors:
                        id_colors[id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                    # 文字列をクリーニングしてから浮動小数点数に変換
                    x1, y1, x2, y2 = map(float, [s.strip('[,]') for s in data[1:]])

                    print(f"{id},{x1},{y1},{x2},{y2},{image_size_source}")

                    p3d_sphere_t, intersection_point = image2ground_base((x1 + x2) / 2, y2, image_size_source, 
                                                                         sensor_position, sensor_orientation_deg, config.dome_radius)                    
                    p2d = None
                    if intersection_point is not None:
                        print(f'intersection_point:{intersection_point}')
                        p2d = camera2xy(intersection_point, config.img_w, config.img_h, config.Wmin, config.Wmax, config.Hmin, config.Hmax)
                        #print('image2ground_base')
                        #print((x1 + x2) / 2, y2, image_size_source, sensor_position, sensor_orientation_deg, config.dome_radius)
                        #print(f'p2d:{p2d} intersection_point:{intersection_point}')
                        #exit(0)
                    # is invalid point?
                    if p2d is None:
                        continue

                    # IDに割り当てられた色を取得
                    color = id_colors[id]

                    cv2.circle(img, (int(p2d[0]), int(p2d[1])), 5, color, 2)
                    cv2.putText(img, f'ID: {id}', (int(p2d[0]), int(p2d[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # バウンディングボックスを動画フレームに描画
                    #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # 緑色の矩形

                    # IDを動画フレームに描画
                    #cv2.putText(frame, f'ID: {id}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:
            break

        # 描画されたフレームを出力動画に書き込む
        output_video.write(img)

    # 動画ファイルを解放
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

    print("動画処理が完了しました。")


if __name__ == "__main__":

    print(f'If colors and data points size mismatch, no give color is used!!!')

    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--frame_step", default=10, type=int, help="Int frame step. >1 skip frames.")
    parser.add_argument("--render_step", default=10, type=int, help="Int point cloud step. Higher is the value, sparser the points.")

    args = parser.parse_args()
    frame_step = args.frame_step
    render_step = args.render_step

    process_video(frame_step=frame_step, render_step=render_step)
