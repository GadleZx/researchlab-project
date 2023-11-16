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
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

import numpy as np
import cv2
from Open3D.dictionary import AttrDict

def load_label_danger(path):
    ''' Read a file with the danger interval 
    '''
    intervals = []
    labels = []
    with open(path) as file:
        for line in file:
            #print(line.rstrip())    
            words = line.split(' ')
            for i in range(0, len(words), 3):
                frame_in = int(words[i])
                frame_out = int(words[i+1])
                label = words[i+2]
                intervals.append(frame_in)
                intervals.append(frame_out)
                labels.append(label)
    return intervals, labels

def which_label(num, intervals, labels):
    for i in range(0, len(intervals), 2):
        if num >= intervals[i] and num <= intervals[i+1]:
            label = labels[int(i / 2)]
            return label
    return None

def load_localizationXY(path):
    ''' Read a file with the localization on the XY plane 
    '''
    res = dict()

    with open(path) as file:
        for line in file:
            #print(line.rstrip())    
            words = line.split(' ')
            num_frame = int(words[0])
            id = int(words[1])
            X = float(words[2])
            Y = float(words[3])
            Z = float(words[4])
            if num_frame not in res:
                res[num_frame] = dict()
            res[num_frame][id] = np.array([X, Y, Z])
    return res


class TrackedPosition:
    def __init__(self):
        self.container = []

    def update(self, t, position):
        self.container.append([t, np.array(position)])

    def direction_vector(self, interval):
        ''' num frame interval
        '''
        t_now = self.container[-1][0]
        idx = -1
        for i in range(len(self.container) - 1, -1, -1):
            t = self.container[i][0]
            if t_now - t > interval:
                idx = i
                break
        if idx >= 0:
            # difference between two points
            difference_vector = self.container[-1][1] - self.container[idx][1]
            # Normalize the difference vector to get the direction vector
            direction_vector = difference_vector / np.linalg.norm(difference_vector)
            return direction_vector
        return None

def danger_level_to_string(level):
    if level == 0:
        return 'Safe'
    elif level == 1:
        return 'Potential Collision'
    elif level == 2:
        return 'Danger'
    return 'Unknown'

def estimate_danger_level(distance, direction, distance_threshold, direction_threshold, distance_direction_threshold):
    if distance < distance_threshold:
        print(f'distance:{distance} distance_threshold:{distance_threshold}')
        return 2
    if direction is not None and direction > direction_threshold and distance < distance_direction_threshold:
        print(f'distance:{distance} distance_threshold:{distance_threshold} direction:{direction} direction_threshold:{direction_threshold} distance_direction_threshold:{distance_direction_threshold}')
        return 1
    return 0

def process_video(input_video_path, output_video_path, tracking_files_directory, localizationXY_path, label_path, frame_step):
    '''
    args
    input_video_path: # 入力動画ファイルのパス
    output_video_path: # 出力動画ファイルのパス
    tracking_files_directory: # テキストファイルが格納されているディレクトリのパス
    do_image2birdview: If true, it process the image source (VERY slow process)
    '''
    # OpenCVを使用して入力動画を読み込み
    input_video = cv2.VideoCapture(input_video_path)

    # Container with the current localization on XY plane
    loc_XY = load_localizationXY(localizationXY_path)
    #print(loc_XY)

    # Labels
    intervals, labels = load_label_danger(label_path)
    print(intervals)
    print(labels)
    label = which_label(225, intervals, labels)

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力フォーマットを指定
    fps = int(input_video.get(cv2.CAP_PROP_FPS))  # 元動画のフレームレートを取得

    config = dict()
    config['x'] = 12#11 # in degrees
    config['y'] = 0#-6 # in degrees
    config['z'] = -20 # in degrees
    config['sensor_height'] = 1.7 # in meters
    config['Wmin'] = -10.0 # in meters
    config['Wmax'] = 10.0 # in meters
    config['Hmin'] = -10.0 # in meters
    config['Hmax'] = 10.0 # in meters
    config['img_w'] = 128 * 2 # in pixels
    config['img_h'] = 128 * 2 # in pixels
    config['dome_radius'] = 15 # in meters
    config['save'] = False
    config['interval'] = 15
    config['distance_threshold'] = 3.0
    config['direction_threshold'] = 0.7
    config['distance_direction_threshold'] = 6.0
    config = AttrDict(config)

    # 出力動画の準備
    output_video = None
    do_save_video = True if len(output_video_path) > 0 else False

    frame_counter = 0
    id_colors = {}  # IDごとの色を保持する辞書
    opencv_wait_s = 1

    # Container with all the tracked position
    container_tracked_position = dict()

    while True:
        # 動画からフレームを読み込む
        ret, frame = input_video.read()
        if not ret:
            break  # 動画の最後に達したらループを終了

        if do_save_video and output_video is None:
            output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

        frame_counter += 1
        if frame_counter % frame_step != 0:
            continue

        # Danger level for the current frame
        # Lower, safer
        danger_level = 0

        # バウンディングボックスの情報をリセット
        bbox_file_path = os.path.join(tracking_files_directory, f'data_{frame_counter}.txt')

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
                    # IDに割り当てられた色を取得
                    color = id_colors[id]

                    # 文字列をクリーニングしてから浮動小数点数に変換
                    x1, y1, x2, y2 = map(float, [s.strip('[,]') for s in data[1:]])

                    # バウンディングボックスを動画フレームに描画
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # 緑色の矩形

                    # IDを動画フレームに描画
                    cv2.putText(frame, f'ID: {id}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Check if the measurement in XY plane does exist
                    if frame_counter in loc_XY:
                        if id in loc_XY[frame_counter]:
                            dist = math.sqrt(loc_XY[frame_counter][id][0]**2 + loc_XY[frame_counter][id][1]**2)
                            formatted_string = f"Pos(m):[{loc_XY[frame_counter][id][0]:.2f} {loc_XY[frame_counter][id][1]:.2f}]"
                            cv2.putText(frame, formatted_string, (int(x1), int(y2 - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, f'D(m){dist}', (int(x1), int(y2 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # add the information to the tracked position
                            if id not in container_tracked_position:
                                container_tracked_position[id] = TrackedPosition()
                            container_tracked_position[id].update(frame_counter, loc_XY[frame_counter][id])

                            # compute if moving toward the camera
                            # The direction is computed as dot product between two points [p_i, p_(i-n)], 
                            # and the last point with the origin [p_i, 0].
                            # If the dot product is >= threshold, it means that the direction is the same
                            # and the vectors are pointing to the origin.
                            #
                            # Eq. d_v_cam = dot( (p_i - p_(i-n))/(|p_i - p_(i-n)|), -p_i/|p_i|)
                            #
                            direction_vector = container_tracked_position[id].direction_vector(config.interval)
                            dot_product_toward_camera = None
                            if direction_vector is not None:
                                #formatted_string = f"[{direction_vector[0]:.2f} {direction_vector[1]:.2f}]"
                                #cv2.putText(frame, formatted_string, (int(x1), int(y2 - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                origin_vector = -loc_XY[frame_counter][id] / np.linalg.norm(-loc_XY[frame_counter][id])
                                dot_product_toward_camera = np.dot(direction_vector, origin_vector)
                                #print(f'direction_vector:{direction_vector} origin_vector:{origin_vector} dot_product:{dot_product_toward_camera}')
                                formatted_string = f"dv(m^2):[{dot_product_toward_camera:.2f}]"
                                cv2.putText(frame, formatted_string, (int(x1), int(y2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            print(f'id:{id}')
                            danger_level_tmp = estimate_danger_level(dist, dot_product_toward_camera, 
                                            config.distance_threshold, config.direction_threshold, 
                                            config.distance_direction_threshold)
                            if danger_level_tmp > danger_level:
                                danger_level = danger_level_tmp


        # Which label
        label = which_label(frame_counter, intervals, labels)
        color_header = (0, 0, 255)
        if label is None: 
            label = 'Safe'
            color_header = (0, 255, 0)
        # Frame number
        cv2.putText(frame, f'#:{frame_counter} label expected:{label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_header, 2)
        if danger_level == 0: color_header = (0, 255, 0)
        if danger_level == 1: color_header = (255, 255, 0)
        if danger_level == 2: color_header = (0, 0, 255)
        cv2.putText(frame, f'label estimated:{danger_level_to_string(danger_level)}', (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_header, 2)

        cv2.imshow('img', frame)
        k =cv2.waitKey(opencv_wait_s)
        if k == 48: # 0
            opencv_wait_s = 0 
        if k == 49: # 1
            opencv_wait_s = 1 
        if k == 27:
            break

        # 描画されたフレームを出力動画に書き込む
        if do_save_video:
            output_video.write(frame)

    # 動画ファイルを解放
    input_video.release()
    if do_save_video:
        output_video.release()
    cv2.destroyAllWindows()

    print("動画処理が完了しました。")

# No save
# python run/DangerPredictionV1.py --input_video_path='data/data_002.mp4' --output_video_path='' --localizationXY_path='data/localizationXY_cam/data_002_locXY.txt' --tracking_files_directory='data/tracking_data_002/' --label_path='data/danger_data_002.txt' --frame_step=1
# Save with the image
# python run/DangerPredictionV1.py --input_video_path='data/data_002.mp4' --output_video_path='danger_output_video.mp4' --localizationXY_path='data/localizationXY_cam/data_002_locXY.txt' --tracking_files_directory='data/tracking_data_002/' --label_path='data/danger_data_002.txt' --frame_step=1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--input_video_path", default='data/data_002.mp4', type=str, help="Video source.")
    parser.add_argument("--output_video_path", default='danger_output_video.mp4', type=str, help="Output video.")
    parser.add_argument("--tracking_files_directory", default='data/tracking_data_002/', type=str, help="Location with the tracking files.")
    parser.add_argument("--localizationXY_path", default='data/localizationXY_cam', type=str, help="Localization in XY space respect the camera.")
    parser.add_argument("--label_path", default='data/danger_data_002.txt', type=str, help="Labels associated to this video.")
    parser.add_argument("--frame_step", default=10, type=int, help="Int frame step. >1 skip frames.")

    args = parser.parse_args()
    print(f'args:{args}')
    process_video(input_video_path=args.input_video_path, output_video_path=args.output_video_path,
                  tracking_files_directory=args.tracking_files_directory, localizationXY_path=args.localizationXY_path,
                  label_path=args.label_path, frame_step=args.frame_step)