import cv2
import os
import numpy
import random

# 入力動画ファイルのパス
input_video_path = "../norfair/demos/camera_motion/src/data_002.mp4"

# 出力動画ファイルのパス
output_video_path = "track_output_video.mp4"

# テキストファイルが格納されているディレクトリのパス
text_files_directory = "../norfair/demos/camera_motion/src/output_data_frames/"

# OpenCVを使用して入力動画を読み込み
input_video = cv2.VideoCapture(input_video_path)

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力フォーマットを指定
fps = int(input_video.get(cv2.CAP_PROP_FPS))  # 元動画のフレームレートを取得
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力動画の準備
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_counter = 0
id_colors = {}  # IDごとの色を保持する辞書

while True:
    # 動画からフレームを読み込む
    ret, frame = input_video.read()
    if not ret:
        break  # 動画の最後に達したらループを終了

    frame_counter += 1

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

                print(f"{id},{x1},{y1},{x2},{y2}")

                # IDに割り当てられた色を取得
                color = id_colors[id]

                # バウンディングボックスを動画フレームに描画
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # 緑色の矩形

                # IDを動画フレームに描画
                cv2.putText(frame, f'ID: {id}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 描画されたフレームを出力動画に書き込む
    output_video.write(frame)

# 動画ファイルを解放
input_video.release()
output_video.release()
cv2.destroyAllWindows()

print("動画処理が完了しました。")
