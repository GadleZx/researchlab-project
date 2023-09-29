import cv2
import os

# 動画ファイルの読み込み
input_video = cv2.VideoCapture('../researchlab-project/data/data_002.mp4')

# 出力動画の設定
output_video_path = '/home/umelab3d/workspace/code/output_video_009.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力フォーマットを指定
fps = int(input_video.get(cv2.CAP_PROP_FPS))  # 元動画のフレームレートを取得
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力動画の準備
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 水平な直線の位置（画像上のY座標）
line_y = 700  # 画像の上から700ピクセルの位置に直線を描画

# 直線の太さを指定 (2ピクセル)
line_thickness = 2

# バウンディングボックスの情報が含まれているテキストファイルのディレクトリパス
bbox_dir = '../../ultralytics/code/runs/detect/predict4/labels'

while True:
    # 動画からフレームを読み込む
    ret, frame = input_video.read()
    if not ret:
        break  # 動画の最後に達したらループを終了

    # 各フレームごとにframe_colorを初期化
    frame_color = (0, 0, 255)  # 初期値は赤色

    # バウンディングボックスの情報をリセット
    change_frame_color = False

    # 各テキストファイルに対して処理を行います
    for bbox_file in sorted(os.listdir(bbox_dir)):
        # バウンディングボックスの情報を読み取ります
        with open(os.path.join(bbox_dir, bbox_file), 'r') as file:
            lines = file.readlines()
            lowest_bbox_bottom_y = float('inf')  # 一番下のバウンディングボックスのy座標を保持する変数
            for line in lines:
                data = line.strip().split()
                x, y, h, w = map(float, data[1:])

                # バウンディングボックスの下の座標を計算
                bbox_bottom_y = int((y + h) * 1000)

                # 一番下にあるバウンディングボックスのy座標を更新
                if bbox_bottom_y < lowest_bbox_bottom_y:
                    lowest_bbox_bottom_y = bbox_bottom_y

            # バウンディングボックスの下の座標が直線よりも下にある場合、枠の色を赤に変更
            if lowest_bbox_bottom_y > line_y:
                frame_color = (0, 0, 255)  # 赤色
                print("low")
            else:
                frame_color = (0, 255, 0)  # 緑色
                print("high")

    # 4つの直線をフレームに描画
    top_line_y = 0
    bottom_line_y = frame_height - 1
    left_line_x = 0
    right_line_x = frame_width - 1

    frame = cv2.line(frame, (left_line_x, top_line_y), (right_line_x, top_line_y), frame_color,
                     line_thickness)
    frame = cv2.line(frame, (left_line_x, bottom_line_y), (right_line_x, bottom_line_y), frame_color,
                     line_thickness)
    frame = cv2.line(frame, (left_line_x, top_line_y), (left_line_x, bottom_line_y), frame_color,
                     line_thickness)
    frame = cv2.line(frame, (right_line_x, top_line_y), (right_line_x, bottom_line_y), frame_color,
                     line_thickness)

    # 水平な直線をフレームに描画
    cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 0, 255), line_thickness)

    # 描画したフレームを出力動画に追加
    output_video.write(frame)

# 後処理
input_video.release()
output_video.release()
cv2.destroyAllWindows()
