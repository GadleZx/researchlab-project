import cv2
import os

# 動画ファイルの読み込み
input_video = cv2.VideoCapture('data/data_002.mp4')

# 出力動画の設定
output_video_path = 'output_video.mp4'
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
bbox_dir = 'data/labels_data_002'

#bbox_file_list = sorted(os.listdir(bbox_dir))
#print(f'bbox_file_list:{bbox_file_list}')

frame_counter = 0
while True:
    # 動画からフレームを読み込む
    ret, frame = input_video.read()
    if not ret:
        break  # 動画の最後に達したらループを終了
        
    frame_counter += 1

    # 各フレームごとにframe_colorを初期化
    frame_color = (0, 0, 255)  # 初期値は赤色

    # バウンディングボックスの情報をリセット
    change_frame_color = False
    
    #print(f'frame:{frame.shape}')

    # 各テキストファイルに対して処理を行います
    #for bbox_file in bbox_file_list:
    if True:
        # バウンディングボックスの情報を読み取ります
        with open(os.path.join(bbox_dir, 'data_002_' + str(frame_counter) + '.txt'), 'r') as file:
            lines = file.readlines()
            lowest_bbox_bottom_y = 0  # 一番下のバウンディングボックスのy座標を保持する変数
            for line in lines:
                data = line.strip().split()
                x, y, w, h = map(float, data[1:])
                
                # convert the normalized coordinates in pixel value
                # x is the center
                x = x * frame.shape[1]
                w = w * frame.shape[1]
                y = y * frame.shape[0]
                h = h * frame.shape[0]
                
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 2) 
                cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0)) 

                # バウンディングボックスの下の座標を計算
                bbox_bottom_y = int((y + h / 2))

                # 一番下にあるバウンディングボックスのy座標を更新
                if bbox_bottom_y > lowest_bbox_bottom_y:
                    lowest_bbox_bottom_y = bbox_bottom_y

            # バウンディングボックスの下の座標が直線よりも下にある場合、枠の色を赤に変更
            if lowest_bbox_bottom_y < line_y:
                frame_color = (0, 255, 0)  # 緑色
            else:
                frame_color = (0, 0, 255)  # 赤色

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
    cv2.line(frame, (0, line_y), (frame_width, line_y), frame_color, line_thickness)
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

    # 描画したフレームを出力動画に追加
    output_video.write(frame)

# 後処理
input_video.release()
output_video.release()
cv2.destroyAllWindows()
