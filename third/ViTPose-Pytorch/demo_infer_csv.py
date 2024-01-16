from src.vitpose_infer import VitInference
import cv2
import csv

vid = cv2.VideoCapture('../../data/data_002.mp4')
model = VitInference('models/vitpose-b.pth',\
            yolo_path='yolov5n.engine',tensorrt=False)
frame_counter =0

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('data_002_pose.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))

# CSVファイルを開く
with open('pose_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    while True:
        ret, frame = vid.read()
        if ret:
            pts, tids, bboxes, drawn_frame, orig_frame = model.inference(frame, frame_counter)

            # ptsの内容をデバッグ出力
            print(f"Frame {frame_counter}: {pts}")

            cv2.imshow("Video", drawn_frame)
            cv2.waitKey(1)
            frame_counter += 1

            out.write(drawn_frame)

            # 骨格点の座標とトラッキングIDを処理
            if pts.size > 0:
                # トラッキングIDごとに座標を集める
                tracking_data = {}
                for pt, tid in zip(pts, tids):
                    x, y = pt[0], pt[1]  # 確信度は無視
                    if tid not in tracking_data:
                        tracking_data[tid] = []
                    tracking_data[tid].append((x, y))

                # 各トラッキングIDとそれに関連する座標をCSVに書き込む
                for tid, coordinates in tracking_data.items():
                    coord_str = '; '.join([f"({x}, {y})" for x, y in coordinates])
                    writer.writerow([frame_counter, tid, coord_str])

        else:
            break


vid.release()
out.release()
cv2.destroyAllWindows()
