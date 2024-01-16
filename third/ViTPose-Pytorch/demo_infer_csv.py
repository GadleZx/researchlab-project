from src.vitpose_infer import VitInference
import cv2
import csv

# CSVファイルを開く（存在しない場合は作成）
with open('output_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # CSVのヘッダーを書き込む
    writer.writerow(['frame_id', 'tracker_id', 'skeleton_coordinates'])

    vid = cv2.VideoCapture('../../data/data_002.mp4')
    model = VitInference('models/vitpose-b-multi-coco.pth', yolo_path='yolov5n.engine', tensorrt=False)
    frame_counter = 0

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    out = cv2.VideoWriter('data_002_pose.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width, frame_height))

    while True:
        ret, frame = vid.read()
        if ret:
            pts, tids, bboxes, drawn_frame, orig_frame = model.inference(frame, frame_counter)
            cv2.imshow("Video", drawn_frame)
            cv2.waitKey(1)
            frame_counter += 1
            out.write(drawn_frame)

            # CSVファイルにデータを書き込む
            for tid, points in zip(tids, pts):
                # フレームID、トラッカーID、骨格点の座標を書き込む
                writer.writerow([frame_counter, tid, points])

        else:
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()
