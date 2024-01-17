from src.vitpose_infer import VitInference
import cv2
import csv
import argparse

def make_keypoint_csv(input_video_path, output_csv_path, output_video_path):
    vid = cv2.VideoCapture(input_video_path)
    model = VitInference('models/vitpose-b.pth', yolo_path='yolov5n.engine', tensorrt=False)
    frame_counter = 0

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60,
                          (frame_width, frame_height))

    # CSVファイルを開く
    with open(output_csv_path, mode='w', newline='') as file:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--input_video_path", default='data/data_002.mp4', type=str, help="Video source.")
    parser.add_argument("--output_csv_path", default='output_keypoint.csv', type=str, help="Output csv")
    parser.add_argument("--output_video_path", default='output_video_vitpose.mp4', type=str, help="Output video.")

    args = parser.parse_args()
    print(f'args:{args}')

    make_keypoint_csv(input_video_path=args.input_video_path, output_csv_path=args.output_csv_path,output_video_path=args.output_video_path)
