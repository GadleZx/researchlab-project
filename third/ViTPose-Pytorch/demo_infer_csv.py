import cv2
import csv
import argparse
from src.vitpose_infer import VitInference

def make_keypoint_csv(input_video_path, output_csv_path, output_video_path, model_path):
    vid = cv2.VideoCapture(input_video_path)
    model = VitInference(model_path, yolo_path='yolov5n.engine', tensorrt=False)
    frame_counter = 0

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60,
                          (frame_width, frame_height))

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        while True:
            ret, frame = vid.read()
            if ret:
                pts, tids, bboxes, drawn_frame, orig_frame = model.inference(frame, frame_counter)

                print(f"Frame {frame_counter}: {pts}")

                cv2.imshow("Video", drawn_frame)
                cv2.waitKey(1)
                frame_counter += 1

                out.write(drawn_frame)

                if pts.size > 0:
                    for pt, tid in zip(pts, tids):
                        row = [frame_counter, tid]
                        for p in pt:
                            row.extend([p[0], p[1]])  # x, y coordinates
                        writer.writerow(row)

            else:
                break

    vid.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--input_video_path", default='data/data_002.mp4', type=str, help="Video source.")
    parser.add_argument("--output_csv_path", default='third/ViTPose-Pytorch/output_keypoint.csv', type=str, help="Output csv")
    parser.add_argument("--output_video_path", default='third/ViTPose-Pytorch/output_video_vitpose.mp4', type=str, help="Output video.")
    parser.add_argument("--model_path", default='third/ViTPose-Pytorch/models/vitpose-b.pth', type=str, help="Model_path.")

    args = parser.parse_args()
    print(f'args:{args}')

    make_keypoint_csv(input_video_path=args.input_video_path, output_csv_path=args.output_csv_path, output_video_path=args.output_video_path, model_path=args.model_path)
