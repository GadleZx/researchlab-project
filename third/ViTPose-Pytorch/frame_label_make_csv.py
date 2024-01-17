import cv2
import pandas as pd
import argparse

# 危険なフレームのリストを読み込む関数
def load_dangerous_frames(filename):
    dangerous_frames = set()
    with open(filename, 'r') as file:
        content = file.read()
        parts = content.split('Danger')
        for part in parts:
            if part.strip():
                start, end = map(int, part.split())
                dangerous_frames.update(range(start, end + 1))
    return dangerous_frames

def make_frame_csv(input_video_path, input_text_path, output_csv_path):
    # 動画ファイルと危険フレームのテキストファイル
    video_file = input_video_path
    dangerous_frames_file = input_text_path

    # 危険なフレームのセットを読み込む
    dangerous_frames = load_dangerous_frames(dangerous_frames_file)

    # 動画を読み込む
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # フレームごとのラベルを作成
    frame_labels = []
    for frame_num in range(frame_count):
        label = 'Danger' if frame_num in dangerous_frames else 'Safe'
        frame_labels.append([frame_num, label])

    # DataFrameを作成してCSVに保存
    df = pd.DataFrame(frame_labels, columns=['Frame', 'Label'])
    #df = pd.DataFrame(frame_labels)
    df.to_csv(output_csv_path, index=False,)

    # 動画のキャプチャを解放
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--input_video_path", default='data/data_002.mp4', type=str, help="Video source.")
    parser.add_argument("--output_csv_path", default='third/ViTPose-Pytorch/output_frame.csv', type=str, help="Output csv")
    parser.add_argument("--input_txt_path", default='data/danger_data_002.txt', type=str, help="Input_txt.")


    args = parser.parse_args()
    print(f'args:{args}')

    make_frame_csv(input_video_path=args.input_video_path, output_csv_path=args.output_csv_path, input_text_path=args.input_txt_path)
