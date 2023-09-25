import math
import argparse
from pathlib import Path

import numpy as np
import cv2

def extract_frames(video_in, path_out, step):
    """
    Create a list of frame files from video file.
    Args:
        - **video_in*: Location of the video.
        - **path_out*: Where to save the frames.
        - **step**: step.
    """
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_in)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print(f"[-] open]:{video_in}")

    # Create targer folder
    Path(path_out).mkdir(parents=True, exist_ok=True)
    
    # Read until video is completed
    num_frame = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        
            # Display the resulting frame
            cv2.imshow('Frame',frame)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if num_frame % step == 0:
                cv2.imwrite(path_out + '//frame_' + str(num_frame) + '.jpg', frame)
        
        # Break the loop
        else: 
            break
        num_frame += 1
    
    # When everything done, release the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()


# python VideoProcessing/1.ImagesFromVideo.py --video_in=./data/data_002.mp4 --path_out=data/result/ --step=10

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--video_in", default='data/data_002.mp4', help="Name of the input video.")
    parser.add_argument("--path_out", default='data/result', help="Path where to save the frames.")
    parser.add_argument("--step", default=10, type=int, help="Frame step.")

    args = parser.parse_args()
    video_in = args.video_in
    path_out = args.path_out
    step = args.step
    extract_frames(video_in=video_in, path_out=path_out, step=step)
