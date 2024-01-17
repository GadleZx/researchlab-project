# How to Setup

```Console
pip install -r requirements.txt
```

# How to Use

## Estimate danger

Clone the project  
Unzip the zip files in the data folder 

```Console
python Open3D/equirectangular2topview_video_V2.py --input_video_path='data/data_002.mp4' --output_video_path='topview_track_output_video.mp4' --output_localizationXY_path='data/localizationXY_cam/data_002_locXY.txt' --tracking_files_directory='data/tracking_data_002/' --frame_step=1 --no-do_image2birdview

python run/DangerPredictionV1.py --input_video_path='data/data_002.mp4' --output_video_path='danger_output_video.mp4' --localizationXY_path='data/localizationXY_cam/data_002_locXY.txt' --tracking_files_directory='data/tracking_data_002/' --label_path='data/danger_data_002.txt' --frame_step=1

python run/DangerPredictionV2.py --input_video_path='data/data_002.mp4' --output_video_path='danger_output_video_v2.mp4' --localizationXY_path='data/localizationXY_cam/data_002_locXY.txt' --tracking_files_directory='data/tracking_data_002/' --label_path='data/danger_data_002.txt' --frame_step=1
```


## Open3D


Intersection Plane-Sphere

```Console
python Open3D/equirectangular2topview_V2.py --file_image=data/frame_sample.jpg --step=100 --mode='sphereplane'
python Open3D/Visualization.py --file_name="Open3D/result.pcd"
```


```Console
python Open3D/create_pcd.py 
python Open3D/Visualization.py --file_name="Open3D/result.pcd"

or 

python Open3D/equirectangular2spherical_pcd.py
python Open3D/Visualization.py --file_name="Open3D/result.pcd"

or 

python Open3D/equirectangular2spherical_pcd.py --file_image=data/frame_sample.jpg
python Open3D/Visualization.py --file_name="Open3D/result.pcd"

or (default data)

python Open3D/Visualization.py

or(bird view)

python Open3D/bird_view.py
python Open3D/Visualization.py --file_name="Open3D/result.pcd"

```

## Create Frames and update poses


Clone the project  
Unzip the zip files in the data folder 

```Console
python VideoProcessing/1.ImagesFromVideo.py --video_in=./data/data_002.mp4 --path_out=data/result/ --step=10
python VideoProcessing/2.Frames2Spherical.py --path_frames_folder=data/result/ --path_trajectory=data/log_pose_data_002/frame_trajectory.txt --video_fps=30
python Open3D/Visualization.py --file_name="Open3D/result.pcd"
```

```Console
python VideoProcessing/3.SingleFrame2Spherical.py --path_frames_folder=data/result/ --path_trajectory=data/log_pose_data_002/frame_trajectory.txt --video_fps=30
python Open3D/Visualization.py --file_name="Open3D/result.pcd"
```



## Projection

Visualize a projection from a sample equirectangular image

```Console
python Open3D/equirectangular2spherical_pcd2.py --file_image=data/frame_sample.jpg --step=1
python Open3D/Visualization.py --file_name="Open3D/result.pcd"
```

## Cubemap

Convert 360degrees images.
https://github.com/sunset1995/py360convert/tree/master


## Top View

Slow processing

```Console
python Open3D/equirectangular2topview.py --file_image=data/frame_sample.jpg --step=1

python Open3D/equirectangular2topview_V2.py --file_image=data/frame_sample.jpg --step=10 --mode='ground2image'
python Open3D/equirectangular2topview_V2.py --file_image=data/frame_sample.jpg --step=10 --mode='image2ground'
python Open3D/equirectangular2topview_V2.py --file_image=frame0.jpg --step=10 --mode='image2ground'

python Open3D/Visualization.py --file_name="Open3D/result.pcd"
```

Video

```Console
python Open3D/equirectangular2topview_video.py --frame_step=1 --render_step=1

python Open3D/equirectangular2topview_video_V2.py --input_video_path='data/data_002.mp4' --output_video_path='topview_track_output_video.mp4' --output_localizationXY_path='data/localizationXY_cam/data_002_locXY.txt' --tracking_files_directory='data/tracking_data_002/' --frame_step=10 --render_step=10 --do_image2birdview

python Open3D/equirectangular2topview_video_V2.py --input_video_path='data/data_002.mp4' --output_video_path='topview_track_output_video.mp4' --output_localizationXY_path='data/localizationXY_cam/data_002_locXY.txt' --tracking_files_directory='data/tracking_data_002/' --frame_step=1 --no-do_image2birdview

```
## norfair

https://github.com/tryolabs/norfair

```Console
python norfair/demo_file_output.py --transformation homography --draw-paths --path-history 150 --distance-threshold 200 --track-boxes --max-points=900 --min-distance=14 --save --model yolov5x --hit-counter-max 4 data/data_002.mp4
```

## Human pose

third/ViTPose-Pytorch  
Please, follow the readme.md for the setup

```Console
Example (path to the video may be different)
python demo_infer.py
```

## Human pose make csv

```Console
python third/ViTPose-Pytorch/demo_infer_csv.py --input_video_path='data/data_002.mp4' --output_csv_path='third/ViTPose-Pytorch/keypoint.csv' --output_video_path='third/ViTPose-Pytorch/vitpose.mp4' --model_path='third/ViTPose-Pytorch/models/vitpose-b.pth'
```

## Frame label make csv

```Console
python third/ViTPose-Pytorch/frame_label_make_csv.py --input_video_path='data/data_002.mp4' --output_csv_path='third/ViTPose-Pytorch/frame_label.csv' --input_txt_path='data/danger_data_002.txt'
```


## Link

[simpy](https://docs.sympy.org/dev/search.html)  
