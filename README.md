# How to Setup

```Console
pip install -r requirements.txt
```

# How to Use

## Open3D

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
python Open3D/Visualization.py --file_name="Open3D/result.pcd"
```

Video

```Console
python Open3D/equirectangular2topview_video.py --frame_step=1 --render_step=1
```
## norfair

https://github.com/tryolabs/norfair

```Console
python norfair/demo_file_output.py --transformation homography --draw-paths --path-history 150 --distance-threshold 200 --track-boxes --max-points=900 --min-distance=14 --save --model yolov5x --hit-counter-max 4 data/data_002.mp4
```
