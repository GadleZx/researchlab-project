import sys
sys.path.append('./')
import os
import math
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

import open3d as o3d

# import the function to convert an image in xyz
from Open3D.equirectangular2spherical_pcd import image2pointcloud

def frame2pc_and_transform(path_img, pose):
    """
    It gets a selected image, convert in spherical and transform the orientation.
    """

    # Load an image
    img = cv2.imread(path_img)
    if img is None:
        print(f'[-] Open:{path_img}')
        return
    else:
        print(f'[+] Open:{path_img}')
    cv2.line(img, (0,800), (1920,800), (255,0,255),2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    pc = image2pointcloud(img, 1, 'image_uv')

    # Get the pointd data as numpy
    # From Open3D to numpy
    np_points = np.asarray(pc.points)
    print(f'np_points:{np_points.shape}')

    # scale down the points size for better visualization (optional)
    np_points *= 1.0

    # Transform the position
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    # Rotate 90 degrees to match the orientation in StellaVSLAM
    r = R.from_euler('x', -90, degrees=True)
    np_points = r.apply(np_points)

    # Apply the camera pose rotation
    #r = R.from_quat(pose[3:])
    #print(f'r:{r}')
    #np_points = r.apply(np_points)
    #print(f'np_pointsA:{np_points[100]}') # debug only

    # translate
    #t = np.asarray(pose[:3])
    #print(f't:{t}')
    #for i in range(0, np_points.shape[0]):
    #    for k in range(0, np_points.shape[1]):
    #        np_points[i,k] = np_points[i,k] + t[k]
    #np_points[:,:] = np_points[:,:] + t  # equivalent to the above code
    #print(f'np_pointsB:{np_points[100]}') # debug only

    # get the colors
    np_colors = np.asarray(pc.colors)

    return np_points, np_colors

    # convert the points back to Open3D
    # From numpy to Open3D
    #pc.points = o3d.utility.Vector3dVector(np_points)    

    # Save the PointCloud object to a PCD file
    #o3d.io.write_point_cloud('Open3D/result.pcd', pc)



def compute_frames(path_frames_folder, path_trajectory, video_fps):
    """
    It elaborates each frame with associated trajectory.
    Args:
        - **path_frames_folder**: Location where the frames are locates.
        - **path_trajectory**: Location with the trajectory data.
        - **video_fps**: Original video fps used to calculate the frame number.
    """

    # list of image frames
    path_frames = os.listdir(path_frames_folder)
    #print(f'path_frames:{path_frames}')

    # How many ms for each frame
    video_frame_ms = 1. / video_fps

    # read the trajectory file
    cam_pose = dict()
    with open(path_trajectory, 'r', encoding='UTF-8') as file:
        while line := file.readline():
            # remove endline and return, and split
            s_cam_pose = line.rstrip()
            data = np.fromstring(s_cam_pose, dtype=float, sep=' ')
            frame_num = data[0] / video_frame_ms
            cam_pose[round(frame_num)] = data[1:]
            #print(f'rame_num:{frame_num} round:{round(frame_num)} data[0]:{data[0]}')
    #print(f'cam_pose:{cam_pose}') 

    # process each frame
    np_points = []
    np_colors = []
    for frame_name in path_frames:
        # get the frame number
        filename = os.path.splitext(frame_name)[0]
        # split on _ and get the third element (the number)
        num = int(filename.split('_')[1])
        #print(f'frame_name:{frame_name} filename:{filename} num:{num}')
        # convert the number in frame
        # get the pose
        if num in cam_pose:
            s_pose = cam_pose[num]
            print(f'num:{num} s_pose:{s_pose}')
            # elaborate the selected frame
            np_points_tmp, np_colors_tmp = frame2pc_and_transform(path_frames_folder + '/' + frame_name, s_pose)

            # collect all the points and color
            np_points.append(np_points_tmp)
            np_colors.append(np_colors_tmp)
            break

    # concatenate points and colors
    np_points_all = np.concatenate(np_points, axis=0)
    np_colors_all = np.concatenate(np_colors, axis=0)

    # Create a PointCloud object
    pc = o3d.geometry.PointCloud()
    # Set the point cloud data
    pc.points = o3d.utility.Vector3dVector(np_points_all)
    # Set the colors
    pc.colors = o3d.utility.Vector3dVector(np_colors_all)
    # Save the PointCloud object to a PCD file
    o3d.io.write_point_cloud('Open3D/result.pcd', pc)

# python VideoProcessing/2.Frames2Spherical.py --path_frames_folder=data/result/ --path_trajectory=data/log_pose/frame_trajectory.txt --video_fps=30
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--path_frames_folder", default='data/result', help="Folder with the image files.")
    parser.add_argument("--path_trajectory", default='data/log_pose/frame_trajectory.txt', help="Path with trajectory data (i.e. StellaVSLAM out).")
    parser.add_argument("--video_fps", default=30, type=float, help="Original video framerate.")

    args = parser.parse_args()
    path_frames_folder = args.path_frames_folder
    path_trajectory = args.path_trajectory
    video_fps = args.video_fps
    compute_frames(path_frames_folder=path_frames_folder, path_trajectory=path_trajectory, video_fps=video_fps)
