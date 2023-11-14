# http://www.paul-reed.co.uk/programming.html
# https://paulbourke.net/panorama/icosahedral/
# https://github.com/rdbch/icosahedral_sampler

import os
import random
import copy
import sys
sys.path.append('.')
import math
import argparse

import numpy as np
import open3d as o3d
from PIL import Image

import numpy as np
import cv2

from common import create_grid, calculate_plane_normal, uv2xyz, xyz_transform, xyz2mesh, xyz2camera, xy2image

def camera2xy(data, width, height):
    """
    The position of the virtual camera is the same of the sensor camera.
    The orientation is facing down.
    """
    # Change from HWC to NC
    data_shape = data.shape
    data = data.reshape((-1, 3))

    # Changed the solution to a static desired maximum/minimum range to support sparse points
    #minx = data[:, 0].min()
    #maxx = data[:, 0].max()
    #miny = data[:, 1].min()
    #maxy = data[:, 1].max()

    minx = -10
    maxx = 10
    miny = -10
    maxy = 10

    for i in range(0, data.shape[0]):
        x = min((data[i, 0] - minx) / (maxx - minx) * width, width - 1)
        y = min((data[i, 1] - miny) / (maxy - miny) * height, height - 1)
        data[i, 0] = x
        data[i, 1] = y
        data[i, 2] = 0

    # remove the last dimension (w)
    # change from NC to HWC
    data = data[:,:3].reshape(data_shape)
    return data


def transform_point(point2d, img_width, img_height, img_out_side):

    # Create the grid of points to transform
    data_xy = np.zeros((1, 1, 2))
    data_xy[0,0,0] = point2d[0]
    data_xy[0,0,1] = point2d[1]

    # Transform the equirectangular image in spherical with center 0,0,0 and radius 1
    data = uv2xyz(data_xy, (img_width, img_height), 1)

    # sensor camera position
    ray_origin = np.array([0, 0, 1.7]) # meters
    sensor_position_at_origin_must_be_0 = np.array([0, 0, 0]) # used only to pass a variable to the function
    sensor_orientation_deg = np.array([-11, -6, 10])# = np.array([-15, 0, 0])

    # Transform the sensor orientation
    # The orientation is empirically estimated (ground slope)
    # NO POSITION modified, since it updates ONLY the direction vector
    data = xyz_transform(data, sensor_position_at_origin_must_be_0, sensor_orientation_deg)

    # Dome parameters
    # Define the 3 points on the plane.
    point_1 = np.array([0.0, 0.0, 0.0])
    point_2 = np.array([1.0, 0.0, 0.0])
    point_3 = np.array([0.0, 1.0, 0.0])
    # Calculate the normal vector of the plane.
    plane_normal = calculate_plane_normal(point_1, point_2, point_3)
    # Dome base size
    dome_radius = 15

    # Use False if it is desired only the base. True for the full dome
    data = xyz2mesh(data, ray_origin, point_1, plane_normal, dome_radius, False)
    # Project on camera
    data = xyz2camera(data, -ray_origin, dome_radius, 30.0)

    # The point is not on the ground
    if data[0,0,0] == 0 and data[0,0,1] == 0:
        return None

    # create output image
    side = img_out_side # 1024
    data_xy = camera2xy(copy.deepcopy(data), side, side)

    return data_xy

def image2pointcloud(img, step, mode, img_out_side):
    """
    Create a pcd file from image file.
    Args:
        - **img*: image (PIL).
        - **step**: Point cloud step.
        - **mode**: Mode the data is elaborated (uv, image_uv)
        - **img_out_side**: Each side (w,h) of the output image
    Return:
        - Point cloud structrue
    Note: The image is rotated 90 degrees anticlockwise.
    """
    # Convert in numpy format
    img_array = np.array(img)

    # image slicing
    if step > 1:
        img_array_slice = img_array[::step,::step,:]
        #print(f'data:{data.shape}')
        #print(f'img_array:{img_array.shape}')
        # Reshape the image data into a 2D array
        data_color = img_array_slice.reshape(-1, 3)
    else:
        # Reshape the image data into a 2D array
        data_color = img_array.reshape(-1, 3)

    if mode == 'uv':

        # Create the grid of points to transform
        data_xy = create_grid((img_array.shape[1], img_array.shape[0]), step)

        # Transform the equirectangular image in spherical with center 0,0,0 and radius 1
        data = uv2xyz(data_xy, (img_array.shape[1], img_array.shape[0]), step)

        # sensor camera position
        ray_origin = np.array([0, 0, 1.7]) # meters
        sensor_position_at_origin_must_be_0 = np.array([0, 0, 0]) # used only to pass a variable to the function
        sensor_orientation_deg = np.array([-11, -6, 10])# = np.array([-15, 0, 0])

        # Transform the sensor orientation
        # The orientation is empirically estimated (ground slope)
        # NO POSITION modified, since it updates ONLY the direction vector
        data = xyz_transform(data, sensor_position_at_origin_must_be_0, sensor_orientation_deg)

        # Dome parameters
        # Define the 3 points on the plane.
        point_1 = np.array([0.0, 0.0, 0.0])
        point_2 = np.array([1.0, 0.0, 0.0])
        point_3 = np.array([0.0, 1.0, 0.0])
        # Calculate the normal vector of the plane.
        plane_normal = calculate_plane_normal(point_1, point_2, point_3)
        # Dome base size
        dome_radius = 15

        # Use False if it is desired only the base. True for the full dome
        data = xyz2mesh(data, ray_origin, point_1, plane_normal, dome_radius, False)
        # Project on camera
        data = xyz2camera(data, -ray_origin, dome_radius, 30.0)

        # create output image
        side = img_out_side # 1024
        data_xy = camera2xy(copy.deepcopy(data), side, side)
        img = xy2image(data_xy, data_color, side, side)
    else:
        assert(f'mode not implemented:{mode}')

    # Reshape the point cloud data into a 2D array
    data = data.reshape(-1, 3)

    # Create a PointCloud object
    pc = o3d.geometry.PointCloud()

    # Set the point cloud data
    pc.points = o3d.utility.Vector3dVector(data)

    # Set the RGB colors for each point in the point cloud
    pc.colors = o3d.utility.Vector3dVector(data_color / 255)

    return pc, img


def process_video(frame_step, render_step):
    # 入力動画ファイルのパス
    input_video_path = "data/data_002.mp4"

    # 出力動画ファイルのパス
    output_video_path = "topview_track_output_video.mp4"

    # テキストファイルが格納されているディレクトリのパス
    text_files_directory = "data/tracking_data_002/"

    # OpenCVを使用して入力動画を読み込み
    input_video = cv2.VideoCapture(input_video_path)

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力フォーマットを指定
    fps = int(input_video.get(cv2.CAP_PROP_FPS))  # 元動画のフレームレートを取得

    img_out_side = 512
    frame_width = img_out_side #int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height =  img_out_side #int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 出力動画の準備
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_counter = 0
    id_colors = {}  # IDごとの色を保持する辞書

    while True:
        # 動画からフレームを読み込む
        ret, frame = input_video.read()
        if not ret:
            break  # 動画の最後に達したらループを終了
        frame_counter += 1
        if frame_counter % frame_step != 0:
            continue

        pc, img = image2pointcloud(frame, render_step, 'uv', img_out_side)

        # バウンディングボックスの情報をリセット
        bbox_file_path = os.path.join(text_files_directory, f'data_{frame_counter}.txt')

        # テキストファイルが存在し、空でない場合に処理を行う
        if os.path.exists(bbox_file_path) and os.path.getsize(bbox_file_path) > 0:
            # バウンディングボックスの情報を読み取ります
            with open(bbox_file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    data = line.split()

                    id = int(data[0])
                    if id not in id_colors:
                        id_colors[id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                    # 文字列をクリーニングしてから浮動小数点数に変換
                    x1, y1, x2, y2 = map(float, [s.strip('[,]') for s in data[1:]])

                    print(f"{id},{x1},{y1},{x2},{y2}")

                    # Transform equirectangular point in top-view
                    p2d = transform_point([(x1 + x2) / 2, y2], frame.shape[1], frame.shape[0], img_out_side)
                    # is invalid point?
                    if p2d is None:
                        continue

                    # IDに割り当てられた色を取得
                    color = id_colors[id]

                    cv2.circle(img, (int(p2d[0,0,0]), int(p2d[0,0,1])), 5, color, 2)
                    cv2.putText(img, f'ID: {id}', (int(p2d[0,0,0]), int(p2d[0,0,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # バウンディングボックスを動画フレームに描画
                    #cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # 緑色の矩形

                    # IDを動画フレームに描画
                    #cv2.putText(frame, f'ID: {id}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(1)

        # 描画されたフレームを出力動画に書き込む
        output_video.write(img)

    # 動画ファイルを解放
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

    print("動画処理が完了しました。")


if __name__ == "__main__":

    print(f'If colors and data points size mismatch, no give color is used!!!')

    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--frame_step", default=10, type=int, help="Int frame step. >1 skip frames.")
    parser.add_argument("--render_step", default=10, type=int, help="Int point cloud step. Higher is the value, sparser the points.")

    args = parser.parse_args()
    frame_step = args.frame_step
    render_step = args.render_step

    process_video(frame_step=frame_step, render_step=render_step)
