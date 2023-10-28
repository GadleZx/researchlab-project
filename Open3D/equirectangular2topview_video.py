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

from Camera.matrix import func_projection_matrix, func_rotation_matrix, func_rotation_translation_matrix, project_points, transform_points

import numpy as np
import cv2

def compute_distance_3d(point_1, point_2):
    """Computes the distance between two 3D points using NumPy arrays.

    Args:
    point_1: A NumPy array representing the first 3D point.
    point_2: A NumPy array representing the second 3D point.

    Returns:
    A float representing the distance between the two 3D points.
    """

    # Compute the difference between the two 3D points.
    difference = point_1 - point_2

    # Compute the L2 norm of the difference vector. This is the distance between the two 3D points.
    distance = np.linalg.norm(difference)

    return distance


def spherical_to_cartesian(azimuth, elevation):
    """Converts spherical coordinates to Cartesian coordinates.

    Args:
        azimuth: A float representing the azimuth angle in degrees.
        elevation: A float representing the elevation angle in degrees.

    Returns:
        A 3D vector representing the Cartesian coordinates.
    """

    x = np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))
    y = np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))
    z = np.sin(np.radians(elevation))

    return np.array([x, y, z])


def ray_dome_intersection(ray_origin, ray_direction, dome_center, dome_radius, dome_opening_angle):
    """Computes the intersection between a ray and a dome.

    Args:
        ray_origin: A 3D vector representing the origin of the ray.
        ray_direction: A 3D vector representing the direction of the ray.
        dome_center: A 3D vector representing the center of the dome.
        dome_radius: A float representing the radius of the dome.
        dome_opening_angle: A float representing the opening angle of the dome in degrees.

    Returns:
        A 3D vector representing the intersection point, or None if the ray does not intersect the dome.
    """

    # Check if the ray intersects the sphere that circumscribes the dome.
    sphere_intersection_point = ray_sphere_intersection(ray_origin, ray_direction, dome_center, dome_radius)
    if sphere_intersection_point is None:
        return None

    # Convert the ray and the dome to spherical coordinates.
    ray_azimuth, ray_elevation = spherical_to_cartesian(ray_direction)
    dome_azimuth, dome_elevation = spherical_to_cartesian(dome_center)

    # Check if the ray's azimuth and elevation angles are within the dome's opening angle.
    if not (dome_azimuth - dome_opening_angle / 2 <= ray_azimuth <= dome_azimuth + dome_opening_angle / 2 and
            dome_elevation - dome_opening_angle / 2 <= ray_elevation <= dome_elevation + dome_opening_angle / 2):
        return None

    # The ray intersects the dome.
    return sphere_intersection_point



def calculate_plane_normal(point_1, point_2, point_3):
    """Calculates the normal vector of a plane using 3 points.

    Args:
    point_1: A 3D vector representing the first point on the plane.
    point_2: A 3D vector representing the second point on the plane.
    point_3: A 3D vector representing the third point on the plane.

    Returns:
    A 3D vector representing the normal vector of the plane.
    """

    # Calculate the two edge vectors of the plane.
    edge_vector_1 = point_2 - point_1
    edge_vector_2 = point_3 - point_1

    # Calculate the cross product of the two edge vectors to get the normal vector.
    normal_vector = np.cross(edge_vector_1, edge_vector_2)

    # Normalize the normal vector.
    normal_vector /= np.linalg.norm(normal_vector)

    return normal_vector

def ray_plane_intersection(ray_origin, ray_direction, plane_point, plane_normal):
    """Calculates the intersection between a ray and a plane.

    Args:
    ray_origin: A 3D vector representing the origin of the ray.
    ray_direction: A 3D vector representing the direction of the ray.
    plane_point: A 3D vector representing a point on the plane.
    plane_normal: A 3D vector representing the normal vector to the plane.

    Returns:
    A 3D vector representing the intersection point, or None if the ray does not
    intersect the plane.
    """
    # Calculate the denominator of the intersection formula.
    denominator = plane_normal.dot(ray_direction)

    # If the denominator is zero, the ray is parallel to the plane and does not
    # intersect it.
    if denominator == 0:
        return None

    # Calculate the distance from the ray origin to the plane.
    t = (plane_point - ray_origin).dot(plane_normal) / denominator

    # If the distance is negative, the intersection point is behind the ray origin
    # and does not exist.
    if t < 0:
        return None

    # Calculate the intersection point.
    intersection_point = ray_origin + t * ray_direction

    return intersection_point


def ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius):
    """Calculates the intersection between a ray and a sphere.

    Args:
        ray_origin: A 3D vector representing the origin of the ray.
        ray_direction: A 3D vector representing the direction of the ray.
        sphere_center: A 3D vector representing the center of the sphere.
        sphere_radius: A float representing the radius of the sphere.

    Returns:
        A 3D vector representing the intersection point, or None if the ray does not
        intersect the sphere.
    """

    # Calculate the vector from the ray origin to the sphere center.
    v = sphere_center - ray_origin

    # Calculate the dot product of the ray direction and the vector from the ray
    # origin to the sphere center.
    b = ray_direction.dot(v)

    # Calculate the discriminant of the intersection formula.
    discriminant = b**2 - v.dot(v) + sphere_radius**2

    # If the discriminant is negative, the ray does not intersect the sphere.
    if discriminant < 0:
        return None

    # Calculate the two possible intersection points.
    t1 = -b + discriminant**0.5
    t2 = -b - discriminant**0.5

    # If both intersection points are behind the ray origin, the ray does not
    # intersect the sphere.
    if t1 < 0 and t2 < 0:
        return None

    # If both intersection points are in front of the ray origin, return the
    # closer intersection point.
    if t1 >= 0 and t2 >= 0:
        return ray_origin + t1 * ray_direction

    # If one intersection point is behind the ray origin and the other is in
    # front of the ray origin, return the intersection point that is in front of
    # the ray origin.
    return ray_origin + max(t1, t2) * ray_direction

def create_grid(image_size, step):
    """
    It converts the content of the data structure from image pixels to cartesian coordinates
    Args:
        - **image_size**: Size of the image source (WH). i.e. (640,480).
        - **step**: Points step
    Return:
        - **data**: Container with the pixel points (2D) to transform.

    The function expects to return  data in the format HWC (height,width,channels)
    Channel 0 will contain the position in X.
    Channel 1 will contain the position in Y.
    """
    #print(f'imge_size:{image_size} bin:{bin}')
    data = np.zeros((round(image_size[1] / step), round(image_size[0] / step), 2))
    print(f'create_grid data:{data.shape}')
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            x = float(j * step)
            y = float(i * step)
            data[i, j, 0] = x
            data[i, j, 1] = y
    return data


def uv2xyz(data_in, image_size, step):
    """
    It converts the content of the data structure from image pixels to cartesian coordinates
    Args:
        - **image_size**: Size of the image source (WH). i.e. (640,480).
        - **step**: Points step
    Removed:
        - **data**: Container with the pixel points (2D) to transform.

    The function expects to return  data in the format HWC (height,width,channels)
    Channel 0 will contain the position in X.
    Channel 1 will contain the position in Y.
    Channel 2 will contain the position in Z.
    """
    #print(f'imge_size:{image_size} bin:{bin}')
    data = np.zeros((data_in.shape[0], data_in.shape[1], 3))
    print(f'uv2xyz data:{data.shape}')
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            # convert in uv -> spherical -> cartesian coordinates
            u = data_in[i, j, 0] / image_size[0]
            v = data_in[i, j, 1] / image_size[1]
            theta = u * 2.0 * math.pi
            phi = v * math.pi

            x = math.cos(theta) * math.sin(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(phi)
            #if z > 0.2: 
            #    x = 0
            #    y = 0
            #    z = 0
            data[i, j, 0] = x
            data[i, j, 1] = y
            data[i, j, 2] = z
    return data


def xyz_transform(data, position, orientation):
    """
    Transform a set of points
    Args:
        - **data**: Collection of points in form [w, h, xyz(3)] 
        - **ray_origin**: Center of the xyz

    """
    #print(f'imge_size:{image_size} bin:{bin}')
    print(f'xyz_transform data:{data.shape}')

    # Change from HWC to NC
    data_shape = data.shape
    data = data.reshape((-1, 3))

    # Transform the original sensor camera, to align to the ground (i.e. slope)
    rotation_matrix = func_rotation_matrix(orientation[0], orientation[1], orientation[2])
    roto_translation_matrix = func_rotation_translation_matrix(rotation_matrix, position)
    projection_matrix_identity = np.array([[1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,0],
                                 [0,0,0,1]])
    data = project_points(roto_translation_matrix, projection_matrix_identity, data)

    # remove the last dimension (w)
    # change from NC to HWC
    data = data[:,:3].reshape(data_shape)
    return data

def xyz2mesh(data, ray_origin, plane_point, plane_normal, dome_radius, do_intersect_top):
    """
    It intersects rays to mesh
    Args:
        - **data**: Collection of points in form [w, h, xyz(3)] 
        - **ray_origin**: Center of the xyz

    """
    #print(f'imge_size:{image_size} bin:{bin}')
    print(f'xyz2mesh data:{data.shape}')

    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            ray_direction = np.array([data[i, j, 0], data[i, j, 1], data[i, j, 2]])
            intersection_point = ray_plane_intersection(ray_origin, ray_direction, plane_point, plane_normal)
            # The intersection with the ground is used if the base is true
            # otherwise the intersection with the sphere is used.

            # Too far (point to the edge)
            if intersection_point is not None and (abs(intersection_point[0]) > dome_radius or abs(intersection_point[1]) > dome_radius or abs(intersection_point[2]) > dome_radius):
                intersection_point = None

            # Point out of dome base
            if intersection_point is not None :
                dist = compute_distance_3d(plane_point, intersection_point)
                if dist > dome_radius:
                    intersection_point = None

            # If there is no intersection with the base and the full dome intersection is desired
            if intersection_point is None and do_intersect_top:
                # The center of the sphere is inverted respect the plane point (confirm why)
                intersection_point = ray_sphere_intersection(ray_origin, ray_direction, -plane_point, dome_radius)

            # No intersection. The point is out of bound. Collapse to the center for visualization
            if intersection_point is None:
                intersection_point = np.array([0,0,0])

            # Update the 3D point
            data[i, j, 0] = intersection_point[0]
            data[i, j, 1] = intersection_point[1]
            data[i, j, 2] = intersection_point[2]
    return data


def xyz2camera(data, position_sensor_camera, dome_radius):
    """
    The position of the virtual camera is the same of the sensor camera.
    The orientation is facing down.
    """
    
    # Change from HWC to NC
    data_shape = data.shape
    data = data.reshape((-1, 3))

    # Transform the world around the camera located at the origin.
    position = position_sensor_camera # np.array([0.0, 0.0, 0.0])
    orientation = np.array([0.0, 0.0, 0.0])
    rotation_matrix = func_rotation_matrix(orientation[0], orientation[1], orientation[2])
    roto_translation_matrix = func_rotation_translation_matrix(rotation_matrix, position)
    #print(f'roto_translation_matrix:{roto_translation_matrix}')
    projection_matrix = func_projection_matrix(30.0, 16. / 16., 0.1, 100.0)
    projection_matrix_identity = np.array([[1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,0],
                                 [0,0,0,1]])
    print(f'projection_matrix:{projection_matrix}')
    data = project_points(roto_translation_matrix, projection_matrix, data)
    # Convert NaN and Inf values to 0
    data = np.nan_to_num(data)

    # clip the points out of bound
    max_size_clip = dome_radius
    for i in range(0, data.shape[0]):
        if data[i, 0] < -max_size_clip or data[i, 0] > max_size_clip or data[i, 1] < -max_size_clip or data[i, 1] > max_size_clip or data[i, 2] < -max_size_clip or data[i, 2] > max_size_clip:
            data[i, 0] = 0
            data[i, 1] = 0
            data[i, 2] = 0

    # remove the last dimension (w)
    # change from NC to HWC
    data = data[:,:3].reshape(data_shape)
    #print(f'data:{data}')
    return data


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


def xy2image(data, data_color, width, height):
    """
    The position of the virtual camera is the same of the sensor camera.
    The orientation is facing down.
    """
    # Change from HWC to NC
    data = data.reshape((-1, 3))
    img = np.zeros((width, height, 3), dtype = "uint8")
    print(img.shape, data_color.shape)
    for i in range(0, data.shape[0]):
        x = data[i, 0]
        y = data[i, 1]
        #print(x, y, data[i, :], data_color[i,:])
        img[int(y), int(x), 0] = data_color[i, 0]
        img[int(y), int(x), 1] = data_color[i, 1]
        img[int(y), int(x), 2] = data_color[i, 2]
    return img


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
    data = xyz2camera(data, -ray_origin, dome_radius)

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
        data = xyz2camera(data, -ray_origin, dome_radius)

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
