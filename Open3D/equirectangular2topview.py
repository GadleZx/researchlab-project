# http://www.paul-reed.co.uk/programming.html
# https://paulbourke.net/panorama/icosahedral/
# https://github.com/rdbch/icosahedral_sampler

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

from common import create_grid, calculate_plane_normal, uv2xyz, xyz_transform, xyz2mesh, xyz2camera, xy2image, ray_sphere_intersection_v2, direction_vector, xy2xyz, xyz2xy, xyz_transform, intersect_dome, xyz_transformV2

def camera2xy(data, width, height):
    """
    The position of the virtual camera is the same of the sensor camera.
    The orientation is facing down.
    """
    # Change from HWC to NC
    data_shape = data.shape
    data = data.reshape((-1, 3))

    minx = data[:, 0].min()
    maxx = data[:, 0].max()
    miny = data[:, 1].min()
    maxy = data[:, 1].max()
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


def image2pointcloud(img, step, mode):
    """
    Create a pcd file from image file.
    Args:
        - **img*: image (PIL).
        - **step**: Point cloud step.
        - **mode**: Mode the data is elaborated (uv, image_uv)
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
        sensor_orientation_deg = np.array([-15, 0, 0])

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
        side = 512 # 1024
        data_xy = camera2xy(copy.deepcopy(data), side, side)
        img = xy2image(data_xy, data_color, side, side)
        cv2.imwrite('img.jpg', img)
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

    return pc


def ground2image(img, step, mode, config):
    """
    Create a pcd file from image file.
    Args:
        - **img*: image (PIL).
        - **step**: Point cloud step.
        - **mode**: Mode the data is elaborated (uv, image_uv)
    Return:
        - Point cloud structrue
    Note: The image is rotated 90 degrees anticlockwise.
    """
    # Convert in numpy format
    img_array = np.array(img)

    data_res = []
    data_color_res = []
    if mode == 'uv':

        # sensor camera position
        sensor_position_at_origin_must_be_0 = np.array([0, 0, 0]) # used only to pass a variable to the function
        # NOTE: Rotation is in the order ZXY (Z+ up, X+ right, Y+ front)
        #       Pan, Tilt, Roll
        sensor_orientation_deg = np.array([15, 0, 25])

        # Source image
        image_size_source = [img_array.shape[1], img_array.shape[0]]

        # sensor camera position
        sensor_position = np.array([0, 0, 1.7]) # meters

        Wmin = -10.0
        Wmax = 10.0
        Hmin = -10.0
        Hmax = 10.0
        img_w = 128#256
        img_h = 128#256
        Wbin = (Wmax - Wmin) / float(img_w)
        Hbin = (Hmax - Hmin) / float(img_h)
        z_ground = 0

        img_out = np.zeros((img_h, img_w, 3), dtype = "uint8")
        print(f'img_out:{img_out.shape}')

        # Compute the points interval respect the dome
        if True:
            for y in range(img_h):
                for x in range(img_w):

                    # Ground to Image

                    # Compute current x on the plane ground
                    x_plane = Wbin * x + Wmin
                    y_plane = Hbin * y + Hmin
                    # Compute the direction vector
                    p_ground = np.array([x_plane, y_plane, z_ground])
                    dir_vect = direction_vector(p_ground, sensor_position)

                    # The center of the sphere is inverted respect the plane point (confirm why)
                    intersection_point = ray_sphere_intersection_v2(p_ground, dir_vect, sensor_position, 1.0)
                    if intersection_point is not None:

                        # The intersection point must be aligned to the origin and rotated to the expected sensor camera orientation
                        p3d = np.array([intersection_point[0], intersection_point[1], intersection_point[2]])
                        # translate to the origin
                        p3d -= sensor_position
                        # rotate the intersection point (on a unitary sphere) to align to the camera orientation
                        p3d_t = xyz_transform(p3d, sensor_position_at_origin_must_be_0, sensor_orientation_deg, order='xyz')

                        # Transform the point in the sphere to image coordinate to get the color value in
                        xi, yi = xyz2xy(p3d_t[0], p3d_t[1], p3d_t[2], image_size_source)
                        if xi < 0: xi = image_size_source[0] + xi
                        if xi < 0: xi = 0
                        if xi >= image_size_source[0]: xi = image_size_source[0] - 1
                        if yi < 0: yi = 0
                        if yi >= image_size_source[1]: yi = image_size_source[1] - 1

                        # return the point on the sphere to the sensor coordinates
                        p3d = p3d_t + sensor_position

                        img_out[int(y), int(x), :] = img_array[int(yi), int(xi), :]
                        data_color_res.append(img_array[int(yi), int(xi), :])
                        data_color_res.append(img_array[int(yi), int(xi), :])

                        #print(f'p_ground:{p_ground} p_norm:{p_norm} sensor_position:{sensor_position} intersection_point:{intersection_point}')
                        data_res.append(p_ground) # point to ground
                        #data_res.append(intersection_point)
                        data_res.append(p3d) # point to sphere


                        # Image to Ground
                        if False:

                            # Sphere coordinates
                            xx, yy, zz = xy2xyz(xi, yi, image_size_source, 1)
                            p3d_sphere = np.array([xx, yy, zz])
                            # Rotate and translate the image to align to the camera orientation
                            p3d_sphere_t = xyz_transform(p3d_sphere, sensor_position, -sensor_orientation_deg, order='zyx')

                            # Dome parameters
                            # Define the 3 points on the plane.
                            point_1 = np.array([0.0, 0.0, 0.0])
                            point_2 = np.array([1.0, 0.0, 0.0])
                            point_3 = np.array([0.0, 1.0, 0.0])
                            # Calculate the normal vector of the plane.
                            plane_normal = calculate_plane_normal(point_1, point_2, point_3)
                            # Dome base size
                            dome_radius = 25

                            intersection_point = intersect_dome(sensor_position, p3d_sphere_t - sensor_position, point_1, plane_normal, dome_radius, True)

                            #print(x_plane, y_plane, intersection_point)
                            #data_res.append(p3d_sphere_t)

                            data_res.append(intersection_point + np.array([0, 0, -1]))
                            data_color_res.append(img_array[int(yi), int(xi), :])


            cv2.imwrite('img0.jpg', img_out)
        data_res = np.array(data_res)
        data_color_res = np.array(data_color_res)
    else:
        assert(f'mode not implemented:{mode}')

    print(f'data_res:{data_res}')

    # Reshape the point cloud data into a 2D array
    data_res = data_res.reshape(-1, 3)
    data_color_res = data_color_res.reshape(-1, 3)

    # Create a PointCloud object
    pc = o3d.geometry.PointCloud()

    # Set the point cloud data
    pc.points = o3d.utility.Vector3dVector(data_res)

    # Set the RGB colors for each point in the point cloud
    pc.colors = o3d.utility.Vector3dVector(data_color_res / 255)

    return pc


def image2ground(img, step, mode):
    """
    Create a pcd file from image file.
    Args:
        - **img*: image (PIL).
        - **step**: Point cloud step.
        - **mode**: Mode the data is elaborated (uv, image_uv)
    Return:
        - Point cloud structrue
    Note: The image is rotated 90 degrees anticlockwise.
    """
    # Convert in numpy format
    img_array = np.array(img)

    data_res = []
    data_color_res = []
    if mode == 'uv':

        # sensor camera position
        sensor_position_at_origin_must_be_0 = np.array([0, 0, 0]) # used only to pass a variable to the function
        # NOTE: Rotation is in the order ZXY (Z+ up, X+ right, Y+ front)
        #       Pan, Tilt, Roll
        sensor_orientation_deg = np.array([15, 0, 25])

        # Source image
        image_size_source = [img_array.shape[1], img_array.shape[0]]

        # sensor camera position
        sensor_position = np.array([0, 0, 1.7]) # meters

        Wmin = -10.0
        Wmax = 10.0
        Hmin = -10.0
        Hmax = 10.0
        img_w = 128#256
        img_h = 128#256
        Wbin = (Wmax - Wmin) / float(img_w)
        Hbin = (Hmax - Hmin) / float(img_h)
        z_ground = 0

        img_out = np.zeros((img_h, img_w, 3), dtype = "uint8")
        print(f'img_out:{img_out.shape}')

        # Compute the points interval respect the dome
        if True:
            for y in range(0, image_size_source[1], 10):
                for x in range(0, image_size_source[0], 10):

                    # Image to Ground

                    # Sphere coordinates
                    xx, yy, zz = xy2xyz(x, y, image_size_source, 1)
                    p3d_sphere = np.array([xx, yy, zz])
                    # Rotate and translate the image to align to the camera orientation
                    p3d_sphere_t = xyz_transform(p3d_sphere, sensor_position, -sensor_orientation_deg, order='zyx')

                    # Dome parameters
                    # Define the 3 points on the plane.
                    point_1 = np.array([0.0, 0.0, 0.0])
                    point_2 = np.array([1.0, 0.0, 0.0])
                    point_3 = np.array([0.0, 1.0, 0.0])
                    # Calculate the normal vector of the plane.
                    plane_normal = calculate_plane_normal(point_1, point_2, point_3)
                    # Dome base size
                    dome_radius = 15

                    intersection_point = intersect_dome(sensor_position, p3d_sphere_t - sensor_position, point_1, plane_normal, dome_radius, False)
                    if intersection_point is not None:
                        data_res.append(p3d_sphere_t)
                        data_color_res.append(img_array[int(y), int(x), :])
                        data_res.append(intersection_point)
                        data_color_res.append(img_array[int(y), int(x), :])

            cv2.imwrite('img1.jpg', img_out)
        data_res = np.array(data_res)
        data_color_res = np.array(data_color_res)
    else:
        assert(f'mode not implemented:{mode}')

    print(f'data_res:{data_res}')

    # Reshape the point cloud data into a 2D array
    data_res = data_res.reshape(-1, 3)
    data_color_res = data_color_res.reshape(-1, 3)

    # Create a PointCloud object
    pc = o3d.geometry.PointCloud()

    # Set the point cloud data
    pc.points = o3d.utility.Vector3dVector(data_res)

    # Set the RGB colors for each point in the point cloud
    pc.colors = o3d.utility.Vector3dVector(data_color_res / 255)

    return pc


def save_image_pointcloud(fname_in_img, fname_out_pcd, step):
    """
    Create a pcd file from image file.
    Args:
        - **fname_in_img*: Filename with the image.
        - **fname_out_pcd*: Filename where to save the pcd file.
        - **step**: Point cloud step.
    Note: The image is rotated 90 degrees anticlockwise.
    """
    # Load an image
    img = Image.open(fname_in_img)
    if img is None:
        print(f'[-] Open:{fname_in_img}')
        return
    
    #pc = image2pointcloud(img, step, 'uv')
    #pc = ground2image(img, step, 'uv')
    pc = image2ground(img, step, 'uv')

    # Save the PointCloud object to a PCD file
    o3d.io.write_point_cloud(fname_out_pcd, pc)


if __name__ == "__main__":

    print(f'If colors and data points size mismatch, no give color is used!!!')

    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--file_image", default='data/equirect001.jpeg', help="name of the equirectangular image.")
    parser.add_argument("--file_out", default='Open3D/result.pcd', help="name of the point cloud file (ply,pcd)")
    parser.add_argument("--step", default=10, type=int, help="Int point cloud step. Higher is the value, sparser the points.")

    args = parser.parse_args()
    fname_img = args.file_image
    fname_out = args.file_out
    step = args.step
    save_image_pointcloud(fname_in_img=fname_img, fname_out_pcd=fname_out, step=step)
