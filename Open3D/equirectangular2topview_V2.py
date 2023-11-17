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

from dictionary import AttrDict
from common import create_grid, calculate_plane_normal, uv2xyz, xyz_transform, xyz2mesh, xyz2camera, xy2image, ray_sphere_intersection_v2, direction_vector, xy2xyz, xyz2xy, xyz_transform, intersect_dome, xyz_transformV2, camera2xy, is_point_on_plane


def ground2image_base(p_ground, sensor_position, sensor_position_at_origin_must_be_0, sensor_orientation_deg, image_size_source):
    # Compute the direction vector
    dir_vect = direction_vector(p_ground, sensor_position)

    # The center of the sphere is inverted respect the plane point (confirm why)
    intersection_point = ray_sphere_intersection_v2(p_ground, dir_vect, sensor_position, 1.0)
    p3d, xi, yi = None, None, None
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
    return p3d, xi, yi

def image2ground_base(x, y, image_size_source, sensor_position, sensor_orientation_deg, dome_radius):
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

    intersection_point = intersect_dome(sensor_position, p3d_sphere_t - sensor_position, point_1, plane_normal, dome_radius, False)
    if intersection_point is not None:
        return p3d_sphere_t, intersection_point
    return None, None


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
        sensor_orientation_deg = np.array([config.x, config.y, config.z])

        # Source image
        image_size_source = [img_array.shape[1], img_array.shape[0]]

        # sensor camera position
        sensor_position = np.array([0, 0, config.sensor_height]) # meters

        Wbin = (config.Wmax - config.Wmin) / float(config.img_w)
        Hbin = (config.Hmax - config.Hmin) / float(config.img_h)
        z_ground = 0

        # Compute the points interval respect the dome
        if True:
            # Scan an image that represents a plane on the ground
            for y in range(config.img_h):
                for x in range(config.img_w):

                    # Compute current x on the plane ground
                    x_plane = Wbin * x + config.Wmin
                    y_plane = Hbin * y + config.Hmin
                    p_ground = np.array([x_plane, y_plane, z_ground])

                    p3d, xi, yi = ground2image_base(p_ground, sensor_position, sensor_position_at_origin_must_be_0, 
                                            sensor_orientation_deg, image_size_source)

                    if p3d is not None:                        
                        #print(f'p_ground:{p_ground} p_norm:{p_norm} sensor_position:{sensor_position} intersection_point:{intersection_point}')
                        data_res.append(p_ground) # point to ground
                        data_color_res.append(img_array[int(yi), int(xi), :])
                        #data_res.append(intersection_point)
                        data_res.append(p3d) # point to sphere
                        data_color_res.append(img_array[int(yi), int(xi), :])

            #cv2.imwrite('img0.jpg', img_out)
        data_res = np.array(data_res)
        data_color_res = np.array(data_color_res)

        # Image 2D
        img_out = np.zeros((config.img_h, config.img_w, 3), dtype = "uint8")
        print(f'img_out:{img_out.shape}')
        data = camera2xy(data_res.copy(), img_out.shape[1], img_out.shape[0], config.Wmin, config.Wmax, config.Hmin, config.Hmax)
        for i in range(data.shape[0]):
            img_out[int(data[i][1]), int(data[i][0]), 0] = data_color_res[i, 2]
            img_out[int(data[i][1]), int(data[i][0]), 1] = data_color_res[i, 1]
            img_out[int(data[i][1]), int(data[i][0]), 2] = data_color_res[i, 0]
        if config.save:
            cv2.imwrite('img0.jpg', img_out)
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

    return pc, img_out


def image2ground(img, step, mode, config):
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
        sensor_orientation_deg = np.array([config.x, config.y, config.z])

        # Source image
        image_size_source = [img_array.shape[1], img_array.shape[0]]

        # sensor camera position
        sensor_position = np.array([0, 0, config.sensor_height]) # meters

        # Compute the points interval respect the dome
        if True:
            for y in range(0, image_size_source[1], step):
                for x in range(0, image_size_source[0], step):

                    p3d_sphere_t, intersection_point = image2ground_base(x, y, image_size_source, 
                                                                         sensor_position, sensor_orientation_deg, config.dome_radius)                    
                    if intersection_point is not None:
                        data_res.append(p3d_sphere_t)
                        data_color_res.append(img_array[int(y), int(x), :])
                        data_res.append(intersection_point)
                        data_color_res.append(img_array[int(y), int(x), :])

        data_res = np.array(data_res)
        data_color_res = np.array(data_color_res)

        # Image 2D
        img_out = np.zeros((config.img_h, config.img_w, 3), dtype = "uint8")
        print(f'img_out:{img_out.shape}')
        data = camera2xy(data_res.copy(), img_out.shape[1], img_out.shape[0], config.Wmin, config.Wmax, config.Hmin, config.Hmax)
        for i in range(data.shape[0]):
            img_out[int(data[i][1]), int(data[i][0]), 0] = data_color_res[i, 2]
            img_out[int(data[i][1]), int(data[i][0]), 1] = data_color_res[i, 1]
            img_out[int(data[i][1]), int(data[i][0]), 2] = data_color_res[i, 0]
        if config.save:
            cv2.imwrite('img1.jpg', img_out)

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

    return pc, img_out


###############################################################################

def circle_points(sphere_center, sphere_radius, plane_point, plane_normal, num_points=32):
    # Calculate the distance between the sphere center and the plane
    distance = np.abs(np.dot(plane_normal, sphere_center - plane_point))

    # Check if the sphere and plane intersect
    if distance > sphere_radius:
        return []
    elif distance == sphere_radius:
        return [sphere_center - sphere_radius * plane_normal]
    else:
        # Calculate the intersection circle
        projection = sphere_center - distance * plane_normal
        radius = np.sqrt(sphere_radius ** 2 - distance ** 2)

        # Calculate the direction vector from the sphere center to the point on the plane
        direction_vector = projection - plane_point
        direction_vector /= np.linalg.norm(direction_vector) 
        print(f'direction_vector:{direction_vector} d:{np.dot(direction_vector, plane_normal)}')

        # Check the sign of the projection to determine the direction of the basis vectors
        if np.dot(direction_vector, plane_normal) > 0:
            # Use the original calculation for u and v
            u = np.cross(plane_normal, direction_vector)
            u /= np.linalg.norm(u)
        else:
            # Reverse the direction of the basis vectors
            u = -np.cross(plane_normal, direction_vector)
            u /= np.linalg.norm(u)
            #projection = -projection

        # Calculate the second basis vector (v)
        v = np.cross(plane_normal, u)
        v /= np.linalg.norm(v)

        # Check for NaN values in u and v
        if np.isnan(u).any() or np.isnan(v).any():
            # Handle NaN values in u or v
            print("NaN detected in u or v. Switching to default basis.")
            u = np.array([1, 0, 0])
            v = np.array([0, 1, 0])
            #if np.dot(direction_vector, plane_normal) < 0:
            #    projection = -projection

        # Calculate the circle points
        circle_points = [
            projection + radius * (np.cos(theta) * u + np.sin(theta) * v)
            for theta in np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        ]

        # check if original point is in the same plane
        normal = calculate_plane_normal(circle_points[0], circle_points[1], circle_points[2])

        # check if the point is on the same plane of the circle points
        # if False, invert the projection
        if is_point_on_plane(plane_point, circle_points[0], normal) is False:
            # Calculate the circle points with inverted projection
            circle_points = [
                -projection + radius * (np.cos(theta) * u + np.sin(theta) * v)
                for theta in np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            ]

        return circle_points

def estimate_circle_intesection_plane_sphere(p0, p1, p2, image_size, num_points):
    print(p0, p1, p2)
    if p0 is None or p1 is None or p2 is None:
        return None, None, None, None, None
    xyz0 = np.array(xy2xyz(p0[0], p0[1], image_size, 1))
    xyz1 = np.array(xy2xyz(p1[0], p1[1], image_size, 1))
    xyz2 = np.array(xy2xyz(p2[0], p2[1], image_size, 1))
    print(xyz0, xyz1, xyz2)

    normal = calculate_plane_normal(xyz0, xyz1, xyz2)
    sphere_center = np.array([0,0,0])
    sphere_radius = 1
    points = circle_points(sphere_center, sphere_radius, xyz0, normal, num_points=num_points)

    #print(f'points:{points}')
    points_image = []
    for p in points:
        xi, yi = xyz2xy(p[0], p[1], p[2], image_size)

        if xi < 0: xi = image_size[0] + xi
        if xi < 0: xi = 0
        if xi >= image_size[0]: xi = image_size[0] - 1
        if yi < 0: yi = 0
        if yi >= image_size[1]: yi = image_size[1] - 1
        points_image.append([xi, yi])
    #print(f'points_image:{points_image}')
    return xyz0, xyz1, xyz2, points, points_image


def sphere_plane_intersect(img, step, mode, config):
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
        sensor_orientation_deg = np.array([config.x, config.y, config.z])

        # Source image
        image_size_source = [img_array.shape[1], img_array.shape[0]]

        # sensor camera position
        sensor_position = np.array([0, 0, config.sensor_height]) # meters

        # Compute the points interval respect the dome
        if True:
            for y in range(0, image_size_source[1], step):
                for x in range(0, image_size_source[0], step):

                    xyz = np.array(xy2xyz(x, y, image_size_source, 1))

                    if xyz is not None:
                        data_res.append(xyz)
                        data_color_res.append(img_array[int(y), int(x), :])

            # Get 3 points on a sphere and compute the circle around it
            p0 = np.array([100, 200])
            p1 = np.array([900, 200])
            p2 = np.array([1700, 200])

            p0 = np.array([1142.16058203, 622.74247871])
            p1 = np.array([1119.99422627, 623.60575899])
            p2 = np.array([1094.90233285,  629.64594686])

            p0 = np.array([1612.90201193,  602.11861073])
            p1 = np.array([1595.5182475,   644.67360772])
            p2 = np.array([1507.36600957,  695.79626495])

            xyz0, xyz1, xyz2, points, points_image = estimate_circle_intesection_plane_sphere(p0, p1, p2, image_size_source, 100)
            data_res.append(xyz0)
            data_color_res.append(np.array([255, 0, 255]))
            data_res.append(xyz1)
            data_color_res.append(np.array([255, 0, 255]))
            data_res.append(xyz2)
            data_color_res.append(np.array([255, 0, 255]))
            for p in points:
                data_res.append(p)
                data_color_res.append(np.array([0, 255, 0]))

            # Draw the circle on the image source
            print(f'points_image:{points_image}')
            for p in points_image:
                cv2.circle(img_array, (int(p[0]), int(p[1])), 2, (0, 255, 0), 2)
            if config.save:
                cv2.imwrite('img_array.jpg', img_array)

            # Get 3 points on a sphere and compute the circle around it
            p0 = np.array([100, 880])
            p1 = np.array([900, 880])
            p2 = np.array([1700, 880])
            xyz0, xyz1, xyz2, points, points_image = estimate_circle_intesection_plane_sphere(p0, p1, p2, image_size_source, 110)
            data_res.append(xyz0)
            data_color_res.append(np.array([0, 255, 255]))
            data_res.append(xyz1)
            data_color_res.append(np.array([0, 255, 255]))
            data_res.append(xyz2)
            data_color_res.append(np.array([0, 255, 255]))
            for p in points:
                data_res.append(p)
                data_color_res.append(np.array([0, 255, 255]))

        data_res = np.array(data_res)
        data_color_res = np.array(data_color_res)

    else:
        assert(f'mode not implemented:{mode}')

    #print(f'data_res:{data_res}')

    # Reshape the point cloud data into a 2D array
    data_res = data_res.reshape(-1, 3)
    data_color_res = data_color_res.reshape(-1, 3)

    # Create a PointCloud object
    pc = o3d.geometry.PointCloud()

    # Set the point cloud data
    pc.points = o3d.utility.Vector3dVector(data_res)

    # Set the RGB colors for each point in the point cloud
    pc.colors = o3d.utility.Vector3dVector(data_color_res / 255)

    return pc, None

###############################################################################


def save_image_pointcloud(fname_in_img, fname_out_pcd, step, mode):
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

    config = dict()
    config['x'] = 15
    config['y'] = 0
    config['z'] = 25
    config['sensor_height'] = 1.7
    config['Wmin'] = -10.0
    config['Wmax'] = 10.0
    config['Hmin'] = -10.0
    config['Hmax'] = 10.0
    config['img_w'] = 512 #128
    config['img_h'] = 512 #128
    config['dome_radius'] = 15
    config['save'] = True
    config = AttrDict(config)


    if False:
        image_size_source = [1920, 1080]
        sensor_orientation_deg = np.array([config.x, config.y, config.z])
        # sensor camera position
        sensor_position = np.array([0, 0, config.sensor_height]) # meters
        p3d_sphere_t, intersection_point = image2ground_base(1400.8951381657796, 668.4494223502126, 
                                                             image_size_source, sensor_position, sensor_orientation_deg, config.dome_radius)
        print(f'intersection:{intersection_point}')
        exit(0)



    if mode == 'ground2image':
        pc, img = ground2image(img, step, 'uv', config)
    elif mode == 'image2ground':
        pc, img = image2ground(img, step, 'uv', config)
    elif mode =='sphereplane':
        pc, img = sphere_plane_intersect(img, step, 'uv', config)

    # Save the PointCloud object to a PCD file
    o3d.io.write_point_cloud(fname_out_pcd, pc)


# python Open3D/equirectangular2topview_V2.py --file_image=data/frame_sample.jpg --step=10 --mode='ground2image'
# python Open3D/equirectangular2topview_V2.py --file_image=data/frame_sample.jpg --step=10 --mode='image2ground'
# python Open3D/equirectangular2topview_V2.py --file_image=frame0.jpg --step=10 --mode='image2ground'
# python Open3D/equirectangular2topview_V2.py --file_image=data/frame_sample.jpg --step=10 --mode='sphereplane'

if __name__ == "__main__":

    print(f'If colors and data points size mismatch, no give color is used!!!')

    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--file_image", default='data/equirect001.jpeg', help="name of the equirectangular image.")
    parser.add_argument("--file_out", default='Open3D/result.pcd', help="name of the point cloud file (ply,pcd)")
    parser.add_argument("--step", default=10, type=int, help="Int point cloud step. Higher is the value, sparser the points.")
    parser.add_argument("--mode", default='ground2image', type=str, help="ground2image/image2ground")

    args = parser.parse_args()
    fname_img = args.file_image
    fname_out = args.file_out
    step = args.step
    save_image_pointcloud(fname_in_img=fname_img, fname_out_pcd=fname_out, step=step, mode=args.mode)
