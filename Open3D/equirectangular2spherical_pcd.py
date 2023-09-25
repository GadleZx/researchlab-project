# http://www.paul-reed.co.uk/programming.html
# https://paulbourke.net/panorama/icosahedral/
# https://github.com/rdbch/icosahedral_sampler

import math
import argparse

import numpy as np
import open3d as o3d
from PIL import Image

def uv2xyz(image_size, step):
    """
    It converts the content of the data structure from image pixels to cartesian coordinates
    Args:
        - **image_size**-: Size of the image source (WH). i.e. (640,480).
        - **step**: Points step
    Removed:
        - **data**: Container with the pixel points (2D) to transform.

    The function expects to return  data in the format HWC (height,width,channels)
    Channel 0 will contain the position in X.
    Channel 1 will contain the position in Y.
    Channel 2 will contain the position in Z.
    """
    #print(f'imge_size:{image_size} bin:{bin}')
    data = np.zeros((round(image_size[1] / step), round(image_size[0] / step), 3))
    print(f'data:{data.shape}')
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            # convert in uv -> spherical -> cartesian coordinates
            u = float(j * step) / image_size[0]
            v = float(i * step) / image_size[1]
            theta = u * 2.0 * math.pi
            phi = v * math.pi

            x = math.cos(theta) * math.sin(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(phi)
            data[i, j, 0] = x
            data[i, j, 1] = y
            data[i, j, 2] = z
    return data

def image2pointcloud(img, step):
    """
    Create a pcd file from image file.
    Args:
        - **img*: image (PIL).
        - **step**: Point cloud step.
    Return:
        - Point cloud structrue
    Note: The image is rotated 90 degrees anticlockwise.
    """
    # Convert in numpy format
    img_array = np.array(img)

    # Create a numpy array of point cloud data with the same shape as the image
    # The output is in the range [0,1[
    #data = np.zeros((img_array.shape[0], img_array.shape[1], 3))
    #for i in range(img_array.shape[0]):
    #    for j in range(img_array.shape[1]):
    #        data[i, j, 0] = i / img_array.shape[0]
    #        data[i, j, 1] = j / img_array.shape[1]
    #        data[i, j, 2] = 0 # Fix z on the origin
    data = uv2xyz((img_array.shape[1], img_array.shape[0]), step)

    # Reshape the point cloud data into a 2D array
    data = data.reshape(-1, 3)

    # Create a PointCloud object
    pc = o3d.geometry.PointCloud()

    # Set the point cloud data
    pc.points = o3d.utility.Vector3dVector(data)

    # image slicing
    if step > 1:
        img_array = img_array[::step,::step,:]
        print(f'img_array:{img_array.shape}')
    # Reshape the image data into a 2D array
    data_color = img_array.reshape(-1, 3)

    # Set the RGB colors for each point in the point cloud
    pc.colors = o3d.utility.Vector3dVector(data_color / 255)

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

    pc = image2pointcloud(img, step)

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
