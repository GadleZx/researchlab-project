import math
import numpy as np
import open3d as o3d
from PIL import Image

def uv2xyz(image_size, step):
    """
    It converts the content of the data structure from image pixels to cartesian coordinates.
    Args:
        - image_size: Size of the image source (width, height), e.g. (640, 480).
        - step: Point spacing.
    Returns:
        A NumPy array with the shape (height, width, 3), where:
        - Channel 0 contains the X coordinate.
        - Channel 1 contains the Y coordinate.
        - Channel 2 contains the Z coordinate.
    """
    data = np.zeros((round(image_size[1] / step), round(image_size[0] / step), 3))
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
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

def xyz2uv(data, image_size, step):
    new_uvs = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x, y, z = data[i, j]

            theta = math.atan2(y, x)
            phi = math.acos(z)
            new_phi = phi + (math.pi / 2.0)

            new_u = theta / (2.0 * math.pi)
            new_v = new_phi / math.pi

            new_pixel_u = int(new_u * image_size[0] / step)
            new_pixel_v = int(new_v * image_size[1] / step)

            new_uvs.append((new_pixel_u, new_pixel_v))

    return new_uvs

def main(fname_in_img, fname_out_pcd):
    # Load an image
    img = Image.open(fname_in_img)
    img_array = np.array(img)

    # Create a numpy array of point cloud data using uv2xyz
    data = uv2xyz((img_array.shape[1], img_array.shape[0]), 1)

    # Use xyztouv to transform the data
    new_uvs = xyz2uv(data, (img_array.shape[1], img_array.shape[0]), 1)

    # Create a PointCloud object
    pc = o3d.geometry.PointCloud()

    # Set the point cloud data
    pc.points = o3d.utility.Vector3dVector(data.reshape(-1, 3))

    # Reshape the image data into a 2D array
    colors = img_array.reshape(-1, 3) / 255.0

    # Set the RGB colors for each point in the point cloud
    pc.colors = o3d.utility.Vector3dVector(colors)

    # Save the PointCloud object to a PCD file
    o3d.io.write_point_cloud(fname_out_pcd, pc)

if __name__ == "__main__":
    fname_img = 'data/frame_sample.jpg'
    fname_out = 'Open3D/result.pcd'
    main(fname_img, fname_out)
