import numpy as np
import open3d as o3d
from PIL import Image

def main(fname_in_img, fname_out_pcd):
    """
    Create a pcd file from image file.
    Args:
        - **fname_in_img*: Filename with the image.
        - **fname_out_pcd*: Filename where to save the pcd file.
    Note: The image is rotated 90 degrees anticlockwise.
    """
    # Load an image
    img = Image.open(fname_in_img)
    # Convert in numpy format
    img_array = np.array(img)

    # Create a numpy array of point cloud data with the same shape as the image
    # The output is in the range [0,1[
    data = np.zeros((img_array.shape[0], img_array.shape[1], 3))
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            data[i, j, 0] = i / img_array.shape[0]
            data[i, j, 1] = j / img_array.shape[1]
            data[i, j, 2] = 0 # Fix z on the origin

    # Reshape the point cloud data into a 2D array
    data = data.reshape(-1, 3)

    # Create a PointCloud object
    pc = o3d.geometry.PointCloud()

    # Set the point cloud data
    pc.points = o3d.utility.Vector3dVector(data)

    # Reshape the image data into a 2D array
    data_color = img_array.reshape(-1, 3)

    # Set the RGB colors for each point in the point cloud
    pc.colors = o3d.utility.Vector3dVector(data_color / 255)

    # Save the PointCloud object to a PCD file
    o3d.io.write_point_cloud(fname_out_pcd, pc)

if __name__ == "__main__":
    fname_img = 'data/equirect001.jpeg'
    fname_out = 'Open3D/result.pcd'
    main(fname_in_img=fname_img, fname_out_pcd=fname_out)
