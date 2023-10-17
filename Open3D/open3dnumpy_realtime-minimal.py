# examples/Python/Basic/working_with_numpy.py

import copy
import numpy as np
import open3d as o3d

if __name__ == "__main__":

    # generate some neat n times 3 matrix using a variant of sync function
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    #xyz = np.array([[1,2,3],[4,5,6]])
    xyz = np.random.randint(5, size=(2, 3))
    print('xyz')
    print(xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    threshold = 0.05
    icp_iteration = 100
    save_image = False

    while True:
        xyz = np.random.uniform(0, 1, size=(100, 3))
        #print('xyz:{}'.format(xyz))
        pcd.points = o3d.utility.Vector3dVector(xyz)
        color = np.random.uniform(0, 1, size=(100, 3))
        #print('color:{}'.format(color))
        pcd.colors = o3d.utility.Vector3dVector(color)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        #exit(0)

    vis.destroy_window()