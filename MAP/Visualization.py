# examples/Python/Basic/visualization.py
# Other:
#  https://stackoverflow.com/questions/65774814/adding-new-points-to-point-cloud-in-real-time-open3d

import argparse

import numpy as np
import open3d as o3d

def main(fname_ply):
    """ 
    Example of visualization program with Open3D
    Args:
        - **fname_ply*: Filename with ply data
    """
    try:
        print("Load a ply point cloud, print it, and render it")
        pcd = o3d.io.read_point_cloud(fname_ply)
    except:
        print(f'Failed to open:{fname_ply}')
        return

    try:
        print("Load a ply point cloud, print it, and render it")
        line_set = o3d.io.read_line_set('Open3D/my_lines.ply')
    except:
        print(f'Failed to open lines:filensma')
        return

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6,
                                                        origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, line_set, mesh_frame])
    return

    print("Let\'s draw some primitives")
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3, height=4.0)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6,
                                                        origin=[-2, -2, -2])

    print("We draw a few primitives using collection.")
    o3d.visualization.draw_geometries(
        [mesh_box, mesh_sphere, mesh_cylinder, mesh_frame])

    print("We draw a few primitives using + operator of mesh.")
    o3d.visualization.draw_geometries(
        [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])

    print("Let\'s draw a cubic using o3d.geometry.LineSet")
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
              [0, 1, 1], [1, 1, 1]]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([line_set])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--file_name", default="Open3D/fragment.ply", help="name of the point cloud file (ply,pcd)")

    args = parser.parse_args()
    fname_ply = args.file_name #'Open3D/fragment.ply'
    main(fname_ply=fname_ply)

