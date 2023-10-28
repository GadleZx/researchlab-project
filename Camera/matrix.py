import numpy as np

def func_rotation_matrix(x, y, z):
  """Generates a rotation matrix from a tuple of 3 values (x, y, z).

  Args:
    x: The x-axis rotation angle in degrees.
    y: The y-axis rotation angle in degrees.
    z: The z-axis rotation angle in degrees.

  Returns:
    A 4x4 NumPy array representing the rotation matrix.
  """

  # Convert the rotation angles to radians.
  x_rad = np.radians(x)
  y_rad = np.radians(y)
  z_rad = np.radians(z)

  # Create the rotation matrices for each axis.
  rx = np.array([[1, 0, 0, 0],
                  [0, np.cos(x_rad), -np.sin(x_rad), 0],
                  [0, np.sin(x_rad), np.cos(x_rad), 0],
                  [0, 0, 0, 1]])

  ry = np.array([[np.cos(y_rad), 0, np.sin(y_rad), 0],
                  [0, 1, 0, 0],
                  [-np.sin(y_rad), 0, np.cos(y_rad), 0],
                  [0, 0, 0, 1]])

  rz = np.array([[np.cos(z_rad), -np.sin(z_rad), 0, 0],
                  [np.sin(z_rad), np.cos(z_rad), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

  # Multiply the rotation matrices to get the combined rotation matrix.
  rotation_matrix = rx @ ry @ rz

  return rotation_matrix


def func_rotation_translation_matrix(rotation, translation):
  """Creates a rotation translation matrix from the given parameters.

  Args:
    rotation: A 3x3 NumPy array representing the rotation matrix.
    translation: A 3x1 NumPy array representing the translation vector.

  Returns:
    A 4x4 NumPy array representing the rotation translation matrix.
  """

  rotation_translation_matrix = np.array([
      [rotation[0, 0], rotation[0, 1], rotation[0, 2], translation[0]],
      [rotation[1, 0], rotation[1, 1], rotation[1, 2], translation[1]],
      [rotation[2, 0], rotation[2, 1], rotation[2, 2], translation[2]],
      [0.0, 0.0, 0.0, 1.0]
  ])

  return rotation_translation_matrix


def func_rotation_translation_matrix(rotation, translation):
  """Creates a rotation translation matrix from the given parameters.

  Args:
    rotation: A 3x3 NumPy array representing the rotation matrix.
    translation: A 3x1 NumPy array representing the translation vector.

  Returns:
    A 4x4 NumPy array representing the rotation translation matrix.
  """

  rotation_translation_matrix = np.array([
      [rotation[0, 0], rotation[0, 1], rotation[0, 2], 0],
      [rotation[1, 0], rotation[1, 1], rotation[1, 2], 0],
      [rotation[2, 0], rotation[2, 1], rotation[2, 2], 0],
      [translation[0], translation[1], translation[2], 1.0]
  ])

  return rotation_translation_matrix


def func_projection_matrix(fov, aspect_ratio, near_plane, far_plane):
  """Creates a projection matrix from the given parameters.

  Args:
    fov: The field of view in degrees.
    aspect_ratio: The aspect ratio of the image.
    near_plane: The distance to the near plane.
    far_plane: The distance to the far plane.

  Returns:
    A 4x4 NumPy array representing the projection matrix.
  """

  f = 1.0 / np.tan(fov * np.pi / 360.0)

  projection_matrix = np.array([
      [f / aspect_ratio, 0.0, 0.0, 0.0],
      [0.0, f, 0.0, 0.0],
      [0.0, 0.0, -(far_plane + near_plane) / (far_plane - near_plane), -1.0],
      [0.0, 0.0, -2.0 * far_plane * near_plane / (far_plane - near_plane), 1.0]
  ])

  return projection_matrix


def func_camera_matrix_ng(position, orientation, fov, aspect_ratio, near_plane, far_plane):
  """Creates a camera matrix from the given parameters.

  Returns:
    A 4x4 NumPy array representing the camera matrix.
  """

  # Calculate the projection matrix.
  projection_matrix = func_projection_matrix(fov, aspect_ratio, near_plane, far_plane)
  print(f'projection_matrix:{projection_matrix}')
  rotation_matrix = func_rotation_matrix(orientation[0], orientation[1], orientation[2])
  print(f'rotation_matrix:{projection_matrix}')
  rotation_translation_matrix = func_rotation_translation_matrix(rotation_matrix, position)
  print(f'rotation_translation_matrix:{rotation_translation_matrix}')

  # Rotate the projection matrix to the camera orientation.
  camera_matrix = projection_matrix @ rotation_translation_matrix
  print(f'camera_matrix:{camera_matrix}')

  return camera_matrix


def func_camera_matrix(position, orientation, field_of_view, aspect_ratio):
  """Creates a camera matrix from the given parameters.

  Args:
    position: A 3x1 NumPy array representing the position of the camera.
    orientation: A 3x3 NumPy array representing the orientation of the camera.
    field_of_view: The field of view in degrees.
    aspect_ratio: The aspect ratio of the image.

  Returns:
    A 4x4 NumPy array representing the camera matrix.

  Note:
    depth = -camera_matrix[2,3] / camera_matrix[2,0] * projected_point[0] + 
            -camera_matrix[2,3] / camera_matrix[2,1] * projected_point[1] + 
            camera_matrix[2,3]
  """

  # Calculate the focal length.

  focal_length = 1.0 / np.tan(field_of_view * np.pi / 360.0)

  # Calculate the distance between the camera and the points being projected

  distance = position[2]

  # Create the camera matrix.

  camera_matrix = np.array([
      [focal_length / aspect_ratio, 0.0, position[0], 0.0],
      [0.0, focal_length, position[1], 0.0],
      [0.0, 0.0, -1.0, 0.0],
      [0.0, 0.0, -distance, 1.0]
  ])

  # Rotate the camera matrix by the camera orientation.

  camera_matrix = camera_matrix @ orientation

  return camera_matrix


def apply_camera_matrix(point_cloud, projection_matrix):
  """Applies the given projection matrix to the given point cloud data.

  Args:
    point_cloud: A 3D NumPy array representing the point cloud data.
    projection_matrix: A 4x4 NumPy array representing the projection matrix.

  Returns:
    A 2D NumPy array representing the projected point cloud data.
  """

  # Add a fourth column to the point cloud data to represent the homogeneous coordinate.
  point_cloud_with_homogeneous_coordinate = np.hstack([point_cloud, np.ones((point_cloud.shape[0], 1))])

  # Transform the point cloud data with the projection matrix.
  projected_point_cloud = point_cloud_with_homogeneous_coordinate @ projection_matrix

  print(f'projected_point_cloud:{projected_point_cloud}')

  r = projected_point_cloud[:, 3]
  print(f'r:{r}')

  # Divide the projected point cloud data by the fourth column to get the normalized projection.
  projected_point_cloud = projected_point_cloud / projected_point_cloud[:, 3][:, None]

  # Set the projected points that are behind the camera to zero.
  projected_point_cloud[projected_point_cloud[:, 3] < 0] = 0

  return projected_point_cloud


def transform_points(roto_translation_matrix, points):
  """Transform 3D points in 3D

  Args:
    roto_translation_matrix: A 4x4 NumPy array representing the rotation-translation matrix.
    projection_matrix: A 4x4 NumPy array representing the projection matrix.
    points: A Nx3 NumPy array representing the 3D points.

  Returns:
    A Nx3 NumPy array representing the transformed 3D points.
  """

  #print('#####################################')
  #print(f'roto_translation_matrix:{roto_translation_matrix}')
  #print(f'projection_matrix:{projection_matrix}')
  #print(f'points:{points}')

  # Add a fourth column to the point cloud data to represent the homogeneous coordinate.
  point_cloud_with_homogeneous_coordinate = np.hstack([points, np.ones((points.shape[0], 1))])  

  #print(f'point_cloud_with_homogeneous_coordinate:{point_cloud_with_homogeneous_coordinate}')

  # Transform the 3D points by the rotation-translation matrix.
  transformed_points = point_cloud_with_homogeneous_coordinate @ roto_translation_matrix

  # Remove the last dimension
  return transformed_points[:, :3]


def project_points(roto_translation_matrix, projection_matrix, points):
  """Projects 3D points into 2D pixel coordinates using the given parameters.

  Args:
    roto_translation_matrix: A 4x4 NumPy array representing the rotation-translation matrix.
    projection_matrix: A 4x4 NumPy array representing the projection matrix.
    points: A Nx3 NumPy array representing the 3D points.

  Returns:
    A Nx2 NumPy array representing the projected 2D points.
  """

  #print('#####################################')
  #print(f'roto_translation_matrix:{roto_translation_matrix}')
  #print(f'projection_matrix:{projection_matrix}')
  #print(f'points:{points}')

  # Add a fourth column to the point cloud data to represent the homogeneous coordinate.
  point_cloud_with_homogeneous_coordinate = np.hstack([points, np.ones((points.shape[0], 1))])  

  #print(f'point_cloud_with_homogeneous_coordinate:{point_cloud_with_homogeneous_coordinate}')

  # Transform the 3D points by the rotation-translation matrix.
  transformed_points = point_cloud_with_homogeneous_coordinate @ roto_translation_matrix

  #print(f'transformed_points:{transformed_points}')

  # Project the transformed 3D points into 2D pixel coordinates.
  projected_points = transformed_points @ projection_matrix

  #print(f'projected_points:{projected_points}')

  # Normalize the projected points.
  projected_points /= projected_points[:, 3][:, None]

  #print(f'projected_points_norm:{projected_points}')

  # Set the projected points that are behind the camera to zero.
  projected_points[projected_points[:, 3] < 0] = 0

  return projected_points


if False:
  position = np.array([0.0, 0.0, 0.0])
  orientation = np.array([0.0, 0.0, 0.0])
  vertical_field_of_view = 60.0
  horizontal_field_of_view = 90.0

  camera_matrix = func_camera_matrix(position, orientation, 90, 16. / 16., 0.1, 100.0)

  print(camera_matrix)

  # Example usage:

  point_cloud = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0]
  ])

  projection_matrix = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, -1.0],
    [0.0, 0.0, 0.0, 1.0]
  ])


  projected_point_cloud = apply_camera_matrix(point_cloud, projection_matrix)
  print(projected_point_cloud)


  projected_point_cloud = apply_camera_matrix(point_cloud, camera_matrix)
  print(f'projection_matrix:{projection_matrix}')
  print(f'projected_point_cloud:{projected_point_cloud}')
