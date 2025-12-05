"""
3D Geometry utilities for back-projecting 2D points to 3D using depth maps.

This module provides functions to convert 2D pixel coordinates and depth values
into 3D points in camera coordinate space using the pinhole camera model.
"""

import numpy as np


def backproject_point_to_3d(x, y, depth_map, camera_matrix, depth_scale=1.0):
    """
    Back-project a 2D point to 3D using depth and camera intrinsics.

    Uses the pinhole camera model:
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        Z = depth

    Args:
        x, y: 2D pixel coordinates (float)
        depth_map: Depth map (H, W) with depth values
        camera_matrix: 3x3 camera intrinsic matrix K
                      [[fx,  0, cx],
                       [ 0, fy, cy],
                       [ 0,  0,  1]]
        depth_scale: Optional scale factor for depth values (default 1.0)

    Returns:
        (X, Y, Z): 3D point in camera coordinate system
        Returns (None, None, None) if coordinates are invalid

    Coordinate System:
        X: right (increasing x in image)
        Y: down (increasing y in image)
        Z: forward (away from camera)
    """
    h, w = depth_map.shape

    # Validate bounds
    x_int, y_int = int(round(x)), int(round(y))
    if not (0 <= x_int < w and 0 <= y_int < h):
        return None, None, None

    # Sample depth value at (x, y)
    Z = depth_map[y_int, x_int] * depth_scale

    # Check for invalid depth
    if Z <= 0 or not np.isfinite(Z):
        return None, None, None

    # Extract camera intrinsics
    fx = camera_matrix[0, 0]  # Focal length x
    fy = camera_matrix[1, 1]  # Focal length y
    cx = camera_matrix[0, 2]  # Principal point x
    cy = camera_matrix[1, 2]  # Principal point y

    # Back-projection equations (pinhole camera model)
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    return X, Y, Z


def backproject_lines_to_3d(lines_2d, depth_map, camera_matrix, depth_scale=1.0):
    """
    Back-project 2D lines to 3D lines in camera space.

    For each 2D line defined by two endpoints [x1, y1, x2, y2],
    this function back-projects both endpoints to 3D coordinates.

    Args:
        lines_2d: Array of 2D lines (N, 4) with format [x1, y1, x2, y2]
        depth_map: Depth map (H, W)
        camera_matrix: 3x3 camera intrinsic matrix
        depth_scale: Optional scale factor for depth values (default 1.0)

    Returns:
        lines_3d: Array of 3D lines (M, 6) with format [X1, Y1, Z1, X2, Y2, Z2]
                  where M <= N (lines with invalid depth are filtered out)
        valid_indices: List of indices of valid 3D lines
                      maps lines_3d back to lines_2d
                      e.g., lines_3d[i] corresponds to lines_2d[valid_indices[i]]

    Note:
        Lines are filtered out if either endpoint has invalid depth:
        - Out of bounds
        - Depth <= 0
        - Depth is inf or nan
    """
    if depth_map is None or camera_matrix is None:
        return np.array([]), []

    lines_3d = []
    valid_indices = []

    for idx, line in enumerate(lines_2d):
        x1, y1, x2, y2 = line

        # Back-project start point (x1, y1)
        X1, Y1, Z1 = backproject_point_to_3d(x1, y1, depth_map, camera_matrix, depth_scale)

        # Back-project end point (x2, y2)
        X2, Y2, Z2 = backproject_point_to_3d(x2, y2, depth_map, camera_matrix, depth_scale)

        # Check if both endpoints are valid
        if X1 is not None and X2 is not None:
            lines_3d.append([X1, Y1, Z1, X2, Y2, Z2])
            valid_indices.append(idx)

    return np.array(lines_3d), valid_indices


def compute_3d_line_length(line_3d):
    """
    Compute the Euclidean length of a 3D line.

    Args:
        line_3d: 3D line as array [X1, Y1, Z1, X2, Y2, Z2]

    Returns:
        length: Euclidean distance between the two 3D endpoints
    """
    X1, Y1, Z1, X2, Y2, Z2 = line_3d
    length = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)
    return length


def compute_3d_line_midpoint(line_3d):
    """
    Compute the midpoint of a 3D line.

    Args:
        line_3d: 3D line as array [X1, Y1, Z1, X2, Y2, Z2]

    Returns:
        midpoint: 3D coordinates of midpoint [X_mid, Y_mid, Z_mid]
    """
    X1, Y1, Z1, X2, Y2, Z2 = line_3d
    midpoint = np.array([(X1 + X2) / 2, (Y1 + Y2) / 2, (Z1 + Z2) / 2])
    return midpoint
