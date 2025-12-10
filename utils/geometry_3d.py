# 3D geometry stuff for the SLAM project
# converts 2D image coordinates to 3D using depth maps
# based on pinhole camera model (standard computer vision)

import numpy as np


def backproject_point_to_3d(x, y, depth_map, camera_matrix, depth_scale=1.0):
    # takes a 2D pixel coordinate and converts it to 3D using depth
    # uses the pinhole camera model (X = (x-cx)*Z/fx, etc.)

    # Args:
    #   x, y: pixel coordinates
    #   depth_map: the depth image
    #   camera_matrix: intrinsics matrix (from calibration)
    #   depth_scale: scale factor if needed

    h, w = depth_map.shape

    # make sure point is inside image
    xInt, yInt = int(round(x)), int(round(y))
    if not (0 <= xInt < w and 0 <= yInt < h):
        return None, None, None  # out of bounds

    # get depth value at this pixel
    Z = depth_map[yInt, xInt] * depth_scale

    # check if depth is valid
    if Z <= 0 or not np.isfinite(Z):
        return None, None, None  # invalid depth

    # extract camera parameters
    fx = camera_matrix[0, 0]  # focal length x
    fy = camera_matrix[1, 1]  # focal length y
    cx = camera_matrix[0, 2]  # principal point x (center)
    cy = camera_matrix[1, 2]  # principal point y

    # backprojection equations (standard pinhole model)
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    return X, Y, Z


def backproject_lines_to_3d(lines_2d, depth_map, camera_matrix, depth_scale=1.0):
    # converts 2D lines to 3D lines using depth
    # basically backprojects both endpoints of each line
    # returns only lines where both endpoints have valid depth

    if depth_map is None or camera_matrix is None:
        return np.array([]), []

    lines3d = []
    valid_idx = []

    # go through each 2D line
    for idx, line in enumerate(lines_2d):
        x1, y1, x2, y2 = line

        # backproject start point
        X1, Y1, Z1 = backproject_point_to_3d(x1, y1, depth_map, camera_matrix, depth_scale)

        # backproject end point
        X2, Y2, Z2 = backproject_point_to_3d(x2, y2, depth_map, camera_matrix, depth_scale)

        # only keep line if both points are valid
        if X1 is not None and X2 is not None:
            lines3d.append([X1, Y1, Z1, X2, Y2, Z2])
            valid_idx.append(idx)

    return np.array(lines3d), valid_idx


def compute_3d_line_length(line_3d):
    # computes the 3D length of a line (euclidean distance)
    X1, Y1, Z1, X2, Y2, Z2 = line_3d
    len3d = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)
    return len3d


def compute_3d_line_midpoint(line_3d):
    # gets the midpoint of a 3D line (average of endpoints)
    X1, Y1, Z1, X2, Y2, Z2 = line_3d
    mid = np.array([(X1 + X2) / 2, (Y1 + Y2) / 2, (Z1 + Z2) / 2])
    return mid
