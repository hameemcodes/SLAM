"""
3D Line Map for Visual SLAM
Stores 3D lines in world frame with descriptors for matching
"""
import numpy as np
from typing import Tuple, List, Optional


class LineMap:
    """Stores 3D lines in world frame with their descriptors."""
    
    def __init__(self):
        self.lines_3d = []        # List of [X1,Y1,Z1,X2,Y2,Z2] in world frame
        self.descriptors = []     # LBD descriptors for each line
        self.frame_ids = []       # Which frame each line came from
        
    def add_lines(self, lines_3d_cam: np.ndarray, descriptors: np.ndarray, 
                  R: np.ndarray, t: np.ndarray, frame_id: int):
        """
        Add 3D lines to map by transforming from camera frame to world frame.
        
        Args:
            lines_3d_cam: Nx6 array of [X1,Y1,Z1,X2,Y2,Z2] in camera frame
            descriptors: NxD array of LBD descriptors
            R: 3x3 rotation matrix (camera to world)
            t: 3x1 translation vector (camera to world)
            frame_id: Frame index for tracking
        """
        # Convert OpenCV camera convention -> ARKit camera convention
        # before applying ARKit's camera-to-world transform
        F = np.diag([1.0, -1.0, -1.0])
        for i, line in enumerate(lines_3d_cam):
            # Transform both endpoints to world frame
            P1_arkit_cam = F @ line[:3]
            P2_arkit_cam = F @ line[3:]
            P1_world = R @ P1_arkit_cam + t
            P2_world = R @ P2_arkit_cam + t
            
            self.lines_3d.append(np.concatenate([P1_world, P2_world]))
            self.descriptors.append(descriptors[i])
            self.frame_ids.append(frame_id)
    
    
    def get_lines_and_descriptors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return all 3D lines and descriptors as arrays."""
        if len(self.lines_3d) == 0:
            return np.array([]).reshape(0, 6), np.array([])
        return np.array(self.lines_3d), np.array(self.descriptors)
    
    def __len__(self):
        return len(self.lines_3d)


def backproject_lines(lines_2d: np.ndarray, depth: np.ndarray, K: np.ndarray,
                      rgb_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Back-project 2D lines to 3D using depth map.
    
    Args:
        lines_2d: Nx4 array of [x1,y1,x2,y2] in RGB pixel coordinates
        depth: HxW depth map (meters)
        K: 3x3 intrinsic matrix (for RGB resolution)
        rgb_shape: (H, W) of RGB image
        
    Returns:
        lines_3d: Mx6 array of valid 3D lines [X1,Y1,Z1,X2,Y2,Z2] in camera frame
        valid_indices: Which input lines have valid depth
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    depth_h, depth_w = depth.shape
    rgb_h, rgb_w = rgb_shape
    
    lines_3d = []
    valid_indices = []
    
    for i, line in enumerate(lines_2d):
        x1, y1, x2, y2 = line
        
        # Scale coordinates from RGB to depth resolution
        u1 = int(x1 * depth_w / rgb_w)
        v1 = int(y1 * depth_h / rgb_h)
        u2 = int(x2 * depth_w / rgb_w)
        v2 = int(y2 * depth_h / rgb_h)
        
        # Clamp to valid range
        u1, u2 = np.clip([u1, u2], 0, depth_w - 1)
        v1, v2 = np.clip([v1, v2], 0, depth_h - 1)
        
        # Get depth values
        Z1 = depth[v1, u1]
        Z2 = depth[v2, u2]
        
        # Skip if invalid depth
        if np.isnan(Z1) or np.isnan(Z2) or Z1 <= 0 or Z2 <= 0:
            continue
        
        # Back-project to 3D (pinhole model)
        X1 = (x1 - cx) * Z1 / fx
        Y1 = (y1 - cy) * Z1 / fy
        X2 = (x2 - cx) * Z2 / fx
        Y2 = (y2 - cy) * Z2 / fy
        
        lines_3d.append([X1, Y1, Z1, X2, Y2, Z2])
        valid_indices.append(i)
    
    if len(lines_3d) == 0:
        return np.array([]).reshape(0, 6), np.array([])
    
    return np.array(lines_3d), np.array(valid_indices)