"""
load data from Record3D app which contains LiDAR data and RGB images.
"""
import json
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Optional, List, Tuple, Generator
from dataclasses import dataclass

@dataclass
class FrameData:
    "data container which contains the RGB image, depth, camera calibration and pose of camera per frame"
    rgb_image: np.ndarray
    depth: np.ndarray
    K: np.ndarray #are we defining these as numpy arrays?
    pose: np.ndarray
    frame_idx: int #frame index
    timestamp: float #timestamp of the frame
class Record3DLoader:
    "lOADING FILES FROM RECORD3D APP"
    def __init__(self, data_path: str, max_frames: Optional[int] = None):
        self.data_path = Path(data_path)  # Convert string path to Path object
    
     # Load JSON file with metadata
        self.metadata = self._load_metadata()
    
        # Parse camera intrinsics (focal length, principal point)
        self.K = self._parse_intrinsics()

        # Parse camera poses (position & orientation for each frame)
        self.poses = self._parse_poses()
    
        # Get sorted lists of RGB and depth file paths
        self.rgb_files = self._get_rgb_files()
        self.depth_files = self._get_depth_files()
    
    # Determine total number of frames
        self.num_frames = min(len(self.rgb_files), len(self.depth_files), len(self.poses))
        if max_frames: #allows me to test with a smaller subset of frames if needed
            self.num_frames = min(self.num_frames, max_frames)

    def _load_metadata(self) -> dict:
        "loading metadata from JSON file and storing it in python dictionary"
        metadata_path = self.data_path / "metadata.json" #path to the metadata file

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    def _parse_intrinsics(self) -> np.ndarray:
        "parsing camera intrinsics from the metadata and returning it as a numpy array"
        K_data = self.metadata.get('K',None)

        if len(K_data) ==9: 
            # Flattened format: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
            fx = K_data[0]
            fy = K_data[4]
            cx = K_data[6]
            cy = K_data[7]
        elif len(K_data) >= 4:
            # Compact format: [fx, fy, cx, cy, ...]
            fx, fy, cx, cy = K_data[0], K_data[1], K_data[2], K_data[3]
        #should cover both cases where K is stored as a flattened 3x3 matrix or as a compact list of parameters
        K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
        ], dtype=np.float64)
        return K #forming the intrinsic matrix K using the parsed parameters - project 3D points onto the 2D image plane
    