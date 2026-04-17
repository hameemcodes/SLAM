"""
Record3D Data Loader for EXR + JPG Sequence Export

Loads data exported from Record3D app with "EXR + JPG sequence" option.

Expected folder structure:
    video1/
    ├── metadata.json    # Camera poses, intrinsics, timestamps
    ├── rgb/             # JPG images (0.jpg, 1.jpg, ...)
    └── depth/           # EXR depth maps (0.exr, 1.exr, ...)

Usage:
    loader = Record3DLoader('path/to/video1')
    
    for frame in loader:
        rgb = frame.rgb           # RGB image for M-LSD
        depth = frame.depth       # Depth map in meters
        K = frame.K               # Camera intrinsics
        pose = frame.pose         # 4x4 camera pose (ground truth!)

Requirements:
    pip install opencv-python numpy OpenEXR Imath
    
    If OpenEXR fails to install on Windows, try:
    pip install openexr-python
    
    Or use imageio:
    pip install imageio imageio-ffmpeg
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
    """Container for a single frame's data"""
    rgb: np.ndarray           # RGB image (H, W, 3), uint8
    depth: np.ndarray         # Depth map (H, W), float32, meters
    K: np.ndarray             # 3x3 camera intrinsic matrix
    pose: np.ndarray          # 4x4 camera-to-world transformation matrix
    frame_idx: int            # Frame index
    timestamp: float          # Timestamp in seconds


class Record3DLoader:
    """
    Load Record3D EXR + JPG sequence exports
    
    This loader handles the specific format from Record3D's 
    "EXR + JPG sequence (Touchdesigner-friendly)" export option.
    """
    
    def __init__(self, data_path: str, max_frames: Optional[int] = None):
        """
        Initialize loader
        
        Args:
            data_path: Path to exported folder (containing metadata.json, rgb/, depth/)
            max_frames: Optional limit on number of frames to load
        """
        self.data_path = Path(data_path)
        self.max_frames = max_frames
        
        # Validate path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Parse camera intrinsics
        self.K = self._parse_intrinsics()
        
        # Parse poses
        self.poses = self._parse_poses()
        
        # Get file lists
        self.rgb_files = self._get_rgb_files()
        self.depth_files = self._get_depth_files()
        
        # Determine number of frames
        self.num_frames = min(len(self.rgb_files), len(self.depth_files), len(self.poses))
        if max_frames:
            self.num_frames = min(self.num_frames, max_frames)
        
        print(f"[Record3D] Loaded {self.num_frames} frames from {data_path}")
        print(f"[Record3D] RGB resolution: {self.metadata.get('w', 'unknown')}x{self.metadata.get('h', 'unknown')}")
        print(f"[Record3D] FPS: {self.metadata.get('fps', 'unknown')}")
        print(f"[Record3D] Intrinsics: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}, cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}")
    
    def _load_metadata(self) -> dict:
        """Load metadata.json"""
        metadata_path = self.data_path / 'metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_path}")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def _parse_intrinsics(self) -> np.ndarray:
        """Parse camera intrinsics from metadata"""
        # Record3D stores K as a flattened 3x3 matrix (row-major, transposed):
        # [fx, 0, 0, 0, fy, 0, cx, cy, 1]
        K_data = self.metadata.get('K', None)
        
        if K_data is None:
            raise ValueError("No intrinsics (K) found in metadata")
        
        if len(K_data) == 9:
            # Flattened 3x3 matrix (transposed): [fx, 0, 0, 0, fy, 0, cx, cy, 1]
            fx = K_data[0]
            fy = K_data[4]
            cx = K_data[6]
            cy = K_data[7]
        elif len(K_data) >= 4:
            # Fallback: assume [fx, fy, cx, cy, ...]
            fx, fy, cx, cy = K_data[0], K_data[1], K_data[2], K_data[3]
        else:
            raise ValueError(f"Unexpected intrinsics format: {K_data}")
        
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)
        
        return K
    
    def _parse_poses(self) -> List[np.ndarray]:
        """Parse camera poses from metadata"""
        poses_data = self.metadata.get('poses', [])
        
        if not poses_data:
            print("[WARNING] No poses found in metadata, using identity")
            return [np.eye(4)]
        
        poses = []
        for pose_data in poses_data:
            # Record3D stores poses as 7 values: [qx, qy, qz, qw, tx, ty, tz]
            # Quaternion (x, y, z, w) + Translation (x, y, z)
            if len(pose_data) == 7:
                qx, qy, qz, qw, tx, ty, tz = pose_data
                pose = self._quat_to_matrix(qx, qy, qz, qw, tx, ty, tz)
            elif len(pose_data) == 16:
                # Already a 4x4 matrix
                pose = np.array(pose_data).reshape(4, 4)
            else:
                print(f"[WARNING] Unknown pose format with {len(pose_data)} values")
                pose = np.eye(4)
            
            poses.append(pose)
        
        return poses
    
    def _quat_to_matrix(self, qx: float, qy: float, qz: float, qw: float,
                        tx: float, ty: float, tz: float) -> np.ndarray:
        """Convert quaternion + translation to 4x4 transformation matrix"""
        # Normalize quaternion
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        
        # Build 4x4 matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        
        return T
    
    def _get_rgb_files(self) -> List[Path]:
        """Get sorted list of RGB files"""
        rgb_dir = self.data_path / 'rgb'
        
        if not rgb_dir.exists():
            raise FileNotFoundError(f"rgb/ folder not found in {self.data_path}")
        
        # Find all image files
        files = list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.png'))
        
        # Sort by numeric filename (0.jpg, 1.jpg, ..., 10.jpg, 11.jpg, ...)
        files = sorted(files, key=lambda x: int(x.stem))
        
        return files
    
    def _get_depth_files(self) -> List[Path]:
        """Get sorted list of depth files"""
        depth_dir = self.data_path / 'depth'
        
        if not depth_dir.exists():
            raise FileNotFoundError(f"depth/ folder not found in {self.data_path}")
        
        # Find all EXR files
        files = list(depth_dir.glob('*.exr'))
        
        # Sort by numeric filename
        files = sorted(files, key=lambda x: int(x.stem))
        
        return files
    
    def _load_rgb(self, path: Path) -> np.ndarray:
        """Load RGB image"""
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Failed to load RGB image: {path}")
        # Convert BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _load_depth_exr(self, path: Path) -> np.ndarray:
        """Load depth from EXR file"""
        # Try multiple methods to load EXR
        
        # Method 1: OpenEXR (most reliable)
        try:
            import OpenEXR
            import Imath
            
            exr_file = OpenEXR.InputFile(str(path))
            header = exr_file.header()
            
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # Try to find depth channel (could be 'Z', 'R', or 'Y')
            channels = list(header['channels'].keys())
            if not hasattr(self, '_exr_debug_done'):
                print(f"[EXR Debug] Available channels: {channels}")
                self._exr_debug_done = True
            depth_channel = None
            for ch in ['Z', 'R', 'Y', 'depth']:
                if ch in channels:
                    depth_channel = ch
                    break
            
            if depth_channel is None:
                # Just use first channel
                depth_channel = list(channels)[0]
            
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_str = exr_file.channel(depth_channel, pt)
            depth = np.frombuffer(depth_str, dtype=np.float32).reshape(height, width)
            
            if not hasattr(self, '_exr_method_printed'):
                print(f"[EXR] Loaded with OpenEXR, channel: {depth_channel}")
                self._exr_method_printed = True
            return depth
            
        except ImportError:
            pass
        except Exception as e:
            if not hasattr(self, '_exr_err_printed'):
                print(f"[EXR] OpenEXR error: {e}")
                self._exr_err_printed = True
        
        # Method 2: imageio
        try:
            import imageio.v3 as iio
            depth = iio.imread(str(path))
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]  # Take first channel
            if not hasattr(self, '_exr_method_printed'):
                print(f"[EXR] Loaded with imageio")
                self._exr_method_printed = True
            return depth.astype(np.float32)
        except (ImportError, Exception) as e:
            if not hasattr(self, '_imageio_err_printed'):
                print(f"[EXR] imageio error: {e}")
                self._imageio_err_printed = True
        
        # Method 3: cv2 with EXR support
        try:
            depth = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth is not None:
                if len(depth.shape) == 3:
                    depth = depth[:, :, 0]
                if not hasattr(self, '_exr_method_printed'):
                    print(f"[EXR] Loaded with OpenCV")
                    self._exr_method_printed = True
                return depth.astype(np.float32)
            else:
                print(f"[EXR] OpenCV returned None for {path}")
        except Exception as e:
            print(f"[EXR] OpenCV error: {e}")
        
        raise ImportError(
            "Could not load EXR file. Install one of:\n"
            "  pip install OpenEXR Imath\n"
            "  pip install imageio\n"
            "  Or on Windows: pip install openexr"
        )
    
    def get_frame(self, idx: int) -> Optional[FrameData]:
        """Get a single frame by index"""
        if idx < 0 or idx >= self.num_frames:
            return None
        
        # Load RGB
        rgb = self._load_rgb(self.rgb_files[idx])
        
        # Load depth
        depth = self._load_depth_exr(self.depth_files[idx])
        
        # Get pose
        pose = self.poses[idx] if idx < len(self.poses) else np.eye(4)
        
        # Get timestamp
        timestamps = self.metadata.get('timestamps', [])
        timestamp = timestamps[idx] if idx < len(timestamps) else idx / 60.0
        
        return FrameData(
            rgb=rgb,
            depth=depth,
            K=self.K.copy(),
            pose=pose.copy(),
            frame_idx=idx,
            timestamp=timestamp
        )
    
    def __len__(self) -> int:
        return self.num_frames
    
    def __getitem__(self, idx: int) -> FrameData:
        frame = self.get_frame(idx)
        if frame is None:
            raise IndexError(f"Frame index {idx} out of range [0, {self.num_frames})")
        return frame
    
    def __iter__(self) -> Generator[FrameData, None, None]:
        """Iterate over all frames"""
        for idx in range(self.num_frames):
            yield self.get_frame(idx)
    
    def get_intrinsics(self) -> np.ndarray:
        """Get camera intrinsic matrix"""
        return self.K.copy()
    
    def get_all_poses(self) -> np.ndarray:
        """Get all poses as (N, 4, 4) array"""
        return np.array(self.poses[:self.num_frames])
    
    def get_trajectory(self) -> np.ndarray:
        """Get camera positions as (N, 3) array"""
        poses = self.get_all_poses()
        return poses[:, :3, 3]


def get_depth_at_rgb_point(u: float, v: float, 
                           depth_map: np.ndarray,
                           rgb_shape: Tuple[int, int]) -> float:
    """
    Get depth value at RGB image coordinate
    
    Since RGB and depth have different resolutions, this function
    handles the coordinate scaling.
    
    Args:
        u, v: Pixel coordinates in RGB image
        depth_map: Depth map (different resolution than RGB)
        rgb_shape: (height, width) of RGB image
        
    Returns:
        Depth value in meters
    """
    depth_h, depth_w = depth_map.shape
    rgb_h, rgb_w = rgb_shape
    
    # Scale coordinates
    u_depth = int(u * depth_w / rgb_w)
    v_depth = int(v * depth_h / rgb_h)
    
    # Clamp to valid range
    u_depth = np.clip(u_depth, 0, depth_w - 1)
    v_depth = np.clip(v_depth, 0, depth_h - 1)
    
    return depth_map[v_depth, u_depth]


def get_scaled_intrinsics(K: np.ndarray, 
                          rgb_shape: Tuple[int, int],
                          depth_shape: Tuple[int, int]) -> np.ndarray:
    """
    Get intrinsics scaled for depth map resolution
    
    Args:
        K: Original intrinsics (for RGB resolution)
        rgb_shape: (height, width) of RGB
        depth_shape: (height, width) of depth
        
    Returns:
        K_depth: Intrinsics scaled for depth resolution
    """
    scale_x = depth_shape[1] / rgb_shape[1]
    scale_y = depth_shape[0] / rgb_shape[0]
    
    K_depth = K.copy()
    K_depth[0, 0] *= scale_x  # fx
    K_depth[1, 1] *= scale_y  # fy
    K_depth[0, 2] *= scale_x  # cx
    K_depth[1, 2] *= scale_y  # cy
    
    return K_depth


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python record3d_loader.py <path_to_video_folder>")
        print("\nExample: python record3d_loader.py C:/Users/hamee/Downloads/videos/video1")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    print("=" * 60)
    print("Record3D Loader Test")
    print("=" * 60)
    
    try:
        loader = Record3DLoader(data_path, max_frames=10)
        
        print(f"\nTotal frames: {len(loader)}")
        print(f"\nCamera intrinsics K:")
        print(loader.K)
        
        # Load first frame
        print("\n--- First Frame ---")
        frame = loader[0]
        print(f"RGB shape: {frame.rgb.shape}")
        print(f"Depth shape: {frame.depth.shape}")
        print(f"Depth range: {frame.depth.min():.3f} - {frame.depth.max():.3f} meters")
        print(f"Timestamp: {frame.timestamp:.3f}s")
        print(f"Pose:\n{frame.pose}")
        
        # Show trajectory stats
        trajectory = loader.get_trajectory()
        print(f"\n--- Trajectory ---")
        print(f"Start position: {trajectory[0]}")
        print(f"End position: {trajectory[-1]}")
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        print(f"Total distance: {total_distance:.3f} meters")
        
        # Visualize first frame
        print("\n--- Visualization ---")
        print("Press any key to continue, 'q' to quit")
        
        rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        
        # Depth visualization (normalize and colormap)
        depth_valid = frame.depth.copy()
        depth_valid[depth_valid <= 0] = np.nan
        depth_min, depth_max = np.nanmin(depth_valid), np.nanmax(depth_valid)
        depth_norm = ((depth_valid - depth_min) / (depth_max - depth_min) * 255)
        depth_norm = np.nan_to_num(depth_norm, nan=0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        
        # Resize depth to match RGB for display
        depth_color = cv2.resize(depth_color, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
        
        # Add text
        cv2.putText(rgb_bgr, f"Frame 0 | RGB", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(depth_color, f"Depth: {depth_min:.2f}-{depth_max:.2f}m", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('RGB', rgb_bgr)
        cv2.imshow('Depth', depth_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\n[SUCCESS] Loader test complete!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()