"""
TUM RGB-D Benchmark Dataset Loader

Loads data from the TUM RGB-D benchmark (Sturm et al., 2012) and provides
the same FrameData interface as Record3DLoader so that the SLAM pipeline
works without modification.

Expected folder structure:
    rgbd_dataset_freiburg1_xyz/
    ├── groundtruth.txt    # timestamp tx ty tz qx qy qz qw
    ├── rgb.txt            # timestamp filename
    ├── depth.txt          # timestamp filename
    ├── rgb/               # PNG images (timestamped filenames)
    └── depth/             # 16-bit PNG depth maps (depth_mm / 5000 = metres)

Key differences from Record3D:
    - Depth is 16-bit PNG, divide pixel value by 5000 to get metres
    - RGB and depth have different timestamps → associate by nearest time
    - Ground truth is at yet another rate → associate by nearest time
    - Camera convention is standard OpenCV (no ARKit Y-flip)

The loader adjusts TUM poses so they are compatible with the existing
pipeline's ARKit convention assumption. This means the F = diag(1,-1,-1)
flips in runSLAM2.py and map_3d.py cancel out correctly.

Usage:
    loader = TUMLoader('path/to/rgbd_dataset_freiburg1_xyz')
    for frame in loader:
        rgb = frame.rgb        # (480, 640, 3) uint8
        depth = frame.depth    # (480, 640) float32, metres
        K = frame.K            # 3x3 intrinsics
        pose = frame.pose      # 4x4 camera-to-world

References:
    Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems",
    IROS 2012.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Generator
from dataclasses import dataclass


# Reuse the same FrameData container
@dataclass
class FrameData:
    """Container for a single frame's data (same as LIDAR_loader)."""
    rgb: np.ndarray           # RGB image (H, W, 3), uint8
    depth: np.ndarray         # Depth map (H, W), float32, meters
    K: np.ndarray             # 3x3 camera intrinsic matrix
    pose: np.ndarray          # 4x4 camera-to-world transformation matrix
    frame_idx: int            # Frame index
    timestamp: float          # Timestamp in seconds


# ── TUM freiburg camera intrinsics ──────────────────────────────────────
# Source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats

TUM_INTRINSICS = {
    "freiburg1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3},
    "freiburg2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7},
    "freiburg3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
}


class TUMLoader:
    """
    Load TUM RGB-D benchmark datasets.

    Provides the same interface as Record3DLoader so that LineSLAM
    can use either data source without code changes.
    """

    def __init__(self, data_path: str, max_frames: Optional[int] = None,
                 camera: str = "freiburg1", max_time_diff: float = 0.02):
        """
        Args:
            data_path: Path to dataset folder (e.g. rgbd_dataset_freiburg1_xyz)
            max_frames: Optional limit on number of frames
            camera: Which freiburg camera for intrinsics ("freiburg1", "freiburg2", "freiburg3")
            max_time_diff: Maximum time difference (seconds) when associating
                           RGB, depth, and ground truth timestamps. Pairs with
                           larger gaps are discarded.
        """
        self.data_path = Path(data_path)
        self.max_frames = max_frames
        self.max_time_diff = max_time_diff

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # Parse intrinsics
        intr = TUM_INTRINSICS.get(camera, TUM_INTRINSICS["freiburg1"])
        self.K = np.array([
            [intr["fx"], 0,          intr["cx"]],
            [0,          intr["fy"],  intr["cy"]],
            [0,          0,           1]
        ], dtype=np.float64)

        # Parse file lists
        self.rgb_list = self._parse_file_list("rgb.txt")
        self.depth_list = self._parse_file_list("depth.txt")
        self.gt_list = self._parse_groundtruth("groundtruth.txt")

        # Associate RGB ↔ depth ↔ ground truth by timestamps
        self.associations = self._associate_all()

        self.num_frames = len(self.associations)
        if max_frames:
            self.num_frames = min(self.num_frames, max_frames)

        print(f"[TUM] Loaded {self.num_frames} frames from {data_path}")
        print(f"[TUM] Camera: {camera}")
        print(f"[TUM] Intrinsics: fx={intr['fx']:.1f}, fy={intr['fy']:.1f}, "
              f"cx={intr['cx']:.1f}, cy={intr['cy']:.1f}")
        print(f"[TUM] RGB frames: {len(self.rgb_list)}, "
              f"Depth frames: {len(self.depth_list)}, "
              f"GT poses: {len(self.gt_list)}")
        print(f"[TUM] Associated triplets: {len(self.associations)}")

    # ── File parsing ────────────────────────────────────────────────────

    def _parse_file_list(self, filename: str) -> List[Tuple[float, str]]:
        """Parse rgb.txt or depth.txt → list of (timestamp, filepath)."""
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"{filename} not found in {self.data_path}")

        entries = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                timestamp = float(parts[0])
                path = parts[1]
                entries.append((timestamp, path))
        return entries

    def _parse_groundtruth(self, filename: str) -> List[Tuple[float, np.ndarray]]:
        """Parse groundtruth.txt → list of (timestamp, 4x4 pose matrix)."""
        filepath = self.data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"{filename} not found in {self.data_path}")

        entries = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                timestamp = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = (float(parts[4]), float(parts[5]),
                                   float(parts[6]), float(parts[7]))
                pose = self._quat_to_matrix(qx, qy, qz, qw, tx, ty, tz)
                entries.append((timestamp, pose))
        return entries

    # ── Timestamp association ───────────────────────────────────────────

    def _associate_all(self) -> List[Tuple[int, str, str, np.ndarray, float]]:
        """
        Associate RGB, depth, and ground truth by nearest timestamp.

        Returns list of (index, rgb_path, depth_path, pose_4x4, timestamp).
        """
        rgb_times = np.array([t for t, _ in self.rgb_list])
        depth_times = np.array([t for t, _ in self.depth_list])
        gt_times = np.array([t for t, _ in self.gt_list])

        associations = []
        idx = 0

        for i, (rgb_t, rgb_path) in enumerate(self.rgb_list):
            # Find closest depth
            depth_idx = np.argmin(np.abs(depth_times - rgb_t))
            depth_diff = abs(depth_times[depth_idx] - rgb_t)
            if depth_diff > self.max_time_diff:
                continue

            # Find closest ground truth
            gt_idx = np.argmin(np.abs(gt_times - rgb_t))
            gt_diff = abs(gt_times[gt_idx] - rgb_t)
            if gt_diff > self.max_time_diff:
                continue

            depth_path = self.depth_list[depth_idx][1]
            pose = self.gt_list[gt_idx][1]

            associations.append((idx, rgb_path, depth_path, pose, rgb_t))
            idx += 1

        return associations

    # ── Quaternion → matrix ─────────────────────────────────────────────

    def _quat_to_matrix(self, qx, qy, qz, qw, tx, ty, tz) -> np.ndarray:
        """
        Convert TUM quaternion + translation to 4x4 camera-to-world matrix.

        TUM poses are in standard OpenCV convention. The existing SLAM
        pipeline assumes ARKit convention (Y-up, Z-backward) because of
        the F = diag(1,-1,-1) flip in map_3d.py and runSLAM2.py.

        To make the pipeline work unchanged, we adjust the rotation:
            R_stored = R_tum @ F
        so that when the code does R_stored @ F @ P_opencv, the two F
        matrices cancel and we get R_tum @ P_opencv — the correct result.
        """
        # Normalise quaternion
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])

        # Adjust for pipeline's ARKit assumption: R_stored = R_tum @ F
        F = np.diag([1.0, -1.0, -1.0])
        R_adjusted = R @ F

        T = np.eye(4)
        T[:3, :3] = R_adjusted
        T[:3, 3] = [tx, ty, tz]
        return T

    # ── Frame loading ───────────────────────────────────────────────────

    def get_frame(self, idx: int) -> Optional[FrameData]:
        """Get a single frame by index."""
        if idx < 0 or idx >= self.num_frames:
            return None

        _, rgb_path, depth_path, pose, timestamp = self.associations[idx]

        # Load RGB
        rgb_full = self.data_path / rgb_path
        rgb = cv2.imread(str(rgb_full))
        if rgb is None:
            raise IOError(f"Failed to load RGB: {rgb_full}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load depth (16-bit PNG, divide by 5000 to get metres)
        depth_full = self.data_path / depth_path
        depth_raw = cv2.imread(str(depth_full), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise IOError(f"Failed to load depth: {depth_full}")
        depth = depth_raw.astype(np.float32) / 5000.0
        # Mark invalid depth (0 in raw = no measurement)
        depth[depth_raw == 0] = 0.0

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
        for idx in range(self.num_frames):
            yield self.get_frame(idx)

    def get_intrinsics(self) -> np.ndarray:
        return self.K.copy()

    def get_all_poses(self) -> np.ndarray:
        return np.array([a[3] for a in self.associations[:self.num_frames]])

    def get_trajectory(self) -> np.ndarray:
        poses = self.get_all_poses()
        return poses[:, :3, 3]


# ── Quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python TUM_loader.py <path_to_tum_dataset>")
        print("Example: python TUM_loader.py rgbd_dataset_freiburg1_xyz")
        sys.exit(1)

    loader = TUMLoader(sys.argv[1], max_frames=10)

    frame = loader[0]
    print(f"\nFirst frame:")
    print(f"  RGB shape: {frame.rgb.shape}")
    print(f"  Depth shape: {frame.depth.shape}")
    print(f"  Depth range: {frame.depth[frame.depth > 0].min():.3f} - "
          f"{frame.depth[frame.depth > 0].max():.3f} m")
    print(f"  Pose:\n{frame.pose}")

    traj = loader.get_trajectory()
    total_dist = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    print(f"\nTrajectory: {total_dist:.3f} m over {len(loader)} frames")
