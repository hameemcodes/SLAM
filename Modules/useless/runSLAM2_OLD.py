"""
Line-Based Visual SLAM Pipeline (v2 - Noise Reduced)
Connects: Record3D loader â†’ M-LSD â†’ LBD â†’ SelMap â†’ Back-projection â†’ PnL â†’ Map

Key improvements over v1:
- Only matched+validated lines added to map (not every detection)
- Depth edge filter removes lines at depth discontinuities
- Minimum 3D line length filter removes tiny noisy segments
- Score threshold raised to reduce false line detections
"""
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

# Import modules
from LIDAR_loader import Record3DLoader
from MLSD import pred_lines
from lbd_optimized import LineDescriptorOptimized, LineMatcherOptimized
from minPNL_solver import MinPnLSolver, compute_pose_error
from map_3d import LineMap, backproject_lines
# from depth_edge_filter import backproject_lines_with_edge_filter  # REVERTED


def selmap_filter(matches, lines1, lines2, threshold_factor=1.5):
    """
    SelMap outlier rejection for line matches.
    Filters based on displacement vector consistency (magnitude + angle).
    """
    if len(matches) < 5:
        return matches, 0
    
    centers1 = (lines1[:, :2] + lines1[:, 2:]) / 2
    centers2 = (lines2[:, :2] + lines2[:, 2:]) / 2
    
    vectors = np.array([centers2[m[1]] - centers1[m[0]] for m in matches])
    lengths = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    len_hist, len_edges = np.histogram(lengths, bins=30)
    mode_idx = np.argmax(len_hist)
    mode_len = (len_edges[mode_idx] + len_edges[mode_idx + 1]) / 2
    
    ang_hist, ang_edges = np.histogram(angles, bins=36)
    mode_idx = np.argmax(ang_hist)
    mode_ang = (ang_edges[mode_idx] + ang_edges[mode_idx + 1]) / 2
    
    len_threshold = threshold_factor * np.std(lengths)
    ang_threshold = threshold_factor * np.std(angles)
    
    inliers = []
    for i, m in enumerate(matches):
        len_ok = abs(lengths[i] - mode_len) < len_threshold
        ang_ok = abs(angles[i] - mode_ang) < ang_threshold
        if len_ok and ang_ok:
            inliers.append(m)
    
    return inliers, len(matches) - len(inliers)


def filter_short_lines_2d(lines_2d, min_length=40.0):
    """Remove 2D lines shorter than min_length pixels."""
    lengths = np.sqrt((lines_2d[:, 2] - lines_2d[:, 0])**2 + 
                      (lines_2d[:, 3] - lines_2d[:, 1])**2)
    mask = lengths >= min_length
    return lines_2d[mask], mask


def filter_short_lines_3d(lines_3d, min_length=0.05):
    """Remove 3D lines shorter than min_length metres."""
    lengths = np.sqrt(np.sum((lines_3d[:, 3:] - lines_3d[:, :3])**2, axis=1))
    mask = lengths >= min_length
    return mask


class LineSLAM:
    def __init__(self, mlsd_model_path: str):
        """Initialize SLAM pipeline with M-LSD model."""
        self.interpreter = tf.lite.Interpreter(model_path=mlsd_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.descriptor = LineDescriptorOptimized(num_bands=7, band_width=5, max_lines=150)
        self.matcher = LineMatcherOptimized(descriptor_distance_threshold=0.4, 
                                            geometric_distance_threshold=100.0,
                                            angle_threshold_deg=25.0)
        self.map = LineMap()
        self.pnl_solver = None
        
        self.prev_lines_2d = None
        self.prev_descriptors = None
        self.prev_lines_3d = None
        
        self.estimated_poses = []
        self.gt_poses = []
        self.rotation_errors = []
        self.translation_errors = []

    def process_frame(self, frame, frame_idx: int, use_gt_pose: bool = False):
        """Process a single frame through the SLAM pipeline."""
        if self.pnl_solver is None:
            self.pnl_solver = MinPnLSolver(frame.K, ransac_iters=500, threshold=15.0)
        
        # 1. Detect 2D lines with M-LSD (raised threshold to reduce false detections)
        rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        lines_2d = pred_lines(rgb_bgr, self.interpreter, self.input_details, 
                              self.output_details, score_thr=0.10, dist_thr=20.0)
        
        if len(lines_2d) == 0:
            print(f"  Frame {frame_idx}: No lines detected")
            return None
        
        # 2. Filter short 2D lines (removes tiny noisy detections)
        lines_2d, length_mask = filter_short_lines_2d(lines_2d, min_length=40.0)
        
        if len(lines_2d) == 0:
            print(f"  Frame {frame_idx}: No lines after length filter")
            return None
        
        # 3. Compute LBD descriptors
        descriptors, valid_desc_indices = self.descriptor.compute_descriptors(frame.rgb, lines_2d)
        lines_2d = lines_2d[valid_desc_indices]
        
        if len(lines_2d) == 0:
            print(f"  Frame {frame_idx}: No valid descriptors")
            return None
        
        # 4. Back-project to 3D using LiDAR depth (plain backprojection)
        lines_3d_cam, valid_3d_indices = backproject_lines(
            lines_2d, frame.depth, frame.K, frame.rgb.shape[:2]
        )
        
        lines_2d = lines_2d[valid_3d_indices]
        descriptors = descriptors[valid_3d_indices]
        
        # 5. Filter short 3D lines (removes tiny segments from noisy depth)
        if len(lines_3d_cam) > 0:
            valid_3d_length = filter_short_lines_3d(lines_3d_cam, min_length=0.05)
            lines_3d_cam = lines_3d_cam[valid_3d_length]
            lines_2d = lines_2d[valid_3d_length]
            descriptors = descriptors[valid_3d_length]
        
        if len(lines_3d_cam) < 3:
            print(f"  Frame {frame_idx}: Not enough 3D lines ({len(lines_3d_cam)})")
            return None
        
        # 6. Pose estimation (skip for first frame)
        R_est, t_est = None, None
        n_matches, n_inliers, n_rejected = 0, 0, 0
        matched_indices_curr = []  # Track which current-frame lines were matched
        
        if frame_idx > 0 and self.prev_lines_2d is not None:
            matches = self.matcher.match_lines(
                self.prev_lines_2d, self.prev_descriptors,
                lines_2d, descriptors
            )
            n_matches_raw = len(matches)
            
            if len(matches) >= 5:
                matches, n_rejected = selmap_filter(matches, self.prev_lines_2d, lines_2d)
            else:
                n_rejected = 0
            
            n_matches = len(matches)
            
            # Track which lines in current frame were matched
            matched_indices_curr = [m[1] for m in matches]
            
            if n_matches >= 3:
                lines_2d_matched = np.array([lines_2d[m[1]] for m in matches])
                lines_3d_matched = np.array([self.prev_lines_3d[m[0]] for m in matches])
                
                success, R_est, t_est, inliers = self.pnl_solver.estimate_pose(
                    lines_2d_matched, lines_3d_matched
                )
                n_inliers = inliers.sum() if success else 0
                
                if success:
                    # Compare with ground truth
                    # ARKit pose is camera-to-world; invert to get world-to-camera
                    R_gt_c2w = frame.pose[:3, :3]
                    t_gt_c2w = frame.pose[:3, 3]
                    R_gt_arkit = R_gt_c2w.T
                    t_gt_arkit = -R_gt_c2w.T @ t_gt_c2w
                    
                    # Solver outputs world->OpenCV-camera (Y-down, Z-forward)
                    # ARKit GT is world->ARKit-camera (Y-up, Z-backward)
                    # Convert: R_opencv = F @ R_arkit, t_opencv = F @ t_arkit
                    F = np.diag([1.0, -1.0, -1.0])
                    R_gt = F @ R_gt_arkit
                    t_gt = F @ t_gt_arkit
                    
                    rot_err, trans_err = compute_pose_error(R_est, t_est, R_gt, t_gt)
                    self.rotation_errors.append(rot_err)
                    self.translation_errors.append(trans_err)
                    self.estimated_poses.append((R_est, t_est))
        
        # 7. Get pose for map update
        if use_gt_pose or R_est is None:
            R = frame.pose[:3, :3]
            t = frame.pose[:3, 3]
        else:
            R, t = R_est, t_est
        self.gt_poses.append(frame.pose)
        
        # 8. Add ONLY matched lines to map (key noise reduction!)
        # For frame 0 (no matches yet), add all lines to seed the map
        if frame_idx == 0:
            self.map.add_lines(lines_3d_cam, descriptors, R, t, frame_idx)
        elif len(matched_indices_curr) > 0:
            # Only add lines that were successfully matched
            matched_mask = np.array(matched_indices_curr)
            matched_3d = lines_3d_cam[matched_mask]
            matched_desc = descriptors[matched_mask]
            self.map.add_lines(matched_3d, matched_desc, R, t, frame_idx)
        
        # 9. Store for next frame matching (keep ALL lines for matching flexibility)
        self.prev_lines_2d = lines_2d
        self.prev_descriptors = descriptors
        lines_3d_world = []

        # F converts OpenCV camera convention → ARKit camera convention
        # OpenCV: X-right, Y-down, Z-forward
        # ARKit:  X-right, Y-up,   Z-backward
        F = np.diag([1.0, -1.0, -1.0])
        for line in lines_3d_cam:
            P1_arkit_cam = F @ line[:3]
            P2_arkit_cam = F @ line[3:]
            P1_world = R @ P1_arkit_cam + t
            P2_world = R @ P2_arkit_cam + t
            lines_3d_world.append(np.concatenate([P1_world, P2_world]))
        self.prev_lines_3d = np.array(lines_3d_world) if lines_3d_world else np.array([]).reshape(0,6)
        
        return {
            'frame_idx': frame_idx,
            'n_lines_2d': len(lines_2d),
            'n_lines_3d': len(lines_3d_cam),
            'n_matches': n_matches,
            'n_rejected': n_rejected,
            'n_inliers': n_inliers,
            'n_map_lines': len(self.map),
            'rot_err': self.rotation_errors[-1] if self.rotation_errors else None,
            'trans_err': self.translation_errors[-1] if self.translation_errors else None
        }

    def run(self, data_path: str, max_frames: int = None, use_gt_pose: bool = False):
        """Run SLAM on a Record3D sequence."""
        print("=" * 60)
        print("Line-Based Visual SLAM (v2 - Noise Reduced)")
        print("=" * 60)
        
        loader = Record3DLoader(data_path, max_frames=max_frames)
        print(f"\nProcessing {len(loader)} frames...")
        print(f"Using {'ARKit ground truth' if use_gt_pose else 'estimated'} poses for map\n")
        
        for frame_idx in range(len(loader)):
            frame = loader[frame_idx]
            result = self.process_frame(frame, frame_idx, use_gt_pose)
            
            if result:
                print(f"Frame {frame_idx}: {result['n_lines_2d']} lines, "
                      f"{result['n_matches']} matches (-{result['n_rejected']} rejected), "
                      f"{result['n_inliers']} inliers, "
                      f"map={result['n_map_lines']}", end="")
                if result['rot_err'] is not None:
                    print(f" | Err: {result['rot_err']:.2f}Â° / {result['trans_err']:.1f}%")
                else:
                    print()
        
        # Summary
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total frames processed: {len(self.gt_poses)}")
        print(f"Total 3D lines in map: {len(self.map)}")
        
        if self.rotation_errors:
            print(f"\nPose Estimation Errors (vs ARKit ground truth):")
            print(f"  Rotation:    mean={np.mean(self.rotation_errors):.2f}Â°, "
                  f"median={np.median(self.rotation_errors):.2f}Â°")
            print(f"  Translation: mean={np.mean(self.translation_errors):.1f}%, "
                  f"median={np.median(self.translation_errors):.1f}%")
        
        return self.map, self.estimated_poses, self.gt_poses


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python runSLAM2.py <path_to_record3d_folder> [max_frames]")
        print("\nExample: python runSLAM2.py C:/Users/hamee/Downloads/videos/video1 20")
        sys.exit(1)
    
    data_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    mlsd_model = "tflite_models/M-LSD_512_large_fp32.tflite"
    
    if not Path(mlsd_model).exists():
        print(f"ERROR: M-LSD model not found at {mlsd_model}")
        print("Please ensure the model file is in the correct location.")
        sys.exit(1)
    
    slam = LineSLAM(mlsd_model)
    slam.run(data_path, max_frames=max_frames, use_gt_pose=True)


if __name__ == '__main__':
    main()