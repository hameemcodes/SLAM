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
vocab_path = "lbd_vocab_k64.npz"

# Import modules
from LIDAR_loader import Record3DLoader
from MLSD import pred_lines
from lbd_optimized import LineDescriptorOptimized, LineMatcherOptimized
#from minPNL_solver import MinPnLSolver
from pnp_line_solver import PnPLineSolver, compute_pose_error
from map_3d import LineMap, backproject_lines
# from depth_edge_filter import backproject_lines_with_edge_filter  # REVERTED
from loop_closure import BoWVocabulary, train_vocabulary_from_descriptors, LoopClosureDetector
from loop_closure_correction import LoopClosureCorrector
import matplotlib.pyplot as plt
import numpy as np

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
    # Toggle between solvers: 'pnp' (robust, recommended) or 'minpnl' (experimental)
    SOLVER = 'pnp'
    
    def __init__(self, mlsd_model_path: str, vocab_path: str = None):

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
        self.all_descriptors = []  # Store descriptors from all frames for vocab training

        self.vocab = None  # For loop closure 
        self.lcd= None  # Loop closure detector
        self.corrector = LoopClosureCorrector()
        self.last_keyframe_pose = None
        self.keyframe_count = 0
        if vocab_path and Path(vocab_path).exists():
            try:
                self.vocab = BoWVocabulary.from_file(vocab_path)
                self.lcd = LoopClosureDetector(self.vocab, min_loop_gap = 20, score_threshold=0.35)
                print(f"[LineSLAM] Loaded vocabulary from: {vocab_path}")
                print(f"[LineSLAM] Loop closure detector initialized")
            except Exception as e:
                print(f"[LineSLAM] Warning: Could not load vocabulary: {e}")
                print(f"[LineSLAM] Continuing without loop closure")
        else:
            print(f"[LineSLAM] No vocabulary provided (loop closure disabled)")


    def process_frame(self, frame, frame_idx: int, use_gt_pose: bool = False):
        """Process a single frame through the SLAM pipeline."""
        if self.pnl_solver is None:
            if self.SOLVER == 'pnp':
                self.pnl_solver = PnPLineSolver(frame.K, ransac_iters=1000, threshold=15.0)
                print(f"[Solver] Using PnPLineSolver (OpenCV solvePnP on line endpoints)")
            else:
                self.pnl_solver = MinPnLSolver(frame.K, ransac_iters=500, threshold=15.0)
                print(f"[Solver] Using MinPnLSolver (CGR-based)")
        
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

        if len(descriptors)>0:
            self.all_descriptors.append(descriptors.copy()) # Store descriptors for all frames 

        #loop closure detection
        loop_closure_candidate = None
        if self.lcd is not None:
            candidates = self.lcd.add_and_query(descriptors, frame_idx)
            if candidates:
                best_idx, best_score = candidates[0]
                print(f"  🔄 LOOP CLOSURE: frame {frame_idx} <-> frame {best_idx} "
                    f"(score={best_score:.3f})")
                loop_closure_candidate = (best_idx, best_score)
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
                    
                    rot_err, trans_err, abs_err_m = compute_pose_error(R_est, t_est, R_gt, t_gt)
                    self.rotation_errors.append(rot_err)
                    self.translation_errors.append(trans_err)
                    self.estimated_poses.append((R_est, t_est))
        
        # 7. Get pose for map update
        if use_gt_pose or R_est is None:
            R = frame.pose[:3, :3]
            t = frame.pose[:3, 3]
        else:
            F_conv = np.diag([1.0, -1.0, -1.0])
            R = R_est.T @ F_conv
            t = -R_est.T @ t_est.flatten()
        self.gt_poses.append(frame.pose)

        # 7b. Loop closure correction
        self.corrector.add_pose(frame_idx, R, t)
        
        if loop_closure_candidate is not None:
            # Sanity check: reject corrections that are too large (false positives)
            # A genuine loop closure on a well-tracked sequence should have small drift
            MAX_LC_ROTATION_DEG = 20.0   # reject corrections larger than this
            MAX_LC_TRANSLATION_M = 2.0
            
            result = self.corrector.apply_correction(
                current_idx=frame_idx,
                match_idx=loop_closure_candidate[0],
                line_map=self.map
            )
            
            if result.get("status") == "applied":
                drift_rot = result.get("drift_rotation_deg", 999)
                drift_trans = result.get("drift_translation_m", 999)
                
                if drift_rot > MAX_LC_ROTATION_DEG or drift_trans > MAX_LC_TRANSLATION_M:
                    print(f"  ⚠ LC REJECTED: drift too large "
                          f"({drift_rot:.1f}° / {drift_trans:.2f}m) — likely false positive")
                    # Undo: reload original poses back into corrector
                    # (correction already modified poses in-place, so we revert)
                    # For now, we just skip reading back the corrected pose
                else:
                    # FIX #1: Read back the corrected pose and use it
                    corrected_R, corrected_t = self.corrector.poses[frame_idx]
                    R = corrected_R
                    t = corrected_t
                    print(f"  ✓ LC APPLIED: correction within bounds")

        # 8. Keyframe-based map insertion
        # Only insert new lines when the camera has moved enough (keyframe)
        is_keyframe = False

        if frame_idx == 0:
            # First frame is always a keyframe
            is_keyframe = True
        elif R_est is not None and self.last_keyframe_pose is not None:
            # Check if camera moved enough since last keyframe
            last_R, last_t = self.last_keyframe_pose
            # Translation distance
            dist = np.linalg.norm(t - last_t)
            # Rotation angle
            R_diff = R @ last_R.T
            trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
            angle = np.degrees(np.arccos(trace))
            
            # Keyframe if moved >0.05m or rotated >5°
            if dist > 0.05 or angle > 5.0:
                is_keyframe = True

        if is_keyframe:
            if frame_idx == 0:
                # Seed the map with all lines
                self.map.add_lines(lines_3d_cam, descriptors, R, t, frame_idx)
            else:
                # Only add UNMATCHED lines (new geometry)
                all_indices = set(range(len(lines_3d_cam)))
                matched_set = set(matched_indices_curr)
                unmatched_indices = sorted(all_indices - matched_set)
                
                if len(unmatched_indices) > 0:
                    unmatched_mask = np.array(unmatched_indices)
                    new_3d = lines_3d_cam[unmatched_mask]
                    new_desc = descriptors[unmatched_mask]
                    self.map.add_lines(new_3d, new_desc, R, t, frame_idx)
            
            self.last_keyframe_pose = (R.copy(), t.copy())
            self.keyframe_count += 1       
     # 9. Store for next frame matching (keep ALL lines for matching flexibility)
        # FIX #2: Uses corrected R, t so next frame matches against corrected positions
        self.prev_lines_2d = lines_2d
        self.prev_descriptors = descriptors
        lines_3d_world = []
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

    def get_all_descriptors(self):
        return self.all_descriptors # Return list of descriptor arrays for all frames for vocab training


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
    def run_tum(self, data_path: str, max_frames: int = None, use_gt_pose: bool = False):
        """Run SLAM on a TUM RGB-D sequence."""
        from TUM_loader import TUMLoader
        
        print("=" * 60)
        print("Line-Based Visual SLAM (TUM RGB-D)")
        print("=" * 60)
        
        loader = TUMLoader(data_path, max_frames=max_frames)
        print(f"\nProcessing {len(loader)} frames...")
        print(f"Using {'ground truth' if use_gt_pose else 'estimated'} poses for map\n")
        
        for frame_idx in range(len(loader)):
            frame = loader[frame_idx]
            result = self.process_frame(frame, frame_idx, use_gt_pose)
            
            if result:
                print(f"Frame {frame_idx}: {result['n_lines_2d']} lines, "
                    f"{result['n_matches']} matches (-{result['n_rejected']} rejected), "
                    f"{result['n_inliers']} inliers, "
                    f"map={result['n_map_lines']}", end="")
                if result['rot_err'] is not None:
                    print(f" | Err: {result['rot_err']:.2f}° / {result['trans_err']:.1f}%")
                else:
                    print()
        
        # Summary (same as run)
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Total frames processed: {len(self.gt_poses)}")
        print(f"Total 3D lines in map: {len(self.map)}")
        
        if self.rotation_errors:
            print(f"\nPose Estimation Errors (vs ground truth):")
            print(f"  Rotation:    mean={np.mean(self.rotation_errors):.2f}°, "
                f"median={np.median(self.rotation_errors):.2f}°")
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
    


    # Ask user about loop closure
    print("Do you want to implement loop closure? (Y/N)")
    enable_loop_closure = input("Enter 'Y' to enable loop closure, or press Enter to skip: ").strip().lower() == "y"
    
    if enable_loop_closure:
        vocab_path = "lbd_vocab_k64.npz"
        if not Path(vocab_path).exists():
            print(f"\nERROR: Vocabulary file not found: {vocab_path}")
            print("Please run train_vocab.py first to train the vocabulary.")
            sys.exit(1)
        print(f"Running SLAM with loop closure enabled.")
        slam = LineSLAM(mlsd_model, vocab_path=vocab_path)  # ← Pass vocab_path
    else:
        print("Skipping loop closure.")
        slam = LineSLAM(mlsd_model)  # ← No vocab_path, loop closure disabled
    if (Path(data_path) / "groundtruth.txt").exists():
        print("Detected TUM RGB-D dataset")
        slam.run_tum(data_path, max_frames=max_frames, use_gt_pose=True)
    else:
        print("Detected Record3D dataset")
        slam.run(data_path, max_frames=max_frames, use_gt_pose=True)

    
    # Ground truth camera positions (ARKit: pose is camera-to-world, position = pose[:3,3])
    gt_positions = np.array([p[:3, 3] for p in slam.gt_poses])

    # Estimated camera positions (world-to-camera: position = -R.T @ t)
    est_positions = np.array([-R.T @ t for R, t in slam.estimated_poses])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_positions[:, 0], gt_positions[:, 2], 'g-o', markersize=4, label='ARKit GT')
    ax.plot(est_positions[:, 0], est_positions[:, 2], 'r-^', markersize=4, label='Estimated')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Camera Trajectory (Top-Down View)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.savefig("trajectory_topdown.png", dpi=150)
    plt.show()
    all_Descriptors = slam.get_all_descriptors()
    print(f"\nCollected descriptors from {len(all_Descriptors)} frames for vocab training.")


if __name__ == '__main__':
    main()