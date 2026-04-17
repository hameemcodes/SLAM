"""
Test MinPnL with REALISTIC line correspondences from actual SLAM pipeline
"""
import numpy as np
import sys
import cv2
import tensorflow as tf

sys.path.insert(0, r'C:\Users\hamee\OneDrive\Documents\GitHub\3D-Object-SLAM\Modules')  # Adjust to your path

from TUM_loader import TUMLoader
from MLSD import pred_lines
from lbd_optimized import LineDescriptorOptimized, LineMatcherOptimized
from map_3d import backproject_lines
from minPNL_solver import MinPnLSolver, compute_pose_error
from depth_edge_filter import backproject_lines_with_edge_filter


def test_minpnl_real(data_path, mlsd_model_path):
    """
    Test MinPnL with real line correspondences from SLAM pipeline
    """
    print("\n" + "="*70)
    print("MinPnL TEST WITH REAL LINE CORRESPONDENCES")
    print("="*70)
    
    # Load data
    loader = TUMLoader(data_path, max_frames=3)
    
    # Load M-LSD
    interpreter = tf.lite.Interpreter(model_path=mlsd_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Initialize LBD
    descriptor = LineDescriptorOptimized(num_bands=7, band_width=5, max_lines=100)
    matcher = LineMatcherOptimized()
    
    # Process frame 0
    frame0 = loader[0]
    rgb0_bgr = cv2.cvtColor(frame0.rgb, cv2.COLOR_RGB2BGR)
    lines_2d_0 = pred_lines(rgb0_bgr, interpreter, input_details, output_details, 
                            score_thr=0.1, dist_thr=20.0)
    
    print(f"\nFrame 0: Detected {len(lines_2d_0)} lines")
    
    if len(lines_2d_0) == 0:
        print("No lines detected! Try different M-LSD threshold")
        return
    
    # Compute descriptors
    desc0, valid0 = descriptor.compute_descriptors(frame0.rgb, lines_2d_0)
    lines_2d_0 = lines_2d_0[valid0]
    
    # Back-project to 3D in camera frame
    lines_3d_cam0, valid_3d0 = backproject_lines_with_edge_filter(
        lines_2d_0, 
        frame0.depth, 
        frame0.K, 
        frame0.rgb.shape[:2],
        edge_filter_method='median',     # Start with 'median'
        edge_threshold=0.15,             # Tune between 0.10-0.25
        depth_consistency_threshold=0.20 # Line-level check (optional)
        )
    lines_2d_0 = lines_2d_0[valid_3d0]
    desc0 = desc0[valid_3d0]
    

    print(f"Frame 0: {len(lines_3d_cam0)} lines with valid 3D")
    
    if len(lines_3d_cam0) < 3:
        print("Not enough 3D lines!")
        return
    
    # Transform frame 0 lines to WORLD frame (this is the map)
    R0_ctow = frame0.pose[:3, :3]
    t0_ctow = frame0.pose[:3, 3]
    
    lines_3d_world = []
    for line in lines_3d_cam0:
        P1_world = R0_ctow @ line[:3] + t0_ctow
        P2_world = R0_ctow @ line[3:] + t0_ctow
        lines_3d_world.append(np.concatenate([P1_world, P2_world]))
    lines_3d_world = np.array(lines_3d_world)
    
    print(f"Transformed {len(lines_3d_world)} lines to world frame")
    
    # Process frame 1
    frame1 = loader[1]
    rgb1_bgr = cv2.cvtColor(frame1.rgb, cv2.COLOR_RGB2BGR)
    lines_2d_1 = pred_lines(rgb1_bgr, interpreter, input_details, output_details,
                            score_thr=0.1, dist_thr=20.0)
    
    print(f"\nFrame 1: Detected {len(lines_2d_1)} lines")
    
    if len(lines_2d_1) == 0:
        print("No lines detected in frame 1!")
        return
    
    # Compute descriptors
    desc1, valid1 = descriptor.compute_descriptors(frame1.rgb, lines_2d_1)
    lines_2d_1 = lines_2d_1[valid1]
    
    # Match lines between frame 0 and frame 1
    matches = matcher.match_lines(lines_2d_0, desc0, lines_2d_1, desc1)
    
    print(f"Found {len(matches)} matches between frames")
    
    if len(matches) < 3:
        print("Not enough matches for PnL!")
        return
    # Match lines between frame 0 and frame 1
    matches = matcher.match_lines(lines_2d_0, desc0, lines_2d_1, desc1)

    print(f"Found {len(matches)} raw matches")

    #SelMap filtering (copied from runSLAM2.py)
    if len(matches) >= 5:
        from runSLAM2 import selmap_filter  # Import the function
        matches, n_rejected = selmap_filter(matches, lines_2d_0, lines_2d_1)
        print(f"After SelMap: {len(matches)} matches (rejected {n_rejected})")

    print(f"Final matches for PnL: {len(matches)}")
    # Build correspondences for PnL
    # Current 2D lines from frame 1, corresponding 3D lines from world frame
    matched_2d = np.array([lines_2d_1[m[1]] for m in matches])
    matched_3d = np.array([lines_3d_world[m[0]] for m in matches])
    
    print(f"\nPnL input: {len(matched_2d)} 2D-3D line correspondences")
    
    # Ground truth pose (world-to-camera for frame 1)
    R_gt_ctow = frame1.pose[:3, :3]
    t_gt_ctow = frame1.pose[:3, 3]
    R_gt = R_gt_ctow.T
    t_gt = -R_gt_ctow.T @ t_gt_ctow
    
    print(f"Ground truth pose (world-to-cam):")
    print(f"  R determinant: {np.linalg.det(R_gt):.6f} (should be 1.0)")
    print(f"  t magnitude: {np.linalg.norm(t_gt):.3f}")
    
    # Test MinPnL
    print("\n" + "-"*70)
    print("TESTING MinPnL SOLVER")
    print("-"*70)
    
    solver = MinPnLSolver(frame1.K, ransac_iters=500, threshold=30.0)
    success, R_est, t_est, inliers = solver.estimate_pose(matched_2d, matched_3d, debug=True)
    
    if success:
        rot_err, trans_err = compute_pose_error(R_est, t_est, R_gt, t_gt)
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"✓ Success!")
        print(f"Rotation error: {rot_err:.2f}°")
        print(f"Translation error: {trans_err:.1f}%")
        print(f"Inliers: {inliers.sum()}/{len(matches)}")
        
        # Show estimated pose
        print(f"\nEstimated R determinant: {np.linalg.det(R_est):.6f}")
        print(f"Estimated t magnitude: {np.linalg.norm(t_est):.3f}")
        
        # Check if pose makes sense
        if rot_err > 45:
            print("\n⚠ WARNING: Large rotation error - pose may be flipped")
        if trans_err > 50:
            print("⚠ WARNING: Large translation error")
        

            
    else:
        print(f"\n✗ FAILED - Could not estimate pose")
        print(f"Inliers found: {inliers.sum()}/{len(matches)}")
        
        if inliers.sum() == 0:
            print("\nPossible issues:")
            print("1. Frame convention mismatch (cam-to-world vs world-to-cam)")
            print("2. Lines not corresponding correctly")
            print("3. RANSAC threshold too strict")
            print("4. Numerical issues in solver")
    


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python test_minpnl_realistic.py <data_path> <mlsd_model_path>")
        print("\nExample:")
        print("  python test_minpnl_realistic.py \\")
        print("    C:/Users/hamee/Downloads/videos/video1 \\")
        print("    C:/Users/hamee/Downloads/SLAM/tflite_models/M-LSD_512_large_fp32.tflite")
        sys.exit(1)
    
    data_path = sys.argv[1]
    mlsd_model = sys.argv[2]
    
    test_minpnl_real(data_path, mlsd_model)