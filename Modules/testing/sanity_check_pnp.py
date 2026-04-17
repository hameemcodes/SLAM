"""
Sanity Check: Use OpenCV solvePnP with line endpoints as points
to verify that the front-end data (line detection, matching, 3D backprojection)
is correct BEFORE debugging the MinPnL solver.

If OpenCV PnP works well → front-end is fine, MinPnL solver has bugs
If OpenCV PnP also fails → front-end has issues (matches, depth, conventions)
"""
import numpy as np
import sys
import cv2
import tensorflow as tf

sys.path.insert(0, r'C:\Users\hamee\OneDrive\Documents\GitHub\3D-Object-SLAM\Modules')

from LIDAR_loader import Record3DLoader
from MLSD import pred_lines
from lbd_optimized import LineDescriptorOptimized, LineMatcherOptimized
from map_3d import backproject_lines
from depth_edge_filter import backproject_lines_with_edge_filter


def run_sanity_check(data_path, mlsd_model_path):
    print("\n" + "=" * 70)
    print("SANITY CHECK: OpenCV solvePnP with line endpoints")
    print("=" * 70)

    # ===== Load data =====
    loader = Record3DLoader(data_path, max_frames=3)

    interpreter = tf.lite.Interpreter(model_path=mlsd_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    descriptor = LineDescriptorOptimized(num_bands=7, band_width=5, max_lines=100)
    matcher = LineMatcherOptimized()

    # ===== Print poses for inspection =====
    frame0 = loader[0]
    frame1 = loader[1]

    print(f"\n--- POSE INSPECTION ---")
    print(f"Frame 0 pose (camera-to-world):\n{frame0.pose}")
    print(f"Frame 0 position: {frame0.pose[:3, 3]}")
    print(f"\nFrame 1 pose (camera-to-world):\n{frame1.pose}")
    print(f"Frame 1 position: {frame1.pose[:3, 3]}")

    baseline = np.linalg.norm(frame1.pose[:3, 3] - frame0.pose[:3, 3])
    print(f"\nBaseline (distance between frames): {baseline:.6f} metres")
    if baseline > 1.0:
        print("⚠ WARNING: Baseline > 1m between consecutive 60fps frames is suspicious!")
    elif baseline < 0.0001:
        print("⚠ WARNING: Baseline near zero - frames may be identical")
    else:
        print(f"✓ Baseline looks reasonable for 60fps video")

    # ===== Process Frame 0 =====
    rgb0_bgr = cv2.cvtColor(frame0.rgb, cv2.COLOR_RGB2BGR)
    lines_2d_0 = pred_lines(rgb0_bgr, interpreter, input_details, output_details,
                            score_thr=0.1, dist_thr=20.0)
    print(f"\nFrame 0: {len(lines_2d_0)} lines detected")

    desc0, valid0 = descriptor.compute_descriptors(frame0.rgb, lines_2d_0)
    lines_2d_0 = lines_2d_0[valid0]

    lines_3d_cam0, valid_3d0 = backproject_lines_with_edge_filter(
        lines_2d_0, frame0.depth, frame0.K, frame0.rgb.shape[:2],
        edge_filter_method='median', edge_threshold=0.15,
        depth_consistency_threshold=0.20
    )
    lines_2d_0 = lines_2d_0[valid_3d0]
    desc0 = desc0[valid_3d0]
    print(f"Frame 0: {len(lines_3d_cam0)} lines with valid 3D")

    # Transform to world frame
    R0_c2w = frame0.pose[:3, :3]
    t0_c2w = frame0.pose[:3, 3]

    # Convert OpenCV camera convention -> ARKit camera convention before
    # applying ARKit's camera-to-world transform
    F = np.diag([1.0, -1.0, -1.0])
    lines_3d_world = []
    for line in lines_3d_cam0:
        P1_arkit_cam = F @ line[:3]
        P2_arkit_cam = F @ line[3:]
        P1_world = R0_c2w @ P1_arkit_cam + t0_c2w
        P2_world = R0_c2w @ P2_arkit_cam + t0_c2w
        lines_3d_world.append(np.concatenate([P1_world, P2_world]))
    lines_3d_world = np.array(lines_3d_world)

    # ===== Process Frame 1 =====
    rgb1_bgr = cv2.cvtColor(frame1.rgb, cv2.COLOR_RGB2BGR)
    lines_2d_1 = pred_lines(rgb1_bgr, interpreter, input_details, output_details,
                            score_thr=0.1, dist_thr=20.0)
    print(f"Frame 1: {len(lines_2d_1)} lines detected")

    desc1, valid1 = descriptor.compute_descriptors(frame1.rgb, lines_2d_1)
    lines_2d_1 = lines_2d_1[valid1]

    # ===== Match =====
    matches = matcher.match_lines(lines_2d_0, desc0, lines_2d_1, desc1)
    print(f"Raw matches: {len(matches)}")

    if len(matches) >= 5:
        from runSLAM2 import selmap_filter
        matches, n_rejected = selmap_filter(matches, lines_2d_0, lines_2d_1)
        print(f"After SelMap: {len(matches)} matches (rejected {n_rejected})")

    if len(matches) < 3:
        print("Not enough matches!")
        return

    # ===== Build point correspondences from line endpoints =====
    # Each line gives us 2 point correspondences (both endpoints)
    points_3d = []  # World frame
    points_2d = []  # Frame 1 pixels

    for m in matches:
        idx0, idx1 = m
        line_3d = lines_3d_world[idx0]   # [X1,Y1,Z1, X2,Y2,Z2] in world
        line_2d = lines_2d_1[idx1]       # [x1,y1, x2,y2] in pixels

        # Endpoint 1
        points_3d.append(line_3d[:3])
        points_2d.append(line_2d[:2])

        # Endpoint 2
        points_3d.append(line_3d[3:])
        points_2d.append(line_2d[2:])

    points_3d = np.array(points_3d, dtype=np.float64)
    points_2d = np.array(points_2d, dtype=np.float64)

    print(f"\nOpenCV PnP input: {len(points_3d)} point correspondences "
          f"(from {len(matches)} line matches)")

    # ===== Print some correspondences for inspection =====
    print(f"\n--- SAMPLE CORRESPONDENCES (first 3 lines) ---")
    for i in range(min(3, len(matches))):
        idx0, idx1 = matches[i]
        print(f"  Match {i}: line {idx0} → line {idx1}")
        print(f"    3D endpoint 1 (world): {lines_3d_world[idx0][:3]}")
        print(f"    3D endpoint 2 (world): {lines_3d_world[idx0][3:]}")
        print(f"    2D endpoint 1 (pixels): {lines_2d_1[idx1][:2]}")
        print(f"    2D endpoint 2 (pixels): {lines_2d_1[idx1][2:]}")

    # ===== Ground truth (world-to-camera for frame 1) =====
    R1_c2w = frame1.pose[:3, :3]
    t1_c2w = frame1.pose[:3, 3]
    # World-to-camera: invert the camera-to-world transform (ARKit convention)
    R_gt_arkit = R1_c2w.T
    t_gt_arkit = -R1_c2w.T @ t1_c2w
    # Convert ARKit w2c -> OpenCV w2c (Y-down, Z-forward)
    # solvePnP and reprojection both use OpenCV convention
    F = np.diag([1.0, -1.0, -1.0])
    R_gt_w2c = F @ R_gt_arkit
    t_gt_w2c = F @ t_gt_arkit

    print(f"\n--- GROUND TRUTH (world-to-camera) ---")
    print(f"R_gt determinant: {np.linalg.det(R_gt_w2c):.6f}")
    print(f"t_gt: {t_gt_w2c}")
    print(f"t_gt magnitude: {np.linalg.norm(t_gt_w2c):.3f}")

    # Also show the RELATIVE pose (more intuitive)
    # Relative: frame1 w.r.t. frame0
    R_rel = R1_c2w.T @ R0_c2w  # relative rotation
    t_rel = R1_c2w.T @ (t0_c2w - t1_c2w)  # relative translation
    print(f"\n--- RELATIVE POSE (frame 1 w.r.t. frame 0) ---")
    print(f"Relative translation magnitude: {np.linalg.norm(t_rel):.6f} metres")
    print(f"Relative rotation angle: {np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))):.4f} degrees")

    # ===== Verify: project 3D points with GT pose, check reprojection =====
    print(f"\n--- REPROJECTION CHECK WITH GROUND TRUTH POSE ---")
    K = frame1.K
    reproj_errors = []
    for i in range(len(points_3d)):
        # Transform to camera frame using GT
        P_cam = R_gt_w2c @ points_3d[i] + t_gt_w2c

        if P_cam[2] <= 0:
            print(f"  Point {i}: BEHIND CAMERA (Z={P_cam[2]:.3f})")
            reproj_errors.append(999)
            continue

        # Project to pixels
        p_proj = K @ P_cam
        p_proj = p_proj[:2] / p_proj[2]

        # Error vs observed 2D point
        error = np.linalg.norm(p_proj - points_2d[i])
        reproj_errors.append(error)

    reproj_errors = np.array(reproj_errors)
    valid_errors = reproj_errors[reproj_errors < 900]

    if len(valid_errors) > 0:
        print(f"  Points behind camera: {(reproj_errors >= 900).sum()}/{len(reproj_errors)}")
        print(f"  Reprojection errors (valid points):")
        print(f"    Mean:   {valid_errors.mean():.1f} pixels")
        print(f"    Median: {np.median(valid_errors):.1f} pixels")
        print(f"    Max:    {valid_errors.max():.1f} pixels")
        print(f"    Min:    {valid_errors.min():.1f} pixels")
        print(f"    < 10px: {(valid_errors < 10).sum()}/{len(valid_errors)}")
        print(f"    < 50px: {(valid_errors < 50).sum()}/{len(valid_errors)}")

        if valid_errors.mean() > 100:
            print("\n  ⚠ HIGH REPROJECTION ERROR WITH GT POSE!")
            print("  This means the front-end data is inconsistent:")
            print("  - 3D points (from frame 0) don't match 2D points (from frame 1)")
            print("  - Possible causes: bad matches, wrong depth, convention error")
        elif valid_errors.mean() > 30:
            print("\n  ⚠ MODERATE reprojection error - matches are noisy but partially correct")
        else:
            print("\n  ✓ Low reprojection error - front-end data looks consistent!")
    else:
        print("  ⚠ ALL POINTS BEHIND CAMERA - convention is definitely wrong!")

    # ===== Run OpenCV solvePnP =====
    print(f"\n--- OpenCV solvePnP ---")

    # Method 1: solvePnPRansac (robust to outliers)
    success, rvec, tvec, inliers_cv = cv2.solvePnPRansac(
        points_3d, points_2d, K, distCoeffs=None,
        iterationsCount=1000,
        reprojectionError=15.0,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success and inliers_cv is not None:
        R_est, _ = cv2.Rodrigues(rvec)
        t_est = tvec.flatten()

        print(f"  ✓ solvePnPRansac succeeded!")
        print(f"  Inliers: {len(inliers_cv)}/{len(points_3d)} points "
              f"({len(inliers_cv)//2}/{len(matches)} lines)")

        # Compare with GT
        R_diff = R_est @ R_gt_w2c.T
        rot_err = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
        trans_err_abs = np.linalg.norm(t_est - t_gt_w2c)

        print(f"\n  --- COMPARISON WITH GROUND TRUTH ---")
        print(f"  Rotation error:    {rot_err:.2f}°")
        print(f"  Translation error: {trans_err_abs:.3f} metres")
        print(f"  Est t magnitude:   {np.linalg.norm(t_est):.3f}")
        print(f"  GT  t magnitude:   {np.linalg.norm(t_gt_w2c):.3f}")

        if rot_err < 5:
            print(f"\n  ✓ EXCELLENT - Front-end is fine! Problem is in MinPnL solver.")
        elif rot_err < 15:
            print(f"\n  ✓ GOOD - Front-end is mostly fine. MinPnL solver needs fixing.")
        elif rot_err < 45:
            print(f"\n  ⚠ MODERATE - Some front-end issues but solver should work better.")
        else:
            print(f"\n  ✗ POOR - Front-end data has significant issues.")
    else:
        print(f"  ✗ solvePnPRansac FAILED")
        print(f"  This suggests front-end data is too noisy or inconsistent")

    # Method 2: solvePnP without RANSAC (for comparison)
    print(f"\n--- OpenCV solvePnP (no RANSAC, all points) ---")
    success2, rvec2, tvec2 = cv2.solvePnP(
        points_3d, points_2d, K, distCoeffs=None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success2:
        R_est2, _ = cv2.Rodrigues(rvec2)
        t_est2 = tvec2.flatten()
        R_diff2 = R_est2 @ R_gt_w2c.T
        rot_err2 = np.degrees(np.arccos(np.clip((np.trace(R_diff2) - 1) / 2, -1, 1)))
        print(f"  Rotation error: {rot_err2:.2f}°")
        print(f"  Translation error: {np.linalg.norm(t_est2 - t_gt_w2c):.3f} metres")
    else:
        print(f"  ✗ Failed")

    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}")
    if success and inliers_cv is not None and rot_err < 15:
        print("Front-end data is GOOD. The MinPnL solver has bugs.")
        print("→ Focus on fixing _build_constraints() in minPNL_solver.py")
    elif success and inliers_cv is not None and rot_err < 45:
        print("Front-end data is OKAY but noisy. MinPnL solver also needs fixing.")
        print("→ Fix MinPnL solver first, then improve matching quality")
    else:
        print("Front-end data has ISSUES. Fix these before debugging MinPnL:")
        print("→ Check frame convention (camera-to-world vs world-to-camera)")
        print("→ Check depth scaling and RGB-depth alignment")
        print("→ Check match quality")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python sanity_check_pnp.py <data_path> <mlsd_model_path>")
        print("\nExample:")
        print("  python sanity_check_pnp.py ^")
        print("    C:\\Users\\hamee\\Downloads\\videos\\video1 ^")
        print("    C:\\Users\\hamee\\...\\M-LSD_512_large_fp32.tflite")
        sys.exit(1)

    run_sanity_check(sys.argv[1], sys.argv[2])