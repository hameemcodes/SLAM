"""
Visualization for Line-Based SLAM debugging
Shows: 2D lines on images, 3D map, matches between frames
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def draw_lines_on_image(image, lines, color=(0, 255, 0), thickness=2):
    """Draw 2D lines on an image."""
    vis = image.copy()
    for line in lines:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def visualize_matches(img1, lines1, img2, lines2, matches, title="Matches"):
    """Visualize line matches between two frames."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create side-by-side image
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    vis[:h2, w1:] = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Draw all lines (gray)
    for line in lines1:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(vis, (x1, y1), (x2, y2), (100, 100, 100), 1)
    for line in lines2:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(vis, (x1 + w1, y1), (x2 + w1, y2), (100, 100, 100), 1)
    
    # Draw matches (colored)
    for i, (idx1, idx2) in enumerate(matches):
        color = tuple(map(int, np.random.randint(50, 255, 3)))
        
        # Draw matched lines
        l1, l2 = lines1[idx1], lines2[idx2]
        cv2.line(vis, (int(l1[0]), int(l1[1])), (int(l1[2]), int(l1[3])), color, 2)
        cv2.line(vis, (int(l2[0])+w1, int(l2[1])), (int(l2[2])+w1, int(l2[3])), color, 2)
        
        # Draw connection between midpoints
        mid1 = (int((l1[0]+l1[2])/2), int((l1[1]+l1[3])/2))
        mid2 = (int((l2[0]+l2[2])/2)+w1, int((l2[1]+l2[3])/2))
        cv2.line(vis, mid1, mid2, color, 1)
    
    cv2.putText(vis, f"{title}: {len(matches)} matches", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return vis


def visualize_3d_map(lines_3d, poses_gt=None, poses_est=None, title="3D Line Map"):
    """
    Visualize 3D lines and camera trajectory.
    
    Args:
        lines_3d: Nx6 array of [X1,Y1,Z1,X2,Y2,Z2] in world frame
        poses_gt: List of 4x4 ground truth poses
        poses_est: List of (R, t) estimated poses
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw 3D lines (subsample if too many)
    max_lines = 500
    if len(lines_3d) > max_lines:
        indices = np.random.choice(len(lines_3d), max_lines, replace=False)
        lines_to_draw = lines_3d[indices]
    else:
        lines_to_draw = lines_3d
    
    for line in lines_to_draw:
        X1, Y1, Z1, X2, Y2, Z2 = line
        ax.plot([X1, X2], [Y1, Y2], [Z1, Z2], 'b-', alpha=0.5, linewidth=1.5)
    
    # Draw ground truth trajectory
    if poses_gt is not None and len(poses_gt) > 0:
        positions = np.array([p[:3, 3] for p in poses_gt])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'g-o', markersize=5, linewidth=2, label='ARKit GT')
    
    # Draw estimated trajectory
    if poses_est is not None and len(poses_est) > 0:
        positions = np.array([t for R, t in poses_est])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'r-^', markersize=5, linewidth=2, label='Estimated')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    if len(lines_3d) > 0:
        all_points = lines_3d.reshape(-1, 3)
        center = np.median(all_points, axis=0)
        max_range = np.percentile(np.abs(all_points - center), 95)
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])
    
    plt.tight_layout()
    return fig


def debug_single_frame(frame, lines_2d, lines_3d, frame_idx, output_dir="debug_output"):
    """Save debug visualization for a single frame."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Draw 2D lines on RGB
    vis = draw_lines_on_image(frame.rgb, lines_2d, color=(0, 255, 0), thickness=2)
    
    # Add info text
    cv2.putText(vis, f"Frame {frame_idx}: {len(lines_2d)} 2D lines, {len(lines_3d)} 3D lines", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save
    cv2.imwrite(f"{output_dir}/frame_{frame_idx:03d}_lines.jpg", 
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    # Also save depth visualization
    depth_vis = frame.depth.copy()
    depth_vis = np.nan_to_num(depth_vis, nan=0)
    depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(f"{output_dir}/frame_{frame_idx:03d}_depth.jpg", depth_vis)


def run_visualization(data_path, mlsd_model_path, max_frames=10, output_dir="debug_output"):
    """Run SLAM with visualization for debugging."""
    import tensorflow as tf
    from LIDAR_loader import Record3DLoader
    from MLSD import pred_lines
    from lbd_optimized import LineDescriptorOptimized, LineMatcherOptimized
    from map_3d import LineMap, backproject_lines
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=mlsd_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load data
    loader = Record3DLoader(data_path, max_frames=max_frames)
    
    # Initialize
    descriptor = LineDescriptorOptimized(num_bands=7, band_width=5, max_lines=150)
    matcher = LineMatcherOptimized(descriptor_distance_threshold=0.4,
                                   geometric_distance_threshold=100.0,
                                   angle_threshold_deg=25.0)
    
    all_lines_3d = []
    gt_poses = []
    prev_lines_2d, prev_descriptors, prev_frame = None, None, None
    
    print(f"\nGenerating debug visualizations in {output_dir}/\n")
    
    for frame_idx in range(len(loader)):
        frame = loader[frame_idx]
        gt_poses.append(frame.pose)
        
        # Detect lines
        rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        lines_2d = pred_lines(rgb_bgr, interpreter, input_details, output_details, 
                              score_thr=0.1, dist_thr=20.0)
        
        if len(lines_2d) == 0:
            continue
        
        # Compute descriptors
        descriptors, valid_indices = descriptor.compute_descriptors(frame.rgb, lines_2d)
        lines_2d = lines_2d[valid_indices]
        
        # Back-project
        lines_3d_cam, valid_3d = backproject_lines(lines_2d, frame.depth, frame.K, frame.rgb.shape[:2])
        lines_2d = lines_2d[valid_3d]
        descriptors = descriptors[valid_3d]
        
        # Save frame debug
        debug_single_frame(frame, lines_2d, lines_3d_cam, frame_idx, output_dir)
        
        # Transform to world and accumulate
        R, t = frame.pose[:3, :3], frame.pose[:3, 3]
        for line in lines_3d_cam:
            P1_world = R @ line[:3] + t
            P2_world = R @ line[3:] + t
            all_lines_3d.append(np.concatenate([P1_world, P2_world]))
        
        # Visualize matches
        if prev_lines_2d is not None and len(lines_2d) > 0:
            matches = matcher.match_lines(prev_lines_2d, prev_descriptors, lines_2d, descriptors)
            
            if len(matches) > 0:
                vis = visualize_matches(prev_frame.rgb, prev_lines_2d, 
                                       frame.rgb, lines_2d, matches,
                                       title=f"Frame {frame_idx-1} → {frame_idx}")
                cv2.imwrite(f"{output_dir}/matches_{frame_idx-1:03d}_to_{frame_idx:03d}.jpg",
                           cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        prev_lines_2d, prev_descriptors, prev_frame = lines_2d, descriptors, frame
        print(f"Frame {frame_idx}: {len(lines_2d)} lines, {len(lines_3d_cam)} 3D")
    
    # Save 3D map visualization
    all_lines_3d = np.array(all_lines_3d)
    fig = visualize_3d_map(all_lines_3d, gt_poses, title="3D Line Map (ARKit poses)")
    fig.savefig(f"{output_dir}/3d_map.png", dpi=150)
    plt.close(fig)
    
    print(f"\n✓ Saved {len(loader)} frame visualizations")
    print(f"✓ Saved match visualizations") 
    print(f"✓ Saved 3D map: {output_dir}/3d_map.png")
    print(f"\nTotal 3D lines: {len(all_lines_3d)}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_slam.py <data_path> [max_frames]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    mlsd_model = "tflite_models/M-LSD_512_large_fp32.tflite"
    
    run_visualization(data_path, mlsd_model, max_frames, output_dir="debug_output")