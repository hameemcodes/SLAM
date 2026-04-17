"""
Generate All Figures for Line-Based Visual SLAM Progress Report

Generates:
1. Section 3.1: Reprojection error histogram (sanity check with OpenCV solvePnP)
2. Section 5.1: Box plot of rotation/translation errors (video1)
3. Section 5.2: Side-by-side comparison (video1 success vs video2 failure)
4. Section 5.3: Scatter plot of rotation error vs inter-frame rotation

Usage:
    python generate_report_figures.py <video1_path> <video2_path> <mlsd_model_path>

Example:
    python generate_report_figures.py \\
        C:/Users/hamee/Downloads/videos/video1 \\
        C:/Users/hamee/Downloads/videos/video2 \\
        tflite_models/M-LSD_512_large_fp32.tflite
"""

import numpy as np
import sys
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import SLAM modules
from LIDAR_loader import Record3DLoader
from MLSD import pred_lines
from lbd_optimized import LineDescriptorOptimized, LineMatcherOptimized
from map_3d import backproject_lines
from minPNL_solver import MinPnLSolver, compute_pose_error
from runSLAM2 import selmap_filter, filter_short_lines_2d, filter_short_lines_3d


class ReportFigureGenerator:
    """Generate all figures needed for the progress report"""
    
    def __init__(self, video1_path: str, video2_path: str, mlsd_model_path: str):
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.mlsd_model_path = mlsd_model_path
        self.output_dir = Path("report_figures")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load M-LSD model
        print("Loading M-LSD model...")
        self.interpreter = tf.lite.Interpreter(model_path=mlsd_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Initialize descriptors
        self.descriptor = LineDescriptorOptimized(num_bands=7, band_width=5, max_lines=150)
        self.matcher = LineMatcherOptimized()
        
        print(f"✓ Output directory: {self.output_dir.absolute()}\n")
    
    def generate_all_figures(self):
        """Main function to generate all required figures"""
        print("="*70)
        print("GENERATING FIGURES FOR PROGRESS REPORT")
        print("="*70)
        
        # Figure 1: Sanity check (Section 3.1)
        print("\n[1/4] Figure 1: Sanity Check Reprojection Histogram...")
        self.generate_sanity_check_figure()
        
        # Run SLAM on both datasets
        print("\n[2/4] Running SLAM analysis on both datasets...")
        print("  Processing video1 (30 frames)...")
        video1_results = self.run_slam_analysis(self.video1_path, max_frames=30)
        
        print("  Processing video2 (30 frames)...")
        video2_results = self.run_slam_analysis(self.video2_path, max_frames=30)
        
        # Figure 2: Box plots (Section 5.1)
        print("\n[3/4] Figure 2: Box Plot of Errors (Video1)...")
        self.generate_boxplot_figure(video1_results)
        
        # Figure 3: Side-by-side comparison (Section 5.2)
        print("\n[4/4] Figure 3 & 4: Comparison and Scatter Plot...")
        self.generate_comparison_figure(video1_results, video2_results)
        
        # Figure 4: Scatter plot (Section 5.3)
        self.generate_scatter_plot(video1_results, video2_results)
        
        print("\n" + "="*70)
        print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
        print(f"✓ Location: {self.output_dir.absolute()}")
        print("="*70)
        self.print_summary(video1_results, video2_results)
    
    def generate_sanity_check_figure(self):
        """
        Figure 1 (Section 3.1): Reprojection error histogram
        Validates front-end by comparing GT pose vs OpenCV solvePnP
        """
        loader = Record3DLoader(self.video1_path, max_frames=3)
        
        # Process frames 0 and 1
        frame0, frame1 = loader[0], loader[1]
        
        # Detect lines in frame 0
        rgb0_bgr = cv2.cvtColor(frame0.rgb, cv2.COLOR_RGB2BGR)
        lines_2d_0 = pred_lines(rgb0_bgr, self.interpreter, self.input_details,
                                self.output_details, score_thr=0.1, dist_thr=20.0)
        
        # Compute descriptors
        desc0, valid0 = self.descriptor.compute_descriptors(frame0.rgb, lines_2d_0)
        lines_2d_0 = lines_2d_0[valid0]
        
        # Back-project to 3D
        lines_3d_cam0, valid_3d0 = backproject_lines(
            lines_2d_0, frame0.depth, frame0.K, frame0.rgb.shape[:2]
        )
        lines_2d_0 = lines_2d_0[valid_3d0]
        desc0 = desc0[valid_3d0]
        
        # Transform to world frame
        R0_c2w = frame0.pose[:3, :3]
        t0_c2w = frame0.pose[:3, 3]
        
        lines_3d_world = []
        for line in lines_3d_cam0:
            P1_world = R0_c2w @ line[:3] + t0_c2w
            P2_world = R0_c2w @ line[3:] + t0_c2w
            lines_3d_world.append(np.concatenate([P1_world, P2_world]))
        lines_3d_world = np.array(lines_3d_world)
        
        # Detect lines in frame 1
        rgb1_bgr = cv2.cvtColor(frame1.rgb, cv2.COLOR_RGB2BGR)
        lines_2d_1 = pred_lines(rgb1_bgr, self.interpreter, self.input_details,
                                self.output_details, score_thr=0.1, dist_thr=20.0)
        
        desc1, valid1 = self.descriptor.compute_descriptors(frame1.rgb, lines_2d_1)
        lines_2d_1 = lines_2d_1[valid1]
        
        # Match lines
        matches = self.matcher.match_lines(lines_2d_0, desc0, lines_2d_1, desc1)
        if len(matches) >= 5:
            matches, _ = selmap_filter(matches, lines_2d_0, lines_2d_1)
        
        # Build point correspondences (each line = 2 points)
        points_3d, points_2d = [], []
        for m in matches:
            idx0, idx1 = m
            line_3d = lines_3d_world[idx0]
            line_2d = lines_2d_1[idx1]
            
            points_3d.extend([line_3d[:3], line_3d[3:]])
            points_2d.extend([line_2d[:2], line_2d[2:]])
        
        points_3d = np.array(points_3d, dtype=np.float64)
        points_2d = np.array(points_2d, dtype=np.float64)
        
        # Ground truth pose (world-to-camera)
        R1_c2w = frame1.pose[:3, :3]
        t1_c2w = frame1.pose[:3, 3]
        R_gt_w2c = R1_c2w.T
        t_gt_w2c = -R1_c2w.T @ t1_c2w
        
        # Compute reprojection errors with GT pose
        K = frame1.K
        reproj_gt = []
        
        for i in range(len(points_3d)):
            P_cam = R_gt_w2c @ points_3d[i] + t_gt_w2c
            if P_cam[2] > 0:
                p_proj = K @ P_cam
                p_proj = p_proj[:2] / p_proj[2]
                error = np.linalg.norm(p_proj - points_2d[i])
                if error < 500:
                    reproj_gt.append(error)
        
        # OpenCV solvePnP
        success, rvec, tvec, inliers_cv = cv2.solvePnPRansac(
            points_3d, points_2d, K, distCoeffs=None,
            iterationsCount=1000, reprojectionError=15.0,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        reproj_opencv = []
        rot_err_opencv = 999
        if success:
            R_est, _ = cv2.Rodrigues(rvec)
            t_est = tvec.flatten()
            
            for i in range(len(points_3d)):
                P_cam = R_est @ points_3d[i] + t_est
                if P_cam[2] > 0:
                    p_proj = K @ P_cam
                    p_proj = p_proj[:2] / p_proj[2]
                    error = np.linalg.norm(p_proj - points_2d[i])
                    if error < 500:
                        reproj_opencv.append(error)
            
            # Compute rotation error for reporting
            R_diff = R_est @ R_gt_w2c.T
            rot_err_opencv = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
        
        # Create histogram
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        bins = np.linspace(0, 50, 25)
        ax.hist(reproj_gt, bins=bins, alpha=0.7, label='Ground Truth Pose',
                color='#2ecc71', edgecolor='black', linewidth=1.2)
        
        if reproj_opencv:
            ax.hist(reproj_opencv, bins=bins, alpha=0.7, label='OpenCV solvePnP',
                    color='#3498db', edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Reprojection Error (pixels)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Sanity Check: Reprojection Error Distribution', 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics box
        stats_text = f'Ground Truth:\n  Mean: {np.mean(reproj_gt):.1f}px\n  Median: {np.median(reproj_gt):.1f}px\n'
        if reproj_opencv:
            stats_text += f'\nOpenCV solvePnP:\n  Mean: {np.mean(reproj_opencv):.1f}px\n  Rot Error: {rot_err_opencv:.1f}°'
        
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_sanity_check.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig1_sanity_check.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: fig1_sanity_check.pdf/.png")
        print(f"    - {len(matches)} line matches ({len(points_3d)} point correspondences)")
        print(f"    - GT pose mean error: {np.mean(reproj_gt):.1f} pixels")
        if success:
            print(f"    - OpenCV rotation error: {rot_err_opencv:.1f}°")
    
    def run_slam_analysis(self, data_path: str, max_frames: int = 30) -> Dict:
        """Run SLAM and collect detailed statistics for analysis"""
        loader = Record3DLoader(data_path, max_frames=max_frames)
        
        prev_lines_2d = None
        prev_descriptors = None
        prev_lines_3d = None
        prev_pose = None
        
        results = {
            'rotation_errors': [],
            'translation_errors': [],
            'inlier_counts': [],
            'match_counts': [],
            'inter_frame_rotations': [],
            'all_frames': [],  # Store frame data for visualization
        }
        
        solver = None
        
        for frame_idx in range(len(loader)):
            frame = loader[frame_idx]
            
            if solver is None:
                solver = MinPnLSolver(frame.K, ransac_iters=500, threshold=15.0)
            
            # Detect lines
            rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            lines_2d = pred_lines(rgb_bgr, self.interpreter, self.input_details,
                                  self.output_details, score_thr=0.10, dist_thr=20.0)
            
            if len(lines_2d) == 0:
                continue
            
            # Filter and descriptors
            lines_2d, _ = filter_short_lines_2d(lines_2d, min_length=40.0)
            if len(lines_2d) == 0:
                continue
            
            descriptors, valid_desc = self.descriptor.compute_descriptors(frame.rgb, lines_2d)
            lines_2d = lines_2d[valid_desc]
            
            # Back-project
            lines_3d_cam, valid_3d = backproject_lines(
                lines_2d, frame.depth, frame.K, frame.rgb.shape[:2]
            )
            lines_2d = lines_2d[valid_3d]
            descriptors = descriptors[valid_3d]
            
            # Filter short 3D lines
            if len(lines_3d_cam) > 0:
                valid_3d_length = filter_short_lines_3d(lines_3d_cam, min_length=0.05)
                lines_3d_cam = lines_3d_cam[valid_3d_length]
                lines_2d = lines_2d[valid_3d_length]
                descriptors = descriptors[valid_3d_length]
            
            frame_data = {
                'idx': frame_idx,
                'rgb': frame.rgb.copy(),
                'lines_2d': lines_2d.copy(),
                'n_lines': len(lines_2d)
            }
            
            # Pose estimation (skip first frame)
            if frame_idx > 0 and prev_lines_2d is not None and len(lines_2d) >= 3:
                matches = self.matcher.match_lines(prev_lines_2d, prev_descriptors,
                                                   lines_2d, descriptors)
                
                n_raw_matches = len(matches)
                if len(matches) >= 5:
                    matches, _ = selmap_filter(matches, prev_lines_2d, lines_2d)
                
                frame_data['n_matches'] = len(matches)
                results['match_counts'].append(len(matches))
                
                if len(matches) >= 3:
                    lines_2d_matched = np.array([lines_2d[m[1]] for m in matches])
                    lines_3d_matched = np.array([prev_lines_3d[m[0]] for m in matches])
                    
                    success, R_est, t_est, inliers = solver.estimate_pose(
                        lines_2d_matched, lines_3d_matched
                    )
                    
                    n_inliers = inliers.sum() if success else 0
                    frame_data['n_inliers'] = n_inliers
                    results['inlier_counts'].append(n_inliers)
                    
                    if success and n_inliers >= 3:
                        # Ground truth (world-to-camera)
                        R_gt_c2w = frame.pose[:3, :3]
                        t_gt_c2w = frame.pose[:3, 3]
                        R_gt = R_gt_c2w.T
                        t_gt = -R_gt_c2w.T @ t_gt_c2w
                        
                        rot_err, trans_err = compute_pose_error(R_est, t_est, R_gt, t_gt)
                        
                        results['rotation_errors'].append(rot_err)
                        results['translation_errors'].append(trans_err)
                        
                        frame_data['rot_err'] = rot_err
                        frame_data['trans_err'] = trans_err
                        frame_data['success'] = True
                        
                        # Inter-frame rotation
                        if prev_pose is not None:
                            R_prev = prev_pose[:3, :3]
                            R_curr = frame.pose[:3, :3]
                            R_rel = R_curr.T @ R_prev
                            angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1)))
                            results['inter_frame_rotations'].append(angle)
                            frame_data['inter_frame_rot'] = angle
                    else:
                        frame_data['success'] = False
                else:
                    frame_data['success'] = False
            
            results['all_frames'].append(frame_data)
            
            # Update previous frame data
            R = frame.pose[:3, :3]
            t = frame.pose[:3, 3]
            lines_3d_world = []
            for line in lines_3d_cam:
                P1_world = R @ line[:3] + t
                P2_world = R @ line[3:] + t
                lines_3d_world.append(np.concatenate([P1_world, P2_world]))
            
            prev_lines_2d = lines_2d
            prev_descriptors = descriptors
            prev_lines_3d = np.array(lines_3d_world) if lines_3d_world else np.array([]).reshape(0, 6)
            prev_pose = frame.pose
        
        return results
    
    def generate_boxplot_figure(self, results: Dict):
        """Figure 2 (Section 5.1): Box plot of rotation and translation errors"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Rotation error
        if results['rotation_errors']:
            data = results['rotation_errors']
            bp1 = ax1.boxplot([data], labels=['Video1'], patch_artist=True,
                              showmeans=True, meanline=True,
                              boxprops=dict(facecolor='#3498db', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2),
                              meanprops=dict(color='green', linewidth=2, linestyle='--'),
                              whiskerprops=dict(linewidth=1.5),
                              capprops=dict(linewidth=1.5))
            
            ax1.set_ylabel('Rotation Error (degrees)', fontsize=11, fontweight='bold')
            ax1.set_title('Rotation Error Distribution (Video1)', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            mean_rot = np.mean(data)
            median_rot = np.median(data)
            
            stats_text = f'Mean: {mean_rot:.2f}°\nMedian: {median_rot:.2f}°\nStd: {np.std(data):.2f}°'
            stats_text += f'\nMin: {np.min(data):.2f}°\nMax: {np.max(data):.2f}°'
            stats_text += f'\nFrames: {len(data)}'
            
            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9, family='monospace')
        
        # Translation error
        if results['translation_errors']:
            data = results['translation_errors']
            bp2 = ax2.boxplot([data], labels=['Video1'], patch_artist=True,
                              showmeans=True, meanline=True,
                              boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2),
                              meanprops=dict(color='blue', linewidth=2, linestyle='--'),
                              whiskerprops=dict(linewidth=1.5),
                              capprops=dict(linewidth=1.5))
            
            ax2.set_ylabel('Translation Error (%)', fontsize=11, fontweight='bold')
            ax2.set_title('Translation Error Distribution (Video1)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            mean_trans = np.mean(data)
            median_trans = np.median(data)
            
            stats_text = f'Mean: {mean_trans:.1f}%\nMedian: {median_trans:.1f}%\nStd: {np.std(data):.1f}%'
            stats_text += f'\nMin: {np.min(data):.1f}%\nMax: {np.max(data):.1f}%'
            
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9, family='monospace')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_video1_boxplot.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig2_video1_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: fig2_video1_boxplot.pdf/.png")
        if results['rotation_errors']:
            print(f"    - Median rotation: {median_rot:.2f}°")
            print(f"    - Median translation: {median_trans:.1f}%")
    
    def generate_comparison_figure(self, video1_results: Dict, video2_results: Dict):
        """Figure 3 (Section 5.2): Side-by-side comparison of success vs failure"""
        # Find best success frame from video1
        best_frame = None
        best_error = float('inf')
        for frame in video1_results['all_frames']:
            if frame.get('success', False):
                err = frame.get('rot_err', 999)
                if err < best_error:
                    best_error = err
                    best_frame = frame
        
        # Find worst failure from video2
        worst_frame = None
        worst_error = 0
        for frame in video2_results['all_frames']:
            err = frame.get('rot_err', 0)
            if err > worst_error:
                worst_error = err
                worst_frame = frame
        
        if best_frame is None or worst_frame is None:
            print("  ⚠ Could not find suitable frames for comparison")
            return
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Video1 success
        img1 = best_frame['rgb'].copy()
        for line in best_frame['lines_2d']:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        ax1.imshow(img1)
        title1 = f"Video1 Frame {best_frame['idx']} - SUCCESS"
        title1 += f"\nRot: {best_frame.get('rot_err', 0):.2f}°, Trans: {best_frame.get('trans_err', 0):.1f}%"
        title1 += f"\nInliers: {best_frame.get('n_inliers', 0)}/{best_frame.get('n_matches', 0)}"
        ax1.set_title(title1, fontsize=10, fontweight='bold', color='green')
        ax1.axis('off')
        
        # Video2 failure
        img2 = worst_frame['rgb'].copy()
        for line in worst_frame['lines_2d']:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        ax2.imshow(img2)
        title2 = f"Video2 Frame {worst_frame['idx']} - FAILURE"
        title2 += f"\nRot: {worst_frame.get('rot_err', 999):.2f}°, Trans: {worst_frame.get('trans_err', 999):.1f}%"
        title2 += f"\nInliers: {worst_frame.get('n_inliers', 0)}/{worst_frame.get('n_matches', 0)}"
        ax2.set_title(title2, fontsize=10, fontweight='bold', color='red')
        ax2.axis('off')
        
        plt.suptitle('Comparison: Successful vs Failed Pose Estimation', 
                    fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig3_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: fig3_comparison.pdf/.png")
        print(f"    - Video1 best: {best_error:.2f}° error")
        print(f"    - Video2 worst: {worst_error:.2f}° error")
    
    def generate_scatter_plot(self, video1_results: Dict, video2_results: Dict):
        """Figure 4 (Section 5.3): Scatter plot of rotation error vs inter-frame rotation"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Video1 data
        if video1_results['rotation_errors'] and video1_results['inter_frame_rotations']:
            # Align lengths
            n = min(len(video1_results['rotation_errors']), 
                   len(video1_results['inter_frame_rotations']))
            rot_err_v1 = video1_results['rotation_errors'][:n]
            inter_rot_v1 = video1_results['inter_frame_rotations'][:n]
            
            ax.scatter(inter_rot_v1, rot_err_v1, s=100, alpha=0.6,
                      color='#2ecc71', marker='o', edgecolors='black', linewidth=1,
                      label=f'Video1 Success (n={n})')
        
        # Video2 data
        if video2_results['rotation_errors'] and video2_results['inter_frame_rotations']:
            n = min(len(video2_results['rotation_errors']),
                   len(video2_results['inter_frame_rotations']))
            rot_err_v2 = video2_results['rotation_errors'][:n]
            inter_rot_v2 = video2_results['inter_frame_rotations'][:n]
            
            ax.scatter(inter_rot_v2, rot_err_v2, s=100, alpha=0.6,
                      color='#e74c3c', marker='x', linewidths=2,
                      label=f'Video2 Failure (n={n})')
        
        ax.set_xlabel('Inter-frame Rotation Magnitude (degrees)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Pose Estimation Rotation Error (degrees)', fontsize=11, fontweight='bold')
        ax.set_title('Rotation Error vs Inter-frame Rotation Magnitude',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add threshold line
        ax.axhline(y=5, color='orange', linestyle='--', linewidth=2,
                  label='5° Success Threshold', alpha=0.7)
        
        # Add correlation text if enough points
        if len(inter_rot_v1) > 2:
            corr = np.corrcoef(inter_rot_v1, rot_err_v1)[0, 1]
            ax.text(0.02, 0.98, f'Video1 Correlation: {corr:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                   fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_scatter_rotation.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig4_scatter_rotation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: fig4_scatter_rotation.pdf/.png")
    
    def print_summary(self, video1_results: Dict, video2_results: Dict):
        """Print summary statistics"""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        print("\nVIDEO1:")
        if video1_results['rotation_errors']:
            print(f"  Frames analyzed: {len(video1_results['rotation_errors'])}")
            print(f"  Rotation error:    median={np.median(video1_results['rotation_errors']):.2f}°, "
                  f"mean={np.mean(video1_results['rotation_errors']):.2f}°")
            print(f"  Translation error: median={np.median(video1_results['translation_errors']):.1f}%, "
                  f"mean={np.mean(video1_results['translation_errors']):.1f}%")
            print(f"  Mean inliers: {np.mean(video1_results['inlier_counts']):.1f}")
            all_inlier_frames = sum(1 for x in video1_results['rotation_errors'] if x < 5)
            print(f"  All-inlier frames (<5° error): {all_inlier_frames}/{len(video1_results['rotation_errors'])} "
                  f"({100*all_inlier_frames/len(video1_results['rotation_errors']):.0f}%)")
        
        print("\nVIDEO2:")
        if video2_results['rotation_errors']:
            print(f"  Frames analyzed: {len(video2_results['rotation_errors'])}")
            print(f"  Rotation error:    median={np.median(video2_results['rotation_errors']):.2f}°, "
                  f"mean={np.mean(video2_results['rotation_errors']):.2f}°")
            print(f"  Translation error: median={np.median(video2_results['translation_errors']):.1f}%, "
                  f"mean={np.mean(video2_results['translation_errors']):.1f}%")
            print(f"  Mean inliers: {np.mean(video2_results['inlier_counts']):.1f}")
        
        print()


def main():
    if len(sys.argv) < 4:
        print("Usage: python generate_report_figures.py <video1_path> <video2_path> <mlsd_model_path>")
        print("\nExample:")
        print("  python generate_report_figures.py \\")
        print("    C:/Users/hamee/Downloads/videos/video1 \\")
        print("    C:/Users/hamee/Downloads/videos/video2 \\")
        print("    tflite_models/M-LSD_512_large_fp32.tflite")
        sys.exit(1)
    
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    mlsd_model_path = sys.argv[3]
    
    # Validate paths
    for path, name in [(video1_path, "video1"), (video2_path, "video2"), 
                       (mlsd_model_path, "M-LSD model")]:
        if not Path(path).exists():
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)
    
    # Generate all figures
    generator = ReportFigureGenerator(video1_path, video2_path, mlsd_model_path)
    generator.generate_all_figures()


if __name__ == '__main__':
    main()
