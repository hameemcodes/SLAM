"""
Generate Poster Figures for Line-Based Visual SLAM
===================================================
Generates:
  1. Line detection + matching visualisation
  2. SelMap histogram (magnitudes & angles with mode + rejection)
  3. Camera trajectory vs ground truth (NO loop closure)
  4. 3D map with trajectory overlay
  5. Results metrics table (image for poster)

Usage:
    python generate_poster_figures.py <data_path> [max_frames]
"""

import numpy as np
import cv2
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from MLSD import pred_lines
from lbd_optimized import LineDescriptorOptimized, LineMatcherOptimized
from runSLAM2 import LineSLAM, selmap_filter
import tensorflow as tf

OUTPUT_DIR = Path("poster_figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def is_tum(data_path):
    return (Path(data_path) / "groundtruth.txt").exists()

def get_loader(data_path, max_frames):
    if is_tum(data_path):
        from TUM_loader import TUMLoader
        return TUMLoader(str(data_path), max_frames=max_frames), "TUM"
    else:
        from LIDAR_loader import Record3DLoader
        return Record3DLoader(str(data_path), max_frames=max_frames), "Record3D"


def selmap_filter_with_histdata(matches, lines1, lines2, threshold_factor=1.5):
    if len(matches) < 5:
        return matches, 0, {}
    centers1 = (lines1[:, :2] + lines1[:, 2:]) / 2
    centers2 = (lines2[:, :2] + lines2[:, 2:]) / 2
    vectors = np.array([centers2[m[1]] - centers1[m[0]] for m in matches])
    lengths = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    len_hist, len_edges = np.histogram(lengths, bins=30)
    mode_idx = np.argmax(len_hist)
    mode_len = (len_edges[mode_idx] + len_edges[mode_idx + 1]) / 2
    ang_hist, ang_edges = np.histogram(angles, bins=36)
    mode_idx_a = np.argmax(ang_hist)
    mode_ang = (ang_edges[mode_idx_a] + ang_edges[mode_idx_a + 1]) / 2
    len_threshold = threshold_factor * np.std(lengths)
    ang_threshold = threshold_factor * np.std(angles)
    inlier_mask = np.array([
        abs(lengths[i] - mode_len) < len_threshold and
        abs(angles[i] - mode_ang) < ang_threshold
        for i in range(len(matches))
    ])
    inliers = [m for i, m in enumerate(matches) if inlier_mask[i]]
    hist_data = dict(lengths=lengths, angles=angles, mode_len=mode_len,
                     mode_ang=mode_ang, len_threshold=len_threshold,
                     ang_threshold=ang_threshold, inlier_mask=inlier_mask)
    return inliers, (~inlier_mask).sum(), hist_data


# =====================================================================
#  FIGURE 1 — Line detection + matching
# =====================================================================

def generate_line_detection_and_matching(loader, interpreter, input_details,
                                         output_details, frame_a=0, frame_b=1):
    print("\n[Fig 1] Line detection + matching ...")
    descriptor = LineDescriptorOptimized(num_bands=7, band_width=5, max_lines=150)
    matcher = LineMatcherOptimized()
    fa, fb = loader[frame_a], loader[frame_b]

    def detect(frame):
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        lines = pred_lines(bgr, interpreter, input_details, output_details,
                           score_thr=0.10, dist_thr=20.0)
        if len(lines) == 0:
            return np.array([]), np.array([])
        lens = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
        lines = lines[lens >= 40]
        if len(lines) == 0:
            return np.array([]), np.array([])
        desc, valid = descriptor.compute_descriptors(frame.rgb, lines)
        return lines[valid] if len(valid) else lines, desc

    lines_a, desc_a = detect(fa)
    lines_b, desc_b = detect(fb)
    if len(lines_a) == 0 or len(lines_b) == 0:
        print("  Not enough lines, skipping.")
        return None

    matches_raw = matcher.match_lines(lines_a, desc_a, lines_b, desc_b)

    h, w = fa.rgb.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = fa.rgb.copy()
    canvas[:, w:] = fb.rgb.copy()

    for l in lines_a:
        cv2.line(canvas, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 200, 0), 1)
    for l in lines_b:
        cv2.line(canvas, (int(l[0]) + w, int(l[1])), (int(l[2]) + w, int(l[3])), (0, 200, 0), 1)

    np.random.seed(42)
    for idx_a, idx_b in matches_raw:
        color = tuple(int(c) for c in np.random.randint(80, 255, 3))
        la, lb = lines_a[idx_a], lines_b[idx_b]
        cv2.line(canvas, (int(la[0]), int(la[1])), (int(la[2]), int(la[3])), color, 2)
        cv2.line(canvas, (int(lb[0]) + w, int(lb[1])), (int(lb[2]) + w, int(lb[3])), color, 2)
        mid_a = (int((la[0] + la[2]) / 2), int((la[1] + la[3]) / 2))
        mid_b = (int((lb[0] + lb[2]) / 2) + w, int((lb[1] + lb[3]) / 2))
        cv2.line(canvas, mid_a, mid_b, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"Frame {frame_a}: {len(lines_a)} lines",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, f"Frame {frame_b}: {len(lines_b)} lines",
                (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, f"{len(matches_raw)} matches",
                (w - 80, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    out = OUTPUT_DIR / "fig1_line_detection_matching.png"
    cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"  Saved {out}")
    return matches_raw, lines_a, lines_b


# =====================================================================
#  FIGURE 2 — SelMap histogram
# =====================================================================

def generate_selmap_histogram(matches_raw, lines_a, lines_b):
    print("\n[Fig 2] SelMap filtering histogram ...")
    if len(matches_raw) < 5:
        print("  Not enough matches, skipping.")
        return
    _, _, hd = selmap_filter_with_histdata(matches_raw, lines_a, lines_b)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    inlier_lens = hd['lengths'][hd['inlier_mask']]
    outlier_lens = hd['lengths'][~hd['inlier_mask']]
    bins_l = np.linspace(hd['lengths'].min(), hd['lengths'].max(), 31)
    ax1.hist(inlier_lens, bins=bins_l, color='#2ecc71', alpha=0.8, label='Inliers')
    ax1.hist(outlier_lens, bins=bins_l, color='#e74c3c', alpha=0.7, label='Outliers')
    ax1.axvline(hd['mode_len'], color='blue', lw=2, ls='--', label=f"Mode={hd['mode_len']:.1f}px")
    ax1.axvline(hd['mode_len'] - hd['len_threshold'], color='orange', lw=1.5, ls=':')
    ax1.axvline(hd['mode_len'] + hd['len_threshold'], color='orange', lw=1.5, ls=':', label='Bounds')
    ax1.axvspan(hd['mode_len'] - hd['len_threshold'], hd['mode_len'] + hd['len_threshold'], alpha=0.08, color='green')
    ax1.set_xlabel('Displacement Magnitude (px)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax1.set_title('Displacement Magnitude', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)

    inlier_angs = np.degrees(hd['angles'][hd['inlier_mask']])
    outlier_angs = np.degrees(hd['angles'][~hd['inlier_mask']])
    bins_a = np.linspace(np.degrees(hd['angles']).min(), np.degrees(hd['angles']).max(), 37)
    ax2.hist(inlier_angs, bins=bins_a, color='#2ecc71', alpha=0.8, label='Inliers')
    ax2.hist(outlier_angs, bins=bins_a, color='#e74c3c', alpha=0.7, label='Outliers')
    ax2.axvline(np.degrees(hd['mode_ang']), color='blue', lw=2, ls='--',
                label=f"Mode={np.degrees(hd['mode_ang']):.1f}\u00b0")
    ax2.axvline(np.degrees(hd['mode_ang'] - hd['ang_threshold']), color='orange', lw=1.5, ls=':')
    ax2.axvline(np.degrees(hd['mode_ang'] + hd['ang_threshold']), color='orange', lw=1.5, ls=':', label='Bounds')
    ax2.axvspan(np.degrees(hd['mode_ang'] - hd['ang_threshold']),
                np.degrees(hd['mode_ang'] + hd['ang_threshold']), alpha=0.08, color='green')
    ax2.set_xlabel('Displacement Angle (\u00b0)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax2.set_title('Displacement Angle', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)

    n_in = hd['inlier_mask'].sum()
    n_out = (~hd['inlier_mask']).sum()
    fig.suptitle(f'SelMap Outlier Filtering \u2014 {n_in} inliers, {n_out} rejected',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_selmap_histogram.png"
    fig.savefig(str(out), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# =====================================================================
#  Run SLAM (no loop closure)
# =====================================================================

def run_slam_no_lc(data_path, max_frames):
    mlsd_model = "tflite_models/M-LSD_512_large_fp32.tflite"
    slam = LineSLAM(mlsd_model, vocab_path=None)
    if is_tum(data_path):
        slam.run_tum(str(data_path), max_frames=max_frames, use_gt_pose=False)
    else:
        slam.run(str(data_path), max_frames=max_frames, use_gt_pose=False)
    return slam


# =====================================================================
#  FIGURE 3 — Camera trajectory vs ground truth (NO LC)
# =====================================================================

def generate_trajectory_figure(slam, ds_name):
    print("\n[Fig 3] Camera trajectory vs ground truth ...")

    gt_pos = np.array([p[:3, 3] for p in slam.gt_poses])
    est_pos = np.array([-R.T @ t for R, t in slam.estimated_poses]) if slam.estimated_poses else np.array([])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], 'g-o', markersize=5, linewidth=2, label='Ground Truth')
    if len(est_pos) > 0:
        ax.plot(est_pos[:, 0], est_pos[:, 2], 'r-^', markersize=5, linewidth=2, label='Estimated')
    ax.set_xlabel('X (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Z (m)', fontsize=14, fontweight='bold')
    ax.set_title(f'Camera Trajectory \u2014 {ds_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if slam.rotation_errors:
        stats = (f"Mean rot error: {np.mean(slam.rotation_errors):.2f}\u00b0\n"
                 f"Mean trans error: {np.mean(slam.translation_errors):.1f}%")
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out = OUTPUT_DIR / "fig3_trajectory.png"
    fig.savefig(str(out), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# =====================================================================
#  FIGURE 4 — 3D map with trajectory overlay
# =====================================================================

def generate_3d_map_figure(slam, ds_name):
    print("\n[Fig 4] 3D map with trajectory overlay ...")

    lines_3d, _ = slam.map.get_lines_and_descriptors()
    gt_pos = np.array([p[:3, 3] for p in slam.gt_poses])

    # Outlier filtering
    if len(lines_3d) > 0:
        all_coords = lines_3d.reshape(-1, 3)
        median = np.median(all_coords, axis=0)
        mad = np.median(np.abs(all_coords - median), axis=0)
        threshold = 10.0 * (mad + 1e-6)
        p1 = lines_3d[:, :3]
        p2 = lines_3d[:, 3:]
        valid = np.all(np.abs(p1 - median) < threshold, axis=1) & np.all(np.abs(p2 - median) < threshold, axis=1)
        n_before = len(lines_3d)
        lines_3d = lines_3d[valid]
        if n_before - len(lines_3d) > 0:
            print(f"  Filtered {n_before - len(lines_3d)} outlier lines")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    max_lines = 800
    if len(lines_3d) > max_lines:
        idx = np.random.choice(len(lines_3d), max_lines, replace=False)
        lines_draw = lines_3d[idx]
    else:
        lines_draw = lines_3d

    for line in lines_draw:
        ax.plot([line[0], line[3]], [line[1], line[4]], [line[2], line[5]],
                'c-', alpha=0.4, linewidth=0.8)

    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
            'g-o', markersize=4, linewidth=2.5, label='Camera Trajectory')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'3D Line Map \u2014 {len(lines_3d)} lines, {ds_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)

    if len(lines_3d) > 0:
        pts = lines_3d.reshape(-1, 3)
        centre = np.median(pts, axis=0)
        span = np.percentile(np.abs(pts - centre), 95)
        ax.set_xlim([centre[0] - span, centre[0] + span])
        ax.set_ylim([centre[1] - span, centre[1] + span])
        ax.set_zlim([centre[2] - span, centre[2] + span])

    plt.tight_layout()
    out = OUTPUT_DIR / "fig4_3d_map.png"
    fig.savefig(str(out), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# =====================================================================
#  FIGURE 5 — Results metrics table
# =====================================================================

def generate_metrics_table(slam, ds_name, n_frames):
    print("\n[Fig 5] Results metrics table ...")

    rows = []
    rows.append(["Dataset", ds_name])
    rows.append(["Total Frames", str(n_frames)])
    rows.append(["Successful Pose Estimates", str(len(slam.estimated_poses))])
    rows.append(["3D Map Lines", str(len(slam.map))])

    if slam.rotation_errors:
        rows.append(["Mean Rotation Error", f"{np.mean(slam.rotation_errors):.2f}\u00b0"])
        rows.append(["Median Rotation Error", f"{np.median(slam.rotation_errors):.2f}\u00b0"])
        rows.append(["Mean Translation Error", f"{np.mean(slam.translation_errors):.1f}%"])
        rows.append(["Median Translation Error", f"{np.median(slam.translation_errors):.1f}%"])

    fig, ax = plt.subplots(figsize=(8, len(rows) * 0.55 + 1))
    ax.axis('off')
    ax.set_title('SLAM Pipeline Results', fontsize=18, fontweight='bold', pad=20)

    col_labels = ["Metric", "Value"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colWidths=[0.55, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.0)

    # Style header
    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor('#008B8B')
        cell.set_text_props(color='white', fontweight='bold', fontsize=15)

    # Style data rows — highlight error rows
    for i, row in enumerate(rows):
        for j in range(2):
            cell = table[i + 1, j]
            cell.set_facecolor('#f0f8f8' if i % 2 == 0 else 'white')
            if 'Error' in row[0]:
                cell.set_text_props(fontweight='bold')
                if j == 1:
                    cell.set_text_props(fontweight='bold', color='#006400')

    plt.tight_layout()
    out = OUTPUT_DIR / "fig5_metrics_table.png"
    fig.savefig(str(out), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")

    # Also print to console
    print("\n  " + "=" * 45)
    for label, value in rows:
        print(f"  {label:30s} {value}")
    print("  " + "=" * 45)


# =====================================================================
#  MAIN
# =====================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    data_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None

    mlsd_model = "tflite_models/M-LSD_512_large_fp32.tflite"
    if not Path(mlsd_model).exists():
        print(f"ERROR: M-LSD model not found at {mlsd_model}")
        sys.exit(1)

    interpreter = tf.lite.Interpreter(model_path=mlsd_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    loader, ds_name = get_loader(data_path, max_frames)
    print(f"Dataset: {ds_name}, {len(loader)} frames\n")

    # ── Fig 1: Line detection + matching
    result = generate_line_detection_and_matching(loader, interpreter, inp, out, frame_a=0, frame_b=1)
    if result:
        matches_raw, lines_a, lines_b = result
        # ── Fig 2: SelMap histogram
        generate_selmap_histogram(matches_raw, lines_a, lines_b)

    # ── Figs 3, 4, 5: Run SLAM once (no LC), generate all remaining figures
    print("\n[Fig 3+4+5] Running SLAM (no loop closure) ...")
    slam = run_slam_no_lc(data_path, max_frames)
    generate_trajectory_figure(slam, ds_name)
    generate_3d_map_figure(slam, ds_name)
    generate_metrics_table(slam, ds_name, len(loader))

    print("\n" + "=" * 60)
    print(f"ALL FIGURES SAVED TO: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()