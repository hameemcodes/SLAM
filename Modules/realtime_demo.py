"""
Real-Time SLAM Demo — Open3D + Side-by-Side Matching
======================================================
Panel 1 (OpenCV): Side-by-side previous + current frame with match connections
Panel 2 (Open3D): Interactive 3D map with camera trajectory

At the end, saves a results summary table as poster_figures/results_table.png

Requirements:  pip install open3d

Usage:
    python realtime_demo.py <data_path> [max_frames] [delay_ms]

Controls (OpenCV): q=quit  p=pause
Controls (Open3D): mouse drag=rotate  scroll=zoom  shift+drag=pan
"""

import numpy as np
import cv2
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

from MLSD import pred_lines
from lbd_optimized import LineDescriptorOptimized, LineMatcherOptimized
from map_3d import backproject_lines
from pnp_line_solver import PnPLineSolver, compute_pose_error
from runSLAM2 import selmap_filter

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("WARNING: open3d not installed. pip install open3d")


class Open3DMapViewer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("3D Line Map", width=800, height=600)
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.95, 0.95, 0.95])
        opt.line_width = 2.0
        opt.point_size = 5.0

        self.line_set = o3d.geometry.LineSet()
        self.line_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self.line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=int))
        self.vis.add_geometry(self.line_set)

        self.traj_set = o3d.geometry.LineSet()
        self.traj_set.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self.traj_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=int))
        self.vis.add_geometry(self.traj_set)

        self.cam_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        self.cam_marker.paint_uniform_color([0, 1, 0])
        self.vis.add_geometry(self.cam_marker)

        self.map_points, self.map_lines = [], []
        self.cam_positions = []
        self.n_map_lines = 0
        self.first_view_set = False

    def add_lines(self, lines_world):
        base = len(self.map_points)
        for line in lines_world:
            self.map_points.append(line[:3])
            self.map_points.append(line[3:])
            self.map_lines.append([base, base + 1])
            base += 2
        self.n_map_lines = len(self.map_lines)
        self._update_lines()

    def set_camera(self, pos):
        self.cam_positions.append(pos.copy())
        self._update_traj()
        self.cam_marker.translate(pos - np.array(self.cam_marker.get_center()), relative=True)
        self.vis.update_geometry(self.cam_marker)

    def _update_lines(self):
        if not self.map_points: return
        self.line_set.points = o3d.utility.Vector3dVector(np.array(self.map_points))
        self.line_set.lines = o3d.utility.Vector2iVector(np.array(self.map_lines, dtype=np.int32))
        self.line_set.colors = o3d.utility.Vector3dVector(np.tile([0, 0.75, 0.75], (len(self.map_lines), 1)))
        self.vis.update_geometry(self.line_set)

    def _update_traj(self):
        if len(self.cam_positions) < 2: return
        pts = np.array(self.cam_positions)
        lines = [[i, i+1] for i in range(len(pts)-1)]
        self.traj_set.points = o3d.utility.Vector3dVector(pts)
        self.traj_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
        self.traj_set.colors = o3d.utility.Vector3dVector(np.tile([0, 0.7, 0], (len(lines), 1)))
        self.vis.update_geometry(self.traj_set)

    def poll(self):
        if not self.first_view_set and len(self.map_points) > 6:
            self.vis.reset_view_point(True)
            self.first_view_set = True
        self.vis.poll_events()
        self.vis.update_renderer()

    def destroy(self):
        self.vis.destroy_window()


class FallbackRenderer:
    def __init__(self):
        self.n_map_lines = 0
        self.cam_positions = []
    def add_lines(self, lw): self.n_map_lines += len(lw)
    def set_camera(self, p): self.cam_positions.append(p.copy())
    def poll(self): pass
    def destroy(self): pass


def detect_freiburg_camera(dp):
    n = Path(dp).name.lower()
    if "freiburg2" in n: return "freiburg2"
    if "freiburg3" in n: return "freiburg3"
    return "freiburg1"


def transform_and_filter(lines_3d_cam, R_c2w, t_c2w, existing_lines=None):
    F_flip = np.diag([1.0, -1.0, -1.0])
    good = []
    for line in lines_3d_cam:
        P1 = R_c2w @ (F_flip @ line[:3]) + t_c2w
        P2 = R_c2w @ (F_flip @ line[3:]) + t_c2w
        if np.any(np.abs(P1) > 50) or np.any(np.abs(P2) > 50): continue
        length = np.linalg.norm(P2 - P1)
        if length > 3.0 or length < 0.02: continue
        if existing_lines and len(existing_lines) > 0:
            mid = (P1 + P2) / 2
            d = (P2 - P1) / (length + 1e-10)
            dup = False
            for ex in existing_lines[-200:]:
                if np.linalg.norm(mid - (ex[:3]+ex[3:])/2) < 0.03:
                    de = ex[3:]-ex[:3]; de /= (np.linalg.norm(de)+1e-10)
                    if abs(np.dot(d, de)) > 0.85: dup = True; break
            if dup: continue
        good.append(np.concatenate([P1, P2]))
    return good


def build_match_canvas(prev_rgb, prev_lines, curr_rgb, curr_lines, matches,
                       target_h=400):
    """Build side-by-side matching visualisation."""
    h1, w1 = prev_rgb.shape[:2]
    h2, w2 = curr_rgb.shape[:2]

    # Scale both to target height
    s1 = target_h / h1
    s2 = target_h / h2
    pw = int(w1 * s1)
    cw = int(w2 * s2)
    prev_r = cv2.resize(prev_rgb, (pw, target_h))
    curr_r = cv2.resize(curr_rgb, (cw, target_h))

    canvas = np.zeros((target_h, pw + cw, 3), dtype=np.uint8)
    canvas[:, :pw] = prev_r
    canvas[:, pw:] = curr_r

    # Draw all detected lines (green, thin)
    for l in prev_lines:
        x1, y1, x2, y2 = int(l[0]*s1), int(l[1]*s1), int(l[2]*s1), int(l[3]*s1)
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 180, 0), 1)
    for l in curr_lines:
        x1, y1, x2, y2 = int(l[0]*s2)+pw, int(l[1]*s2), int(l[2]*s2)+pw, int(l[3]*s2)
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 180, 0), 1)

    # Draw matches (colored lines + connections)
    np.random.seed(42)
    for idx_prev, idx_curr in matches:
        color = tuple(int(c) for c in np.random.randint(80, 255, 3))
        lp = prev_lines[idx_prev]
        lc = curr_lines[idx_curr]
        # Matched lines (thicker)
        cv2.line(canvas,
                 (int(lp[0]*s1), int(lp[1]*s1)),
                 (int(lp[2]*s1), int(lp[3]*s1)), color, 2)
        cv2.line(canvas,
                 (int(lc[0]*s2)+pw, int(lc[1]*s2)),
                 (int(lc[2]*s2)+pw, int(lc[3]*s2)), color, 2)
        # Connection between midpoints
        mid_p = (int((lp[0]+lp[2])/2*s1), int((lp[1]+lp[3])/2*s1))
        mid_c = (int((lc[0]+lc[2])/2*s2)+pw, int((lc[1]+lc[3])/2*s2))
        cv2.line(canvas, mid_p, mid_c, color, 1, cv2.LINE_AA)

    # Divider line
    cv2.line(canvas, (pw, 0), (pw, target_h), (100, 100, 100), 1)

    return canvas


def save_results_table(rot_errors, trans_errors, n_frames, n_map_lines, ds_name):
    """Save summary metrics as a poster-ready table image."""
    out_dir = Path("poster_figures")
    out_dir.mkdir(exist_ok=True)

    rows = [
        ["Dataset", ds_name],
        ["Total Frames", str(n_frames)],
        ["Pose Estimates", str(len(rot_errors))],
        ["3D Map Lines", str(n_map_lines)],
    ]
    if rot_errors:
        rows.append(["Mean Rotation Error", f"{np.mean(rot_errors):.2f}\u00b0"])
        rows.append(["Median Rotation Error", f"{np.median(rot_errors):.2f}\u00b0"])
        rows.append(["Mean Translation Error", f"{np.mean(trans_errors):.1f}%"])
        rows.append(["Median Translation Error", f"{np.median(trans_errors):.1f}%"])

    fig, ax = plt.subplots(figsize=(8, len(rows) * 0.55 + 1))
    ax.axis('off')
    ax.set_title('SLAM Pipeline Results', fontsize=18, fontweight='bold', pad=20)

    table = ax.table(
        cellText=rows, colLabels=["Metric", "Value"],
        loc='center', cellLoc='center', colWidths=[0.55, 0.35]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.0)

    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor('#008B8B')
        cell.set_text_props(color='white', fontweight='bold', fontsize=15)

    for i, row in enumerate(rows):
        for j in range(2):
            cell = table[i+1, j]
            cell.set_facecolor('#f0f8f8' if i % 2 == 0 else 'white')
            if 'Error' in row[0] and j == 1:
                cell.set_text_props(fontweight='bold', color='#006400')

    plt.tight_layout()
    out = out_dir / "results_table.png"
    fig.savefig(str(out), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Results table saved: {out}")


def run_demo(data_path, max_frames=100, delay_ms=1):
    mlsd_model = "tflite_models/M-LSD_512_large_fp32.tflite"
    if not Path(mlsd_model).exists():
        print(f"ERROR: M-LSD model not found at {mlsd_model}"); sys.exit(1)

    interpreter = tf.lite.Interpreter(model_path=mlsd_model)
    interpreter.allocate_tensors()
    inp_d = interpreter.get_input_details()
    out_d = interpreter.get_output_details()

    descriptor = LineDescriptorOptimized(num_bands=7, band_width=5, max_lines=150)
    matcher = LineMatcherOptimized(descriptor_distance_threshold=0.4,
                                   geometric_distance_threshold=100.0,
                                   angle_threshold_deg=25.0)
    solver = None

    dp = Path(data_path)
    if (dp / "groundtruth.txt").exists():
        from TUM_loader import TUMLoader
        cam = detect_freiburg_camera(data_path)
        loader = TUMLoader(str(dp), max_frames=max_frames, camera=cam)
        ds_name = f"TUM ({cam})"
    else:
        from LIDAR_loader import Record3DLoader
        loader = Record3DLoader(str(dp), max_frames=max_frames)
        ds_name = "Record3D"

    if HAS_OPEN3D:
        viewer = Open3DMapViewer()
        print(f"Dataset: {ds_name}, {len(loader)} frames  |  Open3D viewer active")
    else:
        viewer = FallbackRenderer()
        print(f"Dataset: {ds_name}, {len(loader)} frames  |  No Open3D")

    print(f"Controls: [q] quit  [p] pause\n")

    prev_lines_2d = prev_desc = prev_lines_3d = None
    prev_rgb = None
    rot_errors, trans_errors = [], []
    last_kf_pos = None
    all_map_lines = []

    for fi in range(len(loader)):
        frame = loader[fi]

        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        lines_2d = pred_lines(bgr, interpreter, inp_d, out_d, score_thr=0.10, dist_thr=20.0)

        if len(lines_2d) > 0:
            lens = np.sqrt((lines_2d[:, 2]-lines_2d[:, 0])**2 + (lines_2d[:, 3]-lines_2d[:, 1])**2)
            lines_2d = lines_2d[lens >= 40]

        desc = np.array([])
        if len(lines_2d) > 0:
            desc, valid = descriptor.compute_descriptors(frame.rgb, lines_2d)
            lines_2d = lines_2d[valid] if len(valid) else lines_2d

        lines_3d_cam = np.array([]).reshape(0, 6)
        if len(lines_2d) > 0 and len(desc) > 0:
            lines_3d_cam, valid_3d = backproject_lines(
                lines_2d, frame.depth, frame.K, frame.rgb.shape[:2])
            lines_2d = lines_2d[valid_3d]; desc = desc[valid_3d]
            if len(lines_3d_cam) > 0:
                l3 = np.sqrt(np.sum((lines_3d_cam[:, 3:]-lines_3d_cam[:, :3])**2, axis=1))
                m3 = l3 >= 0.05
                lines_3d_cam = lines_3d_cam[m3]; lines_2d = lines_2d[m3]; desc = desc[m3]

        n_matches, n_inliers = 0, 0
        matched_indices_curr = []

        if solver is None and len(lines_2d) > 0:
            solver = PnPLineSolver(frame.K, ransac_iters=1000, threshold=15.0)

        matches = []
        if fi > 0 and prev_lines_2d is not None and len(lines_2d) >= 3 and len(desc) >= 3:
            matches = matcher.match_lines(prev_lines_2d, prev_desc, lines_2d, desc)
            if len(matches) >= 5:
                matches, _ = selmap_filter(matches, prev_lines_2d, lines_2d)
            n_matches = len(matches)
            matched_indices_curr = [m[1] for m in matches]

            if n_matches >= 3:
                l2d_m = np.array([lines_2d[m[1]] for m in matches])
                l3d_m = np.array([prev_lines_3d[m[0]] for m in matches])
                ok, R_est, t_est, inliers = solver.estimate_pose(l2d_m, l3d_m)
                n_inliers = inliers.sum() if ok else 0
                if ok:
                    R_gt_c2w = frame.pose[:3, :3]; t_gt_c2w = frame.pose[:3, 3]
                    F = np.diag([1.0, -1.0, -1.0])
                    re, te, _ = compute_pose_error(R_est, t_est, F @ R_gt_c2w.T, F @ (-R_gt_c2w.T @ t_gt_c2w))
                    rot_errors.append(re); trans_errors.append(te)

        R_c2w = frame.pose[:3, :3]; t_c2w = frame.pose[:3, 3]

        # ── Keyframe map insertion ──────────────────────────────────
        is_keyframe = fi == 0 or (last_kf_pos is not None and np.linalg.norm(t_c2w - last_kf_pos) > 0.05)

        if is_keyframe and len(lines_3d_cam) > 0:
            if fi == 0:
                new_lines = transform_and_filter(lines_3d_cam, R_c2w, t_c2w)
            else:
                unmatched = sorted(set(range(len(lines_3d_cam))) - set(matched_indices_curr))
                new_lines = transform_and_filter(
                    lines_3d_cam[np.array(unmatched)] if unmatched else np.array([]).reshape(0,6),
                    R_c2w, t_c2w, existing_lines=all_map_lines
                ) if unmatched else []

            if new_lines:
                viewer.add_lines(np.array(new_lines))
                all_map_lines.extend(new_lines)
            last_kf_pos = t_c2w.copy()

        viewer.set_camera(t_c2w)

        # Store for next frame
        F_flip = np.diag([1.0, -1.0, -1.0])
        if len(lines_2d) > 0 and len(lines_3d_cam) > 0:
            prev_3d = []
            for line in lines_3d_cam:
                P1 = R_c2w @ (F_flip @ line[:3]) + t_c2w
                P2 = R_c2w @ (F_flip @ line[3:]) + t_c2w
                prev_3d.append(np.concatenate([P1, P2]))
            prev_lines_3d = np.array(prev_3d) if prev_3d else np.array([]).reshape(0, 6)

        # ── Side-by-side matching visualisation ─────────────────────
        if fi > 0 and prev_rgb is not None and len(matches) > 0 and prev_lines_2d is not None:
            canvas = build_match_canvas(prev_rgb, prev_lines_2d, frame.rgb,
                                        lines_2d, matches, target_h=400)
        else:
            # First frame or no matches: just show current frame with lines
            vis = frame.rgb.copy()
            for l in lines_2d:
                cv2.line(vis, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, 0), 2)
            canvas = cv2.resize(vis, (int(vis.shape[1] * 400 / vis.shape[0]), 400))

        # HUD overlay
        hud = [
            f"Frame {fi}/{len(loader)-1}  [{ds_name}]",
            f"Lines: {len(lines_2d)}   Matches: {n_matches}   Inliers: {n_inliers}",
            f"Map: {viewer.n_map_lines}   KF: {'Yes' if is_keyframe else '-'}",
        ]
        if rot_errors:
            hud.append(f"Err: {rot_errors[-1]:.2f}deg / {trans_errors[-1]:.1f}%")

        bar_h = 22 * len(hud) + 8
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (canvas.shape[1], bar_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, canvas)
        for j, txt in enumerate(hud):
            cv2.putText(canvas, txt, (8, 18 + j * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Line-Based Visual SLAM", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

        # Update prev for next iteration (AFTER building canvas)
        if len(lines_2d) > 0:
            prev_lines_2d = lines_2d
            prev_desc = desc
            prev_rgb = frame.rgb.copy()

        viewer.poll()

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'):
            print("  Paused")
            while True:
                viewer.poll()
                k = cv2.waitKey(100) & 0xFF
                if k == ord('p'): break
                if k == ord('q'): viewer.destroy(); cv2.destroyAllWindows(); return

        err_str = f" | {rot_errors[-1]:.2f}deg / {trans_errors[-1]:.1f}%" if rot_errors else ""
        kf = " [KF]" if is_keyframe else ""
        print(f"  Frame {fi}: {len(lines_2d)} lines, {n_matches} match, "
              f"map={viewer.n_map_lines}{kf}{err_str}")

    # ── End of run: print + save results table ──────────────────────
    print("\n" + "=" * 55)
    print("DEMO COMPLETE")
    print("=" * 55)
    if rot_errors:
        print(f"  Mean rotation error:    {np.mean(rot_errors):.2f}\u00b0")
        print(f"  Median rotation error:  {np.median(rot_errors):.2f}\u00b0")
        print(f"  Mean translation error: {np.mean(trans_errors):.1f}%")
        print(f"  Median translation error: {np.median(trans_errors):.1f}%")
    print(f"  Total frames: {len(loader)}")
    print(f"  Pose estimates: {len(rot_errors)}")
    print(f"  Total map lines: {viewer.n_map_lines}")

    save_results_table(rot_errors, trans_errors, len(loader),
                       viewer.n_map_lines, ds_name)

    print("\nClose Open3D window or press any key to exit ...")
    cv2.waitKey(0)
    viewer.destroy()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    run_demo(sys.argv[1],
             int(sys.argv[2]) if len(sys.argv) > 2 else 100,
             int(sys.argv[3]) if len(sys.argv) > 3 else 1)