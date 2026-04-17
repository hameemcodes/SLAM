"""
PnP-based Line Pose Estimator

Practical pose estimator for line-based SLAM using OpenCV's solvePnP
on line endpoints. This serves as a robust fallback when the MinPnL 
solver struggles with CGR degeneracy on cluttered / large-motion scenes.

Approach:
  - Each 2D-3D line correspondence yields 2 point correspondences (endpoints)
  - OpenCV solvePnPRansac provides robust initial pose
  - Inlier checking uses line reprojection error (geometric, not point-based)
  - Optional refinement with all inliers via solvePnP(ITERATIVE)

Convention:
  Returns R, t mapping world -> OpenCV camera (P_cam = R @ P_world + t)
  This is the standard OpenCV convention (Y-down, Z-forward).
"""
import numpy as np
import cv2
from typing import Tuple


class PnPLineSolver:
    def __init__(self, K: np.ndarray, ransac_iters: int = 1000, 
                 threshold: float = 10.0, reproj_threshold: float = 8.0):
        """
        Args:
            K: 3x3 camera intrinsic matrix
            ransac_iters: iterations for solvePnPRansac
            threshold: line reprojection error threshold for inlier classification (pixels)
            reproj_threshold: point reprojection threshold for solvePnPRansac (pixels)
        """
        self.K = K.astype(np.float64)
        self.ransac_iters = ransac_iters
        self.threshold = threshold
        self.reproj_threshold = reproj_threshold

    def estimate_pose(self, lines_2d: np.ndarray, lines_3d: np.ndarray,
                      debug: bool = False) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate camera pose from 2D-3D line correspondences.

        Args:
            lines_2d: Nx4 [x1,y1,x2,y2] in pixels
            lines_3d: Nx6 [X1,Y1,Z1,X2,Y2,Z2] in WORLD frame

        Returns:
            success, R (world-to-cam), t (world-to-cam), inlier_mask (per line)
            Convention: P_cam = R @ P_world + t  (OpenCV: Y-down, Z-forward)
        """
        N = len(lines_2d)
        if N < 3:
            return False, np.eye(3), np.zeros(3), np.zeros(N, dtype=bool)

        # Build point correspondences from line endpoints
        points_3d, points_2d = self._lines_to_points(lines_2d, lines_3d)

        # Stage 1: Robust initial estimate via RANSAC
        try:
            success, rvec, tvec, pt_inliers = cv2.solvePnPRansac(
                points_3d, points_2d, self.K, distCoeffs=None,
                iterationsCount=self.ransac_iters,
                reprojectionError=self.reproj_threshold,
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except Exception as e:
            if debug:
                print(f"[PnPLine] solvePnPRansac exception: {e}")
            return False, np.eye(3), np.zeros(3), np.zeros(N, dtype=bool)

        if not success or pt_inliers is None or len(pt_inliers) < 6:
            if debug:
                n_inl = len(pt_inliers) if pt_inliers is not None else 0
                print(f"[PnPLine] solvePnPRansac failed or too few inliers ({n_inl})")
            return False, np.eye(3), np.zeros(3), np.zeros(N, dtype=bool)

        R_est, _ = cv2.Rodrigues(rvec)
        t_est = tvec.flatten()

        # Stage 2: Classify LINE inliers using line reprojection error
        line_inliers = self._get_line_inliers(lines_2d, lines_3d, R_est, t_est)

        if debug:
            print(f"[PnPLine] Point inliers: {len(pt_inliers)}/{len(points_3d)}, "
                  f"Line inliers: {line_inliers.sum()}/{N}")

        # Stage 3: Refine with line-inlier endpoints only
        if line_inliers.sum() >= 3:
            inlier_2d = lines_2d[line_inliers]
            inlier_3d = lines_3d[line_inliers]
            pts3, pts2 = self._lines_to_points(inlier_2d, inlier_3d)

            try:
                success2, rvec2, tvec2 = cv2.solvePnP(
                    pts3, pts2, self.K, distCoeffs=None,
                    rvec=rvec, tvec=tvec,  # use initial as starting point
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success2:
                    R_est, _ = cv2.Rodrigues(rvec2)
                    t_est = tvec2.flatten()
                    # Re-evaluate inliers with refined pose
                    line_inliers = self._get_line_inliers(lines_2d, lines_3d, R_est, t_est)
            except Exception:
                pass  # keep the RANSAC solution

        return line_inliers.sum() >= 3, R_est, t_est, line_inliers

    def _lines_to_points(self, lines_2d, lines_3d):
        """Convert line correspondences to point correspondences (both endpoints)."""
        N = len(lines_2d)
        points_3d = np.zeros((2 * N, 3), dtype=np.float64)
        points_2d = np.zeros((2 * N, 2), dtype=np.float64)

        for i in range(N):
            points_3d[2*i]     = lines_3d[i, :3]
            points_3d[2*i + 1] = lines_3d[i, 3:]
            points_2d[2*i]     = lines_2d[i, :2]
            points_2d[2*i + 1] = lines_2d[i, 2:]

        return points_3d, points_2d

    def _get_line_inliers(self, lines_2d, lines_3d, R, t):
        """Compute per-line inlier mask using line reprojection error."""
        N = len(lines_2d)
        inliers = np.zeros(N, dtype=bool)

        for i in range(N):
            err = self._line_reprojection_error(lines_2d[i], lines_3d[i], R, t)
            inliers[i] = err < self.threshold

        return inliers

    def _line_reprojection_error(self, line_2d, line_3d, R, t):
        """
        Reprojection error for a single line.
        
        Projects both 3D endpoints into the image, forms the projected line,
        then measures the distance from observed 2D endpoints to this line.
        """
        # Project 3D endpoints to camera frame
        P1_cam = R @ line_3d[:3] + t
        P2_cam = R @ line_3d[3:] + t

        # Check if behind camera
        if P1_cam[2] <= 0 or P2_cam[2] <= 0:
            return 1000.0

        # Project to pixel coordinates
        p1_proj = self.K @ P1_cam
        p1_proj = p1_proj[:2] / p1_proj[2]
        p2_proj = self.K @ P2_cam
        p2_proj = p2_proj[:2] / p2_proj[2]

        # Line through projected endpoints (homogeneous)
        l_proj = np.cross(
            np.array([p1_proj[0], p1_proj[1], 1.0]),
            np.array([p2_proj[0], p2_proj[1], 1.0])
        )
        l_norm = np.linalg.norm(l_proj[:2])
        if l_norm < 1e-10:
            return 1000.0
        l_proj = l_proj / l_norm

        # Distance from observed 2D endpoints to projected line
        obs_p1 = line_2d[:2]
        obs_p2 = line_2d[2:]
        d1 = abs(l_proj[0] * obs_p1[0] + l_proj[1] * obs_p1[1] + l_proj[2])
        d2 = abs(l_proj[0] * obs_p2[0] + l_proj[1] * obs_p2[1] + l_proj[2])

        return (d1 + d2) / 2.0


def compute_pose_error(R_est, t_est, R_gt, t_gt):
    """Compute rotation and translation errors between estimated and GT poses."""
    R_diff = R_est @ R_gt.T
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    rot_err = np.degrees(np.arccos(trace))

    t_err = np.linalg.norm(t_est - t_gt)
    t_norm = np.linalg.norm(t_gt)
    trans_err = (t_err / t_norm * 100) if t_norm > 1e-6 else 0
    abs_err_m = np.linalg.norm(t_est - t_gt)  # actual drift in metres

    return rot_err, trans_err, abs_err_m
