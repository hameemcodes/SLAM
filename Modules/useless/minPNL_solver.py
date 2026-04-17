"""
MinPnL: Minimal PnL solver with Gram-Schmidt constraint compression
Based on Zhou et al. 2020 RAL paper

Key innovation: Compress 2N constraints → 3 quadratic equations for numerical stability

FIXES applied:
1. Correct CGR (Cayley-Gibbs-Rodrigues) expansion coefficients in _build_constraints
2. Reprojection-based disambiguation (test both s and -s solutions)
3. Improved numerical stability in Gram-Schmidt and quadratic solver
"""
import numpy as np
from typing import Tuple
from scipy.optimize import least_squares


class MinPnLSolver:
    def __init__(self, K: np.ndarray, ransac_iters: int = 300, threshold: float = 10.0):
        self.K = K.astype(np.float64)
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.ransac_iters = ransac_iters
        self.threshold = threshold
    
    def estimate_pose(self, lines_2d: np.ndarray, lines_3d: np.ndarray, 
                     debug: bool = False) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate camera pose from 2D-3D line correspondences
        
        Args:
            lines_2d: Nx4 [x1,y1,x2,y2] in pixels
            lines_3d: Nx6 [X1,Y1,Z1,X2,Y2,Z2] in WORLD frame
            
        Returns:
            success, R (world-to-cam), t (world-to-cam), inliers
            Convention: P_cam = R @ P_world + t
        """
        N = len(lines_2d)
        if N < 3:
            return False, np.eye(3), np.zeros(3), np.zeros(N, dtype=bool)
        
        s_init = self._get_opencv_warmstart(lines_2d, lines_3d) #using OpenCV PnP RANSAC as warm start for better convergence

        # RANSAC
        best_R, best_t, best_inliers = None, None, np.zeros(N, dtype=bool)
        best_count = 0
        
        for _ in range(self.ransac_iters):
            idx = np.random.choice(N, 3, replace=False)
            success, R, t = self._solve_minimal(lines_2d[idx], lines_3d[idx])
            
            if not success:
                continue
            
            # Count inliers
            inliers = self._get_inliers(lines_2d, lines_3d, R, t)
            count = inliers.sum()
            
            if count > best_count:
                best_count = count
                best_R, best_t, best_inliers = R, t, inliers
                
                if count > 0.8 * N:
                    break
        
        # Refine with all inliers
        if best_count >= 3:
            success, R, t = self._solve_minimal(lines_2d[best_inliers], lines_3d[best_inliers])
            if success:
                best_R, best_t = R, t
                best_inliers = self._get_inliers(lines_2d, lines_3d, R, t)
        
        if debug:
            print(f"[MinPnL] Inliers: {best_count}/{N}")
        
        return best_count >= 3, best_R, best_t, best_inliers
    
    def _get_opencv_warmstart(self, lines_2d, lines_3d):
        """
        Get initial guess for CGR parameters using OpenCV's PnP RANSAC.
        
        After the coordinate convention fix, lines_3d are in ARKit world frame.
        OpenCV solvePnP returns R, t mapping world -> OpenCV camera, which is
        the same convention as our MinPnL solver output. So we pass world 
        points directly — no flipping needed.
        """
        import cv2

        # Each line has two point correspondences (both endpoints)
        points_3d = []
        points_2d = []

        for i in range(len(lines_3d)):
            points_3d.append(lines_3d[i, :3])   # endpoint 1 in world frame
            points_3d.append(lines_3d[i, 3:])   # endpoint 2 in world frame
            points_2d.append([lines_2d[i, 0], lines_2d[i, 1]])  # endpoint 1 pixels
            points_2d.append([lines_2d[i, 2], lines_2d[i, 3]])  # endpoint 2 pixels
            
        points_3d = np.array(points_3d, dtype=np.float64)
        points_2d = np.array(points_2d, dtype=np.float64)
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, self.K, 
                distCoeffs=None,
                iterationsCount=100,
                reprojectionError=8.0,
                confidence=0.99
            )

            if not success:
                return None
            
            # Convert rotation vector to matrix
            R_opencv, _ = cv2.Rodrigues(rvec)
            
            # Convert R to CGR s vector using axis-angle
            # CGR: s = tan(theta/2) * axis
            cos_theta = np.clip((np.trace(R_opencv) - 1) / 2, -1, 1)
            theta = np.arccos(cos_theta)
            
            if theta < 1e-6:
                return np.zeros(3)
            
            # Extract axis from skew-symmetric part of R
            axis = np.array([
                R_opencv[2, 1] - R_opencv[1, 2],
                R_opencv[0, 2] - R_opencv[2, 0],
                R_opencv[1, 0] - R_opencv[0, 1]
            ]) / (2 * np.sin(theta))
            
            s_init = np.tan(theta / 2) * axis
            return s_init
            
        except Exception as e:
            return None

    def _solve_minimal(self, lines_2d: np.ndarray, lines_3d: np.ndarray, s_init = None) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Core MinPnL algorithm with Gram-Schmidt compression"""
        
        # Step 1: Normalize 2D lines
        lines_2d_norm = np.column_stack([
            (lines_2d[:, 0] - self.cx) / self.fx,
            (lines_2d[:, 1] - self.cy) / self.fy,
            (lines_2d[:, 2] - self.cx) / self.fx,
            (lines_2d[:, 3] - self.cy) / self.fy
        ])
        
        # Step 2: Build constraint matrices A and B
        A, B = self._build_constraints(lines_2d_norm, lines_3d)
        
        # Step 3: Gram-Schmidt compression to 3 equations
        C = self._compress_with_gram_schmidt(A, B)
        
        if C is None:
            return False, np.eye(3), np.zeros(3)
        
        # Step 4: Solve 3 quadratic equations numerically
        s_sol = self._solve_quadratic_system(C, s_init=s_init)
        
        if s_sol is None:
            return False, np.eye(3), np.zeros(3)
        
        # Step 5: Recover R and t, testing both s and -s (CGR antipodal ambiguity)
        # The CGR parameterisation has an inherent ambiguity: both s and -s can
        # satisfy the quadratic constraints. We test both and pick the one with
        # lower reprojection error.
        candidates = []
        for s_candidate in [s_sol, -s_sol]:
            R_cand = self._cgr_to_rotation(s_candidate)
            t_cand = self._recover_translation(s_candidate, A, B)
            
            # Compute total reprojection error for this candidate
            total_err = 0.0
            for k in range(len(lines_3d)):
                err = self._line_error(lines_2d[k], lines_3d[k], R_cand, t_cand)
                if err < 500:
                    total_err += err
                else:
                    total_err += 500  # Penalty for behind-camera points
            
            candidates.append((R_cand, t_cand, total_err))
        
        # Pick candidate with lower reprojection error
        best = min(candidates, key=lambda x: x[2])
        R, t = best[0], best[1]
        
        return True, R, t
    
    def _build_constraints(self, lines_2d_norm, lines_3d):
        """
        Build Ar + Bτ = 0 from line correspondences
        
        CORRECT CGR expansion of l^T * R_bar * P where:
        R_bar = (1-|s|²)I + 2[s]× + 2ss^T
        
        Monomial vector: r = [s1², s2², s3², s1s2, s1s3, s2s3, s1, s2, s3, 1]
        """
        N = len(lines_2d_norm)
        A = np.zeros((2*N, 10))
        B = np.zeros((2*N, 3))
        
        for i in range(N):
            # Line normal in normalised image coordinates
            p1 = np.array([lines_2d_norm[i, 0], lines_2d_norm[i, 1], 1.0])
            p2 = np.array([lines_2d_norm[i, 2], lines_2d_norm[i, 3], 1.0])
            l = np.cross(p1, p2)
            l = l / (np.linalg.norm(l[:2]) + 1e-10)
            
            # 3D endpoints
            for j, P in enumerate([lines_3d[i, :3], lines_3d[i, 3:]]):
                row = 2*i + j
                X, Y, Z = P
                lx, ly, lz = l[0], l[1], l[2]
                
                # CORRECT coefficients from expanding l^T * R_bar * P
                # R_bar = (1-|s|²)I + 2[s]× + 2ss^T
                
                # s1²: lx*X - ly*Y - lz*Z
                A[row, 0] = lx*X - ly*Y - lz*Z
                
                # s2²: -lx*X + ly*Y - lz*Z
                A[row, 1] = -lx*X + ly*Y - lz*Z
                
                # s3²: -lx*X - ly*Y + lz*Z
                A[row, 2] = -lx*X - ly*Y + lz*Z
                
                # s1*s2: 2(lx*Y + ly*X)
                A[row, 3] = 2*(lx*Y + ly*X)
                
                # s1*s3: 2(lx*Z + lz*X)
                A[row, 4] = 2*(lx*Z + lz*X)
                
                # s2*s3: 2(ly*Z + lz*Y)
                A[row, 5] = 2*(ly*Z + lz*Y)
                
                # s1: 2(-ly*Z + lz*Y)
                A[row, 6] = 2*(-ly*Z + lz*Y)
                
                # s2: 2(lx*Z - lz*X)
                A[row, 7] = 2*(lx*Z - lz*X)
                
                # s3: 2(-lx*Y + ly*X)
                A[row, 8] = 2*(-lx*Y + ly*X)
                
                # constant: lx*X + ly*Y + lz*Z
                A[row, 9] = lx*X + ly*Y + lz*Z
                
                B[row, :] = l
        
        return A, B
    
    def _compress_with_gram_schmidt(self, A, B):
        """Compress 2N equations to 3 using Gram-Schmidt (key innovation!)"""
        try:
            BtB = B.T @ B
            cond = np.linalg.cond(BtB)
            if cond > 1e10:
                return None
            BtB_inv = np.linalg.inv(BtB)
            K = A - B @ BtB_inv @ B.T @ A
        except:
            return None
        
        K9 = K[:, :9]
        
        indices = self._gram_schmidt_select3(K9)
        
        if len(indices) < 3:
            return None
        
        K3 = K[:, indices]
        mask = np.ones(10, dtype=bool)
        mask[indices] = False
        K7 = K[:, mask]
        
        try:
            KtK = K3.T @ K3
            cond = np.linalg.cond(KtK)
            if cond > 1e10:
                return None
            C7 = np.linalg.inv(KtK) @ K3.T @ K7
        except:
            return None
        
        C = np.zeros((3, 10))
        C[:, indices] = np.eye(3)
        C[:, mask] = C7
        
        return C
    
    def _gram_schmidt_select3(self, K9):
        """Select 3 most linearly independent columns using iterative orthogonalisation"""
        n_cols = K9.shape[1]
        selected = []
        basis = []
        
        for step in range(3):
            best_norm = -1
            best_idx = -1
            best_proj = None
            
            for col_idx in range(n_cols):
                if col_idx in selected:
                    continue
                    
                v = K9[:, col_idx].copy()
                
                for b in basis:
                    v = v - (np.dot(b, v) / (np.dot(b, b) + 1e-12)) * b
                
                norm = np.linalg.norm(v)
                if norm > best_norm:
                    best_norm = norm
                    best_idx = col_idx
                    best_proj = v.copy()
            
            if best_norm < 1e-10:
                break
                
            selected.append(best_idx)
            basis.append(best_proj / (best_norm + 1e-12))
        
        return selected
    
    def _solve_quadratic_system(self, C, s_init=None):
        """Solve 3 quadratic equations for s=[s1, s2, s3]"""
        
        def equations(s):
            s1, s2, s3 = s
            r = np.array([s1**2, s2**2, s3**2, s1*s2, s1*s3, s2*s3, s1, s2, s3, 1.0])
            return C @ r
        
        def jacobian(s):
            s1, s2, s3 = s
            dr_ds = np.array([
                [2*s1, 0, 0, s2, s3, 0, 1, 0, 0, 0],
                [0, 2*s2, 0, s1, 0, s3, 0, 1, 0, 0],
                [0, 0, 2*s3, 0, s1, s2, 0, 0, 1, 0]
            ]).T  # 10x3
            return C @ dr_ds  # 3x3
        
        # Build guess list: warm-start first (if available), then fallbacks
        guesses = []
        
        if s_init is not None:
            guesses.append(s_init.tolist())
            # Small perturbations around the warm start
            guesses.append((s_init + [0.05, 0.0, 0.0]).tolist())
            guesses.append((s_init + [0.0, 0.05, 0.0]).tolist())
            guesses.append((s_init + [0.0, 0.0, 0.05]).tolist())
        
        # Fallback guesses covering small to medium rotations
        guesses += [
            [0.0, 0.0, 0.0],          # identity
            [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
            [0.0, 0.0, 0.1], [0.0, 0.0, -0.1],
            # Medium rotations (~30-45 deg)
            [0.3, 0.0, 0.0], [-0.3, 0.0, 0.0],
            [0.0, 0.3, 0.0], [0.0, -0.3, 0.0],
            [0.0, 0.0, 0.3], [0.0, 0.0, -0.3],
            # Larger rotations (~60 deg)
            [0.6, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 0.6],
            # Diagonal combinations
            [0.3, 0.3, 0.0], [0.3, 0.0, 0.3], [0.0, 0.3, 0.3],
            [-0.3, 0.3, 0.0], [0.3, -0.3, 0.0],
        ]
        
        best_sol = None
        best_residual = np.inf
        
        for guess in guesses:
            try:
                result = least_squares(equations, guess, jac=jacobian, 
                                      method='lm', max_nfev=100)
                residual = np.linalg.norm(result.fun)
                
                if residual < best_residual:
                    best_residual = residual
                    best_sol = result.x
            except:
                continue
        
        return best_sol if best_residual < 0.1 else None
    
    def _cgr_to_rotation(self, s):
        """Convert CGR parameters to rotation matrix"""
        s1, s2, s3 = s
        s_skew = np.array([[0, -s3, s2], [s3, 0, -s1], [-s2, s1, 0]])
        s_norm_sq = s1**2 + s2**2 + s3**2
        
        R_bar = (1 - s_norm_sq) * np.eye(3) + 2 * s_skew + 2 * np.outer(s, s)
        R = R_bar / (1 + s_norm_sq)
        
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = -R
        
        return R
    
    def _recover_translation(self, s, A, B):
        """Recover t from s using τ = (1 + s^T s) t"""
        s1, s2, s3 = s
        r = np.array([s1**2, s2**2, s3**2, s1*s2, s1*s3, s2*s3, s1, s2, s3, 1.0])
        
        tau = -np.linalg.inv(B.T @ B) @ B.T @ A @ r
        t = tau / (1 + s1**2 + s2**2 + s3**2)
        
        return t
    
    def _get_inliers(self, lines_2d, lines_3d, R, t):
        """Get inlier mask based on reprojection error"""
        errors = np.array([self._line_error(lines_2d[i], lines_3d[i], R, t) 
                          for i in range(len(lines_2d))])
        return errors < self.threshold
    
    def _line_error(self, line_2d, line_3d, R, t):
        """Compute reprojection error for one line"""
        P1_cam = R @ line_3d[:3] + t
        P2_cam = R @ line_3d[3:] + t
        
        if P1_cam[2] <= 0 or P2_cam[2] <= 0:
            return 1000.0
        
        p1 = self.K @ P1_cam; p1 = p1[:2] / p1[2]
        p2 = self.K @ P2_cam; p2 = p2[:2] / p2[2]
        
        l_proj = np.cross([p1[0], p1[1], 1], [p2[0], p2[1], 1])
        l_proj = l_proj / (np.linalg.norm(l_proj[:2]) + 1e-10)
        
        obs_p1, obs_p2 = line_2d[:2], line_2d[2:]
        d1 = abs(l_proj @ [obs_p1[0], obs_p1[1], 1])
        d2 = abs(l_proj @ [obs_p2[0], obs_p2[1], 1])
        
        return (d1 + d2) / 2


def compute_pose_error(R_est, t_est, R_gt, t_gt):
    """Compute rotation and translation errors"""
    R_diff = R_est @ R_gt.T
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    rot_err = np.degrees(np.arccos(trace))
    
    t_err = np.linalg.norm(t_est - t_gt)
    t_norm = np.linalg.norm(t_gt)
    trans_err = (t_err / t_norm * 100) if t_norm > 1e-6 else 0
    
    return rot_err, trans_err