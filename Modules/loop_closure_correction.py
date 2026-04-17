"""
Loop Closure Correction for Line-Based Visual SLAM

Implements rigid-body linear drift correction with SLERP rotation
interpolation, inspired by Jang et al. (ACCV 2020).

Algorithm:
    1. When loop closure detected between frame_current and frame_match,
       compute the SE(3) drift error (ΔR, Δt)
    2. Linearly interpolate correction backward through all intermediate
       frames using weight α_i = (i - match) / (current - match)
    3. Use SLERP for rotation (avoids gimbal lock / numerical sensitivity
       of Euler angles) and LERP for translation
    4. Apply corrections to both stored poses and 3D map lines

References:
    - Jang et al., "Pose Correction Algorithm for Relative Frames
      between Keyframes in SLAM", ACCV 2020
      (SLERP for rotation correction, measurement constraint preservation)
    - Grisetti et al., 2010 - rigid body subgraph propagation baseline
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
#  Quaternion utilities (Hamilton convention: [w, x, y, z])
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to unit quaternion [w, x, y, z].
    
    Uses Shepperd's method for numerical stability across all rotations.
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)  # ensure unit quaternion


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)]
    ])


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Spherical Linear Interpolation between two unit quaternions.
    
    Traverses the shortest arc on the unit quaternion sphere, which
    guarantees a valid rotation at every interpolation point.
    
    This avoids the numerical sensitivity of Euler-angle interpolation
    (gimbal lock) and the SO(3) constraint violations of naive matrix
    interpolation, as discussed in Jang et al. Sec. 3.
    
    Args:
        q0: Start quaternion [w, x, y, z] (identity = no correction)
        q1: End quaternion [w, x, y, z] (full correction)
        alpha: Interpolation weight in [0, 1]
        
    Returns:
        Interpolated unit quaternion.
    """
    # Ensure shortest path (flip q1 if dot product is negative)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    # Clamp for numerical safety
    dot = np.clip(dot, -1.0, 1.0)
    
    # If quaternions are very close, fall back to normalised LERP
    # (avoids division by near-zero sin(theta))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    
    # Standard SLERP formula
    theta_0 = np.arccos(dot)          # angle between quaternions
    theta = theta_0 * alpha            # interpolated angle
    sin_theta_0 = np.sin(theta_0)
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


# ---------------------------------------------------------------------------
#  Loop Closure Correction
# ---------------------------------------------------------------------------

class LoopClosureCorrector:
    """
    Applies rigid-body drift correction when a loop closure is detected.
    
    Stores the full trajectory of camera-to-world poses (as used by the
    SLAM pipeline) and, on loop closure, distributes the accumulated
    drift error backward through the trajectory using SLERP (rotation)
    and LERP (translation).
    
    Usage in LineSLAM:
        self.corrector = LoopClosureCorrector()
        
        # After each frame:
        self.corrector.add_pose(frame_idx, R_c2w, t_c2w)
        
        # When loop closure detected:
        self.corrector.apply_correction(
            current_idx, match_idx, self.map
        )
    """
    
    def __init__(self):
        # Store poses as camera-to-world: P_world = R @ P_cam + t
        # Keyed by frame index for random access
        self.poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.frame_order: List[int] = []  # insertion order
        self.corrections_applied: List[dict] = []  # log for analysis
    
    def add_pose(self, frame_idx: int, R_c2w: np.ndarray, t_c2w: np.ndarray):
        """
        Register a frame's camera-to-world pose.
        
        Args:
            frame_idx: Frame index.
            R_c2w: 3x3 rotation matrix, camera-to-world.
            t_c2w: 3x1 translation vector, camera-to-world.
        """
        self.poses[frame_idx] = (R_c2w.copy(), t_c2w.copy().flatten())
        if frame_idx not in self.frame_order:
            self.frame_order.append(frame_idx)
    
    def apply_correction(self,
                         current_idx: int,
                         match_idx: int,
                         line_map,
                         ) -> dict:
        """
        Apply drift correction after a loop closure detection.
        
        Computes the SE(3) error between the current frame's pose and
        the matched frame's pose, then distributes it linearly backward
        through all intermediate frames.
        
        Args:
            current_idx: Index of the current (query) frame.
            match_idx: Index of the matched (database) frame.
            line_map: LineMap instance — its 3D lines will be corrected
                      in-place based on their source frame_id.
                      
        Returns:
            Dictionary with correction diagnostics.
        """
        if current_idx not in self.poses or match_idx not in self.poses:
            print(f"  [LC Correction] WARNING: missing poses for "
                  f"frames {current_idx} or {match_idx}, skipping")
            return {"status": "missing_poses"}
        
        R_current, t_current = self.poses[current_idx]
        R_match, t_match = self.poses[match_idx]
        
        # --- Step 1: Compute drift error in SE(3) ---
        # The match frame's pose is assumed more trustworthy (it was
        # estimated earlier, closer to a known-good region).
        # ΔR @ R_current = R_match  =>  ΔR = R_match @ R_current^T
        # Δt = t_match - ΔR @ t_current
        delta_R = R_match @ R_current.T
        delta_t = t_match - delta_R @ t_current
        
        # Convert ΔR to quaternion for SLERP
        delta_q = rotation_matrix_to_quaternion(delta_R)
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Sanity check: how large is the correction?
        angle_deg = 2.0 * np.arccos(np.clip(abs(delta_q[0]), 0, 1)) * 180 / np.pi
        trans_mag = np.linalg.norm(delta_t)
        print(f"  [LC Correction] Drift: {angle_deg:.2f}° rotation, "
              f"{trans_mag:.4f}m translation")
        
        # --- Step 2: Find frames to correct ---
        # All frames strictly between match_idx and current_idx (inclusive
        # of current_idx, exclusive of match_idx which stays fixed).
        frames_to_correct = [
            idx for idx in self.frame_order
            if match_idx < idx <= current_idx
        ]
        frames_to_correct.sort()
        
        if len(frames_to_correct) == 0:
            print(f"  [LC Correction] No intermediate frames to correct")
            return {"status": "no_frames"}
        
        span = current_idx - match_idx
        
        # --- Step 3: Interpolate and apply corrections ---
        corrected_count = 0
        for idx in frames_to_correct:
            # α = 0 at match (no correction), α = 1 at current (full correction)
            alpha = (idx - match_idx) / span
            
            # SLERP for rotation
            q_correction = slerp(q_identity, delta_q, alpha)
            R_correction = quaternion_to_rotation_matrix(q_correction)
            
            # LERP for translation
            t_correction = alpha * delta_t
            
            # Apply: corrected_pose = correction ∘ original_pose
            R_orig, t_orig = self.poses[idx]
            R_corrected = R_correction @ R_orig
            t_corrected = R_correction @ t_orig + t_correction
            
            # Update stored pose
            self.poses[idx] = (R_corrected, t_corrected)
            corrected_count += 1
        
        # --- Step 4: Correct map lines ---
        lines_corrected = self._correct_map_lines(
            line_map, match_idx, current_idx, 
            delta_q, q_identity, delta_t
        )
        
        result = {
            "status": "applied",
            "match_idx": match_idx,
            "current_idx": current_idx,
            "drift_rotation_deg": angle_deg,
            "drift_translation_m": trans_mag,
            "frames_corrected": corrected_count,
            "lines_corrected": lines_corrected,
        }
        self.corrections_applied.append(result)
        
        print(f"  [LC Correction] Corrected {corrected_count} poses, "
              f"{lines_corrected} map lines")
        
        return result
    
    def _correct_map_lines(self,
                           line_map,
                           match_idx: int,
                           current_idx: int,
                           delta_q: np.ndarray,
                           q_identity: np.ndarray,
                           delta_t: np.ndarray) -> int:
        """
        Apply interpolated corrections to all 3D map lines from
        frames in the affected range.
        
        Each line's world-frame endpoints are transformed by the same
        correction that was applied to its source frame's pose.
        
        Returns:
            Number of lines corrected.
        """
        span = current_idx - match_idx
        count = 0
        
        for i in range(len(line_map.lines_3d)):
            fid = line_map.frame_ids[i]
            
            # Only correct lines from the affected frame range
            if not (match_idx < fid <= current_idx):
                continue
            
            alpha = (fid - match_idx) / span
            
            # Same SLERP + LERP as for poses
            q_correction = slerp(q_identity, delta_q, alpha)
            R_correction = quaternion_to_rotation_matrix(q_correction)
            t_correction = alpha * delta_t
            
            # Transform both endpoints
            line = line_map.lines_3d[i]
            P1 = line[:3]
            P2 = line[3:]
            
            P1_corrected = R_correction @ P1 + t_correction
            P2_corrected = R_correction @ P2 + t_correction
            
            line_map.lines_3d[i] = np.concatenate([P1_corrected, P2_corrected])
            count += 1
        
        return count
    
    def get_corrected_poses(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Return all poses (corrected where applicable)."""
        return self.poses.copy()
    
    def get_correction_log(self) -> List[dict]:
        """Return list of all corrections applied (for analysis/plotting)."""
        return self.corrections_applied
