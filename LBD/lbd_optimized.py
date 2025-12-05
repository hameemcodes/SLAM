"""
Optimized Line Band Descriptor (LBD) Implementation
Performance improvements:
- Cached gradient computation
- Vectorized operations
- Reduced memory allocations
- Optional line filtering
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class LineDescriptorOptimized:
    """Optimized Line Band Descriptor with gradient caching"""

    def __init__(self,
                 num_bands: int = 7,  # Reduced from 9
                 band_width: int = 5,  # Reduced from 7
                 min_line_length: float = 30.0,
                 max_lines: Optional[int] = None):  # Limit number of lines
        """
        Initialize Optimized Line Band Descriptor

        Args:
            num_bands: Number of bands (lower = faster)
            band_width: Width of each band (lower = faster)
            min_line_length: Minimum line length
            max_lines: Maximum number of lines to process (None = no limit)
        """
        self.num_bands = num_bands
        self.band_width = band_width
        self.min_line_length = min_line_length
        self.max_lines = max_lines

        # Cache for gradients
        self._cached_grad_x = None
        self._cached_grad_y = None
        self._cache_hash = None

    def compute_descriptors(self,
                          image: np.ndarray,
                          lines: np.ndarray,
                          use_cache: bool = False) -> Tuple[np.ndarray, List[int]]:
        """
        Compute LBD descriptors with optional gradient caching

        Args:
            image: Input image
            lines: Array of lines (N, 4)
            use_cache: Reuse gradient computation if image hasn't changed

        Returns:
            descriptors: Array of descriptors
            valid_indices: Indices of lines with valid descriptors
        """
        if len(lines) == 0:
            return np.array([]), []

        # Limit number of lines if specified
        if self.max_lines is not None and len(lines) > self.max_lines:
            # Keep longest lines
            lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
            indices = np.argsort(lengths)[::-1][:self.max_lines]
            lines = lines[indices]
        else:
            indices = np.arange(len(lines))

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Compute or use cached gradients
        cache_hash = hash(gray.tobytes()) if use_cache else None

        if use_cache and cache_hash == self._cache_hash and self._cached_grad_x is not None:
            grad_x = self._cached_grad_x
            grad_y = self._cached_grad_y
        else:
            # Compute gradients (this is expensive!)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            if use_cache:
                self._cached_grad_x = grad_x
                self._cached_grad_y = grad_y
                self._cache_hash = cache_hash

        # Pre-compute line properties
        line_vectors = lines[:, 2:] - lines[:, :2]
        line_lengths = np.sqrt(np.sum(line_vectors**2, axis=1))

        # Filter short lines
        valid_mask = line_lengths >= self.min_line_length
        valid_line_indices = np.where(valid_mask)[0]

        if len(valid_line_indices) == 0:
            return np.array([]), []

        valid_lines = lines[valid_mask]
        valid_lengths = line_lengths[valid_mask]
        valid_vectors = line_vectors[valid_mask]

        # Normalize line directions
        line_dirs = valid_vectors / valid_lengths[:, np.newaxis]
        perp_dirs = np.stack([-line_dirs[:, 1], line_dirs[:, 0]], axis=1)

        # Compute descriptors for all valid lines
        descriptors = []

        h, w = gray.shape

        for i, (line, length, line_dir, perp_dir) in enumerate(
                zip(valid_lines, valid_lengths, line_dirs, perp_dirs)):

            descriptor = []

            # Sample along the line
            for band_idx in range(self.num_bands):
                t = (band_idx + 0.5) / self.num_bands
                center_x = line[0] + t * (line[2] - line[0])
                center_y = line[1] + t * (line[3] - line[1])

                # Sample across the band
                band_grads_parallel = []
                band_grads_perp = []

                for offset in range(-self.band_width // 2, self.band_width // 2 + 1):
                    px = int(round(center_x + offset * perp_dir[0]))
                    py = int(round(center_y + offset * perp_dir[1]))

                    if 0 <= px < w and 0 <= py < h:
                        gx = grad_x[py, px]
                        gy = grad_y[py, px]

                        grad_parallel = gx * line_dir[0] + gy * line_dir[1]
                        grad_perp = gx * perp_dir[0] + gy * perp_dir[1]

                        band_grads_parallel.append(grad_parallel)
                        band_grads_perp.append(grad_perp)

                # Compute statistics
                if len(band_grads_parallel) > 0:
                    descriptor.extend([
                        np.mean(band_grads_parallel),
                        np.std(band_grads_parallel),
                        np.mean(band_grads_perp),
                        np.std(band_grads_perp)
                    ])
                else:
                    descriptor.extend([0.0, 0.0, 0.0, 0.0])

            descriptors.append(descriptor)

        # Map back to original indices
        original_indices = [indices[i] for i in valid_line_indices]

        return np.array(descriptors, dtype=np.float32), original_indices


class LineMatcherOptimized:
    """Optimized line matcher with spatial filtering"""

    def __init__(self,
                 descriptor_distance_threshold: float = 0.35,  # Slightly relaxed
                 geometric_distance_threshold: float = 100.0,  # More relaxed
                 angle_threshold_deg: float = 20.0,  # More relaxed
                 use_spatial_filter: bool = True):
        """
        Initialize optimized line matcher

        Args:
            descriptor_distance_threshold: Max descriptor distance
            geometric_distance_threshold: Max pixel distance
            angle_threshold_deg: Max angle difference
            use_spatial_filter: Pre-filter candidates by spatial proximity
        """
        self.desc_threshold = descriptor_distance_threshold
        self.geom_threshold = geometric_distance_threshold
        self.angle_threshold = np.deg2rad(angle_threshold_deg)
        self.use_spatial_filter = use_spatial_filter

    def match_lines(self,
                   lines1: np.ndarray,
                   descriptors1: np.ndarray,
                   lines2: np.ndarray,
                   descriptors2: np.ndarray) -> List[Tuple[int, int]]:
        """
        Match lines with spatial filtering for speed

        Args:
            lines1, descriptors1: Lines and descriptors from image 1
            lines2, descriptors2: Lines and descriptors from image 2

        Returns:
            List of (idx1, idx2) matches
        """
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []

        matches = []

        # Normalize descriptors
        desc1_norm = descriptors1 / (np.linalg.norm(descriptors1, axis=1, keepdims=True) + 1e-6)
        desc2_norm = descriptors2 / (np.linalg.norm(descriptors2, axis=1, keepdims=True) + 1e-6)

        # Pre-compute line midpoints and angles
        mid1 = (lines1[:, :2] + lines1[:, 2:]) / 2
        mid2 = (lines2[:, :2] + lines2[:, 2:]) / 2

        angles1 = np.arctan2(lines1[:, 3] - lines1[:, 1], lines1[:, 2] - lines1[:, 0])
        angles2 = np.arctan2(lines2[:, 3] - lines2[:, 1], lines2[:, 2] - lines2[:, 0])

        # Match each line in image 1
        for i in range(len(lines1)):

            # Spatial filtering: only consider nearby lines
            if self.use_spatial_filter:
                distances = np.linalg.norm(mid2 - mid1[i], axis=1)
                candidates = np.where(distances < self.geom_threshold)[0]

                if len(candidates) == 0:
                    continue

                # Angular filtering
                angle_diffs = np.abs(angles2[candidates] - angles1[i])
                angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
                candidates = candidates[angle_diffs < self.angle_threshold]

                if len(candidates) == 0:
                    continue
            else:
                candidates = np.arange(len(lines2))

            # Compute descriptor distances only for candidates
            desc_distances = np.linalg.norm(desc1_norm[i] - desc2_norm[candidates], axis=1)

            # Find best match
            if len(desc_distances) > 0:
                best_idx = np.argmin(desc_distances)
                best_dist = desc_distances[best_idx]

                # Ratio test
                if len(desc_distances) > 1:
                    sorted_dists = np.sort(desc_distances)
                    ratio = sorted_dists[0] / (sorted_dists[1] + 1e-6)
                else:
                    ratio = 0.0

                if best_dist < self.desc_threshold and ratio < 0.8:
                    actual_idx = candidates[best_idx]
                    matches.append((i, actual_idx))

        return matches


# Simple aliases for easy switching
def create_fast_descriptor():
    """Create descriptor optimized for speed"""
    return LineDescriptorOptimized(
        num_bands=5,
        band_width=5,
        max_lines=100
    )


def create_balanced_descriptor():
    """Create descriptor with balanced speed/accuracy"""
    return LineDescriptorOptimized(
        num_bands=7,
        band_width=5,
        max_lines=150
    )


def create_accurate_descriptor():
    """Create descriptor optimized for accuracy"""
    return LineDescriptorOptimized(
        num_bands=9,
        band_width=7,
        max_lines=None
    )
