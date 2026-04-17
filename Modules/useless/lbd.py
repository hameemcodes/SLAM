"""
Line Band Descriptor (LBD) Implementation
For matching lines between images in SLAM applications

Based on the LBD algorithm:
- Divides lines into bands perpendicular to line direction
- Computes gradient-based descriptors for each band
- Enables efficient line matching across frames
"""

import cv2
import numpy as np
from typing import List, Tuple


class LineDescriptor:
    """Line Band Descriptor (LBD) for line feature matching"""

    def __init__(self,
                 num_bands: int = 9,
                 band_width: int = 7,
                 min_line_length: float = 30.0):
        """
        Initialize Line Band Descriptor

        Args:
            num_bands: Number of bands to divide the line into
            band_width: Width of each band in pixels
            min_line_length: Minimum line length to compute descriptor
        """
        self.num_bands = num_bands
        self.band_width = band_width
        self.min_line_length = min_line_length

    def compute_descriptors(self, image: np.ndarray, lines: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Compute LBD descriptors for all lines in an image

        Args:
            image: Input image (grayscale or BGR)
            lines: Array of lines with shape (N, 4) where each line is [x1, y1, x2, y2]

        Returns:
            descriptors: Array of descriptors with shape (M, descriptor_size)
            valid_indices: List of indices of lines with valid descriptors
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        descriptors = []
        valid_indices = []

        for idx, line in enumerate(lines):
            descriptor = self._compute_single_descriptor(gray, line)
            if descriptor is not None:
                descriptors.append(descriptor)
                valid_indices.append(idx)

        if len(descriptors) == 0:
            return np.array([]), []

        return np.array(descriptors), valid_indices

    def _compute_single_descriptor(self, gray_image: np.ndarray, line: np.ndarray) -> np.ndarray:
        """
        Compute descriptor for a single line

        Args:
            gray_image: Grayscale image
            line: Line coordinates [x1, y1, x2, y2]

        Returns:
            Descriptor vector or None if line is too short
        """
        x1, y1, x2, y2 = line

        # Calculate line length
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if line_length < self.min_line_length:
            return None

        # Calculate line direction (unit vector)
        dx = (x2 - x1) / line_length
        dy = (y2 - y1) / line_length

        # Perpendicular direction (for band sampling)
        perp_dx = -dy
        perp_dy = dx

        # Compute gradient images
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        descriptor = []

        # Sample points along the line
        for i in range(self.num_bands):
            # Position along the line (center of each band)
            t = (i + 0.5) / self.num_bands
            center_x = x1 + t * (x2 - x1)
            center_y = y1 + t * (y2 - y1)

            # Compute mean and std of gradients in the band
            band_descriptor = self._compute_band_descriptor(
                grad_x, grad_y,
                center_x, center_y,
                perp_dx, perp_dy,
                dx, dy
            )

            descriptor.extend(band_descriptor)

        return np.array(descriptor, dtype=np.float32)

    def _compute_band_descriptor(self,
                                 grad_x: np.ndarray,
                                 grad_y: np.ndarray,
                                 center_x: float,
                                 center_y: float,
                                 perp_dx: float,
                                 perp_dy: float,
                                 line_dx: float,
                                 line_dy: float) -> List[float]:
        """
        Compute descriptor for a single band

        Args:
            grad_x, grad_y: Gradient images
            center_x, center_y: Center point of the band
            perp_dx, perp_dy: Perpendicular direction to the line
            line_dx, line_dy: Line direction

        Returns:
            Band descriptor (mean gradients along and perpendicular to line)
        """
        h, w = grad_x.shape

        # Sample points across the band width
        gradients_parallel = []
        gradients_perpendicular = []

        for offset in range(-self.band_width // 2, self.band_width // 2 + 1):
            # Point in the band
            px = center_x + offset * perp_dx
            py = center_y + offset * perp_dy

            # Check bounds
            px_int, py_int = int(round(px)), int(round(py))
            if 0 <= px_int < w and 0 <= py_int < h:
                gx = grad_x[py_int, px_int]
                gy = grad_y[py_int, px_int]

                # Project gradient onto line direction and perpendicular direction
                grad_parallel = gx * line_dx + gy * line_dy
                grad_perp = gx * perp_dx + gy * perp_dy

                gradients_parallel.append(grad_parallel)
                gradients_perpendicular.append(grad_perp)

        # Compute statistics
        if len(gradients_parallel) == 0:
            return [0.0, 0.0, 0.0, 0.0]

        mean_parallel = np.mean(gradients_parallel)
        std_parallel = np.std(gradients_parallel)
        mean_perp = np.mean(gradients_perpendicular)
        std_perp = np.std(gradients_perpendicular)

        return [mean_parallel, std_parallel, mean_perp, std_perp]


class LineMatcher:
    """Match lines between images using LBD descriptors"""

    def __init__(self,
                 descriptor_distance_threshold: float = 0.3,
                 geometric_distance_threshold: float = 50.0,
                 angle_threshold_deg: float = 15.0):
        """
        Initialize Line Matcher

        Args:
            descriptor_distance_threshold: Max normalized distance for descriptor matching
            geometric_distance_threshold: Max pixel distance between line endpoints
            angle_threshold_deg: Max angle difference in degrees
        """
        self.desc_threshold = descriptor_distance_threshold
        self.geom_threshold = geometric_distance_threshold
        self.angle_threshold = np.deg2rad(angle_threshold_deg)

    def match_lines(self,
                   lines1: np.ndarray,
                   descriptors1: np.ndarray,
                   lines2: np.ndarray,
                   descriptors2: np.ndarray) -> List[Tuple[int, int]]:
        """
        Match lines between two sets using descriptors and geometric constraints

        Args:
            lines1: Lines from first image (N1, 4)
            descriptors1: Descriptors from first image (N1, desc_size)
            lines2: Lines from second image (N2, 4)
            descriptors2: Descriptors from second image (N2, desc_size)

        Returns:
            List of matches as (index_in_lines1, index_in_lines2) tuples
        """
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []

        matches = []

        # Normalize descriptors
        desc1_norm = descriptors1 / (np.linalg.norm(descriptors1, axis=1, keepdims=True) + 1e-6)
        desc2_norm = descriptors2 / (np.linalg.norm(descriptors2, axis=1, keepdims=True) + 1e-6)

        # Compute all pairwise distances
        distances = np.linalg.norm(desc1_norm[:, None, :] - desc2_norm[None, :, :], axis=2)

        # For each line in image 1, find best match in image 2
        for i in range(len(lines1)):
            # Find best and second best matches
            sorted_indices = np.argsort(distances[i])
            best_idx = sorted_indices[0]
            best_dist = distances[i, best_idx]

            # Ratio test (Lowe's ratio test)
            if len(sorted_indices) > 1:
                second_best_dist = distances[i, sorted_indices[1]]
                ratio = best_dist / (second_best_dist + 1e-6)
            else:
                ratio = 0.0

            # Check descriptor distance and ratio
            if best_dist < self.desc_threshold and ratio < 0.8:
                # Apply geometric constraints
                if self._check_geometric_constraint(lines1[i], lines2[best_idx]):
                    matches.append((i, best_idx))

        return matches

    def _check_geometric_constraint(self, line1: np.ndarray, line2: np.ndarray) -> bool:
        """
        Check if two lines are geometrically consistent

        Args:
            line1, line2: Lines as [x1, y1, x2, y2]

        Returns:
            True if lines are geometrically similar
        """
        # Check angle similarity
        angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
        angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0])

        angle_diff = np.abs(angle1 - angle2)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

        if angle_diff > self.angle_threshold:
            return False

        # Check endpoint distance (midpoint distance)
        mid1 = np.array([(line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2])
        mid2 = np.array([(line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2])

        distance = np.linalg.norm(mid1 - mid2)

        if distance > self.geom_threshold:
            return False

        return True


def visualize_matches(image1: np.ndarray,
                     lines1: np.ndarray,
                     image2: np.ndarray,
                     lines2: np.ndarray,
                     matches: List[Tuple[int, int]],
                     max_matches: int = 50) -> np.ndarray:
    """
    Visualize line matches between two images

    Args:
        image1, image2: Input images
        lines1, lines2: Line coordinates
        matches: List of (index1, index2) matches
        max_matches: Maximum number of matches to display

    Returns:
        Combined image showing matches
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Create side-by-side image
    h_max = max(h1, h2)
    combined = np.zeros((h_max, w1 + w2, 3), dtype=np.uint8)

    # Place images
    if len(image1.shape) == 2:
        combined[:h1, :w1] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    else:
        combined[:h1, :w1] = image1

    if len(image2.shape) == 2:
        combined[:h2, w1:w1+w2] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    else:
        combined[:h2, w1:w1+w2] = image2

    # Draw lines
    for line in lines1:
        x1, y1, x2, y2 = [int(v) for v in line]
        cv2.line(combined, (x1, y1), (x2, y2), (100, 100, 100), 1)

    for line in lines2:
        x1, y1, x2, y2 = [int(v) for v in line]
        cv2.line(combined, (x1 + w1, y1), (x2 + w1, y2), (100, 100, 100), 1)

    # Draw matches
    np.random.seed(42)
    num_matches = min(len(matches), max_matches)
    selected_matches = np.random.choice(len(matches), num_matches, replace=False) if len(matches) > max_matches else range(len(matches))

    for match_idx in selected_matches:
        i, j = matches[match_idx]

        # Random color for each match
        color = tuple(np.random.randint(50, 255, 3).tolist())

        # Get line midpoints
        line1 = lines1[i]
        line2 = lines2[j]

        mid1_x = int((line1[0] + line1[2]) / 2)
        mid1_y = int((line1[1] + line1[3]) / 2)
        mid2_x = int((line2[0] + line2[2]) / 2 + w1)
        mid2_y = int((line2[1] + line2[3]) / 2)

        # Draw matched lines in color
        cv2.line(combined, (int(line1[0]), int(line1[1])),
                (int(line1[2]), int(line1[3])), color, 2)
        cv2.line(combined, (int(line2[0] + w1), int(line2[1])),
                (int(line2[2] + w1), int(line2[3])), color, 2)

        # Draw connection
        cv2.line(combined, (mid1_x, mid1_y), (mid2_x, mid2_y), color, 1)
        cv2.circle(combined, (mid1_x, mid1_y), 3, color, -1)
        cv2.circle(combined, (mid2_x, mid2_y), 3, color, -1)

    # Add text
    text = f"Matches: {len(matches)}"
    cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    return combined
