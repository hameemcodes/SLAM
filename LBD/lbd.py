# Line Band Descriptor (LBD) implementation for SLAM project
# Based on paper by Zhang & Koch (2013) - took ages to understand this
# Last modified: 1am (debugging session)
# TODO: optimize this more, it's still kind of slow

import cv2
import numpy as np
from typing import List, Tuple


class LineDescriptor:
    # This class handles line descriptors using the LBD method
    # basically splits lines into bands and computes gradients

    def __init__(self,
                 num_bands: int = 9,
                 band_width: int = 7,
                 min_line_length: float = 30.0):
        # Initialize the descriptor parameters
        # num_bands: how many bands to split line into (9 worked best in my tests)
        # band_width: width of each band in pixels (7 seems good)
        # min_line_length: ignore lines shorter than this

        self.num_bands = num_bands
        self.band_width = band_width
        self.min_line_length = min_line_length
        # print(f"LineDescriptor initialized with {num_bands} bands")  # debug

    def compute_descriptors(self, image: np.ndarray, lines: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        # compute descriptors for all lines in the image
        # returns descriptors array and list of which lines were valid

        # convert to grayscale if needed
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image.copy()

        desc_list = []
        valid_idx = []

        # go through each line and compute its descriptor
        for idx, line in enumerate(lines):
            desc = self._compute_single_descriptor(gray_img, line)
            if desc is not None:
                desc_list.append(desc)
                valid_idx.append(idx)

        # return empty arrays if no valid descriptors
        if len(desc_list) == 0:
            return np.array([]), []

        return np.array(desc_list), valid_idx

    def _compute_single_descriptor(self, gray_image: np.ndarray, line: np.ndarray) -> np.ndarray:
        # Compute descriptor for one line
        # Returns None if line is too short

        x1, y1, x2, y2 = line

        # calculate line length
        lineLen = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # skip short lines (they're not reliable)
        if lineLen < self.min_line_length:
            return None

        # get line direction (normalized)
        dx = (x2 - x1) / lineLen
        dy = (y2 - y1) / lineLen

        # perpendicular direction (used for band sampling)
        perpDx = -dy
        perpDy = dx

        # compute gradients using sobel (ksize=3 works well)
        gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        desc = []

        # sample along the line to create descriptor
        for i in range(self.num_bands):
            # find center of this band
            t = (i + 0.5) / self.num_bands
            cx = x1 + t * (x2 - x1)
            cy = y1 + t * (y2 - y1)

            # get gradient stats for this band
            band_desc = self._compute_band_descriptor(
                gx, gy,
                cx, cy,
                perpDx, perpDy,
                dx, dy
            )

            desc.extend(band_desc)

        return np.array(desc, dtype=np.float32)

    def _compute_band_descriptor(self,
                                 grad_x: np.ndarray,
                                 grad_y: np.ndarray,
                                 center_x: float,
                                 center_y: float,
                                 perp_dx: float,
                                 perp_dy: float,
                                 line_dx: float,
                                 line_dy: float) -> List[float]:
        # compute descriptor for a single band
        # samples gradients perpendicular to the line

        h, w = grad_x.shape

        # collect gradients along and perpendicular to line direction
        grads_parallel = []
        grads_perp = []

        # sample across the band width
        for offset in range(-self.band_width // 2, self.band_width // 2 + 1):
            # get point in band
            px = center_x + offset * perp_dx
            py = center_y + offset * perp_dy

            # make sure point is inside image
            px_int, py_int = int(round(px)), int(round(py))
            if 0 <= px_int < w and 0 <= py_int < h:
                gx = grad_x[py_int, px_int]
                gy = grad_y[py_int, px_int]

                # project gradient onto line direction
                g_par = gx * line_dx + gy * line_dy
                g_perp = gx * perp_dx + gy * perp_dy

                grads_parallel.append(g_par)
                grads_perp.append(g_perp)

        # if no valid samples, return zeros
        if len(grads_parallel) == 0:
            return [0.0, 0.0, 0.0, 0.0]

        # compute mean and std dev (this is the descriptor)
        mean_par = np.mean(grads_parallel)
        std_par = np.std(grads_parallel)
        mean_perp = np.mean(grads_perp)
        std_perp = np.std(grads_perp)

        return [mean_par, std_par, mean_perp, std_perp]


class LineMatcher:
    # Matches lines between two images using their LBD descriptors
    # uses both descriptor similarity and geometric constraints

    def __init__(self,
                 descriptor_distance_threshold: float = 0.3,
                 geometric_distance_threshold: float = 50.0,
                 angle_threshold_deg: float = 15.0):
        # Initialize matcher with thresholds
        # these values worked well after some tuning
        # TODO: might need adjustment for different image types

        self.desc_thresh = descriptor_distance_threshold  # 0.3 seems good
        self.geom_thresh = geometric_distance_threshold  # 50 pixels
        self.angle_thresh = np.deg2rad(angle_threshold_deg)  # convert to radians

    def match_lines(self,
                   lines1: np.ndarray,
                   descriptors1: np.ndarray,
                   lines2: np.ndarray,
                   descriptors2: np.ndarray) -> List[Tuple[int, int]]:
        # Match lines between two frames
        # returns list of (idx1, idx2) tuples

        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []  # can't match if either is empty

        match_list = []

        # normalize descriptors (important for distance calculation!)
        desc1_normed = descriptors1 / (np.linalg.norm(descriptors1, axis=1, keepdims=True) + 1e-6)
        desc2_normed = descriptors2 / (np.linalg.norm(descriptors2, axis=1, keepdims=True) + 1e-6)

        # compute distances between all descriptor pairs
        # this creates a matrix of distances
        dists = np.linalg.norm(desc1_normed[:, None, :] - desc2_normed[None, :, :], axis=2)

        # for each line in first image, find best match in second
        for i in range(len(lines1)):
            # sort to find best and second-best matches
            sorted_idx = np.argsort(dists[i])
            best_idx = sorted_idx[0]
            best_distance = dists[i, best_idx]

            # Lowe's ratio test (from SIFT paper)
            # helps filter out ambiguous matches
            if len(sorted_idx) > 1:
                second_best_dist = dists[i, sorted_idx[1]]
                ratio = best_distance / (second_best_dist + 1e-6)
            else:
                ratio = 0.0

            # check if match is good enough
            if best_distance < self.desc_thresh and ratio < 0.8:  # 0.8 is standard ratio threshold
                # also check geometric constraints
                if self._check_geometric_constraint(lines1[i], lines2[best_idx]):
                    match_list.append((i, best_idx))

        return match_list

    def _check_geometric_constraint(self, line1: np.ndarray, line2: np.ndarray) -> bool:
        # check if two lines are geometrically similar
        # (similar angle and close in space)

        # calculate angles of both lines
        ang1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
        ang2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0])

        # get angle difference (handle wrapping)
        ang_diff = np.abs(ang1 - ang2)
        ang_diff = np.minimum(ang_diff, 2 * np.pi - ang_diff)  # wrap around

        # reject if angles are too different
        if ang_diff > self.angle_thresh:
            return False

        # check distance between line midpoints
        midpoint1 = np.array([(line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2])
        midpoint2 = np.array([(line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2])

        dist = np.linalg.norm(midpoint1 - midpoint2)

        # reject if lines are too far apart
        if dist > self.geom_thresh:
            return False

        return True  # passed all checks


def visualize_matches(image1: np.ndarray,
                     lines1: np.ndarray,
                     image2: np.ndarray,
                     lines2: np.ndarray,
                     matches: List[Tuple[int, int]],
                     max_matches: int = 50) -> np.ndarray:
    # creates a side-by-side visualization of line matches
    # really helpful for debugging!

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # make a combined image (side by side)
    max_h = max(h1, h2)
    combined_img = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)

    # put first image on left
    if len(image1.shape) == 2:
        combined_img[:h1, :w1] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    else:
        combined_img[:h1, :w1] = image1

    # put second image on right
    if len(image2.shape) == 2:
        combined_img[:h2, w1:w1+w2] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    else:
        combined_img[:h2, w1:w1+w2] = image2

    # draw all lines in gray first
    for line in lines1:
        x1, y1, x2, y2 = [int(v) for v in line]
        cv2.line(combined_img, (x1, y1), (x2, y2), (100, 100, 100), 1)

    for line in lines2:
        x1, y1, x2, y2 = [int(v) for v in line]
        cv2.line(combined_img, (x1 + w1, y1), (x2 + w1, y2), (100, 100, 100), 1)

    # draw matches in random colors
    np.random.seed(42)  # for consistent colors
    n_matches = min(len(matches), max_matches)
    if len(matches) > max_matches:
        selected = np.random.choice(len(matches), n_matches, replace=False)
    else:
        selected = range(len(matches))

    for match_idx in selected:
        i, j = matches[match_idx]

        # pick random color for this match
        clr = tuple(np.random.randint(50, 255, 3).tolist())

        # get lines
        line1 = lines1[i]
        line2 = lines2[j]

        # calculate midpoints
        mid1x = int((line1[0] + line1[2]) / 2)
        mid1y = int((line1[1] + line1[3]) / 2)
        mid2x = int((line2[0] + line2[2]) / 2 + w1)
        mid2y = int((line2[1] + line2[3]) / 2)

        # draw matched lines in color (thicker)
        cv2.line(combined_img, (int(line1[0]), int(line1[1])),
                (int(line1[2]), int(line1[3])), clr, 2)
        cv2.line(combined_img, (int(line2[0] + w1), int(line2[1])),
                (int(line2[2] + w1), int(line2[3])), clr, 2)

        # draw connecting line between midpoints
        cv2.line(combined_img, (mid1x, mid1y), (mid2x, mid2y), clr, 1)
        cv2.circle(combined_img, (mid1x, mid1y), 3, clr, -1)
        cv2.circle(combined_img, (mid2x, mid2y), 3, clr, -1)

    # add match count text
    txt = f"Matches: {len(matches)}"
    cv2.putText(combined_img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    return combined_img
