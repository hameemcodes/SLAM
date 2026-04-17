"""
Depth Edge Filter for Line-Based SLAM
======================================
Rejects pixels near depth discontinuities BEFORE back-projection.

This prevents M-LSD from detecting false "lines" across occlusion boundaries.

Usage:
    Add this filter to your back-projection pipeline before creating 3D lines.
"""
import numpy as np
import cv2
from scipy.ndimage import median_filter, uniform_filter


def detect_depth_edges(depth_map, method='gradient', threshold=0.15, window_size=5):
    """
    Detect pixels near depth discontinuities.
    
    Args:
        depth_map: HxW depth map in meters
        method: 'gradient' (fast) or 'variance' (robust) or 'median' (balanced)
        threshold: Relative depth change to consider an edge
                   - gradient: 0.10-0.20 (10-20% change)
                   - variance: 0.01-0.05 (1-5% variance ratio)
                   - median: 0.15-0.25 (15-25% deviation from median)
        window_size: Neighborhood size (3, 5, or 7)
    
    Returns:
        edge_mask: Boolean array, True = edge pixel (unreliable)
    """
    h, w = depth_map.shape
    
    # Handle invalid depth
    valid_mask = (depth_map > 0) & ~np.isnan(depth_map)
    
    if method == 'gradient':
        return _detect_edges_gradient(depth_map, valid_mask, threshold)
    
    elif method == 'variance':
        return _detect_edges_variance(depth_map, valid_mask, threshold, window_size)
    
    elif method == 'median':
        return _detect_edges_median(depth_map, valid_mask, threshold, window_size)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _detect_edges_gradient(depth_map, valid_mask, threshold):
    """
    Detect edges using Sobel gradient (FAST, recommended for real-time).
    
    Pros: Very fast, good for real-time applications
    Cons: Sensitive to noise
    """
    # Compute gradients
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize by depth (relative change)
    depth_safe = np.where(depth_map > 0.01, depth_map, 1.0)
    relative_grad = grad_mag / depth_safe
    
    # Mark edges
    edge_mask = (relative_grad > threshold) | ~valid_mask
    
    return edge_mask


def _detect_edges_variance(depth_map, valid_mask, threshold, window_size):
    """
    Detect edges using local variance (ROBUST, best quality).
    
    Pros: Robust to noise, high quality
    Cons: Slower
    """
    # Compute local mean
    depth_filled = np.where(valid_mask, depth_map, 0)
    local_mean = uniform_filter(depth_filled, size=window_size, mode='constant')
    
    # Compute local variance
    depth_sq = depth_filled ** 2
    local_mean_sq = uniform_filter(depth_sq, size=window_size, mode='constant')
    local_var = local_mean_sq - local_mean ** 2
    local_var = np.maximum(local_var, 0)  # Numerical stability
    
    # Normalize by mean depth squared
    mean_safe = np.where(local_mean > 0.01, local_mean, 1.0)
    relative_var = local_var / (mean_safe ** 2)
    
    # Mark high-variance regions as edges
    edge_mask = (relative_var > threshold) | ~valid_mask
    
    return edge_mask


def _detect_edges_median(depth_map, valid_mask, threshold, window_size):
    """
    Detect edges using median filter (BALANCED, recommended).
    
    Pros: Good balance of speed and robustness
    Cons: Slightly slower than gradient
    """
    # Compute local median (robust to outliers)
    depth_filled = np.where(valid_mask, depth_map, np.nan)
    local_median = median_filter(depth_filled, size=window_size)
    
    # Compute deviation from median
    deviation = np.abs(depth_map - local_median)
    median_safe = np.where(local_median > 0.01, local_median, 1.0)
    relative_deviation = deviation / median_safe
    
    # Mark large deviations as edges
    edge_mask = (relative_deviation > threshold) | ~valid_mask
    
    return edge_mask


def filter_depth_map(depth_map, method='median', threshold=0.15, window_size=5):
    """
    Create filtered depth map with edge pixels set to invalid (0 or NaN).
    
    Args:
        depth_map: HxW depth map
        method: 'gradient', 'variance', or 'median'
        threshold: Edge detection sensitivity
        window_size: Neighborhood size
    
    Returns:
        filtered_depth: Depth map with edge pixels removed
        edge_mask: Boolean mask of rejected pixels
    """
    edge_mask = detect_depth_edges(depth_map, method, threshold, window_size)
    
    # Set edge pixels to invalid
    filtered_depth = depth_map.copy()
    filtered_depth[edge_mask] = 0  # or np.nan
    
    n_removed = edge_mask.sum()
    n_total = depth_map.size
    pct_removed = 100 * n_removed / n_total
    
    print(f"    Depth edge filter: removed {n_removed}/{n_total} pixels ({pct_removed:.1f}%)")
    
    return filtered_depth, edge_mask


def backproject_lines_with_edge_filter(lines_2d, depth_map, K, rgb_shape,
                                       edge_filter_method='median',
                                       edge_threshold=0.15,
                                       depth_consistency_threshold=0.20):
    """
    Back-project 2D lines to 3D with TWO-STAGE filtering:
    
    Stage 1: Depth edge filter (pixel-level, your idea)
    Stage 2: Depth consistency filter (line-level, my previous approach)
    
    This combines both approaches for maximum robustness.
    
    Args:
        lines_2d: Nx4 array of [x1, y1, x2, y2]
        depth_map: HxW depth map
        K: 3x3 intrinsics
        rgb_shape: (H, W) of RGB image
        edge_filter_method: 'gradient', 'variance', or 'median'
        edge_threshold: Pixel-level edge detection threshold
        depth_consistency_threshold: Line-level consistency threshold
    
    Returns:
        lines_3d: Mx6 array of valid 3D lines
        valid_indices: Which input lines survived both filters
    """
    from map_3d import backproject_lines  # Your existing function
    
    # Stage 1: Filter depth map to remove edge pixels
    filtered_depth, edge_mask = filter_depth_map(
        depth_map, 
        method=edge_filter_method,
        threshold=edge_threshold
    )
    
    # Stage 2: Back-project using filtered depth
    lines_3d, valid_3d = backproject_lines(lines_2d, filtered_depth, K, rgb_shape)
    
    # Stage 3: Depth consistency check on remaining lines
    if len(lines_3d) > 0:
        lines_3d_filtered, depth_valid = filter_lines_by_depth_consistency(
            lines_2d[valid_3d], 
            lines_3d, 
            max_variation=depth_consistency_threshold
        )
        
        # Map back to original indices
        #final_valid = valid_3d.copy()
        #final_valid[valid_3d] = depth_valid
        # CORRECT:
        final_valid_indices = valid_3d[depth_valid]  # Shape: (41,)
        
        return lines_3d_filtered, final_valid_indices
    else:
        return lines_3d, valid_3d


def filter_lines_by_depth_consistency(lines_2d, lines_3d, max_variation=0.20):
    """Line-level filter (from previous approach)."""
    valid_lines_3d = []
    valid_indices = []
    
    for i, line_3d in enumerate(lines_3d):
        Z1, Z2 = line_3d[2], line_3d[5]
        
        if Z1 < 0.01 or Z2 < 0.01:
            continue
        
        depth_ratio = abs(Z2 - Z1) / min(Z1, Z2)
        
        if depth_ratio < max_variation:
            valid_lines_3d.append(line_3d)
            valid_indices.append(i)
    
    if len(valid_lines_3d) == 0:
        return np.array([]).reshape(0, 6), np.array([], dtype=int)
    
    return np.array(valid_lines_3d), np.array(valid_indices, dtype=int)


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_depth_edges(rgb_image, depth_map, edge_mask):
    """Overlay edge mask on RGB image for debugging."""
    import cv2
    
    vis = rgb_image.copy()
    
    # Color edge pixels red
    vis[edge_mask] = [255, 0, 0]
    
    return vis


def compare_filters(depth_map):
    """Compare different edge detection methods."""
    import matplotlib.pyplot as plt
    
    methods = ['gradient', 'variance', 'median']
    thresholds = [0.15, 0.03, 0.20]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original
    axes[0].imshow(depth_map, cmap='jet')
    axes[0].set_title('Original Depth')
    axes[0].axis('off')
    
    # Each method
    for ax, method, thresh in zip(axes[1:], methods, thresholds):
        edge_mask = detect_depth_edges(depth_map, method=method, threshold=thresh)
        
        # Show edges in red
        vis = np.stack([edge_mask.astype(float)] * 3, axis=-1)
        vis[:, :, 1] = 0  # Remove green
        vis[:, :, 2] = 0  # Remove blue
        
        ax.imshow(vis)
        ax.set_title(f'{method.capitalize()}\n({edge_mask.sum()} pixels)')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
# In your SLAM pipeline (e.g., runSLAM2.py):

from depth_edge_filter import backproject_lines_with_edge_filter

# Replace your old back-projection call:
# lines_3d_cam, valid_3d = backproject_lines(lines_2d, frame.depth, frame.K, frame.rgb.shape[:2])

# With the new filtered version:
lines_3d_cam, valid_3d = backproject_lines_with_edge_filter(
    lines_2d, 
    frame.depth, 
    frame.K, 
    frame.rgb.shape[:2],
    edge_filter_method='median',    # 'gradient' for speed, 'median' for quality
    edge_threshold=0.15,            # Lower = more aggressive filtering
    depth_consistency_threshold=0.20  # Same as before
)

lines_2d = lines_2d[valid_3d]
descriptors = descriptors[valid_3d]
"""
