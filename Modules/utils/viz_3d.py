"""
3D Visualization utilities for displaying 3D lines using matplotlib.

This module provides functions to create and update a live 3D visualization
of detected lines in camera coordinate space.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')  # Non-blocking backend for real-time updates


def initialize_3d_visualization():
    """
    Initialize matplotlib 3D visualization window.

    Creates a non-blocking 3D plot that can be updated in real-time
    while other processing continues.

    Returns:
        fig: Matplotlib figure object
        ax: 3D axes object

    Usage:
        fig, ax = initialize_3d_visualization()
        # ... process data ...
        update_3d_visualization(ax, lines_3d)
    """
    plt.ion()  # Enable interactive mode (non-blocking)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set labels and title
    ax.set_xlabel('X (camera right)', fontsize=10)
    ax.set_ylabel('Y (camera down)', fontsize=10)
    ax.set_zlabel('Z (camera forward)', fontsize=10)
    ax.set_title('3D Line Visualization (Camera Frame)', fontsize=12)

    # Set initial view angle for better perspective
    ax.view_init(elev=20, azim=45)

    # Show without blocking
    plt.show(block=False)
    plt.pause(0.001)

    return fig, ax


def update_3d_visualization(ax, lines_3d, matched_indices=None):
    """
    Update 3D visualization with new lines.

    Clears the previous frame and draws the current set of 3D lines.
    Lines are color-coded based on whether they were matched with
    the previous frame.

    Args:
        ax: Matplotlib 3D axes object (from initialize_3d_visualization)
        lines_3d: Array of 3D lines (N, 6) with format [X1, Y1, Z1, X2, Y2, Z2]
        matched_indices: Optional list of indices for lines that were matched
                        Matched lines are drawn in green, unmatched in red

    Returns:
        None

    Note:
        Uses plt.pause() to update display without blocking execution.
        The visualization window remains interactive and can be rotated
        with the mouse.
    """
    # Clear previous lines
    ax.cla()

    # Reset labels (cleared by cla())
    ax.set_xlabel('X (camera right)', fontsize=10)
    ax.set_ylabel('Y (camera down)', fontsize=10)
    ax.set_zlabel('Z (camera forward)', fontsize=10)
    ax.set_title('3D Line Visualization (Camera Frame)', fontsize=12)

    # Handle empty case
    if len(lines_3d) == 0:
        ax.text(0, 0, 0, 'No valid 3D lines', fontsize=12, ha='center')
        plt.draw()
        plt.pause(0.001)
        return

    # Convert matched indices to set for fast lookup
    matched_set = set(matched_indices) if matched_indices is not None else set()

    # Draw each line
    for idx, line in enumerate(lines_3d):
        X1, Y1, Z1, X2, Y2, Z2 = line

        # Color coding:
        # - Green: matched lines (found in both current and previous frame)
        # - Red: unmatched lines (new or not matched)
        color = 'green' if idx in matched_set else 'red'
        alpha = 0.8 if idx in matched_set else 0.4  # Matched lines more opaque
        linewidth = 2.0 if idx in matched_set else 1.5

        # Plot line segment
        ax.plot([X1, X2], [Y1, Y2], [Z1, Z2],
                color=color, linewidth=linewidth, alpha=alpha)

    # Set axis limits based on data statistics
    # This provides better visualization by filtering out extreme outliers
    if len(lines_3d) > 0:
        # Reshape to (N*2, 3) to get all points
        all_points = lines_3d.reshape(-1, 3)

        # Compute statistics for each axis
        x_median, y_median, z_median = np.median(all_points, axis=0)
        x_std, y_std, z_std = np.std(all_points, axis=0)

        # Set limits to 3 standard deviations around median
        # This excludes extreme outliers while preserving most data
        scale = 3  # Number of standard deviations
        x_min, x_max = x_median - scale * x_std, x_median + scale * x_std
        y_min, y_max = y_median - scale * y_std, y_median + scale * y_std
        z_min, z_max = z_median - scale * z_std, z_median + scale * z_std

        # Ensure minimum range for visibility
        min_range = 1.0
        if x_max - x_min < min_range:
            x_mid = (x_min + x_max) / 2
            x_min, x_max = x_mid - min_range / 2, x_mid + min_range / 2
        if y_max - y_min < min_range:
            y_mid = (y_min + y_max) / 2
            y_min, y_max = y_mid - min_range / 2, y_mid + min_range / 2
        if z_max - z_min < min_range:
            z_mid = (z_min + z_max) / 2
            z_min, z_max = z_mid - min_range / 2, z_mid + min_range / 2

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

    # Add camera origin as reference point
    ax.scatter([0], [0], [0], color='blue', s=100, marker='o', label='Camera Origin')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, label=f'Matched ({len(matched_set)})'),
        Line2D([0], [0], color='red', linewidth=2, alpha=0.4, label=f'Unmatched ({len(lines_3d) - len(matched_set)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Camera')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Update display (non-blocking)
    plt.draw()
    plt.pause(0.001)  # Small pause to allow display update


def render_3d_visualization_to_image(lines_3d, matched_indices=None, figsize=(10, 8)):
    """
    Render 3D visualization to a numpy array (for OpenCV display).

    Creates a matplotlib figure, renders the 3D lines, and converts to BGR image.
    This avoids GIL conflicts with OpenCV's cv2.waitKey().

    Args:
        lines_3d: Array of 3D lines (N, 6) with format [X1, Y1, Z1, X2, Y2, Z2]
        matched_indices: Optional list of indices for lines that were matched
        figsize: Figure size (width, height) in inches

    Returns:
        image: BGR numpy array (H, W, 3) suitable for cv2.imshow()
               Returns None if lines_3d is empty
    """
    import io

    # Handle empty case
    if len(lines_3d) == 0:
        return None

    # Create figure and axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Set labels and title
    ax.set_xlabel('X (camera right)', fontsize=10)
    ax.set_ylabel('Y (camera down)', fontsize=10)
    ax.set_zlabel('Z (camera forward)', fontsize=10)
    ax.set_title('3D Line Visualization (Camera Frame)', fontsize=12)

    # Set view angle
    ax.view_init(elev=20, azim=45)

    # Convert matched indices to set for fast lookup
    matched_set = set(matched_indices) if matched_indices is not None else set()

    # Draw each line
    for idx, line in enumerate(lines_3d):
        X1, Y1, Z1, X2, Y2, Z2 = line
        color = 'green' if idx in matched_set else 'red'
        alpha = 0.8 if idx in matched_set else 0.4
        linewidth = 2.0 if idx in matched_set else 1.5
        ax.plot([X1, X2], [Y1, Y2], [Z1, Z2],
                color=color, linewidth=linewidth, alpha=alpha)

    # Set axis limits based on data statistics
    if len(lines_3d) > 0:
        all_points = lines_3d.reshape(-1, 3)
        x_median, y_median, z_median = np.median(all_points, axis=0)
        x_std, y_std, z_std = np.std(all_points, axis=0)

        scale = 3
        x_min, x_max = x_median - scale * x_std, x_median + scale * x_std
        y_min, y_max = y_median - scale * y_std, y_median + scale * y_std
        z_min, z_max = z_median - scale * z_std, z_median + scale * z_std

        # Ensure minimum range
        min_range = 1.0
        if x_max - x_min < min_range:
            x_mid = (x_min + x_max) / 2
            x_min, x_max = x_mid - min_range / 2, x_mid + min_range / 2
        if y_max - y_min < min_range:
            y_mid = (y_min + y_max) / 2
            y_min, y_max = y_mid - min_range / 2, y_mid + min_range / 2
        if z_max - z_min < min_range:
            z_mid = (z_min + z_max) / 2
            z_min, z_max = z_mid - min_range / 2, z_mid + min_range / 2

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

    # Add camera origin
    ax.scatter([0], [0], [0], color='blue', s=100, marker='o', label='Camera Origin')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, label=f'Matched ({len(matched_set)})'),
        Line2D([0], [0], color='red', linewidth=2, alpha=0.4, label=f'Unmatched ({len(lines_3d) - len(matched_set)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Camera')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Render figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)

    # Convert to numpy array
    import cv2
    image_array = np.frombuffer(buf.read(), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Close figure to free memory
    plt.close(fig)

    return image
