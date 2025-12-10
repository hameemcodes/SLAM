# 3D visualization for SLAM line features
# Uses matplotlib to show 3D lines in real-time
# TODO: this is kinda slow, maybe look into using Open3D instead?

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')  # non-blocking backend (took forever to figure this out)


def initialize_3d_visualization():
    # sets up the matplotlib 3D plot window
    # returns figure and axis objects

    plt.ion()  # interactive mode - makes it non-blocking

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # label the axes
    ax.set_xlabel('X (right)', fontsize=10)
    ax.set_ylabel('Y (down)', fontsize=10)
    ax.set_zlabel('Z (forward)', fontsize=10)
    ax.set_title('3D Lines Visualization', fontsize=12)

    # set viewing angle (20, 45 looks good)
    ax.view_init(elev=20, azim=45)

    # show the window without blocking
    plt.show(block=False)
    plt.pause(0.001)  # small pause needed for window to appear

    return fig, ax


def update_3d_visualization(ax, lines_3d, matched_indices=None):
    # updates the 3D plot with new lines
    # matched lines are green, unmatched are red

    # clear previous stuff
    ax.cla()

    # reset labels (cla clears them)
    ax.set_xlabel('X (right)', fontsize=10)
    ax.set_ylabel('Y (down)', fontsize=10)
    ax.set_zlabel('Z (forward)', fontsize=10)
    ax.set_title('3D Lines', fontsize=12)

    # if no lines, show message
    if len(lines_3d) == 0:
        ax.text(0, 0, 0, 'No lines to show', fontsize=12, ha='center')
        plt.draw()
        plt.pause(0.001)
        return

    # convert to set for faster lookups
    matched_set = set(matched_indices) if matched_indices is not None else set()

    # draw each 3D line
    for idx, line in enumerate(lines_3d):
        X1, Y1, Z1, X2, Y2, Z2 = line

        # color based on whether line was matched
        # green = matched, red = not matched
        if idx in matched_set:
            clr = 'green'
            alph = 0.8
            lw = 2.0
        else:
            clr = 'red'
            alph = 0.4
            lw = 1.5

        # plot the line segment
        ax.plot([X1, X2], [Y1, Y2], [Z1, Z2],
                color=clr, linewidth=lw, alpha=alph)

    # set axis limits based on the data
    # use median and std to avoid outliers messing up the view
    if len(lines_3d) > 0:
        # get all endpoint coordinates
        pts = lines_3d.reshape(-1, 3)

        # compute stats
        x_med, y_med, z_med = np.median(pts, axis=0)
        x_std, y_std, z_std = np.std(pts, axis=0)

        # set limits to +/- 3 std devs
        scale_factor = 3  # this value worked well
        xMin, xMax = x_med - scale_factor * x_std, x_med + scale_factor * x_std
        yMin, yMax = y_med - scale_factor * y_std, y_med + scale_factor * y_std
        zMin, zMax = z_med - scale_factor * z_std, z_med + scale_factor * z_std

        # make sure range isn't too small
        minRange = 1.0
        if xMax - xMin < minRange:
            xMid = (xMin + xMax) / 2
            xMin, xMax = xMid - minRange / 2, xMid + minRange / 2
        if yMax - yMin < minRange:
            yMid = (yMin + yMax) / 2
            yMin, yMax = yMid - minRange / 2, yMid + minRange / 2
        if zMax - zMin < minRange:
            zMid = (zMin + zMax) / 2
            zMin, zMax = zMid - minRange / 2, zMid + minRange / 2

        ax.set_xlim([xMin, xMax])
        ax.set_ylim([yMin, yMax])
        ax.set_zlim([zMin, zMax])

    # show camera origin as blue dot
    ax.scatter([0], [0], [0], color='blue', s=100, marker='o', label='Camera')

    # add legend
    from matplotlib.lines import Line2D
    legend_stuff = [
        Line2D([0], [0], color='green', linewidth=2, label=f'Matched ({len(matched_set)})'),
        Line2D([0], [0], color='red', linewidth=2, alpha=0.4, label=f'Unmatched ({len(lines_3d) - len(matched_set)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Camera')
    ]
    ax.legend(handles=legend_stuff, loc='upper right')

    # update the display
    plt.draw()
    plt.pause(0.001)


def render_3d_visualization_to_image(lines_3d, matched_indices=None, figsize=(10, 8)):
    # renders 3D viz to an image instead of showing in window
    # useful for displaying with opencv (avoids threading issues)

    import io

    # return None if no lines
    if len(lines_3d) == 0:
        return None

    # create new figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # set up labels
    ax.set_xlabel('X (right)', fontsize=10)
    ax.set_ylabel('Y (down)', fontsize=10)
    ax.set_zlabel('Z (forward)', fontsize=10)
    ax.set_title('3D Lines', fontsize=12)

    # set viewing angle
    ax.view_init(elev=20, azim=45)

    # matched lines set
    matched_set = set(matched_indices) if matched_indices is not None else set()

    # draw lines (same as update function)
    for idx, line in enumerate(lines_3d):
        X1, Y1, Z1, X2, Y2, Z2 = line
        clr = 'green' if idx in matched_set else 'red'
        alph = 0.8 if idx in matched_set else 0.4
        lw = 2.0 if idx in matched_set else 1.5
        ax.plot([X1, X2], [Y1, Y2], [Z1, Z2],
                color=clr, linewidth=lw, alpha=alph)

    # set limits based on data
    if len(lines_3d) > 0:
        pts = lines_3d.reshape(-1, 3)
        x_med, y_med, z_med = np.median(pts, axis=0)
        x_std, y_std, z_std = np.std(pts, axis=0)

        sc = 3  # scale factor
        xMin, xMax = x_med - sc * x_std, x_med + sc * x_std
        yMin, yMax = y_med - sc * y_std, y_med + sc * y_std
        zMin, zMax = z_med - sc * z_std, z_med + sc * z_std

        # ensure min range
        minRng = 1.0
        if xMax - xMin < minRng:
            xMid = (xMin + xMax) / 2
            xMin, xMax = xMid - minRng / 2, xMid + minRng / 2
        if yMax - yMin < minRng:
            yMid = (yMin + yMax) / 2
            yMin, yMax = yMid - minRng / 2, yMid + minRng / 2
        if zMax - zMin < minRng:
            zMid = (zMin + zMax) / 2
            zMin, zMax = zMid - minRng / 2, zMid + minRng / 2

        ax.set_xlim([xMin, xMax])
        ax.set_ylim([yMin, yMax])
        ax.set_zlim([zMin, zMax])

    # camera origin
    ax.scatter([0], [0], [0], color='blue', s=100, marker='o', label='Camera')

    # legend
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], color='green', linewidth=2, label=f'Matched ({len(matched_set)})'),
        Line2D([0], [0], color='red', linewidth=2, alpha=0.4, label=f'Unmatched ({len(lines_3d) - len(matched_set)})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Camera')
    ]
    ax.legend(handles=leg, loc='upper right')

    # render to buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
    buffer.seek(0)

    # convert to opencv image
    import cv2
    img_arr = np.frombuffer(buffer.read(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # close figure to free memory (important!)
    plt.close(fig)

    return img
