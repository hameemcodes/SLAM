import cv2
import numpy as np
import tensorflow as tf
import glob
import os
import sys
import time

# Add M-LSD and LBD folders to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'M-LSD'))
sys.path.insert(0, os.path.join(BASE_DIR, 'LBD'))

# Import from M-LSD (renamed from utils.py to M-LSD.py to avoid naming conflict)
from MLSD import pred_lines  # Changed from 'utils' to 'M-LSD'
from lbd import LineDescriptor, LineMatcher, visualize_matches

# Import from our utils package
from utils.depth_utils import initialize_depth_model, estimate_depth, visualize_depth_map
from utils.geometry_3d import backproject_lines_to_3d
from utils.viz_3d import initialize_3d_visualization, update_3d_visualization, render_3d_visualization_to_image

# --- 1. SETUP PATHS ---

CALIB_OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_PATH = os.path.join(BASE_DIR, 'tflite_models', 'M-LSD_512_large_fp32.tflite')

# --- 2. INPUT CONFIGURATION ---
# Set INPUT_MODE to 'video' or 'images'
INPUT_MODE = 'images'

# For video mode
VIDEO_PATH = r"C:\Users\hamee\Downloads\test_videos\IMG_8824.mp4"
OUTPUT_VIDEO_NAME = os.path.join(BASE_DIR, 'mlsd_result.mp4')

# For images mode
IMAGE_FOLDER = r"C:\Users\hamee\Downloads\test_images"
OUTPUT_IMAGE_FOLDER = os.path.join(BASE_DIR, 'mlsd_results_images')

# --- 3. PARAMETERS ---
INPUT_SIZE = 512
SCORE_THR = 0.5
DIST_THR = 20.0

# Image processing parameters
MAX_IMAGE_DIMENSION = 1280  # Resize images larger than this (0 = no resize)

# Line matching parameters
ENABLE_LINE_MATCHING = True  # Set to False to disable matching
NUM_BANDS = 7  # Number of bands for LBD descriptor (reduced for speed)
BAND_WIDTH = 5  # Width of each band in pixels (reduced for speed)

# ==================== DEPTH ESTIMATION CONFIGURATION (NEW) ====================
ENABLE_DEPTH_ESTIMATION = True  # Set to False to disable depth and 3D features
DEPTH_MODEL_PATH = os.path.join(BASE_DIR, 'depth_anything_v2_vitb.pth')  # Local model checkpoint
DEPTH_VISUALIZATION = True  # Show depth map visualization window
ENABLE_3D_VISUALIZATION = True  # Re-enabled: depth_anything_v2 module now available
DEPTH_SCALE_FACTOR = 1.0  # Scale factor for depth values (adjust if needed)
# ==============================================================================

def load_calibration():
    """Load camera calibration data from output folder."""
    print(f"[INFO] Looking for calibration in: {CALIB_OUTPUT_DIR}")
    try:
        mtx_path = os.path.join(CALIB_OUTPUT_DIR, 'camera_matrix.txt')
        dist_path = os.path.join(CALIB_OUTPUT_DIR, 'distortion_coefficients.txt')

        if not os.path.exists(mtx_path):
            print(f"[ERROR] MISSING FILE: {mtx_path}")
            return None, None

        mtx = np.loadtxt(mtx_path)
        dist = np.loadtxt(dist_path)
        print("[SUCCESS] Calibration data loaded.")
        print(f"  fx={mtx[0,0]:.2f}, fy={mtx[1,1]:.2f}, cx={mtx[0,2]:.2f}, cy={mtx[1,2]:.2f}")
        return mtx, dist
    except Exception as e:
        print(f"[ERROR] Error loading calibration: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_image_list(folder):
    """Get list of images from folder."""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder, ext)))
    images = sorted(list(set(images)))  # Deduplicate (Windows case-insensitive) and sort
    return images


def resize_if_needed(image):
    """Resize image if it exceeds MAX_IMAGE_DIMENSION."""
    if MAX_IMAGE_DIMENSION <= 0:
        return image, 1.0

    h, w = image.shape[:2]
    max_dim = max(h, w)

    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  [Resized] {w}x{h} -> {new_w}x{new_h} (scale: {scale:.2f})")
        return resized, scale

    return image, 1.0


def selmap_filter_lines(matches, lines1, lines2, threshold_factor=0.7):
    """
    SelMap outlier rejection for line matches.
    Filters matches based on displacement vector consistency.
    """
    if len(matches) < 5:
        return matches, 0

    # Compute line centers
    centers1 = (lines1[:, :2] + lines1[:, 2:]) / 2
    centers2 = (lines2[:, :2] + lines2[:, 2:]) / 2

    # Compute displacement vectors for matches
    vectors = np.array([centers2[m[1]] - centers1[m[0]] for m in matches])
    lengths = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Build histograms and find modes
    len_hist, len_edges = np.histogram(lengths, bins=50)
    ang_hist, ang_edges = np.histogram(angles, bins=36)
    mode_len = (len_edges[np.argmax(len_hist)] + len_edges[np.argmax(len_hist)+1]) / 2
    mode_ang = (ang_edges[np.argmax(ang_hist)] + ang_edges[np.argmax(ang_hist)+1]) / 2

    # Filter matches near modes
    len_threshold = threshold_factor * np.std(lengths)
    ang_threshold = threshold_factor * np.std(angles)

    inliers = []
    for i, m in enumerate(matches):
        if abs(lengths[i] - mode_len) < len_threshold and abs(angles[i] - mode_ang) < ang_threshold:
            inliers.append(m)

    rejected = len(matches) - len(inliers)
    return inliers, rejected


def process_frame(frame, interpreter, input_details, output_details,
                 depth_model=None, camera_matrix=None): #mtx, dist for undistortion (commented out)
    """
    Process a single frame: detect lines and optionally estimate depth + 3D.

    Args:
        frame: Input frame (BGR numpy array)
        interpreter: TFLite interpreter for M-LSD
        input_details, output_details: TFLite model I/O details
        depth_model: Optional depth estimation model
        camera_matrix: Optional 3x3 camera intrinsic matrix for 3D back-projection

    Returns:
        processed_img: Frame with 2D lines drawn
        lines: Detected 2D lines (N, 4)
        depth_map: Depth map (H, W) or None if depth disabled
        lines_3d: 3D lines (M, 6) or empty array if depth disabled/failed
        valid_3d_indices: Indices of lines with valid 3D (list)
    """
    """
    # Undistortion code (commented out - using original image)
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    """

    processed_img = frame.copy()  # Using original image

    # ==================== 2D LINE DETECTION (EXISTING) ====================
    # Detect Lines using M-LSD
    lines = pred_lines(
        processed_img,
        interpreter,
        input_details,
        output_details,
        input_shape=[INPUT_SIZE, INPUT_SIZE],
        score_thr=SCORE_THR,
        dist_thr=DIST_THR
    )

    # Draw 2D lines on image (red)
    for line in lines:
        x_start, y_start, x_end, y_end = [int(val) for val in line]
        cv2.line(processed_img, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
    # ======================================================================

    # ==================== DEPTH ESTIMATION (NEW) ====================
    depth_map = None
    lines_3d = np.array([])
    valid_3d_indices = []

    if ENABLE_DEPTH_ESTIMATION and depth_model is not None and camera_matrix is not None:
        # Estimate depth for the frame
        depth_map = estimate_depth(depth_model, frame)

        # Back-project 2D lines to 3D if we have depth and calibration
        if depth_map is not None and len(lines) > 0:
            lines_3d, valid_3d_indices = backproject_lines_to_3d(
                lines, depth_map, camera_matrix, DEPTH_SCALE_FACTOR
            )
    # ================================================================

    return processed_img, lines, depth_map, lines_3d, valid_3d_indices


def process_video(interpreter, input_details, output_details,
                 depth_model=None, camera_matrix=None):
    """Process video file with line matching and optional depth estimation."""
    print(f"[INFO] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file. Check path.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"[SUCCESS] Video Opened: {width}x{height} at {fps} FPS")

    # Setup Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, fps, (width, height))

    # Initialize line descriptor and matcher
    if ENABLE_LINE_MATCHING:
        line_descriptor = LineDescriptor(num_bands=NUM_BANDS, band_width=BAND_WIDTH)
        line_matcher = LineMatcher()
        print("[INFO] Line matching enabled")

    # ==================== 3D VISUALIZATION (OPENCV-BASED, NO GIL CONFLICT) ====================
    # No matplotlib window initialization needed - we render to image instead
    if ENABLE_3D_VISUALIZATION and ENABLE_DEPTH_ESTIMATION:
        print("[INFO] 3D visualization will be rendered to OpenCV window")
    # =======================================================================================

    prev_frame = None
    prev_lines = None
    prev_descriptors = None
    frame_count = 0

    print("--- STARTING VIDEO LOOP ---")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            frame_count += 1

            # Resize if needed
            frame_resized, scale = resize_if_needed(frame)

            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}...")

            # ==================== PROCESS FRAME WITH DEPTH (MODIFIED) ====================
            processed_frame, lines, depth_map, lines_3d, valid_3d_indices = process_frame(
                frame_resized, interpreter, input_details, output_details,
                depth_model, camera_matrix
            )
            # =============================================================================

            # ==================== LINE MATCHING (EXISTING) ====================
            matched_indices = []  # Track which lines were matched (for 3D viz)
            if ENABLE_LINE_MATCHING and prev_frame is not None and len(lines) > 0:
                # Compute descriptors for current frame
                descriptors, valid_indices = line_descriptor.compute_descriptors(frame_resized, lines)

                if len(descriptors) > 0 and prev_descriptors is not None and len(prev_descriptors) > 0:
                    # Get valid lines
                    valid_lines = lines[valid_indices]

                    # Match lines
                    matches = line_matcher.match_lines(
                        prev_lines, prev_descriptors,
                        valid_lines, descriptors
                    )

                    # Apply SelMap filtering
                    matches, rejected = selmap_filter_lines(matches, prev_lines, valid_lines)

                    if frame_count % 30 == 0:
                        if rejected > 0:
                            print(f"  Found {len(matches)} matches ({rejected} outliers removed)")
                        else:
                            print(f"  Found {len(matches)} line matches")

                    # Draw matches on processed frame (green overrides red)
                    for match_i, match_j in matches:
                        line = valid_lines[match_j]
                        x1, y1, x2, y2 = [int(v) for v in line]
                        cv2.line(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Track which line index in original lines array was matched
                        original_idx = valid_indices[match_j]
                        matched_indices.append(original_idx)

                # Store current frame data for next iteration
                prev_lines = lines[valid_indices] if len(valid_indices) > 0 else lines
                prev_descriptors = descriptors
            else:
                # First frame or no matching - compute descriptors for next frame
                if ENABLE_LINE_MATCHING and len(lines) > 0:
                    descriptors, valid_indices = line_descriptor.compute_descriptors(frame_resized, lines)
                    prev_lines = lines[valid_indices] if len(valid_indices) > 0 else lines
                    prev_descriptors = descriptors

            prev_frame = frame_resized.copy()
            # ==================================================================

            # ==================== DEPTH & 3D VISUALIZATION (NEW) ====================
            if DEPTH_VISUALIZATION and depth_map is not None:
                depth_colored = visualize_depth_map(depth_map)
                if depth_colored is not None:
                    cv2.imshow('Depth Map', depth_colored)

            if ENABLE_3D_VISUALIZATION and ENABLE_DEPTH_ESTIMATION and len(lines_3d) > 0:
                # Map matched 2D line indices to 3D line indices
                matched_3d_indices = []
                for matched_2d_idx in matched_indices:
                    if matched_2d_idx in valid_3d_indices:
                        matched_3d_indices.append(valid_3d_indices.index(matched_2d_idx))

                # Render 3D visualization to image and display in OpenCV window
                viz_3d_image = render_3d_visualization_to_image(lines_3d, matched_3d_indices)
                if viz_3d_image is not None:
                    cv2.imshow('3D Visualization', viz_3d_image)

                if frame_count % 30 == 0:
                    print(f"  3D lines: {len(lines_3d)}/{len(lines)} (matched: {len(matched_3d_indices)})")
            # =========================================================================

            # Resize if needed for video writer
            if processed_frame.shape[0] != height or processed_frame.shape[1] != width:
                processed_frame = cv2.resize(processed_frame, (width, height))

            # Save & Show
            out.write(processed_frame)
            cv2.imshow('M-LSD Output', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit.")
                break

    except Exception as e:
        print(f"Runtime Error: {e}")
        import traceback
        traceback.print_exc()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Close matplotlib if open
    if fig_3d is not None:
        plt.close(fig_3d)

    print(f"[SUCCESS] Processed {frame_count} frames. Output saved to: {OUTPUT_VIDEO_NAME}")


def process_images(interpreter, input_details, output_details,
                  depth_model=None, camera_matrix=None):
    """Process folder of images with line matching and optional depth estimation."""
    images = get_image_list(IMAGE_FOLDER)

    if not images:
        print(f"[ERROR] No images found in {IMAGE_FOLDER}")
        return

    print(f"[INFO] Found {len(images)} images")

    # Create output folder
    if not os.path.exists(OUTPUT_IMAGE_FOLDER):
        os.makedirs(OUTPUT_IMAGE_FOLDER)

    # Initialize line descriptor and matcher
    if ENABLE_LINE_MATCHING:
        line_descriptor = LineDescriptor(num_bands=NUM_BANDS, band_width=BAND_WIDTH)
        line_matcher = LineMatcher()
        print("[INFO] Line matching enabled")

    # ==================== 3D VISUALIZATION (OPENCV-BASED, NO GIL CONFLICT) ====================
    # No matplotlib window initialization needed - we render to image instead
    if ENABLE_3D_VISUALIZATION and ENABLE_DEPTH_ESTIMATION:
        print("[INFO] 3D visualization will be rendered to OpenCV window")
    # =======================================================================================

    prev_frame = None
    prev_lines = None
    prev_descriptors = None

    print("--- STARTING IMAGE PROCESSING ---")
    print("Press 'n' for next image, 'm' to show matches with previous image, 'q' to quit")

    for idx, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARNING] Could not read: {img_path}")
            continue

        print(f"Processing ({idx+1}/{len(images)}): {os.path.basename(img_path)}")

        # Resize if needed
        frame_resized, scale = resize_if_needed(frame)

        # ==================== PROCESS FRAME WITH DEPTH (MODIFIED) ====================
        processed_frame, lines, depth_map, lines_3d, valid_3d_indices = process_frame(
            frame_resized, interpreter, input_details, output_details,
            depth_model, camera_matrix
        )
        # =============================================================================

        print(f"  Detected {len(lines)} lines")

        # Compute descriptors
        descriptors = None
        valid_indices = []
        if ENABLE_LINE_MATCHING and len(lines) > 0:
            descriptors, valid_indices = line_descriptor.compute_descriptors(frame_resized, lines)
            print(f"  Computed {len(descriptors)} descriptors")

        # ==================== LINE MATCHING (EXISTING) ====================
        matched_indices = []  # Track which lines were matched (for 3D viz)
        matches = []
        if ENABLE_LINE_MATCHING and prev_frame is not None and descriptors is not None and prev_descriptors is not None:
            if len(descriptors) > 0 and len(prev_descriptors) > 0:
                valid_lines = lines[valid_indices]

                matches = line_matcher.match_lines(
                    prev_lines, prev_descriptors,
                    valid_lines, descriptors
                )

                # Apply SelMap filtering
                matches, rejected = selmap_filter_lines(matches, prev_lines, valid_lines)

                if rejected > 0:
                    print(f"  Found {len(matches)} matches ({rejected} outliers removed)")
                else:
                    print(f"  Found {len(matches)} line matches with previous image")

                # Draw matched lines on processed frame in green
                for match_i, match_j in matches:
                    line = valid_lines[match_j]
                    x1, y1, x2, y2 = [int(v) for v in line]
                    cv2.line(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Track which line index in original lines array was matched
                    original_idx = valid_indices[match_j]
                    matched_indices.append(original_idx)
        # ==================================================================

        # ==================== DEPTH & 3D VISUALIZATION (NEW) ====================
        if DEPTH_VISUALIZATION and depth_map is not None:
            depth_colored = visualize_depth_map(depth_map)
            if depth_colored is not None:
                cv2.imshow('Depth Map', depth_colored)

        if ENABLE_3D_VISUALIZATION and ENABLE_DEPTH_ESTIMATION and len(lines_3d) > 0:
            # Map matched 2D line indices to 3D line indices
            matched_3d_indices = []
            for matched_2d_idx in matched_indices:
                if matched_2d_idx in valid_3d_indices:
                    matched_3d_indices.append(valid_3d_indices.index(matched_2d_idx))

            # Render 3D visualization to image and display in OpenCV window
            viz_3d_image = render_3d_visualization_to_image(lines_3d, matched_3d_indices)
            if viz_3d_image is not None:
                cv2.imshow('3D Visualization', viz_3d_image)
            print(f"  3D lines: {len(lines_3d)}/{len(lines)} (matched: {len(matched_3d_indices)})")
        # =========================================================================

        # Save output image
        output_filename = f"mlsd_{os.path.basename(img_path)}"
        output_path = os.path.join(OUTPUT_IMAGE_FOLDER, output_filename)
        cv2.imwrite(output_path, processed_frame)
        print(f"  Saved to: {output_filename}")

        # Display
        cv2.imshow('M-LSD Output', processed_frame)

        # Wait for key press
        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                print("User quit.")
                cv2.destroyAllWindows()
                return
            elif key == ord('n'):
                break
            elif key == ord('m') and len(matches) > 0 and prev_frame is not None:
                # Show side-by-side match visualization
                print("  Showing match visualization...")
                valid_lines = lines[valid_indices]
                match_vis = visualize_matches(
                    prev_frame, prev_lines,
                    frame_resized, valid_lines,
                    matches, max_matches=50
                )

                # Save match visualization
                match_filename = f"matches_{idx-1}_{idx}_{os.path.basename(img_path)}"
                match_path = os.path.join(OUTPUT_IMAGE_FOLDER, match_filename)
                cv2.imwrite(match_path, match_vis)

                cv2.imshow('Line Matches', match_vis)
                print(f"  Match visualization saved to: {match_filename}")
                print("  Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow('Line Matches')
            elif key == ord('m'):
                print("  No matches available (first image or no matches found)")

        # Store current frame data for next iteration
        prev_frame = frame_resized.copy()
        if descriptors is not None and len(descriptors) > 0:
            prev_lines = lines[valid_indices]
            prev_descriptors = descriptors
        else:
            prev_lines = lines
            prev_descriptors = None

    cv2.destroyAllWindows()

    # Close matplotlib if open
    if fig_3d is not None:
        plt.close(fig_3d)

    print(f"[SUCCESS] Finished. Results saved to: {OUTPUT_IMAGE_FOLDER}")


def main():
    print("--- STARTING MAIN ---")
    print(f"[INFO] Mode: {INPUT_MODE.upper()}")

    # ==================== 1. LOAD CALIBRATION (FIXED) ====================
    mtx, dist = None, None
    if ENABLE_DEPTH_ESTIMATION:
        mtx, dist = load_calibration()
        if mtx is None:
            print("[WARNING] Calibration not loaded. 3D features will be disabled.")
            print("[INFO] Continuing with 2D line detection only...")
            # Don't exit - allow 2D features to work
        else:
            print("[INFO] Calibration loaded successfully. 3D features enabled.")
    else:
        print("[INFO] Depth estimation disabled. Running 2D line detection only.")
    # ======================================================================

    # ==================== 2. LOAD M-LSD MODEL (EXISTING) ====================
    print(f"[INFO] Looking for M-LSD model at: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] FILE NOT FOUND: {MODEL_PATH}")
        return

    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("[SUCCESS] M-LSD model loaded.")
    except Exception as e:
        print(f"[ERROR] Error loading M-LSD model: {e}")
        return
    # ========================================================================

    # ==================== 3. INITIALIZE DEPTH MODEL (NEW) ====================
    depth_model = None
    if ENABLE_DEPTH_ESTIMATION:
        print("\n[INFO] Initializing depth estimation...")
        print("[INFO] This may take a moment...")
        depth_model = initialize_depth_model(DEPTH_MODEL_PATH)

        if depth_model is None:
            print("[WARNING] Depth model failed to load. Continuing without depth...")
            mtx = None  # Disable 3D features if depth fails
    # =========================================================================

    # ==================== 4. PROCESS BASED ON MODE (MODIFIED) ====================
    if INPUT_MODE == 'video':
        process_video(interpreter, input_details, output_details,
                     depth_model, mtx)
    elif INPUT_MODE == 'images':
        process_images(interpreter, input_details, output_details,
                      depth_model, mtx)
    else:
        print(f"[ERROR] Unknown INPUT_MODE: {INPUT_MODE}. Use 'video' or 'images'.")
    # =============================================================================


if __name__ == "__main__":
    print("[INFO] Script Launched!")
    main()
    print("[INFO] Script Completed.")