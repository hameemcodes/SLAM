import cv2
import numpy as np
import tensorflow as tf
import glob
import os
import sys
import time

# TODO: Clean up these path additions later - bit messy but works for now
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'M-LSD'))
sys.path.insert(0, os.path.join(BASE_DIR, 'LBD'))

# Import from M-LSD (had to rename from utils.py - was conflicting with our utils)
from MLSD import pred_lines  
from lbd import LineDescriptor, LineMatcher, visualize_matches

# Our custom utility modules
from utils.depth_utils import initialize_depth_model, estimate_depth, visualize_depth_map
from utils.geometry_3d import backproject_lines_to_3d
from utils.viz_3d import initialize_3d_visualization, update_3d_visualization, render_3d_visualization_to_image

# Configuration stuff - probably should move to a config file eventually
CALIB_OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_PATH = os.path.join(BASE_DIR, 'tflite_models', 'M-LSD_512_large_fp32.tflite')

# Input setup - change this based on what you're testing
INPUT_MODE = 'images'  # 'video' or 'images'

# Video settings (if using video mode)
VIDEO_PATH = r"C:\Users\hamee\Downloads\test_videos\IMG_8824.mp4"
OUTPUT_VIDEO_NAME = os.path.join(BASE_DIR, 'mlsd_result.mp4')

# Image settings (if using image mode)
IMAGE_FOLDER = r"C:\Users\hamee\Downloads\test_images"  
OUTPUT_IMAGE_FOLDER = os.path.join(BASE_DIR, 'mlsd_results_images')

# M-LSD parameters - these work pretty well, don't change unless you know what you're doing
INPUT_SIZE = 512
SCORE_THR = 0.5      # Lower = more lines detected (but noisier)
DIST_THR = 20.0      # Distance threshold for line merging

# Image processing
MAX_IMAGE_DIMENSION = 1280  # Resize big images to prevent memory issues

# Line matching config
ENABLE_LINE_MATCHING = True   
NUM_BANDS = 7        # LBD descriptor bands - reduced from default for speed
BAND_WIDTH = 5       # Band width in pixels

# Depth estimation settings - this is the cool new stuff!
ENABLE_DEPTH_ESTIMATION = True
DEPTH_MODEL_PATH = os.path.join(BASE_DIR, 'depth_anything_v2_vitb.pth')  
DEPTH_VISUALIZATION = True    # Show the depth map in a separate window
ENABLE_3D_VISUALIZATION = True
DEPTH_SCALE_FACTOR = 1.0     # Might need to tune this


def load_calibration():
    """Load camera calibration - need this for 3D stuff to work."""
    print(f"[INFO] Searching for calibration files in: {CALIB_OUTPUT_DIR}")
    
    try:
        # Look for the calibration files
        camera_matrix_file = os.path.join(CALIB_OUTPUT_DIR, 'camera_matrix.txt')
        distortion_file = os.path.join(CALIB_OUTPUT_DIR, 'distortion_coefficients.txt')

        if not os.path.exists(camera_matrix_file):
            print(f"[ERROR] Can't find camera matrix file: {camera_matrix_file}")
            return None, None

        # Load the matrices
        camera_matrix = np.loadtxt(camera_matrix_file)
        distortion_coeffs = np.loadtxt(distortion_file)
        
        print("[SUCCESS] Got the calibration data!")
        print(f"  Camera params - fx: {camera_matrix[0,0]:.2f}, fy: {camera_matrix[1,1]:.2f}")
        print(f"                  cx: {camera_matrix[0,2]:.2f}, cy: {camera_matrix[1,2]:.2f}")
        
        return camera_matrix, distortion_coeffs
        
    except Exception as error:
        print(f"[ERROR] Something went wrong loading calibration: {error}")
        import traceback
        traceback.print_exc()
        return None, None


def get_image_list(folder_path):
    """Get all images from a folder - handles different extensions."""
    # Support common image formats
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    all_images = []
    for extension in image_extensions:
        found_images = glob.glob(os.path.join(folder_path, extension))
        all_images.extend(found_images)
    
    # Remove duplicates (Windows file system is case-insensitive) and sort
    unique_images = sorted(list(set(all_images)))
    return unique_images


def resize_if_needed(img):
    """Resize image if it's too big - helps with memory and processing speed."""
    if MAX_IMAGE_DIMENSION <= 0:
        return img, 1.0  # No resizing

    height, width = img.shape[:2]
    max_dimension = max(height, width)

    if max_dimension > MAX_IMAGE_DIMENSION:
        # Calculate new size
        scale_factor = MAX_IMAGE_DIMENSION / max_dimension
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Resize using area interpolation (good for downsampling)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"  [Resized] {width}x{height} -> {new_width}x{new_height} (scale: {scale_factor:.2f})")
        
        return resized_img, scale_factor

    return img, 1.0


def selmap_filter_lines(line_matches, lines_1, lines_2, threshold_factor=0.7):
    """
    SelMap outlier rejection - filters out bad line matches.
    Based on displacement vector consistency (fancy way of saying "do the matches make sense?")
    """
    if len(line_matches) < 5:
        return line_matches, 0  # Not enough matches to filter

    # Calculate line centers (midpoints)
    centers_1 = (lines_1[:, :2] + lines_1[:, 2:]) / 2
    centers_2 = (lines_2[:, :2] + lines_2[:, 2:]) / 2

    # Get displacement vectors for all matches
    displacement_vectors = []
    for match in line_matches:
        vector = centers_2[match[1]] - centers_1[match[0]]
        displacement_vectors.append(vector)
    
    displacement_vectors = np.array(displacement_vectors)
    vector_lengths = np.linalg.norm(displacement_vectors, axis=1)
    vector_angles = np.arctan2(displacement_vectors[:, 1], displacement_vectors[:, 0])

    # Find the most common length and angle (modes)
    length_histogram, length_bins = np.histogram(vector_lengths, bins=50)
    angle_histogram, angle_bins = np.histogram(vector_angles, bins=36)
    
    mode_length = (length_bins[np.argmax(length_histogram)] + 
                   length_bins[np.argmax(length_histogram)+1]) / 2
    mode_angle = (angle_bins[np.argmax(angle_histogram)] + 
                  angle_bins[np.argmax(angle_histogram)+1]) / 2

    # Filter based on how close matches are to the modes
    length_threshold = threshold_factor * np.std(vector_lengths)
    angle_threshold = threshold_factor * np.std(vector_angles)

    good_matches = []
    for i, match in enumerate(line_matches):
        length_diff = abs(vector_lengths[i] - mode_length)
        angle_diff = abs(vector_angles[i] - mode_angle)
        
        if length_diff < length_threshold and angle_diff < angle_threshold:
            good_matches.append(match)

    num_rejected = len(line_matches) - len(good_matches)
    return good_matches, num_rejected


def process_frame(input_frame, mlsd_interpreter, input_details, output_details,
                 depth_estimation_model=None, camera_intrinsics=None):
    """
    Main frame processing function - does line detection and optionally depth + 3D stuff.
    
    Returns a bunch of stuff:
    - processed_image: frame with lines drawn on it
    - detected_lines: the 2D lines we found
    - depth_map: depth estimation result (or None)
    - lines_3d: 3D lines (or empty array)
    - valid_3d_indices: which lines have valid 3D data
    """
    
    # Start with a copy of the input
    processed_image = input_frame.copy()

    # Detect lines using M-LSD
    detected_lines = pred_lines(
        processed_image,
        mlsd_interpreter,
        input_details,
        output_details,
        input_shape=[INPUT_SIZE, INPUT_SIZE],
        score_thr=SCORE_THR,
        dist_thr=DIST_THR
    )

    # Draw the detected lines in red
    for line in detected_lines:
        x1, y1, x2, y2 = [int(coord) for coord in line]
        cv2.line(processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Initialize depth-related variables
    depth_map = None
    lines_3d = np.array([])
    valid_3d_indices = []

    # Do depth estimation if enabled and we have the required models
    if (ENABLE_DEPTH_ESTIMATION and 
        depth_estimation_model is not None and 
        camera_intrinsics is not None):
        
        # Get depth for this frame
        depth_map = estimate_depth(depth_estimation_model, input_frame)

        # Convert 2D lines to 3D if we got a valid depth map
        if depth_map is not None and len(detected_lines) > 0:
            lines_3d, valid_3d_indices = backproject_lines_to_3d(
                detected_lines, depth_map, camera_intrinsics, DEPTH_SCALE_FACTOR
            )

    return processed_image, detected_lines, depth_map, lines_3d, valid_3d_indices


def process_video(mlsd_interpreter, input_details, output_details,
                 depth_model=None, camera_matrix=None):
    """Handle video processing with line matching and optional depth."""
    
    print(f"[INFO] Trying to open video: {VIDEO_PATH}")
    video_capture = cv2.VideoCapture(VIDEO_PATH)
    
    if not video_capture.isOpened():
        print(f"[ERROR] Couldn't open the video file. Check if the path is correct.")
        return

    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    print(f"[SUCCESS] Video opened: {frame_width}x{frame_height} @ {frame_rate}fps")

    # Set up video writer for output
    fourcc_codec = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc_codec, frame_rate, 
                                   (frame_width, frame_height))

    # Initialize line matching stuff
    line_descriptor = None
    line_matcher = None
    if ENABLE_LINE_MATCHING:
        line_descriptor = LineDescriptor(num_bands=NUM_BANDS, band_width=BAND_WIDTH)
        line_matcher = LineMatcher()
        print("[INFO] Line matching is enabled")

    if ENABLE_3D_VISUALIZATION and ENABLE_DEPTH_ESTIMATION:
        print("[INFO] 3D visualization will show in OpenCV window")

    # Variables to store previous frame data for matching
    previous_frame = None
    previous_lines = None
    previous_descriptors = None
    frame_counter = 0

    print("--- STARTING VIDEO PROCESSING ---")
    
    try:
        while True:
            success, current_frame = video_capture.read()
            if not success:
                print("Reached end of video.")
                break

            frame_counter += 1

            # Resize frame if needed
            resized_frame, scaling_factor = resize_if_needed(current_frame)

            # Progress update every 30 frames (about once per second for 30fps video)
            if frame_counter % 30 == 0:
                print(f"Processing frame {frame_counter}...")

            # Process the frame (line detection, depth, 3D)
            processed_frame, current_lines, depth_map, lines_3d, valid_3d_indices = process_frame(
                resized_frame, mlsd_interpreter, input_details, output_details,
                depth_model, camera_matrix
            )

            # Line matching with previous frame
            matched_line_indices = []  # Keep track of which lines matched
            
            if (ENABLE_LINE_MATCHING and 
                previous_frame is not None and 
                len(current_lines) > 0):
                
                # Compute descriptors for current lines
                current_descriptors, valid_line_indices = line_descriptor.compute_descriptors(
                    resized_frame, current_lines)

                # Match with previous frame if we have data
                if (len(current_descriptors) > 0 and 
                    previous_descriptors is not None and 
                    len(previous_descriptors) > 0):
                    
                    valid_current_lines = current_lines[valid_line_indices]

                    # Find matches
                    raw_matches = line_matcher.match_lines(
                        previous_lines, previous_descriptors,
                        valid_current_lines, current_descriptors
                    )

                    # Filter out outliers
                    filtered_matches, num_outliers = selmap_filter_lines(
                        raw_matches, previous_lines, valid_current_lines)

                    if frame_counter % 30 == 0:
                        if num_outliers > 0:
                            print(f"  Found {len(filtered_matches)} good matches "
                                  f"({num_outliers} outliers filtered out)")
                        else:
                            print(f"  Found {len(filtered_matches)} line matches")

                    # Draw matched lines in green (overwrites the red ones)
                    for prev_idx, curr_idx in filtered_matches:
                        line = valid_current_lines[curr_idx]
                        x1, y1, x2, y2 = [int(coord) for coord in line]
                        cv2.line(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Remember which line matched for 3D visualization
                        original_line_index = valid_line_indices[curr_idx]
                        matched_line_indices.append(original_line_index)

                # Update previous frame data
                if len(valid_line_indices) > 0:
                    previous_lines = current_lines[valid_line_indices]
                else:
                    previous_lines = current_lines
                previous_descriptors = current_descriptors
                
            else:
                # First frame or no matching - just compute descriptors for next time
                if ENABLE_LINE_MATCHING and len(current_lines) > 0:
                    current_descriptors, valid_line_indices = line_descriptor.compute_descriptors(
                        resized_frame, current_lines)
                    if len(valid_line_indices) > 0:
                        previous_lines = current_lines[valid_line_indices]
                    else:
                        previous_lines = current_lines
                    previous_descriptors = current_descriptors

            previous_frame = resized_frame.copy()

            # Show depth map if enabled
            if DEPTH_VISUALIZATION and depth_map is not None:
                depth_visualization = visualize_depth_map(depth_map)
                if depth_visualization is not None:
                    cv2.imshow('Depth Map', depth_visualization)

            # 3D visualization
            if (ENABLE_3D_VISUALIZATION and 
                ENABLE_DEPTH_ESTIMATION and 
                len(lines_3d) > 0):
                
                # Figure out which 3D lines correspond to matched 2D lines
                matched_3d_line_indices = []
                for matched_2d_idx in matched_line_indices:
                    if matched_2d_idx in valid_3d_indices:
                        # Find position in 3D array
                        position_in_3d = valid_3d_indices.index(matched_2d_idx)
                        matched_3d_line_indices.append(position_in_3d)

                # Render and show 3D visualization
                viz_3d_img = render_3d_visualization_to_image(lines_3d, matched_3d_line_indices)
                if viz_3d_img is not None:
                    cv2.imshow('3D Visualization', viz_3d_img)

                if frame_counter % 30 == 0:
                    print(f"  3D: {len(lines_3d)}/{len(current_lines)} lines converted "
                          f"(matched: {len(matched_3d_line_indices)})")

            # Make sure output frame is the right size for video writer
            if (processed_frame.shape[0] != frame_height or 
                processed_frame.shape[1] != frame_width):
                processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))

            # Save frame and display
            video_writer.write(processed_frame)
            cv2.imshow('M-LSD Line Detection', processed_frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q' - quitting...")
                break

    except Exception as error:
        print(f"Error during video processing: {error}")
        import traceback
        traceback.print_exc()

    # Clean up
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Close any matplotlib windows if they exist
    # Note: this variable might not be defined - should probably fix this
    try:
        if 'fig_3d' in globals() and fig_3d is not None:
            import matplotlib.pyplot as plt
            plt.close(fig_3d)
    except:
        pass  # Ignore if matplotlib not used

    print(f"[SUCCESS] Processed {frame_counter} frames total. Output saved: {OUTPUT_VIDEO_NAME}")


def process_images(mlsd_interpreter, input_details, output_details,
                  depth_model=None, camera_matrix=None):
    """Process a folder full of images with line matching and depth."""
    
    image_files = get_image_list(IMAGE_FOLDER)

    if not image_files:
        print(f"[ERROR] No images found in {IMAGE_FOLDER}")
        return

    print(f"[INFO] Found {len(image_files)} images to process")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_IMAGE_FOLDER):
        os.makedirs(OUTPUT_IMAGE_FOLDER)
        print(f"[INFO] Created output directory: {OUTPUT_IMAGE_FOLDER}")

    # Set up line matching
    line_descriptor = None
    line_matcher = None
    if ENABLE_LINE_MATCHING:
        line_descriptor = LineDescriptor(num_bands=NUM_BANDS, band_width=BAND_WIDTH)
        line_matcher = LineMatcher()
        print("[INFO] Line matching enabled for image sequence")

    if ENABLE_3D_VISUALIZATION and ENABLE_DEPTH_ESTIMATION:
        print("[INFO] 3D visualization will be shown in OpenCV window")

    # Variables for tracking between images
    prev_frame = None
    prev_lines = None
    prev_descriptors = None

    print("--- STARTING IMAGE SEQUENCE PROCESSING ---")
    print("Controls: 'n' = next image, 'm' = show matches with previous, 'q' = quit")

    for image_idx, image_path in enumerate(image_files):
        current_frame = cv2.imread(image_path)
        if current_frame is None:
            print(f"[WARNING] Couldn't load image: {image_path}")
            continue

        print(f"\nProcessing ({image_idx+1}/{len(image_files)}): {os.path.basename(image_path)}")

        # Resize if the image is too big
        resized_frame, scale_factor = resize_if_needed(current_frame)

        # Process frame for line detection and depth
        processed_frame, detected_lines, depth_map, lines_3d, valid_3d_indices = process_frame(
            resized_frame, mlsd_interpreter, input_details, output_details,
            depth_model, camera_matrix
        )

        print(f"  Found {len(detected_lines)} lines")

        # Compute line descriptors for matching
        current_descriptors = None
        valid_indices = []
        if ENABLE_LINE_MATCHING and len(detected_lines) > 0:
            current_descriptors, valid_indices = line_descriptor.compute_descriptors(
                resized_frame, detected_lines)
            print(f"  Computed descriptors for {len(current_descriptors)} lines")

        # Match with previous image if available
        matched_indices = []
        matches = []
        if (ENABLE_LINE_MATCHING and 
            prev_frame is not None and 
            current_descriptors is not None and 
            prev_descriptors is not None):
            
            if len(current_descriptors) > 0 and len(prev_descriptors) > 0:
                valid_lines = detected_lines[valid_indices]

                # Find matches
                raw_matches = line_matcher.match_lines(
                    prev_lines, prev_descriptors,
                    valid_lines, current_descriptors
                )

                # Filter outliers
                matches, num_rejected = selmap_filter_lines(raw_matches, prev_lines, valid_lines)

                if num_rejected > 0:
                    print(f"  Found {len(matches)} matches ({num_rejected} outliers filtered)")
                else:
                    print(f"  Found {len(matches)} matches with previous image")

                # Draw matched lines in green
                for prev_match_idx, curr_match_idx in matches:
                    line = valid_lines[curr_match_idx]
                    x1, y1, x2, y2 = [int(coord) for coord in line]
                    cv2.line(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Track original line index for 3D viz
                    original_idx = valid_indices[curr_match_idx]
                    matched_indices.append(original_idx)

        # Show depth map if available
        if DEPTH_VISUALIZATION and depth_map is not None:
            depth_colored = visualize_depth_map(depth_map)
            if depth_colored is not None:
                cv2.imshow('Depth Map', depth_colored)

        # 3D visualization
        if (ENABLE_3D_VISUALIZATION and 
            ENABLE_DEPTH_ESTIMATION and 
            len(lines_3d) > 0):
            
            # Map 2D matched indices to 3D line indices
            matched_3d_indices = []
            for matched_2d_idx in matched_indices:
                if matched_2d_idx in valid_3d_indices:
                    pos_in_3d_array = valid_3d_indices.index(matched_2d_idx)
                    matched_3d_indices.append(pos_in_3d_array)

            # Render 3D scene to image and show it
            viz_3d_image = render_3d_visualization_to_image(lines_3d, matched_3d_indices)
            if viz_3d_image is not None:
                cv2.imshow('3D Lines', viz_3d_image)
            
            print(f"  3D: {len(lines_3d)}/{len(detected_lines)} lines reconstructed "
                  f"(matched: {len(matched_3d_indices)})")

        # Save the processed image
        output_filename = f"mlsd_{os.path.basename(image_path)}"
        output_full_path = os.path.join(OUTPUT_IMAGE_FOLDER, output_filename)
        cv2.imwrite(output_full_path, processed_frame)
        print(f"  Saved: {output_filename}")

        # Display and wait for user input
        cv2.imshow('M-LSD Results', processed_frame)

        # User interaction loop
        while True:
            key_pressed = cv2.waitKey(0) & 0xFF

            if key_pressed == ord('q'):
                print("User quit.")
                cv2.destroyAllWindows()
                return
            elif key_pressed == ord('n'):
                break  # Go to next image
            elif key_pressed == ord('m') and len(matches) > 0 and prev_frame is not None:
                # Show detailed match visualization
                print("  Creating match visualization...")
                valid_current_lines = detected_lines[valid_indices]
                
                match_visualization = visualize_matches(
                    prev_frame, prev_lines,
                    resized_frame, valid_current_lines,
                    matches, max_matches=50  # Limit to prevent clutter
                )

                # Save the match visualization
                match_filename = f"matches_{image_idx-1}_to_{image_idx}_{os.path.basename(image_path)}"
                match_output_path = os.path.join(OUTPUT_IMAGE_FOLDER, match_filename)
                cv2.imwrite(match_output_path, match_visualization)

                cv2.imshow('Line Matches', match_visualization)
                print(f"  Match visualization saved: {match_filename}")
                print("  Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyWindow('Line Matches')
                
            elif key_pressed == ord('m'):
                print("  No matches to show (first image or no matches found)")

        # Store current data for next iteration
        prev_frame = resized_frame.copy()
        if current_descriptors is not None and len(current_descriptors) > 0:
            prev_lines = detected_lines[valid_indices]
            prev_descriptors = current_descriptors
        else:
            prev_lines = detected_lines
            prev_descriptors = None

    cv2.destroyAllWindows()

    # Clean up matplotlib if it was used
    try:
        if 'fig_3d' in globals() and fig_3d is not None:
            import matplotlib.pyplot as plt
            plt.close(fig_3d)
    except:
        pass

    print(f"\n[SUCCESS] All done! Results saved to: {OUTPUT_IMAGE_FOLDER}")


def main():
    """Main function - orchestrates everything."""
    print("=== M-LSD LINE DETECTION WITH DEPTH ESTIMATION ===")
    print(f"[INFO] Running in {INPUT_MODE.upper()} mode")

    # Load camera calibration if we need it
    camera_matrix = None
    distortion_coeffs = None
    
    if ENABLE_DEPTH_ESTIMATION:
        camera_matrix, distortion_coeffs = load_calibration()
        if camera_matrix is None:
            print("[WARNING] No calibration data - disabling 3D features")
            print("[INFO] Will continue with 2D line detection only...")
            # Don't exit, just continue without 3D features
        else:
            print("[INFO] Calibration loaded - 3D features are available!")
    else:
        print("[INFO] Depth estimation disabled - running 2D detection only")

    # Load the M-LSD model
    print(f"\n[INFO] Loading M-LSD model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("[ERROR] Make sure you have downloaded the M-LSD TFLite model")
        return

    try:
        # Load TensorFlow Lite model
        tflite_interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        tflite_interpreter.allocate_tensors()
        
        # Get input and output details
        model_input_details = tflite_interpreter.get_input_details()
        model_output_details = tflite_interpreter.get_output_details()
        
        print("[SUCCESS] M-LSD model loaded successfully!")
        
    except Exception as error:
        print(f"[ERROR] Failed to load M-LSD model: {error}")
        return

    # Initialize depth estimation model if needed
    depth_estimation_model = None
    if ENABLE_DEPTH_ESTIMATION:
        print("\n[INFO] Setting up depth estimation model...")
        print("[INFO] This might take a little while on first run...")
        
        depth_estimation_model = initialize_depth_model(DEPTH_MODEL_PATH)

        if depth_estimation_model is None:
            print("[WARNING] Depth model failed to initialize")
            print("[WARNING] Continuing without depth estimation...")
            camera_matrix = None  # Disable 3D features

    # Run the appropriate processing mode
    print(f"\n[INFO] Starting {INPUT_MODE} processing...")
    
    if INPUT_MODE == 'video':
        process_video(tflite_interpreter, model_input_details, model_output_details,
                     depth_estimation_model, camera_matrix)
    elif INPUT_MODE == 'images':
        process_images(tflite_interpreter, model_input_details, model_output_details,
                      depth_estimation_model, camera_matrix)
    else:
        print(f"[ERROR] Invalid INPUT_MODE: '{INPUT_MODE}'. Use 'video' or 'images'.")


if __name__ == "__main__":
    print("[INFO] Starting M-LSD line detection script...")
    main()
    print("[INFO] Script finished!")