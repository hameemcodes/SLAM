import numpy as np
import cv2
import glob
import os
import pickle
import time

# Camera calibration parameters
# You can modify these variables as needed
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners per chessboard row and column
SQUARE_SIZE = 2.45        # Size of a square in centimeters
# Use a raw string with the full Windows path (avoid "\U" unicode escape)
#CALIBRATION_IMAGES_DIR = 'C:\Users\hamee\Downloads\chessboard2*.jpg'  # Path to calibration images
CALIBRATION_IMAGES_DIR = r'C:\Users\hamee\Downloads\chessboard2'  # directory (no wildcard)
IMAGE_PATTERNS = ('*.jpg', '*.jpeg', '*.png', '*.HEIC', '*.DNG')  # include HEIC/heif
OUTPUT_DIRECTORY = 'output'  # Directory to save calibration results
SAVE_UNDISTORTED = False   # Whether to save undistorted images

# Performance optimization settings
MAX_DETECTION_SIZE = 3000  # Max dimension for corner detection (width or height)
MAX_REFINEMENT_SIZE = 3000  # Max dimension for corner refinement
ENABLE_FALLBACK_DETECTION = False  # Disable slow fallback methods

def _get_image_list():
    """Return list of image file paths from CALIBRATION_IMAGES_DIR matching IMAGE_PATTERNS."""
    images = []
    for p in IMAGE_PATTERNS:
        images.extend(glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, p)))
    images.sort()
    return images

def imread_flexible(path):

    img = cv2.imread(path)
    if img is not None:
        return img
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.heic', '.heif'):
        try:
            # lazy import so script still runs if user doesn't need HEIC support
            import pillow_heif  # registers HEIF/HEIC support for Pillow
            from PIL import Image
            pillow_heif.register_heif_opener()  # safe to call
            pil = Image.open(path).convert('RGB')
            arr = np.array(pil)[:, :, ::-1].copy()  # RGB -> BGR for OpenCV compatibility
            return arr
        except Exception as e:
            print(f"HEIC read failed for {path}: {e}")
            return None
    return None

def convert_images_to_jpg():

    print("Scanning for images to convert to JPG...")

    # Get all image files
    all_files = []
    for pattern in IMAGE_PATTERNS:
        all_files.extend(glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, pattern)))

    if not all_files:
        print(f"No images found in {CALIBRATION_IMAGES_DIR}")
        return

    # Filter out files that are already JPG/JPEG
    files_to_convert = [f for f in all_files if not f.lower().endswith(('.jpg', '.jpeg'))]

    if not files_to_convert:
        print("All images are already in JPG format. No conversion needed.")
        return

    print(f"Found {len(files_to_convert)} image(s) to convert to JPG format...")

    converted_count = 0
    failed_count = 0

    for idx, filepath in enumerate(files_to_convert):
        print(f"Converting {idx+1}/{len(files_to_convert)}: {os.path.basename(filepath)}...", end=" ")

        # Load the image using the flexible reader
        img = imread_flexible(filepath)

        if img is None:
            print("FAILED (could not load image)")
            failed_count += 1
            continue

        # Create new filename with .jpg extension
        base_name = os.path.splitext(filepath)[0]
        jpg_filepath = base_name + '.jpg'

        # Save as JPG with maximum quality (100)
        success = cv2.imwrite(jpg_filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        if success:
            # Delete the original file
            try:
                os.remove(filepath)
                print(f"OK (saved as {os.path.basename(jpg_filepath)})")
                converted_count += 1
            except Exception as e:
                print(f"WARNING (saved JPG but could not delete original: {e})")
                converted_count += 1
        else:
            print("FAILED (could not save JPG)")
            failed_count += 1

    print(f"\nConversion complete: {converted_count} converted, {failed_count} failed")

def downscale_image(img, max_dimension):
    """
    Downscale image if it exceeds max_dimension while preserving aspect ratio.
    Returns (downscaled_img, scale_factor)
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)

    if max_dim <= max_dimension:
        return img, 1.0

    scale = max_dimension / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return downscaled, scale

def calibrate_camera():
    """
    Calibrate the camera using chessboard images.
    
    Returns:
        ret: The RMS re-projection error
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    
    # Scale object points by square size (for real-world measurements)
    objp = objp * SQUARE_SIZE
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Store downscaled image size for calibration
    downscaled_size = None
    original_size = None

    # Get list of calibration images
    images = _get_image_list()
    
    if not images:
        print(f"No calibration images found at {CALIBRATION_IMAGES_DIR}")
        return None, None, None, None, None
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    print(f"Found {len(images)} calibration images")

    # Process each calibration image
    for idx, fname in enumerate(images):
        start_time = time.time()

        img = imread_flexible(fname)
        if img is None:
            print(f"Warning: failed to load {fname}, skipping")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Store original dimensions for scaling coordinates
        original_h, original_w = gray.shape[:2]

        # Downscale for corner detection (major speedup)
        gray_detection, detection_scale = downscale_image(gray, MAX_DETECTION_SIZE)

        print(f"  Image {idx+1}/{len(images)}: {original_w}x{original_h} -> {gray_detection.shape[1]}x{gray_detection.shape[0]} (scale: {detection_scale:.3f})", end="")

        # Find corners on downscaled image (much faster)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray_detection, CHESSBOARD_SIZE, flags)

        # Optional: try fallback methods if enabled and first attempt failed
        if not ret and ENABLE_FALLBACK_DETECTION:
            try:
                ret, corners = cv2.findChessboardCornersSB(gray_detection, CHESSBOARD_SIZE, None)
                if ret:
                    # Corners are already in downscaled space, no scaling needed here
                    print(" [fallback SB]", end="")
            except AttributeError:
                pass

        # If found, add object points and image points
        if ret:
            # Verify consistent downscaled dimensions
            current_downscaled_size = (gray_detection.shape[1], gray_detection.shape[0])

            if downscaled_size is None:
                downscaled_size = current_downscaled_size
                original_size = (original_w, original_h)
                print(f" [Using {downscaled_size[0]}x{downscaled_size[1]} for calibration]", end="")
            elif current_downscaled_size != downscaled_size:
                print(f" - SKIPPED (size mismatch: {current_downscaled_size[0]}x{current_downscaled_size[1]} != {downscaled_size[0]}x{downscaled_size[1]})", end="")
                elapsed = time.time() - start_time
                print(f" ({elapsed:.2f}s)")
                continue

            objpoints.append(objp)

            # Refine corners on the SAME downscaled image (single-scaling approach)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray_detection, corners, (11, 11), (-1, -1), criteria)

            # Keep corners at downscaled resolution for calibration
            imgpoints.append(corners2)

            # Scale corners for drawing on full-resolution image
            corners_for_drawing = corners2 / detection_scale
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners_for_drawing, ret)
            
            # Save image with corners drawn
            # Save image with corners drawn
            basename = os.path.basename(fname)
            name_without_ext = os.path.splitext(basename)[0]
            output_filename = f'corners_{name_without_ext}.jpg' # Force .jpg
            output_img_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
            cv2.imwrite(output_img_path, img)

            elapsed = time.time() - start_time
            print(f" - FOUND ({elapsed:.2f}s)")
        else:
            elapsed = time.time() - start_time
            print(f" - NOT FOUND ({elapsed:.2f}s)")
    
    if not objpoints:
        print("No chessboard patterns were detected in any images.")
        return None, None, None, None, None

    print(f"Calibrating camera at downscaled resolution ({downscaled_size[0]}x{downscaled_size[1]})...")

    # Calibrate camera using downscaled image dimensions
    ret, mtx_downscaled, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, downscaled_size, None, None
    )

    # Scale camera matrix to original resolution
    scale_x = original_size[0] / downscaled_size[0]
    scale_y = original_size[1] / downscaled_size[1]

    mtx = mtx_downscaled.copy()
    mtx[0, 0] *= scale_x  # fx
    mtx[1, 1] *= scale_y  # fy
    mtx[0, 2] *= scale_x  # cx
    mtx[1, 2] *= scale_y  # cy

    print(f"Scaled camera matrix to original resolution ({original_size[0]}x{original_size[1]})")
    
    # Save calibration results
    calibration_data = {
        'camera_matrix': mtx,
        'camera_matrix_downscaled': mtx_downscaled,
        'distortion_coefficients': dist,
        'rotation_vectors': rvecs,
        'translation_vectors': tvecs,
        'reprojection_error': ret,
        'calibration_resolution': downscaled_size,
        'target_resolution': original_size
    }
    
    with open(os.path.join(OUTPUT_DIRECTORY, 'calibration_data.pkl'), 'wb') as f:
        pickle.dump(calibration_data, f)
    
    # Save camera matrix and distortion coefficients as text files
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'camera_matrix.txt'), mtx)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'distortion_coefficients.txt'), dist)
    
    print(f"Calibration complete! RMS re-projection error: {ret}")
    print(f"Results saved to {OUTPUT_DIRECTORY}")

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, mtx_downscaled

def undistort_images(mtx, dist):
    """
    Undistort all calibration images using the calibration results.
    
    Args:
        mtx: Camera matrix
        dist: Distortion coefficients
    """
    if not SAVE_UNDISTORTED:
        return
    
    images = _get_image_list()
    
    if not images:
        print(f"No images found at {CALIBRATION_IMAGES_DIR} to undistort")
        return
    
    undistorted_dir = os.path.join(OUTPUT_DIRECTORY, 'undistorted')
    if not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)
    
    print(f"Undistorting {len(images)} images...")
    
    for idx, fname in enumerate(images):
        img = imread_flexible(fname)
        if img is None:
            print(f"Warning: failed to load {fname}, skipping")
            continue
        h, w = img.shape[:2]
        
        # Refine camera matrix based on free scaling parameter
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # Undistort image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Crop the image (optional)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # Save undistorted image
        # Save undistorted image
        basename = os.path.basename(fname)
        name_without_ext = os.path.splitext(basename)[0]
        output_filename = f'undistorted_{name_without_ext}.jpg' # Force .jpg
        output_img_path = os.path.join(undistorted_dir, output_filename)
        cv2.imwrite(output_img_path, dst)

        
        print(f"Undistorted image {idx+1}/{len(images)}: {fname}")
    
    print(f"Undistorted images saved to {undistorted_dir}")

def calculate_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """
    Calculate the reprojection error for each calibration image.
    
    Args:
        objpoints: 3D points in real world space
        imgpoints: 2D points in image plane
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    
    Returns:
        mean_error: Mean reprojection error
    """
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        print(f"Reprojection error for image {i+1}: {error}")
    
    mean_error = total_error / len(objpoints)
    print(f"Mean reprojection error: {mean_error}")
    
    return mean_error

def main():
    """
    Main function to run the camera calibration process.
    """
    print("Starting camera calibration...")

    # Convert all images to JPG format first
    convert_images_to_jpg()

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, mtx_downscaled = calibrate_camera()

    if mtx is None:
        print("Calibration failed. Exiting.")
        return

    # Calculate and display per-image reprojection errors (at downscaled resolution)
    print("\n--- Per-Image Reprojection Errors (at calibration resolution) ---")
    calculate_reprojection_error(objpoints, imgpoints, mtx_downscaled, dist, rvecs, tvecs)

    # Undistort images
    undistort_images(mtx, dist)
    
    print("Camera calibration completed successfully!")

if __name__ == "__main__":
    main()