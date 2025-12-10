# Camera calibration script for SLAM coursework
# Last updated: 2am 


import numpy as np
import cv2
import glob
import os
import pickle
import time

# Camera calibration params - based on the opencv tutorial
# TODO: assess matrix
chessboard_size = (9, 6)  # inner corners - spent forever counting these correctly
square_sz = 2.45   # measured with ruler, in cm (TODO: double check this measurement)

# Change this to wherever your images are
CALIBRATION_IMAGES_DIR = r'C:\Users\hamee\Downloads\chessboard2'  # NOTE: change this path!
img_patterns = ('*.jpg', '*.jpeg', '*.png', '*.HEIC', '*.DNG')
output_dir = 'output'
SAVE_UNDISTORTED = False   # set to True if you want to see undistorted images (takes forever though)

# Performance stuff - these numbers worked best after testing
MAX_DETECTION_SIZE = 3000  # making this bigger is really slow
max_refine_size = 3000
enable_fallback = False  # fallback methods are super slow, disabled for now

def _get_image_list():
    # gets all the calibration images from the folder
    img_list = []
    for pattern in img_patterns:
        found_imgs = glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, pattern))
        img_list.extend(found_imgs)
    img_list.sort()  # make sure they're in order
    return img_list

def imread_flexible(path):
    # this function handles different image formats
    # mainly for HEIC images from iphone (pain to work with!)

    img = cv2.imread(path)
    if img is not None:
        return img

    # check if it's HEIC format
    file_ext = os.path.splitext(path)[1].lower()
    if file_ext in ('.heic', '.heif'):
        try:
            # found this solution on stackoverflow, seems to work
            import pillow_heif
            from PIL import Image
            pillow_heif.register_heif_opener()
            pil_img = Image.open(path).convert('RGB')
            # convert PIL to opencv format (BGR)
            arr = np.array(pil_img)[:, :, ::-1].copy()
            return arr
        except Exception as e:
            print(f"Couldn't read HEIC file {path}: {e}")
            return None

    return None

def convert_images_to_jpg():
    # converts all images to JPG format since opencv likes JPG better
    # TODO: make backup of originals?

    print("Looking for images to convert...")

    # Get all image files from folder
    all_files = []
    for pattern in img_patterns:
        found = glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, pattern))
        all_files.extend(found)

    if not all_files:
        print(f"Didn't find any images in {CALIBRATION_IMAGES_DIR}")
        return

    # skip files that are already jpg
    files_to_convert = []
    for f in all_files:
        if not f.lower().endswith(('.jpg', '.jpeg')):
            files_to_convert.append(f)

    if not files_to_convert:
        print("Everything's already JPG, we're good")
        return

    print(f"Found {len(files_to_convert)} files that need converting...")

    num_converted = 0
    num_failed = 0

    for idx, filepath in enumerate(files_to_convert):
        print(f"Converting {idx+1}/{len(files_to_convert)}: {os.path.basename(filepath)}...", end=" ")

        # try to load the image
        img = imread_flexible(filepath)

        if img is None:
            print("FAILED - couldn't load")
            num_failed += 1
            continue

        # create new filename
        base_name = os.path.splitext(filepath)[0]
        jpg_path = base_name + '.jpg'

        # save with quality=100 (best quality)
        success = cv2.imwrite(jpg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        if success:
            # delete original file
            try:
                os.remove(filepath)
                print(f"OK")
                num_converted += 1
            except Exception as e:
                print(f"WARNING - saved but couldn't delete original")
                num_converted += 1
        else:
            print("FAILED - couldn't save")
            num_failed += 1

    print(f"\nDone! Converted: {num_converted}, Failed: {num_failed}")

def downscale_image(img, max_dim):
    # downscales image if too big (helps with speed)
    # returns the resized image and the scale factor used

    h, w = img.shape[:2]
    maxDim = max(h, w)

    if maxDim <= max_dim:
        return img, 1.0  # no need to resize

    # calculate scale
    scale_factor = max_dim / maxDim
    newW = int(w * scale_factor)
    newH = int(h * scale_factor)

    # resize using INTER_AREA (best for downscaling)
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
    return resized, scale_factor

def calibrate_camera():
    # Main calibration function - this does all the heavy lifting
    # Based on opencv camera calibration tutorial
    # TODO: test 

    # Set up object points for the chessboard pattern
    # these are the 3D coordinates of chessboard corners in real world
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # multiply by square size to get actual measurements
    objp = objp * square_sz

    # these will store the corner positions from all images
    obj_points = []  # 3D points (real world)
    img_points = []  # 2D points (in image)

    # keep track of image sizes
    downscaled_sz = None
    orig_sz = None

    # get all the chessboard images
    imgs = _get_image_list()

    if not imgs:
        print(f"Couldn't find any calibration images in {CALIBRATION_IMAGES_DIR}")
        return None, None, None, None, None

    # make output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Found {len(imgs)} calibration images - let's process them")

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