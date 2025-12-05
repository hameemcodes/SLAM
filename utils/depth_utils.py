"""
Depth Anything V2 utilities for depth estimation.

This module provides functions to load the Depth Anything V2 model from a local checkpoint
and perform depth estimation on images.

Note: Requires minimal Depth Anything V2 code. To install:
    git clone https://github.com/DepthAnything/Depth-Anything-V2
    copy the 'depth_anything_v2' folder to this project root
    OR add to Python path: sys.path.insert(0, 'path/to/Depth-Anything-V2')
"""

import cv2
import numpy as np
import torch
import sys
import os


def initialize_depth_model(model_path=None):
    """
    Initialize Depth Anything V2 model from local checkpoint.

    Args:
        model_path: Path to local .pth checkpoint file

    Returns:
        model: Loaded Depth Anything V2 model on appropriate device
        None if loading fails
    """
    print("[INFO] Initializing Depth Anything V2 model...")

    if model_path is None or not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None

    # Try to import Depth Anything V2
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        print("[ERROR] Could not import depth_anything_v2 module")
        print("[INFO] Please copy the 'depth_anything_v2' folder from:")
        print("       https://github.com/DepthAnything/Depth-Anything-V2")
        print("       to your project root directory")
        return None

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    try:
        # Model configuration for vitb
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        encoder = 'vitb'  # Base model
        print(f"[INFO] Creating model architecture ({encoder})...")

        # Create model
        model = DepthAnythingV2(**model_configs[encoder])

        # Load pretrained weights
        print(f"[INFO] Loading weights from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Move to device and set to eval mode
        model = model.to(device).eval()

        print("[SUCCESS] Depth Anything V2 model loaded successfully")
        return model

    except Exception as e:
        print(f"[ERROR] Failed to load depth model: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_depth(depth_model, frame):
    """
    Estimate depth map for a given frame.

    Args:
        depth_model: Initialized Depth Anything V2 model
        frame: Input image as numpy array (H, W, 3) in BGR format

    Returns:
        depth_map: Depth map as numpy array (H, W) with float values
                   Higher values = farther from camera
                   Returns None if estimation fails
    """
    if depth_model is None:
        return None

    try:
        # Depth Anything V2's infer_image expects BGR format (OpenCV format)
        depth_map = depth_model.infer_image(frame)

        # Validate output
        if depth_map is None or depth_map.size == 0:
            print("[WARNING] Depth estimation returned empty result")
            return None

        # Check for invalid values
        if not np.isfinite(depth_map).all():
            print("[WARNING] Depth map contains inf/nan values, cleaning...")
            depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)

        return depth_map

    except Exception as e:
        print(f"[ERROR] Depth estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_depth_map(depth_map):
    """
    Convert depth map to colorized visualization for display.

    Args:
        depth_map: Depth map as numpy array (H, W)

    Returns:
        vis_image: BGR image for cv2.imshow (H, W, 3) with uint8 dtype
                   Returns None if input is invalid
    """
    if depth_map is None or depth_map.size == 0:
        return None

    try:
        # Normalize to 0-255 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min < 1e-6:  # Avoid division by zero
            print("[WARNING] Depth map has no variation")
            depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

        # Apply colormap (TURBO is good for depth - blue=near, red=far)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

        return depth_colored

    except Exception as e:
        print(f"[ERROR] Depth visualization failed: {e}")
        return None
