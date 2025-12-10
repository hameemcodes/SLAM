# Depth estimation utilities using Depth Anything V2
# TODO: try to obtain depth histogram for better visualization
# Last updated: late night coding session

# NOTE: You need to clone the Depth-Anything-V2 repo and copy the depth_anything_v2 folder here
# git clone https://github.com/DepthAnything/Depth-Anything-V2
# I spent like 2 hours figuring out how to set this up properly

import cv2
import numpy as np
import torch
import sys
import os


def initialize_depth_model(model_path=None):
    # loads the Depth Anything V2 model from checkpoint
    # this is the monocular depth estimation model (estimates depth from single image)

    print("[INFO] Loading depth model...")

    if model_path is None or not os.path.exists(model_path):
        print(f"[ERROR] Can't find model file: {model_path}")
        return None

    # try importing the depth anything v2 module
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        print("[ERROR] depth_anything_v2 module not found!")
        print("[INFO] You need to clone the repo and copy the folder:")
        print("       https://github.com/DepthAnything/Depth-Anything-V2")
        return None

    # check if CUDA is available (way faster than CPU!)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    if device == 'cpu':
        print("[WARNING] Running on CPU will be really slow...")

    try:
        # model configurations (from the github repo)
        # using vitb since it's a good balance of speed vs accuracy
        model_cfgs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},  # this one
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        encoder_type = 'vitb'
        print(f"[INFO] Creating model with {encoder_type} encoder...")

        # create the model
        depth_model = DepthAnythingV2(**model_cfgs[encoder_type])

        # load the pretrained weights
        print(f"[INFO] Loading weights from {model_path}")
        weights = torch.load(model_path, map_location=device)
        depth_model.load_state_dict(weights)

        # move to GPU if available and set to eval mode
        depth_model = depth_model.to(device).eval()

        print("[SUCCESS] Depth model loaded!")
        return depth_model

    except Exception as e:
        print(f"[ERROR] Something went wrong loading the model: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_depth(depth_model, frame):
    # estimates depth map for a single frame
    # input is BGR image (opencv format)
    # returns depth map where higher values = further away

    if depth_model is None:
        return None

    try:
        # run inference (the model expects BGR format which is what opencv uses)
        depthMap = depth_model.infer_image(frame)

        # check if we got valid output
        if depthMap is None or depthMap.size == 0:
            print("[WARNING] Depth estimation returned nothing")
            return None

        # sometimes the model outputs inf or nan values, need to clean those
        if not np.isfinite(depthMap).all():
            print("[WARNING] Got some invalid values in depth map, fixing...")
            depthMap = np.nan_to_num(depthMap, nan=0.0, posinf=0.0, neginf=0.0)

        return depthMap

    except Exception as e:
        print(f"[ERROR] Depth estimation crashed: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_depth_map(depth_map):
    # converts depth map to a colored image for visualization
    # makes it easier to see what's going on

    if depth_map is None or depth_map.size == 0:
        return None

    try:
        # normalize depth to 0-255 range for display
        dMin = depth_map.min()
        dMax = depth_map.max()

        # avoid divide by zero if depth is all the same
        if dMax - dMin < 1e-6:
            print("[WARNING] Depth map is flat (no variation)")
            depth_norm = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            depth_norm = ((depth_map - dMin) / (dMax - dMin) * 255).astype(np.uint8)

        # apply colormap - TURBO looks really nice
        # blue = close, red = far away
        colored_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

        return colored_depth

    except Exception as e:
        print(f"[ERROR] Couldn't visualize depth: {e}")
        return None
