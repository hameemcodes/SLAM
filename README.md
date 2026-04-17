# 3D-Object-SLAM

A visual SLAM (Simultaneous Localization and Mapping) system that combines advanced computer vision techniques to detect, track, and reconstruct 3D line features from video or image sequences.

## Overview

This project integrates multiple state-of-the-art computer vision models to create a complete 3D reconstruction pipeline:

- **2D Line Detection**: M-LSD (Mobile Line Segment Detection) using deep learning
- **Depth Estimation**: Depth Anything V2 for monocular depth prediction
- **Line Matching**: LBD (Line Band Descriptors) for feature tracking across frames
- **Camera Calibration**: Chessboard-based intrinsic parameter estimation
- **3D Reconstruction**: Back-projection of 2D lines to 3D space using depth and camera intrinsics

The system generates 3D point clouds and line reconstructions of scenes from monocular video or image sequences.

## Features

- **Real-time Processing**: Frame-by-frame video processing with visualization
- **Robust Line Matching**: SelMap algorithm filters outliers based on displacement consistency
- **Multi-modal Visualization**: 2D overlays, depth maps, and 3D line reconstructions
- **Interactive Image Mode**: Step through images with keyboard controls
- **Flexible Configuration**: Easily toggle features (depth, matching, visualization)
- **High Performance**: GPU-accelerated depth estimation, TFLite-optimized line detection

## Project Structure

```
3D-Object-SLAM/
├── run_SLAM.py                          # Main SLAM pipeline
├── README.md                            # This file
│
├── M-LSD/
│   └── MLSD.py                          # Line segment detection
│
├── LBD/
│   ├── lbd.py                           # Line Band Descriptor implementation
│   └── lbd_optimized.py                 # Optimized version
│
├── calibration/
│   └── Camera-Calibration.py            # Camera intrinsic calibration
│
├── utils/
│   ├── depth_utils.py                   # Depth model initialization & inference
│   ├── geometry_3d.py                   # 2D→3D back-projection
│   └── viz_3d.py                        # 3D visualization utilities
│
├── depth_anything_v2/                   # Depth model architecture
│   ├── dinov2.py                        # Vision Transformer backbone
│   └── dpt.py                           # DPT decoder
│
├── tflite_models/
│   └── M-LSD_512_large_fp32.tflite     # Line detection model
│
├── depth_anything_v2_vitb.pth           # Depth model checkpoint (390 MB)
│
└── output/                              # Calibration results and visualizations
    ├── camera_matrix.txt
    ├── distortion_coefficients.txt
    └── calibration_data.pkl
```

## Installation

### Requirements

```bash
pip install opencv-python
pip install tensorflow
pip install torch torchvision
pip install numpy
pip install matplotlib
pip install gradio==4.29.0
pip install gradio_imageslider
```

Optional for HEIC/HEIF support:
```bash
pip install pillow-heif
```

### Model Files

1. **M-LSD Model**: Place `M-LSD_512_large_fp32.tflite` in `tflite_models/` directory
2. **Depth Anything V2 Model**: Download `depth_anything_v2_vitb.pth` (390 MB) to project root

## Usage

### 1. Camera Calibration (First-time Setup)

Before running SLAM, calibrate your camera using chessboard images:

```bash
python calibration/Camera-Calibration.py
```

**Requirements**:
- Images of 9x6 chessboard pattern (inner corners)
- Square size: 2.45 cm (or adjust in script)
- Multiple images from different angles/positions

**Output**:
- `output/camera_matrix.txt` - 3x3 intrinsic matrix
- `output/distortion_coefficients.txt` - Distortion coefficients
- `output/calibration_data.pkl` - Complete calibration data
- `output/corners_*.jpg` - Visualization of detected corners

### 2. Running the SLAM Pipeline

Edit configuration in [run_SLAM.py](run_SLAM.py):

```python
# Choose mode: 'video' or 'images'
INPUT_MODE = 'images'

# For video mode
VIDEO_PATH = 'path/to/your/video.mp4'

# For images mode
IMAGE_FOLDER = 'path/to/image/folder'

# Feature toggles
ENABLE_DEPTH_ESTIMATION = True
ENABLE_LINE_MATCHING = True
ENABLE_3D_VISUALIZATION = True
ENABLE_DEPTH_VISUALIZATION = True

# Detection parameters
SCORE_THR = 0.3   # Line confidence threshold (0.0-1.0)
DIST_THR = 20.0   # Distance threshold for M-LSD
```

Run the pipeline:
```bash
python run_SLAM.py
```

**Interactive Controls (Image Mode)**:
- **Space**: Next frame
- **'b'**: Previous frame
- **'q'**: Quit

### 3. Analysis Utilities

Test depth estimation and histogram:
```bash
python test_depth_histogram.py
```

Compare SelMap filtering effectiveness:
```bash
python test_selmap_comparison.py
```

Analyze M-LSD threshold sensitivity:
```bash
python test_threshold_sensitivity.py
```

## Pipeline Workflow

```
INPUT: Video or Image Sequence
    ↓
[1] LOAD CALIBRATION
    ├─ camera_matrix.txt
    └─ distortion_coefficients.txt
    ↓
[2] INITIALIZE MODELS
    ├─ M-LSD (TFLite)
    └─ Depth Anything V2 (PyTorch)
    ↓
[3] FRAME PROCESSING LOOP
    │
    ├─→ 2D LINE DETECTION (M-LSD)
    │   └─ Output: Lines [x1, y1, x2, y2]
    │
    ├─→ DEPTH ESTIMATION (Depth-V2)
    │   └─ Output: Depth map (H, W)
    │
    ├─→ 3D RECONSTRUCTION
    │   └─ Backproject lines using depth + intrinsics
    │
    ├─→ LINE MATCHING (LBD)
    │   ├─ Compute descriptors
    │   ├─ Match with previous frame
    │   └─ Filter with SelMap algorithm
    │
    └─→ VISUALIZATION
        ├─ 2D lines (red=unmatched, green=matched)
        ├─ Depth map (TURBO colormap)
        └─ 3D line reconstruction
    ↓
OUTPUT: Annotated video/images + 3D data
```

## Key Components

### M-LSD (Line Segment Detection)

**Location**: [M-LSD/MLSD.py](M-LSD/MLSD.py)

Detects line segments in images using a TensorFlow Lite deep learning model.

**Key Functions**:
- `pred_lines(image, interpreter, score_thr, dist_thr)` - Detect 2D lines
- `pred_squares(image, interpreter)` - Detect rectangular patterns
- `plot_threshold_sensitivity()` - Analyze detection parameters

**Parameters**:
- `score_thr`: Confidence threshold (default: 0.3)
- `dist_thr`: Distance threshold (default: 20.0)

### Depth Anything V2

**Location**: [utils/depth_utils.py](utils/depth_utils.py)

Monocular depth estimation using Vision Transformer architecture.

**Key Functions**:
- `initialize_depth_model(encoder='vitb')` - Load model (vits/vitb/vitl)
- `estimate_depth(image, model, device)` - Predict depth map
- `visualize_depth_map(depth_map)` - Convert to colored visualization
- `plot_depth_histogram(depth_map, title)` - Statistical analysis

**Model Sizes**:
- `vits`: 24.8M parameters (smallest, fastest)
- `vitb`: 97.5M parameters (balanced)
- `vitl`: 335.3M parameters (largest, most accurate)

### Line Band Descriptors (LBD)

**Location**: [LBD/lbd.py](LBD/lbd.py)

Matches line segments across frames using gradient-based descriptors.

**Components**:
- `LineDescriptor` - Computes descriptors for line segments
- `LineMatcher` - Matches descriptors between frames
- `visualize_matches()` - Visualizes matched lines

**Algorithm**:
1. Divide each line into perpendicular bands
2. Compute gradient descriptors for each band
3. Match using descriptor distance + geometric constraints

### SelMap Outlier Filtering

**Location**: [run_SLAM.py:selmap_filter_lines()](run_SLAM.py)

Robust filtering based on displacement consistency.

**Process**:
1. Compute displacement vectors between matched line centers
2. Build histograms of magnitudes and angles
3. Identify modal (most common) displacement
4. Filter matches as inliers/outliers

### 3D Reconstruction

**Location**: [utils/geometry_3d.py](utils/geometry_3d.py)

Back-projects 2D lines to 3D using pinhole camera model.

**Key Functions**:
- `backproject_point_to_3d(x, y, depth, camera_matrix, depth_scale)` - Single point
- `backproject_lines_to_3d(lines_2d, depth_map, camera_matrix, depth_scale)` - Line segments
- `compute_3d_line_length()` - Euclidean distance in 3D
- `compute_3d_line_midpoint()` - Line center in 3D

**Coordinate Frame**:
- Origin: Camera center
- X-axis: Right
- Y-axis: Down
- Z-axis: Forward (into scene)

## Camera Calibration Details

### Calibration Process

The calibration script uses OpenCV's chessboard detection:

1. **Pattern**: 9x6 inner corners, 2.45 cm squares
2. **Detection**:
   - Images downscaled for faster processing
   - Subpixel corner refinement
   - Multiple views required for accuracy
3. **Optimization**: Computes intrinsics and distortion coefficients
4. **Validation**: Generates reprojection error visualization

### Calibration Parameters

Edit in [calibration/Camera-Calibration.py](calibration/Camera-Calibration.py):

```python
CHESSBOARD_SIZE = (9, 6)       # Inner corners
SQUARE_SIZE = 2.45             # cm
MAX_IMAGE_SIZE = 3000          # Downscale for detection
```

### Output Format

**camera_matrix.txt**:
```
fx  0   cx
0   fy  cy
0   0   1
```
- `fx, fy`: Focal lengths (pixels)
- `cx, cy`: Principal point (image center)

**distortion_coefficients.txt**:
```
[k1, k2, p1, p2, k3]
```
- `k1, k2, k3`: Radial distortion
- `p1, p2`: Tangential distortion

## Configuration Parameters

### Detection Thresholds

- `SCORE_THR`: Line confidence (0.0-1.0, default: 0.3)
  - Lower: More lines detected (more noise)
  - Higher: Fewer, higher-quality lines

- `DIST_THR`: Junction distance threshold (default: 20.0)
  - Controls line segment merging

### Depth Parameters

- `DEPTH_SCALE`: Converts relative depth to metric-like scale (default: 10.0)
- Depth values are relative; scale adjusts visualization/3D coordinates

### SelMap Filtering

- `MAGNITUDE_TOLERANCE`: Displacement magnitude threshold
- `ANGLE_TOLERANCE`: Displacement angle threshold (degrees)

## Performance Notes

- **GPU Recommended**: Depth Anything V2 runs much faster on CUDA-enabled GPU
- **Image Downscaling**: Large images automatically downscaled to ≤1280px for processing
- **TFLite Optimization**: M-LSD uses efficient TensorFlow Lite inference
- **Batch Processing**: Video mode processes frames sequentially without loading all into memory

## Troubleshooting

### Common Issues

1. **Model files not found**:
   - Ensure `M-LSD_512_large_fp32.tflite` is in `tflite_models/`
   - Download `depth_anything_v2_vitb.pth` to project root

2. **Camera calibration fails**:
   - Verify chessboard pattern is 9x6 inner corners
   - Use at least 10-15 images from different angles
   - Ensure good lighting and focus

3. **Depth estimation disabled**:
   - Check CUDA availability for GPU: `torch.cuda.is_available()`
   - Verify model checkpoint exists and is not corrupted

4. **Few lines detected**:
   - Lower `SCORE_THR` to detect more lines
   - Check image quality and contrast

5. **Too many false matches**:
   - Enable SelMap filtering
   - Increase LBD descriptor matching threshold

## Output Files

- `mlsd_result.mp4` - Annotated video with detected/matched lines
- `mlsd_results_images/` - Individual frame outputs
- `output/depth_histograms/` - Depth distribution plots
- `output/selmap_comparison/` - Match filtering comparisons
- `output/reprojection_errors.png` - Calibration quality visualization

## Technical Details

### Depth Anything V2 Architecture

- **Encoder**: DINOv2 Vision Transformer (ViT)
- **Decoder**: Dense Prediction Transformer (DPT)
- **Input**: Any resolution (auto-resized)
- **Output**: Relative depth map (same resolution as input)

### M-LSD Architecture

- **Input**: 512x512 image + confidence channel
- **Output**:
  - Junction points with scores
  - Displacement vectors for line endpoints
- **Post-processing**: Non-maximum suppression, line assembly

### Coordinate Systems

- **Image**: Origin top-left, x-right, y-down
- **Camera 3D**: Origin at camera, X-right, Y-down, Z-forward
- **Depth**: Positive values indicate distance from camera

## References

- **M-LSD**: [Mobile Line Segment Detection](https://github.com/navervision/mlsd)
- **Depth Anything V2**: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **LBD**: Zhang, L., & Koch, R. (2013). An efficient and robust line segment matching approach based on LBD descriptor

## License

Please refer to individual component licenses:
- M-LSD: Apache 2.0
- Depth Anything V2: Apache 2.0

## Contributing

This is a research project. For issues or improvements, please create a GitHub issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System configuration (OS, Python version, GPU)

## Acknowledgments

This project integrates several state-of-the-art computer vision models and techniques:
- Naver Vision Lab (M-LSD)
- Depth Anything V2 team
- OpenCV community
- PyTorch and TensorFlow 
