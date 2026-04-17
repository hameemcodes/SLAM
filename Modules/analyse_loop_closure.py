"""
Analyze loop closure detection performance
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from runSLAM2 import LineSLAM


def analyze_loop_closures(slam):
    """Extract and analyze loop closure candidates from detector"""
    
    if slam.lcd is None:
        print("Loop closure detector not initialized!")
        return None, None  # ← RETURN TWO VALUES (None, None)
    
    # Get all detected loop closures
    detections = slam.lcd.database  # List of (frame_idx, bow_vector)
    
    print(f"\n{'='*60}")
    print("LOOP CLOSURE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total frames processed: {len(detections)}")
    
    # Compute pairwise similarity matrix
    print("\nComputing pairwise similarity matrix...")
    similarity_matrix = np.zeros((len(detections), len(detections)))
    
    for i in range(len(detections)):
        bow_i = detections[i]
        for j in range(len(detections)):
            bow_j = detections[j]
            # L1 distance between BoW vectors
            similarity = 1.0 - (np.linalg.norm(bow_i - bow_j, ord=1) / 2.0)
            similarity_matrix[i, j] = np.clip(similarity, 0, 1)
    
    # Analyze detections
    min_gap = slam.lcd.min_loop_gap
    threshold = slam.lcd.score_threshold
    
    print(f"Detection threshold: {threshold:.3f}")
    print(f"Minimum temporal gap: {min_gap} frames")
    
    # Count valid detections (respecting temporal gap)
    valid_detections = 0
    for i in range(len(detections)):
        for j in range(i - min_gap):  # Only look back with gap
            if similarity_matrix[i, j] > threshold:
                valid_detections += 1
    
    print(f"Valid loop closures detected: {valid_detections}")
    
    # Statistics
    print(f"\nSimilarity statistics (all pairs):")
    print(f"  Mean: {np.mean(similarity_matrix):.4f}")
    print(f"  Median: {np.median(similarity_matrix):.4f}")
    print(f"  Min: {np.min(similarity_matrix):.4f}")
    print(f"  Max: {np.max(similarity_matrix):.4f}")
    print(f"  Std: {np.std(similarity_matrix):.4f}")
    
    # Self-similarity diagonal
    diagonal = np.diag(similarity_matrix)
    print(f"\nDiagonal (self-similarity):")
    print(f"  Mean: {np.mean(diagonal):.4f}")
    print(f"  Should be close to 1.0 ✓" if np.mean(diagonal) > 0.95 else "  WARNING: Low self-similarity!")
    
    return similarity_matrix, detections  # ← RETURN BOTH VALUES


def plot_similarity_heatmap(similarity_matrix, save_path="loop_closure_heatmap.png"):
    """Plot similarity matrix as heatmap"""
    
    if similarity_matrix is None:
        print("No similarity matrix to plot!")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(similarity_matrix, cmap='hot', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Frame Index', fontsize=12)
    ax.set_title('Loop Closure Similarity Matrix (BoW L1 Distance)', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, label='Similarity Score')
    
    # Add grid
    ax.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def plot_trajectory_with_closures(slam, similarity_matrix, save_path="trajectory_with_closures.png"):
    """Plot 3D trajectory and highlight loop closure pairs"""
    
    if similarity_matrix is None:
        print("No similarity matrix to plot trajectory!")
        return
    
    if len(slam.gt_poses) == 0:
        print("No poses to visualize!")
        return
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract trajectory
    positions = np.array([pose[:3, 3] for pose in slam.gt_poses])
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
           'b-', linewidth=2, label='Camera Trajectory')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
              c=range(len(positions)), cmap='viridis', s=20, alpha=0.6)
    
    # Highlight loop closures
    threshold = slam.lcd.score_threshold
    min_gap = slam.lcd.min_loop_gap
    
    loop_pairs = []
    for i in range(len(positions)):
        for j in range(max(0, i - min_gap)):
            if similarity_matrix[i, j] > threshold:
                loop_pairs.append((j, i))
                # Draw line between loop closure pair
                ax.plot([positions[j, 0], positions[i, 0]],
                       [positions[j, 1], positions[i, 1]],
                       [positions[j, 2], positions[i, 2]],
                       'r--', alpha=0.4, linewidth=1)
    
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_title(f'Camera Trajectory with {len(loop_pairs)} Loop Closures', fontsize=12)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyse_loop_closure.py <path_to_record3d_folder> [max_frames]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    mlsd_model = "tflite_models/M-LSD_512_large_fp32.tflite"
    vocab_path = "lbd_vocab_k64.npz"
    
    if not Path(mlsd_model).exists():
        print(f"ERROR: M-LSD model not found")
        sys.exit(1)
    
    if not Path(vocab_path).exists():
        print(f"ERROR: Vocabulary not found. Run train_vocab.py first")
        sys.exit(1)
    
    print("Running SLAM with loop closure...")
    slam = LineSLAM(mlsd_model, vocab_path=vocab_path)
    slam.run(data_path, max_frames=max_frames, use_gt_pose=True)
    
    # Analyze loop closures
    sim_matrix, detections = analyze_loop_closures(slam)  # ← NOW SAFELY UNPACKS
    
    # Visualize (with None checks)
    if sim_matrix is not None:
        plot_similarity_heatmap(sim_matrix)
        plot_trajectory_with_closures(slam, sim_matrix)
        print("\n✓ Analysis complete!")
    else:
        print("\n✗ Analysis failed - no loop closure data available")


if __name__ == '__main__':
    main()