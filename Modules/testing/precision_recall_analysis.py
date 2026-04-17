"""
Generate Precision-Recall curve for loop closure detection
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from runSLAM2 import LineSLAM


def compute_ground_truth_closures(positions, distance_threshold=0.5):
    """
    Determine which frame pairs are truly loop closures based on
    distance between camera positions.
    
    Args:
        positions: (N, 3) camera positions
        distance_threshold: max distance to consider a loop closure
    
    Returns:
        closure_matrix: (N, N) binary matrix, 1 if true loop closure
    """
    N = len(positions)
    closure_matrix = np.zeros((N, N), dtype=bool)
    
    for i in range(N):
        for j in range(i):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < distance_threshold:
                closure_matrix[i, j] = True
                closure_matrix[j, i] = True
    
    return closure_matrix


def compute_precision_recall(similarity_matrix, gt_closures, min_loop_gap,
                             thresholds=None):
    """
    Compute precision and recall at different similarity thresholds
    
    Args:
        similarity_matrix: (N, N) pairwise similarities
        gt_closures: (N, N) binary ground truth
        min_loop_gap: minimum temporal gap
        thresholds: list of thresholds to evaluate
    
    Returns:
        precisions, recalls, thresholds_used
    """
    
    if thresholds is None:
        thresholds = np.linspace(0, 1, 50)
    
    precisions = []
    recalls = []
    
    N = len(similarity_matrix)
    
    # Apply temporal gap constraint to ground truth
    gt_valid = gt_closures.copy()
    for i in range(N):
        for j in range(N):
            if abs(i - j) < min_loop_gap:
                gt_valid[i, j] = False
    
    n_true_closures = np.sum(gt_valid)
    
    if n_true_closures == 0:
        print("WARNING: No ground truth loop closures with temporal constraint!")
        return [], [], []
    
    for threshold in thresholds:
        # Predictions: similarity > threshold AND temporal gap satisfied
        predictions = (similarity_matrix > threshold).astype(bool)
        for i in range(N):
            for j in range(N):
                if abs(i - j) < min_loop_gap:
                    predictions[i, j] = False
        
        # True positives, false positives, false negatives
        tp = np.sum(predictions & gt_valid)
        fp = np.sum(predictions & ~gt_valid)
        fn = np.sum(~predictions & gt_valid)
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls), thresholds


def plot_precision_recall_curve(precisions, recalls, save_path="precision_recall.png"):
    """Plot precision-recall curve"""
    
    # Calculate Area Under Curve (AUC)
    auc = np.trapz(precisions, recalls)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(recalls, precisions, 'b-', linewidth=2.5, label=f'Loop Closure (AUC={auc:.3f})')
    ax.fill_between(recalls, precisions, alpha=0.2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve for Loop Closure Detection', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_similarity_vs_distance(positions, similarity_matrix, save_path="similarity_vs_distance.png"):
    """Plot similarity score vs. actual camera distance"""
    
    distances = []
    similarities = []
    
    N = len(positions)
    for i in range(N):
        for j in range(i):
            dist = np.linalg.norm(positions[i] - positions[j])
            sim = similarity_matrix[i, j]
            distances.append(dist)
            similarities.append(sim)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    scatter = ax.scatter(distances, similarities, alpha=0.5, s=20, c=distances, cmap='viridis')
    
    ax.set_xlabel('Camera Distance (meters)', fontsize=12)
    ax.set_ylabel('BoW Similarity Score', fontsize=12)
    ax.set_title('Loop Closure Candidate: Similarity vs. Distance', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Distance (m)')
    
    # Add trend line
    z = np.polyfit(distances, similarities, 2)
    p = np.poly1d(z)
    x_line = np.linspace(min(distances), max(distances), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend (degree 2 poly)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python precision_recall_analysis.py <path_to_record3d_folder> [max_frames]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    mlsd_model = "tflite_models/M-LSD_512_large_fp32.tflite"
    vocab_path = "lbd_vocab_k64.npz"
    
    print("Running SLAM with loop closure...")
    slam = LineSLAM(mlsd_model, vocab_path=vocab_path)
    slam.run(data_path, max_frames=max_frames, use_gt_pose=True)
    
    print("\n" + "="*60)
    print("PRECISION-RECALL ANALYSIS")
    print("="*60)
    
    # Extract positions
    positions = np.array([pose[:3, 3] for pose in slam.gt_poses])
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    detections = slam.lcd.database
    sim_matrix = np.zeros((len(detections), len(detections)))
    for i in range(len(detections)):
        for j in range(len(detections)):
            sim_matrix[i, j] = 1.0 - (np.linalg.norm(detections[i] - detections[j], ord=1) / 2.0)
    
    # Compute ground truth (camera proximity)
    print("Computing ground truth loop closures...")
    gt_closures = compute_ground_truth_closures(positions, distance_threshold=0.3)
    n_true = np.sum(gt_closures) // 2  # Divide by 2 because matrix is symmetric
    print(f"  Found {n_true} ground truth loop closures (distance < 0.3m)")
    
    # Compute precision-recall
    print("Computing precision-recall curve...")
    precisions, recalls, thresholds = compute_precision_recall(
        sim_matrix, gt_closures, 
        min_loop_gap=slam.lcd.min_loop_gap
    )
    
    # Print statistics
    print(f"\nResults:")
    print(f"  Max precision: {np.max(precisions):.3f}")
    print(f"  Max recall: {np.max(recalls):.3f}")
    
    # Find operating point (threshold where precision ≈ recall)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    print(f"  Best F1 score: {f1_scores[best_idx]:.3f} at threshold={thresholds[best_idx]:.3f}")
    
    # Visualize
    plot_precision_recall_curve(precisions, recalls)
    plot_similarity_vs_distance(positions, sim_matrix)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()