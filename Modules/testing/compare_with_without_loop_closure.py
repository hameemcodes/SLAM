"""
Compare SLAM performance with and without loop closure
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from runSLAM2 import LineSLAM


def run_slam_variants(data_path, max_frames, mlsd_model):
    """Run SLAM with and without loop closure"""
    
    print("="*60)
    print("RUN 1: WITHOUT Loop Closure")
    print("="*60)
    slam_without = LineSLAM(mlsd_model, vocab_path=None)
    slam_without.run(data_path, max_frames=max_frames, use_gt_pose=True)
    
    print("\n" + "="*60)
    print("RUN 2: WITH Loop Closure")
    print("="*60)
    slam_with = LineSLAM(mlsd_model, vocab_path="lbd_vocab_k64.npz")
    slam_with.run(data_path, max_frames=max_frames, use_gt_pose=True)
    
    return slam_without, slam_with


def plot_comparison(slam_without, slam_with, save_path="comparison.png"):
    """Plot side-by-side comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Number of map lines
    ax = axes[0, 0]
    frames = range(len(slam_without.gt_poses))
    map_lines_without = [len(slam_without.map)]  # Simplified - just final count
    map_lines_with = [len(slam_with.map)]
    ax.bar(['Without LC', 'With LC'], [map_lines_without[0], map_lines_with[0]], color=['blue', 'green'])
    ax.set_ylabel('Total 3D Lines in Map', fontsize=11)
    ax.set_title('Map Size Comparison', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Pose errors
    ax = axes[0, 1]
    if slam_without.rotation_errors and slam_with.rotation_errors:
        ax.plot(slam_without.rotation_errors, 'b-', label='Without LC', linewidth=2)
        ax.plot(slam_with.rotation_errors, 'g-', label='With LC', linewidth=2)
        ax.set_xlabel('Frame Index', fontsize=11)
        ax.set_ylabel('Rotation Error (degrees)', fontsize=11)
        ax.set_title('Pose Estimation Error Over Time', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Translation errors
    ax = axes[1, 0]
    if slam_without.translation_errors and slam_with.translation_errors:
        ax.plot(slam_without.translation_errors, 'b-', label='Without LC', linewidth=2)
        ax.plot(slam_with.translation_errors, 'g-', label='With LC', linewidth=2)
        ax.set_xlabel('Frame Index', fontsize=11)
        ax.set_ylabel('Translation Error (%)', fontsize=11)
        ax.set_title('Translation Error Over Time', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = "STATISTICS\n" + "="*40 + "\n\n"
    
    stats_text += "WITHOUT Loop Closure:\n"
    if slam_without.rotation_errors:
        stats_text += f"  Rotation Error: {np.mean(slam_without.rotation_errors):.2f}° ± {np.std(slam_without.rotation_errors):.2f}°\n"
        stats_text += f"  Translation Error: {np.mean(slam_without.translation_errors):.1f}% ± {np.std(slam_without.translation_errors):.1f}%\n"
    stats_text += f"  Map Lines: {len(slam_without.map)}\n\n"
    
    stats_text += "WITH Loop Closure:\n"
    if slam_with.rotation_errors:
        stats_text += f"  Rotation Error: {np.mean(slam_with.rotation_errors):.2f}° ± {np.std(slam_with.rotation_errors):.2f}°\n"
        stats_text += f"  Translation Error: {np.mean(slam_with.translation_errors):.1f}% ± {np.std(slam_with.translation_errors):.1f}%\n"
    stats_text += f"  Map Lines: {len(slam_with.map)}\n\n"
    
    # Improvements
    if slam_without.rotation_errors and slam_with.rotation_errors:
        rot_improvement = (np.mean(slam_without.rotation_errors) - np.mean(slam_with.rotation_errors)) / np.mean(slam_without.rotation_errors) * 100
        trans_improvement = (np.mean(slam_without.translation_errors) - np.mean(slam_with.translation_errors)) / np.mean(slam_without.translation_errors) * 100
        
        stats_text += "IMPROVEMENTS:\n"
        stats_text += f"  Rotation: {rot_improvement:+.1f}%\n"
        stats_text += f"  Translation: {trans_improvement:+.1f}%\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python compare_loop_closure.py <path_to_record3d_folder> [max_frames]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    mlsd_model = "tflite_models/M-LSD_512_large_fp32.tflite"
    vocab_path = "lbd_vocab_k64.npz"
    
    if not Path(vocab_path).exists():
        print(f"ERROR: Vocabulary not found. Run train_vocab.py first")
        sys.exit(1)
    
    slam_without, slam_with = run_slam_variants(data_path, max_frames, mlsd_model)
    plot_comparison(slam_without, slam_with)
    
    print("\n✓ Comparison complete!")


if __name__ == '__main__':
    main()