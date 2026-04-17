import numpy as np
from pathlib import Path
from runSLAM2 import LineSLAM
from loop_closure import train_vocabulary_from_descriptors 

def main():
    data_path = "C:/Users/hamee/Downloads/videos/video1"

    print("=" * 60) 
    print("STEP 1: Running SLAM to collect descriptors...")
    print("=" * 60)
    
    slam = LineSLAM("tflite_models/M-LSD_512_large_fp32.tflite")
    slam.run(data_path, max_frames=100, use_gt_pose=True)
    
    all_descriptors = slam.get_all_descriptors()
    print(f"\nCollected {len(all_descriptors)} frames")
    
    total_lines = sum(len(d) for d in all_descriptors)
    print(f"Total lines: {total_lines}")
    
    print("\n" + "=" * 60)
    print("STEP 2: Training vocabulary...")
    print("=" * 60)
    
    vocab = train_vocabulary_from_descriptors(
        all_descriptors,
        k=64,
        save_path="lbd_vocab_k64.npz"
    )
    
    print(f"\nVocabulary saved to: lbd_vocab_k64.npz")
    print(f"  k={vocab.k} visual words")
    print(f"  Descriptor dimension: {vocab.centres.shape[1]}")
 
if __name__ == "__main__":
     main()
