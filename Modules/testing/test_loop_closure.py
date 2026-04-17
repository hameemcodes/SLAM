"""
Test and demonstration script for loop_closure.py

Part 1: Unit tests with synthetic data (no SLAM dependencies)
Part 2: Integration example showing how to plug into LineSLAM

Run:  python test_loop_closure.py
"""

import numpy as np
import sys
import time


# =====================================================================
#  PART 1: Unit tests with synthetic descriptors
# =====================================================================

def make_synthetic_scene(n_lines: int = 30, desc_dim: int = 28,
                         noise: float = 0.05,
                         rng: np.random.RandomState = None) -> np.ndarray:
    """Generate a synthetic 'scene' as a set of LBD-like descriptors."""
    if rng is None:
        rng = np.random.RandomState(0)
    base = rng.randn(n_lines, desc_dim).astype(np.float32)
    return base + noise * rng.randn(n_lines, desc_dim).astype(np.float32)


def test_vocabulary_training():
    """Test that vocabulary trains and transforms produce valid BoW vectors."""
    from loop_closure import BoWVocabulary

    print("=" * 60)
    print("TEST: Vocabulary training")
    print("=" * 60)

    rng = np.random.RandomState(42)

    # Create 10 synthetic 'scenes' with 30 descriptors each
    scenes = [make_synthetic_scene(30, rng=rng) for _ in range(10)]

    vocab = BoWVocabulary(k=16)
    vocab.train(scenes, max_iter=50, seed=42)

    assert vocab.centres.shape == (16, 28), \
        f"Expected (16, 28), got {vocab.centres.shape}"
    assert vocab.idf.shape == (16,), \
        f"Expected (16,), got {vocab.idf.shape}"

    # Transform a scene into BoW
    bow = vocab.transform(scenes[0])
    assert bow.shape == (16,), f"Expected (16,), got {bow.shape}"
    assert abs(np.abs(bow).sum() - 1.0) < 1e-6, \
        f"BoW vector not L1-normalised: sum={np.abs(bow).sum()}"

    print(f"  Centres shape: {vocab.centres.shape}")
    print(f"  IDF range: [{vocab.idf.min():.2f}, {vocab.idf.max():.2f}]")
    print(f"  BoW non-zero entries: {np.count_nonzero(bow)}/{len(bow)}")
    print("  PASSED\n")
    return vocab, scenes


def test_bow_scoring(vocab, scenes):
    """Test that similar scenes get high scores, different ones get low."""
    from loop_closure import bow_score_l1

    print("=" * 60)
    print("TEST: BoW scoring")
    print("=" * 60)

    rng = np.random.RandomState(99)

    # Same scene with slight noise -> high score
    scene_a = scenes[0]
    scene_a_noisy = scene_a + 0.02 * rng.randn(*scene_a.shape).astype(np.float32)
    bow_a = vocab.transform(scene_a)
    bow_a_noisy = vocab.transform(scene_a_noisy)
    score_same = bow_score_l1(bow_a, bow_a_noisy)

    # Completely different scene -> low score
    scene_b = rng.randn(30, 28).astype(np.float32) * 10
    bow_b = vocab.transform(scene_b)
    score_diff = bow_score_l1(bow_a, bow_b)

    # Self-similarity = 1.0
    score_self = bow_score_l1(bow_a, bow_a)

    print(f"  Self-similarity:       {score_self:.4f} (expect ~1.0)")
    print(f"  Same scene + noise:    {score_same:.4f} (expect high)")
    print(f"  Different scene:       {score_diff:.4f} (expect low)")

    assert abs(score_self - 1.0) < 1e-6, "Self-similarity should be 1.0"
    assert score_same > score_diff, "Same scene should score higher"
    print("  PASSED\n")


def test_loop_closure_detection():
    """Simulate a trajectory that revisits a place and check detection."""
    from loop_closure import BoWVocabulary, LoopClosureDetector

    print("=" * 60)
    print("TEST: Loop closure detection (synthetic trajectory)")
    print("=" * 60)

    rng = np.random.RandomState(42)

    # Create 5 distinct 'places'
    places = [make_synthetic_scene(30, noise=0.01, rng=rng) for _ in range(5)]

    # Simulate trajectory: A B C D E D C B A
    # (revisits places in reverse -> loop closures expected)
    trajectory = [0, 1, 2, 3, 4, 3, 2, 1, 0]
    frame_descriptors = []
    for place_idx in trajectory:
        # Add observation noise to each visit
        noisy = places[place_idx] + 0.03 * rng.randn(30, 28).astype(np.float32)
        frame_descriptors.append(noisy)

    # Train vocabulary on all observations
    vocab = BoWVocabulary(k=16)
    vocab.train(frame_descriptors, max_iter=50, seed=42)

    # Run loop closure detection
    lcd = LoopClosureDetector(vocab, min_loop_gap=3, score_threshold=0.2)

    print(f"  Trajectory: {trajectory}")
    print(f"  min_loop_gap=3, score_threshold=0.2\n")

    detected_loops = []
    for frame_idx, descs in enumerate(frame_descriptors):
        candidates = lcd.add_and_query(descs, frame_idx)
        if candidates:
            best_match, best_score = candidates[0]
            detected_loops.append((frame_idx, best_match, best_score))
            print(f"  Frame {frame_idx} (place {trajectory[frame_idx]}) "
                  f"<-> Frame {best_match} (place {trajectory[best_match]}) "
                  f"score={best_score:.3f}")

    # We expect frames 5-8 to match frames 3-0 respectively
    # (same places visited in reverse)
    assert len(detected_loops) > 0, "Should detect at least one loop closure"

    # Check that at least one correct match was found
    correct = any(
        trajectory[loop[0]] == trajectory[loop[1]]
        for loop in detected_loops
    )
    assert correct, "At least one detected loop should match the same place"

    print(f"\n  Detected {len(detected_loops)} loop closures")
    print("  PASSED\n")


def test_pairwise_scores():
    """Test confusion matrix computation."""
    from loop_closure import BoWVocabulary, LoopClosureDetector

    print("=" * 60)
    print("TEST: Pairwise score matrix (confusion matrix)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    places = [make_synthetic_scene(30, noise=0.01, rng=rng) for _ in range(3)]
    trajectory = [0, 1, 2, 0, 1, 2]  # Visit 3 places twice

    frame_descriptors = []
    for place_idx in trajectory:
        noisy = places[place_idx] + 0.03 * rng.randn(30, 28).astype(np.float32)
        frame_descriptors.append(noisy)

    vocab = BoWVocabulary(k=16)
    vocab.train(frame_descriptors, max_iter=50, seed=42)

    lcd = LoopClosureDetector(vocab, min_loop_gap=2, score_threshold=0.1)
    for i, descs in enumerate(frame_descriptors):
        lcd.add_and_query(descs, i)

    scores = lcd.compute_pairwise_scores()
    assert scores.shape == (6, 6), f"Expected (6,6), got {scores.shape}"
    assert np.allclose(scores, scores.T), "Score matrix should be symmetric"
    assert np.allclose(np.diag(scores), 1.0), "Diagonal should be 1.0"

    # Same-place pairs should score higher than different-place pairs
    # Frame 0 <-> Frame 3 (both place 0), Frame 1 <-> Frame 4 (both place 1)
    same_place_scores = [scores[0, 3], scores[1, 4], scores[2, 5]]
    diff_place_scores = [scores[0, 1], scores[0, 2], scores[1, 2]]

    avg_same = np.mean(same_place_scores)
    avg_diff = np.mean(diff_place_scores)

    print(f"  Score matrix shape: {scores.shape}")
    print(f"  Avg same-place score:  {avg_same:.3f}")
    print(f"  Avg diff-place score:  {avg_diff:.3f}")
    assert avg_same > avg_diff, "Same-place pairs should score higher"
    print("  PASSED\n")


def test_save_load():
    """Test vocabulary serialisation."""
    from loop_closure import BoWVocabulary
    import tempfile, os

    print("=" * 60)
    print("TEST: Vocabulary save/load")
    print("=" * 60)

    rng = np.random.RandomState(42)
    scenes = [make_synthetic_scene(30, rng=rng) for _ in range(10)]

    vocab = BoWVocabulary(k=16)
    vocab.train(scenes, max_iter=30, seed=42)

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name
    try:
        vocab.save(path)
        vocab2 = BoWVocabulary.from_file(path)

        assert np.allclose(vocab.centres, vocab2.centres), "Centres mismatch"
        assert np.allclose(vocab.idf, vocab2.idf), "IDF mismatch"

        # Same transform results
        bow1 = vocab.transform(scenes[0])
        bow2 = vocab2.transform(scenes[0])
        assert np.allclose(bow1, bow2), "Transform results differ after load"
    finally:
        os.unlink(path)

    print("  PASSED\n")


def test_empty_descriptors():
    """Edge case: frames with no descriptors."""
    from loop_closure import BoWVocabulary, LoopClosureDetector

    print("=" * 60)
    print("TEST: Empty descriptor handling")
    print("=" * 60)

    rng = np.random.RandomState(42)
    scenes = [make_synthetic_scene(30, rng=rng) for _ in range(10)]

    vocab = BoWVocabulary(k=16)
    vocab.train(scenes, max_iter=30, seed=42)

    lcd = LoopClosureDetector(vocab, min_loop_gap=3, score_threshold=0.3)

    # Add a normal frame
    lcd.add_and_query(scenes[0], 0)

    # Add an empty frame (no lines detected)
    empty = np.array([]).reshape(0, 28).astype(np.float32)
    candidates = lcd.add_and_query(empty, 1)

    assert candidates == [], "Empty descriptors should return no candidates"
    print("  PASSED\n")


# =====================================================================
#  PART 2: Integration example (does not run - shows the pattern)
# =====================================================================

def print_integration_example():
    """Print how to integrate loop_closure.py into LineSLAM."""
    print("=" * 60)
    print("INTEGRATION GUIDE: Adding to LineSLAM")
    print("=" * 60)

    example = '''
    # In runSLAM2.py, add to imports:
    from loop_closure import BoWVocabulary, LoopClosureDetector, train_vocabulary_from_descriptors

    # ---- OPTION A: Two-pass (recommended for dissertation) ----
    # Pass 1: Run SLAM as normal, collect all descriptors
    # Pass 2: Train vocabulary, then re-run with loop closure

    # After a normal SLAM run, collect descriptors:
    all_descriptors = []  # populated during first run

    # Train vocabulary:
    vocab = train_vocabulary_from_descriptors(all_descriptors, k=64,
                                              save_path="lbd_vocab_k64.npz")

    # ---- OPTION B: Online (vocabulary pre-trained) ----
    # In LineSLAM.__init__():
    self.vocab = BoWVocabulary.from_file("lbd_vocab_k64.npz")
    self.lcd = LoopClosureDetector(self.vocab,
                                   min_loop_gap=20,
                                   score_threshold=0.3)

    # In LineSLAM.process_frame(), after computing descriptors (step 3):
    candidates = self.lcd.add_and_query(descriptors, frame_idx)
    if candidates:
        best_idx, best_score = candidates[0]
        print(f"  LOOP CLOSURE: frame {frame_idx} <-> frame {best_idx} "
              f"(score={best_score:.3f})")

    # ---- Visualisation (for dissertation figures) ----
    # After the full run:
    import matplotlib.pyplot as plt

    scores = slam.lcd.compute_pairwise_scores()
    plt.figure(figsize=(8, 8))
    plt.imshow(scores, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='BoW similarity')
    plt.xlabel('Frame index')
    plt.ylabel('Frame index')
    plt.title('Loop Closure Confusion Matrix')
    plt.savefig('loop_closure_matrix.png', dpi=150)
    '''
    print(example)


# =====================================================================
#  Main
# =====================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  LOOP CLOSURE MODULE - TEST SUITE")
    print("=" * 60 + "\n")

    t0 = time.time()

    vocab, scenes = test_vocabulary_training()
    test_bow_scoring(vocab, scenes)
    test_loop_closure_detection()
    test_pairwise_scores()
    test_save_load()
    test_empty_descriptors()

    elapsed = time.time() - t0

    print("=" * 60)
    print(f"ALL TESTS PASSED ({elapsed:.2f}s)")
    print("=" * 60 + "\n")

    print_integration_example()
