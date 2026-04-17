"""
Loop Closure Detection for Line-Based Visual SLAM

Bag-of-Words (BoW) approach using LBD line descriptors.
Implements visual vocabulary training, TF-IDF weighted BoW vectors,
and database querying for loop closure candidate detection.

Theory:
    1. Train a visual vocabulary by clustering LBD descriptors with k-means
    2. For each frame, quantize its LBD descriptors against the vocabulary
       to produce a sparse BoW histogram (word frequencies)
    3. Weight the histogram with TF-IDF to emphasise distinctive words
    4. Compare BoW vectors between frames using L1-norm scoring
    5. Detect loop closures when score exceeds threshold and temporal
       gap is large enough

References:
    - Galvez-Lopez & Tardos, "Bags of Binary Words for Fast Place
      Recognition in Image Sequences", IEEE T-RO 2012
    - Mur-Artal et al., "ORB-SLAM2", IEEE T-RO 2017, Sec. IV-E

Usage:
    # --- Offline: train vocabulary from a set of descriptors ---
    vocab = BoWVocabulary(k=64)
    vocab.train(all_descriptors_list)  # list of (N_i, 28) arrays
    vocab.save("lbd_vocabulary.npz")

    # --- Online: detect loop closures during SLAM ---
    lcd = LoopClosureDetector(vocab, min_loop_gap=20, score_threshold=0.3)
    for frame_idx, descriptors in enumerate(frame_descriptors):
        candidates = lcd.add_and_query(descriptors, frame_idx)
        if candidates:
            print(f"Loop closure: frame {frame_idx} <-> frame {candidates[0][0]}")
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings


# ---------------------------------------------------------------------------
#  Visual Vocabulary (flat k-means)
# ---------------------------------------------------------------------------

class BoWVocabulary:
    """
    Visual vocabulary for Bag-of-Words representation.

    Clusters LBD descriptors into *k* visual words using k-means.
    Each word centre lives in the same 28-D float space as the raw
    descriptors, so nearest-word lookup is a simple L2 distance.

    Attributes:
        k: Number of visual words (cluster centres).
        centres: (k, D) array of cluster centres after training.
        idf: (k,) array of inverse-document-frequency weights.
    """

    def __init__(self, k: int = 64):
        """
        Args:
            k: Number of visual words. 64-128 is sensible for 28-D LBD.
               Larger values give finer discrimination but need more
               training data and are slower to query.
        """
        self.k = k
        self.centres: Optional[np.ndarray] = None  # (k, D)
        self.idf: Optional[np.ndarray] = None      # (k,)
        self._trained = False

    # ----- training --------------------------------------------------------

    def train(self,
              descriptor_sets: List[np.ndarray],
              max_iter: int = 100,
              seed: int = 42) -> None:
        """
        Train the vocabulary on a collection of descriptor sets.

        Args:
            descriptor_sets: List of arrays, each (N_i, D).  One array
                per training image / frame.
            max_iter: Maximum k-means iterations.
            seed: Random seed for reproducibility.
        """
        # Stack all descriptors into a single (N_total, D) matrix
        all_desc = np.vstack([d for d in descriptor_sets if len(d) > 0])
        if len(all_desc) < self.k:
            raise ValueError(
                f"Need at least k={self.k} descriptors to train, "
                f"got {len(all_desc)}"
            )

        print(f"[Vocabulary] Training k={self.k} words from "
              f"{len(all_desc)} descriptors ({len(descriptor_sets)} frames)...")

        self.centres = self._kmeans(all_desc.astype(np.float64),
                                    self.k, max_iter, seed)

        # Compute IDF from training data
        self._compute_idf(descriptor_sets)
        self._trained = True
        print(f"[Vocabulary] Training complete.")

    def _kmeans(self, data: np.ndarray, k: int,
                max_iter: int, seed: int) -> np.ndarray:
        """Simple k-means (no sklearn dependency required)."""
        rng = np.random.RandomState(seed)
        n = len(data)

        # k-means++ initialisation
        centres = np.empty((k, data.shape[1]), dtype=np.float64)
        centres[0] = data[rng.randint(n)]
        for i in range(1, k):
            dists = np.min(
                np.linalg.norm(data[:, None, :] - centres[None, :i, :],
                               axis=2),
                axis=1
            )
            probs = dists ** 2
            probs /= probs.sum()
            centres[i] = data[rng.choice(n, p=probs)]

        # Iterate
        for iteration in range(max_iter):
            # Assign to nearest centre
            # Process in chunks to avoid huge memory for large datasets
            labels = self._assign_labels(data, centres)

            # Update centres
            new_centres = np.empty_like(centres)
            for j in range(k):
                members = data[labels == j]
                if len(members) == 0:
                    # Re-initialise dead cluster
                    new_centres[j] = data[rng.randint(n)]
                else:
                    new_centres[j] = members.mean(axis=0)

            shift = np.linalg.norm(new_centres - centres)
            centres = new_centres
            if shift < 1e-6:
                print(f"  k-means converged at iteration {iteration + 1}")
                break

        return centres

    @staticmethod
    def _assign_labels(data: np.ndarray,
                       centres: np.ndarray,
                       chunk_size: int = 5000) -> np.ndarray:
        """Assign each descriptor to its nearest centre (chunked for memory)."""
        labels = np.empty(len(data), dtype=np.int32)
        for start in range(0, len(data), chunk_size):
            end = min(start + chunk_size, len(data))
            chunk = data[start:end]
            # (chunk, 1, D) - (1, k, D) -> (chunk, k)
            dists = np.linalg.norm(
                chunk[:, None, :] - centres[None, :, :], axis=2
            )
            labels[start:end] = np.argmin(dists, axis=1)
        return labels

    def _compute_idf(self, descriptor_sets: List[np.ndarray]) -> None:
        """Compute inverse document frequency from training images."""
        n_images = len(descriptor_sets)
        word_doc_count = np.zeros(self.k, dtype=np.float64)

        for descs in descriptor_sets:
            if len(descs) == 0:
                continue
            labels = self._assign_labels(
                descs.astype(np.float64), self.centres
            )
            unique_words = np.unique(labels)
            word_doc_count[unique_words] += 1

        # IDF = log(N / n_i), with smoothing to avoid log(0)
        self.idf = np.log((n_images + 1) / (word_doc_count + 1))

    # ----- inference -------------------------------------------------------

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Convert a set of descriptors into a TF-IDF weighted BoW vector.

        Args:
            descriptors: (N, D) array of LBD descriptors for one frame.

        Returns:
            bow: (k,) sparse-ish vector of TF-IDF weights.
        """
        if not self._trained:
            raise RuntimeError("Vocabulary not trained. Call train() first.")
        if len(descriptors) == 0:
            return np.zeros(self.k, dtype=np.float64)

        labels = self._assign_labels(
            descriptors.astype(np.float64), self.centres
        )

        # Term frequency: fraction of descriptors assigned to each word
        tf = np.bincount(labels, minlength=self.k).astype(np.float64)
        tf /= (tf.sum() + 1e-12)

        # TF-IDF
        bow = tf * self.idf

        # L1-normalise so that score() gives values in [0, 1]
        norm = np.abs(bow).sum()
        if norm > 1e-12:
            bow /= norm

        return bow

    # ----- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        """Save vocabulary to .npz file."""
        if not self._trained:
            raise RuntimeError("Cannot save untrained vocabulary.")
        np.savez(path, centres=self.centres, idf=self.idf, k=self.k)
        print(f"[Vocabulary] Saved to {path}")

    def load(self, path: str) -> None:
        """Load vocabulary from .npz file."""
        data = np.load(path)
        self.centres = data['centres']
        self.idf = data['idf']
        self.k = int(data['k'])
        self._trained = True
        print(f"[Vocabulary] Loaded k={self.k} from {path}")

    @classmethod
    def from_file(cls, path: str) -> 'BoWVocabulary':
        """Factory: create and load a vocabulary from file."""
        vocab = cls()
        vocab.load(path)
        return vocab


# ---------------------------------------------------------------------------
#  BoW Scoring
# ---------------------------------------------------------------------------

def bow_score_l1(bow1: np.ndarray, bow2: np.ndarray) -> float:
    """
    Compute similarity between two BoW vectors using L1-based scoring.

    This is the same scoring function used by DBoW2/DBoW3 (L1_NORM mode):
        score = 1 - 0.5 * ||v1 - v2||_1

    Both vectors are assumed to be L1-normalised, so the score lies in [0, 1]
    where 1 = identical and 0 = maximally different.

    Args:
        bow1, bow2: L1-normalised BoW vectors of the same length.

    Returns:
        Similarity score in [0, 1].
    """
    return 1.0 - 0.5 * np.abs(bow1 - bow2).sum()


# ---------------------------------------------------------------------------
#  Loop Closure Database & Detector
# ---------------------------------------------------------------------------

class LoopClosureDetector:
    """
    Online loop closure detection for visual SLAM.

    Maintains a database of BoW vectors (one per keyframe) and queries
    incoming frames against all previous entries, subject to a temporal
    gap constraint.

    Attributes:
        vocab: The trained BoWVocabulary.
        min_loop_gap: Minimum frame index gap to consider a loop closure.
        score_threshold: Minimum BoW similarity to report a candidate.
        database: Dict mapping frame_idx -> bow vector.
    """

    def __init__(self,
                 vocab: BoWVocabulary,
                 min_loop_gap: int = 20,
                 score_threshold: float = 0.7,
                 max_candidates: int = 5):
        """
        Args:
            vocab: Trained BoWVocabulary instance.
            min_loop_gap: Don't match frames closer than this many indices
                apart (avoids trivially matching sequential frames).
            score_threshold: Minimum similarity score to consider a match
                as a loop closure candidate. Tune this per environment:
                structured scenes (hallways) can use higher thresholds;
                cluttered scenes need lower ones.
            max_candidates: Maximum number of candidates to return per query.
        """
        self.vocab = vocab
        self.min_loop_gap = min_loop_gap
        self.score_threshold = score_threshold
        self.max_candidates = max_candidates

        self.database: Dict[int, np.ndarray] = {}
        self.frame_indices: List[int] = []

        # For analysis / visualisation
        self.all_scores: List[List[Tuple[int, float]]] = []

    def add_and_query(self,
                      descriptors: np.ndarray,
                      frame_idx: int
                      ) -> List[Tuple[int, float]]:
        """
        Add a frame to the database and query for loop closure candidates.

        This is the main entry point called once per SLAM frame.

        Args:
            descriptors: (N, D) LBD descriptors for this frame.
            frame_idx: Current frame index.

        Returns:
            candidates: List of (matched_frame_idx, score) tuples,
                sorted by score descending.  Empty if no loop closure
                detected.
        """
        bow = self.vocab.transform(descriptors)

        # Query against existing database entries
        candidates = []
        for db_idx, db_bow in self.database.items():
            # Temporal gap check
            if abs(frame_idx - db_idx) < self.min_loop_gap:
                continue

            score = bow_score_l1(bow, db_bow)
            if score >= self.score_threshold:
                candidates.append((db_idx, score))

        # Sort by score descending and keep top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:self.max_candidates]

        # Store scores for analysis
        self.all_scores.append(candidates)

        # Add current frame to database
        self.database[frame_idx] = bow
        self.frame_indices.append(frame_idx)

        return candidates

    def query_only(self,
                   descriptors: np.ndarray,
                   frame_idx: int
                   ) -> List[Tuple[int, float]]:
        """
        Query without adding to database (useful for testing).
        """
        bow = self.vocab.transform(descriptors)
        candidates = []
        for db_idx, db_bow in self.database.items():
            if abs(frame_idx - db_idx) < self.min_loop_gap:
                continue
            score = bow_score_l1(bow, db_bow)
            if score >= self.score_threshold:
                candidates.append((db_idx, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.max_candidates]

    def compute_pairwise_scores(self) -> np.ndarray:
        """
        Compute the full confusion matrix of pairwise BoW scores.

        Returns:
            scores: (N, N) symmetric matrix where N = number of frames
                in the database.  Entry (i, j) is the BoW similarity
                between frames i and j.  Useful for visualisation
                (like the confusion matrix from nicolov's repo).
        """
        indices = sorted(self.database.keys())
        n = len(indices)
        scores = np.zeros((n, n), dtype=np.float64)

        bows = [self.database[idx] for idx in indices]

        for i in range(n):
            scores[i, i] = 1.0  # Self-similarity
            for j in range(i + 1, n):
                s = bow_score_l1(bows[i], bows[j])
                scores[i, j] = s
                scores[j, i] = s

        return scores

    @property
    def size(self) -> int:
        return len(self.database)


# ---------------------------------------------------------------------------
#  Convenience: train vocabulary from a SLAM run
# ---------------------------------------------------------------------------

def train_vocabulary_from_descriptors(
        descriptor_sets: List[np.ndarray],
        k: int = 64,
        save_path: Optional[str] = None
) -> BoWVocabulary:
    """
    One-shot vocabulary training from a list of per-frame descriptors.

    For best results, collect descriptors from a representative run
    (or multiple runs) covering the environments you expect to revisit.

    Args:
        descriptor_sets: List of (N_i, D) descriptor arrays, one per frame.
        k: Number of visual words.
        save_path: If given, save the vocabulary here.

    Returns:
        Trained BoWVocabulary.
    """
    vocab = BoWVocabulary(k=k)
    vocab.train(descriptor_sets)
    if save_path:
        vocab.save(save_path)
    return vocab
