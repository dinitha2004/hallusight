"""
modules/distribution_shift.py
================================
HalluShift — Distribution Shift Analyzer

WHAT THIS DOES (simple explanation):
  When LLaMA generates truthful content, its hidden states change smoothly.
  When it hallucinates, the hidden states "jump" abnormally.

  This module detects those jumps using two math tools:
  1. Cosine Similarity  — measures direction change between vectors
  2. Wasserstein Distance — measures how different two distributions are

HOW IT WORKS:
  For each token, compare its hidden state to the average of the
  previous N tokens (rolling window).
  Big difference = suspicious = higher hallucination risk.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from typing import List


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Measures how similar two vectors are in direction.
    Range: -1.0 (opposite) to 1.0 (identical)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def wasserstein_shift(a: np.ndarray, b: np.ndarray, n_dims: int = 100) -> float:
    """
    Wasserstein distance between two hidden state vectors.
    Uses first n_dims dimensions for speed.
    Higher = more different = more suspicious.
    """
    dims = min(n_dims, len(a), len(b))
    return float(wasserstein_distance(a[:dims], b[:dims]))


def distribution_shift_score(
    hidden_states: List[np.ndarray],
    window: int = 5,
    cosine_weight: float = 0.5,
    wasserstein_weight: float = 0.5,
    wasserstein_dims: int = 100,
    wasserstein_scale: float = 10.0,
) -> List[float]:
    """
    Compute distribution shift score for each token.

    Args:
        hidden_states:  List of hidden state vectors from generation
        window:         How many previous tokens to compare against (default 5)

    Returns:
        List of floats 0.0 to 1.0 per token
        0.0 = no shift (likely truthful)
        1.0 = maximum shift (likely hallucinated)
    """
    n = len(hidden_states)
    scores = []

    for i in range(n):
        # Not enough history yet
        if i < window:
            scores.append(0.0)
            continue

        current = hidden_states[i]
        mean_past = np.mean(hidden_states[i - window: i], axis=0)

        # Component 1: Cosine shift (0 = same direction, 1 = opposite)
        cos_sim = cosine_similarity(current, mean_past)
        cosine_shift = (1.0 - cos_sim) / 2.0

        # Component 2: Wasserstein shift (normalised to 0-1)
        wass = wasserstein_shift(current, mean_past, wasserstein_dims)
        wass_norm = min(wass / wasserstein_scale, 1.0)

        # Combined weighted score
        combined = cosine_weight * cosine_shift + wasserstein_weight * wass_norm
        scores.append(float(min(max(combined, 0.0), 1.0)))

    return scores
