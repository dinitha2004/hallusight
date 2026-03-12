"""
modules/feature_clipping.py
==============================
FC + TSV — Feature Clipping and Truthfulness Separator Vector

WHAT THIS DOES (simple explanation):

  Feature Clipping:
    Sometimes hidden state values are extremely large or small (outliers).
    These outliers can confuse our classifiers.
    We "clip" them — cut off anything too extreme.
    Like adjusting the volume — if it's too loud (distorted), turn it down.

  Truthfulness Separator Vector (TSV):
    Imagine drawing a line between two groups of people:
    Group A = truthful responses, Group B = hallucinated responses.
    TSV finds the direction that best separates these two groups.
    We then "subtract" that direction from hidden states to make the
    difference between truthful and hallucinated tokens more obvious.

WHY THIS HELPS:
  After clipping + TSV, our SEP probe and HalluShift detector
  work more accurately because the input data is cleaner.
"""

import numpy as np
from typing import List, Optional


def feature_clipping(hidden_state: np.ndarray, clip_val: float = 5.0) -> np.ndarray:
    """
    Clip hidden state values to [-clip_val, +clip_val].

    Why 5.0? For normalised hidden states, values beyond +-5 standard
    deviations are almost always noise or anomalies.

    Args:
        hidden_state: numpy array of shape (hidden_size,)
        clip_val:     maximum absolute value allowed (default 5.0)

    Returns:
        Clipped numpy array, same shape as input
    """
    return np.clip(hidden_state, -clip_val, clip_val)


def clip_all(hidden_states: List[np.ndarray], clip_val: float = 5.0) -> List[np.ndarray]:
    """Apply feature clipping to all hidden states in a list."""
    return [feature_clipping(h, clip_val) for h in hidden_states]


def compute_tsv(
    truthful_states: List[np.ndarray],
    hallucinated_states: List[np.ndarray],
) -> np.ndarray:
    """
    Compute the Truthfulness Separator Vector.

    This is the normalised difference between the mean of truthful
    hidden states and the mean of hallucinated hidden states.
    It points in the direction that best separates the two classes.

    Args:
        truthful_states:     Hidden states from truthful tokens
        hallucinated_states: Hidden states from hallucinated tokens

    Returns:
        TSV vector of shape (hidden_size,)
    """
    if not truthful_states or not hallucinated_states:
        raise ValueError("Need both truthful and hallucinated states to compute TSV")

    mean_truthful     = np.mean(truthful_states, axis=0)
    mean_hallucinated = np.mean(hallucinated_states, axis=0)

    # Direction from hallucinated toward truthful
    tsv = mean_truthful - mean_hallucinated

    # Normalise so it's a unit vector
    norm = np.linalg.norm(tsv)
    if norm < 1e-8:
        return tsv  # can't normalise zero vector
    return tsv / norm


def apply_tsv(
    hidden_state: np.ndarray,
    tsv_vector: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Project out the TSV direction from a hidden state.

    This subtracts the component of hidden_state that lies along
    the TSV direction, making the representation more neutral.

    Args:
        hidden_state: numpy array (hidden_size,)
        tsv_vector:   The TSV computed from training data (or None to skip)
        alpha:        Strength of TSV application (0 = no effect, 1 = full)

    Returns:
        Modified hidden state, same shape
    """
    if tsv_vector is None:
        return hidden_state  # no TSV available — return unchanged

    # Project hidden_state onto TSV direction
    projection = np.dot(hidden_state, tsv_vector) * tsv_vector

    # Subtract scaled projection
    return hidden_state - alpha * projection


def preprocess_hidden_state(
    hidden_state: np.ndarray,
    clip_val: float = 5.0,
    tsv_vector: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Full preprocessing pipeline for one hidden state:
    1. Feature clipping
    2. TSV adjustment (if TSV available)

    Args:
        hidden_state: Raw hidden state from LLaMA
        clip_val:     Clipping threshold
        tsv_vector:   Optional TSV for adjustment
        alpha:        TSV strength

    Returns:
        Processed hidden state ready for SEP and HalluShift
    """
    h = feature_clipping(hidden_state, clip_val)
    h = apply_tsv(h, tsv_vector, alpha)
    return h


def preprocess_all(
    hidden_states: List[np.ndarray],
    clip_val: float = 5.0,
    tsv_vector: Optional[np.ndarray] = None,
    alpha: float = 1.0,
) -> List[np.ndarray]:
    """Apply full preprocessing to all hidden states."""
    return [preprocess_hidden_state(h, clip_val, tsv_vector, alpha) for h in hidden_states]
