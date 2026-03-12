"""
modules/hidden_state_extractor.py
====================================
TBG — Token-By-Generation Hidden State Extractor

This module wraps the raw hidden state collection from llm_loader.py
and adds useful analysis: normalisation, layer selection, and statistics.

In simple terms:
  - When LLaMA generates each word, it produces an internal "state vector"
  - This vector is like a fingerprint of what the model is "thinking"
  - Normal, confident tokens have consistent, smooth vectors
  - Hallucinated tokens often have unusual or erratic vectors
  - This module organises and prepares those vectors for analysis
"""

import numpy as np
from typing import List, Tuple, Dict


class HiddenStateExtractor:
    """
    Processes and organises hidden state vectors collected during generation.
    
    Usage:
        extractor = HiddenStateExtractor(hidden_size=2048)
        stats = extractor.analyse(hidden_states_list, tokens)
    """

    def __init__(self, hidden_size: int = None):
        """
        Args:
            hidden_size: Dimension of hidden state vectors.
                         Set automatically when you call analyse().
                         OPT-1.3b = 2048, LLaMA-3.2-1B = 2048
        """
        self.hidden_size = hidden_size

    def normalise(self, hidden_state: np.ndarray) -> np.ndarray:
        """
        L2-normalise a vector so its magnitude is 1.
        This makes cosine similarity comparisons fair.
        
        Example: [3, 4] → [0.6, 0.8]  (magnitude was 5, now 1)
        """
        norm = np.linalg.norm(hidden_state)
        if norm < 1e-8:
            return hidden_state  # avoid divide-by-zero
        return hidden_state / norm

    def normalise_all(self, hidden_states: List[np.ndarray]) -> List[np.ndarray]:
        """Normalise every vector in the list."""
        return [self.normalise(h) for h in hidden_states]

    def compute_token_variance(self, hidden_state: np.ndarray) -> float:
        """
        Measures how spread out the values are in one hidden state vector.
        Higher variance = more "activated" representation.
        Very low variance can indicate uncertain or empty representations.
        """
        return float(np.var(hidden_state))

    def compute_token_magnitude(self, hidden_state: np.ndarray) -> float:
        """
        L2 norm (length) of the vector.
        Unusually small magnitudes can indicate low-confidence tokens.
        """
        return float(np.linalg.norm(hidden_state))

    def compute_pairwise_similarity(
        self, h1: np.ndarray, h2: np.ndarray
    ) -> float:
        """
        Cosine similarity between two hidden states.
        Range: -1 (opposite) to +1 (identical direction)
        Tokens in the same factual flow should be similar to each other.
        """
        n1 = np.linalg.norm(h1)
        n2 = np.linalg.norm(h2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(h1, h2) / (n1 * n2))

    def analyse(
        self,
        hidden_states: List[np.ndarray],
        tokens: List[str],
    ) -> Dict:
        """
        Run full analysis on all collected hidden states.

        Args:
            hidden_states: List of vectors from generate_with_hidden_states()
            tokens:        Corresponding list of token strings

        Returns:
            Dict with:
              - normalised:     L2-normalised hidden states (ready for probe)
              - variances:      Per-token variance score
              - magnitudes:     Per-token magnitude (L2 norm)
              - similarities:   Similarity to previous token (0 for first)
              - summary:        Quick stats about the generation
        """
        if not hidden_states:
            return {}

        # Detect hidden size from first vector
        self.hidden_size = hidden_states[0].shape[0]

        normalised     = self.normalise_all(hidden_states)
        variances      = [self.compute_token_variance(h) for h in hidden_states]
        magnitudes     = [self.compute_token_magnitude(h) for h in hidden_states]

        # Similarity to previous token's hidden state
        similarities = [0.0]  # no previous for first token
        for i in range(1, len(hidden_states)):
            sim = self.compute_pairwise_similarity(hidden_states[i], hidden_states[i - 1])
            similarities.append(sim)

        # Build per-token info
        token_info = []
        for i, token in enumerate(tokens):
            token_info.append({
                "index":      i,
                "token":      token,
                "variance":   round(variances[i], 6),
                "magnitude":  round(magnitudes[i], 4),
                "similarity_to_prev": round(similarities[i], 4),
            })

        summary = {
            "total_tokens":     len(tokens),
            "hidden_size":      self.hidden_size,
            "mean_variance":    round(float(np.mean(variances)), 6),
            "mean_magnitude":   round(float(np.mean(magnitudes)), 4),
            "mean_similarity":  round(float(np.mean(similarities[1:])), 4) if len(similarities) > 1 else 0.0,
        }

        return {
            "normalised":    normalised,
            "variances":     variances,
            "magnitudes":    magnitudes,
            "similarities":  similarities,
            "token_info":    token_info,
            "summary":       summary,
        }

    def print_summary(self, analysis_result: Dict):
        """Prints a readable summary of the hidden state analysis."""
        s = analysis_result.get("summary", {})
        print("\n── Hidden State Summary ───────────────────────────")
        print(f"  Total tokens generated : {s.get('total_tokens', 0)}")
        print(f"  Hidden state dimension : {s.get('hidden_size', '?')}")
        print(f"  Mean token variance    : {s.get('mean_variance', 0):.6f}")
        print(f"  Mean token magnitude   : {s.get('mean_magnitude', 0):.4f}")
        print(f"  Mean token similarity  : {s.get('mean_similarity', 0):.4f}")
        print("───────────────────────────────────────────────────\n")

        print("  Per-token breakdown (first 10 tokens):")
        for info in analysis_result.get("token_info", [])[:10]:
            print(
                f"    [{info['index']:>2}] {info['token']:<15}"
                f"  var={info['variance']:.5f}"
                f"  mag={info['magnitude']:.3f}"
                f"  sim={info['similarity_to_prev']:.3f}"
            )
        print()
