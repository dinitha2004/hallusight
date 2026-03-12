"""
modules/semantic_entropy_probe.py
===================================
SEP — Semantic Entropy Probe

WHAT THIS DOES (simple explanation):
  Imagine you are a teacher grading a student's answer.
  You already know which answers were wrong (hallucinated).
  You train yourself to recognise the "feeling" of wrong answers.
  That's exactly what this probe does — but using hidden state vectors.

HOW IT WORKS:
  1. We take hidden state vectors from LLaMA for known hallucinated tokens
  2. We train a tiny logistic regression classifier on those vectors
  3. After training, when we see a new token's hidden state,
     the probe tells us: "probability this is hallucinated = 0.73"

WHY LOGISTIC REGRESSION (not a big neural network)?
  - It's fast (runs in milliseconds)
  - It works well on high-dimensional vectors
  - Easy to interpret
  - Meets NFR1: detection latency <= 2 seconds
"""

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from typing import List


class SemanticEntropyProbe:
    """
    A lightweight classifier that predicts hallucination probability
    from hidden state vectors.

    Usage (training):
        probe = SemanticEntropyProbe()
        probe.train(hidden_states, labels)

    Usage (inference):
        probe = SemanticEntropyProbe()
        probe.load()
        score = probe.score(hidden_state_vector)
    """

    def __init__(self, save_path: str = "sep_probe.pkl"):
        self.save_path = save_path
        self.scaler_path = save_path.replace(".pkl", "_scaler.pkl")

        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, hidden_states: List[np.ndarray], labels: List[int], verbose: bool = True) -> dict:
        """
        Train the probe on labelled hidden states.

        Args:
            hidden_states: List of numpy vectors (hidden_size,)
            labels:        0 = truthful, 1 = hallucinated
        """
        if len(hidden_states) == 0:
            raise ValueError("No training data provided!")
        if len(hidden_states) != len(labels):
            raise ValueError("hidden_states and labels must be same length")

        print(f"\n  Training Semantic Entropy Probe...")
        print(f"  Samples: {len(hidden_states)} | Truthful: {labels.count(0)} | Hallucinated: {labels.count(1)}")

        X = np.array(hidden_states, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        y_pred = self.model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)

        if verbose:
            print(f"  Training accuracy: {acc*100:.1f}%")
            print(classification_report(y, y_pred, target_names=["Truthful", "Hallucinated"]))

        self.save()
        return {"accuracy": acc, "n_samples": len(hidden_states)}

    def save(self):
        joblib.dump(self.model, self.save_path)
        joblib.dump(self.scaler, self.scaler_path)
        print(f"  Probe saved: {self.save_path}")

    def load(self) -> bool:
        if not os.path.exists(self.save_path):
            print(f"  No probe found at {self.save_path} — using neutral scores (0.5)")
            return False
        self.model = joblib.load(self.save_path)
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        self.is_trained = True
        print(f"  Probe loaded: {self.save_path}")
        return True

    def score(self, hidden_state: np.ndarray) -> float:
        """
        Returns hallucination probability for one token: 0.0 (safe) to 1.0 (hallucinated)
        """
        if not self.is_trained:
            return 0.5
        x = np.array(hidden_state, dtype=np.float32).reshape(1, -1)
        try:
            x_scaled = self.scaler.transform(x)
        except Exception:
            x_scaled = x
        proba = self.model.predict_proba(x_scaled)[0]
        return float(proba[1]) if len(proba) > 1 else 0.5

    def score_batch(self, hidden_states: List[np.ndarray]) -> List[float]:
        """Score multiple tokens at once (faster)."""
        if not self.is_trained:
            return [0.5] * len(hidden_states)
        X = np.array(hidden_states, dtype=np.float32)
        try:
            X_scaled = self.scaler.transform(X)
        except Exception:
            X_scaled = X
        probas = self.model.predict_proba(X_scaled)
        return [float(p[1]) if len(p) > 1 else 0.5 for p in probas]

    def is_ready(self) -> bool:
        return self.is_trained
