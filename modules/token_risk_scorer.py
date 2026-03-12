"""
modules/token_risk_scorer.py
==============================
Token-Level Risk Scoring

WHAT THIS DOES (simple explanation):
  We now have TWO scores for each token:
    - SEP score:   from the trained logistic regression probe
    - Shift score: from the HalluShift distribution analyzer

  This module combines them into ONE final risk score per token.

  Think of it like combining two doctor's opinions:
    Doctor 1 (SEP probe) says: "I think this patient has 70% chance of illness"
    Doctor 2 (HalluShift) says: "I think this patient has 60% chance of illness"
    Combined: weighted average = more reliable diagnosis

  Default weights: SEP = 60%, HalluShift = 40%
  (SEP gets more weight because it's trained on labelled data)
"""

from typing import List


def compute_token_risk(
    sep_score: float,
    shift_score: float,
    sep_weight: float = 0.6,
    shift_weight: float = 0.4,
) -> float:
    """
    Combine SEP and distribution shift scores into one risk score.

    Args:
        sep_score:    Hallucination probability from SEP probe (0.0 to 1.0)
        shift_score:  Distribution shift score from HalluShift (0.0 to 1.0)
        sep_weight:   How much to trust the SEP probe (default 0.6 = 60%)
        shift_weight: How much to trust HalluShift (default 0.4 = 40%)

    Returns:
        Combined risk score between 0.0 and 1.0
        0.0 = very safe (truthful)
        1.0 = very risky (hallucinated)
    """
    risk = sep_weight * sep_score + shift_weight * shift_score
    # Clamp to [0.0, 1.0] just in case
    return round(float(min(max(risk, 0.0), 1.0)), 4)


def score_all_tokens(
    sep_scores: List[float],
    shift_scores: List[float],
    sep_weight: float = 0.6,
    shift_weight: float = 0.4,
) -> List[float]:
    """
    Compute risk score for every token at once.

    Args:
        sep_scores:   List of SEP scores (one per token)
        shift_scores: List of shift scores (one per token)

    Returns:
        List of combined risk scores (one per token)
    """
    if len(sep_scores) != len(shift_scores):
        # Pad shorter list with 0.5 (neutral)
        max_len = max(len(sep_scores), len(shift_scores))
        sep_scores = sep_scores + [0.5] * (max_len - len(sep_scores))
        shift_scores = shift_scores + [0.5] * (max_len - len(shift_scores))

    return [
        compute_token_risk(s, h, sep_weight, shift_weight)
        for s, h in zip(sep_scores, shift_scores)
    ]


def classify_risk_level(score: float) -> str:
    """
    Convert a numeric risk score into a human-readable label.

    Args:
        score: Risk score 0.0 to 1.0

    Returns:
        "LOW", "MEDIUM", or "HIGH"
    """
    if score < 0.35:
        return "LOW"
    elif score < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"


def get_risk_summary(risk_scores: List[float]) -> dict:
    """
    Summarise risk scores across all tokens.

    Returns:
        dict with counts of LOW/MEDIUM/HIGH tokens and percentages
    """
    if not risk_scores:
        return {"low": 0, "medium": 0, "high": 0, "total": 0}

    levels = [classify_risk_level(s) for s in risk_scores]
    total = len(levels)

    low    = levels.count("LOW")
    medium = levels.count("MEDIUM")
    high   = levels.count("HIGH")

    return {
        "total":          total,
        "low":            low,
        "medium":         medium,
        "high":           high,
        "low_pct":        round(low / total * 100, 1),
        "medium_pct":     round(medium / total * 100, 1),
        "high_pct":       round(high / total * 100, 1),
        "mean_risk":      round(sum(risk_scores) / total, 4),
        "max_risk":       round(max(risk_scores), 4),
    }
