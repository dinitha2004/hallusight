"""
modules/overall_scorer.py
===========================
Overall Hallucination Score Calculator

WHAT THIS DOES (simple explanation):
  After scoring every token individually, we need ONE number
  that summarises: "How hallucinated is this whole response?"

  This is like a student's overall grade:
    - Some questions were answered correctly (low risk tokens)
    - Some were answered wrong (high risk tokens)
    - The overall score tells you how well they did overall

  We compute a WEIGHTED average — factual tokens (names, dates, numbers)
  get more weight because getting them wrong is more harmful.

OUTPUT:
  A percentage from 0% to 100%
  0%   = fully trustworthy response
  100% = fully hallucinated response
  <30% = mostly safe (green warning)
  30-60% = moderate risk (yellow warning)
  >60% = high risk (red warning — user should verify)
"""

from typing import List, Optional


def overall_hallucination_percentage(
    risk_scores: List[float],
    method: str = "weighted",
    high_risk_threshold: float = 0.6,
    high_risk_boost: float = 1.5,
) -> float:
    """
    Calculate the overall hallucination percentage for a response.

    Args:
        risk_scores:         List of per-token risk scores (0.0 to 1.0)
        method:              "simple" = plain average
                             "weighted" = high-risk tokens count more
        high_risk_threshold: Score above which a token is "high risk"
        high_risk_boost:     How much extra weight high-risk tokens get

    Returns:
        Percentage between 0.0 and 100.0
    """
    if not risk_scores:
        return 0.0

    if method == "simple":
        avg = sum(risk_scores) / len(risk_scores)
        return round(avg * 100, 2)

    # Weighted method: high-risk tokens get more influence
    total_weight = 0.0
    weighted_sum = 0.0

    for score in risk_scores:
        if score >= high_risk_threshold:
            weight = high_risk_boost  # high risk token counts more
        else:
            weight = 1.0
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    avg = weighted_sum / total_weight
    return round(avg * 100, 2)


def get_risk_level(percentage: float) -> str:
    """
    Convert percentage to a risk level label.

    Args:
        percentage: 0.0 to 100.0

    Returns:
        "LOW", "MODERATE", or "HIGH"
    """
    if percentage < 30.0:
        return "LOW"
    elif percentage < 60.0:
        return "MODERATE"
    else:
        return "HIGH"


def get_warning_message(percentage: float) -> str:
    """
    Return a user-facing warning message based on hallucination score.

    This satisfies FR3 from your design: "Notify users"
    """
    level = get_risk_level(percentage)

    if level == "LOW":
        return (
            f"Low hallucination risk ({percentage:.1f}%). "
            "This response appears mostly reliable."
        )
    elif level == "MODERATE":
        return (
            f"Moderate hallucination risk ({percentage:.1f}%). "
            "Some parts of this response may be inaccurate. "
            "Please verify highlighted segments."
        )
    else:
        return (
            f"HIGH hallucination risk ({percentage:.1f}%). "
            "Significant parts of this response may be incorrect. "
            "Do not rely on this response without verification."
        )


def build_final_result(
    tokens: List[str],
    risk_scores: List[float],
    spans: List[dict],
    full_text: str,
    prompt: str = "",
) -> dict:
    """
    Build the complete final result object that the API returns.

    This is the output of the entire HalluSight pipeline.

    Args:
        tokens:      Generated token strings
        risk_scores: Per-token risk scores
        spans:       Hallucinated spans from SpanAggregator
        full_text:   Complete generated text
        prompt:      Original user prompt

    Returns:
        Complete result dict ready for JSON serialisation
    """
    overall_pct = overall_hallucination_percentage(risk_scores)
    risk_level  = get_risk_level(overall_pct)
    warning     = get_warning_message(overall_pct)

    return {
        # Core outputs
        "prompt":              prompt,
        "full_text":           full_text,
        "overall_score":       overall_pct,
        "risk_level":          risk_level,
        "warning_message":     warning,

        # Token-level detail
        "tokens":              tokens,
        "risk_scores":         [round(s, 4) for s in risk_scores],

        # Span-level detail (the highlighted segments)
        "hallucinated_spans":  spans,
        "span_texts":          [s["text"] for s in spans],

        # Stats
        "total_tokens":        len(tokens),
        "high_risk_tokens":    sum(1 for s in risk_scores if s >= 0.6),
        "medium_risk_tokens":  sum(1 for s in risk_scores if 0.35 <= s < 0.6),
        "low_risk_tokens":     sum(1 for s in risk_scores if s < 0.35),
    }
