"""
modules/span_aggregator.py
============================
Span Aggregator

WHAT THIS DOES (simple explanation):
  After scoring every token, we have something like:
    Token:  ["Paris", "is",  "the",  "capital", "of",  "Spain",  "not",  "France"]
    Risk:   [0.1,     0.05,  0.05,   0.1,       0.05,  0.85,     0.2,    0.15   ]

  We don't want to highlight every risky token individually —
  that would look messy. Instead we GROUP consecutive risky tokens
  into meaningful SPANS (phrases).

  Result:
    Hallucinated span: ["Spain"] (risk 0.85 — highlighted in red)

  This directly satisfies your research requirement:
    "Detect hallucination at fine-grained level and highlight
     the exact incorrect segment instead of the whole paragraph"

HOW IT WORKS:
  1. Mark every token above a threshold (default 0.5) as "risky"
  2. Group consecutive risky tokens into spans
  3. Filter out very short spans if needed
  4. Return spans with their text and average risk score
"""

from typing import List, Tuple, Dict


def aggregate_spans(
    tokens: List[str],
    risk_scores: List[float],
    threshold: float = 0.5,
    min_span_length: int = 1,
) -> List[Dict]:
    """
    Group consecutive high-risk tokens into hallucinated spans.

    Args:
        tokens:          List of token strings from generation
        risk_scores:     Risk score per token (0.0 to 1.0)
        threshold:       Minimum risk score to consider a token risky (default 0.5)
        min_span_length: Minimum number of tokens to form a span (default 1)

    Returns:
        List of span dicts:
        [
          {
            "text":       "Spain",        <- the hallucinated text
            "tokens":     ["Spain"],      <- individual tokens in span
            "start":      5,              <- start index in token list
            "end":        5,              <- end index in token list
            "avg_risk":   0.85,           <- average risk score of span
            "max_risk":   0.85,           <- highest risk token in span
          },
          ...
        ]
    """
    if not tokens or not risk_scores:
        return []

    # Align lengths in case of mismatch
    n = min(len(tokens), len(risk_scores))
    tokens = tokens[:n]
    risk_scores = risk_scores[:n]

    spans = []
    current_span_tokens = []
    current_span_scores = []
    current_span_start = -1

    for i, (token, score) in enumerate(zip(tokens, risk_scores)):
        if score >= threshold:
            # Start or continue a span
            if not current_span_tokens:
                current_span_start = i
            current_span_tokens.append(token)
            current_span_scores.append(score)
        else:
            # End current span if it exists
            if current_span_tokens and len(current_span_tokens) >= min_span_length:
                spans.append(_make_span(
                    current_span_tokens,
                    current_span_scores,
                    current_span_start,
                ))
            current_span_tokens = []
            current_span_scores = []
            current_span_start = -1

    # Don't forget the last span if generation ended while in a span
    if current_span_tokens and len(current_span_tokens) >= min_span_length:
        spans.append(_make_span(
            current_span_tokens,
            current_span_scores,
            current_span_start,
        ))

    return spans


def _make_span(tokens: List[str], scores: List[float], start: int) -> Dict:
    """Helper to build a span dictionary."""
    text = " ".join(t.strip() for t in tokens if t.strip())
    avg_risk = sum(scores) / len(scores) if scores else 0.0
    max_risk = max(scores) if scores else 0.0

    return {
        "text":     text,
        "tokens":   tokens,
        "start":    start,
        "end":      start + len(tokens) - 1,
        "avg_risk": round(avg_risk, 4),
        "max_risk": round(max_risk, 4),
    }


def build_highlighted_output(
    tokens: List[str],
    risk_scores: List[float],
    threshold: float = 0.5,
) -> str:
    """
    Build a plain-text version of the output where hallucinated spans
    are marked with [[ ]] brackets.

    Example output:
      "Paris is the capital of [[Spain]] according to the model."

    Args:
        tokens:      Token list
        risk_scores: Risk score per token
        threshold:   Risk threshold for highlighting

    Returns:
        String with hallucinated spans wrapped in [[ ]]
    """
    n = min(len(tokens), len(risk_scores))
    result_parts = []
    in_span = False

    for i in range(n):
        token = tokens[i].strip()
        score = risk_scores[i]

        if score >= threshold:
            if not in_span:
                result_parts.append("[[")
                in_span = True
            result_parts.append(token)
        else:
            if in_span:
                result_parts.append("]]")
                in_span = False
            result_parts.append(token)

    if in_span:
        result_parts.append("]]")

    return " ".join(result_parts)


def get_span_texts(spans: List[Dict]) -> List[str]:
    """Return just the text strings of all spans."""
    return [s["text"] for s in spans]
