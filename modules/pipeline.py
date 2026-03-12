"""
modules/pipeline.py
====================
HalluSight — Full Detection Pipeline
Connects all 10 modules end-to-end.

INPUT:  A question string
OUTPUT: Tokens, risk scores, spans, overall %, highlighted text
"""

import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llm_loader import load_model, generate_with_hidden_states
from modules.semantic_entropy_probe import SemanticEntropyProbe
from modules.distribution_shift import distribution_shift_score
from modules.feature_clipping import preprocess_all
from modules.token_risk_scorer import score_all_tokens
from modules.eat_detector import boost_factual_risk
from modules.span_aggregator import aggregate_spans, build_highlighted_output
from modules.overall_scorer import overall_hallucination_percentage, build_final_result

# ── Globals: loaded once at startup, reused for every request ──
_tokenizer   = None
_model       = None
_sep_probe   = None
_pipeline_ready = False


def initialise_pipeline(probe_path: str = "sep_probe.pkl"):
    """Load LLM + probe into memory. Call ONCE when server starts."""
    global _tokenizer, _model, _sep_probe, _pipeline_ready

    print("\n╔══════════════════════════════════════════╗")
    print("║   HalluSight Pipeline — Initialising     ║")
    print("╚══════════════════════════════════════════╝\n")

    print("[1/2] Loading LLM model (this takes 1-2 min)...")
    _tokenizer, _model = load_model()

    print("[2/2] Loading SEP probe...")
    _sep_probe = SemanticEntropyProbe(save_path=probe_path)
    loaded = _sep_probe.load()
    if not loaded:
        print("  ⚠  No probe found — using neutral 0.5 scores")
        print("  ➜  Run: python train/train_probe.py to fix this")

    _pipeline_ready = True
    print("\n✓ Pipeline ready!\n")


def run_pipeline(prompt: str, max_new_tokens: int = 100) -> dict:
    """
    Run the full hallucination detection pipeline on a prompt.

    Returns dict with:
      full_text, tokens, risk_scores, hallucinated_spans,
      overall_score, risk_level, warning_message,
      highlighted_text, processing_time
    """
    global _tokenizer, _model, _sep_probe, _pipeline_ready

    if not _pipeline_ready:
        initialise_pipeline()

    t0 = time.time()
    print(f"\n▶ Running pipeline on: '{prompt[:55]}...'")

    # 1 ── Generate tokens + hidden states
    tokens, hidden_states, full_text = generate_with_hidden_states(
        prompt, _tokenizer, _model, max_new_tokens=max_new_tokens
    )
    if not tokens:
        return _empty_result(prompt, "No tokens generated")

    # 2 ── Preprocess hidden states
    processed = preprocess_all(hidden_states, clip_val=5.0)

    # 3 ── SEP probe scores
    sep_scores = (_sep_probe.score_batch(processed)
                  if _sep_probe and _sep_probe.is_ready()
                  else [0.5] * len(tokens))

    # 4 ── HalluShift distribution-shift scores
    shift_scores = distribution_shift_score(processed, window=5)

    # 5 ── Combine → per-token risk score
    risk_scores = score_all_tokens(sep_scores, shift_scores)

    # 6 ── EAT: boost factual tokens
    risk_scores = boost_factual_risk(risk_scores, tokens, full_text, boost=0.2)

    # 7 ── Aggregate into spans + build highlighted text
    spans       = aggregate_spans(tokens, risk_scores, threshold=0.5)
    highlighted = build_highlighted_output(tokens, risk_scores, threshold=0.5)

    # 8 ── Overall score
    result = build_final_result(tokens, risk_scores, spans, full_text, prompt)
    result["highlighted_text"] = highlighted
    result["processing_time"]  = round(time.time() - t0, 2)

    print(f"  ✓ Done in {result['processing_time']}s  |  "
          f"Score: {result['overall_score']}%  |  "
          f"Level: {result['risk_level']}")
    return result


def is_ready() -> bool:
    return _pipeline_ready


def _empty_result(prompt, reason):
    return {
        "prompt": prompt, "full_text": "", "tokens": [],
        "risk_scores": [], "hallucinated_spans": [], "span_texts": [],
        "overall_score": 0.0, "risk_level": "LOW",
        "warning_message": f"Could not process: {reason}",
        "highlighted_text": "", "processing_time": 0.0,
        "total_tokens": 0, "high_risk_tokens": 0,
        "medium_risk_tokens": 0, "low_risk_tokens": 0,
    }
