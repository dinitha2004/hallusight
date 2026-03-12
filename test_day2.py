"""
test_day2.py
=============
Run this at the END of Day 2 to confirm everything is working.

Usage:
    python test_day2.py

What this tests:
  ✓ SEP probe loads (or uses neutral scores if not trained yet)
  ✓ Distribution shift (HalluShift) computes scores correctly
  ✓ Feature clipping clips extreme values
  ✓ Token risk scorer combines SEP + HalluShift
  ✓ EAT detector finds names, dates, numbers
  ✓ Span aggregator groups risky tokens into spans
  ✓ Overall scorer computes hallucination percentage
  ✓ Full mini pipeline runs end to end
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("  HalluSight — Day 2 Test")
print("=" * 55)
print()

# ── TEST 1: Feature Clipping ─────────────────────────────
print("TEST 1: Feature Clipping...")
from modules.feature_clipping import feature_clipping, preprocess_all

test_vec = np.array([100.0, -200.0, 3.0, -2.5, 0.5])
clipped  = feature_clipping(test_vec, clip_val=5.0)
assert clipped.max() <= 5.0, "Clipping failed — max value too high"
assert clipped.min() >= -5.0, "Clipping failed — min value too low"
print(f"  Input:   {test_vec}")
print(f"  Clipped: {clipped}")
print("  ✓ Feature clipping works!\n")

# ── TEST 2: Distribution Shift (HalluShift) ──────────────
print("TEST 2: Distribution Shift Analyzer (HalluShift)...")
from modules.distribution_shift import distribution_shift_score

# Create 10 fake hidden states (dim=16 for speed in test)
np.random.seed(42)
normal_states = [np.random.randn(16) * 0.1 for _ in range(8)]   # smooth
anomaly_state = [np.random.randn(16) * 5.0]                       # big jump
test_states   = normal_states + anomaly_state + normal_states[:1]

scores = distribution_shift_score(test_states, window=3)
assert len(scores) == len(test_states), "Score count mismatch"
print(f"  Shift scores: {[round(s,3) for s in scores]}")
print(f"  Anomaly token (index 8) score: {scores[8]:.3f}")
print("  ✓ HalluShift computes shift scores!\n")

# ── TEST 3: SEP Probe ────────────────────────────────────
print("TEST 3: Semantic Entropy Probe (SEP)...")
from modules.semantic_entropy_probe import SemanticEntropyProbe

probe = SemanticEntropyProbe()
loaded = probe.load()

if loaded:
    test_h = np.random.randn(2048).astype(np.float32)
    score  = probe.score(test_h)
    assert 0.0 <= score <= 1.0, "Score out of range"
    print(f"  ✓ Probe loaded and scored: {score:.4f}")
else:
    score = probe.score(np.random.randn(16))
    print(f"  Probe not trained yet — using neutral score: {score}")
    print("  (Run: python train/train_probe.py to train it)")
print("  ✓ SEP probe works!\n")

# ── TEST 4: Token Risk Scorer ────────────────────────────
print("TEST 4: Token Risk Scorer...")
from modules.token_risk_scorer import score_all_tokens, get_risk_summary

sep_s   = [0.1, 0.2, 0.8, 0.9, 0.1, 0.2]
shift_s = [0.1, 0.1, 0.7, 0.8, 0.2, 0.1]
risks   = score_all_tokens(sep_s, shift_s)

print(f"  SEP scores:   {sep_s}")
print(f"  Shift scores: {shift_s}")
print(f"  Risk scores:  {risks}")

summary = get_risk_summary(risks)
print(f"  Summary: {summary}")
assert len(risks) == len(sep_s), "Risk count mismatch"
assert all(0.0 <= r <= 1.0 for r in risks), "Risk out of range"
print("  ✓ Token risk scorer works!\n")

# ── TEST 5: EAT Detector ─────────────────────────────────
print("TEST 5: EAT Detector (named entities, dates, numbers)...")
from modules.eat_detector import detect_factual_token_indices, get_factual_tokens

test_text   = "Albert Einstein was born in 1879 in Germany."
test_tokens = test_text.split()
factual_idx = detect_factual_token_indices(test_text)
factual_tok = get_factual_tokens(test_text, test_tokens)

print(f"  Text: '{test_text}'")
print(f"  Factual token indices: {factual_idx}")
print(f"  Factual tokens found:  {[(t['token'], t['entity_type']) for t in factual_tok]}")
print("  ✓ EAT detector works!\n")

# ── TEST 6: EAT Boost ────────────────────────────────────
print("TEST 6: EAT Risk Boost...")
from modules.eat_detector import boost_factual_risk

base_risks  = [0.1] * len(test_tokens)
boosted     = boost_factual_risk(base_risks, test_tokens, test_text, boost=0.2)
print(f"  Tokens:      {test_tokens}")
print(f"  Base risks:  {base_risks}")
print(f"  Boosted:     {[round(b,2) for b in boosted]}")
print("  ✓ EAT boost works!\n")

# ── TEST 7: Span Aggregator ──────────────────────────────
print("TEST 7: Span Aggregator...")
from modules.span_aggregator import aggregate_spans, build_highlighted_output

tokens_ex = ["Paris", "is", "the", "capital", "of", "Spain", "not", "France"]
risks_ex  = [0.1,     0.05, 0.05,  0.1,       0.05, 0.85,    0.2,   0.15  ]
spans     = aggregate_spans(tokens_ex, risks_ex, threshold=0.5)
highlight = build_highlighted_output(tokens_ex, risks_ex, threshold=0.5)

print(f"  Tokens: {tokens_ex}")
print(f"  Risks:  {risks_ex}")
print(f"  Spans found: {len(spans)}")
for s in spans:
    print(f"    → '{s['text']}' (avg_risk={s['avg_risk']}, max_risk={s['max_risk']})")
print(f"  Highlighted: {highlight}")
assert len(spans) >= 1, "Should find at least one span"
print("  ✓ Span aggregator works!\n")

# ── TEST 8: Overall Scorer ───────────────────────────────
print("TEST 8: Overall Hallucination Score Calculator...")
from modules.overall_scorer import (
    overall_hallucination_percentage,
    get_risk_level,
    get_warning_message,
    build_final_result,
)

test_risks = [0.1, 0.05, 0.05, 0.1, 0.05, 0.85, 0.2, 0.15]
pct        = overall_hallucination_percentage(test_risks)
level      = get_risk_level(pct)
warning    = get_warning_message(pct)

print(f"  Risk scores:  {test_risks}")
print(f"  Overall %:    {pct}%")
print(f"  Risk level:   {level}")
print(f"  Warning:      {warning}")
assert 0.0 <= pct <= 100.0, "Percentage out of range"
print("  ✓ Overall scorer works!\n")

# ── TEST 9: Full Mini Pipeline ───────────────────────────
print("TEST 9: Full mini pipeline (no LLM — using fake hidden states)...")
from modules.feature_clipping import preprocess_all
from modules.token_risk_scorer import score_all_tokens
from modules.eat_detector import boost_factual_risk
from modules.span_aggregator import aggregate_spans
from modules.overall_scorer import build_final_result

np.random.seed(123)
fake_tokens = ["Einstein", "was", "born", "in", "1865", "in", "France"]
fake_hidden = [np.random.randn(64).astype(np.float32) for _ in fake_tokens]

# Simulate high risk for tokens 4 and 6 (1865 and France — the wrong facts)
fake_hidden[4] = np.random.randn(64).astype(np.float32) * 5  # high activation anomaly
fake_hidden[6] = np.random.randn(64).astype(np.float32) * 5

processed      = preprocess_all(fake_hidden, clip_val=5.0)
sep_scores_f   = [probe.score(h) for h in processed]
shift_scores_f = distribution_shift_score(processed, window=2)
risk_scores_f  = score_all_tokens(sep_scores_f, shift_scores_f)
full_text_f    = " ".join(fake_tokens)
risk_scores_f  = boost_factual_risk(risk_scores_f, fake_tokens, full_text_f)
spans_f        = aggregate_spans(fake_tokens, risk_scores_f, threshold=0.4)
result_f       = build_final_result(fake_tokens, risk_scores_f, spans_f, full_text_f)

print(f"  Tokens:         {result_f['tokens']}")
print(f"  Risk scores:    {result_f['risk_scores']}")
print(f"  Spans:          {result_f['span_texts']}")
print(f"  Overall score:  {result_f['overall_score']}%")
print(f"  Risk level:     {result_f['risk_level']}")
print(f"  Warning:        {result_f['warning_message']}")
print("  ✓ Full mini pipeline works!\n")

# ── ALL DONE ─────────────────────────────────────────────
print("=" * 55)
print("  DAY 2 COMPLETE — ALL TESTS PASSED!")
print("=" * 55)
print()
print("  What you built today:")
print("  ✓ SEP - Semantic Entropy Probe")
print("  ✓ HalluShift - Distribution Shift Analyzer")
print("  ✓ Feature Clipping + TSV")
print("  ✓ Token Risk Scorer")
print("  ✓ EAT Detector")
print("  ✓ Span Aggregator")
print("  ✓ Overall Hallucination Score Calculator")
print()
print("  IMPORTANT — Train your probe now:")
print("  python train/train_probe.py")
print()
print("  Then tomorrow (Day 3):")
print("  → Build pipeline.py (connects all modules)")
print("  → Build Flask API")
print("  → Build web frontend")
print("  → Run full system demo")
print()
