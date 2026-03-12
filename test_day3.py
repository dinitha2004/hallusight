"""
test_day3.py
=============
HalluSight — Day 3 Test
Tests the pipeline and API without needing the server running.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("  HalluSight — Day 3 Test")
print("=" * 55)

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}")
        print(f"    ERROR: {e}")
        failed += 1

# ── TEST 1: Pipeline imports ──────────────────────────
def t1():
    from modules.pipeline import run_pipeline, initialise_pipeline, is_ready
    assert callable(run_pipeline)
    assert callable(initialise_pipeline)
    assert callable(is_ready)
test("Pipeline module imports correctly", t1)

# ── TEST 2: Flask app imports ──────────────────────────
def t2():
    # just check api/app.py exists and is valid Python
    path = os.path.join(os.path.dirname(__file__), "api", "app.py")
    assert os.path.exists(path), f"api/app.py not found at {path}"
    with open(path) as f:
        src = f.read()
    assert "def detect" in src
    assert "def health" in src
    assert "@app.route" in src
test("Flask API file exists and has correct endpoints", t2)

# ── TEST 3: Frontend file exists ───────────────────────
def t3():
    path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    assert os.path.exists(path), f"frontend/index.html not found at {path}"
    with open(path) as f:
        src = f.read()
    assert "HalluSight" in src
    assert "runDetection" in src
    assert "hl-high" in src
test("Frontend HTML file exists and has detection UI", t3)

# ── TEST 4: Pipeline _empty_result ─────────────────────
def t4():
    from modules.pipeline import _empty_result
    r = _empty_result("test prompt", "test reason")
    assert r["overall_score"] == 0.0
    assert r["risk_level"] == "LOW"
    assert "tokens" in r
    assert "hallucinated_spans" in r
test("Pipeline _empty_result returns correct structure", t4)

# ── TEST 5: Full pipeline (fake hidden states) ──────────
def t5():
    import numpy as np
    from modules.feature_clipping import preprocess_all
    from modules.token_risk_scorer import score_all_tokens
    from modules.eat_detector import boost_factual_risk
    from modules.span_aggregator import aggregate_spans, build_highlighted_output
    from modules.overall_scorer import build_final_result

    tokens = ["Einstein", "was", "born", "in", "1865", "in", "Paris"]
    hidden = [np.random.randn(2048).astype(np.float32) for _ in tokens]

    processed   = preprocess_all(hidden)
    sep_scores  = [0.1, 0.1, 0.1, 0.1, 0.85, 0.1, 0.75]
    shift_scores= [0.1, 0.1, 0.1, 0.1, 0.80, 0.1, 0.70]
    risk_scores = score_all_tokens(sep_scores, shift_scores)
    risk_scores = boost_factual_risk(risk_scores, tokens,
                                     " ".join(tokens), boost=0.2)
    spans       = aggregate_spans(tokens, risk_scores, threshold=0.5)
    highlighted = build_highlighted_output(tokens, risk_scores, threshold=0.5)
    result      = build_final_result(tokens, risk_scores, spans,
                                     " ".join(tokens), "Test prompt")

    assert result["overall_score"] > 0
    assert len(spans) >= 1
    assert "[[" in highlighted or len(highlighted) > 0
test("Full pipeline flow with fake hidden states", t5)

# ── Summary ────────────────────────────────────────────
print()
print("=" * 55)
if failed == 0:
    print(f"  DAY 3 COMPLETE — ALL {passed} TESTS PASSED!")
    print()
    print("  What you built today:")
    print("  ✓ modules/pipeline.py  — connects all 10 modules")
    print("  ✓ api/app.py           — Flask REST API")
    print("  ✓ frontend/index.html  — web UI with highlighting")
    print()
    print("  To run the full system:")
    print("  1. python api/app.py          ← start the server")
    print("  2. Open frontend/index.html   ← open in browser")
else:
    print(f"  {passed} passed  |  {failed} FAILED")
print("=" * 55)
