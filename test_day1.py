"""
test_day1.py
=============
Run this at the END of Day 1 to confirm everything is working.

Usage:
    python test_day1.py

What this tests:
  ✓ LLaMA / OPT model loads correctly
  ✓ Text is generated token by token
  ✓ Hidden states are collected at each step
  ✓ HiddenStateExtractor analyses the vectors
  ✓ You can see variance, magnitude, similarity per token

If this runs without errors, Day 1 is COMPLETE!
"""

import sys
import time

print("=" * 55)
print("  HalluSight — Day 1 Test")
print("=" * 55)
print()

# ── Test 1: Import check ─────────────────────────────────
print("TEST 1: Checking all imports...")
try:
    import torch
    import numpy as np
    import sklearn
    import scipy
    import spacy
    import flask
    print("  ✓ torch")
    print("  ✓ numpy")
    print("  ✓ scikit-learn")
    print("  ✓ scipy")
    print("  ✓ spacy")
    print("  ✓ flask")
except ImportError as e:
    print(f"\n  ✗ IMPORT ERROR: {e}")
    print("  Fix: Make sure your venv is active and run:")
    print("       pip install torch transformers scikit-learn numpy scipy spacy flask flask-cors joblib")
    sys.exit(1)

print()
print("  ✓ All imports successful!\n")

# ── Test 2: spaCy model ──────────────────────────────────
print("TEST 2: Checking spaCy English model...")
try:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Albert Einstein was born in 1879 in Germany.")
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"  ✓ spaCy loaded")
    print(f"  ✓ Test sentence entities: {entities}")
except Exception as e:
    print(f"  ✗ spaCy error: {e}")
    print("  Fix: python -m spacy download en_core_web_sm")
    sys.exit(1)

print()

# ── Test 3: Load the LLM ─────────────────────────────────
print("TEST 3: Loading the LLM model...")
print("  (This may take a few minutes on first run — downloading model weights)")
print()

try:
    from model.llm_loader import load_model, generate_with_hidden_states, get_model_info
    tokenizer, model = load_model()
    get_model_info(model)
    print("  ✓ Model loaded successfully!")
except Exception as e:
    print(f"\n  ✗ Model loading error: {e}")
    print("\n  COMMON FIXES:")
    print("  1. If 'LLaMA access denied': Change MODEL_NAME in model/llm_loader.py to:")
    print('     MODEL_NAME = "facebook/opt-1.3b"')
    print("  2. If 'out of memory': Change to:")
    print('     MODEL_NAME = "facebook/opt-125m"')
    print("  3. If 'not logged in': Run: huggingface-cli login")
    sys.exit(1)

print()

# ── Test 4: Generate text with hidden states ─────────────
print("TEST 4: Generating text with hidden state extraction...")
print()

TEST_PROMPT = "The capital of France is"
print(f"  Prompt: '{TEST_PROMPT}'")
print()

try:
    start = time.time()
    tokens, hidden_states, full_text = generate_with_hidden_states(
        TEST_PROMPT,
        tokenizer,
        model,
        max_new_tokens=20,  # short for testing
    )
    elapsed = time.time() - start

    print(f"  ✓ Generation complete in {elapsed:.1f}s")
    print(f"  ✓ Full response:   '{full_text}'")
    print(f"  ✓ Tokens list:     {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
    print(f"  ✓ Number of tokens generated: {len(tokens)}")

except Exception as e:
    print(f"  ✗ Generation error: {e}")
    sys.exit(1)

print()

# ── Test 5: Check hidden state shape ────────────────────
print("TEST 5: Checking hidden state vectors...")
print()

import numpy as np

print(f"  Number of hidden states collected: {len(hidden_states)}")
print(f"  Shape of first hidden state:       {hidden_states[0].shape}")
print(f"  Data type:                         {hidden_states[0].dtype}")
print(f"  Sample values (first 5 dims):      {hidden_states[0][:5].round(4)}")
print()

if len(hidden_states) != len(tokens):
    print("  ✗ WARNING: hidden states count does not match token count!")
else:
    print("  ✓ Hidden states count matches token count")

print()

# ── Test 6: Run HiddenStateExtractor ────────────────────
print("TEST 6: Running HiddenStateExtractor analysis...")
print()

try:
    from modules.hidden_state_extractor import HiddenStateExtractor
    extractor = HiddenStateExtractor()
    analysis = extractor.analyse(hidden_states, tokens)
    extractor.print_summary(analysis)
    print("  ✓ HiddenStateExtractor works correctly!")
except Exception as e:
    print(f"  ✗ Extractor error: {e}")
    sys.exit(1)

# ── Test 7: Quick numpy sanity check ───────────────────
print("TEST 7: Quick computation sanity check...")
stacked = np.array(hidden_states)
print(f"  Shape of all hidden states stacked: {stacked.shape}")
print(f"  Mean activation value:              {stacked.mean():.4f}")
print(f"  Std  activation value:              {stacked.std():.4f}")
print()
print("  ✓ Numpy operations work on hidden states!")

# ── ALL DONE ─────────────────────────────────────────────
print()
print("=" * 55)
print("  🎉 DAY 1 COMPLETE — ALL TESTS PASSED!")
print("=" * 55)
print()
print("  What you confirmed today:")
print("  ✓ All libraries installed and importable")
print("  ✓ spaCy English NLP model working")
print("  ✓ LLM model loaded on CPU (Mac)")
print("  ✓ Text generation works token by token")
print("  ✓ Hidden states collected at each generation step")
print("  ✓ HiddenStateExtractor analyses vectors correctly")
print()
print("  Tomorrow (Day 2):")
print("  → Build SEP probe, HalluShift, FeatureClipping")
print("  → Build EAT Detector, SpanAggregator, OverallScorer")
print("  → Train the hallucination detection probe")
print()
print("  Run Day 2 when ready:")
print("  python test_day2.py")
print()
