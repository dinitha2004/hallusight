"""
train/train_probe.py
======================
Script to train the Semantic Entropy Probe (SEP)

WHAT THIS DOES:
  1. Loads the OPT/LLaMA model
  2. For each training sample, generates a response and collects hidden states
  3. Labels each token as 0 (truthful) or 1 (hallucinated)
  4. Trains the logistic regression probe on those labelled hidden states
  5. Saves the trained probe to sep_probe.pkl

HOW LABELLING WORKS:
  We use a clever trick:
  - Feed the CORRECT answer → collect hidden states → label all as 0 (truthful)
  - Feed the WRONG/hallucinated answer → collect hidden states → label all as 1
  - Train the probe to distinguish between these two types of hidden states

HOW TO RUN:
  cd ~/Desktop/hallusight
  source venv/bin/activate
  python train/train_probe.py

TIME: About 5-15 minutes depending on your Mac's speed.
"""

import sys
import os
import json
import time

# Make sure we can import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.llm_loader import load_model, generate_with_hidden_states
from modules.semantic_entropy_probe import SemanticEntropyProbe
from modules.feature_clipping import preprocess_all

print("=" * 55)
print("  HalluSight — SEP Probe Training")
print("=" * 55)
print()

# ── Training configuration ───────────────────────────────
MAX_NEW_TOKENS = 30    # generate up to 30 tokens per sample
CLIP_VAL       = 5.0   # feature clipping value

# Training data: pairs of (prompt, correct_answer, wrong_answer)
# The probe learns to distinguish hidden states of correct vs wrong answers
TRAINING_PAIRS = [
    # Format: (prompt, truthful_continuation, hallucinated_continuation)
    ("The capital of France is", "Paris", "Lyon"),
    ("The capital of Germany is", "Berlin", "Munich"),
    ("The capital of Japan is", "Tokyo", "Osaka"),
    ("Albert Einstein was born in", "1879", "1865"),
    ("The first moon landing was in", "1969", "1972"),
    ("Water freezes at", "0 degrees Celsius", "10 degrees Celsius"),
    ("The speed of light is approximately", "300000 kilometers", "150000 kilometers"),
    ("Shakespeare wrote", "Hamlet and Romeo and Juliet", "The Iliad and the Odyssey"),
    ("The Amazon river flows through", "South America", "Africa"),
    ("The Eiffel Tower is in", "Paris", "London"),
    ("Marie Curie was born in", "Poland", "France"),
    ("The Great Wall of China was built to protect against", "invasions", "floods"),
    ("The human body has", "206 bones", "180 bones"),
    ("DNA stands for", "deoxyribonucleic acid", "digital nucleic array"),
    ("The largest planet in our solar system is", "Jupiter", "Saturn"),
    ("The United States declared independence in", "1776", "1789"),
    ("Gravity was described by", "Isaac Newton", "Albert Einstein"),
    ("The Sahara desert is located in", "Africa", "Asia"),
    ("Python programming language was created by", "Guido van Rossum", "James Gosling"),
    ("The chemical symbol for gold is", "Au", "Ag"),
]

print(f"  Training pairs: {len(TRAINING_PAIRS)}")
print(f"  Tokens per sample: up to {MAX_NEW_TOKENS}")
print(f"  Total samples expected: ~{len(TRAINING_PAIRS) * 2 * MAX_NEW_TOKENS}")
print()

# ── Load model ───────────────────────────────────────────
print("Loading model (this may take a minute)...")
tokenizer, model = load_model()
print()

# ── Collect training data ────────────────────────────────
all_hidden_states = []
all_labels = []

print("Collecting hidden states from training pairs...")
print("(This generates text for each pair — may take several minutes)")
print()

for i, (prompt, correct, hallucinated) in enumerate(TRAINING_PAIRS):
    print(f"  [{i+1}/{len(TRAINING_PAIRS)}] Prompt: '{prompt}'")

    # ── Truthful tokens (label = 0) ──────────────────────
    truthful_prompt = prompt + " " + correct
    try:
        _, hidden_states_t, _ = generate_with_hidden_states(
            truthful_prompt, tokenizer, model, max_new_tokens=MAX_NEW_TOKENS
        )
        hidden_states_t_processed = preprocess_all(hidden_states_t, clip_val=CLIP_VAL)
        all_hidden_states.extend(hidden_states_t_processed)
        all_labels.extend([0] * len(hidden_states_t_processed))
        print(f"    Truthful: {len(hidden_states_t_processed)} tokens collected")
    except Exception as e:
        print(f"    WARNING: Truthful pass failed: {e}")

    # ── Hallucinated tokens (label = 1) ──────────────────
    hallucinated_prompt = prompt + " " + hallucinated
    try:
        _, hidden_states_h, _ = generate_with_hidden_states(
            hallucinated_prompt, tokenizer, model, max_new_tokens=MAX_NEW_TOKENS
        )
        hidden_states_h_processed = preprocess_all(hidden_states_h, clip_val=CLIP_VAL)
        all_hidden_states.extend(hidden_states_h_processed)
        all_labels.extend([1] * len(hidden_states_h_processed))
        print(f"    Hallucinated: {len(hidden_states_h_processed)} tokens collected")
    except Exception as e:
        print(f"    WARNING: Hallucinated pass failed: {e}")

print()
print(f"  Total hidden states collected: {len(all_hidden_states)}")
print(f"  Truthful (label 0): {all_labels.count(0)}")
print(f"  Hallucinated (label 1): {all_labels.count(1)}")
print()

if len(all_hidden_states) < 10:
    print("ERROR: Not enough training data collected. Check model loading.")
    sys.exit(1)

# ── Train the probe ──────────────────────────────────────
print("Training Semantic Entropy Probe...")
probe = SemanticEntropyProbe(save_path="sep_probe.pkl")
result = probe.train(all_hidden_states, all_labels, verbose=True)

print()
print("=" * 55)
print("  TRAINING COMPLETE!")
print("=" * 55)
print(f"  Training accuracy: {result['accuracy']*100:.1f}%")
print(f"  Probe saved to:    sep_probe.pkl")
print()
print("  You can now run:")
print("  python test_day2.py")
print()
