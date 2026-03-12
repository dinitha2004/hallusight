"""
model/llm_loader.py
====================
This file does two things:
  1. Loads the LLaMA (or OPT fallback) model onto your Mac
  2. Generates text token by token AND collects hidden states at each step

WHY HIDDEN STATES MATTER:
  - Normally an LLM just gives you the final text output
  - We turn on output_hidden_states=True so we can also see
    the internal "thinking" vectors at every generation step
  - These vectors are what we analyse for hallucination signals
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# MODEL SELECTION
# Change MODEL_NAME if you cannot access LLaMA
# ─────────────────────────────────────────────

# Option 1: LLaMA (requires HuggingFace access request)
# MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Option 2: Facebook OPT - FREE, no access needed, works the same way
# Use this if LLaMA download is slow or you didn't get access yet
MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Option 3: Smallest possible model for testing (125M params, very fast)
# MODEL_NAME = "facebook/opt-125m"

# Global variables so we only load the model once
_tokenizer = None
_model = None


def load_model(model_name: str = MODEL_NAME):
    """
    Downloads and loads the model on first call.
    On Mac (no GPU), this runs on CPU automatically.
    
    Args:
        model_name: HuggingFace model identifier string
    
    Returns:
        tokenizer: converts text <-> tokens
        model:     the LLM itself
    """
    global _tokenizer, _model

    # If already loaded, return cached version
    if _tokenizer is not None and _model is not None:
        print("✓ Model already loaded (using cached version)")
        return _tokenizer, _model

    print(f"\n Loading model: {model_name}")
    print("  (First run will download ~2-3 GB — this is normal)")
    print("  After first download, it loads from your Mac cache quickly\n")

    start = time.time()

    # Load tokenizer
    print("  Step 1/2: Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if missing (some models need this)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Load model
    # output_hidden_states=True  → gives us internal vectors at each layer
    # torch_dtype=torch.float32  → use 32-bit precision (safe for CPU/Mac)
    # device_map="cpu"           → forces CPU, safe for all Macs
    print("  Step 2/2: Loading model weights (may take 1-2 minutes)...")
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    # Set to evaluation mode (disables dropout, etc.)
    _model.eval()

    elapsed = time.time() - start
    print(f"\n✓ Model loaded successfully in {elapsed:.1f} seconds!")
    print(f"  Model has {sum(p.numel() for p in _model.parameters()) / 1e6:.0f}M parameters\n")

    return _tokenizer, _model


def generate_with_hidden_states(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
):
    """
    Generates a response to the prompt, token by token.
    At each step, collects the hidden state vector for that token.

    Args:
        prompt:         The user's question/input text
        tokenizer:      The loaded tokenizer
        model:          The loaded LLM
        max_new_tokens: Maximum words/tokens to generate (default 100)
        temperature:    Randomness of generation (1.0 = normal, lower = more focused)

    Returns:
        tokens_generated:  List of string tokens, e.g. ["Paris", " is", " the", ...]
        hidden_states_list: List of numpy arrays, one per token (shape: [hidden_size])
        full_text:         Complete generated string
    """

    # Convert the prompt text into token IDs the model understands
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]

    tokens_generated = []
    hidden_states_list = []

    print(f"  Generating response (up to {max_new_tokens} tokens)...")

    with torch.no_grad():  # no_grad = faster, less memory (we are not training)
        for step in range(max_new_tokens):

            # Run the model forward pass
            # This gives us: logits (next token probabilities) + hidden states
            output = model(input_ids, output_hidden_states=True)

            # ── Extract hidden state ──────────────────────────────────────
            # output.hidden_states is a tuple: one tensor per layer
            # We take the LAST layer (index -1) — most informative
            # Shape of each: [batch=1, sequence_length, hidden_size]
            # We take the last position [-1] = the current token being generated
            last_layer_hidden = output.hidden_states[-1]   # shape: [1, seq_len, hidden_size]
            current_token_hidden = last_layer_hidden[0, -1, :]  # shape: [hidden_size]

            # Convert to numpy for later processing (sklearn, scipy, numpy)
            hidden_states_list.append(current_token_hidden.numpy())

            # ── Pick the next token ───────────────────────────────────────
            # logits = raw scores for every possible next token
            # argmax picks the most likely one (greedy decoding)
            next_token_logits = output.logits[0, -1, :]  # shape: [vocab_size]
            next_token_id = torch.argmax(next_token_logits).unsqueeze(0)

            # Decode the token ID back to readable text
            token_text = tokenizer.decode(next_token_id, skip_special_tokens=True)
            tokens_generated.append(token_text)

            # ── Stop if end-of-sequence token ────────────────────────────
            if next_token_id.item() == tokenizer.eos_token_id:
                print(f"  ✓ Generation stopped at step {step+1} (end of sequence)")
                break

            # ── Append new token to input for next step ──────────────────
            # This is called "autoregressive" generation — each step
            # uses all previous tokens to predict the next one
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)

            # Print progress every 10 tokens
            if (step + 1) % 10 == 0:
                print(f"  ... {step+1} tokens generated")

    full_text = tokenizer.decode(
        input_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print(f"  ✓ Done! Generated {len(tokens_generated)} tokens\n")

    return tokens_generated, hidden_states_list, full_text


def get_model_info(model):
    """
    Prints useful information about the loaded model.
    Helpful for understanding the hidden state dimensions.
    """
    config = model.config
    print("\n── Model Information ──────────────────────────────")
    print(f"  Model type:        {config.model_type}")
    print(f"  Hidden size:       {config.hidden_size}  ← this is your vector dimension")
    print(f"  Number of layers:  {config.num_hidden_layers}")
    if hasattr(config, 'num_attention_heads'):
        print(f"  Attention heads:   {config.num_attention_heads}")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total parameters:  {total_params:.0f}M")
    print("───────────────────────────────────────────────────\n")
