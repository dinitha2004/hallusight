"""
modules/eat_detector.py
=========================
EAT — Exact Answer Token Detector

WHAT THIS DOES (simple explanation):
  Not all words are equally important in a hallucination.
  If LLaMA says "Einstein was born in 1879 in Germany" — all words matter.
  But if it says "Einstein was born in 1899 in France" — the wrong parts
  are specifically "1899" (should be 1879) and "France" (should be Germany).

  These critical factual tokens are called Exact Answer Tokens (EAT):
    - Person names     (e.g. "Einstein", "Napoleon", "Musk")
    - Place names      (e.g. "Paris", "Germany", "Mars")
    - Dates & years    (e.g. "1879", "March", "2024")
    - Numbers          (e.g. "42", "3.14", "1000")
    - Organizations    (e.g. "NASA", "Google", "WHO")

HOW IT WORKS:
  We use spaCy's Named Entity Recognition (NER) to find these tokens.
  If a token is a factual EAT token, we BOOST its risk score slightly.
  This makes the system more sensitive to hallucinated facts specifically.

WHY THIS IS IMPORTANT (satisfies FR2 from your design):
  "Identify hallucination patterns" — EAT focuses detection on the
  exact tokens that matter most for factual accuracy.
"""

import re
import spacy
from typing import List, Set

# Load the spaCy English model
# en_core_web_sm is small and fast — perfect for real-time detection
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    print("WARNING: spaCy model not found. Run: python -m spacy download en_core_web_sm")
    _nlp = None

# Entity types that are factual and hallucination-prone
FACTUAL_ENTITY_TYPES = {
    "PERSON",   # People's names
    "ORG",      # Organisations
    "GPE",      # Countries, cities, states
    "LOC",      # Locations
    "DATE",     # Dates and times
    "TIME",     # Time expressions
    "CARDINAL", # Numbers
    "ORDINAL",  # Ordinal numbers (first, second)
    "QUANTITY", # Measurements
    "MONEY",    # Monetary values
    "PERCENT",  # Percentages
    "PRODUCT",  # Products
    "EVENT",    # Named events
    "LAW",      # Named laws or documents
    "LANGUAGE", # Languages
    "NORP",     # Nationalities or religious groups
    "FAC",      # Facilities (buildings, airports)
    "WORK_OF_ART", # Titles of books, songs
}


def detect_factual_token_indices(text: str) -> Set[int]:
    """
    Find word indices in the text that are factual (EAT) tokens.

    Args:
        text: The full generated text string

    Returns:
        Set of word indices that are factual tokens
    """
    if _nlp is None:
        return set()

    doc = _nlp(text)
    factual_indices = set()

    # Named entities from spaCy NER
    for ent in doc.ents:
        if ent.label_ in FACTUAL_ENTITY_TYPES:
            for token in ent:
                factual_indices.add(token.i)

    # Also add any numeric tokens not caught by NER
    for token in doc:
        if token.like_num:
            factual_indices.add(token.i)
        # Matches patterns like: 42, 3.14, 1879, 2024
        if re.match(r'^\d+\.?\d*$', token.text):
            factual_indices.add(token.i)

    return factual_indices


def boost_factual_risk(
    token_risks: List[float],
    tokens: List[str],
    full_text: str,
    boost: float = 0.2,
) -> List[float]:
    """
    Boost risk scores for factual (EAT) tokens.

    Factual tokens get their risk score increased by `boost` because
    hallucinations in factual tokens are most harmful.

    Args:
        token_risks: Current risk scores for each token (0.0 to 1.0)
        tokens:      List of token strings (from generation)
        full_text:   Full generated text (used for NER analysis)
        boost:       Amount to add to factual token scores (default 0.2)

    Returns:
        Updated risk scores list
    """
    if _nlp is None or not full_text.strip():
        return token_risks

    factual_indices = detect_factual_token_indices(full_text)

    boosted = list(token_risks)  # copy so we don't modify original

    for idx in factual_indices:
        if idx < len(boosted):
            # Add boost and clamp to maximum 1.0
            boosted[idx] = min(boosted[idx] + boost, 1.0)

    return boosted


def get_factual_tokens(text: str, tokens: List[str]) -> List[dict]:
    """
    Return a list of factual tokens found in the text with their entity type.

    Useful for debugging and for the final output module.

    Args:
        text:   Full generated text
        tokens: List of token strings

    Returns:
        List of dicts: [{"token": "Paris", "entity_type": "GPE", "index": 3}, ...]
    """
    if _nlp is None:
        return []

    doc = _nlp(text)
    factual_tokens = []

    for ent in doc.ents:
        if ent.label_ in FACTUAL_ENTITY_TYPES:
            for token in ent:
                factual_tokens.append({
                    "token":       token.text,
                    "entity_type": ent.label_,
                    "index":       token.i,
                })

    return factual_tokens
