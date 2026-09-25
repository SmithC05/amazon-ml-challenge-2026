"""
src/features.py
===============
Baseline pair-level feature extraction for the Amazon ML Challenge 2026.

Member 3 deliverable — baseline matching module.

This module extracts the agreed 16-feature baseline vector for a
(S1 entity, candidate entity) pair.  Normalization is NOT reimplemented
here; all text cleaning is delegated entirely to src/preprocess.py
(Member 2's deliverable).

Feature set
-----------
Name similarity (6):
    name_exact              — 1 if normalized names are identical
    name_jaccard            — Jaccard similarity of character 3-grams
    name_token_overlap      — Jaccard similarity of token sets
    name_levenshtein_ratio  — Normalized Levenshtein similarity (0–1)
    name_length_difference  — |len(n1) - len(n2)| / max(len, 1)
    name_token_count_diff   — |tokens(n1) - tokens(n2)| / max(count, 1)

Address similarity (7):
    address_exact           — 1 if normalized addresses are identical
    address_jaccard         — Jaccard similarity of character 3-grams
    address_token_overlap   — Jaccard similarity of token sets
    address_levenshtein_ratio
    address_length_difference
    address_token_count_diff
    address_missing         — 1 if EITHER normalized address is empty

Other (3):
    country_match           — 1 if country strings are equal (raw, lowercased)
    source_is_s2            — 1 if the candidate entity_id starts with "S2-"
    source_is_s3            — 1 if the candidate entity_id starts with "S3-"

Total: 16 features.

Design principles
-----------------
  • Pure functions: given identical inputs the output is always identical.
  • No side effects, no global mutable state.
  • Normalization is read from pre-computed _norm columns; this module
    never calls normalize_name / normalize_address directly on raw values.
  • Levenshtein is computed via a pure-Python DP implementation so there
    is no hard dependency on python-Levenshtein / editdistance at feature
    extraction time.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Agreed feature names (canonical order — must match train.py and predict.py)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES: list[str] = [
    # Name
    "name_exact",
    "name_jaccard",
    "name_token_overlap",
    "name_levenshtein_ratio",
    "name_length_difference",
    "name_token_count_diff",
    # Address
    "address_exact",
    "address_jaccard",
    "address_token_overlap",
    "address_levenshtein_ratio",
    "address_length_difference",
    "address_token_count_diff",
    "address_missing",
    # Other
    "country_match",
    "source_is_s2",
    "source_is_s3",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """Return the set of character n-grams in *text*."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i: i + n] for i in range(len(text) - n + 1)}


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|.  Returns 0.0 if both empty."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _token_set(text: str) -> set[str]:
    """Split *text* on whitespace and return the set of non-empty tokens."""
    return set(text.split())


def _levenshtein(a: str, b: str) -> int:
    """Pure-Python O(|a|·|b|) Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Single-row DP (space-efficient)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,          # deletion
                curr[j - 1] + 1,      # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = curr
    return prev[-1]


def _lev_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return 1.0 - _levenshtein(a, b) / max_len


def _length_diff(a: str, b: str) -> float:
    """Relative length difference: |len(a) - len(b)| / max(len, 1)."""
    la, lb = len(a), len(b)
    return abs(la - lb) / max(la, lb, 1)


def _token_count_diff(a: str, b: str) -> float:
    """Relative token-count difference."""
    ta = len(a.split()) if a else 0
    tb = len(b.split()) if b else 0
    return abs(ta - tb) / max(ta, tb, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Text-pair similarity block  (reused for name and address)
# ─────────────────────────────────────────────────────────────────────────────

def _text_features(a: str, b: str) -> tuple[float, float, float, float, float, float]:
    """
    Return (exact, jaccard, token_overlap, lev_ratio, length_diff, tok_count_diff)
    for a normalized text pair.
    """
    exact = float(a == b)
    jaccard = _jaccard(_char_ngrams(a), _char_ngrams(b))
    token_overlap = _jaccard(_token_set(a), _token_set(b))
    lev = _lev_ratio(a, b)
    ldiff = _length_diff(a, b)
    tcdiff = _token_count_diff(a, b)
    return exact, jaccard, token_overlap, lev, ldiff, tcdiff


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(row: pd.Series) -> dict[str, float]:
    """
    Extract the 16 baseline features for a single candidate pair.

    The input *row* must contain pre-computed _norm columns produced by
    src/preprocess.py (Member 2).  Raw columns are never read here.

    Expected columns
    ----------------
    From the S1 side:
        s1_name_norm, s1_address_norm, s1_country

    From the candidate side:
        cand_name_norm, cand_address_norm, cand_country, cand_entity_id

    Returns
    -------
    dict mapping each feature name (in FEATURE_NAMES order) to a float.
    """
    s1_name = str(row.get("s1_name_norm", "") or "")
    s1_addr = str(row.get("s1_address_norm", "") or "")
    s1_ctry = str(row.get("s1_country", "") or "").strip().lower()

    cn_name = str(row.get("cand_name_norm", "") or "")
    cn_addr = str(row.get("cand_address_norm", "") or "")
    cn_ctry = str(row.get("cand_country", "") or "").strip().lower()
    cand_id = str(row.get("cand_entity_id", "") or "")

    # Name block
    n_exact, n_jac, n_tok, n_lev, n_ldiff, n_tcdiff = _text_features(s1_name, cn_name)

    # Address block
    a_exact, a_jac, a_tok, a_lev, a_ldiff, a_tcdiff = _text_features(s1_addr, cn_addr)
    addr_missing = float(s1_addr == "" or cn_addr == "")

    # Other
    country_match = float(s1_ctry == cn_ctry and s1_ctry != "")
    source_is_s2 = float(cand_id.startswith("S2-"))
    source_is_s3 = float(cand_id.startswith("S3-"))

    return {
        "name_exact":            n_exact,
        "name_jaccard":          n_jac,
        "name_token_overlap":    n_tok,
        "name_levenshtein_ratio": n_lev,
        "name_length_difference": n_ldiff,
        "name_token_count_diff": n_tcdiff,
        "address_exact":         a_exact,
        "address_jaccard":       a_jac,
        "address_token_overlap": a_tok,
        "address_levenshtein_ratio": a_lev,
        "address_length_difference": a_ldiff,
        "address_token_count_diff":  a_tcdiff,
        "address_missing":       addr_missing,
        "country_match":         country_match,
        "source_is_s2":          source_is_s2,
        "source_is_s3":          source_is_s3,
    }


def build_feature_matrix(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply extract_features() to every row of *pairs_df* and return a
    DataFrame with columns exactly equal to FEATURE_NAMES (in order).

    Parameters
    ----------
    pairs_df : DataFrame
        Each row is a candidate pair with the columns described in
        extract_features().

    Returns
    -------
    DataFrame of shape (len(pairs_df), 16) with float64 dtype.
    """
    records = [extract_features(row) for _, row in pairs_df.iterrows()]
    return pd.DataFrame(records, columns=FEATURE_NAMES, index=pairs_df.index)
