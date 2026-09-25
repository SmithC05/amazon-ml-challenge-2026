"""
features.py
-----------
Member 3 – Amazon ML Challenge 2026 – Business Entity Resolution

REUSABLE PAIR-LEVEL FEATURE EXTRACTION MODULE
==============================================

This module exposes functions for computing the 16-feature baseline
feature vector for a (S1 entity, candidate entity) pair.

It is the canonical implementation that BOTH train.py and predict.py
should import from.  The feature names, normalisation logic, and all
similarity formulas are identical to those used in the Colab baseline
(student_resource/src/matching/extract_features.py) so that downstream
model training and inference are compatible with the pre-computed
baseline_features.tsv.

Design principles
-----------------
* Pure functions – no global mutable state.
* Safe for missing / empty / NaN inputs at every level.
* Non-destructive – input dicts / DataFrames are never modified.
* Deterministic – same inputs always produce the same numeric outputs.
* No NaN in the output vector; all values are finite floats or ints.

Feature set
-----------
NAME (6)
  name_exact                – 1 if normalised names match exactly
  name_jaccard              – Jaccard similarity on token sets
  name_token_overlap        – |A∩B| / max(|A|,|B|)
  name_levenshtein_ratio    – 1 − edit_dist / max(len(A),len(B))
  name_length_difference    – |len(A)−len(B)| / max(len(A),len(B))  ∈ [0,1]
  name_token_count_diff     – |#tokens(A) − #tokens(B)|  (raw int)

ADDRESS (7)
  address_exact             – 1 if normalised addresses match exactly
  address_jaccard
  address_token_overlap
  address_levenshtein_ratio
  address_length_difference
  address_token_count_diff
  address_missing           – 1 if either address is empty / NaN
  NOTE: when address_missing==1 all other address features are set to 0.

OTHER (3)
  country_match             – 1 if both countries are identical and non-empty
  source_is_s2              – 1 if candidate_source == "S2"
  source_is_s3              – 1 if candidate_source == "S3"

Usage example
-------------
    from src.features import FEATURE_COLS, extract_pair_features, extract_features_batch

    record_a = {"business_name": "Acme Corp", "business_address": "123 Main St", "country": "US"}
    record_b = {"business_name": "ACME Corporation", "business_address": "", "country": "US"}

    feat_dict = extract_pair_features(record_a, record_b, candidate_source="S2")
    # → {"name_exact": 0, "name_jaccard": 0.5, ..., "source_is_s2": 1, ...}

    # DataFrame-level batch helper:
    feat_df = extract_features_batch(pairs_df, records_lookup)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

#: Ordered list of the 16 feature column names produced by this module.
#: This order matches the Colab baseline and is the contract for downstream
#: train.py / predict.py consumers.
FEATURE_COLS: list[str] = [
    "name_exact",
    "name_jaccard",
    "name_token_overlap",
    "name_levenshtein_ratio",
    "name_length_difference",
    "name_token_count_diff",
    "address_exact",
    "address_jaccard",
    "address_token_overlap",
    "address_levenshtein_ratio",
    "address_length_difference",
    "address_token_count_diff",
    "address_missing",
    "country_match",
    "source_is_s2",
    "source_is_s3",
]

# ---------------------------------------------------------------------------
# TEXT NORMALISATION
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text: Any) -> str:
    """Normalise a raw field value to a clean, comparable string.

    Pipeline (matches Colab baseline exactly):
      1. Return ``""`` for any non-string / NaN / None input.
      2. Unicode NFC normalisation.
      3. Lowercase.
      4. Replace all non-word, non-space characters with a space.
      5. Collapse internal whitespace and strip leading/trailing spaces.

    Parameters
    ----------
    text:
        Raw value from the dataset (may be str, float NaN, None, etc.).

    Returns
    -------
    str
        Clean, normalised string, never ``None``.

    Examples
    --------
    >>> normalise("  Acme, Corp.  ")
    'acme corp'
    >>> normalise(None)
    ''
    >>> normalise(float('nan'))
    ''
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> list[str]:
    """Split a normalised string into word tokens.

    Parameters
    ----------
    text:
        A normalised (already lowercased / cleaned) string.

    Returns
    -------
    list[str]
        List of whitespace-separated tokens; empty list for empty input.

    Examples
    --------
    >>> tokenise("acme corp llc")
    ['acme', 'corp', 'llc']
    >>> tokenise("")
    []
    """
    return text.split() if text else []


# ---------------------------------------------------------------------------
# PRIMITIVE SIMILARITY FUNCTIONS
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.

    Uses the two-row DP algorithm – O(m·n) time, O(min(m,n)) space.
    Correct and fast for the short strings typical of business names /
    addresses (≤ ~200 chars).

    Parameters
    ----------
    a, b:
        Strings to compare (assumed already normalised).

    Returns
    -------
    int
        Minimum number of single-character edits (insertions, deletions,
        substitutions) to transform ``a`` into ``b``.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalised Levenshtein similarity ∈ [0, 1].

    Defined as ``1 − edit_distance(a, b) / max(len(a), len(b))``.
    Returns 1.0 when both strings are empty (perfect match by convention).

    Parameters
    ----------
    a, b:
        Normalised strings.

    Returns
    -------
    float
        Similarity score in [0.0, 1.0].  Higher = more similar.

    Examples
    --------
    >>> levenshtein_ratio("acme", "acme")
    1.0
    >>> levenshtein_ratio("", "")
    1.0
    >>> levenshtein_ratio("abc", "xyz")
    0.0
    """
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / max_len


def jaccard(toks_a: list[str], toks_b: list[str]) -> float:
    """Jaccard similarity on two token lists (treated as sets).

    Defined as ``|A ∩ B| / |A ∪ B|``.
    Returns 1.0 when both token lists are empty.

    Parameters
    ----------
    toks_a, toks_b:
        Token lists from ``tokenise()``.

    Returns
    -------
    float
        Jaccard similarity ∈ [0.0, 1.0].

    Examples
    --------
    >>> jaccard(["acme", "corp"], ["acme", "llc"])
    0.3333333333333333
    >>> jaccard([], [])
    1.0
    """
    sa, sb = set(toks_a), set(toks_b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def token_overlap(toks_a: list[str], toks_b: list[str]) -> float:
    """Overlap coefficient on two token lists (treated as sets).

    Defined as ``|A ∩ B| / max(|A|, |B|)``.
    Returns 1.0 when both token lists are empty.

    Parameters
    ----------
    toks_a, toks_b:
        Token lists from ``tokenise()``.

    Returns
    -------
    float
        Overlap coefficient ∈ [0.0, 1.0].

    Examples
    --------
    >>> token_overlap(["a", "b", "c"], ["a", "b"])
    0.6666666666666666
    >>> token_overlap([], [])
    1.0
    """
    sa, sb = set(toks_a), set(toks_b)
    denom = max(len(sa), len(sb))
    if denom == 0:
        return 1.0
    return len(sa & sb) / denom


def length_diff_norm(a: str, b: str) -> float:
    """Normalised absolute character-length difference ∈ [0, 1].

    Defined as ``|len(a) − len(b)| / max(len(a), len(b))``.
    Returns 0.0 when both strings are empty (no difference).

    Parameters
    ----------
    a, b:
        Normalised strings.

    Returns
    -------
    float
        Normalised length difference ∈ [0.0, 1.0].  0 = same length.

    Examples
    --------
    >>> length_diff_norm("abc", "abcdef")
    0.5
    >>> length_diff_norm("", "")
    0.0
    """
    la, lb = len(a), len(b)
    denom = max(la, lb)
    if denom == 0:
        return 0.0
    return abs(la - lb) / denom


# ---------------------------------------------------------------------------
# PAIR-LEVEL FEATURE EXTRACTION  (main public API)
# ---------------------------------------------------------------------------

def extract_pair_features(
    record_a: dict[str, Any],
    record_b: dict[str, Any],
    candidate_source: str,
) -> dict[str, float | int]:
    """Extract all 16 baseline features for a single (S1, candidate) pair.

    This is the single-pair entry point.  Both ``train.py`` and
    ``predict.py`` can call this function directly, or use the batch
    helper ``extract_features_batch`` for DataFrame-level processing.

    The function is non-destructive: ``record_a`` and ``record_b`` are
    read but never modified.

    Parameters
    ----------
    record_a:
        Dict for the S1 entity.  Expected keys (all optional, missing
        keys default to empty string):
          - ``"business_name"``   (str)
          - ``"business_address"`` (str)
          - ``"country"``         (str)
    record_b:
        Dict for the candidate entity (S2 or S3).  Same key schema as
        ``record_a``.
    candidate_source:
        Source identifier of the candidate entity, either ``"S2"`` or
        ``"S3"``.

    Returns
    -------
    dict[str, float | int]
        Mapping of ``FEATURE_COLS`` names to their computed values.
        All values are finite (no NaN, no None).

    Examples
    --------
    >>> rec_a = {"business_name": "Acme Corp", "business_address": "123 Main St", "country": "US"}
    >>> rec_b = {"business_name": "ACME Corporation", "business_address": "", "country": "US"}
    >>> feats = extract_pair_features(rec_a, rec_b, "S2")
    >>> feats["name_exact"]
    0
    >>> feats["country_match"]
    1
    >>> feats["address_missing"]
    1
    >>> feats["source_is_s2"]
    1
    """
    # ── 1. Normalise raw strings ──────────────────────────────────────────
    n1 = normalise(record_a.get("business_name", ""))
    n2 = normalise(record_b.get("business_name", ""))
    a1 = normalise(record_a.get("business_address", ""))
    a2 = normalise(record_b.get("business_address", ""))
    c1 = (record_a.get("country") or "").strip().lower()
    c2 = (record_b.get("country") or "").strip().lower()

    # ── 2. Tokenise ───────────────────────────────────────────────────────
    nt1, nt2 = tokenise(n1), tokenise(n2)
    at1, at2 = tokenise(a1), tokenise(a2)

    # ── 3. Name features ─────────────────────────────────────────────────
    name_exact            = int(n1 == n2)
    name_jac              = jaccard(nt1, nt2)
    name_tok_ov           = token_overlap(nt1, nt2)
    name_lev              = levenshtein_ratio(n1, n2)
    name_len_diff         = length_diff_norm(n1, n2)
    name_tok_diff         = abs(len(nt1) - len(nt2))

    # ── 4. Address features ───────────────────────────────────────────────
    # Gate: if either address is missing/empty, similarity scores are 0.
    addr_missing = int((not a1) or (not a2))
    if addr_missing:
        addr_exact    = 0
        addr_jac      = 0.0
        addr_tok_ov   = 0.0
        addr_lev      = 0.0
        addr_len_diff = 0.0
        addr_tok_diff = 0
    else:
        addr_exact    = int(a1 == a2)
        addr_jac      = jaccard(at1, at2)
        addr_tok_ov   = token_overlap(at1, at2)
        addr_lev      = levenshtein_ratio(a1, a2)
        addr_len_diff = length_diff_norm(a1, a2)
        addr_tok_diff = abs(len(at1) - len(at2))

    # ── 5. Other features ────────────────────────────────────────────────
    # country_match is 0 when either country is empty (avoids false matches)
    country_match = int(c1 == c2 and c1 != "")
    source_is_s2  = int(candidate_source == "S2")
    source_is_s3  = int(candidate_source == "S3")

    return {
        "name_exact":              name_exact,
        "name_jaccard":            name_jac,
        "name_token_overlap":      name_tok_ov,
        "name_levenshtein_ratio":  name_lev,
        "name_length_difference":  name_len_diff,
        "name_token_count_diff":   name_tok_diff,
        "address_exact":           addr_exact,
        "address_jaccard":         addr_jac,
        "address_token_overlap":   addr_tok_ov,
        "address_levenshtein_ratio": addr_lev,
        "address_length_difference": addr_len_diff,
        "address_token_count_diff":  addr_tok_diff,
        "address_missing":         addr_missing,
        "country_match":           country_match,
        "source_is_s2":            source_is_s2,
        "source_is_s3":            source_is_s3,
    }


# ---------------------------------------------------------------------------
# BATCH HELPER  (DataFrame-level)
# ---------------------------------------------------------------------------

def extract_features_batch(
    pairs: pd.DataFrame,
    records: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Compute features for a batch of pairs stored in a DataFrame.

    This is a convenience wrapper around ``extract_pair_features`` for
    use in train.py and predict.py where pairs arrive as a DataFrame.

    Parameters
    ----------
    pairs:
        DataFrame with AT LEAST the following columns:
          - ``"s1_entity_id"``       (str)
          - ``"candidate_entity_id"`` (str)
          - ``"candidate_source"``    (str, ``"S2"`` or ``"S3"``)
        Any additional columns (e.g. ``"label"``) are preserved and
        prepended to the output DataFrame.
    records:
        Lookup dict ``entity_id → {"business_name", "business_address",
        "country"}``.  Missing entity IDs are handled gracefully (all
        fields default to empty string, producing safe neutral feature
        values).

    Returns
    -------
    pd.DataFrame
        Original ``pairs`` columns followed by all 16 feature columns in
        ``FEATURE_COLS`` order.  No NaN values in the feature columns.

    Raises
    ------
    KeyError
        If ``pairs`` is missing any of the required identifier columns.

    Examples
    --------
    >>> import pandas as pd
    >>> pairs_df = pd.DataFrame([
    ...     {"s1_entity_id": "S1-1", "candidate_entity_id": "S2-1",
    ...      "candidate_source": "S2", "label": 1},
    ... ])
    >>> recs = {
    ...     "S1-1": {"business_name": "Acme", "business_address": "123 St", "country": "US"},
    ...     "S2-1": {"business_name": "Acme", "business_address": "123 St", "country": "US"},
    ... }
    >>> feat_df = extract_features_batch(pairs_df, recs)
    >>> feat_df["name_exact"].iloc[0]
    1
    """
    # Validate required columns
    required = {"s1_entity_id", "candidate_entity_id", "candidate_source"}
    missing_cols = required - set(pairs.columns)
    if missing_cols:
        raise KeyError(
            f"extract_features_batch: pairs DataFrame is missing required "
            f"columns: {sorted(missing_cols)}"
        )

    feature_rows: list[dict] = []

    for row in pairs.itertuples(index=False):
        s1_id   = row.s1_entity_id
        cand_id = row.candidate_entity_id
        src     = row.candidate_source

        r1 = records.get(s1_id, {})
        r2 = records.get(cand_id, {})

        feat = extract_pair_features(r1, r2, candidate_source=src)
        feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows, columns=FEATURE_COLS)

    # Reset pairs index to align before concatenation
    pairs_reset = pairs.reset_index(drop=True)
    return pd.concat([pairs_reset, feat_df], axis=1)
