"""
src/candidate_generation.py
============================
Member 4 deliverable — blocking / candidate generation.

Reduces the S1 × (S2 ∪ S3) cross-product to a tractable candidate set before
the M3 baseline matcher applies its features and model.

Design constraints
------------------
  • NEVER normalize raw text here.  All text is read from M2 Parquet cache
    columns: name_norm, address_norm, country.
  • NEVER compare every S1 row against every S2/S3 row (no O(N²)).
    All blocking uses inverted indices / dictionaries.
  • NEVER make final match decisions.  The ML model (M3) does that.
  • NEVER call M3 train.py, predict.py, or features.py.
  • Candidate generation for S2 and S3 runs separately, then results are
    unioned and deduplicated per S1 entity.

Blocking strategies (applied in parallel, results unioned)
----------------------------------------------------------
Block 1 — Exact normalized name
    Inverted index: name_norm → list of entity IDs.
    Any S1 entity whose normalized name exactly matches any S2/S3 entity
    is a candidate pair.

Block 2 — Distinctive-token overlap (name)
    Each entity is indexed by each token in its normalized name.
    A (S1, S2/S3) pair is a candidate when they share at least one
    "distinctive" token (≥ 4 characters and not a stop-word).
    This handles abbreviations and reordering.

Block 3 — Address-number blocking
    Entities that share at least one numeric token from the normalized
    address are grouped.  Street numbers are highly selective and rarely
    coincide by accident.  When neither entity has an address number, this
    block is skipped for that pair.

Block 4 — First-3-character name prefix (trigram prefix)
    Fast character-level prefix index: entities with the same 3-char prefix
    of their normalized name are grouped.  Catches OCR/typo variants where
    the first few characters are preserved.

Block 5 — Country + name-token
    For entities with the same country (when country is non-empty), add
    name-token candidates.  This restricts token blocking to the same
    geography and reduces false positives.

Public API
----------
  generate_name_candidates(s1, rhs, min_token_len)  → internal pairs df
  generate_address_candidates(s1, rhs)               → internal pairs df
  generate_prefix_candidates(s1, rhs, prefix_len)   → internal pairs df
  generate_country_token_candidates(s1, rhs)         → internal pairs df
  generate_candidates(s1, s2, s3, ...)               → (internal_df, official_df)
  evaluate_candidates(official_df, truth_map, n_s2, n_s3) → metrics dict

Input DataFrame columns (from M2 cache)
----------------------------------------
  entity_id, name_norm, address_norm, country
  (other columns are ignored)

Internal output
---------------
DataFrame with columns: source1_entity_id, candidate_entity_id

Official output
---------------
Produced by src/candidates.py:to_official_format()
  source1_entity_id, candidate_entity_ids   (one row per S1 entity)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Common business / address stop-words — excluded from token blocking
_STOP_TOKENS: frozenset[str] = frozenset(
    "the a an and or of in for to at by co company corporation limited "
    "private incorporated partnership llp ltd pvt inc corp group holdings "
    "services solutions enterprise enterprises international global national "
    "street avenue road drive lane court place highway parkway apartment suite "
    "avenue boulevard".split()
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_tokens(row, col_norm: str, col_tokens: str) -> list[str]:
    """
    Extract tokens for a row, preferring the pre-tokenized list from the
    new M2 cache (col_tokens) when available.  Falls back to splitting the
    normalized string (col_norm) when the list column is absent or None.

    This makes the blocking functions compatible with both:
      - new M2 cache (has name_tokens / address_tokens as Python lists)
      - synthetic DataFrames used in tests / notebooks (only have name_norm)
    """
    # Try pre-tokenized list first
    token_list = row.get(col_tokens) if hasattr(row, 'get') else getattr(row, col_tokens, None)
    if token_list is not None and isinstance(token_list, (list, tuple)):
        return [str(t) for t in token_list]
    # Fall back to splitting the norm string
    norm = row.get(col_norm) if hasattr(row, 'get') else getattr(row, col_norm, None)
    norm = str(norm or "")
    return norm.split()


def _distinctive_tokens(text_or_tokens, min_len: int = 4) -> list[str]:
    """Return tokens that are long enough and not stop-words."""
    tokens = text_or_tokens if isinstance(text_or_tokens, (list, tuple)) else str(text_or_tokens or "").split()
    return [
        t for t in tokens
        if len(t) >= min_len and t not in _STOP_TOKENS
    ]


def _numeric_tokens(text_or_tokens) -> list[str]:
    """Return tokens from *text_or_tokens* that contain at least one digit."""
    tokens = text_or_tokens if isinstance(text_or_tokens, (list, tuple)) else str(text_or_tokens or "").split()
    return [t for t in tokens if any(c.isdigit() for c in t)]


def _build_inverted_index(df: pd.DataFrame, key_fn) -> dict[str, list[str]]:
    """
    Build {key → [entity_id, ...]} from a DataFrame using *key_fn(row)*.
    *key_fn* returns a list of keys for each row.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        for key in key_fn(row):
            if key:
                index[key].append(row["entity_id"])
    return index


def _pairs_from_indices(
    s1: pd.DataFrame,
    rhs_index: dict[str, list[str]],
    key_fn,
    source_prefix: str,
) -> list[tuple[str, str]]:
    """
    For each S1 entity, look up its keys in *rhs_index* and emit
    (s1_entity_id, rhs_entity_id) tuples.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for _, row in s1.iterrows():
        s1_id = row["entity_id"]
        for key in key_fn(row):
            if not key:
                continue
            for cid in rhs_index.get(key, []):
                pair = (s1_id, cid)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)

    return pairs


def _to_df(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    return pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])


# ─────────────────────────────────────────────────────────────────────────────
# Block 1 — Exact normalized name
# ─────────────────────────────────────────────────────────────────────────────

def generate_name_exact_candidates(
    s1: pd.DataFrame,
    rhs: pd.DataFrame,
    rhs_index: dict | None = None,
) -> pd.DataFrame:
    """
    Exact normalized business name blocking.

    Parameters
    ----------
    s1, rhs : DataFrames with columns entity_id, name_norm.

    Returns
    -------
    Internal pairs DataFrame.
    """
    if rhs_index is None:
        rhs_index = defaultdict(list)
        for _, row in rhs.iterrows():
            name = str(row["name_norm"] or "")
            if name:
                rhs_index[name].append(row["entity_id"])

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in s1.iterrows():
        s1_id = row["entity_id"]
        name = str(row["name_norm"] or "")
        if not name:
            continue
        for cid in rhs_index.get(name, []):
            p = (s1_id, cid)
            if p not in seen:
                seen.add(p)
                pairs.append(p)

    return _to_df(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Block 2 — Distinctive name-token overlap
# ─────────────────────────────────────────────────────────────────────────────

def generate_name_candidates(
    s1: pd.DataFrame,
    rhs: pd.DataFrame,
    min_token_len: int = 4,
    rhs_index: dict | None = None,
) -> pd.DataFrame:
    """
    Distinctive-token blocking on normalized business name.

    Indexes RHS by every distinctive token (len ≥ min_token_len, not a
    stop-word).  For each S1 entity, emits a candidate pair with any RHS
    entity that shares at least one such token.

    Parameters
    ----------
    s1, rhs          : DataFrames with entity_id, name_norm.
    min_token_len    : Minimum token length to be considered distinctive.
    """
    # Build RHS index: token → [entity_ids]
    if rhs_index is None:
        rhs_index = defaultdict(list)
        for _, row in rhs.iterrows():
            for tok in _distinctive_tokens(_get_tokens(row, "name_norm", "name_tokens"), min_token_len):
                rhs_index[tok].append(row["entity_id"])

    # Look up each S1 entity
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in s1.iterrows():
        s1_id = row["entity_id"]
        for tok in _distinctive_tokens(_get_tokens(row, "name_norm", "name_tokens"), min_token_len):
            for cid in rhs_index.get(tok, []):
                p = (s1_id, cid)
                if p not in seen:
                    seen.add(p)
                    pairs.append(p)

    return _to_df(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Block 3 — Address-number blocking
# ─────────────────────────────────────────────────────────────────────────────

def generate_address_candidates(
    s1: pd.DataFrame,
    rhs: pd.DataFrame,
    rhs_index: dict | None = None,
) -> pd.DataFrame:
    """
    Address street-number blocking.

    Indexes RHS by numeric tokens in the normalized address.  Street numbers
    are highly selective; entities sharing a street number are likely
    co-located.  Pairs where either side has no numeric address tokens are
    skipped (no false signal from missing addresses).

    Parameters
    ----------
    s1, rhs : DataFrames with entity_id, address_norm.
    """
    if rhs_index is None:
        rhs_index = defaultdict(list)
        for _, row in rhs.iterrows():
            for tok in _numeric_tokens(_get_tokens(row, "address_norm", "address_tokens")):
                rhs_index[tok].append(row["entity_id"])

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in s1.iterrows():
        s1_id = row["entity_id"]
        num_toks = _numeric_tokens(_get_tokens(row, "address_norm", "address_tokens"))
        if not num_toks:
            continue
        for tok in num_toks:
            for cid in rhs_index.get(tok, []):
                p = (s1_id, cid)
                if p not in seen:
                    seen.add(p)
                    pairs.append(p)

    return _to_df(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Block 4 — Name prefix (first N characters)
# ─────────────────────────────────────────────────────────────────────────────

def generate_prefix_candidates(
    s1: pd.DataFrame,
    rhs: pd.DataFrame,
    prefix_len: int = 4,
) -> pd.DataFrame:
    """
    Character-prefix blocking on normalized business name.

    Entities whose normalized name starts with the same *prefix_len*-character
    prefix are grouped.  Catches OCR noise, one-character typos at the start,
    and closely-named businesses.

    Parameters
    ----------
    s1, rhs      : DataFrames with entity_id, name_norm.
    prefix_len   : Number of leading characters to use as key (default 4).
    """
    rhs_index: dict[str, list[str]] = defaultdict(list)
    for _, row in rhs.iterrows():
        name = str(row["name_norm"] or "")
        prefix = name[:prefix_len]
        if len(prefix) == prefix_len:          # only index if name is long enough
            rhs_index[prefix].append(row["entity_id"])

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in s1.iterrows():
        s1_id = row["entity_id"]
        name = str(row["name_norm"] or "")
        prefix = name[:prefix_len]
        if len(prefix) < prefix_len:
            continue
        for cid in rhs_index.get(prefix, []):
            p = (s1_id, cid)
            if p not in seen:
                seen.add(p)
                pairs.append(p)

    return _to_df(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Block 5 — Country + distinctive name-token
# ─────────────────────────────────────────────────────────────────────────────

def generate_country_token_candidates(
    s1: pd.DataFrame,
    rhs: pd.DataFrame,
    min_token_len: int = 4,
    rhs_index: dict | None = None,
) -> pd.DataFrame:
    """
    Country-scoped distinctive-token blocking.

    Only entities with the same non-empty country are compared.  Within each
    country bucket, the distinctive-token blocking is applied.  This reduces
    false positives from token blocking across geographies.

    Parameters
    ----------
    s1, rhs          : DataFrames with entity_id, name_norm, country.
    min_token_len    : Minimum token length for distinctive-token selection.
    """
    # Build RHS index: (country, token) → [entity_ids]
    if rhs_index is None:
        rhs_index = defaultdict(list)
        for _, row in rhs.iterrows():
            ctry = str(row.get("country", "") or "").strip().lower()
            if not ctry:
                continue
            for tok in _distinctive_tokens(_get_tokens(row, "name_norm", "name_tokens"), min_token_len):
                rhs_index[(ctry, tok)].append(row["entity_id"])

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in s1.iterrows():
        s1_id = row["entity_id"]
        ctry = str(row.get("country", "") or "").strip().lower()
        if not ctry:
            continue
        for tok in _distinctive_tokens(_get_tokens(row, "name_norm", "name_tokens"), min_token_len):
            for cid in rhs_index.get((ctry, tok), []):
                p = (s1_id, cid)
                if p not in seen:
                    seen.add(p)
                    pairs.append(p)

    return _to_df(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Block 6 — rapidfuzz token-sort ratio (top-k per S1 bucket)
# ─────────────────────────────────────────────────────────────────────────────

def generate_fuzzy_candidates(
    s1: pd.DataFrame,
    rhs: pd.DataFrame,
    prefix_len: int = 3,
    score_cutoff: float = 80.0,
    max_per_s1: int = 50,
    rhs_buckets: dict | None = None,
) -> pd.DataFrame:
    """
    Fuzzy name blocking using rapidfuzz token_sort_ratio.

    Within each name-prefix bucket (first *prefix_len* chars), compute
    token_sort_ratio between S1 and RHS normalized names.  Emit pairs
    above *score_cutoff*.  Capped at *max_per_s1* per S1 entity to bound
    volume.

    This catches name-reorder, minor spelling variation, and legal-suffix
    differences not resolved by normalization.

    Parameters
    ----------
    s1, rhs       : DataFrames with entity_id, name_norm.
    prefix_len    : Prefix length for bucket partitioning (default 3).
    score_cutoff  : Minimum token_sort_ratio score (0–100, default 80).
    max_per_s1    : Maximum candidates emitted per S1 entity (default 50).
    """
    try:
        from rapidfuzz.fuzz import token_sort_ratio
    except ImportError:
        # Graceful degradation — skip fuzzy block if rapidfuzz not installed
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])

    # Build RHS bucket: prefix → [(entity_id, name_norm)]
    if rhs_buckets is None:
        rhs_buckets = defaultdict(list)
        for _, row in rhs.iterrows():
            name = str(row["name_norm"] or "")
            prefix = name[:prefix_len]
            if len(prefix) == prefix_len:
                rhs_buckets[prefix].append((row["entity_id"], name))

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in s1.iterrows():
        s1_id = row["entity_id"]
        name = str(row["name_norm"] or "")
        prefix = name[:prefix_len]
        if len(prefix) < prefix_len:
            continue

        bucket = rhs_buckets.get(prefix, [])
        count = 0
        for cid, cand_name in bucket:
            if count >= max_per_s1:
                break
            if token_sort_ratio(name, cand_name) >= score_cutoff:
                p = (s1_id, cid)
                if p not in seen:
                    seen.add(p)
                    pairs.append(p)
                    count += 1

    return _to_df(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Master generate_candidates()
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    blocks: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full multi-block candidate generation pipeline.

    Parameters
    ----------
    s1, s2, s3 : DataFrames loaded from M2 Parquet cache.
                 Required columns: entity_id, name_norm,
                 address_norm, country.
    blocks     : List of block names to run. Defaults to all six blocks:
                 ["exact", "token", "address", "prefix", "country_token", "fuzzy"]
    verbose    : If True, print per-block and total statistics.

    Returns
    -------
    (internal_df, official_df)
      internal_df : row-per-pair DataFrame with columns
                    [source1_entity_id, candidate_entity_id]
      official_df : one-row-per-S1 DataFrame (via src/candidates.to_official_format)
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from candidates import to_official_format

    if blocks is None:
        blocks = ["exact", "token", "address", "prefix", "country_token", "fuzzy"]

    all_s1_ids = s1["entity_id"].tolist()

    # --- Run each block for S2 and S3 independently, then union ---
    block_fns = {
        "exact":         generate_name_exact_candidates,
        "token":         generate_name_candidates,
        "address":       generate_address_candidates,
        "prefix":        generate_prefix_candidates,
        "country_token": generate_country_token_candidates,
        "fuzzy":         generate_fuzzy_candidates,
    }

    all_parts: list[pd.DataFrame] = []
    failed_blocks: list[str] = []

    for block_name in blocks:
        fn = block_fns.get(block_name)
        if fn is None:
            continue
        for rhs_name, rhs in [("S2", s2), ("S3", s3)]:
            try:
                part = fn(s1, rhs)
                if verbose:
                    print(f"  Block [{block_name:12s}] x {rhs_name}: {len(part):>8,} pairs")
                all_parts.append(part)
            except Exception as exc:
                tag = f"{block_name}/{rhs_name}"
                failed_blocks.append(tag)
                print(f"  Block [{block_name}] x {rhs_name} FAILED: {exc}", flush=True)

    if failed_blocks:
        print(
            f"\n  *** {len(failed_blocks)} block(s) FAILED: {failed_blocks} ***",
            flush=True,
        )
        raise RuntimeError(
            f"Candidate generation failed for {len(failed_blocks)} block(s): "
            f"{failed_blocks}"
        )

    if not all_parts:
        internal_df = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id"]
        )
    else:
        raw = pd.concat(all_parts, ignore_index=True)
        # Deduplicate across blocks
        internal_df = raw.drop_duplicates(
            subset=["source1_entity_id", "candidate_entity_id"]
        ).reset_index(drop=True)

    if verbose:
        print(f"\n  Total unique pairs (after dedup): {len(internal_df):,}")
        s1_covered = internal_df["source1_entity_id"].nunique()
        print(f"  S1 entities with >=1 candidate: {s1_covered:,} / {len(all_s1_ids):,}")

    official_df = to_official_format(internal_df, all_s1_ids=all_s1_ids)
    return internal_df, official_df


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_candidates(
    official_df: pd.DataFrame,
    truth_map: dict[str, set[str]],
    n_s2: int,
    n_s3: int,
) -> dict:
    """
    Evaluate candidate generation against ground truth.

    Parameters
    ----------
    official_df : Output of to_official_format() — one row per S1 entity.
    truth_map   : {s1_id → set of true matched IDs} from train_ground_truth.tsv.
    n_s2, n_s3  : Total S2 and S3 entity counts (for reduction ratio).

    Returns
    -------
    dict with metrics:
      n_s1, total_candidate_ids, avg_per_s1, median_per_s1, max_per_s1,
      n_zero_candidates, cross_product_size, reduction_ratio,
      candidate_recall, true_matches_found, true_matches_lost
    """
    import numpy as np

    counts = official_df["candidate_entity_ids"].apply(
        lambda x: len(x.split(",")) if isinstance(x, str) and x.strip() else 0
    )

    n_s1  = len(official_df)
    total = int(counts.sum())
    cross = n_s1 * (n_s2 + n_s3)
    reduction = 1.0 - total / cross if cross > 0 else 0.0

    # Recall
    found = missed = 0
    for _, row in official_df.iterrows():
        sid  = row["source1_entity_id"]
        raw  = row["candidate_entity_ids"]
        cids = set(raw.split(",")) if isinstance(raw, str) and raw.strip() else set()
        true_set = truth_map.get(sid, set())
        found  += len(true_set & cids)
        missed += len(true_set - cids)

    total_true = found + missed

    return {
        "n_s1":               n_s1,
        "total_candidate_ids": total,
        "avg_per_s1":         round(float(counts.mean()), 2),
        "median_per_s1":      float(np.median(counts)),
        "max_per_s1":         int(counts.max()),
        "n_zero_candidates":  int((counts == 0).sum()),
        "cross_product_size": cross,
        "reduction_ratio":    round(reduction, 6),
        "true_matches_found": found,
        "true_matches_lost":  missed,
        "candidate_recall":   round(found / total_true, 6) if total_true else None,
    }




def generate_candidates_memory_safe(
    split: str,
    cache_dir: str | Path,
    blocks: list[str] | None = None,
    verbose: bool = True,
    chunk_size: int = 50_000
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int]:
    """
    Memory-safe version of generate_candidates that processes one RHS at a time,
    chunks S1, and spills intermediate block results to disk. DuckDB is used for
    out-of-core merging.
    """
    import gc
    import math
    import tempfile
    import sys
    import duckdb
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cache import load_cache
    from candidates import to_official_format
    from collections import defaultdict

    if blocks is None:
        blocks = ["exact", "token", "address", "prefix", "country_token", "fuzzy"]

    cols = ["entity_id", "name_norm", "address_norm", "country"]

    if verbose:
        print(f"\nLoading S1 ({split}) with {cols}...", flush=True)
    
    s1 = load_cache(split, "source1", cache_dir, columns=cols)
    n_s1 = len(s1)
    all_s1_ids = s1["entity_id"].tolist()
    
    failed_blocks: list[str] = []
    
    n_s2 = 0
    n_s3 = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        part_files = []

        for rhs_name, src_name in [("S2", "source2"), ("S3", "source3")]:
            if verbose:
                print(f"\nLoading {rhs_name} ({src_name}) with {cols}...", flush=True)
            rhs = load_cache(split, src_name, cache_dir, columns=cols)
            
            if src_name == "source2":
                n_s2 = len(rhs)
            else:
                n_s3 = len(rhs)
                
            n_chunks = math.ceil(n_s1 / chunk_size)
            
            for block_name in blocks:
                try:
                    # 1. Build RHS index once per block
                    if verbose:
                        print(f"  Building index for {block_name} x {rhs_name}...", flush=True)
                    
                    rhs_index = None
                    rhs_buckets = None
                    
                    if block_name == "exact":
                        rhs_index = defaultdict(list)
                        for _, row in rhs.iterrows():
                            name = str(row["name_norm"] or "")
                            if name:
                                rhs_index[name].append(row["entity_id"])
                    
                    elif block_name == "token":
                        rhs_index = defaultdict(list)
                        for _, row in rhs.iterrows():
                            for tok in _distinctive_tokens(_get_tokens(row, "name_norm", "name_tokens"), 4):
                                rhs_index[tok].append(row["entity_id"])
                                
                    elif block_name == "address":
                        rhs_index = defaultdict(list)
                        for _, row in rhs.iterrows():
                            for tok in _numeric_tokens(_get_tokens(row, "address_norm", "address_tokens")):
                                rhs_index[tok].append(row["entity_id"])
                                
                    elif block_name == "prefix":
                        rhs_index = defaultdict(list)
                        for _, row in rhs.iterrows():
                            name = str(row["name_norm"] or "")
                            prefix = name[:3]
                            if len(prefix) == 3:
                                rhs_index[prefix].append(row["entity_id"])
                                
                    elif block_name == "country_token":
                        rhs_index = defaultdict(list)
                        for _, row in rhs.iterrows():
                            ctry = str(row.get("country", "") or "").strip().lower()
                            if not ctry:
                                continue
                            for tok in _distinctive_tokens(_get_tokens(row, "name_norm", "name_tokens"), 4):
                                rhs_index[(ctry, tok)].append(row["entity_id"])
                                
                    elif block_name == "fuzzy":
                        rhs_buckets = defaultdict(list)
                        for _, row in rhs.iterrows():
                            name = str(row["name_norm"] or "")
                            prefix = name[:3]
                            if len(prefix) == 3:
                                rhs_buckets[prefix].append((row["entity_id"], name))

                    # 2. Process S1 in chunks
                    block_pairs_found = 0
                    for chunk_idx in range(n_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min((chunk_idx + 1) * chunk_size, n_s1)
                        s1_chunk = s1.iloc[start_idx:end_idx]
                        
                        if block_name == "exact":
                            part = generate_name_exact_candidates(s1_chunk, rhs, rhs_index=rhs_index)
                        elif block_name == "token":
                            part = generate_name_candidates(s1_chunk, rhs, min_token_len=4, rhs_index=rhs_index)
                        elif block_name == "address":
                            part = generate_address_candidates(s1_chunk, rhs, rhs_index=rhs_index)
                        elif block_name == "prefix":
                            part = generate_prefix_candidates(s1_chunk, rhs, prefix_len=3, rhs_index=rhs_index)
                        elif block_name == "country_token":
                            part = generate_country_token_candidates(s1_chunk, rhs, min_token_len=4, rhs_index=rhs_index)
                        elif block_name == "fuzzy":
                            part = generate_fuzzy_candidates(s1_chunk, rhs, prefix_len=3, score_cutoff=80.0, max_per_s1=50, rhs_buckets=rhs_buckets)
                        else:
                            continue
                            
                        block_pairs_found += len(part)
                        
                        if not part.empty:
                            out_path = temp_dir_path / f"{rhs_name}_{block_name}_{chunk_idx}.parquet"
                            part.to_parquet(out_path, index=False, engine="pyarrow")
                            part_files.append(str(out_path))
                            
                        del part
                        gc.collect()
                        
                        if verbose and (chunk_idx + 1) % 10 == 0:
                            print(f"  source={rhs_name} block={block_name} chunk={chunk_idx+1}/{n_chunks} pairs={block_pairs_found:,}", flush=True)
                            
                    if verbose:
                        print(f"  Block [{block_name:12s}] x {rhs_name}: {block_pairs_found:>8,} pairs TOTAL", flush=True)

                    del rhs_index
                    del rhs_buckets
                    gc.collect()

                except Exception as exc:
                    tag = f"{block_name}/{rhs_name}"
                    failed_blocks.append(tag)
                    print(f"  Block [{block_name}] x {rhs_name} FAILED: {exc}", flush=True)
            
            del rhs
            gc.collect()

        if failed_blocks:
            print(
                f"\n*** {len(failed_blocks)} block(s) FAILED: {failed_blocks} ***",
                flush=True,
            )
            raise RuntimeError(
                f"Candidate generation failed for {len(failed_blocks)} block(s): "
                f"{failed_blocks}"
            )

        if not part_files:
            internal_df = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
        else:
            if verbose:
                print(f"\nMerging {len(part_files)} block files with DuckDB...", flush=True)
            
            con = duckdb.connect(database=':memory:')
            
            # Using read_parquet with a list of files handles all of them natively out-of-core
            # Note: We must format the string list properly for DuckDB
            parquet_list_str = ", ".join([f"'{p}'" for p in part_files])
            query = f"""
                SELECT DISTINCT source1_entity_id, candidate_entity_id
                FROM read_parquet([{parquet_list_str}])
            """
            internal_df = con.execute(query).df()
            con.close()
            gc.collect()

    if verbose:
        print(f"\n  Total unique pairs (after dedup): {len(internal_df):,}")
        s1_covered = internal_df["source1_entity_id"].nunique()
        print(f"  S1 entities with >=1 candidate: {s1_covered:,} / {len(all_s1_ids):,}")

    official_df = to_official_format(internal_df, all_s1_ids=all_s1_ids)
    
    return internal_df, official_df, n_s1, n_s2, n_s3


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cache import load_all_cache, build_cache, cache_exists
    from candidates import write_candidates, validate_format

    parser = argparse.ArgumentParser(description="M4 candidate generation.")
    parser.add_argument("--data-dir",   default="dataset/train")
    parser.add_argument("--cache-dir",  default="cache")
    parser.add_argument("--output",     default="output/candidate_pairs.tsv")
    parser.add_argument("--split",      default="train", choices=["train", "test"])
    parser.add_argument("--gt",         default=None, help="Ground truth TSV for evaluation")
    parser.add_argument("--blocks",     nargs="+", default=None)
    parser.add_argument("--build-cache", action="store_true")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    _all_cached = all(
        cache_exists(args.split, src, cache_dir)
        for src in ("source1", "source2", "source3")
    )
    if args.build_cache or not _all_cached:
        print("Building M2 cache...")
        from cache import build_cache
        for src in ("source1", "source2", "source3"):
            build_cache(args.split, src, data_dir, cache_dir, force=args.build_cache)

    print("\nRunning memory-safe candidate generation...")
    internal_df, official_df, n_s1, n_s2, n_s3 = generate_candidates_memory_safe(
        split=args.split, cache_dir=cache_dir, blocks=args.blocks, verbose=True
    )

    # Hard-stop proxy: every S1 entity must appear exactly once in official_df.
    # If blocks silently failed in a way that corrupted the aggregation, this catches it.
    if len(official_df) != n_s1:
        print(f"\nFATAL: official_df has {len(official_df)} rows, expected {n_s1}. Aborting.")
        sys.exit(1)

    # Validate format
    errors = validate_format(official_df)
    if errors:
        for e in errors:
            print(f"FORMAT ERROR: {e}")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_candidates(official_df, args.output)
    print(f"\nWrote: {args.output}  ({len(official_df):,} rows)")

    # Evaluate if ground truth provided
    if args.gt:
        import pandas as pd
        gt = pd.read_csv(args.gt, sep="\t")
        truth_map: dict[str, set[str]] = {}
        for _, row in gt.iterrows():
            val = row["matched_entity_ids"]
            truth_map[row["source1_entity_id"]] = (
                {x.strip() for x in str(val).split(",") if x.strip()}
                if pd.notna(val) and str(val).strip() else set()
            )
        metrics = evaluate_candidates(official_df, truth_map, len(s2), len(s3))
        print("\nCandidate evaluation:")
        print(json.dumps(metrics, indent=2))

        # Save stats alongside output file
        stats_path = str(Path(args.output).with_suffix(".stats.json"))
        with open(stats_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Stats  : {stats_path}")

