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
    out_file: str | Path,
    blocks: list[str] | None = None,
    verbose: bool = True,
    chunk_size: int = 25_000
) -> tuple[int, int, int]:
    import gc
    import duckdb
    import pyarrow.parquet as pq
    import sys
    import tempfile
    import math
    import time
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from candidate_generation import _STOP_TOKENS
    
    if blocks is None:
        blocks = ["exact", "token", "address", "prefix", "country_token", "fuzzy"]

    cache_dir = Path(cache_dir)
    s1_path = cache_dir / f"{split}_source1.parquet"
    
    n_s1 = pq.read_metadata(s1_path).num_rows
    con = duckdb.connect(':memory:')
    
    if verbose:
        print()
        print(f"Loading S1 into DuckDB for chunking...", flush=True)
    con.execute(f"CREATE TABLE s1_table AS SELECT *, row_number() OVER () - 1 as rn FROM read_parquet('{s1_path}')")
    
    all_s1_ids = con.execute("SELECT entity_id FROM s1_table ORDER BY rn").df()['entity_id'].tolist()
    
    con.execute("CREATE TABLE stop_words (token VARCHAR)")
    con.executemany("INSERT INTO stop_words VALUES (?)", [(t,) for t in _STOP_TOKENS])
    
    failed_blocks = []
    
    n_s2 = pq.read_metadata(cache_dir / f"{split}_source2.parquet").num_rows
    n_s3 = pq.read_metadata(cache_dir / f"{split}_source3.parquet").num_rows

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        temp_dir_path = Path(temp_dir)
        part_files = []
        
        n_chunks = math.ceil(n_s1 / chunk_size)
        
        for rhs_name, src_name in [("S2", "source2"), ("S3", "source3")]:
            rhs_path = cache_dir / f"{split}_{src_name}.parquet"
            if verbose:
                print()
                print(f"Processing {rhs_name} ({src_name})...", flush=True)
                
            con.execute(f"CREATE OR REPLACE VIEW rhs_view AS SELECT * FROM read_parquet('{rhs_path}')")
            
            if verbose:
                print(f"  Building RHS relations for {rhs_name}...", flush=True)
            
            t0 = time.time()
            if "token" in blocks:
                con.execute("""
                    CREATE OR REPLACE TABLE rhs_token_dist AS
                    SELECT entity_id, token FROM (
                        SELECT entity_id, unnest(string_split(name_norm, ' ')) as token
                        FROM rhs_view WHERE name_norm IS NOT NULL
                    )
                    WHERE length(token) >= 4 AND token NOT IN (SELECT token FROM stop_words)
                """)
            if "address" in blocks:
                con.execute("""
                    CREATE OR REPLACE TABLE rhs_addr_num AS
                    SELECT entity_id, token FROM (
                        SELECT entity_id, unnest(string_split(address_norm, ' ')) as token
                        FROM rhs_view WHERE address_norm IS NOT NULL
                    )
                    WHERE regexp_matches(token, '[0-9]')
                """)
            if "country_token" in blocks:
                con.execute("""
                    CREATE OR REPLACE TABLE rhs_country_dist AS
                    SELECT entity_id, ctry, token FROM (
                        SELECT entity_id, lower(trim(country)) as ctry, unnest(string_split(name_norm, ' ')) as token
                        FROM rhs_view WHERE name_norm IS NOT NULL AND country IS NOT NULL AND trim(country) != ''
                    )
                    WHERE length(token) >= 4 AND token NOT IN (SELECT token FROM stop_words)
                """)
            if verbose:
                print(f"  RHS relations built in {time.time() - t0:.1f}s", flush=True)
            
            for block_name in blocks:
                if verbose:
                    print(f"  Running Block [{block_name:12s}] x {rhs_name}...", flush=True)
                
                t_block = time.time()
                block_pairs = 0
                
                try:
                    for chunk_idx in range(n_chunks):
                        start_row = chunk_idx * chunk_size
                        end_row = start_row + chunk_size
                        
                        con.execute(f"CREATE OR REPLACE VIEW s1_chunk_view AS SELECT * FROM s1_table WHERE rn >= {start_row} AND rn < {end_row}")
                        
                        out_path = temp_dir_path / f"{rhs_name}_{block_name}_{chunk_idx}.parquet"
                        
                        if block_name == "exact":
                            query = f"""
                                COPY (
                                    SELECT DISTINCT s1.entity_id AS source1_entity_id, rhs.entity_id AS candidate_entity_id
                                    FROM s1_chunk_view s1
                                    JOIN rhs_view rhs ON s1.name_norm = rhs.name_norm
                                    WHERE s1.name_norm IS NOT NULL AND s1.name_norm != ''
                                ) TO '{out_path}' (FORMAT PARQUET)
                            """
                            con.execute(query)
                            
                        elif block_name == "prefix":
                            query = f"""
                                COPY (
                                    SELECT DISTINCT s1.entity_id AS source1_entity_id, rhs.entity_id AS candidate_entity_id
                                    FROM s1_chunk_view s1
                                    JOIN rhs_view rhs ON substr(s1.name_norm, 1, 4) = substr(rhs.name_norm, 1, 4)
                                    WHERE s1.name_norm IS NOT NULL AND length(s1.name_norm) >= 4
                                      AND rhs.name_norm IS NOT NULL AND length(rhs.name_norm) >= 4
                                ) TO '{out_path}' (FORMAT PARQUET)
                            """
                            con.execute(query)
                            
                        elif block_name == "token":
                            query = f"""
                                COPY (
                                    WITH s1_tokens AS (
                                        SELECT entity_id, unnest(string_split(name_norm, ' ')) as token
                                        FROM s1_chunk_view WHERE name_norm IS NOT NULL
                                    ),
                                    s1_dist AS (
                                        SELECT entity_id, token FROM s1_tokens
                                        WHERE length(token) >= 4 AND token NOT IN (SELECT token FROM stop_words)
                                    )
                                    SELECT DISTINCT s1_dist.entity_id AS source1_entity_id, rhs_token_dist.entity_id AS candidate_entity_id
                                    FROM s1_dist
                                    JOIN rhs_token_dist ON s1_dist.token = rhs_token_dist.token
                                ) TO '{out_path}' (FORMAT PARQUET)
                            """
                            con.execute(query)
                            
                        elif block_name == "address":
                            query = f"""
                                COPY (
                                    WITH s1_tokens AS (
                                        SELECT entity_id, unnest(string_split(address_norm, ' ')) as token
                                        FROM s1_chunk_view WHERE address_norm IS NOT NULL
                                    ),
                                    s1_num AS (
                                        SELECT entity_id, token FROM s1_tokens
                                        WHERE regexp_matches(token, '[0-9]')
                                    )
                                    SELECT DISTINCT s1_num.entity_id AS source1_entity_id, rhs_addr_num.entity_id AS candidate_entity_id
                                    FROM s1_num
                                    JOIN rhs_addr_num ON s1_num.token = rhs_addr_num.token
                                ) TO '{out_path}' (FORMAT PARQUET)
                            """
                            con.execute(query)
                            
                        elif block_name == "country_token":
                            query = f"""
                                COPY (
                                    WITH s1_tokens AS (
                                        SELECT entity_id, lower(trim(country)) as ctry, unnest(string_split(name_norm, ' ')) as token
                                        FROM s1_chunk_view WHERE name_norm IS NOT NULL AND country IS NOT NULL AND trim(country) != ''
                                    ),
                                    s1_dist AS (
                                        SELECT entity_id, ctry, token FROM s1_tokens
                                        WHERE length(token) >= 4 AND token NOT IN (SELECT token FROM stop_words)
                                    )
                                    SELECT DISTINCT s1_dist.entity_id AS source1_entity_id, rhs_country_dist.entity_id AS candidate_entity_id
                                    FROM s1_dist
                                    JOIN rhs_country_dist ON s1_dist.ctry = rhs_country_dist.ctry AND s1_dist.token = rhs_country_dist.token
                                ) TO '{out_path}' (FORMAT PARQUET)
                            """
                            con.execute(query)
                            
                        elif block_name == "fuzzy":
                            try:
                                from rapidfuzz.fuzz import token_sort_ratio
                            except ImportError:
                                continue
                                
                            fuzzy_join_path = temp_dir_path / f"{rhs_name}_fuzzy_join_{chunk_idx}.parquet"
                            con.execute(f"""
                                COPY (
                                    SELECT s1.entity_id as s1_id, s1.name_norm as s1_name,
                                           rhs.entity_id as rhs_id, rhs.name_norm as rhs_name
                                    FROM s1_chunk_view s1
                                    JOIN rhs_view rhs ON substr(s1.name_norm, 1, 3) = substr(rhs.name_norm, 1, 3)
                                    WHERE s1.name_norm IS NOT NULL AND length(s1.name_norm) >= 3
                                      AND rhs.name_norm IS NOT NULL AND length(rhs.name_norm) >= 3
                                ) TO '{fuzzy_join_path}' (FORMAT PARQUET)
                            """)
                            
                            import pandas as pd
                            import pyarrow as pa
                            try:
                                parquet_file = pq.ParquetFile(fuzzy_join_path)
                                writer = None
                                raw_out = temp_dir_path / f"{out_path.name}_raw.parquet"
                                
                                for batch in parquet_file.iter_batches(batch_size=200_000):
                                    df_batch = batch.to_pandas()
                                    df_batch["score"] = df_batch.apply(lambda row: token_sort_ratio(row["s1_name"], row["rhs_name"]), axis=1)
                                    filtered = df_batch[df_batch["score"] >= 80.0]
                                    
                                    if not filtered.empty:
                                        batch_out = pa.RecordBatch.from_pandas(filtered[["s1_id", "rhs_id", "score"]])
                                        if writer is None:
                                            writer = pq.ParquetWriter(raw_out, batch_out.schema)
                                        writer.write_batch(batch_out)
                                        
                                if writer is not None:
                                    writer.close()
                                    con.execute(f"""
                                        COPY (
                                            WITH ranked AS (
                                                SELECT s1_id AS source1_entity_id, rhs_id AS candidate_entity_id,
                                                       row_number() OVER (PARTITION BY s1_id ORDER BY score DESC) as rn
                                                FROM read_parquet('{raw_out}')
                                            )
                                            SELECT source1_entity_id, candidate_entity_id
                                            FROM ranked
                                            WHERE rn <= 50
                                        ) TO '{out_path}' (FORMAT PARQUET)
                                    """)
                                    raw_out.unlink(missing_ok=True)
                            except FileNotFoundError:
                                pass
                            finally:
                                fuzzy_join_path.unlink(missing_ok=True)

                        if out_path.exists():
                            chunk_pairs = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
                            if chunk_pairs > 0:
                                part_files.append(str(out_path))
                                block_pairs += chunk_pairs
                            else:
                                out_path.unlink(missing_ok=True)
                                
                        if verbose and (chunk_idx + 1) % 5 == 0:
                            print(f"    source={rhs_name} block={block_name} chunk={chunk_idx+1}/{n_chunks}", flush=True)

                    if verbose:
                        print(f"  -> {block_name} x {rhs_name}: {block_pairs:>8,} pairs TOTAL ({time.time() - t_block:.1f}s)", flush=True)

                except Exception as exc:
                    tag = f"{block_name}/{rhs_name}"
                    failed_blocks.append(tag)
                    print(f"  Block [{block_name}] x {rhs_name} FAILED: {exc}", flush=True)
                    
            con.execute("DROP VIEW IF EXISTS rhs_view")
            con.execute("DROP TABLE IF EXISTS rhs_token_dist")
            con.execute("DROP TABLE IF EXISTS rhs_addr_num")
            con.execute("DROP TABLE IF EXISTS rhs_country_dist")
            gc.collect()

        if failed_blocks:
            print()
            print(f"  *** {len(failed_blocks)} block(s) FAILED: {failed_blocks} ***", flush=True)
            raise RuntimeError(f"Candidate generation failed for {len(failed_blocks)} block(s): {failed_blocks}")

        if not part_files:
            import pandas as pd
            pd.DataFrame(columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(out_file, sep="	", index=False)
        else:
            if verbose:
                print()
                print(f"Merging {len(part_files)} block files with DuckDB...", flush=True)
            
            parquet_list_str = ", ".join([f"'{p}'" for p in part_files])
            
            t_merge = time.time()
            con.execute(f"""
                COPY (
                    WITH distinct_pairs AS (
                        SELECT DISTINCT source1_entity_id, candidate_entity_id
                        FROM read_parquet([{parquet_list_str}])
                    ),
                    agg_pairs AS (
                        SELECT source1_entity_id, string_agg(candidate_entity_id, ',') as candidate_entity_ids
                        FROM distinct_pairs
                        GROUP BY source1_entity_id
                    ),
                    s1_all AS (
                        SELECT entity_id as source1_entity_id
                        FROM s1_table
                    )
                    SELECT s1_all.source1_entity_id, COALESCE(agg_pairs.candidate_entity_ids, '') as candidate_entity_ids
                    FROM s1_all
                    LEFT JOIN agg_pairs ON s1_all.source1_entity_id = agg_pairs.source1_entity_id
                    ORDER BY s1_all.source1_entity_id
                ) TO '{out_file}' (FORMAT CSV, DELIMITER '	', HEADER)
            """)
            if verbose:
                print(f"  Merged in {time.time() - t_merge:.1f}s", flush=True)

    if verbose:
        s1_covered = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{out_file}', delim='	', header=True) WHERE candidate_entity_ids IS NOT NULL AND candidate_entity_ids != ''").fetchone()[0]
        print()
        print(f"  S1 entities with >=1 candidate: {s1_covered:,} / {n_s1:,}")

    con.close()
    return n_s1, n_s2, n_s3


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
    out_path_obj = Path(args.output)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    n_s1, n_s2, n_s3 = generate_candidates_memory_safe(
        split=args.split, cache_dir=cache_dir, out_file=args.output, blocks=args.blocks, verbose=True
    )

    import duckdb
    row_count = duckdb.execute(f"SELECT COUNT(*) FROM read_csv_auto('{args.output}', delim='\t', header=True)").fetchone()[0]
    if row_count != n_s1:
        print(f"\nFATAL: output file has {row_count} rows, expected {n_s1}. Aborting.")
        sys.exit(1)

    print(f"\nWrote: {args.output}  ({row_count:,} rows)")

    # Evaluate if ground truth provided
    if args.gt:
        import pandas as pd
        import csv
        gt = pd.read_csv(args.gt, sep="\t")
        truth_map: dict[str, set[str]] = {}
        for _, row in gt.iterrows():
            val = row["matched_entity_ids"]
            truth_map[row["source1_entity_id"]] = (
                {x.strip() for x in str(val).split(",") if x.strip()}
                if pd.notna(val) and str(val).strip() else set()
            )
            
        found = 0
        missed = 0
        n_with_candidates = 0
        n_empty = 0
        total_candidate_ids = 0
        max_per_s1 = 0
        
        with open(args.output, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sid = row["source1_entity_id"]
                cids_str = row["candidate_entity_ids"] or ""
                if cids_str.strip():
                    cids = set(cids_str.split(","))
                    n_with_candidates += 1
                else:
                    cids = set()
                    n_empty += 1
                
                count = len(cids)
                total_candidate_ids += count
                if count > max_per_s1:
                    max_per_s1 = count
                    
                true_matches = truth_map.get(sid, set())
                found += len(true_matches & cids)
                missed += len(true_matches - cids)
                
        total_true = found + missed
        metrics = {
            "n_s1": n_s1,
            "n_with_candidates": n_with_candidates,
            "n_empty": n_empty,
            "total_candidate_ids": total_candidate_ids,
            "avg_per_s1": round(total_candidate_ids / n_s1, 2) if n_s1 else 0.0,
            "max_per_s1": max_per_s1,
            "candidate_recall": round(found / total_true, 6) if total_true else None,
            "true_matches_lost": missed,
            "reduction_ratio": None
        }
        
        print("\nCandidate evaluation:")
        print(json.dumps(metrics, indent=2))

        # Save stats alongside output file
        stats_path = str(Path(args.output).with_suffix(".stats.json"))
        with open(stats_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Stats  : {stats_path}")

