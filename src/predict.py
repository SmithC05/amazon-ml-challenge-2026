"""
src/predict.py
==============
Reusable inference wrapper for the Amazon ML Challenge 2026 baseline model.

Member 3 deliverable — baseline matching module.

Responsibility
--------------
  • Load the trained baseline model (models/matcher.pkl).
  • Load model configuration (models/model_config.json).
  • Load candidate_pairs.tsv  ← produced by Member 4 (M4).
  • Load source records from M2 cache (never re-normalize).
  • Extract the agreed 16 baseline features.
  • Score every candidate pair and apply the saved threshold.
  • Produce output/matching_results.tsv — one row per test S1 entity,
    empty matched_entity_ids for entities with no predicted match.

Memory-safety architecture (v2 — streaming)
--------------------------------------------
  Candidate TSV is consumed in bounded *chunk_size* rows at a time.
  The full TSV is NEVER loaded into RAM.  For each chunk:
    1. Parse candidate rows → expand comma-separated IDs into pair tuples.
    2. Look up S1 / candidate fields from the in-memory entity lookup dicts
       (entity lookup is bounded: 3 DataFrames of ~1-2 M rows each, read once).
    3. Compute 16 features for each pair in the chunk.
    4. Scale + predict with the already-fit scaler and model.
    5. Apply threshold, accumulate matches into a compact dict
       {s1_id → set(match_id)}.
  The accumulator dict holds AT MOST one entry per S1 entity —
  bounded memory even when the candidate file has billions of pairs.

Boundaries
----------
  • Candidate generation / blocking is NOT done here.  This wrapper
    expects candidate_pairs.tsv to be provided externally by M4.
  • Normalization is NOT reimplemented; M2 cache is the sole source.
  • The model is loaded from disk; training is NOT re-run.

Usage
-----
    python src/predict.py \\
        --data-dir  dataset/test \\
        --candidates output/candidate_pairs.tsv \\
        --model-dir  models \\
        --output-dir output \\
        --chunk-size 50000   # optional; default 50000

Output format (matching_results.tsv)
-------------------------------------
source1_entity_id   matched_entity_ids
S1-000001           S2-001,S3-042
S1-000002           (empty — no predicted match)
...
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import normalize_name, normalize_address, preprocess_dataframe  # noqa: E402  (M2)
from features import build_feature_matrix_from_dicts, FEATURE_NAMES             # noqa: E402  (M3)
from cache import load_all_cache, cache_exists                                   # noqa: E402  (M2 cache)

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK = 50_000


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path):
    """Load (scaler, model) tuple and config dict from *model_dir*."""
    model_path  = model_dir / "matcher.pkl"
    config_path = model_dir / "model_config.json"

    with open(model_path, "rb") as f:
        scaler, model = pickle.load(f)

    with open(config_path) as f:
        config = json.load(f)

    threshold = float(config["threshold"])
    saved_features = config["feature_names"]

    if saved_features != FEATURE_NAMES:
        raise ValueError(
            f"Feature mismatch between model config and current FEATURE_NAMES.\n"
            f"  Config : {saved_features}\n"
            f"  Current: {FEATURE_NAMES}"
        )

    return scaler, model, threshold, config


def load_sources(
    data_dir: Path,
    cache_dir: Path | None = None,
    split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load source TSVs from M2 Parquet cache (preferred) or raw TSVs.
    Returns three DataFrames: s1, s2, s3.
    """
    if cache_dir is not None and all(
        cache_exists(split, src, cache_dir)
        for src in ("source1", "source2", "source3")
    ):
        print(f"  Loading {split} sources from M2 Parquet cache...")
        s1, s2, s3 = load_all_cache(cache_dir, split)
    else:
        if cache_dir is not None:
            print(f"  Cache not found — falling back to raw TSV + M2 normalization")
        s1 = preprocess_dataframe(pd.read_csv(data_dir / f"{split}_source1.tsv", sep="\t", dtype=str))
        s2 = preprocess_dataframe(pd.read_csv(data_dir / f"{split}_source2.tsv", sep="\t", dtype=str))
        s3 = preprocess_dataframe(pd.read_csv(data_dir / f"{split}_source3.tsv", sep="\t", dtype=str))

    print(f"Data  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
    return s1, s2, s3


def _build_entity_lookups(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> dict[str, dict]:
    """
    Build compact per-entity lookup dicts (NOT DataFrames) for fast O(1) access.
    Returns {entity_id: {"name_norm": ..., "address_norm": ..., "country": ...}}
    for both S2 and S3 combined.

    Using plain dicts instead of DataFrame indexing avoids pandas overhead per lookup
    and keeps memory bounded to the actual entity count.
    """
    cand_lookup: dict[str, dict] = {}
    for df in (s2, s3):
        for row in df[["entity_id", "name_norm", "address_norm", "country"]].itertuples(index=False):
            cand_lookup[row.entity_id] = {
                "name_norm":    row.name_norm,
                "address_norm": row.address_norm,
                "country":      row.country,
            }
    return cand_lookup


def _build_s1_lookup(s1: pd.DataFrame) -> dict[str, dict]:
    """Build compact S1 lookup dict."""
    s1_lookup: dict[str, dict] = {}
    for row in s1[["entity_id", "name_norm", "address_norm", "country"]].itertuples(index=False):
        s1_lookup[row.entity_id] = {
            "name_norm":    row.name_norm,
            "address_norm": row.address_norm,
            "country":      row.country,
        }
    return s1_lookup


def _expand_chunk(
    chunk_df: pd.DataFrame,
    s1_lookup: dict[str, dict],
    cand_lookup: dict[str, dict],
) -> list[dict]:
    """
    Expand a chunk of candidate rows into a list of pair dicts ready for
    feature extraction.  Each dict contains s1 and candidate fields plus IDs.

    Pairs where S1 or candidate ID is missing from the lookups are silently
    skipped (they cannot be scored).
    """
    pairs: list[dict] = []
    for row in chunk_df.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        raw_ids = str(row.candidate_entity_ids) if row.candidate_entity_ids else ""
        if not raw_ids or raw_ids.lower() in ("nan", "none", ""):
            continue

        s1_rec = s1_lookup.get(s1_id)
        if s1_rec is None:
            continue

        cand_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
        for cid in cand_ids:
            cand_rec = cand_lookup.get(cid)
            if cand_rec is None:
                continue
            pairs.append({
                "source1_entity_id": s1_id,
                "cand_entity_id":    cid,
                "s1_name_norm":      s1_rec["name_norm"],
                "s1_address_norm":   s1_rec["address_norm"],
                "s1_country":        s1_rec["country"],
                "cand_name_norm":    cand_rec["name_norm"],
                "cand_address_norm": cand_rec["address_norm"],
                "cand_country":      cand_rec["country"],
            })
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main prediction entry-point
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    data_dir: Path,
    candidates_path: Path,
    model_dir: Path,
    output_dir: Path,
    cache_dir: Path | None = None,
    split: str = "test",
    chunk_size: int = _DEFAULT_CHUNK,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model (scaler + model already fit — no re-fitting here)
    scaler, model, threshold, config = load_model(model_dir)
    print(f"Model loaded  — threshold={threshold:.2f}  type={config['model_type']}")

    # 2. Load source data — reads ALL three sources once, holds them in RAM
    #    (bounded: ~300-500 MB for 1.7 M entities, acceptable)
    s1, s2, s3 = load_sources(data_dir, cache_dir=cache_dir, split=split)
    all_s1_ids = s1["entity_id"].tolist()

    # 3. Build compact lookup dicts — one-time O(N) build, then O(1) per pair
    print("Building entity lookup dicts...")
    s1_lookup   = _build_s1_lookup(s1)
    cand_lookup = _build_entity_lookups(s2, s3)
    del s2, s3  # free DataFrame RAM — lookups are now in plain dicts
    print(f"  S1 lookup : {len(s1_lookup):,} entities")
    print(f"  Cand lookup: {len(cand_lookup):,} entities (S2+S3)")

    # 4. Stream candidate TSV in chunks — NEVER load the full file
    pred_map: dict[str, set[str]] = {}   # {s1_id → set of matched IDs}
    total_cand_rows    = 0
    total_pair_rows    = 0
    total_scored_pairs = 0
    total_match_pairs  = 0
    chunk_num          = 0

    print(f"\nStreaming candidates in chunks of {chunk_size:,}...")
    for chunk_df in pd.read_csv(
        candidates_path,
        sep="\t",
        dtype=str,
        chunksize=chunk_size,
        keep_default_na=False,
    ):
        chunk_num += 1
        total_cand_rows += len(chunk_df)

        # 4a. Expand candidate rows → pair dicts
        pair_dicts = _expand_chunk(chunk_df, s1_lookup, cand_lookup)
        n_pairs = len(pair_dicts)
        total_pair_rows += n_pairs

        if n_pairs == 0:
            print(f"  chunk {chunk_num:4d}: {len(chunk_df):6,} cand rows → 0 pairs (skipped)")
            continue

        # 4b. Compute 16 features (vectorised over the chunk)
        X = build_feature_matrix_from_dicts(pair_dicts)   # shape (n_pairs, 16)

        # 4c. Scale + predict (already-fit scaler/model)
        X_sc = scaler.transform(X)
        proba = model.predict_proba(X_sc)[:, 1]
        total_scored_pairs += n_pairs

        # 4d. Apply threshold → accumulate into pred_map
        n_match = 0
        for i, d in enumerate(pair_dicts):
            if proba[i] >= threshold:
                sid = d["source1_entity_id"]
                cid = d["cand_entity_id"]
                if sid not in pred_map:
                    pred_map[sid] = set()
                pred_map[sid].add(cid)
                n_match += 1
        total_match_pairs += n_match

        print(
            f"  chunk {chunk_num:4d}: {len(chunk_df):6,} cand rows "
            f"→ {n_pairs:7,} pairs scored → {n_match:5,} matches"
        )

    print(
        f"\nStreaming complete:"
        f"\n  candidate rows processed : {total_cand_rows:,}"
        f"\n  pair rows scored          : {total_scored_pairs:,}"
        f"\n  matches emitted           : {total_match_pairs:,}"
        f"\n  S1 with ≥1 match          : {len(pred_map):,}"
    )

    # 5. Build output — one row per S1 entity, deterministic order
    #    Zero-candidate S1 entities still get a row (empty matched_entity_ids).
    n_zero_match = 0
    out_rows = []
    for sid in all_s1_ids:
        matched = pred_map.get(sid, set())
        if not matched:
            n_zero_match += 1
        out_rows.append({
            "source1_entity_id":  sid,
            "matched_entity_ids": ",".join(sorted(matched)),
        })

    result_df = pd.DataFrame(out_rows, columns=["source1_entity_id", "matched_entity_ids"])
    out_path  = output_dir / "matching_results.tsv"
    result_df.to_csv(out_path, sep="\t", index=False)

    print(f"\nOutput saved  → {out_path}")
    print(f"  Total S1 rows    : {len(result_df):,}")
    print(f"  Entities matched : {len(result_df) - n_zero_match:,}")
    print(f"  Entities empty   : {n_zero_match:,}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Baseline inference wrapper (streaming).")
    parser.add_argument("--data-dir",    default="dataset/test",           help="Directory with test TSVs")
    parser.add_argument("--candidates",  default="output/candidate_pairs.tsv")
    parser.add_argument("--model-dir",   default="models")
    parser.add_argument("--output-dir",  default="output")
    parser.add_argument("--cache-dir",   default=None,                      help="M2 Parquet cache dir (optional)")
    parser.add_argument("--split",       default="test", choices=["train", "test"])
    parser.add_argument("--chunk-size",  default=_DEFAULT_CHUNK, type=int,  help="Candidate rows per scoring chunk")
    args = parser.parse_args()

    predict(
        data_dir=Path(args.data_dir),
        candidates_path=Path(args.candidates),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        split=args.split,
        chunk_size=args.chunk_size,
    )
