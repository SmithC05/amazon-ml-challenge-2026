"""
src/predict.py
==============
Reusable inference wrapper for the Amazon ML Challenge 2026 baseline model.

Member 3 deliverable — baseline matching module.

Memory-safety architecture (v3 — fully disk-backed)
-----------------------------------------------------
  Problem 4 (FIXED): The global pred_map dict that held all 12M+ matches in RAM
  is replaced by a DuckDB in-process database backed by a temporary file.
  For each scored batch only positive matches are INSERT-ed into DuckDB.
  After all chunks are processed, a single grouped SQL query generates the final
  TSV — no large Python dict/set ever holds the full result set.

Streaming flow
--------------
  1. Load entity data once (3 DataFrames → flat Python dicts, bounded).
  2. Open a temporary DuckDB database for match accumulation.
  3. Stream candidate TSV with pd.read_csv(..., chunksize=N):
       a. expand chunk → pair dicts (bounded)
       b. compute 16 features (bounded array)
       c. scaler.transform → model.predict_proba (bounded)
       d. INSERT matches (prob ≥ threshold) into DuckDB — release batch
  4. After streaming: query DuckDB grouped by source1_entity_id to produce
     comma-separated matched_entity_ids; outer-join with all S1 IDs so
     zero-match entities get an empty field.
  5. Write matching_results.tsv directly from DuckDB COPY.

Boundaries
----------
  • Candidate generation / blocking: NOT done here (M4 contract).
  • Normalization: NOT reimplemented (M2 cache is the source).
  • Model: loaded from disk; never re-fit.

Usage
-----
    python src/predict.py \\
        --data-dir  dataset/test \\
        --candidates output/candidate_pairs.tsv \\
        --model-dir  models \\
        --output-dir output \\
        --chunk-size 50000

Output format (matching_results.tsv)
-------------------------------------
source1_entity_id<TAB>matched_entity_ids
S1-000001           S2-001,S3-042
S1-000002           (empty — no predicted match)
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import normalize_name, normalize_address, preprocess_dataframe  # noqa: E402
from features import build_feature_matrix_from_dicts, FEATURE_NAMES             # noqa: E402
from cache import load_all_cache, cache_exists                                   # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_CHUNK = 50_000


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: Path):
    """Load (scaler, model) tuple and config dict from *model_dir*."""
    with open(model_dir / "matcher.pkl", "rb") as f:
        scaler, model = pickle.load(f)
    with open(model_dir / "model_config.json") as f:
        config = json.load(f)

    threshold      = float(config["threshold"])
    saved_features = config["feature_names"]

    if saved_features != FEATURE_NAMES:
        raise ValueError(
            f"Feature mismatch.\n  Config : {saved_features}\n  Current: {FEATURE_NAMES}"
        )
    return scaler, model, threshold, config


# ─────────────────────────────────────────────────────────────────────────────
# Source loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sources(
    data_dir: Path,
    cache_dir: Path | None = None,
    split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cache_dir is not None and all(
        cache_exists(split, src, cache_dir)
        for src in ("source1", "source2", "source3")
    ):
        print(f"  Loading {split} sources from M2 Parquet cache...")
        s1, s2, s3 = load_all_cache(cache_dir, split)
    else:
        if cache_dir is not None:
            print("  Cache not found — falling back to raw TSV + M2 normalization")
        s1 = preprocess_dataframe(pd.read_csv(data_dir / f"{split}_source1.tsv", sep="\t", dtype=str))
        s2 = preprocess_dataframe(pd.read_csv(data_dir / f"{split}_source2.tsv", sep="\t", dtype=str))
        s3 = preprocess_dataframe(pd.read_csv(data_dir / f"{split}_source3.tsv", sep="\t", dtype=str))
    print(f"Data  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
    return s1, s2, s3


# ─────────────────────────────────────────────────────────────────────────────
# Entity lookup dict builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_entity_lookups(s2: pd.DataFrame, s3: pd.DataFrame) -> dict[str, dict]:
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
    s1_lookup: dict[str, dict] = {}
    for row in s1[["entity_id", "name_norm", "address_norm", "country"]].itertuples(index=False):
        s1_lookup[row.entity_id] = {
            "name_norm":    row.name_norm,
            "address_norm": row.address_norm,
            "country":      row.country,
        }
    return s1_lookup


# ─────────────────────────────────────────────────────────────────────────────
# Chunk expander
# ─────────────────────────────────────────────────────────────────────────────

def _expand_chunk(
    chunk_df: pd.DataFrame,
    s1_lookup: dict[str, dict],
    cand_lookup: dict[str, dict],
) -> list[dict]:
    """
    Expand one candidate TSV chunk into a list of pair dicts.
    Bounded: at most chunk_size × avg_candidates_per_row dicts.
    """
    pairs: list[dict] = []
    for row in chunk_df.itertuples(index=False):
        s1_id   = str(row.source1_entity_id)
        raw_ids = str(row.candidate_entity_ids) if row.candidate_entity_ids else ""
        if not raw_ids or raw_ids.lower() in ("nan", "none", ""):
            continue
        s1_rec = s1_lookup.get(s1_id)
        if s1_rec is None:
            continue
        for cid in (x.strip() for x in raw_ids.split(",") if x.strip()):
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
# DuckDB match accumulator
# ─────────────────────────────────────────────────────────────────────────────

def _open_match_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Open a DuckDB database at *db_path* and create the matches table.
    Using UNIQUE (source1_entity_id, matched_entity_id) to prevent duplicates
    at the storage level without requiring deduplication in Python.
    """
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            source1_entity_id VARCHAR NOT NULL,
            matched_entity_id VARCHAR NOT NULL,
            PRIMARY KEY (source1_entity_id, matched_entity_id)
        )
    """)
    return con


def _insert_matches(
    con: duckdb.DuckDBPyConnection,
    pair_dicts: list[dict],
    probas: np.ndarray,
    threshold: float,
) -> int:
    """
    Insert positive matches for one batch into the DuckDB accumulator.
    Only inserts pairs where proba >= threshold.
    Returns the count of new matches inserted in this batch.
    """
    s1_ids  = []
    cand_ids = []
    for i, d in enumerate(pair_dicts):
        if probas[i] >= threshold:
            s1_ids.append(d["source1_entity_id"])
            cand_ids.append(d["cand_entity_id"])

    if not s1_ids:
        return 0

    # Use DuckDB's parameterised VALUES approach for safety
    # INSERT OR IGNORE skips duplicate (s1, cand) pairs
    rows_df = pd.DataFrame({"s1": s1_ids, "cid": cand_ids})
    con.execute("""
        INSERT OR IGNORE INTO matches
        SELECT s1 AS source1_entity_id, cid AS matched_entity_id
        FROM rows_df
    """)
    return len(s1_ids)


def _write_output_from_db(
    con: duckdb.DuckDBPyConnection,
    all_s1_ids: list[str],
    out_path: Path,
) -> tuple[int, int]:
    """
    Generate the final matching_results.tsv from the DuckDB accumulator.

    Loads all S1 IDs into a DuckDB temp table, LEFT JOINs with matches,
    aggregates comma-separated matched IDs, and writes the TSV.
    Never materialises a large Python DataFrame.

    Returns (n_with_matches, n_zero_matches).
    """
    # Register S1 ordering table
    s1_df = pd.DataFrame({"source1_entity_id": all_s1_ids})
    con.register("s1_all", s1_df)

    # Write TSV directly from DuckDB
    # Deterministic: ORDER BY source1_entity_id, matches joined as comma list sorted
    sql = f"""
        COPY (
            SELECT
                s.source1_entity_id,
                COALESCE(
                    array_to_string(
                        array_agg(m.matched_entity_id ORDER BY m.matched_entity_id),
                        ','
                    ),
                    ''
                ) AS matched_entity_ids
            FROM s1_all s
            LEFT JOIN matches m ON s.source1_entity_id = m.source1_entity_id
            GROUP BY s.source1_entity_id
            ORDER BY s.source1_entity_id
        ) TO '{str(out_path).replace(chr(92), '/')}' (HEADER TRUE, DELIMITER '\\t', QUOTE '')
    """
    con.execute(sql)

    # Count stats
    n_with = con.execute(
        "SELECT COUNT(DISTINCT source1_entity_id) FROM matches"
    ).fetchone()[0]
    n_zero = len(all_s1_ids) - n_with
    return n_with, n_zero


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

    # 1. Load model (already fit — never re-fit)
    scaler, model, threshold, config = load_model(model_dir)
    print(f"Model loaded  — threshold={threshold:.2f}  type={config['model_type']}")

    # 2. Load source data — three DataFrames read once (~300-500 MB, bounded)
    s1, s2, s3 = load_sources(data_dir, cache_dir=cache_dir, split=split)
    all_s1_ids = s1["entity_id"].tolist()

    # 3. Build compact lookup dicts
    print("Building entity lookup dicts...")
    s1_lookup   = _build_s1_lookup(s1)
    cand_lookup = _build_entity_lookups(s2, s3)
    del s2, s3
    print(f"  S1 lookup   : {len(s1_lookup):,} entities")
    print(f"  Cand lookup : {len(cand_lookup):,} entities (S2+S3)")

    # 4. Open DuckDB match accumulator (disk-backed temp file)
    db_path = str(output_dir / ".match_acc.duckdb")
    con = _open_match_db(db_path)

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
        chunk_num       += 1
        total_cand_rows += len(chunk_df)

        # 4a. Expand
        pair_dicts = _expand_chunk(chunk_df, s1_lookup, cand_lookup)
        n_pairs = len(pair_dicts)
        total_pair_rows += n_pairs

        if n_pairs == 0:
            print(f"  chunk {chunk_num:4d}: {len(chunk_df):6,} cand rows → 0 pairs (skipped)")
            continue

        # 4b. Feature extraction (bounded: n_pairs × 16 floats)
        X    = build_feature_matrix_from_dicts(pair_dicts)
        X_sc = scaler.transform(X)
        probas = model.predict_proba(X_sc)[:, 1]
        total_scored_pairs += n_pairs

        # 4c. Insert matches into DuckDB accumulator — release batch immediately
        n_inserted = _insert_matches(con, pair_dicts, probas, threshold)
        total_match_pairs += n_inserted
        del pair_dicts, X, X_sc, probas

        print(
            f"  chunk {chunk_num:4d}: {len(chunk_df):6,} cand rows "
            f"→ {n_pairs:7,} pairs scored → {n_inserted:5,} matches inserted"
        )

    print(
        f"\nStreaming complete:"
        f"\n  candidate rows processed : {total_cand_rows:,}"
        f"\n  pair rows scored          : {total_scored_pairs:,}"
        f"\n  match rows in accumulator : {total_match_pairs:,}"
    )

    # 5. Write final TSV from DuckDB — never materialises a giant Python dict
    out_path = output_dir / "matching_results.tsv"
    print(f"\nWriting output via DuckDB → {out_path} ...")
    n_with, n_zero = _write_output_from_db(con, all_s1_ids, out_path)

    con.close()
    # Clean up temp DB file
    try:
        Path(db_path).unlink(missing_ok=True)
        Path(db_path + ".wal").unlink(missing_ok=True)
    except Exception:
        pass

    print(f"\nOutput saved  → {out_path}")
    print(f"  Total S1 rows    : {len(all_s1_ids):,}")
    print(f"  Entities matched : {n_with:,}")
    print(f"  Entities empty   : {n_zero:,}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Baseline inference wrapper (streaming v3).")
    parser.add_argument("--data-dir",    default="dataset/test")
    parser.add_argument("--candidates",  default="output/candidate_pairs.tsv")
    parser.add_argument("--model-dir",   default="models")
    parser.add_argument("--output-dir",  default="output")
    parser.add_argument("--cache-dir",   default=None)
    parser.add_argument("--split",       default="test", choices=["train", "test"])
    parser.add_argument("--chunk-size",  default=_DEFAULT_CHUNK, type=int)
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
