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
  • Load source records and apply M2 normalization.
  • Extract the agreed 16 baseline features.
  • Score every candidate pair and apply the saved threshold.
  • Produce output/matching_results.tsv — one row per test S1 entity,
    empty matched_entity_ids for entities with no predicted match.

Boundaries
----------
  • Candidate generation / blocking is NOT done here.  This wrapper
    expects candidate_pairs.tsv to be provided externally by M4.
  • Normalization is NOT reimplemented; src/preprocess.py is used.
  • The model is loaded from disk; training is NOT re-run.

Usage
-----
    python src/predict.py \\
        --data-dir  dataset/test \\
        --candidates output/candidate_pairs.tsv \\
        --model-dir  models \\
        --output-dir output

Output format (matching_results.tsv)
-------------------------------------
source1_entity_id   matched_entity_ids
S1-000001           S2-001,S3-042
S1-000002           (empty — no predicted match)
...
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import normalize_name, normalize_address, preprocess_dataframe  # noqa: E402  (M2)
from features import build_feature_matrix, FEATURE_NAMES  # noqa: E402  (M3)
from cache import load_all_cache, cache_exists             # noqa: E402  (M2 cache)


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
    Load test source TSVs and apply M2 normalization.
    Uses Parquet cache when available; falls back to raw TSV + normalization.
    """
    if cache_dir is not None and all(
        cache_exists(split, src, cache_dir)
        for src in ("source1", "source2", "source3")
    ):  # type: ignore[arg-type]
        print(f"  Loading {split} sources from M2 Parquet cache...")
        s1, s2, s3 = load_all_cache(cache_dir, split)              # type: ignore[arg-type]
    else:
        if cache_dir is not None:
            print(f"  Cache not found — falling back to raw TSV + M2 normalization")
        s1 = preprocess_dataframe(pd.read_csv(data_dir / f"{split}_source1.tsv", sep="\t", dtype=str))
        s2 = preprocess_dataframe(pd.read_csv(data_dir / f"{split}_source2.tsv", sep="\t", dtype=str))
        s3 = preprocess_dataframe(pd.read_csv(data_dir / f"{split}_source3.tsv", sep="\t", dtype=str))

    print(f"Data  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")
    return s1, s2, s3


def build_pair_rows(
    candidates: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> pd.DataFrame:
    """Expand candidate pairs into flat (s1, candidate) rows for scoring."""
    s2_lkp = s2.set_index("entity_id")[
        ["name_norm", "address_norm", "country"]
    ].rename(columns={
        "name_norm": "cand_name_norm",
        "address_norm": "cand_address_norm",
        "country": "cand_country",
    })
    s3_lkp = s3.set_index("entity_id")[
        ["name_norm", "address_norm", "country"]
    ].rename(columns={
        "name_norm": "cand_name_norm",
        "address_norm": "cand_address_norm",
        "country": "cand_country",
    })
    cand_lkp = pd.concat([s2_lkp, s3_lkp])

    s1_lkp = s1.set_index("entity_id")[
        ["name_norm", "address_norm", "country"]
    ].rename(columns={
        "name_norm": "s1_name_norm",
        "address_norm": "s1_address_norm",
        "country": "s1_country",
    })

    rows = []
    for _, cand_row in candidates.iterrows():
        s1_id = cand_row["source1_entity_id"]
        raw_ids = str(cand_row.get("candidate_entity_ids", "") or "")
        cand_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]

        if s1_id not in s1_lkp.index:
            continue
        s1_rec = s1_lkp.loc[s1_id]

        for cid in cand_ids:
            if cid not in cand_lkp.index:
                continue
            cand_rec = cand_lkp.loc[cid]
            rows.append({
                "source1_entity_id": s1_id,
                "cand_entity_id":    cid,
                "s1_name_norm":      s1_rec["s1_name_norm"],
                "s1_address_norm":   s1_rec["s1_address_norm"],
                "s1_country":        s1_rec["s1_country"],
                "cand_name_norm":    cand_rec["cand_name_norm"],
                "cand_address_norm": cand_rec["cand_address_norm"],
                "cand_country":      cand_rec["cand_country"],
            })

    return pd.DataFrame(rows)


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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    scaler, model, threshold, config = load_model(model_dir)
    print(f"Model loaded  — threshold={threshold:.2f}  "
          f"type={config['model_type']}")

    # 2. Load source data (cache-aware)
    s1, s2, s3 = load_sources(data_dir, cache_dir=cache_dir, split=split)
    all_s1_ids = s1["entity_id"].tolist()

    # 3. Load candidates (M4 artifact)
    candidates = pd.read_csv(candidates_path, sep="\t")
    print(f"Candidates    — {len(candidates):,} rows")

    # 4. Build pair rows
    pair_df = build_pair_rows(candidates, s1, s2, s3)
    print(f"Pair rows     — {len(pair_df):,}")

    # 5. Extract features and score
    if len(pair_df) == 0:
        print("WARNING: No scorable pairs found — all matches will be empty.")
        pred_map: dict[str, set[str]] = {}
    else:
        X = build_feature_matrix(pair_df).values
        X_sc = scaler.transform(X)
        proba = model.predict_proba(X_sc)[:, 1]

        pred_map = {}
        pair_df = pair_df.copy()
        pair_df["score"] = proba
        pair_df["match"] = proba >= threshold

        for sid, grp in pair_df.groupby("source1_entity_id"):
            matched = set(grp.loc[grp["match"], "cand_entity_id"])
            pred_map[str(sid)] = matched

    # 6. Build output — one row per S1 entity (empty string if no matches)
    rows = []
    for sid in all_s1_ids:
        matched = pred_map.get(sid, set())
        rows.append({
            "source1_entity_id": sid,
            "matched_entity_ids": ",".join(sorted(matched)),
        })

    result_df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])

    out_path = output_dir / "matching_results.tsv"
    result_df.to_csv(out_path, sep="\t", index=False)

    n_matched = (result_df["matched_entity_ids"] != "").sum()
    print(f"\nOutput saved  → {out_path}")
    print(f"  Total S1 rows   : {len(result_df):,}")
    print(f"  Entities matched: {n_matched:,}")
    print(f"  Entities empty  : {len(result_df) - n_matched:,}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline inference wrapper.")
    parser.add_argument("--data-dir",   default="dataset/test",   help="Directory with test TSVs")
    parser.add_argument("--candidates", default="output/candidate_pairs.tsv")
    parser.add_argument("--model-dir",  default="models")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--cache-dir",  default=None, help="M2 Parquet cache directory (optional)")
    parser.add_argument("--split",      default="test", choices=["train", "test"])
    args = parser.parse_args()

    predict(
        data_dir=Path(args.data_dir),
        candidates_path=Path(args.candidates),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        split=args.split,
    )
