"""
predict.py
----------
Member 3 - Amazon ML Challenge 2026 - Business Entity Resolution

INFERENCE PIPELINE
==================

Loads the trained model (models/matcher.pkl) and its configuration
(models/model_config.json), computes pair-level features for every
candidate pair supplied by Member 4, and applies the saved probability
threshold to produce final match predictions.

This script does NOT perform blocking or candidate generation.
It receives a pre-built candidate_pairs.tsv from Member 4 and produces
submission-ready output.

Inputs (all paths overridable via CLI)
---------------------------------------
  models/matcher.pkl
      Trained sklearn Pipeline (StandardScaler + LogisticRegression).

  models/model_config.json
      Saved threshold and feature names written by train.py.

  output/candidate_pairs.tsv          (--candidates)
      Member 4's candidate pairs.
      Required columns: source1_entity_id, candidate_entity_id, candidate_source
      Optional column : label  (used only when computing optional stats)

  dataset/test/test_source1.tsv       (--s1)
  dataset/test/test_source2.tsv       (--s2)
  dataset/test/test_source3.tsv       (--s3)
      Raw test entity records.

Outputs
-------
  output/matching_results.tsv
      Submission-ready file.  One row per S1 entity in test_source1.tsv.
      Columns: source1_entity_id, matched_entity_ids
      matched_entity_ids is comma-separated; empty string when no match.

  output/prediction_scores.tsv  (optional, written unless --no-scores)
      Columns: source1_entity_id, candidate_entity_id, candidate_source,
               match_probability, predicted_label

Submission format rules enforced
---------------------------------
  - One row per S1 entity (even if no match -> empty string).
  - Comma-separated matched IDs, no duplicates within a row.
  - Only S2-/S3- IDs returned (never S1-).
  - Only IDs from the supplied candidate set returned.
  - Output is UTF-8 tab-separated.

Usage
-----
  python src/predict.py                               # all defaults
  python src/predict.py --candidates output/candidate_pairs.tsv
  python src/predict.py --s1 dataset/test/test_source1.tsv \\
                         --s2 dataset/test/test_source2.tsv \\
                         --s3 dataset/test/test_source3.tsv
  python src/predict.py --threshold 0.80              # override saved threshold
  python src/predict.py --no-scores                   # skip prediction_scores.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# -- make repo root importable when called as  python src/predict.py ----------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.features import FEATURE_COLS, extract_features_batch  # noqa: E402

# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PKL   = "models/matcher.pkl"
DEFAULT_MODEL_CFG   = "models/model_config.json"
DEFAULT_CANDIDATES  = "output/candidate_pairs.tsv"
DEFAULT_S1          = "dataset/test/test_source1.tsv"
DEFAULT_S2          = "dataset/test/test_source2.tsv"
DEFAULT_S3          = "dataset/test/test_source3.tsv"
DEFAULT_RESULTS     = "output/matching_results.tsv"
DEFAULT_SCORES      = "output/prediction_scores.tsv"

SOURCE_RECORD_COLS  = ["entity_id", "business_name", "business_address", "country"]


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    """Timestamped console logger."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M3 Inference Pipeline - Amazon ML Challenge 2026"
    )
    p.add_argument("--model",      default=DEFAULT_MODEL_PKL,
                   help=f"Path to matcher.pkl. Default: {DEFAULT_MODEL_PKL}")
    p.add_argument("--config",     default=DEFAULT_MODEL_CFG,
                   help=f"Path to model_config.json. Default: {DEFAULT_MODEL_CFG}")
    p.add_argument("--candidates", default=DEFAULT_CANDIDATES,
                   help=f"Candidate pairs TSV from Member 4. Default: {DEFAULT_CANDIDATES}")
    p.add_argument("--s1",         default=DEFAULT_S1,
                   help=f"test_source1.tsv path. Default: {DEFAULT_S1}")
    p.add_argument("--s2",         default=DEFAULT_S2,
                   help=f"test_source2.tsv path. Default: {DEFAULT_S2}")
    p.add_argument("--s3",         default=DEFAULT_S3,
                   help=f"test_source3.tsv path. Default: {DEFAULT_S3}")
    p.add_argument("--output",     default=DEFAULT_RESULTS,
                   help=f"Output matching_results.tsv path. Default: {DEFAULT_RESULTS}")
    p.add_argument("--scores",     default=DEFAULT_SCORES,
                   help=f"Output prediction_scores.tsv path. Default: {DEFAULT_SCORES}")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Override the threshold from model_config.json.")
    p.add_argument("--no-scores",  action="store_true",
                   help="Skip writing prediction_scores.tsv.")
    p.add_argument("--chunk-size", type=int, default=100_000,
                   help="Pairs to process per batch (memory control). Default: 100000")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_model_and_config(
    model_path: str,
    config_path: str,
) -> tuple[Any, dict]:
    """Load the trained sklearn Pipeline and its saved configuration.

    Parameters
    ----------
    model_path  : path to matcher.pkl
    config_path : path to model_config.json

    Returns
    -------
    (pipeline, config_dict)

    Raises
    ------
    FileNotFoundError  if either artifact is missing.
    ValueError         if the config feature_names differ from FEATURE_COLS.
    """
    for path in (model_path, config_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required model artifact not found: {path}\n"
                f"Run src/train.py first to produce the model."
            )

    log(f"Loading model from: {model_path}")
    pipeline = joblib.load(model_path)
    log(f"  -> {type(pipeline).__name__} loaded.")

    log(f"Loading config from: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Validate that the saved feature set matches this codebase's FEATURE_COLS
    saved_feats = config.get("feature_names", [])
    if saved_feats != FEATURE_COLS:
        raise ValueError(
            f"model_config.json feature_names does not match FEATURE_COLS in "
            f"src/features.py.\n"
            f"  Config: {saved_feats}\n"
            f"  Code:   {FEATURE_COLS}\n"
            f"Re-train the model to synchronise features."
        )

    log(f"  -> model_type     : {config.get('model_type')}")
    log(f"  -> threshold      : {config.get('threshold')}")
    log(f"  -> baseline_version: {config.get('baseline_version')}")
    log(f"  -> train P/R/F0.5 : "
        f"{config.get('precision'):.6f} / "
        f"{config.get('recall'):.6f} / "
        f"{config.get('f0_5'):.6f}")

    return pipeline, config


def load_source_records(*source_paths: str) -> dict[str, dict[str, str]]:
    """Load entity records from one or more source TSV files.

    Parameters
    ----------
    *source_paths : paths to test_source{1,2,3}.tsv

    Returns
    -------
    dict  entity_id -> {"business_name", "business_address", "country"}
    """
    records: dict[str, dict[str, str]] = {}
    for path in source_paths:
        if not os.path.exists(path):
            log(f"  WARNING: source file not found, skipping: {path}")
            continue
        log(f"  Loading {path} ...")
        df = pd.read_csv(
            path, sep="\t", dtype=str, na_filter=False, usecols=SOURCE_RECORD_COLS
        )
        for row in df.itertuples(index=False):
            records[row.entity_id] = {
                "business_name":    row.business_name,
                "business_address": row.business_address,
                "country":          row.country,
            }
        log(f"    -> {len(df):,} records. Cumulative lookup: {len(records):,}")
    return records


def load_candidates(candidates_path: str) -> pd.DataFrame:
    """Load and validate Member 4's candidate pairs.

    Required columns: source1_entity_id, candidate_entity_id, candidate_source
    Optional column : label

    Parameters
    ----------
    candidates_path : path to candidate_pairs.tsv

    Returns
    -------
    DataFrame with at least the three required columns.
    """
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(
            f"Candidate pairs file not found: {candidates_path}\n"
            f"Member 4's blocking output is required before running predict.py."
        )

    log(f"Loading candidates from: {candidates_path}")
    df = pd.read_csv(
        candidates_path, sep="\t",
        dtype={"source1_entity_id": str, "candidate_entity_id": str,
               "candidate_source": str},
    )

    required = {"source1_entity_id", "candidate_entity_id", "candidate_source"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"candidate_pairs.tsv is missing required columns: {sorted(missing_cols)}\n"
            f"Found columns: {df.columns.tolist()}"
        )

    # Deduplicate silently (safety against upstream duplicates)
    n_before = len(df)
    df = df.drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"]
    ).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        log(f"  WARNING: dropped {n_dropped:,} duplicate candidate pairs.")

    log(f"  -> {len(df):,} unique candidate pairs for "
        f"{df['source1_entity_id'].nunique():,} S1 entities.")

    return df


def load_s1_entity_ids(s1_path: str) -> list[str]:
    """Return the ordered list of all S1 entity IDs in test_source1.tsv.

    Every S1 entity must appear in the output, even with no match.
    """
    if not os.path.exists(s1_path):
        raise FileNotFoundError(
            f"test_source1.tsv not found: {s1_path}\n"
            f"Required to determine the full set of S1 entities in the output."
        )
    df = pd.read_csv(s1_path, sep="\t", dtype=str, na_filter=False,
                     usecols=["entity_id"])
    ids = df["entity_id"].tolist()
    log(f"  -> {len(ids):,} S1 entities in test_source1.tsv.")
    return ids


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------
def predict_candidates(
    candidates: pd.DataFrame,
    records: dict[str, dict[str, str]],
    pipeline: Any,
    feature_names: list[str],
    threshold: float,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Score all candidate pairs and return predictions.

    Processes pairs in chunks to bound memory usage.  Feature extraction
    reuses src/features.extract_features_batch exactly as in training.

    Parameters
    ----------
    candidates   : DataFrame with source1_entity_id, candidate_entity_id,
                   candidate_source (and optionally label).
    records      : entity lookup dict from load_source_records().
    pipeline     : loaded sklearn Pipeline.
    feature_names: list of feature column names from model_config.json.
    threshold    : probability threshold for a positive prediction.
    chunk_size   : number of pairs per processing batch.

    Returns
    -------
    DataFrame with columns:
      source1_entity_id, candidate_entity_id, candidate_source,
      match_probability, predicted_label
      (plus label if it was present in the input)
    """
    n_total = len(candidates)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    log(f"  Scoring {n_total:,} pairs in {n_chunks} chunk(s) of up to "
        f"{chunk_size:,} ...")

    result_parts: list[pd.DataFrame] = []

    for chunk_idx in range(n_chunks):
        chunk = candidates.iloc[
            chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size
        ].copy().reset_index(drop=True)

        # extract_features_batch expects "s1_entity_id"; the candidate file
        # from Member 4 (and submission format) uses "source1_entity_id".
        # Rename for the feature call, then restore the original name in output.
        has_s1_col = "source1_entity_id" in chunk.columns
        feat_input = chunk.rename(
            columns={"source1_entity_id": "s1_entity_id"}
        ) if has_s1_col else chunk

        # Feature extraction via src/features.py (identical to training path)
        feat_df = extract_features_batch(feat_input, records)

        # Build feature matrix in the exact order from model_config.json
        X = feat_df[feature_names].values.astype(float)

        # Predict probability of label=1
        proba = pipeline.predict_proba(X)[:, 1]

        # Build output DataFrame using the original column name
        s1_col = "source1_entity_id" if has_s1_col else "s1_entity_id"
        out = chunk[[s1_col, "candidate_entity_id",
                      "candidate_source"]].copy().reset_index(drop=True)
        if has_s1_col:
            out = out.rename(columns={s1_col: "source1_entity_id"})
        if "label" in chunk.columns:
            out["label"] = chunk["label"].values
        out["match_probability"] = proba
        out["predicted_label"]   = (proba >= threshold).astype(int)

        result_parts.append(out)

        if (chunk_idx + 1) % max(1, n_chunks // 5) == 0 or chunk_idx == n_chunks - 1:
            processed = min((chunk_idx + 1) * chunk_size, n_total)
            log(f"    {processed:,}/{n_total:,} pairs scored ...")

    return pd.concat(result_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# OUTPUT ASSEMBLY
# ---------------------------------------------------------------------------
def build_matching_results(
    scores_df: pd.DataFrame,
    all_s1_ids: list[str],
) -> pd.DataFrame:
    """Build the submission-ready matching_results.tsv.

    Rules enforced here:
      - One row per S1 entity (from test_source1.tsv), even if no match.
      - matched_entity_ids = comma-separated unique S2/S3 IDs.
      - Empty string when predicted_label == 0 for all candidates.
      - Only IDs with predicted_label == 1 are included.
      - No S1- IDs (already guaranteed by candidate source, but double-checked).
      - No duplicate IDs within a row.

    Parameters
    ----------
    scores_df  : output of predict_candidates()
    all_s1_ids : ordered list of S1 IDs from test_source1.tsv

    Returns
    -------
    DataFrame with columns: source1_entity_id, matched_entity_ids
    """
    # Keep only predicted matches
    matches = scores_df[scores_df["predicted_label"] == 1].copy()

    # Safety: remove any accidental S1- IDs (should never happen from M4 candidates)
    bad_prefix_mask = matches["candidate_entity_id"].str.startswith("S1-")
    if bad_prefix_mask.any():
        n_bad = bad_prefix_mask.sum()
        log(f"  WARNING: dropping {n_bad} candidate(s) with S1- prefix (not allowed).")
        matches = matches[~bad_prefix_mask]

    # Group matches per S1 entity -> set of matched IDs (deduplication guaranteed)
    if len(matches) > 0:
        grouped = (
            matches.groupby("source1_entity_id")["candidate_entity_id"]
            .agg(lambda ids: ",".join(sorted(set(ids))))
            .reset_index()
            .rename(columns={"candidate_entity_id": "matched_entity_ids"})
        )
        match_map: dict[str, str] = dict(
            zip(grouped["source1_entity_id"], grouped["matched_entity_ids"])
        )
    else:
        match_map = {}

    # Construct one row per S1 entity in the test set (ordered)
    rows = [
        {"source1_entity_id": s1_id,
         "matched_entity_ids": match_map.get(s1_id, "")}
        for s1_id in all_s1_ids
    ]
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])


# ---------------------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------------------
def print_summary(
    results_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    threshold: float,
    config: dict,
    elapsed: float,
) -> None:
    """Print prediction summary to stdout."""
    total_s1   = len(results_df)
    matched_s1 = (results_df["matched_entity_ids"] != "").sum()
    empty_s1   = total_s1 - matched_s1

    total_pairs   = len(scores_df)
    pos_pairs     = (scores_df["predicted_label"] == 1).sum()
    neg_pairs     = total_pairs - pos_pairs

    # Unique matched IDs across all S1 entities
    all_matched_ids = set()
    for s in results_df["matched_entity_ids"]:
        if s:
            all_matched_ids.update(s.split(","))

    # Match count distribution
    match_counts = results_df["matched_entity_ids"].apply(
        lambda s: len(s.split(",")) if s else 0
    )

    print()
    print("=" * 60)
    print("  PREDICTION SUMMARY")
    print("=" * 60)
    print(f"  Model type         : {config.get('model_type')}")
    print(f"  Baseline version   : {config.get('baseline_version')}")
    print(f"  Features used      : {len(config.get('feature_names', []))}")
    print(f"  Threshold applied  : {threshold}")
    print()
    print(f"  Candidate pairs    : {total_pairs:,}")
    print(f"  Predicted matches  : {pos_pairs:,}  ({100*pos_pairs/max(total_pairs,1):.2f}%)")
    print(f"  Predicted non-match: {neg_pairs:,}  ({100*neg_pairs/max(total_pairs,1):.2f}%)")
    print()
    print(f"  S1 entities total  : {total_s1:,}")
    print(f"  S1 with matches    : {matched_s1:,}  ({100*matched_s1/max(total_s1,1):.2f}%)")
    print(f"  S1 with no match   : {empty_s1:,}")
    print(f"  Unique matched IDs : {len(all_matched_ids):,}")
    print()
    print(f"  Match count per S1:")
    print(f"    min  : {match_counts.min()}")
    print(f"    max  : {match_counts.max()}")
    print(f"    mean : {match_counts.mean():.2f}")
    print(f"    median: {match_counts.median():.1f}")
    print()
    print(f"  Elapsed            : {elapsed:.1f}s")
    print("=" * 60)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    t_start = time.time()

    # Create output directories
    for path in (args.output, args.scores):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    print()
    print("=" * 60)
    print("  M3 INFERENCE PIPELINE - Amazon ML Challenge 2026")
    print("=" * 60)

    # -- Step 1: Load model and config ----------------------------------------
    log("Step 1/6  Load model + config")
    pipeline, config = load_model_and_config(args.model, args.config)

    # Determine threshold: CLI override > saved config
    threshold = args.threshold if args.threshold is not None else config["threshold"]
    if args.threshold is not None:
        log(f"  NOTE: threshold overridden via CLI: {threshold} "
            f"(config has {config['threshold']})")
    else:
        log(f"  Using saved threshold: {threshold}")

    feature_names: list[str] = config["feature_names"]

    # -- Step 2: Load source records ------------------------------------------
    log("Step 2/6  Load source entity records")
    records = load_source_records(args.s1, args.s2, args.s3)

    # -- Step 3: Load S1 entity list (for complete output) --------------------
    log("Step 3/6  Load test S1 entity list")
    all_s1_ids = load_s1_entity_ids(args.s1)

    # -- Step 4: Load candidate pairs -----------------------------------------
    log("Step 4/6  Load candidate pairs (Member 4 output)")
    candidates = load_candidates(args.candidates)

    # Safety check: all candidate S1 IDs must be in test_source1
    test_s1_set = set(all_s1_ids)
    cand_s1_set = set(candidates["source1_entity_id"].unique())
    extra_s1 = cand_s1_set - test_s1_set
    if extra_s1:
        n_extra = len(extra_s1)
        log(f"  WARNING: {n_extra:,} S1 IDs in candidates are NOT in "
            f"test_source1.tsv -- these pairs will be scored but the S1 "
            f"entities will NOT appear in matching_results.tsv.")

    # -- Step 5: Score all candidate pairs ------------------------------------
    log("Step 5/6  Feature extraction + scoring")
    scores_df = predict_candidates(
        candidates=candidates,
        records=records,
        pipeline=pipeline,
        feature_names=feature_names,
        threshold=threshold,
        chunk_size=args.chunk_size,
    )

    pos_count = (scores_df["predicted_label"] == 1).sum()
    neg_count = len(scores_df) - pos_count
    log(f"  -> {pos_count:,} predicted matches, {neg_count:,} non-matches "
        f"(threshold={threshold}).")

    # -- Step 6: Write outputs ------------------------------------------------
    log("Step 6/6  Writing output files")

    # 6a. matching_results.tsv (submission format)
    results_df = build_matching_results(scores_df, all_s1_ids)
    results_df.to_csv(args.output, sep="\t", index=False, encoding="utf-8")
    log(f"  -> matching_results.tsv written: {args.output}")
    log(f"     {len(results_df):,} rows  "
        f"({(results_df['matched_entity_ids'] != '').sum():,} with matches, "
        f"{(results_df['matched_entity_ids'] == '').sum():,} empty)")

    # 6b. prediction_scores.tsv (optional diagnostic)
    if not args.no_scores:
        score_cols = ["source1_entity_id", "candidate_entity_id",
                      "candidate_source", "match_probability", "predicted_label"]
        if "label" in scores_df.columns:
            score_cols = ["source1_entity_id", "candidate_entity_id",
                          "candidate_source", "label",
                          "match_probability", "predicted_label"]
        scores_out = scores_df[
            [c for c in score_cols if c in scores_df.columns]
        ].copy()
        scores_out["match_probability"] = scores_out["match_probability"].round(6)
        scores_out.to_csv(args.scores, sep="\t", index=False, encoding="utf-8")
        log(f"  -> prediction_scores.tsv written: {args.scores}")

    # -- Summary --------------------------------------------------------------
    print_summary(
        results_df=results_df,
        scores_df=scores_df,
        threshold=threshold,
        config=config,
        elapsed=time.time() - t_start,
    )


if __name__ == "__main__":
    main()
