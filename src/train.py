"""
train.py
--------
Member 3 - Amazon ML Challenge 2026 - Business Entity Resolution

BASELINE TRAINING PIPELINE
===========================

Trains a Logistic Regression classifier on the 16 baseline features to
predict whether a (S1, candidate) entity pair is a true match (label=1).

Inputs
------
  output/training_pairs_baseline.tsv
      Pre-generated labelled pair dataset (s1_entity_id, candidate_entity_id,
      candidate_source, label).  Produced by generate_pairs.py.

  dataset/train/train_source{1,2,3}.tsv
      Raw entity records used to compute live features via src/features.py.
      If the pre-computed baseline_features.tsv is found alongside the pairs
      file it is loaded directly (faster); otherwise features are recomputed
      on the fly from the source TSVs (allows the pipeline to run wherever
      the raw data lives without re-running extract_features.py).

Outputs
-------
  models/matcher.pkl                    - trained sklearn Pipeline
  models/model_config.json              - full run config + validation metrics
  output/baseline_training_results.json - detailed run results (mirrors config)
  output/baseline_threshold_results.tsv - per-threshold P/R/F0.5 table

Entity-level split
------------------
  Unique s1_entity_ids are split 80/20 (train/validation) with random_state=42.
  ZERO pair-level overlap across splits is guaranteed by construction.

Metric
------
  Entity-level macro Precision, Recall, F0.5 on the validation set.
  For each S1 entity the set of predicted matches (at a given threshold)
  is compared with the ground-truth set.  Per-entity P/R are averaged.

Usage
-----
  python src/train.py
  python src/train.py --pairs output/training_pairs_baseline.tsv
  python src/train.py --source-dir dataset/train --pairs output/training_pairs_baseline.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ?? make src/ importable when running as  python src/train.py ??????????????
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.features import FEATURE_COLS, extract_features_batch  # noqa: E402

# ?????????????????????????????????????????????????????????????????????????????
# DEFAULTS  (all overridable via CLI)
# ?????????????????????????????????????????????????????????????????????????????
DEFAULT_PAIRS     = "output/training_pairs_baseline.tsv"
DEFAULT_SOURCE_DIR = "dataset/train"
DEFAULT_FEATURES  = "output/baseline_features.tsv"   # optional pre-computed cache
DEFAULT_MODEL_OUT  = "models/matcher.pkl"
DEFAULT_CONFIG_OUT = "models/model_config.json"
DEFAULT_RESULTS    = "output/baseline_training_results.json"
DEFAULT_THRESH_TSV = "output/baseline_threshold_results.tsv"

RANDOM_STATE  = 42
TRAIN_RATIO   = 0.80
THRESHOLDS    = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]
BASELINE_VERSION = "baseline-v1"


# ?????????????????????????????????????????????????????????????????????????????
# LOGGING
# ?????????????????????????????????????????????????????????????????????????????
def log(msg: str) -> None:
    """Timestamped console logger."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ?????????????????????????????????????????????????????????????????????????????
# ARGUMENT PARSER
# ?????????????????????????????????????????????????????????????????????????????
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="M3 Baseline Training Pipeline - Amazon ML Challenge 2026"
    )
    p.add_argument("--pairs", default=DEFAULT_PAIRS,
                   help=f"Path to labelled pairs TSV. Default: {DEFAULT_PAIRS}")
    p.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR,
                   help=f"Dir containing train_source{{1,2,3}}.tsv. "
                        f"Default: {DEFAULT_SOURCE_DIR}")
    p.add_argument("--features", default=DEFAULT_FEATURES,
                   help=f"Pre-computed features TSV (optional). "
                        f"Default: {DEFAULT_FEATURES}")
    p.add_argument("--model-out", default=DEFAULT_MODEL_OUT,
                   help=f"Output path for trained Pipeline. Default: {DEFAULT_MODEL_OUT}")
    p.add_argument("--config-out", default=DEFAULT_CONFIG_OUT,
                   help=f"Output path for model config JSON. Default: {DEFAULT_CONFIG_OUT}")
    p.add_argument("--results", default=DEFAULT_RESULTS,
                   help=f"Output path for detailed results JSON. Default: {DEFAULT_RESULTS}")
    p.add_argument("--thresh-tsv", default=DEFAULT_THRESH_TSV,
                   help=f"Output path for threshold TSV. Default: {DEFAULT_THRESH_TSV}")
    return p.parse_args()


# ?????????????????????????????????????????????????????????????????????????????
# DATA LOADING
# ?????????????????????????????????????????????????????????????????????????????
def load_pairs(pairs_path: str) -> pd.DataFrame:
    """Load the labelled pair dataset."""
    log(f"Loading pairs from: {pairs_path}")
    df = pd.read_csv(
        pairs_path,
        sep="\t",
        dtype={"s1_entity_id": str, "candidate_entity_id": str,
               "candidate_source": str, "label": int},
    )
    log(f"  -> {len(df):,} pairs | {df['label'].sum():,} positive | "
        f"{(df['label'] == 0).sum():,} negative")
    return df


def load_source_records(source_dir: str) -> dict[str, dict]:
    """Load entity records from train_source{1,2,3}.tsv into a lookup dict."""
    COLS = ["entity_id", "business_name", "business_address", "country"]
    records: dict[str, dict] = {}
    for src_file in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(source_dir, src_file)
        log(f"  Loading {path} ...")
        df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False, usecols=COLS)
        for row in df.itertuples(index=False):
            records[row.entity_id] = {
                "business_name":    row.business_name,
                "business_address": row.business_address,
                "country":          row.country,
            }
        log(f"    -> {len(df):,} records. Lookup size: {len(records):,}")
    return records


def load_or_compute_features(pairs: pd.DataFrame,
                              features_path: str,
                              source_dir: str) -> pd.DataFrame:
    """
    Load pre-computed baseline_features.tsv if available; otherwise compute
    features from raw source records via src/features.py.

    The pre-computed file is ~134 MB and already aligned with the pairs TSV.
    On the first cold run (no cached file), features are recomputed live.
    """
    if os.path.exists(features_path):
        log(f"Pre-computed features found at: {features_path}")
        log("  Loading (this may take 15-30 s for ~946 K rows) ...")
        feat_df = pd.read_csv(
            features_path,
            sep="\t",
            dtype={"s1_entity_id": str, "candidate_entity_id": str,
                   "candidate_source": str, "label": int},
        )
        # Verify the feature columns are present
        missing = set(FEATURE_COLS) - set(feat_df.columns)
        if missing:
            raise ValueError(
                f"Pre-computed features file is missing columns: {missing}\n"
                f"Delete the file and re-run to recompute."
            )
        log(f"  -> {len(feat_df):,} rows loaded with {len(FEATURE_COLS)} feature cols.")
        return feat_df
    else:
        log("Pre-computed features NOT found. Computing from source records ...")
        log("  Loading source entity records ...")
        records = load_source_records(source_dir)
        log("  Running feature extraction (may take several minutes for ~946 K rows)...")
        feat_df = extract_features_batch(pairs, records)
        log(f"  -> Features computed for {len(feat_df):,} pairs.")
        return feat_df


# ?????????????????????????????????????????????????????????????????????????????
# ENTITY-LEVEL SPLIT
# ?????????????????????????????????????????????????????????????????????????????
def entity_level_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split pairs into train / validation sets by unique s1_entity_id.

    Guarantees ZERO pair-level overlap: every s1_entity_id appears in
    exactly one of the two splits.

    Parameters
    ----------
    df : DataFrame with column 's1_entity_id'
    train_ratio : fraction of unique S1 entities for training
    random_state : RNG seed for reproducibility

    Returns
    -------
    (train_df, val_df)
    """
    unique_s1 = df["s1_entity_id"].unique()
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(unique_s1))

    n_train = int(len(unique_s1) * train_ratio)
    train_ids = set(unique_s1[idx[:n_train]])
    val_ids   = set(unique_s1[idx[n_train:]])

    # Safety: no overlap
    assert len(train_ids & val_ids) == 0, "SAFETY FAIL: s1_entity_id overlap!"

    train_df = df[df["s1_entity_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["s1_entity_id"].isin(val_ids)].reset_index(drop=True)

    log(f"  Entity split -- train: {len(train_ids):,} S1 entities "
        f"({len(train_df):,} pairs) | "
        f"val: {len(val_ids):,} S1 entities ({len(val_df):,} pairs)")

    return train_df, val_df


# ?????????????????????????????????????????????????????????????????????????????
# ENTITY-LEVEL EVALUATION METRIC
# ?????????????????????????????????????????????????????????????????????????????
def entity_level_metrics(
    val_df: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """
    Compute entity-level macro Precision, Recall, and F0.5 at a given threshold.

    For each S1 entity:
      - predicted_set = {candidate_entity_id where proba >= threshold}
      - true_set      = {candidate_entity_id where label == 1}
      - entity_precision = |predicted & true| / |predicted|   (1.0 if |pred|=0)
      - entity_recall    = |predicted & true| / |true|         (1.0 if |true|=0)

    Macro average is taken over all validation S1 entities.
    F0.5 weights precision twice as heavily as recall.

    Implementation uses vectorized pandas groupby -- no Python loops over entities.

    Parameters
    ----------
    val_df    : validation DataFrame with 's1_entity_id', 'candidate_entity_id', 'label'
    proba     : 1-D array of P(label=1) probabilities, aligned with val_df rows
    threshold : classification threshold

    Returns
    -------
    (precision, recall, f0_5)  - all in [0, 1]
    """
    df = val_df[["s1_entity_id", "label"]].copy()
    pred_arr = (proba >= threshold).astype(np.int8)
    df["pred"] = pred_arr
    # TP per entity: pred AND label = pred * label  (both 0/1, product=1 iff both=1)
    df["tp_flag"] = pred_arr * df["label"].values.astype(np.int8)

    # Single groupby-sum -- fully vectorized C-level, no Python loops per entity
    agg = df.groupby("s1_entity_id", sort=False)[["tp_flag", "pred", "label"]].sum()

    tp   = agg["tp_flag"].values.astype(float)   # TP count per entity
    pp   = agg["pred"].values.astype(float)       # predicted positives per entity
    tp_g = agg["label"].values.astype(float)      # ground-truth positives per entity


    # Entity-level precision (1.0 when no predictions to avoid penalising cautious entities)
    prec_per = np.where(pp > 0, tp / pp, 1.0)
    # Entity-level recall (1.0 when no ground-truth matches for this entity)
    rec_per  = np.where(tp_g > 0, tp / tp_g, 1.0)

    macro_prec = float(prec_per.mean())
    macro_rec  = float(rec_per.mean())

    # F0.5: beta=0.5 -> weights precision twice as heavily as recall
    beta_sq = 0.25  # 0.5 ** 2
    denom = beta_sq * macro_prec + macro_rec
    f0_5 = (1 + beta_sq) * macro_prec * macro_rec / denom if denom > 0 else 0.0

    return macro_prec, macro_rec, float(f0_5)


# ?????????????????????????????????????????????????????????????????????????????
# ERROR ANALYSIS
# ?????????????????????????????????????????????????????????????????????????????
def error_analysis(
    val_df: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
    n_examples: int = 5,
) -> dict:
    """
    Identify false positives and false negatives at the chosen threshold.

    Returns a dict with:
      - fp_count : total false positive pairs
      - fn_count : total false negative pairs
      - fp_examples : list of up to n_examples FP rows (as dicts)
      - fn_examples : list of up to n_examples FN rows (as dicts)
      - fp_feature_means : mean feature values for FP pairs
      - fn_feature_means : mean feature values for FN pairs
    """
    analysis_df = val_df.copy().reset_index(drop=True)
    analysis_df["proba"] = proba
    analysis_df["pred"]  = (proba >= threshold).astype(int)

    fp_mask = (analysis_df["pred"] == 1) & (analysis_df["label"] == 0)
    fn_mask = (analysis_df["pred"] == 0) & (analysis_df["label"] == 1)

    fp_df = analysis_df[fp_mask]
    fn_df = analysis_df[fn_mask]

    # Feature means for error groups
    feat_fp_means = (
        fp_df[FEATURE_COLS].mean().round(6).to_dict()
        if len(fp_df) > 0 else {}
    )
    feat_fn_means = (
        fn_df[FEATURE_COLS].mean().round(6).to_dict()
        if len(fn_df) > 0 else {}
    )

    # Sample examples
    example_cols = ["s1_entity_id", "candidate_entity_id",
                    "candidate_source", "label", "proba"] + FEATURE_COLS
    available_cols = [c for c in example_cols if c in analysis_df.columns]

    def to_examples(df: pd.DataFrame) -> list[dict]:
        return (
            df[available_cols]
            .head(n_examples)
            .round(6)
            .to_dict(orient="records")
        )

    return {
        "fp_count":         int(fp_mask.sum()),
        "fn_count":         int(fn_mask.sum()),
        "fp_examples":      to_examples(fp_df),
        "fn_examples":      to_examples(fn_df),
        "fp_feature_means": feat_fp_means,
        "fn_feature_means": feat_fn_means,
    }


# ?????????????????????????????????????????????????????????????????????????????
# MAIN PIPELINE
# ?????????????????????????????????????????????????????????????????????????????
def main() -> None:
    args = parse_args()
    t_start = time.time()

    # ?? 0. Create output directories ????????????????????????????????????????
    for d in [os.path.dirname(args.model_out),
              os.path.dirname(args.config_out),
              os.path.dirname(args.results),
              os.path.dirname(args.thresh_tsv)]:
        if d:
            os.makedirs(d, exist_ok=True)

    print()
    print("=" * 68)
    print("  M3 BASELINE TRAINING PIPELINE - Amazon ML Challenge 2026")
    print("=" * 68)

    # ?? 1. Load pairs ????????????????????????????????????????????????????????
    log("Step 1/7  Load labelled pairs")
    pairs = load_pairs(args.pairs)

    # ?? 2. Load / compute features ???????????????????????????????????????????
    log("Step 2/7  Load / compute features")
    feat_df = load_or_compute_features(pairs, args.features, args.source_dir)

    # Ensure label column is integer
    feat_df["label"] = feat_df["label"].astype(int)

    # ?? 3. Entity-level train / validation split ??????????????????????????????
    log("Step 3/7  Entity-level split (80/20, random_state=42)")
    train_df, val_df = entity_level_split(
        feat_df, train_ratio=TRAIN_RATIO, random_state=RANDOM_STATE
    )

    # Verify zero overlap
    train_s1 = set(train_df["s1_entity_id"].unique())
    val_s1   = set(val_df["s1_entity_id"].unique())
    assert len(train_s1 & val_s1) == 0, "CRITICAL: s1_entity_id overlap found!"
    log(f"  OK Zero s1_entity_id overlap confirmed between train and validation.")

    train_s1_count = len(train_s1)
    val_s1_count   = len(val_s1)
    train_pair_count = len(train_df)
    val_pair_count   = len(val_df)

    # ?? 4. Prepare feature matrices ??????????????????????????????????????????
    log("Step 4/7  Build feature matrices")
    X_train = train_df[FEATURE_COLS].values.astype(float)
    y_train = train_df["label"].values.astype(int)
    X_val   = val_df[FEATURE_COLS].values.astype(float)
    y_val   = val_df["label"].values.astype(int)

    log(f"  X_train: {X_train.shape}  |  y_train positives: {y_train.sum():,}")
    log(f"  X_val:   {X_val.shape}    |  y_val positives:   {y_val.sum():,}")

    # Sanity: no NaN in feature matrices
    nan_train = np.isnan(X_train).sum()
    nan_val   = np.isnan(X_val).sum()
    if nan_train or nan_val:
        raise ValueError(
            f"NaN detected in feature matrices: "
            f"train={nan_train}, val={nan_val}. "
            f"Check features.py or the source data."
        )
    log("  OK No NaN values in feature matrices.")

    # ?? 5. Train sklearn Pipeline ????????????????????????????????????????????
    log("Step 5/7  Train sklearn Pipeline (StandardScaler + LogisticRegression)")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ])
    t_fit = time.time()
    pipeline.fit(X_train, y_train)
    log(f"  OK Training complete in {time.time() - t_fit:.1f}s")

    # ?? 6. Threshold sweep on validation set ????????????????????????????????
    log("Step 6/7  Threshold sweep on validation set")
    proba_val = pipeline.predict_proba(X_val)[:, 1]

    threshold_rows: list[dict] = []
    best_f05        = -1.0
    best_threshold  = THRESHOLDS[0]
    best_precision  = 0.0
    best_recall     = 0.0

    print()
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F0.5':>10}")
    print("  " + "-" * 46)

    for thr in THRESHOLDS:
        prec, rec, f05 = entity_level_metrics(val_df, proba_val, thr)
        threshold_rows.append({
            "threshold": thr,
            "precision": round(prec, 6),
            "recall":    round(rec,  6),
            "f0_5":      round(f05,  6),
        })
        marker = " <- best" if f05 > best_f05 else ""
        print(f"  {thr:>10.2f}  {prec:>10.6f}  {rec:>10.6f}  {f05:>10.6f}{marker}")

        if f05 > best_f05:
            best_f05       = f05
            best_threshold = thr
            best_precision = prec
            best_recall    = rec

    print()
    log(f"  Best threshold : {best_threshold}")
    log(f"  Precision      : {best_precision:.6f}")
    log(f"  Recall         : {best_recall:.6f}")
    log(f"  F0.5           : {best_f05:.6f}")

    # ?? 7. Error analysis ????????????????????????????????????????????????????
    log("Step 7/7  Error analysis at chosen threshold")
    err = error_analysis(val_df, proba_val, best_threshold)
    log(f"  False positives : {err['fp_count']:,}")
    log(f"  False negatives : {err['fn_count']:,}")

    # ?? 8. Save artifacts ????????????????????????????????????????????????????
    log("Saving artifacts ...")

    # ---- 8a. Model pickle ----
    joblib.dump(pipeline, args.model_out)
    log(f"  OK Model saved -> {args.model_out}")

    # ---- 8b. Threshold sweep TSV ----
    thresh_df = pd.DataFrame(threshold_rows)
    thresh_df.to_csv(args.thresh_tsv, sep="\t", index=False)
    log(f"  OK Threshold table -> {args.thresh_tsv}")

    # ---- 8c. model_config.json ----
    config = {
        "model_type":          "LogisticRegression",
        "feature_names":       FEATURE_COLS,
        "threshold":           best_threshold,
        "random_seed":         RANDOM_STATE,
        "train_s1_count":      train_s1_count,
        "validation_s1_count": val_s1_count,
        "train_pair_count":    train_pair_count,
        "validation_pair_count": val_pair_count,
        "precision":           round(best_precision, 6),
        "recall":              round(best_recall,    6),
        "f0_5":                round(best_f05,       6),
        "baseline_version":    BASELINE_VERSION,
    }
    with open(args.config_out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    log(f"  OK Model config -> {args.config_out}")

    # ---- 8d. baseline_training_results.json ----
    results = {
        **config,
        "elapsed_seconds":     round(time.time() - t_start, 2),
        "threshold_sweep":     threshold_rows,
        "error_analysis":      err,
        "sklearn_pipeline":    str(pipeline),
    }
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"  OK Detailed results -> {args.results}")

    # ?? Final summary ????????????????????????????????????????????????????????
    elapsed = time.time() - t_start
    print()
    print("=" * 68)
    print("  TRAINING COMPLETE")
    print("=" * 68)
    print(f"  Train S1 entities     : {train_s1_count:,}")
    print(f"  Validation S1 entities: {val_s1_count:,}")
    print(f"  Train pairs           : {train_pair_count:,}")
    print(f"  Validation pairs      : {val_pair_count:,}")
    print()
    print(f"  Selected threshold    : {best_threshold}")
    print(f"  Precision             : {best_precision:.6f}")
    print(f"  Recall                : {best_recall:.6f}")
    print(f"  F0.5                  : {best_f05:.6f}")
    print()
    print(f"  Model artifact        : {os.path.abspath(args.model_out)}")
    print(f"  Config                : {os.path.abspath(args.config_out)}")
    print(f"  Threshold table       : {os.path.abspath(args.thresh_tsv)}")
    print(f"  Detailed results      : {os.path.abspath(args.results)}")
    print(f"  Elapsed               : {elapsed:.1f}s")
    print("=" * 68)


if __name__ == "__main__":
    main()
