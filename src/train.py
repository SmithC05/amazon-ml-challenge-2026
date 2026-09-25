"""
src/train.py
============
Baseline training pipeline for the Amazon ML Challenge 2026.

Member 3 deliverable — baseline matching module.

Pipeline
--------
1. Load train_source1/2/3.tsv and train_ground_truth.tsv.
2. Apply M2 normalization (src/preprocess.py) to name and address fields.
3. Load candidate_pairs.tsv produced by Member 4 (M4).
4. Build labeled pair rows: join candidate pairs with source records.
5. Extract the 16 baseline features (src/features.py).
6. Entity-level train/validation split (80/20, random_state=42).
7. Train StandardScaler + LogisticRegression baseline.
8. Threshold sweep (0.50 → 0.95) using the official entity-level macro F0.5.
9. Save model artifacts and results.

Official metric
---------------
Entity-level macro F0.5 — computed as follows for EACH labeled S1 entity:

    truth = set of true matched IDs for that entity
    predicted = set of candidate IDs predicted as matches at threshold t

    If truth == {} and predicted == {}:  entity_f05 = 1.0
    If truth == {} and predicted != {}:  entity_f05 = 0.0
    Otherwise:
        precision = |truth ∩ predicted| / |predicted|  (0 if predicted empty)
        recall    = |truth ∩ predicted| / |truth|      (0 if truth empty)
        entity_f05 = (1 + 0.5²) · P · R / (0.5² · P + R)  if P+R > 0 else 0

Final score = mean(entity_f05) across all validation S1 entities.

Outputs
-------
models/matcher.pkl           — (scaler, model) tuple
models/model_config.json     — full experiment metadata
output/baseline_training_results.json
output/baseline_threshold_results.tsv
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import normalize_name, normalize_address   # noqa: E402  (M2)
from features import build_feature_matrix, FEATURE_NAMES  # noqa: E402  (M3)
from cache import load_all_cache, cache_exists             # noqa: E402  (M2 cache)

RANDOM_STATE = 42
BETA = 0.5   # F0.5: precision-weighted


# ─────────────────────────────────────────────────────────────────────────────
# Official metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    """F-beta score for a single (precision, recall) pair."""
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def entity_f05(truth: set[str], predicted: set[str]) -> float:
    """
    Entity-level F0.5 following the competition definition.

      truth empty  AND predicted empty  → 1.0
      truth empty  AND predicted non-empty → 0.0
      otherwise standard precision/recall/F0.5
    """
    if not truth and not predicted:
        return 1.0
    if not truth:
        return 0.0
    tp = len(truth & predicted)
    prec = tp / len(predicted) if predicted else 0.0
    rec  = tp / len(truth)
    return f_beta(prec, rec)


def macro_f05(
    s1_ids: list[str],
    truth_map: dict[str, set[str]],
    pred_map: dict[str, set[str]],
) -> float:
    """
    Mean entity-level F0.5 across all S1 entities in *s1_ids*.

    Parameters
    ----------
    s1_ids   : list of S1 entity IDs to evaluate
    truth_map: {s1_id → set of true matched IDs}
    pred_map : {s1_id → set of predicted IDs at this threshold}
    """
    scores = [
        entity_f05(truth_map.get(sid, set()), pred_map.get(sid, set()))
        for sid in s1_ids
    ]
    return float(np.mean(scores)) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sources(
    data_dir: Path,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and normalize all four training TSVs.

    When *cache_dir* is supplied and the M2 Parquet cache is present,
    source records are loaded from cache (no re-normalization).  Falls back
    to raw TSV + M2 normalization when cache is absent.
    """
    gt = pd.read_csv(data_dir / "train_ground_truth.tsv", sep="\t")

    if cache_dir is not None and cache_exists(cache_dir, "train"):
        print("  Loading sources from M2 Parquet cache...")
        s1, s2, s3 = load_all_cache(cache_dir, "train")
    else:
        if cache_dir is not None:
            print("  Cache not found — falling back to raw TSV + M2 normalization")
        s1 = pd.read_csv(data_dir / "train_source1.tsv", sep="\t")
        s2 = pd.read_csv(data_dir / "train_source2.tsv", sep="\t")
        s3 = pd.read_csv(data_dir / "train_source3.tsv", sep="\t")
        # Apply M2 normalization — never reimplemented here
        for df in (s1, s2, s3):
            df["business_name_norm"]    = df["business_name"].apply(normalize_name)
            df["business_address_norm"] = df["business_address"].apply(normalize_address)

    print(f"Loaded  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  GT={len(gt):,}")
    return s1, s2, s3, gt


def build_truth_map(gt: pd.DataFrame) -> dict[str, set[str]]:
    """
    Build {s1_id → set of true matched IDs} from the ground-truth frame.
    Zero-match rows (NaN / empty matched_entity_ids) map to an empty set.
    """
    truth: dict[str, set[str]] = {}
    for _, row in gt.iterrows():
        sid = row["source1_entity_id"]
        val = row["matched_entity_ids"]
        if pd.isna(val) or str(val).strip() == "":
            truth[sid] = set()
        else:
            truth[sid] = {x.strip() for x in str(val).split(",") if x.strip()}
    return truth


def build_pair_rows(
    candidates: pd.DataFrame,
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    truth_map: dict[str, set[str]],
) -> pd.DataFrame:
    """
    Join candidate pairs with source records and ground truth labels.

    candidate_pairs.tsv columns: source1_entity_id, candidate_entity_ids
    (candidate_entity_ids may be comma-separated; one row per candidate)
    """
    # Build lookup tables keyed by entity_id
    s2_lookup = s2.set_index("entity_id")[
        ["business_name_norm", "business_address_norm", "country"]
    ].rename(columns={
        "business_name_norm": "cand_name_norm",
        "business_address_norm": "cand_address_norm",
        "country": "cand_country",
    })
    s3_lookup = s3.set_index("entity_id")[
        ["business_name_norm", "business_address_norm", "country"]
    ].rename(columns={
        "business_name_norm": "cand_name_norm",
        "business_address_norm": "cand_address_norm",
        "country": "cand_country",
    })
    cand_lookup = pd.concat([s2_lookup, s3_lookup])

    s1_lookup = s1.set_index("entity_id")[
        ["business_name_norm", "business_address_norm", "country"]
    ].rename(columns={
        "business_name_norm": "s1_name_norm",
        "business_address_norm": "s1_address_norm",
        "country": "s1_country",
    })

    rows = []
    for _, cand_row in candidates.iterrows():
        s1_id = cand_row["source1_entity_id"]
        raw_ids = str(cand_row.get("candidate_entity_ids", "") or "")
        cand_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]

        if s1_id not in s1_lookup.index:
            continue
        s1_rec = s1_lookup.loc[s1_id]
        truth_set = truth_map.get(s1_id, set())

        for cid in cand_ids:
            if cid not in cand_lookup.index:
                continue
            cand_rec = cand_lookup.loc[cid]
            rows.append({
                "source1_entity_id":  s1_id,
                "cand_entity_id":     cid,
                "s1_name_norm":       s1_rec["s1_name_norm"],
                "s1_address_norm":    s1_rec["s1_address_norm"],
                "s1_country":         s1_rec["s1_country"],
                "cand_name_norm":     cand_rec["cand_name_norm"],
                "cand_address_norm":  cand_rec["cand_address_norm"],
                "cand_country":       cand_rec["cand_country"],
                "label":              int(cid in truth_set),
            })

    return pd.DataFrame(rows)


def entity_split(
    pair_df: pd.DataFrame,
    train_frac: float = 0.80,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *pair_df* by unique S1 entity (never row-level).

    Returns (train_pairs, val_pairs).  The intersection of train and val
    S1 IDs is guaranteed to be empty.
    """
    s1_ids = pair_df["source1_entity_id"].unique()
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(s1_ids)
    n_train = int(len(shuffled) * train_frac)
    train_ids = set(shuffled[:n_train])
    val_ids   = set(shuffled[n_train:])

    assert train_ids & val_ids == set(), "Split leak: shared S1 IDs!"

    train_df = pair_df[pair_df["source1_entity_id"].isin(train_ids)].copy()
    val_df   = pair_df[pair_df["source1_entity_id"].isin(val_ids)].copy()
    return train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep(
    val_df: pd.DataFrame,
    proba: np.ndarray,
    truth_map: dict[str, set[str]],
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate a range of decision thresholds on the validation set.

    Returns a DataFrame with columns: threshold, precision, recall, f0_5.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]

    val_s1_ids = val_df["source1_entity_id"].unique().tolist()
    results = []

    for t in thresholds:
        pred_positive = proba >= t
        pred_map: dict[str, set[str]] = {}
        for sid in val_s1_ids:
            pred_map[sid] = set()

        mask_df = val_df.copy()
        mask_df["_pos"] = pred_positive
        for sid, grp in mask_df.groupby("source1_entity_id"):
            pred_map[sid] = set(grp.loc[grp["_pos"], "cand_entity_id"])

        # Per-entity precision / recall (for the aggregate report)
        all_prec, all_rec, all_f05 = [], [], []
        for sid in val_s1_ids:
            tr = truth_map.get(sid, set())
            pr = pred_map.get(sid, set())
            tp = len(tr & pr)
            p = tp / len(pr) if pr else 0.0
            r = tp / len(tr) if tr else 0.0
            all_prec.append(p)
            all_rec.append(r)
            all_f05.append(entity_f05(tr, pr))

        results.append({
            "threshold": t,
            "precision": float(np.mean(all_prec)),
            "recall":    float(np.mean(all_rec)),
            "f0_5":      float(np.mean(all_f05)),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry-point
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_dir: Path,
    candidates_path: Path,
    output_dir: Path,
    models_dir: Path,
    cache_dir: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data (uses cache when available)
    s1, s2, s3, gt = load_sources(data_dir, cache_dir=cache_dir)
    truth_map = build_truth_map(gt)

    # 2. Load candidate pairs (M4 artifact)
    candidates = pd.read_csv(candidates_path, sep="\t")
    print(f"Candidate rows: {len(candidates):,}")

    # 3. Build labeled pair rows
    pair_df = build_pair_rows(candidates, s1, s2, s3, truth_map)
    print(f"Labeled pairs : {len(pair_df):,}  "
          f"(positive={pair_df['label'].sum():,}, "
          f"negative={(pair_df['label'] == 0).sum():,})")

    # 4. Entity-level split
    train_df, val_df = entity_split(pair_df)
    train_s1 = set(train_df["source1_entity_id"].unique())
    val_s1   = set(val_df["source1_entity_id"].unique())
    print(f"Train S1={len(train_s1):,}  pairs={len(train_df):,}")
    print(f"Val   S1={len(val_s1):,}   pairs={len(val_df):,}")
    assert train_s1 & val_s1 == set(), "Entity split leak!"

    # 5. Feature extraction
    X_train = build_feature_matrix(train_df).values
    y_train = train_df["label"].values
    X_val   = build_feature_matrix(val_df).values
    y_val   = val_df["label"].values

    # 6. Train baseline: StandardScaler + LogisticRegression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train_sc, y_train)
    print("Model trained.")

    # 7. Threshold sweep on validation set
    val_proba = model.predict_proba(X_val_sc)[:, 1]
    sweep_df = threshold_sweep(val_df, val_proba, truth_map)
    best_row = sweep_df.loc[sweep_df["f0_5"].idxmax()]
    best_threshold = float(best_row["threshold"])
    best_f05       = float(best_row["f0_5"])
    best_prec      = float(best_row["precision"])
    best_rec       = float(best_row["recall"])

    print(f"\nThreshold sweep results:")
    print(sweep_df.to_string(index=False))
    print(f"\nSelected threshold : {best_threshold:.2f}")
    print(f"Validation F0.5    : {best_f05:.4f}")
    print(f"Validation Precision: {best_prec:.4f}")
    print(f"Validation Recall  : {best_rec:.4f}")

    # 8. Save model
    model_path = models_dir / "matcher.pkl"
    with open(model_path, "wb") as f:
        pickle.dump((scaler, model), f)
    print(f"\nModel saved → {model_path}")

    # 9. Save model config
    config = {
        "model_type":            "StandardScaler + LogisticRegression",
        "feature_names":         FEATURE_NAMES,
        "threshold":             best_threshold,
        "random_seed":           RANDOM_STATE,
        "train_s1_count":        len(train_s1),
        "validation_s1_count":   len(val_s1),
        "train_pair_count":      int(len(train_df)),
        "validation_pair_count": int(len(val_df)),
        "precision":             round(best_prec, 6),
        "recall":                round(best_rec, 6),
        "f0_5":                  round(best_f05, 6),
        "baseline_version":      "1.0",
        "metric":                "entity-level macro F0.5 (competition official)",
        "candidates_source":     str(candidates_path),
    }
    config_path = models_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved  → {config_path}")

    # 10. Save training results
    train_results_path = output_dir / "baseline_training_results.json"
    with open(train_results_path, "w") as f:
        json.dump(config, f, indent=2)

    # 11. Save threshold sweep table
    sweep_path = output_dir / "baseline_threshold_results.tsv"
    sweep_df.to_csv(sweep_path, sep="\t", index=False)
    print(f"Sweep saved   → {sweep_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train baseline matcher.")
    parser.add_argument("--data-dir",   default="dataset/train",   help="Directory with train TSVs")
    parser.add_argument("--candidates", default="output/candidate_pairs.tsv", help="M4 candidate pairs TSV")
    parser.add_argument("--output-dir", default="output",  help="Directory for result artifacts")
    parser.add_argument("--models-dir", default="models",  help="Directory for model artifacts")
    parser.add_argument("--cache-dir",  default=None,      help="M2 Parquet cache directory (optional)")
    args = parser.parse_args()

    train(
        data_dir=Path(args.data_dir),
        candidates_path=Path(args.candidates),
        output_dir=Path(args.output_dir),
        models_dir=Path(args.models_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
