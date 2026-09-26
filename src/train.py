"""
src/train.py
============
Baseline training pipeline for the Amazon ML Challenge 2026.

Member 3 deliverable — baseline matching module.

Pipeline
--------
1. Load train_source1/2/3.tsv and train_ground_truth.tsv.
2. Load entity data from M2 Parquet cache (never re-normalize).
3. Load candidate_pairs.tsv produced by Member 4 (M4) — STREAMED in chunks.
4. Build labeled pair rows: join candidate pairs with source records in batches.
5. Accumulate features in bounded batches into a Parquet spool file on disk.
6. Entity-level train/validation split (80/20, random_state=42).
7. Load the full feature matrix from the spool (bounded by max_train_pairs
   for training, full val set for threshold sweep).
8. Train StandardScaler + LogisticRegression baseline.
9. Threshold sweep (0.50 → 0.95) using the official entity-level macro F0.5.
10. Save model artifacts and results.

Memory-safety architecture (v2 — streaming)
--------------------------------------------
  The candidate TSV is NEVER loaded all at once.  Pairs are built in
  bounded batches (--chunk-size rows from the candidate file at a time) and
  written to a temporary Parquet spool on disk.

  After all candidates are processed, the spool is read back:
    • Training rows    — up to max_train_pairs (default 2_000_000).
    • Validation rows  — all (not subject to the training cap).

  If the full pair volume exceeds max_train_pairs the code applies a
  DETERMINISTIC PER-ENTITY NEGATIVE-SAMPLING STRATEGY:
    • All positive (labeled=1) pairs are retained.
    • Negatives are sampled per S1 entity to fill the remaining budget,
      with fixed seed = RANDOM_STATE.
  This is explicit and reported; it never silently truncates data.

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
import logging
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import normalize_name, normalize_address, preprocess_dataframe  # noqa: E402  (M2)
from features import build_feature_matrix_from_dicts, FEATURE_NAMES            # noqa: E402  (M3)
from cache import load_all_cache, cache_exists                                   # noqa: E402  (M2 cache)

logger = logging.getLogger(__name__)

RANDOM_STATE    = 42
BETA            = 0.5   # F0.5: precision-weighted
_DEFAULT_CHUNK  = 50_000
# Default cap for training pairs.  At 16 features × 8 bytes × 2_000_000 rows
# the feature matrix is ~256 MB — safely in RAM for LogisticRegression.
_DEFAULT_MAX_TRAIN_PAIRS = 2_000_000


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
    """Mean entity-level F0.5 across all S1 entities in *s1_ids*."""
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
    Uses M2 Parquet cache when available.
    """
    gt = pd.read_csv(data_dir / "train_ground_truth.tsv", sep="\t")

    if cache_dir is not None and all(
        cache_exists("train", src, cache_dir)
        for src in ("source1", "source2", "source3")
    ):
        print("  Loading sources from M2 Parquet cache...")
        s1, s2, s3 = load_all_cache(cache_dir, "train")
    else:
        if cache_dir is not None:
            print("  Cache not found — falling back to raw TSV + M2 normalization")
        s1 = preprocess_dataframe(pd.read_csv(data_dir / "train_source1.tsv", sep="\t", dtype=str))
        s2 = preprocess_dataframe(pd.read_csv(data_dir / "train_source2.tsv", sep="\t", dtype=str))
        s3 = preprocess_dataframe(pd.read_csv(data_dir / "train_source3.tsv", sep="\t", dtype=str))

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


def _build_entity_lookups(
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> dict[str, dict]:
    """Build compact O(1) candidate lookup dicts from S2 and S3 DataFrames."""
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
    """Build compact O(1) S1 lookup dict."""
    s1_lookup: dict[str, dict] = {}
    for row in s1[["entity_id", "name_norm", "address_norm", "country"]].itertuples(index=False):
        s1_lookup[row.entity_id] = {
            "name_norm":    row.name_norm,
            "address_norm": row.address_norm,
            "country":      row.country,
        }
    return s1_lookup


def _expand_and_label_chunk(
    chunk_df: pd.DataFrame,
    s1_lookup: dict[str, dict],
    cand_lookup: dict[str, dict],
    truth_map: dict[str, set[str]],
    val_s1_set: set[str],
) -> list[dict]:
    """
    Expand a candidate chunk into labeled pair dicts.
    Pairs where S1/candidate are not in the lookups are skipped.
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
        truth_set = truth_map.get(s1_id, set())
        is_val = s1_id in val_s1_set

        for cid in cand_ids:
            cand_rec = cand_lookup.get(cid)
            if cand_rec is None:
                continue
            pairs.append({
                "source1_entity_id":  s1_id,
                "cand_entity_id":     cid,
                "s1_name_norm":       s1_rec["name_norm"],
                "s1_address_norm":    s1_rec["address_norm"],
                "s1_country":         s1_rec["country"],
                "cand_name_norm":     cand_rec["name_norm"],
                "cand_address_norm":  cand_rec["address_norm"],
                "cand_country":       cand_rec["country"],
                "label":              int(cid in truth_set),
                "is_val":             int(is_val),
            })
    return pairs


def entity_split(
    all_s1_ids: list[str],
    train_frac: float = 0.80,
    random_state: int = RANDOM_STATE,
) -> tuple[set[str], set[str]]:
    """
    Split S1 entity population into train and validation sets.
    Returns (train_ids_set, val_ids_set).
    """
    population = np.array(all_s1_ids, dtype=object)
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(population)
    n_train = int(len(shuffled) * train_frac)
    train_ids = set(shuffled[:n_train])
    val_ids   = set(shuffled[n_train:])
    assert train_ids & val_ids == set(), "Split leak: shared S1 IDs!"
    return train_ids, val_ids


def _apply_train_cap(
    train_df: pd.DataFrame,
    max_pairs: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    If the training set has more than *max_pairs* rows, apply deterministic
    negative sampling while retaining ALL positive pairs.

    Strategy
    --------
    1. Extract all positives (label=1) — always kept.
    2. If positives alone exceed max_pairs, just return all positives (log warning).
    3. Otherwise: sample negatives per S1 entity proportionally until the
       budget (max_pairs - n_positives) is filled.  Sampling uses fixed seed.

    This is never silent: the function always prints exactly how many pairs
    were available vs used, and the sampling settings.

    Returns the capped training DataFrame (deterministic).
    """
    n_available = len(train_df)
    if n_available <= max_pairs:
        print(f"  Training pairs available : {n_available:,} (≤ cap of {max_pairs:,}, no sampling needed)")
        return train_df

    pos_df = train_df[train_df["label"] == 1].copy()
    neg_df = train_df[train_df["label"] == 0].copy()
    n_pos  = len(pos_df)
    n_neg_budget = max_pairs - n_pos

    print(
        f"\n⚠ TRAINING PAIR CAP APPLIED"
        f"\n  Available pairs  : {n_available:,}  (pos={n_pos:,}  neg={len(neg_df):,})"
        f"\n  Max training cap : {max_pairs:,}"
        f"\n  Positives kept   : {n_pos:,} (ALL — never sampled)"
        f"\n  Neg budget       : {n_neg_budget:,}"
        f"\n  Sampling seed    : {random_state}"
    )

    if n_pos >= max_pairs:
        print(f"  ⚠ Positives alone ({n_pos:,}) exceed cap — returning all positives only.")
        return pos_df

    # Sample negatives: per-entity proportional then global cap
    neg_sampled = (
        neg_df
        .groupby("source1_entity_id", group_keys=False)
        .apply(lambda g: g.sample(frac=1.0, random_state=random_state))  # shuffle within entity
        .sample(
            n=min(n_neg_budget, len(neg_df)),
            random_state=random_state,
            replace=False,
        )
    )

    result = pd.concat([pos_df, neg_sampled], ignore_index=True)
    print(f"  Sampled negatives: {len(neg_sampled):,}")
    print(f"  Final training   : {len(result):,} pairs (pos={n_pos:,} neg={len(neg_sampled):,})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep(
    val_df: pd.DataFrame,
    proba: np.ndarray,
    truth_map: dict[str, set[str]],
    thresholds: list[float] | None = None,
    val_s1_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Evaluate a range of decision thresholds on the validation set.

    Parameters
    ----------
    val_df     : Candidate-pair rows assigned to validation.
    proba      : Predicted positive probabilities (aligned with val_df rows).
    truth_map  : {s1_id → set of true matched IDs}.
    thresholds : Thresholds to evaluate.  Defaults to 0.50 … 0.95 step 0.05.
    val_s1_ids : COMPLETE list of validation S1 entity IDs, including those
                 with zero candidate pairs.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]

    if val_s1_ids is None:
        val_s1_ids = val_df["source1_entity_id"].unique().tolist()

    results = []

    for t in thresholds:
        pred_positive = proba >= t
        pred_map: dict[str, set[str]] = {sid: set() for sid in val_s1_ids}

        mask_df = val_df.copy()
        mask_df["_pos"] = pred_positive
        for sid, grp in mask_df.groupby("source1_entity_id"):
            pred_map[str(sid)] = set(grp.loc[grp["_pos"], "cand_entity_id"])

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
    chunk_size: int = _DEFAULT_CHUNK,
    max_train_pairs: int = _DEFAULT_MAX_TRAIN_PAIRS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data (uses cache when available)
    s1, s2, s3, gt = load_sources(data_dir, cache_dir=cache_dir)
    truth_map = build_truth_map(gt)
    all_s1_ids = s1["entity_id"].tolist()

    # 2. Entity-level split BEFORE processing candidates — ensures zero-candidate
    #    entities are properly assigned to train/val.
    train_s1_set, val_s1_set = entity_split(all_s1_ids)
    train_s1_list = sorted(train_s1_set)
    val_s1_list   = sorted(val_s1_set)
    print(f"Split: Train S1={len(train_s1_set):,}  Val S1={len(val_s1_set):,}")

    # 3. Build compact entity lookup dicts (one-time)
    print("Building entity lookup dicts...")
    s1_lookup   = _build_s1_lookup(s1)
    cand_lookup = _build_entity_lookups(s2, s3)
    del s2, s3  # free DataFrame RAM
    print(f"  S1 lookup  : {len(s1_lookup):,}")
    print(f"  Cand lookup: {len(cand_lookup):,} (S2+S3)")

    # 4. Stream candidate TSV → spool pairs to a temp Parquet file
    #    to avoid holding all pair rows in RAM.
    spool_dir  = output_dir / ".pair_spool"
    spool_dir.mkdir(exist_ok=True)
    spool_path = spool_dir / "pairs.parquet"

    total_cand_rows = 0
    total_pair_rows = 0
    total_pos       = 0
    total_neg       = 0
    chunk_num       = 0
    spool_frames: list[pd.DataFrame] = []  # batch before writing
    SPOOL_BATCH = 10  # write every 10 chunks to avoid too many small files

    print(f"\nStreaming candidates (chunk_size={chunk_size:,}, spooling to {spool_path}) ...")
    all_frames_for_write: list[pd.DataFrame] = []

    for chunk_df in pd.read_csv(
        candidates_path,
        sep="\t",
        dtype=str,
        chunksize=chunk_size,
        keep_default_na=False,
    ):
        chunk_num       += 1
        total_cand_rows += len(chunk_df)

        pair_dicts = _expand_and_label_chunk(chunk_df, s1_lookup, cand_lookup, truth_map, val_s1_set)
        n_pairs = len(pair_dicts)
        total_pair_rows += n_pairs

        if n_pairs == 0:
            print(f"  chunk {chunk_num:4d}: {len(chunk_df):6,} cand rows → 0 pairs (skipped)")
            continue

        chunk_pair_df = pd.DataFrame(pair_dicts)
        n_pos_chunk = int(chunk_pair_df["label"].sum())
        total_pos += n_pos_chunk
        total_neg += n_pairs - n_pos_chunk

        all_frames_for_write.append(chunk_pair_df)
        print(
            f"  chunk {chunk_num:4d}: {len(chunk_df):6,} cand rows → "
            f"{n_pairs:7,} pairs (pos={n_pos_chunk:,})"
        )

    # Write spool
    if all_frames_for_write:
        spool_df = pd.concat(all_frames_for_write, ignore_index=True)
        spool_df.to_parquet(spool_path, index=False)
        del all_frames_for_write, spool_df

    print(
        f"\nIngestion complete:"
        f"\n  candidate rows : {total_cand_rows:,}"
        f"\n  pair rows      : {total_pair_rows:,}  (pos={total_pos:,}  neg={total_neg:,})"
    )

    if not spool_path.exists() or total_pair_rows == 0:
        raise RuntimeError("No pair rows were produced — cannot train. Check candidate file.")

    # 5. Load spool and split into train/val
    print("\nLoading spool from disk...")
    all_pairs_df = pd.read_parquet(spool_path)
    train_df = all_pairs_df[all_pairs_df["is_val"] == 0].drop(columns=["is_val"]).copy()
    val_df   = all_pairs_df[all_pairs_df["is_val"] == 1].drop(columns=["is_val"]).copy()
    del all_pairs_df

    # Count zero-candidate val entities
    val_s1_with_pairs = set(val_df["source1_entity_id"].unique())
    n_zero_cand_val   = len(val_s1_set - val_s1_with_pairs)

    print(f"Train S1={len(train_s1_set):,}  train pairs={len(train_df):,}")
    print(f"Val   S1={len(val_s1_set):,}   val   pairs={len(val_df):,}")
    print(f"Val   S1 with zero candidate rows : {n_zero_cand_val:,}")
    assert train_s1_set & val_s1_set == set(), "Entity split leak!"

    # 6. Apply training pair cap with deterministic negative sampling if needed
    train_df = _apply_train_cap(train_df, max_train_pairs)
    n_train_pos = int(train_df["label"].sum())
    n_train_neg = len(train_df) - n_train_pos
    sampling_applied = len(train_df) < total_pair_rows - len(val_df)

    # 7. Feature extraction (chunked within RAM — these matrices are bounded)
    print("\nExtracting features...")
    train_dicts = train_df.to_dict("records")
    val_dicts   = val_df.to_dict("records")

    X_train = build_feature_matrix_from_dicts(train_dicts)
    y_train = train_df["label"].values
    X_val   = build_feature_matrix_from_dicts(val_dicts)
    y_val   = val_df["label"].values
    del train_dicts, val_dicts

    print(f"  X_train shape: {X_train.shape}  X_val shape: {X_val.shape}")

    # 8. Train baseline: StandardScaler + LogisticRegression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train_sc, y_train)
    print("Model trained.")

    # 9. Threshold sweep — pass complete val population (incl. zero-cand entities)
    val_proba = model.predict_proba(X_val_sc)[:, 1]
    sweep_df = threshold_sweep(
        val_df, val_proba, truth_map,
        val_s1_ids=val_s1_list,
    )
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

    # 10. Save model
    model_path = models_dir / "matcher.pkl"
    with open(model_path, "wb") as f:
        pickle.dump((scaler, model), f)
    print(f"\nModel saved → {model_path}")

    # 11. Save model config
    config = {
        "model_type":               "StandardScaler + LogisticRegression",
        "feature_names":            FEATURE_NAMES,
        "threshold":                best_threshold,
        "random_seed":              RANDOM_STATE,
        "train_s1_count":           len(train_s1_set),
        "validation_s1_count":      len(val_s1_set),
        "val_s1_zero_cand":         n_zero_cand_val,
        "available_candidate_pairs":total_pair_rows,
        "train_pair_count":         int(len(train_df)),
        "train_positive_pairs":     n_train_pos,
        "train_negative_pairs":     n_train_neg,
        "validation_pair_count":    int(len(val_df)),
        "sampling_applied":         sampling_applied,
        "max_train_pairs_cap":      max_train_pairs,
        "sampling_seed":            RANDOM_STATE if sampling_applied else None,
        "precision":                round(best_prec, 6),
        "recall":                   round(best_rec, 6),
        "f0_5":                     round(best_f05, 6),
        "baseline_version":         "2.0-streaming",
        "metric":                   "entity-level macro F0.5 (competition official, zero-cand entities included)",
        "candidates_source":        str(candidates_path),
        "cache_source":             str(cache_dir) if cache_dir else "raw-tsv",
    }
    config_path = models_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved  → {config_path}")

    # 12. Save training results
    train_results_path = output_dir / "baseline_training_results.json"
    with open(train_results_path, "w") as f:
        json.dump(config, f, indent=2)

    # 13. Save threshold sweep table
    sweep_path = output_dir / "baseline_threshold_results.tsv"
    sweep_df.to_csv(sweep_path, sep="\t", index=False)
    print(f"Sweep saved   → {sweep_path}")

    # Clean up spool
    try:
        spool_path.unlink()
        spool_dir.rmdir()
    except Exception:
        pass

    print("\nTraining complete.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train baseline matcher (streaming).")
    parser.add_argument("--data-dir",        default="dataset/train",             help="Directory with train TSVs")
    parser.add_argument("--candidates",       default="output/candidate_pairs.tsv", help="M4 candidate pairs TSV")
    parser.add_argument("--output-dir",       default="output",                    help="Directory for result artifacts")
    parser.add_argument("--models-dir",       default="models",                    help="Directory for model artifacts")
    parser.add_argument("--cache-dir",        default=None,                        help="M2 Parquet cache dir (optional)")
    parser.add_argument("--chunk-size",       default=_DEFAULT_CHUNK,    type=int, help="Candidate rows per chunk (default 50000)")
    parser.add_argument("--max-train-pairs",  default=_DEFAULT_MAX_TRAIN_PAIRS, type=int,
                        help=f"Max training pair rows before neg sampling (default {_DEFAULT_MAX_TRAIN_PAIRS:,})")
    args = parser.parse_args()

    train(
        data_dir=Path(args.data_dir),
        candidates_path=Path(args.candidates),
        output_dir=Path(args.output_dir),
        models_dir=Path(args.models_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        chunk_size=args.chunk_size,
        max_train_pairs=args.max_train_pairs,
    )
