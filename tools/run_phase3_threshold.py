"""
tools/run_phase3_threshold.py
=============================
Phase 3 — Final threshold selection for M3 baseline matcher.

Loads the existing Phase 2 model and validation data, runs a fine-grained
threshold sweep (0.50 → 0.95, step 0.01 = 46 thresholds), selects the
threshold maximising entity-level macro F0.5, and persists all evidence.

Usage:
    python tools/run_phase3_threshold.py \
        --data-dir     output/local_10k/data \
        --cache-dir    output/local_10k/cache \
        --candidates   output/local_10k/candidate_pairs_10k.tsv \
        --models-dir   output/local_10k/models \
        --output-dir   output/local_10k/output

No new training is performed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_NAMES, build_feature_matrix  # noqa: E402
from train import (  # noqa: E402
    build_pair_rows,
    build_truth_map,
    entity_split,
    entity_f05,
    load_sources,
    RANDOM_STATE,
)


# ── Fine-grained threshold grid ───────────────────────────────────────────────
THRESHOLDS = [round(t, 2) for t in np.arange(0.50, 0.96, 0.01)]


def threshold_sweep_fine(
    val_df: pd.DataFrame,
    proba: np.ndarray,
    truth_map: dict[str, set[str]],
    val_s1_ids: list[str],
    thresholds: list[float],
) -> pd.DataFrame:
    """
    Entity-level macro F0.5 sweep over the given thresholds.

    val_s1_ids is the COMPLETE validation population including
    zero-candidate entities (they start with pred_map[sid] = set()).
    """
    results = []
    for t in thresholds:
        pred_positive = proba >= t
        # Initialise ALL val entities to empty prediction
        pred_map: dict[str, set[str]] = {sid: set() for sid in val_s1_ids}

        tmp = val_df.copy()
        tmp["_pos"] = pred_positive
        for sid, grp in tmp.groupby("source1_entity_id"):
            pred_map[str(sid)] = set(grp.loc[grp["_pos"], "cand_entity_id"])

        all_prec, all_rec, all_f05 = [], [], []
        for sid in val_s1_ids:
            tr = truth_map.get(sid, set())
            pr = pred_map.get(sid, set())
            tp = len(tr & pr)
            p  = tp / len(pr) if pr else 0.0
            r  = tp / len(tr) if tr else 0.0
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 threshold selection for M3 baseline."
    )
    parser.add_argument("--data-dir",   required=True)
    parser.add_argument("--cache-dir",  required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    cache_dir   = Path(args.cache_dir)
    cands_path  = Path(args.candidates)
    models_dir  = Path(args.models_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path  = models_dir / "matcher.pkl"
    config_path = models_dir / "model_config.json"

    # ── 1. Verify Phase 2 artifacts exist ─────────────────────────────────────
    print("=" * 65)
    print("PHASE 3 — M3 THRESHOLD SELECTION")
    print("=" * 65)

    for p, label in [
        (model_path,  "matcher.pkl"),
        (config_path, "model_config.json"),
        (cands_path,  "candidates"),
    ]:
        if not p.exists():
            print(f"  MISSING: {label} ({p})")
            sys.exit(1)
        print(f"  OK  {label}")

    # ── 2. Load Phase 2 model ─────────────────────────────────────────────────
    print("\n[1] Loading Phase 2 model ...")
    with open(model_path, "rb") as f:
        scaler, model = pickle.load(f)

    with open(config_path) as f:
        p2_config = json.load(f)

    saved_features = p2_config["feature_names"]
    if saved_features != FEATURE_NAMES:
        print("FEATURE MISMATCH!")
        print(f"  saved : {saved_features}")
        print(f"  current: {FEATURE_NAMES}")
        sys.exit(1)
    print(f"  Model type  : {p2_config['model_type']}")
    print(f"  Feature count: {model.coef_.shape[1]}")
    print(f"  Features match current FEATURE_NAMES: YES")

    # ── 3. Load sources + build validation pairs ───────────────────────────────
    print("\n[2] Loading sources ...")
    s1, s2, s3, gt = load_sources(data_dir, cache_dir=cache_dir)
    truth_map = build_truth_map(gt)

    print("\n[3] Loading candidates and building pairs ...")
    candidates = pd.read_csv(cands_path, sep="\t")
    pair_df    = build_pair_rows(candidates, s1, s2, s3, truth_map)

    n_pos = int(pair_df["label"].sum())
    n_neg = int((pair_df["label"] == 0).sum())
    print(f"  Total S1       : {len(s1):,}")
    print(f"  Labeled pairs  : {len(pair_df):,}  (pos={n_pos:,}  neg={n_neg:,})")

    # ── 4. Entity split using COMPLETE S1 population ──────────────────────────
    print("\n[4] Reconstructing entity split (random_state=42) ...")
    all_s1_ids = s1["entity_id"].tolist()
    train_df, val_df, train_s1_list, val_s1_list = entity_split(
        pair_df,
        all_s1_ids=all_s1_ids,
        random_state=RANDOM_STATE,
    )
    train_s1 = set(train_s1_list)
    val_s1   = set(val_s1_list)

    val_s1_with_pairs  = set(val_df["source1_entity_id"].unique())
    n_zero_cand_val    = len(val_s1 - val_s1_with_pairs)
    overlap            = len(train_s1 & val_s1)

    n_val_pos = int(val_df["label"].sum())
    n_val_neg = int((val_df["label"] == 0).sum())

    print(f"  Train S1       : {len(train_s1):,}")
    print(f"  Val   S1       : {len(val_s1):,}")
    print(f"  Zero-cand val  : {n_zero_cand_val:,}")
    print(f"  Train/val overlap: {overlap}")
    assert overlap == 0, "Entity split leakage!"

    print(f"\n  Val pairs      : {len(val_df):,}  (pos={n_val_pos:,}  neg={n_val_neg:,})")
    print(f"  Val proba rows : {len(val_df):,}")

    # ── 5. Reconstruct validation probabilities ────────────────────────────────
    print("\n[5] Computing validation probabilities ...")
    X_val    = build_feature_matrix(val_df).values
    X_val_sc = scaler.transform(X_val)
    val_proba = model.predict_proba(X_val_sc)[:, 1]
    print(f"  proba shape    : {val_proba.shape}")
    print(f"  proba min/max  : {val_proba.min():.4f} / {val_proba.max():.4f}")

    # ── 6. Fine-grained threshold sweep ───────────────────────────────────────
    print(f"\n[6] Running fine-grained threshold sweep ({len(THRESHOLDS)} thresholds) ...")
    sweep_df = threshold_sweep_fine(val_df, val_proba, truth_map, val_s1_list, THRESHOLDS)

    print("\nThreshold sweep (step 0.01):")
    print(sweep_df.to_string(index=False))

    # ── 7. Select best threshold ───────────────────────────────────────────────
    best_idx       = sweep_df["f0_5"].idxmax()
    best_row       = sweep_df.loc[best_idx]
    best_threshold = float(best_row["threshold"])
    best_f05       = float(best_row["f0_5"])
    best_prec      = float(best_row["precision"])
    best_rec       = float(best_row["recall"])

    # Stability window
    row_minus = sweep_df[sweep_df["threshold"] == round(best_threshold - 0.01, 2)]
    row_plus  = sweep_df[sweep_df["threshold"] == round(best_threshold + 0.01, 2)]
    f05_minus = float(row_minus["f0_5"].values[0]) if len(row_minus) else float("nan")
    f05_plus  = float(row_plus["f0_5"].values[0])  if len(row_plus)  else float("nan")

    print(f"\nSelected threshold : {best_threshold:.2f}")
    print(f"Validation Precision: {best_prec:.6f}")
    print(f"Validation Recall  : {best_rec:.6f}")
    print(f"Validation F0.5    : {best_f05:.6f}")
    print(f"\nStability:")
    print(f"  F0.5 at t={best_threshold-0.01:.2f} : {f05_minus:.6f}")
    print(f"  F0.5 at t={best_threshold:.2f}     : {best_f05:.6f}  ← selected")
    print(f"  F0.5 at t={best_threshold+0.01:.2f} : {f05_plus:.6f}")

    # ── 8. Persist fine-grained sweep ─────────────────────────────────────────
    sweep_path = output_dir / "phase3_threshold_sweep.tsv"
    sweep_df.to_csv(sweep_path, sep="\t", index=False)
    print(f"\nFine sweep saved → {sweep_path}")

    # Also overwrite baseline file for consistency
    baseline_path = output_dir / "baseline_threshold_results.tsv"
    sweep_df.to_csv(baseline_path, sep="\t", index=False)
    print(f"Baseline sweep saved → {baseline_path}")

    # ── 9. Update model_config.json with Phase 3 values ───────────────────────
    config = {
        "model_type":            p2_config["model_type"],
        "feature_names":         FEATURE_NAMES,
        "feature_count":         len(FEATURE_NAMES),
        "threshold":             best_threshold,
        "random_seed":           RANDOM_STATE,
        "train_s1_count":        len(train_s1),
        "validation_s1_count":   len(val_s1),
        "val_s1_zero_cand":      n_zero_cand_val,
        "train_pair_count":      int(len(train_df)),
        "validation_pair_count": int(len(val_df)),
        "val_positive_pairs":    n_val_pos,
        "val_negative_pairs":    n_val_neg,
        "precision":             round(best_prec,  6),
        "recall":                round(best_rec,   6),
        "f0_5":                  round(best_f05,   6),
        "f0_5_at_t_minus_0_01":  round(f05_minus,  6) if not np.isnan(f05_minus) else None,
        "f0_5_at_t_plus_0_01":   round(f05_plus,   6) if not np.isnan(f05_plus)  else None,
        "baseline_version":      "1.0-phase3",
        "metric":                "entity-level macro F0.5 (competition official, zero-cand entities included)",
        "sweep_step":            0.01,
        "sweep_range":           f"0.50-0.95 step 0.01 ({len(THRESHOLDS)} thresholds)",
        "candidates_source":     str(cands_path),
    }

    # Write to local Phase 3 models dir
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config updated → {config_path}")

    # Also update the repo-level models/model_config.json
    repo_config_path = ROOT / "models" / "model_config.json"
    repo_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(repo_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Repo config updated → {repo_config_path}")

    # Also update output/baseline_training_results.json
    results_path = output_dir / "baseline_training_results.json"
    with open(results_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Training results updated → {results_path}")

    # ── 10. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 3 COMPLETE")
    print("=" * 65)
    print(f"  Selected threshold   : {best_threshold}")
    print(f"  Validation precision : {best_prec:.6f}")
    print(f"  Validation recall    : {best_rec:.6f}")
    print(f"  Validation F0.5      : {best_f05:.6f}")
    print(f"  Zero-cand val S1     : {n_zero_cand_val}")
    print(f"  Train/val overlap    : {overlap}")
    print(f"\nArtifacts:")
    for p in [
        models_dir / "matcher.pkl",
        models_dir / "model_config.json",
        output_dir / "baseline_training_results.json",
        output_dir / "baseline_threshold_results.tsv",
        output_dir / "phase3_threshold_sweep.tsv",
        ROOT / "models" / "model_config.json",
    ]:
        print(f"  {'OK' if p.exists() else 'MISSING':7s} {p}")


if __name__ == "__main__":
    main()
