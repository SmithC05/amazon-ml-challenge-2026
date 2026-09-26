"""
tools/run_m3_10k.py
===================
Phase 2 controlled 10K training runner for the Amazon ML Challenge 2026.

Verifies all inputs exist, runs the M3 train() function on the 10K dataset,
then verifies all artifacts were produced and performs a model sanity check.

Usage (Colab / local):
    python tools/run_m3_10k.py \
        --data-dir     /content/m3_10k/data \
        --cache-dir    /content/m3_10k/cache \
        --candidates   /content/union_10k/candidate_pairs_10k.tsv \
        --output-dir   /content/m3_10k/output \
        --models-dir   /content/m3_10k/models

All directories are created automatically if they do not exist.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_NAMES, build_feature_matrix  # noqa: E402
from train import train  # noqa: E402


def _check_inputs(
    data_dir: Path,
    cache_dir: Path,
    candidates_path: Path,
) -> None:
    """Verify all required input files exist before training."""
    required = {
        "candidates": candidates_path,
        "train_source1.parquet": cache_dir / "train_source1.parquet",
        "train_source2.parquet": cache_dir / "train_source2.parquet",
        "train_source3.parquet": cache_dir / "train_source3.parquet",
        "train_ground_truth.tsv": data_dir / "train_ground_truth.tsv",
    }
    all_ok = True
    for label, path in required.items():
        ok = path.exists()
        print(f"  {'✅' if ok else '❌'} {label}: {path}")
        if not ok:
            all_ok = False
    if not all_ok:
        raise FileNotFoundError(
            "One or more required input files are missing (see above)."
        )


def _verify_artifacts(models_dir: Path, output_dir: Path) -> bool:
    """Check all expected training artifacts exist."""
    artifacts = {
        "matcher.pkl":                      models_dir / "matcher.pkl",
        "model_config.json":               models_dir / "model_config.json",
        "baseline_training_results.json":  output_dir / "baseline_training_results.json",
        "baseline_threshold_results.tsv":  output_dir / "baseline_threshold_results.tsv",
    }
    all_ok = True
    for name, path in artifacts.items():
        ok = path.exists()
        print(f"  {'✅' if ok else '❌'} {name}")
        if not ok:
            all_ok = False
    return all_ok


def _sanity_check_model(models_dir: Path) -> None:
    """Load the saved model and verify it accepts 16-feature input."""
    model_path = models_dir / "matcher.pkl"
    config_path = models_dir / "model_config.json"

    with open(model_path, "rb") as f:
        scaler, model = pickle.load(f)

    with open(config_path) as f:
        config = json.load(f)

    saved_features = config["feature_names"]
    if saved_features != FEATURE_NAMES:
        raise ValueError(
            f"Feature mismatch!\n  Config : {saved_features}\n  Current: {FEATURE_NAMES}"
        )

    n_features = model.coef_.shape[1]
    if n_features != 16:
        raise ValueError(f"Model has {n_features} features, expected 16")

    # Smoke-test predict_proba with a dummy row
    dummy = np.zeros((1, 16), dtype=float)
    dummy_sc = scaler.transform(dummy)
    proba = model.predict_proba(dummy_sc)
    assert proba.shape == (1, 2), f"Unexpected proba shape: {proba.shape}"

    print(f"  ✅ scaler loads")
    print(f"  ✅ model loads  — {n_features} input features")
    print(f"  ✅ predict_proba() works  → dummy proba={proba[0, 1]:.4f}")
    print(f"  ✅ feature_names match FEATURE_NAMES")
    print(f"  threshold in config : {config['threshold']}")
    print(f"  val F0.5 in config  : {config['f0_5']}")
    print(f"  val precision       : {config['precision']}")
    print(f"  val recall          : {config['recall']}")
    print(f"  zero-cand val S1    : {config.get('val_s1_zero_cand', 'n/a')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 controlled M3 10K training runner."
    )
    parser.add_argument("--data-dir",   required=True, help="Dir with train_ground_truth.tsv")
    parser.add_argument("--cache-dir",  required=True, help="Dir with M2 Parquet cache")
    parser.add_argument("--candidates", required=True, help="Path to candidate_pairs_10k.tsv")
    parser.add_argument("--output-dir", required=True, help="Dir for training result artifacts")
    parser.add_argument("--models-dir", required=True, help="Dir for model artifacts")
    args = parser.parse_args()

    data_dir      = Path(args.data_dir)
    cache_dir     = Path(args.cache_dir)
    candidates    = Path(args.candidates)
    output_dir    = Path(args.output_dir)
    models_dir    = Path(args.models_dir)

    print("=" * 65)
    print("PHASE 2 — M3 10K TRAINING RUNNER")
    print("=" * 65)

    # ── Step 1: verify inputs ─────────────────────────────────────────────────
    print("\n[1] Input verification")
    _check_inputs(data_dir, cache_dir, candidates)

    # ── Step 2: run training ──────────────────────────────────────────────────
    print("\n[2] Running train()")
    print("-" * 65)
    train(
        data_dir=data_dir,
        candidates_path=candidates,
        output_dir=output_dir,
        models_dir=models_dir,
        cache_dir=cache_dir,
    )
    print("-" * 65)

    # ── Step 3: verify artifacts ──────────────────────────────────────────────
    print("\n[3] Artifact verification")
    ok = _verify_artifacts(models_dir, output_dir)
    if not ok:
        print("❌ Some artifacts missing — training may have failed.")
        sys.exit(1)

    # ── Step 4: model sanity check ────────────────────────────────────────────
    print("\n[4] Model sanity check")
    _sanity_check_model(models_dir)

    print("\n" + "=" * 65)
    print("✅ PHASE 2 COMPLETE — all checks passed")
    print("=" * 65)


if __name__ == "__main__":
    main()
