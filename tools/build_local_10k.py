"""
tools/build_local_10k.py
========================
Build a self-contained local 10K M3 training environment from the raw
train TSV files (dataset/train/).

Steps:
  1. Read the first 10,000 S1 entities from train_source1.tsv
  2. Apply M2 normalization and write parquet cache (train_source1.parquet)
  3. Read ALL of S2 and S3, normalize, write parquet caches
  4. Extract GT rows for the 10K S1 entities
  5. Build a mini candidate file:
       - For each S1, include its true matches as candidates
       - Plus sample up to 9 random S2/S3 non-match candidates per S1
       (This gives a realistic mix of positives and negatives for the
        classifier; it does NOT simulate real M4 blocking, but is
        sufficient for Phase 2 runtime verification.)

Usage:
    python tools/build_local_10k.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from preprocess import preprocess_dataframe  # noqa: E402  (M2)

SEED = 42
N_S1 = 10_000
N_NEG_PER_S1 = 9  # negative candidates per S1 entity

DATA_DIR   = REPO / "dataset" / "train"
OUT_DIR    = REPO / "output" / "local_10k"
CACHE_DIR  = OUT_DIR / "cache"
DATA_OUT   = OUT_DIR / "data"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)


def main() -> None:
    print("=" * 65)
    print("LOCAL 10K TRAINING ENVIRONMENT BUILDER")
    print("=" * 65)

    # ── 1. Load & normalize S1 (first 10K rows) ──────────────────────────────
    print(f"\n[1] Loading first {N_S1:,} S1 rows ...")
    s1_raw = pd.read_csv(DATA_DIR / "train_source1.tsv", sep="\t", dtype=str,
                         nrows=N_S1)
    print(f"    Read {len(s1_raw):,} S1 rows")
    s1 = preprocess_dataframe(s1_raw)
    s1.to_parquet(CACHE_DIR / "train_source1.parquet", index=False)
    print(f"    ✅ train_source1.parquet written ({len(s1):,} rows)")

    # ── 2. Load & normalize S2 ────────────────────────────────────────────────
    print("\n[2] Loading S2 ...")
    s2_raw = pd.read_csv(DATA_DIR / "train_source2.tsv", sep="\t", dtype=str)
    print(f"    Read {len(s2_raw):,} S2 rows")
    s2 = preprocess_dataframe(s2_raw)
    s2.to_parquet(CACHE_DIR / "train_source2.parquet", index=False)
    print(f"    ✅ train_source2.parquet written ({len(s2):,} rows)")

    # ── 3. Load & normalize S3 ────────────────────────────────────────────────
    print("\n[3] Loading S3 ...")
    s3_raw = pd.read_csv(DATA_DIR / "train_source3.tsv", sep="\t", dtype=str)
    print(f"    Read {len(s3_raw):,} S3 rows")
    s3 = preprocess_dataframe(s3_raw)
    s3.to_parquet(CACHE_DIR / "train_source3.parquet", index=False)
    print(f"    ✅ train_source3.parquet written ({len(s3):,} rows)")

    # ── 4. Extract GT for the 10K S1 entities ─────────────────────────────────
    print("\n[4] Building mini ground truth ...")
    s1_id_set = set(s1["entity_id"])
    gt_full = pd.read_csv(DATA_DIR / "train_ground_truth.tsv", sep="\t",
                          dtype=str)
    gt_mini = gt_full[gt_full["source1_entity_id"].isin(s1_id_set)].copy()
    print(f"    GT rows for 10K S1: {len(gt_mini):,}")
    gt_mini.to_csv(DATA_OUT / "train_ground_truth.tsv", sep="\t", index=False)
    print(f"    ✅ train_ground_truth.tsv written")

    # ── 5. Build mini candidate file ──────────────────────────────────────────
    print("\n[5] Building mini candidate file ...")

    s2_ids = s2["entity_id"].tolist()
    s3_ids = s3["entity_id"].tolist()

    # Truth map for the 10K
    truth_map: dict[str, set[str]] = {}
    for _, row in gt_mini.iterrows():
        val = row["matched_entity_ids"]
        if pd.isna(val) or str(val).strip() == "":
            truth_map[row["source1_entity_id"]] = set()
        else:
            truth_map[row["source1_entity_id"]] = {
                x.strip() for x in str(val).split(",") if x.strip()
            }

    # For S1 entities not in GT, truth is empty
    for sid in s1_id_set:
        if sid not in truth_map:
            truth_map[sid] = set()

    rows = []
    all_cand_ids = s2_ids + s3_ids

    for sid in s1["entity_id"]:
        true_matches = list(truth_map.get(sid, set()))
        # sample negatives that are not true matches
        neg_pool = [c for c in all_cand_ids if c not in truth_map.get(sid, set())]
        neg_sample = random.sample(neg_pool, min(N_NEG_PER_S1, len(neg_pool)))
        cands = list(dict.fromkeys(true_matches + neg_sample))  # preserve order, dedup
        rows.append({
            "source1_entity_id":  sid,
            "candidate_entity_ids": ",".join(cands),
        })

    cand_df = pd.DataFrame(rows)
    cand_path = OUT_DIR / "candidate_pairs_10k.tsv"
    cand_df.to_csv(cand_path, sep="\t", index=False)
    print(f"    ✅ candidate_pairs_10k.tsv written ({len(cand_df):,} rows)")

    # Stats
    n_with_cands = (cand_df["candidate_entity_ids"] != "").sum()
    total_pair_ids = sum(
        len(x.split(",")) if x else 0
        for x in cand_df["candidate_entity_ids"]
    )
    print(f"    S1 with candidates: {n_with_cands:,}")
    print(f"    Total candidate IDs: {total_pair_ids:,}")

    print("\n" + "=" * 65)
    print("✅ LOCAL 10K ENVIRONMENT READY")
    print(f"   data    : {DATA_OUT}")
    print(f"   cache   : {CACHE_DIR}")
    print(f"   cands   : {cand_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
