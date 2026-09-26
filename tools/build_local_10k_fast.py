"""
tools/build_local_10k_fast.py
=============================
Build a self-contained local 10K M3 training environment from the raw
train TSV files (dataset/train/).

FAST VERSION: Only normalizes S2/S3 entities that actually appear in the
candidate file, rather than the entire S2/S3 datasets.

Steps:
  1. Read first 10,000 S1 entities from train_source1.tsv
  2. Build mini candidate file:
       - For each S1, include its true matches
       - Plus sample up to 9 random negatives from the first 50K S2/S3 rows
  3. Collect all candidate entity IDs (S2+S3) from the candidate file
  4. Normalize ONLY those S2/S3 entities needed for feature extraction
  5. Write filtered parquet caches for S1, S2, S3
  6. Extract GT rows for the 10K S1 entities
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
N_NEG_PER_S1 = 9   # negative candidates per S1 entity (S2 + S3 each)
NEG_POOL_SIZE = 50_000  # rows from S2/S3 to use as negative pool

DATA_DIR   = REPO / "dataset" / "train"
OUT_DIR    = REPO / "output" / "local_10k"
CACHE_DIR  = OUT_DIR / "cache"
DATA_OUT   = OUT_DIR / "data"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)


def build_truth_map(gt_mini: pd.DataFrame) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    for _, row in gt_mini.iterrows():
        val = row["matched_entity_ids"]
        if pd.isna(val) or str(val).strip() == "":
            truth[row["source1_entity_id"]] = set()
        else:
            truth[row["source1_entity_id"]] = {
                x.strip() for x in str(val).split(",") if x.strip()
            }
    return truth


def main() -> None:
    print("=" * 65)
    print("LOCAL 10K TRAINING ENVIRONMENT BUILDER (FAST)")
    print("=" * 65)

    # ── 1. Load S1 (first 10K rows, raw) ─────────────────────────────────────
    print(f"\n[1] Loading first {N_S1:,} S1 rows ...")
    s1_raw = pd.read_csv(DATA_DIR / "train_source1.tsv", sep="\t", dtype=str,
                         nrows=N_S1)
    print(f"    S1 raw: {len(s1_raw):,} rows")
    s1_id_set = set(s1_raw["entity_id"])

    # ── 2. Load GT for these 10K S1 entities ─────────────────────────────────
    print("\n[2] Loading ground truth ...")
    gt_chunks = []
    for chunk in pd.read_csv(DATA_DIR / "train_ground_truth.tsv", sep="\t",
                             dtype=str, chunksize=100_000):
        sub = chunk[chunk["source1_entity_id"].isin(s1_id_set)]
        if len(sub):
            gt_chunks.append(sub)
    gt_mini = pd.concat(gt_chunks, ignore_index=True) if gt_chunks else pd.DataFrame(
        columns=["source1_entity_id", "matched_entity_ids"])
    print(f"    GT rows for {N_S1:,} S1: {len(gt_mini):,}")

    truth_map = build_truth_map(gt_mini)
    # Ensure every S1 entity has an entry
    for sid in s1_id_set:
        if sid not in truth_map:
            truth_map[sid] = set()

    # ── 3. Collect all true-match candidate IDs ───────────────────────────────
    print("\n[3] Collecting true-match candidate IDs ...")
    true_match_ids: set[str] = set()
    for matches in truth_map.values():
        true_match_ids.update(matches)
    s2_true = {c for c in true_match_ids if c.startswith("S2-")}
    s3_true = {c for c in true_match_ids if c.startswith("S3-")}
    print(f"    True-match S2 IDs: {len(s2_true):,}")
    print(f"    True-match S3 IDs: {len(s3_true):,}")

    # ── 4. Load a negative pool from S2/S3 (first 50K rows each) ────────────
    print(f"\n[4] Loading negative pool from S2/S3 (first {NEG_POOL_SIZE:,} rows each) ...")
    s2_pool_raw = pd.read_csv(DATA_DIR / "train_source2.tsv", sep="\t", dtype=str,
                               nrows=NEG_POOL_SIZE)
    s3_pool_raw = pd.read_csv(DATA_DIR / "train_source3.tsv", sep="\t", dtype=str,
                               nrows=NEG_POOL_SIZE)
    print(f"    S2 pool: {len(s2_pool_raw):,} rows")
    print(f"    S3 pool: {len(s3_pool_raw):,} rows")

    s2_pool_ids = s2_pool_raw["entity_id"].tolist()
    s3_pool_ids = s3_pool_raw["entity_id"].tolist()

    # ── 5. Build candidate file ───────────────────────────────────────────────
    print("\n[5] Building candidate file ...")
    rows = []
    for sid in s1_raw["entity_id"]:
        tm = list(truth_map.get(sid, set()))
        # Negatives: sample from pool, excluding true matches
        neg_s2 = [c for c in s2_pool_ids if c not in truth_map.get(sid, set())]
        neg_s3 = [c for c in s3_pool_ids if c not in truth_map.get(sid, set())]
        neg_s2_sample = random.sample(neg_s2, min(N_NEG_PER_S1, len(neg_s2)))
        neg_s3_sample = random.sample(neg_s3, min(N_NEG_PER_S1, len(neg_s3)))
        cands = list(dict.fromkeys(tm + neg_s2_sample + neg_s3_sample))
        rows.append({
            "source1_entity_id":   sid,
            "candidate_entity_ids": ",".join(cands),
        })

    cand_df = pd.DataFrame(rows)
    cand_path = OUT_DIR / "candidate_pairs_10k.tsv"
    cand_df.to_csv(cand_path, sep="\t", index=False)
    total_cands = sum(len(r.split(",")) if r else 0
                      for r in cand_df["candidate_entity_ids"])
    print(f"    Candidate file: {len(cand_df):,} S1 rows, {total_cands:,} total candidate IDs")

    # ── 6. Collect ALL needed S2/S3 IDs ──────────────────────────────────────
    print("\n[6] Collecting all candidate entity IDs needed for feature extraction ...")
    needed_s2: set[str] = set()
    needed_s3: set[str] = set()
    for val in cand_df["candidate_entity_ids"]:
        for cid in (val or "").split(","):
            cid = cid.strip()
            if cid.startswith("S2-"):
                needed_s2.add(cid)
            elif cid.startswith("S3-"):
                needed_s3.add(cid)
    print(f"    Needed S2 entities: {len(needed_s2):,}")
    print(f"    Needed S3 entities: {len(needed_s3):,}")

    # ── 7. Filter S2/S3 to only needed entities and normalize ─────────────────
    print("\n[7] Filtering and normalizing S2 ...")
    s2_needed_raw = s2_pool_raw[s2_pool_raw["entity_id"].isin(needed_s2)].copy()
    # Also include any true-match S2 not in pool (load from full file)
    missing_s2 = s2_true - set(s2_pool_raw["entity_id"])
    if missing_s2:
        print(f"    Loading {len(missing_s2):,} true-match S2 not in pool ...")
        # Scan full S2 for missing true matches
        for chunk in pd.read_csv(DATA_DIR / "train_source2.tsv", sep="\t",
                                  dtype=str, chunksize=100_000):
            found = chunk[chunk["entity_id"].isin(missing_s2)]
            if len(found):
                s2_needed_raw = pd.concat([s2_needed_raw, found], ignore_index=True)
                missing_s2 -= set(found["entity_id"])
            if not missing_s2:
                break
    s2 = preprocess_dataframe(s2_needed_raw)
    s2.to_parquet(CACHE_DIR / "train_source2.parquet", index=False)
    print(f"    train_source2.parquet: {len(s2):,} rows")

    print("\n[8] Filtering and normalizing S3 ...")
    s3_needed_raw = s3_pool_raw[s3_pool_raw["entity_id"].isin(needed_s3)].copy()
    missing_s3 = s3_true - set(s3_pool_raw["entity_id"])
    if missing_s3:
        print(f"    Loading {len(missing_s3):,} true-match S3 not in pool ...")
        for chunk in pd.read_csv(DATA_DIR / "train_source3.tsv", sep="\t",
                                  dtype=str, chunksize=100_000):
            found = chunk[chunk["entity_id"].isin(missing_s3)]
            if len(found):
                s3_needed_raw = pd.concat([s3_needed_raw, found], ignore_index=True)
                missing_s3 -= set(found["entity_id"])
            if not missing_s3:
                break
    s3 = preprocess_dataframe(s3_needed_raw)
    s3.to_parquet(CACHE_DIR / "train_source3.parquet", index=False)
    print(f"    train_source3.parquet: {len(s3):,} rows")

    # ── 9. Normalize S1 and write parquet ────────────────────────────────────
    print("\n[9] Normalizing S1 ...")
    s1 = preprocess_dataframe(s1_raw)
    s1.to_parquet(CACHE_DIR / "train_source1.parquet", index=False)
    print(f"    train_source1.parquet: {len(s1):,} rows")

    # ── 10. Write mini GT ─────────────────────────────────────────────────────
    print("\n[10] Writing mini ground truth ...")
    gt_mini.to_csv(DATA_OUT / "train_ground_truth.tsv", sep="\t", index=False)
    print(f"     train_ground_truth.tsv: {len(gt_mini):,} rows")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("LOCAL 10K ENVIRONMENT READY")
    print(f"  S1 : {len(s1):,}  (10K subset)")
    print(f"  S2 : {len(s2):,}  (filtered to candidates only)")
    print(f"  S3 : {len(s3):,}  (filtered to candidates only)")
    print(f"  GT : {len(gt_mini):,} rows for 10K S1")
    print(f"  Candidates: {len(cand_df):,} rows, {total_cands:,} pairs")
    print(f"\n  data    : {DATA_OUT}")
    print(f"  cache   : {CACHE_DIR}")
    print(f"  cands   : {cand_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
