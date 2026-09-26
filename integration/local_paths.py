"""
integration/local_paths.py
==========================
PART 1 — Local Integration Path Configuration
LOCAL ONLY — not a production source file.
This file is .gitignore'd (integration/ is local-only).

All other integration scripts import from here.
Do NOT hardcode these paths elsewhere.
"""

from pathlib import Path

# ── Repository root ───────────────────────────────────────────────────────────
REPO_ROOT = Path(r"E:\Amazon\amazon-ml-challenge-2026")

# ── Raw dataset locations ─────────────────────────────────────────────────────
TRAIN_DATA_DIR = REPO_ROOT / "dataset" / "train"
TEST_DATA_DIR  = REPO_ROOT / "dataset" / "test"

# ── M2 Parquet cache (canonical) ─────────────────────────────────────────────
CACHE_DIR = REPO_ROOT / "dataset" / "processed" / "m2_cache"

# ── Ground truth ──────────────────────────────────────────────────────────────
TRAIN_GT_PATH = TRAIN_DATA_DIR / "train_ground_truth.tsv"

# ── Pipeline artifacts ────────────────────────────────────────────────────────
OUTPUT_DIR     = REPO_ROOT / "output"
MODEL_DIR      = REPO_ROOT / "models"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_PATH  = OUTPUT_DIR / "matching_results.tsv"

# ── Source file shortcuts ─────────────────────────────────────────────────────
TEST_S1_PATH = TEST_DATA_DIR / "test_source1.tsv"
TEST_S2_PATH = TEST_DATA_DIR / "test_source2.tsv"
TEST_S3_PATH = TEST_DATA_DIR / "test_source3.tsv"

# ── Expected row counts (M2 verified 2026-09-26) ──────────────────────────────
EXPECTED_ROWS = {
    "train_source1": 2_206_821,
    "train_source2": 5_034_616,
    "train_source3": 5_285_603,
    "test_source1":  1_732_544,
    "test_source2":  4_887_273,
    "test_source3":  5_082_316,
}

# ── M2 cache file paths ───────────────────────────────────────────────────────
CACHE_FILES = {
    k: CACHE_DIR / f"{k}.parquet" for k in EXPECTED_ROWS
}

# ── Required headers ──────────────────────────────────────────────────────────
MATCHING_HEADER  = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]

# ── Required M2 columns (from preprocess_dataframe) ──────────────────────────
M2_REQUIRED_COLUMNS = {
    "entity_id", "business_name", "business_address", "country",
    "name_norm", "address_norm",
    "name_tokens", "address_tokens",
    "name_token_count", "address_token_count",
    "name_length", "address_length",
    "name_digits", "address_digits",
}

if __name__ == "__main__":
    print("Integration Path Configuration")
    print(f"  REPO_ROOT      : {REPO_ROOT}")
    print(f"  TRAIN_DATA_DIR : {TRAIN_DATA_DIR}")
    print(f"  TEST_DATA_DIR  : {TEST_DATA_DIR}")
    print(f"  CACHE_DIR      : {CACHE_DIR}")
    print(f"  TRAIN_GT_PATH  : {TRAIN_GT_PATH}")
    print(f"  CANDIDATE_PATH : {CANDIDATE_PATH}")
    print(f"  MATCHING_PATH  : {MATCHING_PATH}")
    print(f"  MODEL_DIR      : {MODEL_DIR}")
    for k, p in CACHE_FILES.items():
        exists = "✓" if p.exists() else "✗"
        print(f"  cache {k}: {exists}")
