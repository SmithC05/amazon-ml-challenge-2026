# -*- coding: utf-8 -*-
"""
tools/phase3_effectiveness.py
==============================
Phase 3 -- Full Training Normalization Effectiveness

Vectorised pandas implementation (no iterrows loops).
Uses str.split()+explode() for GT parsing, merge() for joins,
and .map() for normalization -- all vectorised for large-scale data.

Usage (run from repo root):
    python tools/phase3_effectiveness.py ^
        --data-dir "D:/6ab10eb3b23ba_student_resource/student_resource/dataset/train"

Scope constraints:
  * No normalization rules added or changed.
  * No blocking, candidate generation, fuzzy matching, or ML.
  * No cache modification.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")
)

from preprocess import normalize_name, normalize_address  # canonical M2


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_tsv(path: Path, desc: str) -> pd.DataFrame:
    print(f"  Loading {desc} from {path.name} ...", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    print(f"    -> {len(df):,} rows  columns: {list(df.columns)}", flush=True)
    return df


def show_examples(df: pd.DataFrame, cols_name: list, n: int = 8) -> None:
    sample = df.head(n).reset_index(drop=True)
    for _, row in sample.iterrows():
        for c in cols_name:
            print(f"    {c:<35}: {row.get(c, '')!r}")
        print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run(data_dir: Path) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("PHASE 3 -- NORMALIZATION EFFECTIVENESS ANALYSIS")
    print(f"{sep}\n")

    # ----------------------------------------------------------------
    # 1. Load
    # ----------------------------------------------------------------
    print("STEP 1: Loading data", flush=True)
    s1 = load_tsv(data_dir / "train_source1.tsv", "Source 1")
    s2 = load_tsv(data_dir / "train_source2.tsv", "Source 2")
    s3 = load_tsv(data_dir / "train_source3.tsv", "Source 3")
    gt = load_tsv(data_dir / "train_ground_truth.tsv", "Ground Truth")
    print()

    # ----------------------------------------------------------------
    # 2. Parse ground truth -- vectorised
    # ----------------------------------------------------------------
    print("STEP 2: Parsing ground truth ...", flush=True)

    total_gt_rows = len(gt)

    # Rows with missing / empty matched_entity_ids
    no_match_mask = gt["matched_entity_ids"].isna() | (gt["matched_entity_ids"].str.strip() == "")
    rows_no_match   = int(no_match_mask.sum())
    rows_with_match = total_gt_rows - rows_no_match

    # Keep only rows with known matches
    gt_matched = gt[~no_match_mask].copy()

    # Explode: one row per matched id
    gt_exploded = (
        gt_matched
        .assign(matched_entity_id=gt_matched["matched_entity_ids"].str.split(","))
        .explode("matched_entity_id")
    )
    gt_exploded["matched_entity_id"] = gt_exploded["matched_entity_id"].str.strip()
    gt_exploded = gt_exploded[gt_exploded["matched_entity_id"] != ""].copy()

    # Source label from entity_id prefix
    gt_exploded["matched_source"] = gt_exploded["matched_entity_id"].str[:2]

    total_pairs = len(gt_exploded)
    s2_pairs    = int((gt_exploded["matched_source"] == "S2").sum())
    s3_pairs    = int((gt_exploded["matched_source"] == "S3").sum())

    print(f"  Total ground-truth rows              : {total_gt_rows:,}")
    print(f"  S1 rows WITH known matches           : {rows_with_match:,}")
    print(f"  S1 rows with missing/empty GT        : {rows_no_match:,}")
    print(f"  Total true labeled pairs             : {total_pairs:,}")
    print(f"    S1->S2 pairs                       : {s2_pairs:,}")
    print(f"    S1->S3 pairs                       : {s3_pairs:,}")
    print()

    # ----------------------------------------------------------------
    # 3. Build comparison frame -- vectorised merge
    # ----------------------------------------------------------------
    print("STEP 3: Joining pairs with source records ...", flush=True)

    # Narrow source frames for merge (only needed columns)
    s1_narrow = s1[["entity_id", "business_name", "business_address"]].rename(
        columns={"entity_id": "source1_entity_id",
                 "business_name": "s1_business_name",
                 "business_address": "s1_business_address"}
    )

    s2_narrow = s2[["entity_id", "business_name", "business_address"]].rename(
        columns={"entity_id": "matched_entity_id",
                 "business_name": "target_business_name",
                 "business_address": "target_business_address"}
    )
    s3_narrow = s3[["entity_id", "business_name", "business_address"]].rename(
        columns={"entity_id": "matched_entity_id",
                 "business_name": "target_business_name",
                 "business_address": "target_business_address"}
    )

    # Stack S2 + S3 targets
    rhs = pd.concat([s2_narrow, s3_narrow], ignore_index=True)

    # Merge GT pairs with S1 info
    pairs = gt_exploded[["source1_entity_id", "matched_entity_id", "matched_source"]].merge(
        s1_narrow, on="source1_entity_id", how="inner"
    )
    # Merge with target info
    pairs = pairs.merge(
        rhs, on="matched_entity_id", how="inner"
    )

    n = len(pairs)
    skipped = total_pairs - n
    if skipped > 0:
        print(f"  WARNING: {skipped:,} pairs skipped (entity ID not found in source files)")
    print(f"  Resolved pairs for analysis          : {n:,}")
    print()

    # ----------------------------------------------------------------
    # 4. Apply normalization -- vectorised .map()
    # ----------------------------------------------------------------
    print("STEP 4: Applying normalization ...", flush=True)
    pairs["s1_business_name_norm"]         = pairs["s1_business_name"].map(normalize_name)
    pairs["target_business_name_norm"]     = pairs["target_business_name"].map(normalize_name)
    pairs["s1_business_address_norm"]      = pairs["s1_business_address"].map(normalize_address)
    pairs["target_business_address_norm"]  = pairs["target_business_address"].map(normalize_address)
    print("  Done.", flush=True)
    print()

    # ----------------------------------------------------------------
    # 5. Name effectiveness
    # ----------------------------------------------------------------
    print("STEP 5: Computing name effectiveness ...", flush=True)

    s1_name  = pairs["s1_business_name"].fillna("")
    tgt_name = pairs["target_business_name"].fillna("")

    raw_name_eq = (
        (s1_name != "") &
        (tgt_name != "") &
        (s1_name == tgt_name)
    )
    raw_name_count = int(raw_name_eq.sum())
    raw_name_pct   = 100.0 * raw_name_count / n

    norm_name_eq   = (pairs["s1_business_name_norm"] == pairs["target_business_name_norm"])
    norm_name_count = int(norm_name_eq.sum())
    norm_name_pct   = 100.0 * norm_name_count / n
    name_improvement = norm_name_pct - raw_name_pct

    # Newly exact: raw not equal but norm is equal
    name_newly_mask  = (~raw_name_eq) & norm_name_eq
    name_newly_exact = pairs[name_newly_mask][[
        "source1_entity_id", "matched_entity_id", "matched_source",
        "s1_business_name", "target_business_name",
        "s1_business_name_norm", "target_business_name_norm",
    ]].reset_index(drop=True)
    n_name_newly = len(name_newly_exact)

    print(f"  Raw exact name matches               : {raw_name_count:,} / {n:,}  ({raw_name_pct:.4f}%)")
    print(f"  Normalized exact name matches        : {norm_name_count:,} / {n:,}  ({norm_name_pct:.4f}%)")
    print(f"  Improvement                          : +{name_improvement:.4f} pp")
    print(f"  Newly exact pairs                    : {n_name_newly:,}")
    print()

    # ----------------------------------------------------------------
    # 6. Address effectiveness
    # ----------------------------------------------------------------
    print("STEP 6: Computing address effectiveness ...", flush=True)

    s1_addr  = pairs["s1_business_address"].fillna("").str.strip()
    tgt_addr = pairs["target_business_address"].fillna("").str.strip()

    # Two missing values do NOT count as an exact match
    raw_addr_eq = (
        (s1_addr != "") &
        (tgt_addr != "") &
        (s1_addr == tgt_addr)
    )
    raw_addr_count = int(raw_addr_eq.sum())
    raw_addr_pct   = 100.0 * raw_addr_count / n

    s1_addr_norm  = pairs["s1_business_address_norm"].str.strip()
    tgt_addr_norm = pairs["target_business_address_norm"].str.strip()
    norm_addr_eq  = (
        (s1_addr_norm != "") &
        (tgt_addr_norm != "") &
        (s1_addr_norm == tgt_addr_norm)
    )
    norm_addr_count = int(norm_addr_eq.sum())
    norm_addr_pct   = 100.0 * norm_addr_count / n
    addr_improvement = norm_addr_pct - raw_addr_pct

    # Newly exact address
    addr_newly_mask  = (~raw_addr_eq) & norm_addr_eq
    addr_newly_exact = pairs[addr_newly_mask][[
        "source1_entity_id", "matched_entity_id", "matched_source",
        "s1_business_address", "target_business_address",
        "s1_business_address_norm", "target_business_address_norm",
    ]].reset_index(drop=True)
    n_addr_newly = len(addr_newly_exact)

    print(f"  Raw exact address matches            : {raw_addr_count:,} / {n:,}  ({raw_addr_pct:.4f}%)")
    print(f"  Normalized exact address matches     : {norm_addr_count:,} / {n:,}  ({norm_addr_pct:.4f}%)")
    print(f"  Improvement                          : +{addr_improvement:.4f} pp")
    print(f"  Newly exact pairs                    : {n_addr_newly:,}")
    print()

    # ----------------------------------------------------------------
    # 7. Examples
    # ----------------------------------------------------------------
    name_cols = [
        "source1_entity_id", "matched_entity_id", "matched_source",
        "s1_business_name", "target_business_name",
        "s1_business_name_norm", "target_business_name_norm",
    ]
    addr_cols = [
        "source1_entity_id", "matched_entity_id", "matched_source",
        "s1_business_address", "target_business_address",
        "s1_business_address_norm", "target_business_address_norm",
    ]

    print(sep)
    print("NAME NEWLY-EXACT EXAMPLES (raw name != target, normalized name == target)")
    print(sep)
    if n_name_newly > 0:
        show_examples(name_newly_exact, name_cols, n=8)
    else:
        print("  (none found)\n")

    print(sep)
    print("ADDRESS NEWLY-EXACT EXAMPLES (raw addr != target, normalized addr == target)")
    print(sep)
    if n_addr_newly > 0:
        show_examples(addr_newly_exact, addr_cols, n=8)
    else:
        print("  (none found)\n")

    # ----------------------------------------------------------------
    # 8. Final report
    # ----------------------------------------------------------------
    print(sep)
    print("PHASE 3 NORMALIZATION EFFECTIVENESS REPORT")
    print(sep)
    print()
    print(f"True labeled pairs: {n:,}")
    print()
    print("Name:")
    print(f"  Raw exact matches        = {raw_name_count:,}")
    print(f"  Raw exact match rate     = {raw_name_pct:.4f}%")
    print(f"  Normalized exact matches = {norm_name_count:,}")
    print(f"  Normalized exact rate    = {norm_name_pct:.4f}%")
    print(f"  Improvement              = +{name_improvement:.4f} percentage points")
    print(f"  Newly exact pairs        = {n_name_newly:,}")
    print()
    print("Address:")
    print(f"  Raw exact matches        = {raw_addr_count:,}")
    print(f"  Raw exact match rate     = {raw_addr_pct:.4f}%")
    print(f"  Normalized exact matches = {norm_addr_count:,}")
    print(f"  Normalized exact rate    = {norm_addr_pct:.4f}%")
    print(f"  Improvement              = +{addr_improvement:.4f} percentage points")
    print(f"  Newly exact pairs        = {n_addr_newly:,}")
    print()

    print("Summary table:")
    print(f"  {'Metric':<40} {'Count':>10}  {'Percentage':>12}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*12}")
    print(f"  {'Raw exact name':<40} {raw_name_count:>10,}  {raw_name_pct:>11.4f}%")
    print(f"  {'Normalized exact name':<40} {norm_name_count:>10,}  {norm_name_pct:>11.4f}%")
    print(f"  {'Name newly exact':<40} {n_name_newly:>10,}  {100.0*n_name_newly/n:>11.4f}%")
    print(f"  {'Raw exact address':<40} {raw_addr_count:>10,}  {raw_addr_pct:>11.4f}%")
    print(f"  {'Normalized exact address':<40} {norm_addr_count:>10,}  {norm_addr_pct:>11.4f}%")
    print(f"  {'Address newly exact':<40} {n_addr_newly:>10,}  {100.0*n_addr_newly/n:>11.4f}%")
    print()

    print("PHASE 3 STATUS: COMPLETE")
    print()

    # Return results dict for callers (e.g. report generation)
    return {
        "total_gt_rows":        total_gt_rows,
        "rows_with_match":      rows_with_match,
        "rows_no_match":        rows_no_match,
        "total_true_pairs":     n,
        "s2_pairs":             s2_pairs,
        "s3_pairs":             s3_pairs,
        "raw_name_count":       raw_name_count,
        "raw_name_pct":         raw_name_pct,
        "norm_name_count":      norm_name_count,
        "norm_name_pct":        norm_name_pct,
        "name_improvement":     name_improvement,
        "n_name_newly":         n_name_newly,
        "raw_addr_count":       raw_addr_count,
        "raw_addr_pct":         raw_addr_pct,
        "norm_addr_count":      norm_addr_count,
        "norm_addr_pct":        norm_addr_pct,
        "addr_improvement":     addr_improvement,
        "n_addr_newly":         n_addr_newly,
        "name_newly_exact":     name_newly_exact,
        "addr_newly_exact":     addr_newly_exact,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3 -- Normalization Effectiveness on Training Data"
    )
    parser.add_argument(
        "--data-dir",
        default="D:/6ab10eb3b23ba_student_resource/student_resource/dataset/train",
        help="Directory containing train_source1/2/3.tsv and train_ground_truth.tsv",
    )
    parser.add_argument(
        "--out-file",
        default=None,
        help="Optional path to write results as UTF-8 text file (avoids PowerShell pipe encoding issues).",
    )
    args = parser.parse_args()

    if args.out_file:
        import io
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _orig_stdout = sys.stdout
        sys.stdout = io.TextIOWrapper(
            open(out_path, "wb"), encoding="utf-8", line_buffering=True
        )
        try:
            run(Path(args.data_dir))
        finally:
            sys.stdout.flush()
            sys.stdout.close()
            sys.stdout = _orig_stdout
        print(f"Results written to {out_path}")
    else:
        run(Path(args.data_dir))
