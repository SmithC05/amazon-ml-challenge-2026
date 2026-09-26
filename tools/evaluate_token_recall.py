"""
tools/evaluate_token_recall.py
===============================
Stage 1: Ground-truth recall evaluation for M4 token-only benchmark outputs.

Compares token-only candidate TSVs (produced by benchmark_duckdb.py) against
train_ground_truth.tsv, restricted to the first N S1 rows and S2-only true
matches (because the benchmark runs against S2 only with an empty S3).

Does NOT modify candidate_generation.py or regenerate any candidates.

Usage
-----
    python tools/evaluate_token_recall.py \\
        --cache-dir   /content/drive/MyDrive/amazon_cache \\
        --gt          /content/drive/MyDrive/train_ground_truth.tsv \\
        --candidates  /content/bench_maxdf100.tsv \\
                      /content/bench_maxdf500.tsv \\
                      /content/bench_maxdf1000.tsv \\
        [--labels     100 500 1000] \\
        [--n-s1       10000]

Arguments
---------
--cache-dir     Path to M2 Parquet cache (needs train_source1.parquet for IDs)
--gt            Path to train_ground_truth.tsv
--candidates    One or more candidate TSV files (one per token_max_df)
--labels        Optional labels for each file (default: basename)
--n-s1          Number of S1 rows to use from benchmark (default 10000)

Ground-truth TSV columns
------------------------
    source1_entity_id  |  matched_entity_ids   (comma-separated)

Candidate TSV columns (M4 output format)
-----------------------------------------
    source1_entity_id  |  candidate_entity_ids  (comma-separated)

Evaluation rules
----------------
1. Use exactly the first --n-s1 entity_ids from train_source1.parquet.
2. Filter ground-truth to those S1 IDs only.
3. Keep only S2-* IDs from matched_entity_ids (S3 matches are invisible
   to a token+S2-only benchmark run).
4. For each candidate file:
     - Validate that it contains exactly those S1 IDs.
     - Stream row by row; compute all metrics.
5. Print a comparison table.

Metrics
-------
    total_s2_gt         - total S2 true-match IDs across all sampled S1
    found               - S2 true matches present in the candidate set
    lost                - S2 true matches absent from the candidate set
    recall              - found / total_s2_gt
    cand_pairs          - total candidate IDs across all S1 rows
    avg_per_s1          - cand_pairs / n_s1
    max_per_s1          - maximum candidate IDs for any single S1
    s1_with_cands       - S1 rows that received >=1 candidate
    s1_with_s2_truth    - S1 rows that have >=1 S2 true match
    s1_fully_recovered  - S1 rows where ALL S2 true matches are in candidates
    cand_true_frac      - found / cand_pairs  (diagnostic, NOT precision)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _load_s1_ids(cache_dir: Path, n: int) -> list[str]:
    """Return the first *n* entity_ids from train_source1.parquet."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: pip install pyarrow")

    path = cache_dir / "train_source1.parquet"
    if not path.exists():
        sys.exit(f"Cannot find {path}")

    table = pq.read_table(path, columns=["entity_id"])
    ids   = table["entity_id"].to_pylist()
    return ids[:n]


def _load_gt(gt_path: Path, s1_ids: set[str]) -> dict[str, set[str]]:
    """
    Stream train_ground_truth.tsv.
    Return {s1_id -> set of S2-* matched IDs} for s1_ids only.
    """
    truth: dict[str, set[str]] = {sid: set() for sid in s1_ids}

    with open(gt_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        # Normalise column name: the GT may use matched_entity_ids or
        # candidate_entity_ids depending on the file version.
        fieldnames = reader.fieldnames or []
        match_col  = next(
            (c for c in fieldnames if "matched" in c.lower() or "candidate" in c.lower()),
            None,
        )
        if match_col is None:
            sys.exit(
                f"Cannot find a 'matched_entity_ids' column in {gt_path}.\n"
                f"Columns found: {fieldnames}"
            )

        id_col = next(
            (c for c in fieldnames if "source1" in c.lower() or c.lower() == "entity_id"),
            None,
        )
        if id_col is None:
            sys.exit(
                f"Cannot find a 'source1_entity_id' column in {gt_path}.\n"
                f"Columns found: {fieldnames}"
            )

        for row in reader:
            sid = row[id_col].strip()
            if sid not in s1_ids:
                continue
            raw = row[match_col].strip() if row[match_col] else ""
            if not raw:
                continue
            s2_ids = {x.strip() for x in raw.split(",")
                      if x.strip().startswith("S2-")}
            truth[sid] = s2_ids

    return truth


def _load_candidates(cand_path: Path, s1_ids: set[str]) -> dict[str, set[str]]:
    """
    Stream a candidate TSV (M4 output format).
    Return {s1_id -> set of candidate IDs} for s1_ids only.
    Validates that all s1_ids are present.
    """
    cands: dict[str, set[str]] = {}
    unknown_ids: list[str] = []

    with open(cand_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        fieldnames = reader.fieldnames or []
        id_col     = next(
            (c for c in fieldnames if "source1" in c.lower()),
            None,
        )
        cand_col   = next(
            (c for c in fieldnames if "candidate" in c.lower()),
            None,
        )
        if id_col is None or cand_col is None:
            sys.exit(
                f"Candidate file {cand_path} must have "
                f"source1_entity_id and candidate_entity_ids columns.\n"
                f"Columns found: {fieldnames}"
            )

        for row in reader:
            sid = row[id_col].strip()
            if sid not in s1_ids:
                unknown_ids.append(sid)
                continue
            raw = row[cand_col].strip() if row[cand_col] else ""
            cands[sid] = {x.strip() for x in raw.split(",") if x.strip()} if raw else set()

    missing = s1_ids - set(cands.keys())
    if missing:
        n_show = min(5, len(missing))
        print(
            f"  WARNING: {cand_path.name} is missing {len(missing):,} S1 IDs "
            f"(first {n_show}: {list(missing)[:n_show]})",
            file=sys.stderr,
        )
    if unknown_ids:
        print(
            f"  WARNING: {cand_path.name} contains {len(unknown_ids):,} rows "
            f"not in the S1 benchmark set.",
            file=sys.stderr,
        )

    return cands


def _evaluate(
    truth:    dict[str, set[str]],
    cands:    dict[str, set[str]],
    s1_ids:   list[str],
) -> dict:
    """Compute all metrics for one (truth, cands) pair."""
    total_gt          = 0
    found             = 0
    total_cand_pairs  = 0
    max_cand          = 0
    s1_with_cands     = 0
    s1_with_s2_truth  = 0
    s1_fully_recovered = 0

    for sid in s1_ids:
        true_s2  = truth.get(sid, set())
        cand_set = cands.get(sid, set())

        n_cand = len(cand_set)
        n_true = len(true_s2)

        total_gt         += n_true
        total_cand_pairs += n_cand
        max_cand          = max(max_cand, n_cand)

        if n_cand > 0:
            s1_with_cands += 1
        if n_true > 0:
            s1_with_s2_truth += 1

        hits = len(true_s2 & cand_set)
        found += hits
        if n_true > 0 and hits == n_true:
            s1_fully_recovered += 1

    n_s1 = len(s1_ids)
    lost = total_gt - found

    return {
        "total_s2_gt":          total_gt,
        "found":                 found,
        "lost":                  lost,
        "recall":                found / total_gt if total_gt else None,
        "cand_pairs":            total_cand_pairs,
        "avg_per_s1":            total_cand_pairs / n_s1 if n_s1 else 0.0,
        "max_per_s1":            max_cand,
        "s1_with_cands":         s1_with_cands,
        "s1_with_s2_truth":      s1_with_s2_truth,
        "s1_fully_recovered":    s1_fully_recovered,
        # Diagnostic only — NOT model precision
        "cand_true_frac":        found / total_cand_pairs if total_cand_pairs else None,
    }


def _fmt_pct(v: float | None, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}" if v is not None else "  n/a  "


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: token-only GT recall evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cache-dir",   required=True,  help="M2 Parquet cache directory")
    parser.add_argument("--gt",          required=True,  help="train_ground_truth.tsv path")
    parser.add_argument("--candidates",  required=True,  nargs="+",
                        help="Candidate TSV files (one per token_max_df)")
    parser.add_argument("--labels",      nargs="+",      default=None,
                        help="Labels for each candidate file (default: filename)")
    parser.add_argument("--n-s1",        type=int,       default=10_000,
                        help="Number of S1 rows from benchmark (default 10000)")
    args = parser.parse_args()

    cache_dir   = Path(args.cache_dir)
    gt_path     = Path(args.gt)
    cand_paths  = [Path(p) for p in args.candidates]
    n_s1        = args.n_s1

    labels = args.labels or [p.name for p in cand_paths]
    if len(labels) != len(cand_paths):
        sys.exit("--labels count must match --candidates count")

    for p in cand_paths:
        if not p.exists():
            sys.exit(f"Candidate file not found: {p}")
    if not gt_path.exists():
        sys.exit(f"Ground-truth file not found: {gt_path}")

    # ── Load S1 benchmark IDs ─────────────────────────────────────────────────
    print(f"Loading first {n_s1:,} S1 IDs from train_source1.parquet…", flush=True)
    s1_ids_list = _load_s1_ids(cache_dir, n_s1)
    s1_ids_set  = set(s1_ids_list)
    print(f"  Loaded {len(s1_ids_list):,} S1 IDs.", flush=True)

    # ── Stream ground truth (S2-only) ─────────────────────────────────────────
    print(f"Loading ground truth from {gt_path.name}…", flush=True)
    truth = _load_gt(gt_path, s1_ids_set)
    n_with_s2_gt = sum(1 for v in truth.values() if v)
    total_s2_gt  = sum(len(v) for v in truth.values())
    print(
        f"  S1 rows with >=1 S2 true match : {n_with_s2_gt:,} / {n_s1:,}",
        flush=True,
    )
    print(f"  Total S2 true-match IDs        : {total_s2_gt:,}", flush=True)

    # ── Evaluate each candidate file ──────────────────────────────────────────
    all_results: list[tuple[str, dict]] = []

    for label, path in zip(labels, cand_paths):
        print(f"\nLoading candidates: {path.name}…", flush=True)
        cands = _load_candidates(path, s1_ids_set)
        metrics = _evaluate(truth, cands, s1_ids_list)
        all_results.append((label, metrics))
        print(
            f"  recall={_fmt_pct(metrics['recall'])}  "
            f"pairs={metrics['cand_pairs']:,}  "
            f"avg/S1={metrics['avg_per_s1']:.1f}",
            flush=True,
        )

    # ── Comparison table ──────────────────────────────────────────────────────
    W = 78
    print()
    print("=" * W)
    print("TOKEN BLOCK GROUND-TRUTH EVALUATION")
    print(f"  S1 sample : {n_s1:,}")
    print(f"  GT file   : {gt_path.name}")
    print("  Scope     : S2 true matches only (benchmark has empty S3)")
    print("=" * W)

    # Header row
    col_w = max(len(l) for l in labels) + 2
    hdr = (
        f"{'label':<{col_w}}  {'GT_S2':>7}  {'found':>7}  {'lost':>5}"
        f"  {'recall':>7}  {'cand_pairs':>12}  {'avg/S1':>7}  {'max/S1':>7}"
    )
    print(hdr)
    print("-" * W)

    for label, m in all_results:
        print(
            f"{label:<{col_w}}  "
            f"{m['total_s2_gt']:>7,}  "
            f"{m['found']:>7,}  "
            f"{m['lost']:>5,}  "
            f"{_fmt_pct(m['recall']):>7}  "
            f"{m['cand_pairs']:>12,}  "
            f"{m['avg_per_s1']:>7.1f}  "
            f"{m['max_per_s1']:>7,}"
        )

    print("-" * W)
    print()

    # Detailed block per label
    print("DETAIL PER LABEL")
    print("-" * W)
    for label, m in all_results:
        print(f"  [{label}]")
        print(f"    S1 with >=1 candidate          : {m['s1_with_cands']:>8,}")
        print(f"    S1 with >=1 S2 true match      : {m['s1_with_s2_truth']:>8,}")
        print(f"    S1 with ALL S2 matches found   : {m['s1_fully_recovered']:>8,}  "
              f"({m['s1_fully_recovered']/m['s1_with_s2_truth']*100:.1f}% of truth-bearing S1)"
              if m['s1_with_s2_truth'] else "")
        cand_tf = _fmt_pct(m['cand_true_frac'], 5)
        print(f"    Candidate true-match frac(*)   : {cand_tf}")
        print(f"    (*) = found / cand_pairs  — diagnostic only, NOT precision")
        print()

    print("=" * W)
    print("NEXT STEP: run Stage 2 (all 6 blocks + S2+S3) for the 1-2 best thresholds.")
    print("=" * W)


if __name__ == "__main__":
    main()
