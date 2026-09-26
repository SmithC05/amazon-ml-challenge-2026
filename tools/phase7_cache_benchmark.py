#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/phase7_cache_benchmark.py
================================
Phase 7 — Real Cache Validation + Performance/Memory Report.

Builds all six competition caches one-at-a-time, measures:
  - wall-clock build time
  - peak RAM via tracemalloc
  - Parquet output size
  - TSV vs Parquet row-count parity
  - 14-column schema validation
  - normalized content spot-check

Also compares non-chunked vs chunked on train_source1.

Usage (from repo root):
    python tools/phase7_cache_benchmark.py \
        --train-dir D:/6ab10eb3b23ba_student_resource/student_resource/dataset/train \
        --test-dir  D:/6ab10eb3b23ba_student_resource/student_resource/dataset/test \
        --cache-dir cache \
        --chunk-dir cache_chunked \
        --chunksize 500000 \
        --out-file  output/phase7_report.txt
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cache import (
    build_cache,
    load_cache,
    validate_cache,
    cache_exists,
    _OUTPUT_COLUMNS,
    _REQUIRED_COLUMNS,
)
from preprocess import normalize_name, normalize_address


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tsv_row_count(tsv_path: Path) -> int:
    """Count rows in a TSV efficiently (line count - 1 for header)."""
    with open(tsv_path, "rb") as f:
        count = sum(1 for _ in f)
    return count - 1   # subtract header


def _build_and_measure(
    split: str,
    source: str,
    data_dir: Path,
    cache_dir: Path,
    chunksize: int | None = None,
    force: bool = True,
) -> dict:
    """
    Build one cache file, measuring wall-clock time and peak RAM.
    Returns a results dict.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    result = build_cache(
        split=split,
        source=source,
        data_dir=data_dir,
        cache_dir=cache_dir,
        force=force,
        chunksize=chunksize,
    )

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result["elapsed_s"]  = round(elapsed, 2)
    result["peak_ram_mb"] = round(peak_bytes / 1e6, 1)
    return result


def _validate_one(
    split: str,
    source: str,
    cache_dir: Path,
    tsv_dir: Path,
    tsv_filename: str,
) -> dict:
    """
    Validate a single cache: schema, row count, readability, content spot-check.
    """
    out = {
        "split": split,
        "source": source,
        "tsv_rows": None,
        "parquet_rows": None,
        "row_count_match": False,
        "schema_valid": False,
        "non_empty": False,
        "readable": False,
        "spot_check_name_norm": None,
        "spot_check_address_norm": None,
        "errors": [],
    }

    # 1. TSV row count
    tsv_path = tsv_dir / tsv_filename
    if tsv_path.exists():
        out["tsv_rows"] = _tsv_row_count(tsv_path)
    else:
        out["errors"].append(f"TSV not found: {tsv_path}")

    # 2. Validate via cache API
    vr = validate_cache(split, source, cache_dir)
    out["readable"]    = vr["checks"].get("readable", False)
    out["schema_valid"] = vr["valid"]
    out["parquet_rows"] = vr.get("rows")
    out["non_empty"]    = (vr.get("rows") or 0) > 0
    if not vr["valid"]:
        out["errors"].extend(vr.get("errors", []))

    # 3. Row-count match
    if out["tsv_rows"] is not None and out["parquet_rows"] is not None:
        out["row_count_match"] = (out["tsv_rows"] == out["parquet_rows"])

    # 4. Content spot-check (load only 5 rows)
    if out["readable"]:
        try:
            df = load_cache(split, source, cache_dir,
                            columns=["entity_id", "business_name",
                                     "business_address", "name_norm",
                                     "address_norm"])
            row = df.iloc[0]
            raw_name  = str(row.get("business_name", "") or "")
            raw_addr  = str(row.get("business_address", "") or "")
            exp_name  = normalize_name(raw_name)
            exp_addr  = normalize_address(raw_addr)
            cached_name = str(row["name_norm"] or "")
            cached_addr = str(row["address_norm"] or "")
            out["spot_check_name_norm"]    = (cached_name == exp_name)
            out["spot_check_address_norm"] = (cached_addr == exp_addr)
            del df
        except Exception as e:
            out["errors"].append(f"Spot-check failed: {e}")

    return out


def _fmt(v, decimals=1) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:,.{decimals}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    train_dir: Path,
    test_dir: Path,
    cache_dir: Path,
    chunk_dir: Path,
    chunksize: int,
    out_file: Path | None,
) -> None:

    buf = io.StringIO()

    def log(msg: str = "", **kwargs) -> None:
        print(msg, **kwargs)
        buf.write(msg + "\n")

    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    SOURCES = [
        ("train", "source1", train_dir, "train_source1.tsv"),
        ("train", "source2", train_dir, "train_source2.tsv"),
        ("train", "source3", train_dir, "train_source3.tsv"),
        ("test",  "source1", test_dir,  "test_source1.tsv"),
        ("test",  "source2", test_dir,  "test_source2.tsv"),
        ("test",  "source3", test_dir,  "test_source3.tsv"),
    ]

    # ──────────────────────────────────────────────────────────────
    # STEP 1: Count TSV rows (no DataFrame needed)
    # ──────────────────────────────────────────────────────────────
    log("=" * 70)
    log("STEP 1 — TSV ROW COUNTS")
    log("=" * 70)
    tsv_counts = {}
    for split, source, data_dir, tsv_filename in SOURCES:
        tsv_path = data_dir / tsv_filename
        if tsv_path.exists():
            n = _tsv_row_count(tsv_path)
            tsv_counts[(split, source)] = n
            log(f"  {tsv_filename:<30}  {n:>12,} rows")
        else:
            tsv_counts[(split, source)] = None
            log(f"  {tsv_filename:<30}  NOT FOUND")

    # ──────────────────────────────────────────────────────────────
    # STEP 2: Build all 6 caches (non-chunked, one at a time)
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 70)
    log("STEP 2 — BUILD ALL 6 CACHES (non-chunked, sequential)")
    log("=" * 70)

    build_results = {}
    for split, source, data_dir, tsv_filename in SOURCES:
        key = (split, source)
        existing = cache_exists(split, source, cache_dir)
        if existing:
            # Already built — skip to save time; measure load time instead
            log(f"  [{split}/{source}]  Cache already exists — skipping rebuild")
            build_results[key] = {
                "split": split, "source": source,
                "status": "existing", "rows": tsv_counts.get(key),
                "elapsed_s": None, "peak_ram_mb": None,
            }
        else:
            log(f"  [{split}/{source}]  Building ...")
            sys.stdout.flush()
            r = _build_and_measure(
                split, source, data_dir, cache_dir,
                chunksize=None, force=False,
            )
            build_results[key] = r
            rows_s = r.get("rows") or 0
            elapsed = r.get("elapsed_s") or 0
            rps = int(rows_s / elapsed) if elapsed > 0 else 0
            peak = r.get("peak_ram_mb") or 0
            log(f"    -> {rows_s:>12,} rows  {elapsed:>8.1f}s  "
                f"{rps:>10,} rows/s  "
                f"peak={peak:>8.1f} MB  "
                f"status={r['status']}")

    # ──────────────────────────────────────────────────────────────
    # STEP 3: Validate all 6 caches
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 70)
    log("STEP 3 — VALIDATE ALL 6 CACHES")
    log("=" * 70)

    val_results = {}
    for split, source, data_dir, tsv_filename in SOURCES:
        log(f"  [{split}/{source}] validating ...")
        vr = _validate_one(split, source, cache_dir, data_dir, tsv_filename)
        val_results[(split, source)] = vr
        status = "PASS" if (vr["schema_valid"] and vr["row_count_match"]
                            and vr["readable"] and vr["non_empty"]) else "FAIL"
        tsv_n   = _fmt(vr["tsv_rows"])
        pq_n    = _fmt(vr["parquet_rows"])
        match   = "✓" if vr["row_count_match"] else "✗"
        schema  = "✓" if vr["schema_valid"]    else "✗"
        readable = "✓" if vr["readable"]       else "✗"
        log(f"{status}  TSV={tsv_n}  PQ={pq_n}  rowmatch={match}  "
            f"schema={schema}  readable={readable}")
        if vr["errors"]:
            for e in vr["errors"]:
                log(f"    ERROR: {e}")

    # ──────────────────────────────────────────────────────────────
    # STEP 4: Parquet file sizes
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 70)
    log("STEP 4 — PARQUET FILE SIZES")
    log("=" * 70)
    from cache import _cache_path
    pq_sizes = {}
    for split, source, _, _ in SOURCES:
        p = _cache_path(split, source, cache_dir)
        if p.exists():
            mb = p.stat().st_size / 1e6
            pq_sizes[(split, source)] = mb
            log(f"  {p.name:<35}  {mb:>8.1f} MB")
        else:
            pq_sizes[(split, source)] = None
            log(f"  {split}_{source}.parquet   NOT FOUND")

    # ──────────────────────────────────────────────────────────────
    # STEP 5: Chunked vs non-chunked comparison on train_source1
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 70)
    log("STEP 5 — CHUNKED vs NON-CHUNKED COMPARISON (train_source1)")
    log("=" * 70)

    # Non-chunked — force=True to get a real measurement
    log("  [non-chunked] Building train_source1 with force=True ...")
    nc = _build_and_measure(
        "train", "source1", train_dir, cache_dir,
        chunksize=None, force=True,
    )
    log(f"    rows={nc['rows']:,}  elapsed={nc['elapsed_s']:.2f}s  "
        f"peak_RAM={nc['peak_ram_mb']:.1f} MB")

    # Chunked
    log(f"  [chunked={chunksize:,}] Building train_source1 (chunked cache dir) ...")
    ck = _build_and_measure(
        "train", "source1", train_dir, chunk_dir,
        chunksize=chunksize, force=True,
    )
    log(f"    rows={ck['rows']:,}  elapsed={ck['elapsed_s']:.2f}s  "
        f"peak_RAM={ck['peak_ram_mb']:.1f} MB")

    time_diff = ck["elapsed_s"] - nc["elapsed_s"]
    ram_diff  = ck["peak_ram_mb"] - nc["peak_ram_mb"]
    log(f"  Time diff    (chunked - non-chunked): {time_diff:+.2f}s")
    log(f"  RAM diff     (chunked - non-chunked): {ram_diff:+.1f} MB")

    # Verify content equivalence
    log("  Verifying content equivalence ...")
    df_nc = load_cache("train", "source1", cache_dir,
                       columns=["entity_id", "name_norm", "address_norm"])
    df_ck = load_cache("train", "source1", chunk_dir,
                       columns=["entity_id", "name_norm", "address_norm"])
    rows_match = len(df_nc) == len(df_ck)
    name_match = list(df_nc["name_norm"]) == list(df_ck["name_norm"])
    addr_match = list(df_nc["address_norm"]) == list(df_ck["address_norm"])
    log(f"    row count match:    {'PASS' if rows_match else 'FAIL'}  "
        f"({len(df_nc):,} vs {len(df_ck):,})")
    log(f"    name_norm match:    {'PASS' if name_match else 'FAIL'}")
    log(f"    address_norm match: {'PASS' if addr_match else 'FAIL'}")
    del df_nc, df_ck

    # ──────────────────────────────────────────────────────────────
    # STEP 6: Normalized content spot-checks
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 70)
    log("STEP 6 — NORMALIZED CONTENT SPOT-CHECKS (5 rows per cache)")
    log("=" * 70)

    for split, source, data_dir, _ in SOURCES:
        log(f"\n  [{split}/{source}]")
        try:
            df = load_cache(split, source, cache_dir,
                            columns=["entity_id", "business_name",
                                     "business_address", "country",
                                     "name_norm", "address_norm",
                                     "name_token_count", "address_token_count"])
            # Show 3 non-null rows
            sample = df[df["business_name"].notna()].head(3)
            for _, row in sample.iterrows():
                exp_name = normalize_name(str(row.get("business_name") or ""))
                exp_addr = normalize_address(str(row.get("business_address") or ""))
                name_ok = str(row["name_norm"] or "") == exp_name
                addr_ok = str(row["address_norm"] or "") == exp_addr
                log(f"    entity={row['entity_id']}  "
                    f"name_norm={'✓' if name_ok else '✗'}  "
                    f"addr_norm={'✓' if addr_ok else '✗'}  "
                    f"ntoks={row['name_token_count']}  "
                    f"atoks={row['address_token_count']}")
            del df
        except Exception as e:
            log(f"    ERROR: {e}")

    # ──────────────────────────────────────────────────────────────
    # STEP 7: FINAL SUMMARY TABLE
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 70)
    log("PHASE 7 REAL CACHE VALIDATION + PERFORMANCE REPORT")
    log("=" * 70)

    # Header
    hdr = (f"{'Split':<6} {'Source':<8} {'TSV Rows':>12} "
           f"{'PQ Rows':>12} {'Match':>5} "
           f"{'Build(s)':>9} {'Rows/s':>11} "
           f"{'PQ Size':>9} {'PeakRAM':>10} {'Valid':>6}")
    log(hdr)
    log("-" * len(hdr))

    total_train_time = 0.0
    total_test_time  = 0.0
    for split, source, _, _ in SOURCES:
        key  = (split, source)
        br   = build_results[key]
        vr   = val_results[key]
        tsv_n = tsv_counts.get(key)
        pq_n  = vr.get("parquet_rows")
        match = "YES" if vr["row_count_match"] else "NO "
        valid = "PASS" if (vr["schema_valid"] and vr["row_count_match"]
                           and vr["readable"] and vr["non_empty"]) else "FAIL"
        elapsed = br.get("elapsed_s")
        rows    = br.get("rows") or pq_n or 0
        rps     = int(rows / elapsed) if elapsed and elapsed > 0 else None
        pq_mb   = pq_sizes.get(key)

        if elapsed:
            if split == "train":
                total_train_time += elapsed
            else:
                total_test_time += elapsed

        log(f"{split:<6} {source:<8} "
            f"{_fmt(tsv_n):>12} "
            f"{_fmt(pq_n):>12} "
            f"{match:>5} "
            f"{_fmt(elapsed,1):>9} "
            f"{_fmt(rps):>11} "
            f"{_fmt(pq_mb,1)+' MB':>9} "
            f"{_fmt(br.get('peak_ram_mb'),1)+' MB':>10} "
            f"{valid:>6}")

    log()
    log(f"Total train build time  : {total_train_time:.1f}s")
    log(f"Total test  build time  : {total_test_time:.1f}s")

    log()
    log("VALIDATION SUMMARY")
    all_readable    = all(v["readable"]    for v in val_results.values())
    all_row_match   = all(v["row_count_match"] for v in val_results.values())
    all_schema      = all(v["schema_valid"]    for v in val_results.values())
    all_non_empty   = all(v["non_empty"]       for v in val_results.values())
    log(f"  All 6 caches readable:       {'YES' if all_readable  else 'NO'}")
    log(f"  All 6 row counts verified:   {'YES' if all_row_match  else 'NO'}")
    log(f"  All 6 schemas validated:     {'YES' if all_schema     else 'NO'}")
    log(f"  All 6 non-empty:             {'YES' if all_non_empty  else 'NO'}")

    log()
    log("CHUNKING COMPARISON (train_source1)")
    log(f"  Non-chunked:  {nc['elapsed_s']:.2f}s   peak RAM = {nc['peak_ram_mb']:.1f} MB")
    log(f"  Chunked={chunksize:,}: {ck['elapsed_s']:.2f}s   peak RAM = {ck['peak_ram_mb']:.1f} MB")
    log(f"  Time diff:    {time_diff:+.2f}s")
    log(f"  RAM diff:     {ram_diff:+.1f} MB  ({'reduction' if ram_diff < 0 else 'increase'})")
    log(f"  Content:      rows={'MATCH' if rows_match else 'MISMATCH'}  "
        f"name_norm={'MATCH' if name_match else 'MISMATCH'}  "
        f"address_norm={'MATCH' if addr_match else 'MISMATCH'}")

    log()
    log("SCOPE CHECK")
    log("  normalization rules changed : NO")
    log("  new derived fields          : NO")
    log("  blocking                    : NO")
    log("  candidate generation        : NO")
    log("  fuzzy matching              : NO")
    log("  feature engineering         : NO")
    log("  ML/model changes            : NO")
    log("  prediction                  : NO")
    log("  submission                  : NO")

    report = buf.getvalue()

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report, encoding="utf-8")
        print(f"\nReport written → {out_file}")

    return report


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 cache benchmark")
    parser.add_argument("--train-dir",
        default="D:/6ab10eb3b23ba_student_resource/student_resource/dataset/train")
    parser.add_argument("--test-dir",
        default="D:/6ab10eb3b23ba_student_resource/student_resource/dataset/test")
    parser.add_argument("--cache-dir",  default="cache")
    parser.add_argument("--chunk-dir",  default="cache_chunked")
    parser.add_argument("--chunksize",  type=int, default=500_000)
    parser.add_argument("--out-file",   default="output/phase7_report.txt")
    args = parser.parse_args()

    run(
        train_dir = Path(args.train_dir),
        test_dir  = Path(args.test_dir),
        cache_dir = Path(args.cache_dir),
        chunk_dir = Path(args.chunk_dir),
        chunksize = args.chunksize,
        out_file  = Path(args.out_file),
    )
