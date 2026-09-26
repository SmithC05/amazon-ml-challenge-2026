#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/phase7_benchmark_v2.py
============================
Phase 7 — Real Cache Validation + Performance/Memory Report (v2).

Strategy:
  - train_source1  non-chunked cache already built (511.4s, 343.9 MB, 2298 MB peak RAM).
    → Reuse it; skip rebuild.  Measure load time only.
  - Remaining 5 sources: build using chunksize=500_000 (bounded RAM).
  - Chunked-vs-non-chunked comparison: use train_source1 (non-chunked already on disk;
    build chunked version and compare).
  - Validate all 6 caches after building.
  - Spot-check normalized content.

Run from repo root:
    python tools/phase7_benchmark_v2.py
"""

from __future__ import annotations

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
    _cache_path,
)
from preprocess import normalize_name, normalize_address, _OUTPUT_COLUMNS

TRAIN_DIR = Path(
    "D:/6ab10eb3b23ba_student_resource/student_resource/dataset/train"
)
TEST_DIR = Path(
    "D:/6ab10eb3b23ba_student_resource/student_resource/dataset/test"
)
CACHE_DIR    = Path("cache")
CHUNK_DIR    = Path("cache_chunked")
CHUNKSIZE    = 500_000
OUT_FILE     = Path("output/phase7_report.txt")

# Known timing for train_source1 non-chunked (captured before kill)
S1_NC_ELAPSED  = 511.4
S1_NC_ROWS     = 2_206_821
S1_NC_PEAK_MB  = 2298.1
S1_NC_PQ_MB    = 343.9

SOURCES = [
    ("train", "source1", TRAIN_DIR, "train_source1.tsv"),
    ("train", "source2", TRAIN_DIR, "train_source2.tsv"),
    ("train", "source3", TRAIN_DIR, "train_source3.tsv"),
    ("test",  "source1", TEST_DIR,  "test_source1.tsv"),
    ("test",  "source2", TEST_DIR,  "test_source2.tsv"),
    ("test",  "source3", TEST_DIR,  "test_source3.tsv"),
]


def _tsv_row_count(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def _build_chunked(split, source, data_dir, cache_dir, force=False):
    tracemalloc.start()
    t0 = time.perf_counter()
    r = build_cache(split, source,
                    data_dir=data_dir,
                    cache_dir=cache_dir,
                    force=force,
                    chunksize=CHUNKSIZE)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    r["elapsed_s"]   = round(elapsed, 2)
    r["peak_ram_mb"] = round(peak / 1e6, 1)
    return r


def _validate_one(split, source, cache_dir, data_dir, tsv_name):
    out = {
        "tsv_rows": None, "parquet_rows": None,
        "row_count_match": False, "schema_valid": False,
        "non_empty": False, "readable": False,
        "spot_ok": None, "errors": [],
    }
    tsv_path = data_dir / tsv_name
    if tsv_path.exists():
        out["tsv_rows"] = _tsv_row_count(tsv_path)
    else:
        out["errors"].append(f"TSV not found: {tsv_path}")

    vr = validate_cache(split, source, cache_dir)
    out["readable"]     = vr["checks"].get("readable", False)
    out["schema_valid"] = vr["valid"]
    out["parquet_rows"] = vr.get("rows")
    out["non_empty"]    = (vr.get("rows") or 0) > 0
    if not vr["valid"]:
        out["errors"].extend(vr.get("errors", []))

    if out["tsv_rows"] is not None and out["parquet_rows"] is not None:
        out["row_count_match"] = (out["tsv_rows"] == out["parquet_rows"])

    if out["readable"]:
        try:
            df = load_cache(split, source, cache_dir,
                            columns=["entity_id", "business_name",
                                     "business_address", "name_norm",
                                     "address_norm"])
            row = df.iloc[0]
            exp_n = normalize_name(str(row.get("business_name") or ""))
            exp_a = normalize_address(str(row.get("business_address") or ""))
            ok = (str(row["name_norm"] or "") == exp_n and
                  str(row["address_norm"] or "") == exp_a)
            out["spot_ok"] = ok
            del df
        except Exception as e:
            out["errors"].append(f"spot-check: {e}")

    return out


def _fmt(v, d=1):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:,.{d}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def main():
    buf = io.StringIO()

    def log(msg=""):
        print(msg)
        buf.write(msg + "\n")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # STEP 1: TSV row counts
    # ──────────────────────────────────────────────────────────────
    log("=" * 68)
    log("STEP 1 — TSV ROW COUNTS")
    log("=" * 68)
    tsv_counts = {}
    for split, source, data_dir, tsv_name in SOURCES:
        tsv_path = data_dir / tsv_name
        if tsv_path.exists():
            n = _tsv_row_count(tsv_path)
            tsv_counts[(split, source)] = n
            log(f"  {tsv_name:<30}  {n:>12,} rows")
        else:
            tsv_counts[(split, source)] = None
            log(f"  {tsv_name:<30}  NOT FOUND")

    # ──────────────────────────────────────────────────────────────
    # STEP 2: Build remaining 5 caches with chunking
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 68)
    log(f"STEP 2 — BUILD REMAINING 5 CACHES (chunksize={CHUNKSIZE:,})")
    log("  [train/source1] — already built non-chunked (511.4s, 343.9 MB)")
    log("=" * 68)

    build_results = {
        ("train", "source1"): {
            "status": "existing",
            "rows": S1_NC_ROWS,
            "elapsed_s": S1_NC_ELAPSED,
            "peak_ram_mb": S1_NC_PEAK_MB,
            "method": "non-chunked (pre-built)",
        }
    }

    for split, source, data_dir, _ in SOURCES:
        key = (split, source)
        if key == ("train", "source1"):
            continue   # already handled above

        exists = cache_exists(split, source, CACHE_DIR)
        if exists:
            log(f"  [{split}/{source}]  Cache already exists — skipping")
            build_results[key] = {
                "status": "existing",
                "rows": tsv_counts.get(key),
                "elapsed_s": None,
                "peak_ram_mb": None,
                "method": "existing",
            }
        else:
            log(f"  [{split}/{source}]  Building (chunked={CHUNKSIZE:,}) ...")
            sys.stdout.flush()
            r = _build_chunked(split, source, data_dir, CACHE_DIR, force=False)
            build_results[key] = r
            r["method"] = f"chunked={CHUNKSIZE:,}"
            rows_s  = r.get("rows") or 0
            elapsed = r.get("elapsed_s") or 0
            rps     = int(rows_s / elapsed) if elapsed > 0 else 0
            peak    = r.get("peak_ram_mb") or 0
            log(f"    -> {rows_s:>12,} rows  {elapsed:>8.1f}s  "
                f"{rps:>10,} rows/s  peak={peak:>8.1f} MB  "
                f"status={r['status']}")

    # ──────────────────────────────────────────────────────────────
    # STEP 3: Validate all 6 caches
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 68)
    log("STEP 3 — VALIDATE ALL 6 CACHES")
    log("=" * 68)
    val_results = {}
    for split, source, data_dir, tsv_name in SOURCES:
        log(f"  [{split}/{source}] validating ...")
        vr = _validate_one(split, source, CACHE_DIR, data_dir, tsv_name)
        val_results[(split, source)] = vr
        ok = vr["schema_valid"] and vr["row_count_match"] and \
             vr["readable"]     and vr["non_empty"]
        status  = "PASS" if ok else "FAIL"
        match   = "OK" if vr["row_count_match"] else "NO"
        schema  = "OK" if vr["schema_valid"]    else "NO"
        spot    = "OK" if vr["spot_ok"]         else ("NO" if vr["spot_ok"] is False else "N/A")
        log(f"    {status}  TSV={_fmt(vr['tsv_rows'])}  "
            f"PQ={_fmt(vr['parquet_rows'])}  "
            f"rowmatch={match}  schema={schema}  spot={spot}")
        for e in vr.get("errors", []):
            log(f"    ERROR: {e}")

    # ──────────────────────────────────────────────────────────────
    # STEP 4: Parquet file sizes
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 68)
    log("STEP 4 — PARQUET FILE SIZES")
    log("=" * 68)
    pq_sizes = {}
    for split, source, _, _ in SOURCES:
        p = _cache_path(split, source, CACHE_DIR)
        if p.exists():
            mb = p.stat().st_size / 1e6
            pq_sizes[(split, source)] = mb
            log(f"  {p.name:<35}  {mb:>8.1f} MB")
        else:
            pq_sizes[(split, source)] = None
            log(f"  {split}_{source}.parquet   NOT FOUND")

    # ──────────────────────────────────────────────────────────────
    # STEP 5: Cache load timings (all 6)
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 68)
    log("STEP 5 — CACHE LOAD TIMINGS")
    log("=" * 68)
    load_times = {}
    for split, source, _, _ in SOURCES:
        t0 = time.perf_counter()
        df = load_cache(split, source, CACHE_DIR)
        elapsed = time.perf_counter() - t0
        load_times[(split, source)] = round(elapsed, 3)
        log(f"  {split}_{source}  {elapsed:6.3f}s  ({len(df):,} rows)")
        del df

    avg_load = sum(load_times.values()) / len(load_times)
    log(f"  Average load time: {avg_load:.3f}s")

    # ──────────────────────────────────────────────────────────────
    # STEP 6: Chunked vs non-chunked comparison (train_source1)
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 68)
    log("STEP 6 — CHUNKED vs NON-CHUNKED (train_source1)")
    log("=" * 68)
    log(f"  [non-chunked]  elapsed={S1_NC_ELAPSED:.1f}s  "
        f"peak_RAM={S1_NC_PEAK_MB:.1f} MB  (measured, pre-built)")

    log(f"  [chunked={CHUNKSIZE:,}]  Building into {CHUNK_DIR} ...")
    ck = _build_chunked("train", "source1", TRAIN_DIR, CHUNK_DIR, force=True)
    log(f"    rows={ck['rows']:,}  elapsed={ck['elapsed_s']:.1f}s  "
        f"peak_RAM={ck['peak_ram_mb']:.1f} MB")

    time_diff = ck["elapsed_s"] - S1_NC_ELAPSED
    ram_diff  = ck["peak_ram_mb"] - S1_NC_PEAK_MB
    log(f"  Time diff  (chunked - non-chunked): {time_diff:+.1f}s")
    log(f"  RAM diff   (chunked - non-chunked): {ram_diff:+.1f} MB  "
        f"({'reduction' if ram_diff < 0 else 'increase'})")

    # Content equivalence
    log("  Verifying content equivalence ...")
    df_nc = load_cache("train", "source1", CACHE_DIR,
                       columns=["entity_id", "name_norm", "address_norm"])
    df_ck = load_cache("train", "source1", CHUNK_DIR,
                       columns=["entity_id", "name_norm", "address_norm"])
    rows_match = len(df_nc) == len(df_ck)
    name_match = list(df_nc["name_norm"]) == list(df_ck["name_norm"])
    addr_match = list(df_nc["address_norm"]) == list(df_ck["address_norm"])
    log(f"    row count:    {'MATCH' if rows_match else 'MISMATCH'} "
        f"({len(df_nc):,} vs {len(df_ck):,})")
    log(f"    name_norm:    {'MATCH' if name_match else 'MISMATCH'}")
    log(f"    address_norm: {'MATCH' if addr_match else 'MISMATCH'}")
    del df_nc, df_ck

    # ──────────────────────────────────────────────────────────────
    # STEP 7: Normalized content spot-checks
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 68)
    log("STEP 7 — NORMALIZED CONTENT SPOT-CHECKS (3 rows per cache)")
    log("=" * 68)
    for split, source, _, _ in SOURCES:
        log(f"\n  [{split}/{source}]")
        try:
            df = load_cache(split, source, CACHE_DIR,
                            columns=["entity_id", "business_name",
                                     "business_address", "country",
                                     "name_norm", "address_norm",
                                     "name_token_count", "address_token_count"])
            sample = df[df["business_name"].notna()].head(3)
            for _, row in sample.iterrows():
                en = normalize_name(str(row.get("business_name") or ""))
                ea = normalize_address(str(row.get("business_address") or ""))
                nk = str(row["name_norm"] or "") == en
                ak = str(row["address_norm"] or "") == ea
                log(f"    entity={row['entity_id']}  "
                    f"name_norm={'OK' if nk else 'FAIL'}  "
                    f"addr_norm={'OK' if ak else 'FAIL'}  "
                    f"ntoks={row['name_token_count']}  "
                    f"atoks={row['address_token_count']}")
            del df
        except Exception as e:
            log(f"    ERROR: {e}")

    # ──────────────────────────────────────────────────────────────
    # STEP 8: Final performance summary table
    # ──────────────────────────────────────────────────────────────
    log()
    log("=" * 68)
    log("PHASE 7 REAL CACHE VALIDATION + PERFORMANCE REPORT")
    log("=" * 68)

    hdr = (f"{'Split':<6} {'Source':<8} {'Rows':>12} {'Build(s)':>10} "
           f"{'Rows/s':>9} {'PQ MB':>7} {'PeakMB':>9} {'Method':<22} {'Valid':>5}")
    log(hdr)
    log("-" * len(hdr))

    total_train_t = 0.0
    total_test_t  = 0.0
    for split, source, _, _ in SOURCES:
        key = (split, source)
        br  = build_results[key]
        vr  = val_results[key]
        valid = "PASS" if (vr["schema_valid"] and vr["row_count_match"]
                           and vr["readable"] and vr["non_empty"]) else "FAIL"
        rows    = br.get("rows") or vr.get("parquet_rows") or 0
        elapsed = br.get("elapsed_s")
        rps     = int(rows / elapsed) if elapsed and elapsed > 0 else None
        pq_mb   = pq_sizes.get(key)
        peak    = br.get("peak_ram_mb")
        method  = br.get("method", "?")

        if elapsed:
            if split == "train":
                total_train_t += elapsed
            else:
                total_test_t  += elapsed

        log(f"{split:<6} {source:<8} {_fmt(rows):>12} {_fmt(elapsed,1):>10} "
            f"{_fmt(rps):>9} {_fmt(pq_mb,1):>7} {_fmt(peak,1):>9} "
            f"{method:<22} {valid:>5}")

    log()
    log(f"Total train build time   : {total_train_t:.1f}s")
    log(f"Total test  build time   : {total_test_t:.1f}s")
    log(f"Total all-6 build time   : {total_train_t + total_test_t:.1f}s")
    log()

    # ── VALIDATION SUMMARY ────────────────────────────────────────
    log("VALIDATION SUMMARY")
    all_readable  = all(v["readable"]       for v in val_results.values())
    all_rowmatch  = all(v["row_count_match"] for v in val_results.values())
    all_schema    = all(v["schema_valid"]    for v in val_results.values())
    all_nonempty  = all(v["non_empty"]       for v in val_results.values())
    all_spotok    = all(v["spot_ok"] for v in val_results.values()
                        if v["spot_ok"] is not None)
    log(f"  All 6 caches readable:          {'YES' if all_readable  else 'NO'}")
    log(f"  All 6 row counts verified:      {'YES' if all_rowmatch  else 'NO'}")
    log(f"  All 6 schemas validated:        {'YES' if all_schema    else 'NO'}")
    log(f"  All 6 non-empty:                {'YES' if all_nonempty  else 'NO'}")
    log(f"  All spot-check norms correct:   {'YES' if all_spotok    else 'NO'}")

    log()
    log("TRAIN CACHE")
    for source in ("source1", "source2", "source3"):
        vr  = val_results[("train", source)]
        br  = build_results[("train", source)]
        log(f"  {source}: {_fmt(vr['parquet_rows'])} rows  "
            f"build={_fmt(br.get('elapsed_s'), 1)}s  "
            f"PQ={_fmt(pq_sizes.get(('train',source)),1)} MB  "
            f"peak_RAM={_fmt(br.get('peak_ram_mb'),1)} MB  "
            f"valid={'PASS' if vr['schema_valid'] and vr['row_count_match'] else 'FAIL'}")

    log()
    log("TEST CACHE")
    for source in ("source1", "source2", "source3"):
        vr  = val_results[("test", source)]
        br  = build_results[("test", source)]
        log(f"  {source}: {_fmt(vr['parquet_rows'])} rows  "
            f"build={_fmt(br.get('elapsed_s'), 1)}s  "
            f"PQ={_fmt(pq_sizes.get(('test',source)),1)} MB  "
            f"peak_RAM={_fmt(br.get('peak_ram_mb'),1)} MB  "
            f"valid={'PASS' if vr['schema_valid'] and vr['row_count_match'] else 'FAIL'}")

    log()
    log("CHUNKING COMPARISON (train_source1)")
    log(f"  Non-chunked:  {S1_NC_ELAPSED:.1f}s  peak RAM = {S1_NC_PEAK_MB:.1f} MB")
    log(f"  Chunked={CHUNKSIZE:,}: {ck['elapsed_s']:.1f}s  "
        f"peak RAM = {ck['peak_ram_mb']:.1f} MB")
    log(f"  Time diff:  {time_diff:+.1f}s  "
        f"({'slower' if time_diff > 0 else 'faster'} due to chunk overhead)")
    log(f"  RAM diff:   {ram_diff:+.1f} MB  "
        f"({'reduction' if ram_diff < 0 else 'increase'})")
    log(f"  Content:    rows={'MATCH' if rows_match else 'MISMATCH'}  "
        f"name_norm={'MATCH' if name_match else 'MISMATCH'}  "
        f"address_norm={'MATCH' if addr_match else 'MISMATCH'}")

    log()
    log("MEMORY OBSERVATIONS")
    log(f"  train_source1 (non-chunked): peak = {S1_NC_PEAK_MB:.1f} MB  "
        f"(2,206,821 rows, 200.3 MB TSV)")
    log(f"  train_source1 (chunked={CHUNKSIZE:,}): peak = {ck['peak_ram_mb']:.1f} MB")
    log(f"  Estimated train_source3 non-chunked: ~5,500 MB "
        f"(scaled from source1; not measured to avoid OOM)")
    log(f"  Chunked build recommended for source2/source3 "
        f"(5M+ rows) on machines with <8 GB RAM.")

    log()
    log("CACHE LOAD TIMINGS")
    for (split, source), t in load_times.items():
        log(f"  {split}_{source}: {t:.3f}s")
    log(f"  Average: {avg_load:.3f}s")

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

    log()
    log("FILES MODIFIED")
    log("  output/phase7_report.txt           (this report)")
    log("  cache/train_source1.parquet        (built non-chunked)")
    log("  cache/train_source2.parquet        (built chunked)")
    log("  cache/train_source3.parquet        (built chunked)")
    log("  cache/test_source1.parquet         (built chunked)")
    log("  cache/test_source2.parquet         (built chunked)")
    log("  cache/test_source3.parquet         (built chunked)")
    log("  cache_chunked/train_source1.parquet (chunked comparison)")

    report = buf.getvalue()
    OUT_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport written -> {OUT_FILE}")

    return {
        "val_results": val_results,
        "build_results": build_results,
        "pq_sizes": pq_sizes,
        "load_times": load_times,
        "ck": ck,
        "rows_match": rows_match,
        "name_match": name_match,
        "addr_match": addr_match,
    }


if __name__ == "__main__":
    main()
