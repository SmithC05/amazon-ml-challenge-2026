"""
tools/phase5_build_all_caches.py
=================================
Phase 5 — Build all 6 competition caches into dataset/processed/m2_cache/
Uses the finalized M2 implementation only:
  - src/preprocess.py  (canonical normalization, unchanged)
  - src/cache.py       (chunked Parquet writer with Phase 2 schema fix)

No normalization changes. No new columns. No M3/M4 changes.
Run from repo root:
    python tools/phase5_build_all_caches.py
"""
import sys, time, shutil, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from cache import build_cache, validate_cache, cache_exists, _OUTPUT_COLUMNS

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR  = pathlib.Path("dataset/processed/m2_cache")
TRAIN_DIR  = pathlib.Path("D:/6ab10eb3b23ba_student_resource/student_resource/dataset/train")
TEST_DIR   = pathlib.Path("D:/6ab10eb3b23ba_student_resource/student_resource/dataset/test")
OLD_CACHE  = pathlib.Path("cache")   # train_source1 already exists here

SOURCES = [
    ("train", "source1", TRAIN_DIR),
    ("train", "source2", TRAIN_DIR),
    ("train", "source3", TRAIN_DIR),
    ("test",  "source1", TEST_DIR),
    ("test",  "source2", TEST_DIR),
    ("test",  "source3", TEST_DIR),
]

EXPECTED_ROWS = {
    "train_source1": 2_206_821,
    "train_source2": 5_034_616,
    "train_source3": 5_285_603,
    "test_source1":  1_732_544,
    "test_source2":  4_887_273,
    "test_source3":  5_082_316,
}

CHUNKSIZE = 500_000   # memory-safe for 5M-row sources

# ── Setup ─────────────────────────────────────────────────────────────────────
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Cache directory: {CACHE_DIR.resolve()}")
print()

# ── Step 1: Copy already-validated train_source1 if not yet in m2_cache ──────
src1_target = CACHE_DIR / "train_source1.parquet"
src1_origin = OLD_CACHE / "train_source1.parquet"

if not src1_target.exists():
    if src1_origin.exists():
        print(f"[COPY] train_source1: {src1_origin} → {src1_target}")
        t0 = time.perf_counter()
        shutil.copy2(src1_origin, src1_target)
        print(f"       done in {time.perf_counter()-t0:.1f}s  "
              f"({src1_target.stat().st_size/1e6:.1f} MB)")
    else:
        print("[WARN] cache/train_source1.parquet not found — will build from TSV")
else:
    print(f"[SKIP] train_source1: already exists in m2_cache")
print()

# ── Step 2: Build remaining caches (chunked, memory-safe) ────────────────────
print("=" * 68)
print(f"BUILDING CACHES (chunksize={CHUNKSIZE:,}) into {CACHE_DIR}")
print("=" * 68)

results = {}
for split, source, data_dir in SOURCES:
    key = f"{split}_{source}"
    already = cache_exists(split, source, CACHE_DIR)
    if already:
        print(f"[SKIP] {key}: already exists")
        results[key] = "skipped"
        continue

    print(f"[BUILD] {key} ...", flush=True)
    t0 = time.perf_counter()
    r = build_cache(
        split     = split,
        source    = source,
        data_dir  = str(data_dir),
        cache_dir = str(CACHE_DIR),
        force     = False,
        chunksize = CHUNKSIZE,
    )
    elapsed = time.perf_counter() - t0
    if r["status"] == "built":
        mb = pathlib.Path(r["cache_path"]).stat().st_size / 1e6
        rate = r["rows"] / elapsed
        print(f"  OK   {r['rows']:>10,} rows  {elapsed:>7.1f}s  "
              f"{rate:>6.0f} rows/s  {mb:.1f} MB")
        results[key] = "built"
    else:
        print(f"  ERR  status={r['status']}  error={r['error']}")
        results[key] = f"ERROR: {r['error']}"

print()

# ── Step 3: Validate all 6 caches ────────────────────────────────────────────
print("=" * 68)
print("VALIDATING ALL 6 CACHES")
print("=" * 68)

def tsv_row_count(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1

val_results = {}
for split, source, data_dir in SOURCES:
    key  = f"{split}_{source}"
    tsv_name = f"{split}_{source}.tsv"
    tsv  = data_dir / tsv_name
    print(f"\n  [{key}]")

    vr = validate_cache(split, source, CACHE_DIR)
    pq_rows = vr["rows"] or 0
    pq_cols = vr["columns"] or []

    # TSV row count (binary, fast)
    if tsv.exists():
        tsv_rows = tsv_row_count(tsv)
    else:
        tsv_rows = None
        print(f"    WARN: TSV not found at {tsv}")

    exp_rows = EXPECTED_ROWS[key]
    row_match  = (pq_rows == exp_rows) if vr["valid"] else False
    schema_ok  = (pq_cols == _OUTPUT_COLUMNS) if vr["valid"] else False

    print(f"    valid:        {vr['valid']}")
    print(f"    expected rows:{exp_rows:>12,}")
    print(f"    parquet rows: {pq_rows:>12,}  {'MATCH' if row_match else 'MISMATCH'}")
    if tsv_rows is not None:
        print(f"    tsv rows:     {tsv_rows:>12,}  {'MATCH' if tsv_rows == pq_rows else 'MISMATCH'}")
    print(f"    schema(14):   {'PASS' if schema_ok else 'FAIL'}")
    if vr["error"]:
        print(f"    error:        {vr['error']}")
    for ck, ok in vr.get("checks", {}).items():
        if not ok:
            print(f"    FAIL check:   {ck}")

    val_results[key] = {
        "valid": vr["valid"],
        "pq_rows": pq_rows,
        "tsv_rows": tsv_rows,
        "exp_rows": exp_rows,
        "row_match": row_match,
        "schema_ok": schema_ok,
        "error": vr["error"],
    }

# ── Step 4: Summary table ─────────────────────────────────────────────────────
print()
print("=" * 68)
print("FINAL 6-CACHE STATUS TABLE")
print("=" * 68)
hdr = f"{'Cache':<18} {'ExpRows':>10} {'PQRows':>10} {'RowMatch':>9} {'Schema':>7} {'Valid':>6}"
print(hdr)
print("-" * len(hdr))
all_ok = True
for key, v in val_results.items():
    rm = "PASS" if v["row_match"] else "FAIL"
    sc = "PASS" if v["schema_ok"] else "FAIL"
    vd = "PASS" if v["valid"] else "FAIL"
    if not v["valid"]:
        all_ok = False
    print(f"{key:<18} {v['exp_rows']:>10,} {v['pq_rows']:>10,} "
          f"{rm:>9} {sc:>7} {vd:>6}")

print()
print("=" * 68)
print("SUMMARY")
print("=" * 68)
total_v   = sum(1 for v in val_results.values() if v["valid"])
total_fail = 6 - total_v
print(f"  Caches verified: {total_v}/6")
print(f"  PASS: {total_v}   FAIL: {total_fail}")
print(f"  All OK: {all_ok}")
print()
print(f"  Cache directory: {CACHE_DIR.resolve()}")
if all_ok:
    print()
    print("  ALL 6 CACHES: READY")
else:
    print()
    for key, v in val_results.items():
        if not v["valid"]:
            print(f"  FAIL: {key}  error={v['error']}")
