"""
Phase 1 cache verification script — read-only, no modifications.
Verifies all 6 M2 caches one at a time.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from cache import _cache_path, _OUTPUT_COLUMNS, _REQUIRED_COLUMNS
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
CACHE_CANDIDATES = [
    pathlib.Path("cache"),
    pathlib.Path("dataset/processed/m2_cache"),
    pathlib.Path("cache_chunked"),
]

TRAIN_DIR = pathlib.Path("D:/6ab10eb3b23ba_student_resource/student_resource/dataset/train")
TEST_DIR  = pathlib.Path("D:/6ab10eb3b23ba_student_resource/student_resource/dataset/test")

SOURCES = [
    ("train", "source1", TRAIN_DIR, "train_source1.tsv"),
    ("train", "source2", TRAIN_DIR, "train_source2.tsv"),
    ("train", "source3", TRAIN_DIR, "train_source3.tsv"),
    ("test",  "source1", TEST_DIR,  "test_source1.tsv"),
    ("test",  "source2", TEST_DIR,  "test_source2.tsv"),
    ("test",  "source3", TEST_DIR,  "test_source3.tsv"),
]

EXPECTED_COLUMNS = list(_OUTPUT_COLUMNS)
EXPECTED_PREFIX  = {"source1": "S1-", "source2": "S2-", "source3": "S3-"}

# ── Helpers ──────────────────────────────────────────────────────────────────
def tsv_row_count(path):
    """Count rows by byte-level newline scan (memory-efficient)."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1  # subtract header


def find_cache(split, source):
    """Find the first existing cache file across candidates."""
    for base in CACHE_CANDIDATES:
        p = _cache_path(split, source, base)
        if p.exists():
            return p
    return None


def verify_one(split, source, tsv_dir, tsv_name):
    """Verify a single cache. Returns a result dict."""
    r = {
        "split": split, "source": source,
        "cache_path": None, "cache_size_mb": None,
        "tsv_rows": None, "cache_rows": None,
        "row_count_match": False,
        "readable": False,
        "schema_ok": False,
        "schema_order_ok": False,
        "entity_id_ok": False,
        "entity_id_prefix_ok": False,
        "invalid_entity_ids": None,
        "norm_fields_ok": False,
        "raw_fields_ok": False,
        "consistency_ok": False,
        "errors": [],
    }

    # 1. File existence
    p = find_cache(split, source)
    if p is None:
        r["errors"].append("Cache file NOT FOUND in any candidate directory")
        return r
    r["cache_path"] = str(p)
    r["cache_size_mb"] = round(p.stat().st_size / 1e6, 1)

    # 2. TSV row count
    tsv_path = tsv_dir / tsv_name
    if tsv_path.exists():
        try:
            r["tsv_rows"] = tsv_row_count(tsv_path)
        except Exception as e:
            r["errors"].append(f"TSV read error: {e}")
    else:
        r["errors"].append(f"TSV not found: {tsv_path}")

    # 3. Parquet readability + schema
    try:
        # Read only metadata first (fast, no full load)
        import pyarrow.parquet as pq
        pf = pq.read_table(p, memory_map=True)
        r["readable"] = True
        r["cache_rows"] = len(pf)

        cols_actual = pf.schema.names

        # Schema check: all 14 required columns present
        missing = set(EXPECTED_COLUMNS) - set(cols_actual)
        r["schema_ok"] = len(missing) == 0
        if missing:
            r["errors"].append(f"Missing columns: {sorted(missing)}")

        # Schema order check
        r["schema_order_ok"] = (cols_actual == EXPECTED_COLUMNS)
        if not r["schema_order_ok"] and r["schema_ok"]:
            r["errors"].append(f"Column order wrong: {cols_actual}")

        # 4. Row count match
        if r["tsv_rows"] is not None:
            r["row_count_match"] = (r["tsv_rows"] == r["cache_rows"])
            if not r["row_count_match"]:
                r["errors"].append(
                    f"Row count mismatch: TSV={r['tsv_rows']:,} vs PQ={r['cache_rows']:,}")

        # 5. Entity ID check (load only entity_id column)
        eid_col = pf.column("entity_id").to_pylist()
        empty_ids = sum(1 for x in eid_col if not x)
        r["entity_id_ok"] = (empty_ids == 0)
        if empty_ids:
            r["errors"].append(f"Empty entity_ids: {empty_ids}")

        expected_pfx = EXPECTED_PREFIX[source]
        bad_prefix = sum(1 for x in eid_col if x and not str(x).startswith(expected_pfx))
        r["entity_id_prefix_ok"] = (bad_prefix == 0)
        r["invalid_entity_ids"] = bad_prefix
        if bad_prefix:
            r["errors"].append(f"Wrong entity_id prefix (expected {expected_pfx}): {bad_prefix}")

        # 6. Normalized fields presence + type spot-check (5 rows)
        norm_cols = ["name_norm", "address_norm", "name_tokens", "address_tokens",
                     "name_token_count", "address_token_count",
                     "name_length", "address_length", "name_digits", "address_digits"]
        norm_present = all(c in cols_actual for c in norm_cols)
        r["norm_fields_ok"] = norm_present
        if not norm_present:
            missing_norm = [c for c in norm_cols if c not in cols_actual]
            r["errors"].append(f"Missing norm fields: {missing_norm}")

        # 7. Raw fields
        raw_cols = ["entity_id", "business_name", "business_address", "country"]
        raw_present = all(c in cols_actual for c in raw_cols)
        r["raw_fields_ok"] = raw_present
        if not raw_present:
            r["errors"].append(f"Missing raw fields: {[c for c in raw_cols if c not in cols_actual]}")

        # 8. Consistency: schema matches expected exactly
        r["consistency_ok"] = r["schema_ok"] and r["schema_order_ok"]

        del pf, eid_col

    except Exception as e:
        r["readable"] = False
        r["errors"].append(f"Parquet read error: {e}")

    return r


# ── Main ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print("PHASE 1 — M2 CACHE VERIFICATION (read-only)")
print("=" * 70)

print("\n=== CACHE DIRECTORY SEARCH ===")
for base in CACHE_CANDIDATES:
    exists = base.exists()
    print(f"  {base.resolve()}  exists={exists}")
    if exists:
        parquets = sorted(base.rglob("*.parquet"))
        for pf in parquets:
            mb = pf.stat().st_size / 1e6
            print(f"    {pf.name}  {mb:.1f} MB")
        if not parquets:
            print("    (empty)")

print("\n=== EXPECTED SCHEMA (14 columns) ===")
for i, col in enumerate(EXPECTED_COLUMNS, 1):
    print(f"  {i:>2}. {col}")

print()
results = []
for split, source, tsv_dir, tsv_name in SOURCES:
    key = f"{split}_{source}"
    print(f"\n{'─'*60}")
    print(f"VERIFYING: {key}")
    print(f"{'─'*60}")
    r = verify_one(split, source, tsv_dir, tsv_name)
    results.append(r)

    print(f"  Cache path    : {r['cache_path'] or 'NOT FOUND'}")
    print(f"  Size          : {r['cache_size_mb']} MB" if r['cache_size_mb'] else "  Size          : N/A")
    print(f"  TSV rows      : {r['tsv_rows']:,}" if r['tsv_rows'] else "  TSV rows      : N/A")
    print(f"  Cache rows    : {r['cache_rows']:,}" if r['cache_rows'] else "  Cache rows    : N/A")
    print(f"  Row count     : {'PASS' if r['row_count_match'] else 'FAIL'}")
    print(f"  Readable      : {'PASS' if r['readable'] else 'FAIL'}")
    print(f"  Schema        : {'PASS' if r['schema_ok'] else 'FAIL'}")
    print(f"  Schema order  : {'PASS' if r['schema_order_ok'] else 'FAIL'}")
    print(f"  entity_id     : {'PASS' if r['entity_id_ok'] else 'FAIL'}")
    print(f"  Prefix check  : {'PASS' if r['entity_id_prefix_ok'] else 'FAIL'} "
          f"(bad={r['invalid_entity_ids']})")
    print(f"  Norm fields   : {'PASS' if r['norm_fields_ok'] else 'FAIL'}")
    print(f"  Raw fields    : {'PASS' if r['raw_fields_ok'] else 'FAIL'}")
    print(f"  Consistency   : {'PASS' if r['consistency_ok'] else 'FAIL'}")
    if r["errors"]:
        for e in r["errors"]:
            print(f"  ERROR: {e}")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("VERIFICATION TABLE")
print("=" * 70)
header = (
    f"{'Cache':<18} {'TSV Rows':>12} {'PQ Rows':>12} "
    f"{'RowCnt':>7} {'Schema':>7} {'Read':>5} "
    f"{'EntID':>6} {'NormF':>6} {'Consis':>7}"
)
print(header)
print("-" * len(header))
for r in results:
    key = f"{r['split']}_{r['source']}"
    p = lambda b: "PASS" if b else "FAIL"
    print(
        f"{key:<18} "
        f"{r['tsv_rows'] or 0:>12,} "
        f"{r['cache_rows'] or 0:>12,} "
        f"{p(r['row_count_match']):>7} "
        f"{p(r['schema_ok']):>7} "
        f"{p(r['readable']):>5} "
        f"{p(r['entity_id_ok']):>6} "
        f"{p(r['norm_fields_ok']):>6} "
        f"{p(r['consistency_ok']):>7}"
    )

# ── PHASE 1 SUMMARY ──────────────────────────────────────────────────────────
print()
print("=" * 70)
print("PHASE 1 CACHE VERIFICATION SUMMARY")
print("=" * 70)
total_pass = 0
for r in results:
    key = f"{r['split']}_{r['source']}"
    all_ok = (r["readable"] and r["row_count_match"] and r["schema_ok"]
              and r["schema_order_ok"] and r["entity_id_ok"]
              and r["norm_fields_ok"] and r["raw_fields_ok"])
    mark = "✅" if all_ok else "❌"
    if all_ok:
        total_pass += 1
    print(f"  {key:<18}: {mark}")

print()
print(f"Total caches verified: {total_pass}/6")

errors_found = [e for r in results for e in r["errors"]]
print()
print("Errors found:")
if errors_found:
    for e in errors_found:
        print(f"  - {e}")
else:
    print("  NONE")

print()
print("Exact cache path(s):")
seen = set()
for r in results:
    if r["cache_path"] and r["cache_path"] not in seen:
        seen.add(r["cache_path"])
        print(f"  {r['cache_path']}")

print()
print("Files modified: NONE")
print("Phase 1 complete.")
