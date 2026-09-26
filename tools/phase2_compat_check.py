"""
Phase 2 — backward compatibility check.
Lightweight: uses only in-memory synthetic data.
No large TSV reads. No cache rebuilds.
Read-only except for a tiny temp file that is cleaned up.
"""
import sys, pathlib, tempfile, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from preprocess import (
    normalize_text, normalize_name, normalize_address,
    preprocess_dataframe, _OUTPUT_COLUMNS,
)
from cache import (
    build_cache, load_cache, load_all_cache,
    cache_exists, validate_cache, _cache_path,
)

RESULTS = {}

# ── Helper ────────────────────────────────────────────────────────────────────
def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    RESULTS[name] = status
    mark = "✓" if condition else "✗"
    print(f"  [{mark}] {name}: {status}" + (f"  ({detail})" if detail else ""))
    return condition

# ─────────────────────────────────────────────────────────────────────────────
# 1. preprocess.py canonical normalization
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== preprocess.py ===")
check("normalize_text None",       normalize_text(None) == "")
check("normalize_text NaN",        normalize_text(float("nan")) == "")
check("normalize_text empty",      normalize_text("") == "")
check("normalize_text punctuation",normalize_text("ABC, Inc.!") == "abc inc")
check("normalize_text unicode",    normalize_text("Ｈｅｌｌｏ") == "hello")
check("normalize_name pvt ltd",    normalize_name("Pvt Ltd ABC") == "private limited abc")
check("normalize_name ltd",        normalize_name("ABC ltd") == "abc limited")
check("normalize_name inc",        normalize_name("Co Inc") == "co incorporated")
check("normalize_address ave",     normalize_address("123 Ave") == "123 avenue")
check("normalize_address blvd",    normalize_address("Main Blvd") == "main boulevard")
check("normalize_address rd",      normalize_address("Old Rd") == "old road")
check("normalize_address None",    normalize_address(None) == "")

# preprocess_dataframe produces exactly _OUTPUT_COLUMNS
df_in = pd.DataFrame([
    {"entity_id": "S1-001", "business_name": "ABC Pvt Ltd",
     "business_address": "1 Main Ave", "country": "US"},
    {"entity_id": "S1-002", "business_name": None,
     "business_address": None, "country": "IN"},
])
df_out = preprocess_dataframe(df_in)
check("preprocess_dataframe 14 cols",    list(df_out.columns) == _OUTPUT_COLUMNS,
      f"got {list(df_out.columns)}")
check("preprocess_dataframe row count",  len(df_out) == 2)
check("preprocess_dataframe name_norm",  df_out.loc[0, "name_norm"] == "abc private limited")
check("preprocess_dataframe addr_norm",  df_out.loc[0, "address_norm"] == "1 main avenue")
check("preprocess_dataframe null name",  df_out.loc[1, "name_norm"] == "")
check("preprocess_dataframe null addr",  df_out.loc[1, "address_norm"] == "")
check("preprocess_dataframe name_tokens",df_out.loc[0, "name_tokens"] == ["abc", "private", "limited"])
check("preprocess_dataframe digits",     df_out.loc[0, "address_digits"] == 1)
check("input not mutated",               "name_norm" not in df_in.columns)

# ─────────────────────────────────────────────────────────────────────────────
# 2. cache.py API — using a temp dir so nothing touches cache/
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== cache.py API ===")
with tempfile.TemporaryDirectory() as tmp:
    tmp_data  = pathlib.Path(tmp) / "data"
    tmp_cache = pathlib.Path(tmp) / "cache"
    tmp_data.mkdir()
    tmp_cache.mkdir()

    # Write a tiny synthetic TSV
    tsv_path = tmp_data / "train_source1.tsv"
    tsv_path.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tABC Pvt Ltd\t1 Main Ave\tUS\n"
        "S1-002\tXYZ Corp\t5 Oak Rd\tIN\n"
        "S1-003\t\t\tDE\n",
        encoding="utf-8",
    )

    # A. cache_exists — before build
    check("cache_exists before build", not cache_exists("train", "source1", tmp_cache))

    # B. build_cache chunksize=None (whole-file)
    r = build_cache("train", "source1",
                    data_dir=tmp_data, cache_dir=tmp_cache,
                    force=False, chunksize=None)
    check("build_cache None status",    r["status"] == "built",    r["status"])
    check("build_cache None rows",      r["rows"] == 3,            str(r["rows"]))
    check("build_cache None no error",  r["error"] is None,        str(r["error"]))
    cache_nc = pathlib.Path(r["cache_path"])
    check("build_cache None file exists", cache_nc.is_file())

    # C. Idempotency: calling again with force=False → skipped
    r2 = build_cache("train", "source1",
                     data_dir=tmp_data, cache_dir=tmp_cache,
                     force=False, chunksize=None)
    check("build_cache idempotent skip", r2["status"] == "skipped", r2["status"])

    # D. cache_exists — after build
    check("cache_exists after build", cache_exists("train", "source1", tmp_cache))

    # E. load_cache
    df_loaded = load_cache("train", "source1", tmp_cache)
    check("load_cache row count",   len(df_loaded) == 3)
    check("load_cache 14 columns",  list(df_loaded.columns) == _OUTPUT_COLUMNS)
    check("load_cache name_norm",   df_loaded.loc[0, "name_norm"] == "abc private limited")
    check("load_cache addr_norm",   df_loaded.loc[0, "address_norm"] == "1 main avenue")
    check("load_cache null safe",   df_loaded.loc[2, "name_norm"] == "")

    # F. load_cache with columns= subset
    df_sub = load_cache("train", "source1", tmp_cache,
                        columns=["entity_id", "name_norm"])
    check("load_cache columns subset", list(df_sub.columns) == ["entity_id", "name_norm"])

    # G. load_cache raises FileNotFoundError for missing cache
    try:
        load_cache("train", "source2", tmp_cache)
        check("load_cache missing raises", False, "no exception raised")
    except FileNotFoundError:
        check("load_cache missing raises", True)

    # H. validate_cache
    vr = validate_cache("train", "source1", tmp_cache)
    check("validate_cache valid",         vr["valid"])
    check("validate_cache 7 checks",      len(vr["checks"]) == 7)
    check("validate_cache rows",          vr["rows"] == 3)
    check("validate_cache no missing",    vr["missing_columns"] == [])
    check("validate_cache file_exists",   vr["checks"]["file_exists"])
    check("validate_cache readable",      vr["checks"]["parquet_readable"])
    check("validate_cache 14 cols",       vr["checks"]["all_14_columns"])
    check("validate_cache entity_id",     vr["checks"]["entity_id_present"])
    check("validate_cache raw_cols",      vr["checks"]["raw_columns_present"])
    check("validate_cache norm_cols",     vr["checks"]["norm_columns_present"])
    check("validate_cache non_empty",     vr["checks"]["non_empty"])

    # I. validate_cache on missing → valid=False
    vr_miss = validate_cache("train", "source2", tmp_cache)
    check("validate_cache missing valid=False", not vr_miss["valid"])

    # J. build_cache chunksize=1 (one row per chunk — tests incremental writer)
    tmp_cache2 = pathlib.Path(tmp) / "cache_chunked"
    tmp_cache2.mkdir()
    rc = build_cache("train", "source1",
                     data_dir=tmp_data, cache_dir=tmp_cache2,
                     force=False, chunksize=1)
    check("build_cache chunked status",  rc["status"] == "built",  rc["status"])
    check("build_cache chunked rows",    rc["rows"] == 3,          str(rc["rows"]))
    check("build_cache chunked no error",rc["error"] is None,      str(rc["error"]))

    # K. Content equivalence: chunked == non-chunked
    df_ck = load_cache("train", "source1", tmp_cache2)
    check("chunked 14 cols",       list(df_ck.columns) == _OUTPUT_COLUMNS)
    check("chunked row count",     len(df_ck) == len(df_loaded))
    check("chunked name_norm eq",  list(df_ck["name_norm"]) == list(df_loaded["name_norm"]))
    check("chunked addr_norm eq",  list(df_ck["address_norm"]) == list(df_loaded["address_norm"]))
    check("chunked ntoks eq",      list(df_ck["name_token_count"]) == list(df_loaded["name_token_count"]))

    # L. build_cache chunksize=2 (multi-chunk, 2 rows + 1 row)
    tmp_cache3 = pathlib.Path(tmp) / "cache_chunked2"
    tmp_cache3.mkdir()
    rc2 = build_cache("train", "source1",
                      data_dir=tmp_data, cache_dir=tmp_cache3,
                      force=False, chunksize=2)
    df_ck2 = load_cache("train", "source1", tmp_cache3)
    check("chunked2 row count",   len(df_ck2) == 3)
    check("chunked2 content eq",  list(df_ck2["name_norm"]) == list(df_loaded["name_norm"]))

    # M. force=True triggers rebuild
    r_force = build_cache("train", "source1",
                          data_dir=tmp_data, cache_dir=tmp_cache,
                          force=True, chunksize=None)
    check("build_cache force rebuilt", r_force["status"] == "built", r_force["status"])
    check("build_cache force rows",    r_force["rows"] == 3)

    # N. Invalid split raises ValueError
    try:
        cache_exists("invalid", "source1", tmp_cache)
        check("invalid split raises", False)
    except ValueError:
        check("invalid split raises", True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== PHASE 2 BACKWARD COMPATIBILITY SUMMARY ===")
passed = sum(1 for v in RESULTS.values() if v == "PASS")
failed = sum(1 for v in RESULTS.values() if v == "FAIL")
print(f"  Total: {passed + failed}  PASS: {passed}  FAIL: {failed}")
if failed:
    print("  FAILING CHECKS:")
    for k, v in RESULTS.items():
        if v == "FAIL":
            print(f"    - {k}")
else:
    print("  ALL CHECKS PASS")
