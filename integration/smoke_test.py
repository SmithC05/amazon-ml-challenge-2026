"""
integration/smoke_test.py
=========================
PART 3 — Integration Smoke Test

Verifies the integration environment WITHOUT running the full pipeline.
Clearly distinguishes: READY / SAMPLE_INCOMPLETE / FAIL

Does NOT require:
  - Full candidate generation
  - Full model training
  - Full prediction

Usage:
    python integration/smoke_test.py
"""

import sys
import os
import csv
from pathlib import Path

# ── Bootstrap path so we can import local_paths and src ──────────────────────
INTEGRATION_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = INTEGRATION_DIR.parent
sys.path.insert(0, str(REPO_ROOT_DEFAULT))
sys.path.insert(0, str(REPO_ROOT_DEFAULT / "src"))

try:
    from integration.local_paths import (
        REPO_ROOT, TRAIN_DATA_DIR, TEST_DATA_DIR, CACHE_DIR,
        TRAIN_GT_PATH, CANDIDATE_PATH, MATCHING_PATH, MODEL_DIR,
        TEST_S1_PATH, EXPECTED_ROWS, CACHE_FILES,
        MATCHING_HEADER, CANDIDATE_HEADER, M2_REQUIRED_COLUMNS,
    )
except ImportError:
    # Fallback if run from repo root
    sys.path.insert(0, str(INTEGRATION_DIR))
    from local_paths import (
        REPO_ROOT, TRAIN_DATA_DIR, TEST_DATA_DIR, CACHE_DIR,
        TRAIN_GT_PATH, CANDIDATE_PATH, MATCHING_PATH, MODEL_DIR,
        TEST_S1_PATH, EXPECTED_ROWS, CACHE_FILES,
        MATCHING_HEADER, CANDIDATE_HEADER, M2_REQUIRED_COLUMNS,
    )

from src.cache import validate_cache, load_cache

SEP = "=" * 70
sep = "-" * 50

PASS    = "PASS"
FAIL    = "FAIL"
WARN    = "WARN"
SKIP    = "SKIP"

issues  = []
warns   = []
results = []

def ok(label, note=""):
    results.append((PASS, label, note))

def fail(label, note=""):
    issues.append((FAIL, label, note))
    results.append((FAIL, label, note))

def warn(label, note=""):
    warns.append((WARN, label, note))
    results.append((WARN, label, note))

def skip(label, note=""):
    results.append((SKIP, label, note))


print(SEP)
print("INTEGRATION SMOKE TEST")
print(SEP)
print()

# ── CHECK 1: Repository paths exist ──────────────────────────────────────────
print("CHECK 1 — Repository paths")
print(sep)
for name, path in [
    ("REPO_ROOT",       REPO_ROOT),
    ("TRAIN_DATA_DIR",  TRAIN_DATA_DIR),
    ("TEST_DATA_DIR",   TEST_DATA_DIR),
    ("CACHE_DIR",       CACHE_DIR),
    ("OUTPUT_DIR",      MATCHING_PATH.parent),
    ("MODEL_DIR",       MODEL_DIR),
    ("integration/",    INTEGRATION_DIR),
]:
    if path.exists():
        ok(f"Path exists: {name}", str(path))
        print(f"  ✓ {name}")
    else:
        fail(f"Path missing: {name}", str(path))
        print(f"  ✗ {name}  MISSING: {path}")

# ── CHECK 2: All six M2 caches exist ─────────────────────────────────────────
print()
print("CHECK 2 — M2 cache files exist")
print(sep)
for key, p in CACHE_FILES.items():
    if p.exists() and p.stat().st_size > 1000:
        ok(f"Cache exists: {key}")
        print(f"  ✓ {key}  ({p.stat().st_size/1e6:.1f} MB)")
    else:
        fail(f"Cache missing/empty: {key}", str(p))
        print(f"  ✗ {key}  MISSING or empty: {p}")

# ── CHECK 3: validate_cache() all six ────────────────────────────────────────
print()
print("CHECK 3 — validate_cache() all six")
print(sep)
for split in ("train", "test"):
    for source in ("source1", "source2", "source3"):
        key = f"{split}_{source}"
        v = validate_cache(split, source, CACHE_DIR)
        if v["valid"]:
            rows = v["rows"] or 0
            exp  = EXPECTED_ROWS.get(key)
            row_note = ""
            if exp and rows != exp:
                row_note = f"  ⚠ expected {exp:,}"
                warn(f"Row count mismatch: {key}", f"got {rows:,}, expected {exp:,}")
            print(f"  ✓ {key}  {rows:,} rows{row_note}")
            ok(f"validate_cache PASS: {key}", f"{rows:,} rows")
        else:
            fail(f"validate_cache FAIL: {key}", str(v.get("error", v["checks"])))
            print(f"  ✗ {key}  FAIL — {v.get('error', '')}")
            for chk, ck_ok in v["checks"].items():
                if not ck_ok:
                    print(f"       failed check: {chk}")

# ── CHECK 4 & 5: Load each cache individually; check required columns ─────────
print()
print("CHECK 4/5 — Individual load + column check")
print(sep)
for split in ("train", "test"):
    for source in ("source1", "source2", "source3"):
        key = f"{split}_{source}"
        try:
            df = load_cache(split, source, CACHE_DIR,
                            columns=list(M2_REQUIRED_COLUMNS))
            missing_cols = M2_REQUIRED_COLUMNS - set(df.columns)
            if missing_cols:
                fail(f"Missing columns in {key}", str(sorted(missing_cols)))
                print(f"  ✗ {key}  missing columns: {sorted(missing_cols)}")
            else:
                ok(f"Load OK + all columns: {key}")
                print(f"  ✓ {key}  loaded {len(df):,} rows, all 14 columns present")
            del df
            import gc; gc.collect()
        except Exception as e:
            fail(f"Load FAILED: {key}", str(e))
            print(f"  ✗ {key}  LOAD ERROR: {e}")

# ── CHECK 6: entity_id non-empty strings ─────────────────────────────────────
print()
print("CHECK 6 — entity_id string/non-empty spot-check")
print(sep)
for split in ("train", "test"):
    for source in ("source1", "source2", "source3"):
        key = f"{split}_{source}"
        try:
            df = load_cache(split, source, CACHE_DIR, columns=["entity_id"])
            bad_null  = df["entity_id"].isna().sum()
            bad_empty = (df["entity_id"].astype(str).str.strip() == "").sum()
            bad_type  = (df["entity_id"].apply(lambda x: not isinstance(x, str))).sum()
            if bad_null + bad_empty + bad_type == 0:
                ok(f"entity_id OK: {key}")
                print(f"  ✓ {key}  entity_id: all non-null, non-empty strings")
            else:
                fail(f"entity_id issues: {key}",
                     f"null={bad_null} empty={bad_empty} non-str={bad_type}")
                print(f"  ✗ {key}  entity_id: null={bad_null} empty={bad_empty} non-str={bad_type}")
            del df
            import gc; gc.collect()
        except Exception as e:
            fail(f"entity_id check FAILED: {key}", str(e))
            print(f"  ✗ {key}  ERROR: {e}")

# ── CHECK 7: Train GT exists ──────────────────────────────────────────────────
print()
print("CHECK 7 — Train ground truth")
print(sep)
if TRAIN_GT_PATH.exists() and TRAIN_GT_PATH.stat().st_size > 0:
    ok("Train GT exists", str(TRAIN_GT_PATH))
    print(f"  ✓ {TRAIN_GT_PATH.name}  ({TRAIN_GT_PATH.stat().st_size/1e6:.1f} MB)")
else:
    fail("Train GT missing", str(TRAIN_GT_PATH))
    print(f"  ✗ train_ground_truth.tsv MISSING: {TRAIN_GT_PATH}")

# ── CHECK 8: Test S1 exists ───────────────────────────────────────────────────
print()
print("CHECK 8 — Test Source-1 TSV")
print(sep)
if TEST_S1_PATH.exists() and TEST_S1_PATH.stat().st_size > 0:
    ok("test_source1.tsv exists", str(TEST_S1_PATH))
    print(f"  ✓ test_source1.tsv  ({TEST_S1_PATH.stat().st_size/1e6:.1f} MB)")
else:
    fail("test_source1.tsv missing", str(TEST_S1_PATH))
    print(f"  ✗ test_source1.tsv MISSING")

# ── CHECK 9: Output directories exist ────────────────────────────────────────
print()
print("CHECK 9 — Output directories")
print(sep)
for dname, dpath in [("output/", MATCHING_PATH.parent), ("models/", MODEL_DIR),
                     ("integration/", INTEGRATION_DIR)]:
    if dpath.exists():
        ok(f"Dir exists: {dname}")
        print(f"  ✓ {dname}")
    else:
        fail(f"Dir missing: {dname}", str(dpath))
        print(f"  ✗ {dname}  MISSING")

# ── CHECK 10 & 11: candidate_pairs.tsv header + completeness ─────────────────
print()
print("CHECK 10/11 — candidate_pairs.tsv (if present)")
print(sep)

CANDIDATE_STATUS = "ABSENT"
if CANDIDATE_PATH.exists():
    # Check header
    with open(CANDIDATE_PATH, encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
    cols = header_line.split("\t")
    if cols == CANDIDATE_HEADER:
        ok("candidate_pairs.tsv header correct")
        print(f"  ✓ Header correct: {CANDIDATE_HEADER}")
    else:
        fail("candidate_pairs.tsv header wrong", f"got {cols}")
        print(f"  ✗ Header wrong: got {cols}, expected {CANDIDATE_HEADER}")

    # Check completeness: count S1 IDs in candidate file vs test S1
    # Stream both files to avoid memory issues
    if TEST_S1_PATH.exists():
        test_s1_ids = set()
        with open(TEST_S1_PATH, encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                s = line.split("\t", 1)[0].strip()
                if s:
                    test_s1_ids.add(s)

        cand_s1_ids = set()
        with open(CANDIDATE_PATH, encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                s = line.split("\t", 1)[0].strip()
                if s:
                    cand_s1_ids.add(s)

        n_test   = len(test_s1_ids)
        n_cand   = len(cand_s1_ids)
        missing  = test_s1_ids - cand_s1_ids
        extra    = cand_s1_ids - test_s1_ids

        if missing:
            CANDIDATE_STATUS = "SAMPLE_INCOMPLETE"
            warn(
                "candidate_pairs.tsv SAMPLE/INCOMPLETE",
                f"covers {n_cand:,}/{n_test:,} test S1 entities; "
                f"{len(missing):,} missing"
            )
            print(f"  ⚠ SAMPLE/INCOMPLETE: covers {n_cand:,}/{n_test:,} test S1 entities")
            print(f"    Missing {len(missing):,} S1 entities (e.g. {sorted(missing)[:3]})")
        elif extra:
            CANDIDATE_STATUS = "SAMPLE_INCOMPLETE"
            warn(
                "candidate_pairs.tsv has extra S1 IDs not in test set",
                f"{len(extra):,} extra IDs"
            )
            print(f"  ⚠ Extra S1 IDs not in test set: {len(extra):,}")
        else:
            CANDIDATE_STATUS = "COMPLETE"
            ok("candidate_pairs.tsv COMPLETE — covers all test S1 entities")
            print(f"  ✓ COMPLETE: covers all {n_test:,} test S1 entities")

        print(f"    File: {CANDIDATE_PATH}  ({CANDIDATE_PATH.stat().st_size/1e6:.2f} MB)")
        del test_s1_ids, cand_s1_ids
else:
    CANDIDATE_STATUS = "ABSENT"
    warn("candidate_pairs.tsv not present", str(CANDIDATE_PATH))
    print(f"  ⚠ ABSENT: {CANDIDATE_PATH}")
    print("    M4 must deliver this file before M3 can run.")

# ── MATCHING FILE STATUS ───────────────────────────────────────────────────────
print()
print("CHECK 12 — matching_results.tsv (if present)")
print(sep)
if MATCHING_PATH.exists():
    with open(MATCHING_PATH, encoding="utf-8") as f:
        mhdr = f.readline().rstrip("\n").split("\t")
    if mhdr == MATCHING_HEADER:
        ok("matching_results.tsv header correct")
        print(f"  ✓ Header correct (file present but may be sample)")
    else:
        fail("matching_results.tsv header wrong", str(mhdr))
        print(f"  ✗ Header wrong: {mhdr}")
else:
    skip("matching_results.tsv not present yet (M3 not run)")
    print(f"  ⚪ ABSENT (expected — M3 not yet run): {MATCHING_PATH}")

# ── FINAL REPORT ─────────────────────────────────────────────────────────────
print()
print(SEP)
print("SMOKE TEST RESULTS")
print(SEP)

n_fail = len(issues)
n_warn = len(warns)

print(f"\n  {'Check':<45}  {'Status'}")
print("  " + "-" * 58)
for status, label, note in results:
    marker = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "SKIP": "⚪"}.get(status, "?")
    print(f"  {marker} {label[:44]:<44}  {status}")

print()
print(f"  Total PASS : {sum(1 for s,_,_ in results if s==PASS)}")
print(f"  Total WARN : {n_warn}")
print(f"  Total FAIL : {n_fail}")
print(f"  Total SKIP : {sum(1 for s,_,_ in results if s==SKIP)}")
print()
print(f"  Candidate file status: {CANDIDATE_STATUS}")
print()

print(SEP)
if n_fail > 0:
    print("INTEGRATION STATUS: FAIL")
    for _, label, note in issues:
        print(f"  ✗ {label}" + (f"  ({note})" if note else ""))
elif CANDIDATE_STATUS == "SAMPLE_INCOMPLETE":
    print("INTEGRATION STATUS: READY (M2 complete; candidate file is SAMPLE/INCOMPLETE — awaiting M4)")
elif CANDIDATE_STATUS == "ABSENT":
    print("INTEGRATION STATUS: READY (M2 complete; awaiting M4 candidate_pairs.tsv)")
else:
    print("INTEGRATION STATUS: READY")

print()
print("Pipeline component status:")
print(f"  M2 CACHE  : {'READY' if n_fail == 0 else 'ISSUES'}")
print(f"  M4 OUTPUT : {CANDIDATE_STATUS}")
print(f"  M3 OUTPUT : {'PRESENT' if MATCHING_PATH.exists() else 'NOT YET RUN'}")
print(SEP)

sys.exit(1 if n_fail > 0 else 0)
