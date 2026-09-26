"""
integration/final_pre_submission_check.py
==========================================
PART 6 — Final Pre-Submission Check Script (Memory-Conscious)

Checks all requirements before claiming submission ready.
Streams files — no giant pandas loads for TSVs.

Checks:
  A. matching_results.tsv exists
  B. candidate_pairs.tsv exists
  C. Matching header exactly correct
  D. Candidate header exactly correct
  E. Every test S1 entity appears exactly once in matching
  F. No duplicate IDs within matched/candidate lists
  G. No S1 IDs appear in matched/candidate ID lists
  H. IDs have S2-/S3- prefixes
  I. Matched IDs are a subset of candidate IDs (per S1)
  J. No unexpected S1 IDs appear
  K. Official validator passes

Usage:
    python integration/final_pre_submission_check.py
"""

import sys
import os
import subprocess
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parent
REPO_ROOT_DEFAULT = INTEGRATION_DIR.parent
sys.path.insert(0, str(REPO_ROOT_DEFAULT))

try:
    from integration.local_paths import (
        REPO_ROOT, TEST_DATA_DIR, CANDIDATE_PATH, MATCHING_PATH,
        TEST_S1_PATH, MATCHING_HEADER, CANDIDATE_HEADER,
    )
except ImportError:
    sys.path.insert(0, str(INTEGRATION_DIR))
    from local_paths import (
        REPO_ROOT, TEST_DATA_DIR, CANDIDATE_PATH, MATCHING_PATH,
        TEST_S1_PATH, MATCHING_HEADER, CANDIDATE_HEADER,
    )

SEP = "=" * 70
sep = "-" * 50
DELIM = "\t"

errors = []
warnings = []

def err(msg):
    errors.append(msg)
    print(f"  ✗ {msg}")

def warn(msg):
    warnings.append(msg)
    print(f"  ⚠ {msg}")

def ok(msg):
    print(f"  ✓ {msg}")


def stream_ids_from_col1(path: Path) -> set:
    """Stream first-column IDs from a TSV, skipping header. Memory-light."""
    ids = set()
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            s = line.split(DELIM, 1)[0].strip()
            if s:
                ids.add(s)
    return ids


def read_header(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return f.readline().rstrip("\n").split(DELIM)


def validate_id_list_file_stream(
    path: Path,
    expected_header: list,
    col_label: str,
    required_s1: set,
) -> tuple[bool, dict]:
    """
    Stream-validate a matching/candidate TSV.
    Returns (ok, {s1_id: set_of_ids}) — only keeps ID sets in memory.
    For huge files this is the memory-safe approach vs pandas.
    """
    mapping = {}   # s1 → frozenset of matched/candidate ids
    seen_s1        = set()
    dup_s1         = set()
    intra_dupes    = set()
    self_matches   = set()
    wrong_prefix   = set()
    empties        = 0
    n_rows         = 0

    # Header check
    if DELIM not in open(path, encoding="utf-8").readline() and \
       "," in open(path, encoding="utf-8").readline():
        err(f"{path.name}: looks CSV not TSV (no TAB in header)")
        return False, {}

    hdr = read_header(path)
    if hdr != expected_header:
        err(f"{path.name}: wrong header {hdr} (expected {expected_header})")
        return False, {}

    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for lineno, line in enumerate(f, 2):
            s1, tab, rest = line.rstrip("\n").partition(DELIM)
            if not tab:
                if s1.strip():
                    err(f"{path.name}: malformed row (no tab) at line {lineno}")
                continue

            n_rows += 1
            if s1 in seen_s1:
                dup_s1.add(s1)
            seen_s1.add(s1)

            ids_str = rest.strip()
            if not ids_str:
                empties += 1
                mapping[s1] = frozenset()
                continue

            ids_list = ids_str.split(",")
            ids_set  = set(ids_list)
            if len(ids_list) != len(ids_set):
                intra_dupes.add(s1)

            mapping[s1] = frozenset(ids_set)
            for mid in ids_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)

    # Aggregate findings
    file_ok = True

    def _ex(s, n=5):
        items = sorted(s)[:n]
        suffix = f", ... ({len(s)} total)" if len(s) > n else ""
        return ", ".join(items) + suffix

    if dup_s1:
        err(f"{path.name}: duplicate source1_entity_id rows: {_ex(dup_s1)}")
        file_ok = False
    if intra_dupes:
        err(f"{path.name}: duplicate IDs within {col_label} list for: {_ex(intra_dupes)}")
        file_ok = False
    if self_matches:
        err(f"{path.name}: {col_label} contains S1 IDs (self-matches): {_ex(self_matches)}")
        file_ok = False
    if wrong_prefix:
        err(f"{path.name}: {col_label} IDs without S2-/S3- prefix: {_ex(wrong_prefix)}")
        file_ok = False

    missing_s1 = required_s1 - seen_s1
    extra_s1   = seen_s1 - required_s1

    if missing_s1:
        err(f"{path.name}: {len(missing_s1):,} required S1 entities missing: {_ex(missing_s1)}")
        file_ok = False
    if extra_s1:
        err(f"{path.name}: {len(extra_s1):,} unknown S1 IDs: {_ex(extra_s1)}")
        file_ok = False

    print(f"    {n_rows} rows total ({empties} empty, {n_rows-empties} non-empty)")
    return file_ok, mapping


print(SEP)
print("FINAL PRE-SUBMISSION CHECK")
print(SEP)
print()

# ── CHECK A & B: Files exist ──────────────────────────────────────────────────
print("A/B — File existence")
print(sep)
matching_exists  = MATCHING_PATH.exists()
candidate_exists = CANDIDATE_PATH.exists()

if matching_exists:
    ok(f"matching_results.tsv exists ({MATCHING_PATH.stat().st_size/1e6:.2f} MB)")
else:
    err(f"matching_results.tsv NOT FOUND: {MATCHING_PATH}")

if candidate_exists:
    ok(f"candidate_pairs.tsv exists ({CANDIDATE_PATH.stat().st_size/1e6:.2f} MB)")
else:
    err(f"candidate_pairs.tsv NOT FOUND: {CANDIDATE_PATH}")

# If either file missing, cannot proceed with deeper checks
if not matching_exists or not candidate_exists:
    print()
    print(SEP)
    print("SUBMISSION READY: NO")
    print(f"  {len(errors)} blocking error(s). Fix files before re-running.")
    print(SEP)
    sys.exit(1)

# ── Load required S1 IDs (streaming) ─────────────────────────────────────────
print()
print("Loading test S1 entity IDs (streaming)...")
required_s1 = stream_ids_from_col1(TEST_S1_PATH)
print(f"  Required test S1 entities: {len(required_s1):,}")

# ── CHECK C & E–J: Validate matching file ────────────────────────────────────
print()
print("C/E-J — matching_results.tsv deep validation")
print(sep)
matching_ok, matching_map = validate_id_list_file_stream(
    MATCHING_PATH, MATCHING_HEADER, "matched_entity_ids", required_s1
)

# ── CHECK D & E–J: Validate candidate file ────────────────────────────────────
print()
print("D/E-J — candidate_pairs.tsv deep validation")
print(sep)
candidate_ok, candidate_map = validate_id_list_file_stream(
    CANDIDATE_PATH, CANDIDATE_HEADER, "candidate_entity_ids", required_s1
)

# ── CHECK I: Matched IDs ⊆ Candidate IDs per S1 ──────────────────────────────
print()
print("I — Matched IDs ⊆ Candidate IDs (per S1 entity)")
print(sep)
if matching_map and candidate_map:
    offenders = {
        s1: mids - candidate_map.get(s1, frozenset())
        for s1, mids in matching_map.items()
        if mids - candidate_map.get(s1, frozenset())
    }
    if offenders:
        total_leaked = sum(len(v) for v in offenders.values())
        warn(
            f"{len(offenders):,} S1 entities have {total_leaked:,} matched IDs "
            f"outside their candidate set. Pipeline bug likely."
        )
    else:
        ok("All matched IDs are within their candidate sets")

# ── CHECK K: Official validator ───────────────────────────────────────────────
print()
print("K — Official validator (utils/validate_submission.py)")
print(sep)
validator = REPO_ROOT / "utils" / "validate_submission.py"
if not validator.exists():
    err(f"Official validator not found: {validator}")
else:
    cmd = [
        sys.executable, str(validator),
        "--matching",   str(MATCHING_PATH),
        "--candidate",  str(CANDIDATE_PATH),
        "--test-dir",   str(TEST_DATA_DIR),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    for line in result.stdout.splitlines():
        print(f"    {line}")
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"    STDERR: {line}")
    if result.returncode == 0:
        ok("Official validator PASSED")
    else:
        err(f"Official validator FAILED (exit code {result.returncode})")

# ── FINAL VERDICT ─────────────────────────────────────────────────────────────
print()
print(SEP)
n_err  = len(errors)
n_warn = len(warnings)

if n_err > 0:
    print(f"SUBMISSION READY: NO — {n_err} blocking error(s)")
    for e in errors:
        print(f"  ✗ {e}")
    if warnings:
        print(f"\n  {n_warn} warning(s):")
        for w in warnings:
            print(f"  ⚠ {w}")
else:
    # Check that the files are the real final outputs, not samples
    # A real output must cover all test S1 entities
    n_matching  = len(matching_map)
    n_required  = len(required_s1)
    if n_matching < n_required:
        print(f"SUBMISSION READY: NO — matching_results covers only "
              f"{n_matching:,}/{n_required:,} test S1 entities")
    else:
        if warnings:
            print(f"SUBMISSION READY: YES (with {n_warn} warning(s))")
            for w in warnings:
                print(f"  ⚠ {w}")
        else:
            print("SUBMISSION READY: YES — all checks passed")

print(SEP)
sys.exit(1 if n_err > 0 else 0)
