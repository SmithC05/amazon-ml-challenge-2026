"""
integration/test_validator_fixtures.py
=======================================
PART 4 — Test the Official Submission Validator

Creates synthetic fixtures to test utils/validate_submission.py
WITHOUT touching real competition output.

Test cases:
  1. Correct matching_results.tsv
  2. Missing S1 row
  3. Duplicate S1 row
  4. Wrong ID prefix (no S2-/S3-)
  5. Repeated matched ID within a row
  6. CSV instead of TSV (wrong delimiter)
  7. Empty matched IDs (valid: means no match)
  8. Candidate/matching mismatch (matched not in candidates)

Usage:
    python integration/test_validator_fixtures.py
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTEGRATION_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

VALIDATOR = REPO_ROOT / "utils" / "validate_submission.py"
TEST_DIR  = REPO_ROOT / "dataset" / "test"

SEP = "=" * 70
sep = "-" * 50

results = []

# ── Synthetic test S1 data ────────────────────────────────────────────────────
# We create a tiny fake test_source1.tsv so the validator knows which S1 IDs exist.
# We use 5 known IDs drawn from the real test set (safe — not ground truth).
FAKE_S1_IDS = ["S1-100001", "S1-100002", "S1-100003", "S1-100004", "S1-100005"]
FAKE_S2_IDS = ["S2-200001", "S2-200002", "S2-200003"]
FAKE_S3_IDS = ["S3-300001", "S3-300002"]

FIXTURE_DIR = REPO_ROOT / "integration" / "fixtures"
FIXTURE_DIR.mkdir(exist_ok=True)

# Write a fake test_source1.tsv (the validator only reads the entity_id column)
fake_s1_tsv = FIXTURE_DIR / "fake_test_source1.tsv"
with open(fake_s1_tsv, "w", encoding="utf-8") as f:
    f.write("entity_id\tbusiness_name\n")
    for sid in FAKE_S1_IDS:
        f.write(f"{sid}\tFake Business {sid}\n")

# Write a fake candidate_pairs.tsv (valid, covers all S1 IDs)
fake_cand = FIXTURE_DIR / "fake_candidate_pairs.tsv"
with open(fake_cand, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    f.write(f"S1-100001\tS2-200001,S2-200002,S3-300001\n")
    f.write(f"S1-100002\tS2-200003,S3-300002\n")
    f.write(f"S1-100003\tS2-200001\n")
    f.write(f"S1-100004\t\n")
    f.write(f"S1-100005\tS3-300001,S3-300002\n")


def run_validator(matching_path, candidate_path=None, test_dir=None, extra_args=None):
    """Run the validator and return (returncode, stdout, stderr)."""
    cmd = [
        sys.executable, str(VALIDATOR),
        "--matching", str(matching_path),
        "--test-dir", str(test_dir or FIXTURE_DIR),
    ]
    if candidate_path:
        cmd += ["--candidate", str(candidate_path)]
    if extra_args:
        cmd += extra_args
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    return r.returncode, r.stdout, r.stderr


def report(case_num, name, expected_pass, returncode, stdout, stderr, expected_contains=None):
    stdout_all = stdout + stderr
    actual_pass = (returncode == 0)
    correct = (actual_pass == expected_pass)
    detected = all(e in stdout_all for e in (expected_contains or []))

    status = "✓ CORRECT" if (correct and detected) else "✗ WRONG"
    results.append({
        "case": case_num, "name": name,
        "expected_pass": expected_pass, "actual_pass": actual_pass,
        "correct": correct and detected,
    })
    print(f"  Case {case_num}: {name}")
    print(f"    Expected: {'PASS' if expected_pass else 'FAIL'}  "
          f"Got: {'PASS' if actual_pass else 'FAIL'}  [{status}]")
    if expected_contains:
        for e in expected_contains:
            found = e in stdout_all
            print(f"    Expected text: {e!r:.60}  → {'FOUND ✓' if found else 'NOT FOUND ✗'}")
    # Show first 3 non-empty output lines
    for line in [l for l in stdout.splitlines() if l.strip()][:5]:
        print(f"    OUT: {line}")
    print()


print(SEP)
print("PART 4 — VALIDATOR FIXTURE TESTS")
print(SEP)
print(f"  Validator:  {VALIDATOR}")
print(f"  Fixture dir: {FIXTURE_DIR}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# CASE 1: Correct matching_results.tsv (must PASS)
# ──────────────────────────────────────────────────────────────────────────────
print(sep)
print("CASE 1 — Correct matching_results.tsv")
fix1 = FIXTURE_DIR / "case1_correct.tsv"
with open(fix1, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    f.write("S1-100001\tS2-200001,S3-300001\n")
    f.write("S1-100002\tS2-200003\n")
    f.write("S1-100003\tS2-200001\n")
    f.write("S1-100004\t\n")
    f.write("S1-100005\tS3-300002\n")
rc, out, err = run_validator(fix1, fake_cand)
report(1, "Correct format", expected_pass=True, returncode=rc, stdout=out, stderr=err,
       expected_contains=["PASS"])

# ──────────────────────────────────────────────────────────────────────────────
# CASE 2: Missing S1 row (must FAIL)
# ──────────────────────────────────────────────────────────────────────────────
print(sep)
print("CASE 2 — Missing S1 row")
fix2 = FIXTURE_DIR / "case2_missing_s1.tsv"
with open(fix2, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    f.write("S1-100001\tS2-200001\n")
    # S1-100002, S1-100003, S1-100004, S1-100005 missing
rc, out, err = run_validator(fix2, fake_cand)
report(2, "Missing S1 row", expected_pass=False, returncode=rc, stdout=out, stderr=err,
       expected_contains=["required S1 entity"])

# ──────────────────────────────────────────────────────────────────────────────
# CASE 3: Duplicate S1 row (must FAIL)
# ──────────────────────────────────────────────────────────────────────────────
print(sep)
print("CASE 3 — Duplicate S1 row")
fix3 = FIXTURE_DIR / "case3_dup_s1.tsv"
with open(fix3, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    f.write("S1-100001\tS2-200001\n")
    f.write("S1-100001\tS2-200002\n")  # duplicate!
    f.write("S1-100002\tS2-200003\n")
    f.write("S1-100003\t\n")
    f.write("S1-100004\t\n")
    f.write("S1-100005\t\n")
rc, out, err = run_validator(fix3, fake_cand)
report(3, "Duplicate S1 row", expected_pass=False, returncode=rc, stdout=out, stderr=err,
       expected_contains=["duplicate source1_entity_id"])

# ──────────────────────────────────────────────────────────────────────────────
# CASE 4: Wrong ID prefix (must FAIL)
# ──────────────────────────────────────────────────────────────────────────────
print(sep)
print("CASE 4 — Wrong ID prefix (no S2-/S3-)")
fix4 = FIXTURE_DIR / "case4_wrong_prefix.tsv"
with open(fix4, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    f.write("S1-100001\tBAD-000001,S2-200001\n")   # BAD- prefix
    f.write("S1-100002\tS2-200003\n")
    f.write("S1-100003\t\n")
    f.write("S1-100004\t\n")
    f.write("S1-100005\t\n")
rc, out, err = run_validator(fix4, fake_cand)
report(4, "Wrong ID prefix", expected_pass=False, returncode=rc, stdout=out, stderr=err,
       expected_contains=["without an S2-/S3- prefix"])

# ──────────────────────────────────────────────────────────────────────────────
# CASE 5: Repeated matched ID within a row (must FAIL)
# ──────────────────────────────────────────────────────────────────────────────
print(sep)
print("CASE 5 — Repeated matched ID within a row")
fix5 = FIXTURE_DIR / "case5_intra_dup.tsv"
with open(fix5, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    f.write("S1-100001\tS2-200001,S2-200001\n")   # S2-200001 duplicated!
    f.write("S1-100002\tS2-200003\n")
    f.write("S1-100003\t\n")
    f.write("S1-100004\t\n")
    f.write("S1-100005\t\n")
rc, out, err = run_validator(fix5, fake_cand)
report(5, "Repeated matched ID within row", expected_pass=False, returncode=rc, stdout=out, stderr=err,
       expected_contains=["repeated ID inside a matched_entity_ids"])

# ──────────────────────────────────────────────────────────────────────────────
# CASE 6: CSV instead of TSV (must FAIL)
# ──────────────────────────────────────────────────────────────────────────────
print(sep)
print("CASE 6 — CSV instead of TSV")
fix6 = FIXTURE_DIR / "case6_csv.tsv"
with open(fix6, "w", encoding="utf-8") as f:
    f.write("source1_entity_id,matched_entity_ids\n")   # COMMA delimiter!
    f.write("S1-100001,S2-200001\n")
    f.write("S1-100002,S2-200003\n")
rc, out, err = run_validator(fix6, fake_cand)
report(6, "CSV instead of TSV", expected_pass=False, returncode=rc, stdout=out, stderr=err,
       expected_contains=["COMMA-separated"])

# ──────────────────────────────────────────────────────────────────────────────
# CASE 7: Empty matched IDs (valid — means no match predicted → should PASS)
# ──────────────────────────────────────────────────────────────────────────────
print(sep)
print("CASE 7 — All empty matched IDs (valid: 'no match' for all)")
fix7 = FIXTURE_DIR / "case7_all_empty.tsv"
with open(fix7, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in FAKE_S1_IDS:
        f.write(f"{sid}\t\n")
rc, out, err = run_validator(fix7, fake_cand)
report(7, "All empty matched IDs (PASS expected)", expected_pass=True, returncode=rc, stdout=out, stderr=err,
       expected_contains=["PASS"])

# ──────────────────────────────────────────────────────────────────────────────
# CASE 8: Candidate/matching mismatch (matched ID not in candidates → WARNING not FAIL)
# ──────────────────────────────────────────────────────────────────────────────
print(sep)
print("CASE 8 — Matched ID not in candidate set (WARNING, not FAIL)")
fix8 = FIXTURE_DIR / "case8_cand_mismatch.tsv"
with open(fix8, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    f.write("S1-100001\tS2-200001,S3-300001\n")       # S3-300001 is in candidates ✓
    f.write("S1-100002\tS3-999999\n")                 # S3-999999 NOT in fake_cand ← mismatch
    f.write("S1-100003\t\n")
    f.write("S1-100004\t\n")
    f.write("S1-100005\t\n")
rc, out, err = run_validator(fix8, fake_cand)
# According to validate_submission.py, mismatch is a WARNING, not FAIL
report(8, "Candidate/matching mismatch (WARN only, PASS exit)", expected_pass=True,
       returncode=rc, stdout=out, stderr=err,
       expected_contains=["WARNING"])

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print()
print(SEP)
print("VALIDATOR FIXTURE TEST SUMMARY")
print(SEP)
n_correct = sum(1 for r in results if r["correct"])
n_total   = len(results)

print(f"\n  {'Case':<5}  {'Name':<45}  {'Expected':<8}  {'Got':<8}  {'Result'}")
print("  " + "-" * 80)
for r in results:
    expected = "PASS" if r["expected_pass"] else "FAIL"
    actual   = "PASS" if r["actual_pass"]   else "FAIL"
    status   = "✓ CORRECT" if r["correct"] else "✗ WRONG"
    print(f"  {r['case']:<5}  {r['name']:<45}  {expected:<8}  {actual:<8}  {status}")

print()
print(f"  {n_correct}/{n_total} cases behaved as expected")
print()

if n_correct == n_total:
    print("VALIDATOR TESTS: ALL CORRECT ✓")
    print("  The official validator correctly detects all tested error categories.")
else:
    print(f"VALIDATOR TESTS: {n_total - n_correct} unexpected result(s)")
    print("  Review the cases above. Do NOT modify validate_submission.py unless")
    print("  a genuine bug (incorrect PASS/FAIL) is confirmed.")
print(SEP)

# Clean up fixtures after test (keep them for reference)
# FIXTURE_DIR contents preserved for manual inspection.
print(f"\n  Fixture files preserved in: {FIXTURE_DIR}")

sys.exit(0 if n_correct == n_total else 1)
