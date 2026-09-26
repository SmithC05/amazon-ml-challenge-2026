# INTEGRATION GUIDE
## Amazon ML Challenge 2026 — Integration / Final Submission Owner

> **Machine:** `E:\Amazon\amazon-ml-challenge-2026`  
> **Commit:** `a34fa540ca14` · branch `main`  
> **Last Updated:** 2026-09-26  

---

## 1. Where Raw Data Lives

| File | Path |
|---|---|
| `train_source1.tsv` | `dataset/train/train_source1.tsv` (200.3 MB) |
| `train_source2.tsv` | `dataset/train/train_source2.tsv` (466.6 MB) |
| `train_source3.tsv` | `dataset/train/train_source3.tsv` (480.4 MB) |
| `train_ground_truth.tsv` | `dataset/train/train_ground_truth.tsv` (121.1 MB) |
| `test_source1.tsv` | `dataset/test/test_source1.tsv` (166.9 MB) |
| `test_source2.tsv` | `dataset/test/test_source2.tsv` (485.9 MB) |
| `test_source3.tsv` | `dataset/test/test_source3.tsv` (482.6 MB) |

> **Note:** Raw TSVs live at `dataset/train/` and `dataset/test/` (not `dataset/raw/`).  
> **Do NOT move or rename** these files.

---

## 2. Where M2 Cache Lives

```
dataset/processed/m2_cache/
  train_source1.parquet   (360.6 MB,  2,206,821 rows)
  train_source2.parquet   (906.5 MB,  5,034,616 rows)
  train_source3.parquet   (930.6 MB,  5,285,603 rows)
  test_source1.parquet    (293.2 MB,  1,732,544 rows)
  test_source2.parquet    (918.8 MB,  4,887,273 rows)
  test_source3.parquet    (921.9 MB,  5,082,316 rows)
```

All 6 caches have been built, validated, and verified as of 2026-09-26.  
**Do NOT regenerate unless a cache is corrupt** (use `integration/smoke_test.py` to verify).

### M2 Cache Schema (14 columns)

| Column | Type | Description |
|---|---|---|
| `entity_id` | str | Original entity ID |
| `business_name` | str | Raw business name |
| `business_address` | str | Raw business address |
| `country` | str | Country |
| `name_norm` | str | Normalized name (M2) |
| `address_norm` | str | Normalized address (M2) |
| `name_tokens` | str | Space-joined name tokens |
| `address_tokens` | str | Space-joined address tokens |
| `name_token_count` | int | Token count of name |
| `address_token_count` | int | Token count of address |
| `name_length` | int | Character length of name |
| `address_length` | int | Character length of address |
| `name_digits` | int | Digit count in name |
| `address_digits` | int | Digit count in address |

---

## 3. What M4 Must Deliver

**File:** `output/candidate_pairs.tsv`

**Format** (TSV, UTF-8, no BOM):
```
source1_entity_id\tcandidate_entity_ids
S1-000001\tS2-001,S2-002,S3-042
S1-000002\t
```

**Requirements:**
- Every `test_source1.tsv` entity must appear **exactly once**
- `candidate_entity_ids` is comma-separated S2-/S3- IDs, or empty string (no match)
- No S1-prefixed IDs in the candidate list
- No duplicate IDs within a row's candidate list
- Row count = 1,732,544 (all test S1 entities)

### ⚠️ CRITICAL: Cache-Dir for M4

`candidate_generation.py` defaults `--cache-dir` to `./cache` (relative CWD).  
**This is WRONG for this machine.** The canonical cache is at:

```
E:\Amazon\amazon-ml-challenge-2026\dataset\processed\m2_cache
```

**M4 must be run with this exact command:**

```powershell
python src/candidate_generation.py `
    --split test `
    --data-dir "E:\Amazon\amazon-ml-challenge-2026\dataset\test" `
    --cache-dir "E:\Amazon\amazon-ml-challenge-2026\dataset\processed\m2_cache" `
    --output "E:\Amazon\amazon-ml-challenge-2026\output\candidate_pairs.tsv"
```

Or from repo root:

```powershell
python src/candidate_generation.py `
    --split test `
    --data-dir dataset/test `
    --cache-dir dataset/processed/m2_cache `
    --output output/candidate_pairs.tsv
```

**Do NOT** run without `--cache-dir` — it will fail to find the cache.

---

## 4. What M3 Must Deliver

**Training artifact:** `models/matcher.pkl`  
**Config:** `models/model_config.json`

M3 training command (run AFTER M4 delivers `output/candidate_pairs.tsv`):

```powershell
python src/train.py `
    --data-dir dataset/train `
    --candidates output/candidate_pairs.tsv `
    --cache-dir dataset/processed/m2_cache `
    --output-dir output `
    --models-dir models
```

---

## 5. Exact Prediction Command

Once M3 training is complete and M4 has delivered `output/candidate_pairs.tsv`:

```powershell
python src/predict.py `
    --data-dir dataset/test `
    --candidates output/candidate_pairs.tsv `
    --cache-dir dataset/processed/m2_cache `
    --model-dir models `
    --output-dir output `
    --split test
```

**Output:** `output/matching_results.tsv`

---

## 6. Exact Validator Command

```powershell
python utils/validate_submission.py `
    --matching output/matching_results.tsv `
    --candidate output/candidate_pairs.tsv `
    --test-dir dataset/test
```

With ID existence check (more thorough, uses more RAM):

```powershell
python utils/validate_submission.py `
    --matching output/matching_results.tsv `
    --candidate output/candidate_pairs.tsv `
    --test-dir dataset/test `
    --check-ids
```

Exit code 0 = safe to submit. Exit code 1 = fix errors first.

---

## 7. Final Submission File Requirements

The submission zip must contain both:

| File | Format | Required |
|---|---|---|
| `output/matching_results.tsv` | TSV, TAB-delimited, UTF-8 | **Required** (scored) |
| `output/candidate_pairs.tsv` | TSV, TAB-delimited, UTF-8 | **Required** (expected in zip) |

### matching_results.tsv format

```
source1_entity_id\tmatched_entity_ids
S1-000001\tS2-001,S3-042
S1-000002\t
```

- Header must be exactly `source1_entity_id\tmatched_entity_ids`
- Every test S1 entity must appear exactly once
- Empty `matched_entity_ids` = no match predicted
- No S1-prefixed IDs in matched list
- IDs comma-separated within a row

### Pipeline: M2 → M4 → M3 → Prediction → Submission

```
M2 (Parquet cache)
    ↓  name_norm, address_norm, country, name_tokens, ...
M4 (candidate_generation.py)
    ↓  source1_entity_id, candidate_entity_ids
M3 (train.py → matcher.pkl)
    ↓  threshold, trained model
predict.py
    ↓  source1_entity_id, matched_entity_ids
validate_submission.py → SUBMIT
```

---

## 8. Cache-Path Requirement

| Script | `--cache-dir` default | Status |
|---|---|---|
| `src/candidate_generation.py` | `cache` (relative) | ⚠ **Must override** |
| `src/train.py` | `None` (must supply) | ✓ Pass at runtime |
| `src/predict.py` | `None` (must supply) | ✓ Pass at runtime |

**Always pass:**
```
--cache-dir "E:\Amazon\amazon-ml-challenge-2026\dataset\processed\m2_cache"
```
or, from the repo root:
```
--cache-dir dataset/processed/m2_cache
```

---

## 9. Integration Scripts

| Script | Purpose |
|---|---|
| `integration/local_paths.py` | Single source of truth for all local paths |
| `integration/smoke_test.py` | Run before and after any pipeline step |
| `integration/final_pre_submission_check.py` | Run before submitting |
| `integration/INTEGRATION_GUIDE.md` | This document |
| `integration/m2_local_status.txt` | M2 verification report |

**Run smoke test at any time:**
```powershell
python integration/smoke_test.py
```

**Run final check when matching_results.tsv is ready:**
```powershell
python integration/final_pre_submission_check.py
```

---

## 10. Known Current Status

| Component | Status | Notes |
|---|---|---|
| **M2 Cache** | ✅ **READY** | All 6 Parquet files verified 2026-09-26 |
| **M4 Candidates** | ⏳ **IN PROGRESS** | `candidate_pairs.tsv` not yet final |
| **M3 Training** | ⏳ **IN PROGRESS** | Requires M4 output first |
| **Final Prediction** | ❌ **NOT READY** | Requires M3 model + M4 candidates |
| **Submission** | ❌ **NOT READY** | Requires Final Prediction |

### M2 Cache Row Counts (Verified)

| Source | Rows |
|---|---|
| `train_source1` | 2,206,821 |
| `train_source2` | 5,034,616 |
| `train_source3` | 5,285,603 |
| `test_source1` | 1,732,544 |
| `test_source2` | 4,887,273 |
| `test_source3` | 5,082,316 |

---

## 11. Interface Contracts

### M2 → M4 (cache columns consumed by candidate_generation.py)

```
entity_id      — S1-/S2-/S3- prefixed entity identifier
name_norm      — normalized business name
address_norm   — normalized address
country        — country string
name_tokens    — (implicitly via name_norm split)
```

Source: `src/candidate_generation.py` docstring, lines 58–61.

### M4 → M3 (input to train.py / predict.py)

```
source1_entity_id    — S1- prefixed
candidate_entity_ids — comma-separated S2-/S3- IDs (or empty)
```

Source: `src/train.py` line 430, `src/predict.py` line 248.

### M3 → Final Submission

```
source1_entity_id   — S1- prefixed
matched_entity_ids  — comma-separated S2-/S3- IDs (or empty)
```

Source: `src/predict.py` line 231, `utils/validate_submission.py` line 48.

---

> **This document is authoritative for integration on this machine.**  
> Do not modify M3/M4 source files on this machine.  
> Do not push raw data, Parquet caches, or generated outputs to GitHub.
