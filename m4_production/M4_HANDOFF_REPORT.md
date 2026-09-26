# M4 HANDOFF REPORT: Candidate Generation

## 1. FINAL FROZEN BLOCKING CONFIGURATION
- **Blocks Used:** exact, token, address, prefix
- **token_max_df:** 500
- **address_max_df:** 100
- **prefix_max_df:** 100
- **chunk_size:** 25,000 (S1 entities per merge chunk)
- **memory_limit:** 6GB
- **threads:** 2

## 2. FULL TRAIN RESULTS
- **TRAIN S1 row count:** 2,206,821
- **Total candidate IDs:** 311,707,961
- **S1 with candidates:** 2,038,507
- **S1 empty:** 168,314
- **Average candidates/S1:** 141.25
- **Maximum candidates/S1:** 2,679
- **Actual candidate recall against train_ground_truth.tsv:** 59.01%
- **Generation runtime:** ~33 minutes
- **Validation runtime:** ~4.5 minutes

## 3. FULL TEST RESULTS
- **TEST S1 row count:** 1,732,544
- **Total candidate IDs:** 275,235,863
- **S1 with candidates:** 1,624,298
- **S1 empty:** 108,246
- **Average candidates/S1:** 158.86
- **Maximum candidates/S1:** 3,145
- **Final file size:** 3.40 GB
- **Generation runtime:** ~20 minutes
- **Validation status:** PASSED (Independent structural validation successfully confirmed 1.73M strict TSV rows, exactly one row per S1, and zero S1/duplicate leaks).

## 4. BLOCK BENCHMARKS
*Summary of isolated block evaluations:*
- **exact:** Primary precise matching.
- **token500:** Heavy candidate generator; max_df clamped at 500 to prevent stop-word explosions.
- **address100:** Spatial matching; constrained to max_df 100.
- **prefix100:** Lexical fallback; constrained to max_df 100.
- **country+token attempt/result:** Excluded/Abandoned due to severe over-generation and memory risks.
- **fuzzy attempt/result:** Excluded. Attempting `jaccard`/fuzzy matching inside DuckDB caused immediate Cartesian explosion and OOM errors during the initial benchmark phases.

## 5. BLOCK-WISE CONTRIBUTION
**Exact incremental per-block contribution cannot be derived from existing artifacts.** 

During the M4 phase, only fully unified combinations were evaluated on the 10,000-entity sample. We do not have the isolated "exact alone" or "exact + token500" metrics saved. 

The existing recorded metrics are strictly for the unified configurations:
- **UNION A (exact + token500 + address100 + prefix100):** 59.63% Recall, 1,458,854 pairs (Selected for production)
- **UNION B (exact + token500 + address100 + prefix500):** 61.25% Recall, 2,079,096 pairs
- **UNION C (exact + token500 + address500 + prefix500):** 67.22% Recall, 3,100,195 pairs

## 6. FINAL HANDOFF FILES
Confirmed present and finalized:
- `src/candidate_generation.py` (Patched with deterministic `ORDER BY entity_id` chunking)
- `m4_production/train_candidate_pairs.tsv`
- `output/candidate_pairs.tsv` (Validated TEST file)
- `m4_production/train_candidate_stats.json`
- `m4_production/test_candidate_stats.json`
- `m4_production/M4_HANDOFF_REPORT.md` (This file)

## 7. EXACT REPRODUCTION COMMANDS
**TRAIN Command:**
```powershell
py -3.10 src\candidate_generation.py --data-dir dataset\train --cache-dir cache --output m4_production\train_candidate_pairs.tsv --split train --blocks exact token address prefix --token-max-df 500 --address-max-df 100 --prefix-max-df 100 --gt dataset\train\train_ground_truth.tsv
```

**TEST Command:**
```powershell
py -3.10 src\candidate_generation.py --data-dir dataset\test --cache-dir cache --output m4_production\test_candidate_pairs.tsv --split test --blocks exact token address prefix --token-max-df 500 --address-max-df 100 --prefix-max-df 100
```

## 8. M4 SCOPE CHECK
I explicitly confirm that M4 strictly focused on Candidate Generation and did **NOT** perform:
- Model training (M3)
- Feature engineering
- Threshold tuning
- Final matching
- Submission packaging

## 9. FINAL STATUS
**M4 is COMPLETE.** The entire candidate generation pipeline is memory-safe, deterministic, fully validated, and successfully scaled to the full TRAIN and TEST datasets. All output files are securely staged for Deliverable #4 (Model Training/Predictions).
