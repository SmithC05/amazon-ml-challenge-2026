# M3 Baseline Model Report
## Amazon ML Challenge 2026 — Entity Matching

> **Role boundary:** Member 3 owns the baseline pair matcher, baseline validation, threshold tuning, and the reusable inference wrapper documented here. Preprocessing/normalization is Member 2's deliverable (`src/preprocess.py`). Candidate generation/blocking is Member 4's deliverable (`output/candidate_pairs.tsv`). Advanced model experimentation and final model selection belong to the team lead.

---

## 1. Responsibility

| Artifact | Owner |
|---|---|
| `src/preprocess.py` | **Member 2** (do not modify) |
| `output/candidate_pairs.tsv` | **Member 4** |
| `src/features.py` | Member 3 |
| `src/train.py` | Member 3 |
| `src/predict.py` | Member 3 |
| `models/matcher.pkl` | Member 3 |
| `models/model_config.json` | Member 3 |
| `output/baseline_training_results.json` | Member 3 |
| `output/baseline_threshold_results.tsv` | Member 3 |
| `notebooks/04_models.ipynb` | Member 3 |
| Advanced model experiments | Team lead |

---

## 2. Pipeline Overview

```
Raw TSVs  (train_source1/2/3 + train_ground_truth)
    │
    ▼  src/preprocess.py  (Member 2)
Normalized text columns (_norm)
    │
    ▼  output/candidate_pairs.tsv  (Member 4)
Candidate (S1, S2/S3) pairs
    │
    ▼  src/features.py  (Member 3)
16-feature baseline vector per pair
    │
    ▼  src/train.py / src/predict.py  (Member 3)
Baseline Logistic Regression predictions
    │
    ▼
output/matching_results.tsv
```

---

## 3. Normalization (Member 2)

Normalization is **not reimplemented** in this module. The following functions from `src/preprocess.py` are imported and called:

| Function | Applied to |
|---|---|
| `normalize_name(value)` | `business_name` → `business_name_norm` |
| `normalize_address(value)` | `business_address` → `business_address_norm` |

The pipeline applied by M2 is: Unicode NFKC → lowercase → punctuation → whitespace collapse → validated legal-suffix / address-abbreviation token maps.

Raw columns (`business_name`, `business_address`) are never overwritten and are preserved alongside the normalized columns.

---

## 4. Baseline Feature Set

Sixteen deterministic, pair-level features are computed by `src/features.py`.  
All text inputs are read from the `_norm` columns; raw fields are never used.

### Name similarity (6 features)

| Feature | Description |
|---|---|
| `name_exact` | 1 if normalized names are character-for-character identical |
| `name_jaccard` | Jaccard similarity of character 3-gram sets |
| `name_token_overlap` | Jaccard similarity of whitespace-token sets |
| `name_levenshtein_ratio` | 1 − edit_distance / max_length |
| `name_length_difference` | \|len(a) − len(b)\| / max(len, 1) |
| `name_token_count_diff` | \|tokens(a) − tokens(b)\| / max(count, 1) |

### Address similarity (7 features)

| Feature | Description |
|---|---|
| `address_exact` | 1 if normalized addresses are identical |
| `address_jaccard` | Jaccard similarity of character 3-gram sets |
| `address_token_overlap` | Jaccard similarity of token sets |
| `address_levenshtein_ratio` | Normalized Levenshtein |
| `address_length_difference` | Relative length difference |
| `address_token_count_diff` | Relative token-count difference |
| `address_missing` | 1 if either address normalizes to `""` |

### Other (3 features)

| Feature | Description |
|---|---|
| `country_match` | 1 if raw country strings are equal (lowercased, non-empty) |
| `source_is_s2` | 1 if the candidate entity_id starts with `S2-` |
| `source_is_s3` | 1 if the candidate entity_id starts with `S3-` |

---

## 5. Baseline Model Architecture

```
StandardScaler         — zero-mean, unit-variance scaling of the 16 features
LogisticRegression     — binary classifier; random_state=42, max_iter=1000
```

- Input: 16-dimensional feature vector (float64)
- Output: probability of match for each candidate pair
- Decision: `match` if `P(match) ≥ threshold`

No ensemble, boosting, or deep-learning components are used. Those experiments are reserved for the team lead.

---

## 6. Entity-Level Train / Validation Split

Splitting is performed **by unique S1 entity**, not by row, to prevent data leakage.

| Split | Fraction | random_state |
|---|---|---|
| Train | 80 % of labeled S1 entities | 42 |
| Validation | 20 % of labeled S1 entities | 42 |

The intersection of train and validation S1 IDs is verified to be empty at runtime.

---

## 7. Official Metric: Entity-Level Macro F0.5

The competition scores submissions on **entity-level macro F0.5**.  
This is computed as follows for **each** labeled S1 entity:

```
truth     = set of true matched IDs for that S1 entity
predicted = set of candidate IDs predicted as matches at threshold t

Case 1: truth = {} AND predicted = {}   → entity_f0.5 = 1.0
Case 2: truth = {} AND predicted ≠ {}   → entity_f0.5 = 0.0
Case 3: otherwise
    precision = |truth ∩ predicted| / |predicted|  (0 if predicted empty)
    recall    = |truth ∩ predicted| / |truth|
    entity_f0.5 = (1 + 0.5²) · P · R / (0.5² · P + R)   if P + R > 0
                = 0                                         otherwise

Final score = mean(entity_f0.5)  across all validation S1 entities
```

> **Important:** do NOT compute F0.5 from a single global precision and recall. Each entity contributes one F0.5 score to the macro average.

---

## 8. Threshold Sweep

The decision threshold is not hard-coded. A sweep over `[0.50, 0.55, …, 0.95]` is run on the validation set and the threshold maximizing the entity-level macro F0.5 is selected and saved to `models/model_config.json`.

Results are saved in full to `output/baseline_threshold_results.tsv`.

---

## 9. Baseline Validation Results

*(To be filled after first full run with M4's candidate pairs.)*

| Metric | Value |
|---|---|
| Selected threshold | TBD |
| Validation Precision (avg per-entity) | TBD |
| Validation Recall (avg per-entity) | TBD |
| **Validation entity-level macro F0.5** | **TBD** |
| Train S1 entities | TBD |
| Validation S1 entities | TBD |
| Train pairs | TBD |
| Validation pairs | TBD |

---

## 10. Error Analysis Observations

*(To be completed after first run. Guidance for what to look at:)*

**False positives** (predicted match, wrong):
- Common when business names are similar but addresses differ (e.g., chain stores in the same city).
- `address_missing = 1` pairs are higher-risk — address cannot disambiguate.

**False negatives** (missed match):
- Entities with large name surface variation (abbreviation, language, transliteration) have low `name_jaccard` and `name_levenshtein_ratio` — the baseline cannot recover these.
- These motivate M4's semantic/fuzzy candidate generation and future model experiments.

---

## 11. Inference Interface

```bash
# After training:
python src/predict.py \
    --data-dir  dataset/test \
    --candidates output/candidate_pairs.tsv \
    --model-dir  models \
    --output-dir output
```

Produces `output/matching_results.tsv`:
- One row per test S1 entity.
- `matched_entity_ids` is empty for entities with no predicted match.
- Format is tab-separated, UTF-8 — compatible with `utils/validate_submission.py`.

---

## 12. What This Baseline Is Not

- This is **not** the final competition model.
- XGBoost, CatBoost, deep learning, and ensemble experiments belong to the team lead.
- This baseline establishes the reference F0.5 score that all future experiments must beat.
