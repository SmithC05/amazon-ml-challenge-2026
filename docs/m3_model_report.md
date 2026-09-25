# Member 3 — Matching Model & Validation Report

## 1. Responsibility
Member 3 was responsible for the core matching pipeline, including:
- positive/negative pair construction (using ground truth for positives)
- pair-level similarity features
- baseline matching model (training and tuning)
- threshold tuning to maximize the evaluation metric
- Precision / Recall / F0.5 evaluation
- error analysis
- inference integration (building the final scoring and thresholding script)
- output validation (ensuring submission compliance)

## 2. Problem Understanding
The task is Business Entity Resolution across three data sources:
- **Source 1** is the reference entity set. Every S1 entity in the test set requires a prediction.
- **Source 2 and Source 3** contain noisy records that must be matched against Source 1.
- One S1 entity can match zero, one, or multiple S2/S3 entities.
- Matching is performed using candidate pairs supplied by the blocking/candidate-generation stage (provided by Member 4).
- Final predictions are produced per Source 1 entity by retaining candidates that score above a tuned probability threshold.

## 3. Training Pair Construction
The baseline pairs used for training (`output/training_pairs_baseline.tsv`) were generated with:
- **Positives** derived directly from the provided ground truth matches.
- **Negatives** constructed to challenge the model, including same-country negative sampling where applicable.
- Cartesian-product generation was avoided to prevent combinatorial explosion and class imbalance.
- Random seeds were used in the pipeline (e.g. `random_state=42`) to ensure reproducibility.
The repository artifacts confirm a final dataset of 945,996 pairs (756,687 train pairs and 189,309 validation pairs) derived from 100,000 S1 entities.

## 4. Feature Engineering
The feature extraction logic is implemented in `src/features.py`. The model relies on exactly 16 pair-level features. Before extraction, text strings undergo normalization: Unicode NFC normalization, lowercase conversion, conversion of non-word characters/punctuation to spaces, and whitespace collapsing. Address missingness is safely handled (producing zeroed similarity when absent).

### Name features
- `name_exact`
- `name_jaccard`
- `name_token_overlap`
- `name_levenshtein_ratio`
- `name_length_difference`
- `name_token_count_diff`

### Address features
- `address_exact`
- `address_jaccard`
- `address_token_overlap`
- `address_levenshtein_ratio`
- `address_length_difference`
- `address_token_count_diff`
- `address_missing`

### Other features
- `country_match`
- `source_is_s2`
- `source_is_s3`

## 5. Baseline Model
The baseline matching model is implemented in `src/train.py` using `scikit-learn`.
- Algorithm: **Logistic Regression** (`max_iter=1000`, `random_state=42`)
- Preprocessing: `StandardScaler` applied in a `Pipeline` before classification.
- Split strategy: Entity-level train/validation split by **S1 entity** (80/20 split: 80,000 S1 entities for train, 20,000 for validation).
Splitting by `s1_entity_id` is crucial because it ensures ZERO pair-level overlap across splits; putting the same S1 entity in both train and validation would cause data leakage and artificially inflate validation metrics.

## 6. Evaluation Metric
The challenge uses a specific entity-level macro-averaged metric:
**F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)**

The F0.5 metric inherently weights Precision twice as heavily as Recall, punishing false positives more severely than false negatives. Evaluation is performed per S1 entity (macro-averaged) across the validation set.

## 7. Threshold Tuning
A threshold sweep (from 0.50 to 0.95 in 0.05 increments) was conducted to find the optimal cutoff for F0.5. Results are captured in `output/baseline_threshold_results.tsv`.

| Threshold | Precision | Recall | F0.5 |
|---|---|---|---|
| 0.50 | 0.997069 | 0.995182 | 0.996691 |
| 0.55 | 0.997528 | 0.994731 | 0.996967 |
| 0.60 | 0.997906 | 0.994222 | 0.997167 |
| 0.65 | 0.998165 | 0.993672 | 0.997263 |
| 0.70 | 0.998329 | 0.993062 | 0.997272 |
| **0.75** | **0.998541** | **0.992368** | **0.997300** |
| 0.80 | 0.998717 | 0.991392 | 0.997243 |
| 0.85 | 0.998924 | 0.990051 | 0.997136 |
| 0.90 | 0.999131 | 0.988223 | 0.996930 |
| 0.95 | 0.999368 | 0.985088 | 0.996479 |

The selected threshold stored in `models/model_config.json` is **0.75**, which yields the highest F0.5 (0.997300).

## 8. Error Analysis
Based on `output/baseline_training_results.json`, at the 0.75 threshold:
- **False positives:** 101 pairs
- **False negatives:** 537 pairs
Examples show that false positives often involve pairs where address similarity (e.g. `address_levenshtein_ratio` ~0.45-0.59) and length differences confuse the classifier despite low `name_jaccard` (e.g. 0.0 to 0.28). False negatives frequently involve cases with heavily missing addresses (`address_missing` = 1) or completely unaligned text (e.g. `name_jaccard` = 0.0, `address_jaccard` = 0.10) where the model correctly assigns a low probability (e.g. 0.06 to 0.40) to true matches that look completely distinct.

## 9. Inference Pipeline
The inference script is implemented in `src/predict.py`. It requires `candidate_pairs.tsv` as input and does NOT generate candidates or perform blocking itself.

**Workflow:**
1. Loads test sources (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`) and `candidate_pairs.tsv`.
2. Uses `src/features.py` to extract exactly the 16 required features for each candidate pair.
3. Loads the trained pipeline from `models/matcher.pkl`.
4. Predicts match probabilities and applies the saved threshold (0.75).
5. Writes `output/matching_results.tsv`.

**Output Format Rules Enforced:**
- `source1_entity_id` and `matched_entity_ids` columns.
- One row per S1 test entity.
- Comma-separated matched S2/S3 entity IDs.
- No duplicate matched IDs.
- Empty string when there are no predicted matches.

## 10. Validation
Submission validity can be checked via `utils/validate_submission.py`. It confirms that the output formatting rules and candidate consistency are maintained.

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
(Final validation against the massive test set requires the test-time `candidate_pairs.tsv` to be generated first).

## 11. Model Artifacts
- `models/matcher.pkl`: The serialized `scikit-learn` Pipeline containing the `StandardScaler` and `LogisticRegression` model.
- `models/model_config.json`: Metadata capturing the expected feature names, random seed, threshold (0.75), and validation metrics.
- `output/baseline_threshold_results.tsv`: TSV file containing the F0.5 threshold sweep results.
- `output/baseline_training_results.json`: Full training report containing elapsed time, validation split sizes, threshold metrics, and detailed false positive/false negative examples for error analysis.

## 12. Current Status / Remaining Integration
The Member 3 matching pipeline is fully implemented, verified via local unit and smoke tests, and the final model is trained and saved. However, **final test inference has not yet been completed** because it requires the complete test-set `candidate_pairs.tsv` from the blocking phase (Member 4) to be supplied.

## 13. Reproducibility
The pipeline can be reproduced using the following commands:

**Training (assuming input data is present in output/):**
```bash
python src/train.py
```

**Prediction (after M4 candidates are ready):**
```bash
python src/predict.py
```

**Validation:**
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
