# Final Approach (Member 3)

The matching pipeline relies on a supervised classification model that scores pre-generated candidate pairs. The technical approach is strictly divided into steps:

1. **Candidate Supply:** Candidate pairs are supplied by the blocking stage (Member 4 output) via `candidate_pairs.tsv`. The matching pipeline does not perform blocking.
2. **Feature Extraction:** Pair-level name and address similarity features (Jaccard, token overlap, Levenshtein ratio, length differences) are extracted for every candidate pair via `src/features.py`.
3. **Scoring:** A `StandardScaler` + `LogisticRegression` pipeline scores each pair to predict the probability of being a true match.
4. **Configuration Mapping:** The serialized `models/model_config.json` specifies the exact feature order expected by the model and the pre-tuned probability threshold (0.75).
5. **Thresholding:** Candidate pairs with a predicted probability greater than or equal to the threshold are retained as final matches.
6. **Formatting:** Predictions are grouped by S1 entity.
7. **Output Generation:** Final output is written as `matching_results.tsv`, containing one row per S1 test entity and comma-separated S2/S3 entity IDs. Empty strings denote no matches.
8. **Validation:** Final submission format and candidate consistency are validated using `utils/validate_submission.py`.
