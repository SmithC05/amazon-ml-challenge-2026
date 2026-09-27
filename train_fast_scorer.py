import duckdb
import time
import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
sys.path.insert(0, str(Path(".").resolve() / "src"))
from train import build_truth_map, entity_f05

query_features = """
SELECT 
    source1_entity_id, cand_entity_id, label, is_val,
    (s1_name_norm == cand_name_norm)::INT as name_exact,
    (s1_address_norm == cand_address_norm)::INT as address_exact,
    (s1_country == cand_country)::INT as country_match,
    ABS(LENGTH(s1_name_norm) - LENGTH(cand_name_norm)) / GREATEST(LENGTH(s1_name_norm), LENGTH(cand_name_norm), 1) as name_length_diff,
    ABS(LENGTH(s1_address_norm) - LENGTH(cand_address_norm)) / GREATEST(LENGTH(s1_address_norm), LENGTH(cand_address_norm), 1) as address_length_diff,
    (SUBSTRING(s1_name_norm, 1, 5) == SUBSTRING(cand_name_norm, 1, 5))::INT as name_prefix5,
    (SUBSTRING(s1_address_norm, 1, 5) == SUBSTRING(cand_address_norm, 1, 5))::INT as address_prefix5,
    CASE WHEN LENGTH(s1_name_norm) >= 2 AND LENGTH(cand_name_norm) >= 2 THEN jaccard(s1_name_norm, cand_name_norm) ELSE 0.0 END as name_jaccard,
    CASE WHEN LENGTH(s1_address_norm) >= 2 AND LENGTH(cand_address_norm) >= 2 THEN jaccard(s1_address_norm, cand_address_norm) ELSE 0.0 END as address_jaccard,
    levenshtein(s1_name_norm, cand_name_norm) as name_levenshtein,
    levenshtein(s1_address_norm, cand_address_norm) as address_levenshtein,
    jaro_winkler_similarity(s1_name_norm, cand_name_norm) as name_jaro,
    jaro_winkler_similarity(s1_address_norm, cand_address_norm) as address_jaro
FROM (
    SELECT * FROM 'output/.pair_spool/pairs.parquet' WHERE is_val = 1
    UNION ALL
    SELECT * FROM (
        SELECT * FROM 'output/.pair_spool/pairs.parquet' WHERE is_val = 0 AND label = 1
        UNION ALL
        SELECT * FROM (SELECT * FROM 'output/.pair_spool/pairs.parquet' WHERE is_val = 0 AND label = 0 USING SAMPLE 2000000)
    )
)
"""

con = duckdb.connect()

print("Extracting features from spool in DuckDB...")
t0 = time.time()
df = con.execute(query_features).df()
print(f"Extraction done in {time.time()-t0:.2f}s. Rows: {len(df)}")

df_train = df[df['is_val'] == 0].copy()
df_val = df[df['is_val'] == 1].copy()

n_pos = df_train['label'].sum()
df_train_pos = df_train[df_train['label'] == 1]
df_train_neg = df_train[df_train['label'] == 0]
if len(df_train_neg) > max(0, 2000000 - n_pos):
    df_train_neg = df_train_neg.sample(n=max(0, 2000000 - n_pos), random_state=42)
df_train_bal = pd.concat([df_train_pos, df_train_neg])

feature_cols = [c for c in df.columns if c not in ['source1_entity_id', 'cand_entity_id', 'label', 'is_val']]

print(f"Training LogisticRegression on {len(df_train_bal)} rows...")
X_train = df_train_bal[feature_cols].values
y_train = df_train_bal['label'].values

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)

print("Scoring validation set...")
X_val = df_val[feature_cols].values
X_val_sc = scaler.transform(X_val)
probas = model.predict_proba(X_val_sc)[:, 1]
df_val['proba'] = probas

print("Loading ground truth for F0.5...")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
truth_map = build_truth_map(gt)
val_s1_ids = df_val['source1_entity_id'].unique().tolist()

best_f05 = 0
best_th = 0
for th in np.arange(0.5, 0.96, 0.05):
    preds = df_val[df_val['proba'] >= th]
    pred_map = preds.groupby('source1_entity_id')['cand_entity_id'].apply(set).to_dict()
    
    f05s = []
    for sid in val_s1_ids:
        tr = truth_map.get(sid, set())
        pr = pred_map.get(sid, set())
        f05s.append(entity_f05(tr, pr))
    
    avg_f05 = np.mean(f05s)
    print(f"  TH={th:.2f}  F0.5={avg_f05:.4f}")
    if avg_f05 > best_f05:
        best_f05 = avg_f05
        best_th = th

print(f"Best validation F0.5: {best_f05:.4f} at TH={best_th:.2f}")

import pickle
with open("output/fast_matcher.pkl", "wb") as f:
    pickle.dump((scaler, model, feature_cols, best_th), f)
