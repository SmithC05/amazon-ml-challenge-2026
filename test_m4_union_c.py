import sys
import time
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(".").resolve() / "src"))
from candidate_generation import generate_candidates_memory_safe, evaluate_candidates
from train import build_truth_map

print("Loading ground truth...")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
truth_map = build_truth_map(gt)

print("Running candidate generation UNION C (exact, token, address, prefix)...")
t0 = time.time()
generate_candidates_memory_safe(
    split="train",
    cache_dir="cache_10k",
    out_file="output/union_c_10k.tsv",
    blocks=["exact", "token", "address", "prefix"],
    token_max_df=500,
    address_max_df=500,
    prefix_max_df=500,
    db_path="m4_10k.duckdb"
)
t1 = time.time()

print(f"Candidate generation done in {t1 - t0:.2f}s")

print("Evaluating candidates...")
cands = pd.read_csv("output/union_c_10k.tsv", sep="\t", dtype=str)

n_s2 = pq.read_metadata("cache_10k/train_source2.parquet").num_rows
n_s3 = pq.read_metadata("cache_10k/train_source3.parquet").num_rows

metrics = evaluate_candidates(cands, truth_map, n_s2, n_s3)
for k, v in metrics.items():
    print(f"{k}: {v}")
