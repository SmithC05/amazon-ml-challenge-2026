import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import pyarrow.parquet as pq

# Monkeypatch features._levenshtein
import features
import Levenshtein
import rapidfuzz

def _levenshtein_fast(a: str, b: str) -> int:
    return Levenshtein.distance(a, b)

features._levenshtein = _levenshtein_fast

def run_benchmark(n_pairs):
    pf = pq.ParquetFile(r"output\.pair_spool\pairs.parquet")
    batch = next(pf.iter_batches(batch_size=n_pairs))
    df = batch.to_pandas()
    pair_dicts = df.to_dict("records")
    
    start = time.time()
    _ = features.build_feature_matrix_from_dicts(pair_dicts)
    end = time.time()
    
    elapsed = end - start
    throughput = n_pairs / elapsed
    print(f"FAST {n_pairs} pairs: {elapsed:.2f} sec (Throughput: {throughput:.2f} pairs/sec)")
    return pair_dicts

if __name__ == "__main__":
    print("--- FAST BENCHMARK ---")
    run_benchmark(10000)
    run_benchmark(50000)
    run_benchmark(100000)
