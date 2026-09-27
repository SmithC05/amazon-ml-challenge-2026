import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cProfile
import pstats
import pyarrow.parquet as pq
import pandas as pd
from features import build_feature_matrix_from_dicts

def run_benchmark(n_pairs):
    pf = pq.ParquetFile(r"output\.pair_spool\pairs.parquet")
    batch = next(pf.iter_batches(batch_size=n_pairs))
    df = batch.to_pandas()
    pair_dicts = df.to_dict("records")
    
    start = time.time()
    _ = build_feature_matrix_from_dicts(pair_dicts)
    end = time.time()
    
    elapsed = end - start
    throughput = n_pairs / elapsed
    print(f"{n_pairs} pairs: {elapsed:.2f} sec (Throughput: {throughput:.2f} pairs/sec)")
    return pair_dicts

if __name__ == "__main__":
    print("--- CURRENT BENCHMARK ---")
    run_benchmark(10000)
    run_benchmark(50000)
    pair_dicts_100k = run_benchmark(100000)
    
    print("\n--- PROFILING 100K ---")
    profiler = cProfile.Profile()
    profiler.enable()
    build_feature_matrix_from_dicts(pair_dicts_100k)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(10)
    
    print("\n--- INSTALLED LIBRARIES ---")
    try:
        import Levenshtein
        print("Levenshtein: YES")
    except:
        print("Levenshtein: NO")
    try:
        import editdistance
        print("editdistance: YES")
    except:
        print("editdistance: NO")
    try:
        import rapidfuzz
        print("rapidfuzz: YES")
    except:
        print("rapidfuzz: NO")
