import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from candidate_generation import generate_candidates_memory_safe
import pyarrow.parquet as pq

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python benchmark_duckdb.py <cache_dir> <output_tsv>")
        sys.exit(1)
        
    cache_dir = sys.argv[1]
    out_tsv = sys.argv[2]
    
    print(f"Running benchmark on first 10,000 rows of S1 in {cache_dir}")
    
    # We will temporarily modify the s1_path read inside the duckdb function 
    # to only take 10k rows. 
    # Actually it's easier to just create a mini cache dir with 10k rows
    import tempfile
    import pyarrow as pa
    
    with tempfile.TemporaryDirectory() as temp_cache:
        temp_cache_path = Path(temp_cache)
        
        # S1: take 10,000 rows
        s1_orig = pq.read_table(Path(cache_dir) / "train_source1.parquet")
        s1_mini = s1_orig.slice(0, 10000)
        pq.write_table(s1_mini, temp_cache_path / "train_source1.parquet")
        
        # S2 and S3: link or copy
        # Since duckdb reads them directly, we can just symlink them
        (temp_cache_path / "train_source2.parquet").symlink_to(Path(cache_dir) / "train_source2.parquet")
        (temp_cache_path / "train_source3.parquet").symlink_to(Path(cache_dir) / "train_source3.parquet")
        
        print("Starting memory-safe generation...")
        n_s1, n_s2, n_s3 = generate_candidates_memory_safe(
            split="train",
            cache_dir=temp_cache_path,
            out_file=out_tsv,
            verbose=True,
            chunk_size=10000
        )
        
        print(f"Benchmark finished. output saved to {out_tsv}")
