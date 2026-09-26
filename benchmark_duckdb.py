"""
benchmark_duckdb.py
====================
Developer benchmark mode (requirement L).

Runs:
  - 10,000 S1 rows
  - token block only
  - S2 only

Reports:
  - peak RSS (via psutil, optional)
  - elapsed time
  - candidate pair count

Usage:
    python benchmark_duckdb.py <cache_dir> <output_tsv>

Example (Colab):
    python benchmark_duckdb.py /content/drive/MyDrive/amazon_cache /content/bench_out.tsv
"""

import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from candidate_generation import generate_candidates_memory_safe


def _rss_gb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python benchmark_duckdb.py <cache_dir> <output_tsv>")
        sys.exit(1)

    cache_dir = Path(sys.argv[1])
    out_tsv   = Path(sys.argv[2])
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("M4 BENCHMARK MODE")
    print("  S1 rows : 10,000")
    print("  blocks  : token only")
    print("  sources : S2 only")
    print("=" * 60)

    rss_before = _rss_gb()
    if rss_before is not None:
        print(f"RSS before : {rss_before:.2f} GB")

    # Build a mini cache dir with 10k S1 rows, S2 symlink (or copy on Windows)
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # S1: slice first 10,000 rows
        s1_orig  = pq.read_table(cache_dir / "train_source1.parquet")
        s1_mini  = s1_orig.slice(0, 10_000)
        pq.write_table(s1_mini, tmp / "train_source1.parquet")
        print(f"Mini S1 written: {len(s1_mini):,} rows", flush=True)

        # S2: symlink on Unix, copy on Windows (symlinks need elevated perms)
        s2_src = cache_dir / "train_source2.parquet"
        s2_dst = tmp / "train_source2.parquet"
        try:
            s2_dst.symlink_to(s2_src.resolve())
        except (OSError, NotImplementedError):
            import shutil
            shutil.copy2(s2_src, s2_dst)
        print(f"S2 ready ({pq.read_metadata(s2_dst).num_rows:,} rows)", flush=True)

        # S3: just write an empty Parquet so the function doesn't crash
        # (benchmark uses S2 only — S3 will produce 0 pairs quickly)
        s3_schema = pq.read_schema(s2_src)
        empty_s3  = pa.table({name: pa.array([], type=s3_schema.field(name).type)
                               for name in s3_schema.names})
        pq.write_table(empty_s3, tmp / "train_source3.parquet")
        print("S3 dummy written (empty)", flush=True)

        t0 = time.time()
        print("\nStarting benchmark run...\n", flush=True)

        n_s1, n_s2, n_s3 = generate_candidates_memory_safe(
            split      = "train",
            cache_dir  = tmp,
            out_file   = out_tsv,
            blocks     = ["token"],          # token only
            verbose    = True,
            chunk_size = 10_000,             # single chunk = 10k rows
            memory_limit = "6GB",
            threads    = 2,
        )

    elapsed = time.time() - t0
    rss_after = _rss_gb()

    # Count candidate pairs in output
    import duckdb
    row_count  = duckdb.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{out_tsv}', delim='\\t', header=True)"
    ).fetchone()[0]
    pair_count = duckdb.execute(
        f"SELECT SUM(len(string_split(candidate_entity_ids, ','))) "
        f"FROM read_csv_auto('{out_tsv}', delim='\\t', header=True) "
        f"WHERE candidate_entity_ids IS NOT NULL AND candidate_entity_ids != ''"
    ).fetchone()[0] or 0

    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print(f"  S1 rows processed : {n_s1:,}")
    print(f"  S2 rows indexed   : {n_s2:,}")
    print(f"  Output TSV rows   : {row_count:,}")
    print(f"  Candidate pairs   : {pair_count:,}")
    print(f"  Elapsed time      : {elapsed:.1f}s")
    if rss_before is not None and rss_after is not None:
        print(f"  RSS before        : {rss_before:.2f} GB")
        print(f"  RSS after         : {rss_after:.2f} GB")
        print(f"  RSS delta         : {rss_after - rss_before:+.2f} GB")
    print(f"  Output            : {out_tsv}")
    print("=" * 60)
