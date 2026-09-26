"""
benchmark_duckdb.py
====================
Developer benchmark mode (requirement L).

Runs:
  - 10,000 S1 rows
  - token block only
  - S2 only

Sweeps token_max_df across [100, 500, 1000] by default, or a single
value if --max-df is passed.

Reports per sweep:
  - token index stats (total / usable / max_df / median_df)
  - elapsed time
  - candidate pair count
  - peak RSS delta (via psutil, optional)

Usage:
    python benchmark_duckdb.py <cache_dir> <output_tsv> [--max-df N [N ...]]

Examples (Colab):
    # sweep three values (default)
    python benchmark_duckdb.py /content/drive/MyDrive/cache /content/bench.tsv

    # single value
    python benchmark_duckdb.py /content/drive/MyDrive/cache /content/bench.tsv --max-df 200
"""

import argparse
import shutil
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


def _run_one(cache_dir: Path, out_tsv: Path, max_df: int) -> dict:
    """Run one benchmark configuration and return a results dict."""
    import duckdb

    rss_before = _rss_gb()
    t0 = time.time()

    n_s1, n_s2, _ = generate_candidates_memory_safe(
        split        = "train",
        cache_dir    = cache_dir,
        out_file     = out_tsv,
        blocks       = ["token"],
        verbose      = True,
        chunk_size   = 10_000,
        memory_limit = "6GB",
        threads      = 2,
        token_max_df = max_df,
    )

    elapsed   = time.time() - t0
    rss_after = _rss_gb()

    # Count candidate pairs in output
    pair_count = duckdb.execute(
        f"SELECT SUM(len(string_split(candidate_entity_ids, ','))) "
        f"FROM read_csv_auto('{out_tsv}', delim='\\t', header=True) "
        f"WHERE candidate_entity_ids IS NOT NULL AND candidate_entity_ids != ''"
    ).fetchone()[0] or 0

    return {
        "max_df":     max_df,
        "n_s1":       n_s1,
        "n_s2":       n_s2,
        "pairs":      pair_count,
        "avg_pairs":  round(pair_count / n_s1, 1) if n_s1 else 0,
        "elapsed_s":  round(elapsed, 1),
        "rss_before": rss_before,
        "rss_after":  rss_after,
    }


def main():
    parser = argparse.ArgumentParser(description="M4 token benchmark: sweeps token_max_df")
    parser.add_argument("cache_dir", help="Path to M2 Parquet cache dir")
    parser.add_argument("output_tsv", help="Path for the output TSV")
    parser.add_argument(
        "--max-df", nargs="+", type=int, default=[100, 500, 1000],
        metavar="N",
        help="One or more token_max_df values to sweep (default: 100 500 1000)",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_base  = Path(args.output_tsv)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # ── Build mini S1 (10k rows) + S2 symlink/copy + empty S3 ────────────────
    import tempfile
    print("=" * 60)
    print("M4 BENCHMARK MODE  (token block only, S2 only, 10k S1 rows)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # S1: first 10,000 rows
        s1_orig = pq.read_table(cache_dir / "train_source1.parquet")
        s1_mini = s1_orig.slice(0, 10_000)
        pq.write_table(s1_mini, tmp / "train_source1.parquet")
        print(f"Mini S1 : {len(s1_mini):,} rows", flush=True)

        # S2: symlink (Unix) or copy (Windows/elevated required)
        s2_src = cache_dir / "train_source2.parquet"
        s2_dst = tmp / "train_source2.parquet"
        try:
            s2_dst.symlink_to(s2_src.resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(s2_src, s2_dst)
        print(f"S2      : {pq.read_metadata(s2_dst).num_rows:,} rows", flush=True)

        # S3: empty schema-matching Parquet (benchmark is S2-only)
        s3_schema = pq.read_schema(s2_src)
        empty_s3  = pa.table({
            name: pa.array([], type=s3_schema.field(name).type)
            for name in s3_schema.names
        })
        pq.write_table(empty_s3, tmp / "train_source3.parquet")
        print("S3      : empty (benchmark uses S2 only)", flush=True)
        print()

        results = []
        for max_df in args.max_df:
            print(f"\n{'─'*60}")
            print(f"  token_max_df = {max_df}")
            print(f"{'─'*60}")
            out_tsv = out_base.with_stem(f"{out_base.stem}_maxdf{max_df}")
            try:
                r = _run_one(tmp, out_tsv, max_df)
                results.append(r)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                import traceback; traceback.print_exc()

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print(f"{'max_df':>8}  {'pairs':>14}  {'avg/S1':>9}  {'time(s)':>8}  {'RSS Δ(GB)':>10}")
    print("-" * 60)
    for r in results:
        rss_delta = (
            f"{r['rss_after'] - r['rss_before']:+.2f}"
            if r["rss_before"] is not None and r["rss_after"] is not None
            else "   n/a"
        )
        print(
            f"{r['max_df']:>8}  {r['pairs']:>14,}  {r['avg_pairs']:>9,.1f}"
            f"  {r['elapsed_s']:>8.1f}  {rss_delta:>10}"
        )
    print("=" * 60)
    print()
    print("TARGET: <=1-2M pairs for 10k S1 (<=100-200 avg/S1)")
    print("Do NOT run full-data generation until benchmark passes.")


if __name__ == "__main__":
    main()
