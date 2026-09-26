"""
tools/evaluate_stage2_blocks.py
===============================
Stage 2 M4 blocking evaluation.

Measures candidate recall and candidate volume for the multi-block M4 pipeline 
without running the full 2.2M S1 dataset. Evaluates exact, address, prefix, 
and country_token blocks alongside token (sweeping max_df). 
Excludes fuzzy to establish a baseline before adding expensive string matching.

Ground Truth: uses BOTH S2 and S3 matches for the sampled 10,000 S1 IDs.
Generates candidates block-by-block, evaluates each, then computes the union 
and evaluates the combined candidate set.

Usage
-----
    python tools/evaluate_stage2_blocks.py \\
        --cache-dir /content/drive/MyDrive/amazon_cache \\
        --gt /content/drive/MyDrive/train_ground_truth.tsv \\
        --output-dir /content/stage2_eval \\
        --n-s1 10000 \\
        --max-df 500 1000
"""

import argparse
import csv
import sys
import time
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from candidate_generation import generate_candidates_memory_safe


def _load_s1_ids(s1_path: Path) -> list[str]:
    table = pq.read_table(s1_path, columns=["entity_id"])
    return table["entity_id"].to_pylist()


def _load_gt(gt_path: Path, s1_ids: set[str]) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {sid: set() for sid in s1_ids}
    with open(gt_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        match_col = next((c for c in fieldnames if "matched" in c.lower() or "candidate" in c.lower()), None)
        id_col = next((c for c in fieldnames if "source1" in c.lower() or c.lower() == "entity_id"), None)
        if not match_col or not id_col:
            sys.exit(f"GT columns missing. Found: {fieldnames}")

        for row in reader:
            sid = row[id_col].strip()
            if sid not in s1_ids:
                continue
            raw = row[match_col].strip() if row[match_col] else ""
            if raw:
                # Includes BOTH S2-* and S3-* (as opposed to token-only Stage 1)
                truth[sid] = {x.strip() for x in raw.split(",") if x.strip()}
    return truth


def _evaluate_tsv(tsv_path: Path, truth: dict[str, set[str]], s1_ids: list[str]) -> dict:
    tsv_path_str = str(tsv_path).replace("\\", "/")
    con = duckdb.connect()
    stats = con.execute(f"""
        SELECT 
            COALESCE(SUM(
                CASE WHEN candidate_entity_ids IS NULL OR candidate_entity_ids = '' THEN 0
                ELSE len(string_split(candidate_entity_ids, ',')) END
            ), 0) AS cand_pairs,
            COALESCE(MAX(
                CASE WHEN candidate_entity_ids IS NULL OR candidate_entity_ids = '' THEN 0
                ELSE len(string_split(candidate_entity_ids, ',')) END
            ), 0) AS max_cand,
            COUNT(*) FILTER (WHERE candidate_entity_ids IS NOT NULL AND candidate_entity_ids != '') AS s1_with_cands
        FROM read_csv_auto('{tsv_path_str}', delim='\\t', header=True)
    """).fetchone()
    con.close()

    total_cand_pairs = int(stats[0])
    max_cand = int(stats[1])
    s1_with_cands = int(stats[2])

    found = 0
    s1_fully_recovered = 0
    total_gt = sum(len(truth.get(sid, set())) for sid in s1_ids)

    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        id_col = next((c for c in fieldnames if "source1" in c.lower()), None)
        cand_col = next((c for c in fieldnames if "candidate" in c.lower()), None)
        
        for row in reader:
            sid = row[id_col].strip()
            true_set = truth.get(sid, set())
            n_true = len(true_set)
            
            if n_true > 0:
                raw = row[cand_col].strip() if row[cand_col] else ""
                if raw:
                    cand_set = {x.strip() for x in raw.split(",") if x.strip()}
                    hits = len(true_set & cand_set)
                    found += hits
                    if hits == n_true:
                        s1_fully_recovered += 1

    n_s1 = len(s1_ids)
    lost = total_gt - found
    recall = found / total_gt if total_gt else 0.0

    return {
        "cand_pairs": total_cand_pairs,
        "s1_with_cands": s1_with_cands,
        "max_per_s1": max_cand,
        "avg_per_s1": total_cand_pairs / n_s1 if n_s1 else 0.0,
        "found": found,
        "lost": lost,
        "recall": recall,
        "s1_fully_recovered": s1_fully_recovered,
        "total_gt": total_gt
    }


def _rss_gb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Stage 2 M4 blocking evaluation.")
    parser.add_argument("--cache-dir", required=True, help="Path to M2 cache containing Parquet files")
    parser.add_argument("--gt", required=True, help="Path to train_ground_truth.tsv")
    parser.add_argument("--output-dir", required=True, help="Directory to save intermediate and union TSVs")
    parser.add_argument("--n-s1", type=int, default=10000, help="Number of S1 rows to evaluate")
    parser.add_argument("--max-df", nargs="+", type=int, default=[500, 1000], help="token_max_df thresholds to sweep")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    gt_path = Path(args.gt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_s1 = args.n_s1

    print("=" * 60)
    print("STAGE 2 M4 EVALUATION (No Fuzzy)")
    print(f"S1 rows : {n_s1:,}")
    print("=" * 60)

    # 1. Setup mini cache
    mini_cache = out_dir / "mini_cache"
    mini_cache.mkdir(exist_ok=True)
    
    s1_orig = pq.read_table(cache_dir / "train_source1.parquet")
    s1_mini = s1_orig.slice(0, n_s1)
    pq.write_table(s1_mini, mini_cache / "train_source1.parquet")
    
    # Symlink (or copy) S2 and S3 for full rhs coverage
    for src in ["source2", "source3"]:
        src_path = cache_dir / f"train_{src}.parquet"
        dst_path = mini_cache / f"train_{src}.parquet"
        if not dst_path.exists():
            try:
                dst_path.symlink_to(src_path.resolve())
            except (OSError, NotImplementedError):
                print(f"Copying {src_path.name} (symlink failed)...")
                shutil.copy2(src_path, dst_path)

    s1_ids = _load_s1_ids(mini_cache / "train_source1.parquet")
    truth = _load_gt(gt_path, set(s1_ids))

    # 2. Run blocks independently
    static_blocks = ["exact", "address", "prefix", "country_token"]
    block_metrics = {}

    for b in static_blocks:
        print(f"\n--- Running block: {b} ---")
        out_tsv = out_dir / f"block_{b}.tsv"
        t0 = time.time()
        rss_before = _rss_gb()
        
        generate_candidates_memory_safe(
            split="train",
            cache_dir=mini_cache,
            out_file=out_tsv,
            blocks=[b],
            verbose=False,
            chunk_size=10_000,
            threads=2,
            db_path=out_dir / "m4_work.duckdb"
        )
        
        elapsed = time.time() - t0
        rss_after = _rss_gb()
        metrics = _evaluate_tsv(out_tsv, truth, s1_ids)
        metrics['elapsed'] = elapsed
        metrics['rss_delta'] = (rss_after - rss_before) if rss_before and rss_after else 0.0
        block_metrics[b] = metrics
        
        print(f"Done in {elapsed:.1f}s. Recall: {metrics['recall']:.2%}, Pairs: {metrics['cand_pairs']:,}")

    # 3. Run token blocks
    for mdf in args.max_df:
        b = f"token_{mdf}"
        print(f"\n--- Running block: {b} ---")
        out_tsv = out_dir / f"block_{b}.tsv"
        t0 = time.time()
        rss_before = _rss_gb()
        
        generate_candidates_memory_safe(
            split="train",
            cache_dir=mini_cache,
            out_file=out_tsv,
            blocks=["token"],
            verbose=False,
            chunk_size=10_000,
            threads=2,
            token_max_df=mdf,
            db_path=out_dir / "m4_work.duckdb"
        )
        
        elapsed = time.time() - t0
        rss_after = _rss_gb()
        metrics = _evaluate_tsv(out_tsv, truth, s1_ids)
        metrics['elapsed'] = elapsed
        metrics['rss_delta'] = (rss_after - rss_before) if rss_before and rss_after else 0.0
        block_metrics[b] = metrics
        
        print(f"Done in {elapsed:.1f}s. Recall: {metrics['recall']:.2%}, Pairs: {metrics['cand_pairs']:,}")

    # 4. Compute Unions using DuckDB
    con = duckdb.connect(str(out_dir / "union.duckdb"))
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA preserve_insertion_order=false")
    tmp_dir_str = str(out_dir).replace('\\', '/')
    con.execute(f"PRAGMA temp_directory='{tmp_dir_str}'")
    
    union_metrics = {}
    for mdf in args.max_df:
        print(f"\n--- Computing UNION for max_df={mdf} ---")
        union_tsv = out_dir / f"union_maxdf{mdf}.tsv"
        
        # Files to union: the static blocks + the specific token block
        tsv_files = [str(out_dir / f"block_{b}.tsv") for b in static_blocks]
        tsv_files.append(str(out_dir / f"block_token_{mdf}.tsv"))
        
        union_queries = []
        for f in tsv_files:
            f_esc = f.replace('\\', '/')
            union_queries.append(
                f"SELECT source1_entity_id, unnest(string_split(candidate_entity_ids, ',')) as candidate_entity_id "
                f"FROM read_csv_auto('{f_esc}', delim='\\t', header=True) "
                f"WHERE candidate_entity_ids != ''"
            )
            
        full_union_query = " UNION ".join(union_queries)
        s1_parquet = str(mini_cache / "train_source1.parquet").replace('\\', '/')
        union_tsv_str = str(union_tsv).replace('\\', '/')
        
        con.execute(f"""
            COPY (
                WITH all_pairs AS (
                    {full_union_query}
                ),
                distinct_pairs AS (
                    SELECT DISTINCT source1_entity_id, candidate_entity_id 
                    FROM all_pairs
                ),
                agg_pairs AS (
                    SELECT source1_entity_id, string_agg(candidate_entity_id, ',') AS candidate_entity_ids
                    FROM distinct_pairs
                    GROUP BY source1_entity_id
                ),
                s1_all AS (
                    SELECT entity_id AS source1_entity_id FROM read_parquet('{s1_parquet}')
                )
                SELECT
                    s1_all.source1_entity_id,
                    COALESCE(agg_pairs.candidate_entity_ids, '') AS candidate_entity_ids
                FROM s1_all
                LEFT JOIN agg_pairs ON s1_all.source1_entity_id = agg_pairs.source1_entity_id
            ) TO '{union_tsv_str}' (FORMAT CSV, DELIMITER '\\t', HEADER)
        """)
        
        metrics = _evaluate_tsv(union_tsv, truth, s1_ids)
        union_metrics[mdf] = metrics
        print(f"Union max_df={mdf} Recall: {metrics['recall']:.2%}, Pairs: {metrics['cand_pairs']:,}")

    con.close()

    # 5. Print final report
    print("\n" + "="*80)
    print("STAGE 2 EVALUATION REPORT (10k S1, S2+S3, No Fuzzy)")
    print("="*80)
    
    print("\n1. PER-BLOCK PERFORMANCE")
    print(f"{'Block':<15} | {'Pairs':>10} | {'S1 w/Cands':>10} | {'Found':>8} | {'Lost':>8} | {'Recall':>7}")
    print("-" * 72)
    for b in static_blocks:
        m = block_metrics[b]
        print(f"{b:<15} | {m['cand_pairs']:>10,} | {m['s1_with_cands']:>10,} | {m['found']:>8,} | {m['lost']:>8,} | {m['recall']:>7.2%}")
        
    for mdf in args.max_df:
        b = f"token_{mdf}"
        m = block_metrics[b]
        print(f"{b:<15} | {m['cand_pairs']:>10,} | {m['s1_with_cands']:>10,} | {m['found']:>8,} | {m['lost']:>8,} | {m['recall']:>7.2%}")

    print("\n2. UNION PERFORMANCE")
    for mdf in args.max_df:
        m = union_metrics[mdf]
        print(f"\n--- UNION (token_max_df={mdf}) ---")
        print(f"  Total unique pairs      : {m['cand_pairs']:,}")
        print(f"  Avg candidates / S1     : {m['avg_per_s1']:.1f}")
        print(f"  Max candidates / S1     : {m['max_per_s1']:,}")
        print(f"  S1 with >=1 candidate   : {m['s1_with_cands']:,}")
        print(f"  Total true matches      : {m['total_gt']:,}")
        print(f"  True matches found      : {m['found']:,}")
        print(f"  True matches lost       : {m['lost']:,}")
        print(f"  Candidate recall        : {m['recall']:.2%}")
        print(f"  S1 all true recovered   : {m['s1_fully_recovered']:,}")

    print("\n" + "="*80)
    print("NEXT STEP: Compare the Union blocks above. Evaluate fuzzy only if recall is lacking")
    print("and the total candidate pool size leaves room for fuzzy expansion.")
    print("="*80)


if __name__ == "__main__":
    main()
