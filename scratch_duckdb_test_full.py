import duckdb
import pandas as pd
import time
from pathlib import Path

def _select_train_pairs_from_spool(
    spool_path: Path,
    max_pairs: int,
    random_state: int = 42,
) -> pd.DataFrame:
    con = duckdb.connect()

    res = con.execute(f"""
        SELECT 
            SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as pos_count,
            SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) as neg_count
        FROM '{spool_path}'
        WHERE is_val = 0
    """).fetchone()

    n_pos = int(res[0]) if res[0] is not None else 0
    n_neg_avail = int(res[1]) if res[1] is not None else 0

    if n_pos >= max_pairs:
        n_neg_budget = min(n_neg_avail, n_pos)
        effective_cap = n_pos + n_neg_budget
    else:
        n_neg_budget = max_pairs - n_pos

    query = f"""
        SELECT * EXCLUDE (is_val) FROM (
            SELECT * FROM '{spool_path}'
            WHERE is_val = 0 AND label = 1
            
            UNION ALL
            
            SELECT * FROM (
                SELECT * FROM '{spool_path}'
                WHERE is_val = 0 AND label = 0
                ORDER BY hash(source1_entity_id || cand_entity_id || '{random_state}')
                LIMIT {n_neg_budget}
            )
        )
    """
    
    print("Executing query...")
    start = time.time()
    result = con.execute(query).df()
    print(f"Query completed in {time.time()-start:.2f} seconds")
    
    n_used = len(result)
    n_neg_used = n_used - n_pos

    print(f"Sampled negatives    : {n_neg_used:,} / {n_neg_avail:,} available")
    print(f"Final training pairs : {n_used:,}  (pos={n_pos:,}  neg={n_neg_used:,})")
    
    return result

spool_path = Path(r"d:\projects\HACKATHON\amazon ml\amazon-ml-challenge-2026\output\.pair_spool\pairs.parquet")
df1 = _select_train_pairs_from_spool(spool_path, max_pairs=1000)
# run again to test determinism
df2 = _select_train_pairs_from_spool(spool_path, max_pairs=1000)

assert df1.equals(df2), "Dataframes are not identical!"
print("Determinism test PASSED")
