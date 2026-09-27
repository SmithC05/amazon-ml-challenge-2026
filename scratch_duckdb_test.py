import duckdb
import time

spool_path = r"d:\projects\HACKATHON\amazon ml\amazon-ml-challenge-2026\output\.pair_spool\pairs.parquet"
con = duckdb.connect()

print('Counting positives and negatives...')
start = time.time()
query = f"""
    SELECT 
        SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as pos_count,
        SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) as neg_count
    FROM '{spool_path}'
    WHERE is_val = 0
"""
res = con.execute(query).fetchone()
pos = int(res[0]) if res[0] is not None else 0
neg = int(res[1]) if res[1] is not None else 0
print(f'Train Positives: {pos:,}')
print(f'Train Negatives: {neg:,}')
print(f'Query took: {time.time() - start:.2f} seconds')

n_neg_budget = min(neg, pos)
print(f'Target negatives to sample: {n_neg_budget:,}')

print('Testing DuckDB deterministic selection...')
start = time.time()
sample_query = f"""
    SELECT *
    FROM '{spool_path}'
    WHERE is_val = 0 AND label = 0
    ORDER BY hash(source1_entity_id || cand_entity_id)
    LIMIT {n_neg_budget}
"""
neg_df = con.execute(sample_query).df()
print(f'DuckDB selection took: {time.time() - start:.2f} seconds')
print(f'Result shape: {neg_df.shape}')
print(neg_df.head(2))

