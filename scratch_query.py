import duckdb
import time

spool_path = r"d:\projects\HACKATHON\amazon ml\amazon-ml-challenge-2026\output\.pair_spool\pairs.parquet"
print('Querying spool with DuckDB...')
start = time.time()
con = duckdb.connect()

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
