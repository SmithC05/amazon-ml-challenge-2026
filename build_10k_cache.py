import duckdb
from pathlib import Path
Path('cache_10k').mkdir(exist_ok=True)
con = duckdb.connect()
con.execute("COPY (SELECT * FROM read_parquet('cache/train_source1.parquet') LIMIT 10000) TO 'cache_10k/train_source1.parquet' (FORMAT 'parquet')")
con.execute("COPY (SELECT * FROM read_parquet('cache/train_source2.parquet')) TO 'cache_10k/train_source2.parquet' (FORMAT 'parquet')")
con.execute("COPY (SELECT * FROM read_parquet('cache/train_source3.parquet')) TO 'cache_10k/train_source3.parquet' (FORMAT 'parquet')")
print('10K cache built.')
