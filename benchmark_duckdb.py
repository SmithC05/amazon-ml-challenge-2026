import duckdb
import time

con = duckdb.connect()

print("Testing DuckDB string functions:")
try:
    res = con.execute("SELECT jaccard('hello', 'hallo'), jaro_similarity('hello', 'hallo'), levenshtein('hello', 'hallo')").fetchone()
    print(f"Functions available: {res}")
except Exception as e:
    print(f"Error: {e}")
    try:
        res = con.execute("SELECT jaro_similarity('hello', 'hallo'), levenshtein('hello', 'hallo')").fetchone()
        print(f"Without Jaccard: {res}")
    except Exception as e2:
        print(f"Error2: {e2}")

# Run benchmark on 100K pairs
print("\nRunning DuckDB benchmark on 100K pairs...")
con.execute("""
    CREATE OR REPLACE TABLE sample_pairs AS 
    SELECT * FROM 'output/.pair_spool/pairs.parquet' LIMIT 100000;
""")

query_features = """
SELECT 
    source1_entity_id, cand_entity_id, label, is_val,
    (s1_name_norm == cand_name_norm)::INT as name_exact,
    (s1_address_norm == cand_address_norm)::INT as address_exact,
    (s1_country == cand_country)::INT as country_match,
    ABS(LENGTH(s1_name_norm) - LENGTH(cand_name_norm)) / GREATEST(LENGTH(s1_name_norm), LENGTH(cand_name_norm), 1) as name_length_diff,
    ABS(LENGTH(s1_address_norm) - LENGTH(cand_address_norm)) / GREATEST(LENGTH(s1_address_norm), LENGTH(cand_address_norm), 1) as address_length_diff,
    CASE WHEN LENGTH(s1_name_norm) >= 2 AND LENGTH(cand_name_norm) >= 2 THEN jaccard(s1_name_norm, cand_name_norm) ELSE 0.0 END as name_jaccard,
    CASE WHEN LENGTH(s1_address_norm) >= 2 AND LENGTH(cand_address_norm) >= 2 THEN jaccard(s1_address_norm, cand_address_norm) ELSE 0.0 END as address_jaccard,
    levenshtein(s1_name_norm, cand_name_norm) as name_levenshtein,
    levenshtein(s1_address_norm, cand_address_norm) as address_levenshtein,
    jaro_winkler_similarity(s1_name_norm, cand_name_norm) as name_jaro,
    jaro_winkler_similarity(s1_address_norm, cand_address_norm) as address_jaro
FROM sample_pairs
"""

t0 = time.time()
try:
    df = con.execute(query_features).df()
    t1 = time.time()
    elapsed = t1 - t0
    throughput = 100000 / elapsed
    print(f"DuckDB 100K pairs: {elapsed:.3f} sec -> {throughput:.0f} pairs/sec")
    print(f"Result DataFrame shape: {df.shape}")
except Exception as e:
    print(f"Query Error: {e}")
