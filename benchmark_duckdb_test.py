import duckdb
import time

con = duckdb.connect()

print("Expanding 1M test candidates...")
con.execute("""
    CREATE OR REPLACE TABLE test_candidates AS 
    SELECT 
        source1_entity_id, 
        UNNEST(STRING_SPLIT(candidate_entity_ids, ',')) as cand_entity_id
    FROM read_csv_auto('dataset/candidate_pairs.tsv', header=True)
    WHERE candidate_entity_ids IS NOT NULL AND candidate_entity_ids != ''
    LIMIT 10000;
""")
con.execute("CREATE OR REPLACE TABLE test_1m AS SELECT * FROM test_candidates LIMIT 1000000;")

print("Materializing RHS with index (approx 1-2 mins)...")
t_idx_start = time.time()
con.execute("""
    CREATE OR REPLACE TABLE rhs AS 
    SELECT entity_id, name_norm, address_norm, country FROM read_parquet('cache/test_source2.parquet')
    UNION ALL
    SELECT entity_id, name_norm, address_norm, country FROM read_parquet('cache/test_source3.parquet');
""")
con.execute("CREATE UNIQUE INDEX rhs_idx ON rhs(entity_id);")
print(f"RHS materialized in {time.time() - t_idx_start:.2f}s")

print("Materializing S1 with index...")
con.execute("""
    CREATE OR REPLACE TABLE s1 AS 
    SELECT entity_id, name_norm, address_norm, country FROM read_parquet('cache/test_source1.parquet');
""")
con.execute("CREATE UNIQUE INDEX s1_idx ON s1(entity_id);")

query = """
    SELECT 
        c.source1_entity_id, 
        c.cand_entity_id,
        (s1.name_norm == s2.name_norm)::INT as name_exact,
        (s1.address_norm == s2.address_norm)::INT as address_exact,
        (s1.country == s2.country)::INT as country_match,
        ABS(LENGTH(s1.name_norm) - LENGTH(s2.name_norm)) / GREATEST(LENGTH(s1.name_norm), LENGTH(s2.name_norm), 1) as name_length_diff,
        ABS(LENGTH(s1.address_norm) - LENGTH(s2.address_norm)) / GREATEST(LENGTH(s1.address_norm), LENGTH(s2.address_norm), 1) as address_length_diff,
        CASE WHEN LENGTH(s1.name_norm) >= 2 AND LENGTH(s2.name_norm) >= 2 THEN jaccard(s1.name_norm, s2.name_norm) ELSE 0.0 END as name_jaccard,
        CASE WHEN LENGTH(s1.address_norm) >= 2 AND LENGTH(s2.address_norm) >= 2 THEN jaccard(s1.address_norm, s2.address_norm) ELSE 0.0 END as address_jaccard,
        levenshtein(s1.name_norm, s2.name_norm) as name_levenshtein,
        levenshtein(s1.address_norm, s2.address_norm) as address_levenshtein,
        jaro_winkler_similarity(s1.name_norm, s2.name_norm) as name_jaro,
        jaro_winkler_similarity(s1.address_norm, s2.address_norm) as address_jaro
    FROM test_1m c
    JOIN s1 ON c.source1_entity_id = s1.entity_id
    JOIN rhs s2 ON c.cand_entity_id = s2.entity_id
"""

print("Benchmarking JOIN + FEATURES...")
t0 = time.time()
df = con.execute(query).df()
t1 = time.time()
elapsed = t1 - t0
print(f"DuckDB Test Benchmark: {elapsed:.2f} s -> {len(df)/elapsed:.0f} pairs/sec (rows {len(df)})")
