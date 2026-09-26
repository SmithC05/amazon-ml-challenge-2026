import duckdb
import time
from pathlib import Path
import hashlib
import sys

def get_file_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def run_baseline():
    start = time.time()
    
    # DuckDB config
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='2GB'")
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA temp_directory='emergency_tmp'")

    s1_path = "dataset/processed/m2_cache/test_source1.parquet"
    s2_path = "dataset/processed/m2_cache/test_source2.parquet"
    s3_path = "dataset/processed/m2_cache/test_source3.parquet"
    out_path = "output/trial_submission/matching_results.tsv"

    print("Creating views...")
    con.execute(f"""
        CREATE VIEW s1 AS SELECT entity_id as id, name_norm, address_norm FROM '{s1_path}';
        CREATE VIEW s2 AS SELECT entity_id as id, name_norm, address_norm FROM '{s2_path}';
        CREATE VIEW s3 AS SELECT entity_id as id, name_norm, address_norm FROM '{s3_path}';
    """)

    bucket_limit = 50
    print(f"Finding valid S2/S3 name buckets (max size {bucket_limit})...")
    con.execute(f"""
        CREATE TABLE s2_name_buckets AS 
        SELECT name_norm, id as s2_id 
        FROM s2 
        WHERE name_norm IS NOT NULL AND name_norm != '' 
          AND name_norm IN (SELECT name_norm FROM s2 WHERE name_norm IS NOT NULL AND name_norm != '' GROUP BY name_norm HAVING count(*) <= {bucket_limit});

        CREATE TABLE s3_name_buckets AS 
        SELECT name_norm, id as s3_id 
        FROM s3 
        WHERE name_norm IS NOT NULL AND name_norm != '' 
          AND name_norm IN (SELECT name_norm FROM s3 WHERE name_norm IS NOT NULL AND name_norm != '' GROUP BY name_norm HAVING count(*) <= {bucket_limit});
          
        CREATE TABLE s2_addr_buckets AS 
        SELECT address_norm, id as s2_id 
        FROM s2 
        WHERE address_norm IS NOT NULL AND address_norm != '' 
          AND address_norm IN (SELECT address_norm FROM s2 WHERE address_norm IS NOT NULL AND address_norm != '' GROUP BY address_norm HAVING count(*) <= {bucket_limit});

        CREATE TABLE s3_addr_buckets AS 
        SELECT address_norm, id as s3_id 
        FROM s3 
        WHERE address_norm IS NOT NULL AND address_norm != '' 
          AND address_norm IN (SELECT address_norm FROM s3 WHERE address_norm IS NOT NULL AND address_norm != '' GROUP BY address_norm HAVING count(*) <= {bucket_limit});
    """)

    print("Executing joins...")
    con.execute("""
        CREATE TABLE matches AS 
        SELECT s1.id as s1_id, s2.s2_id as match_id
        FROM s1 JOIN s2_name_buckets s2 ON s1.name_norm = s2.name_norm
        WHERE s1.name_norm IS NOT NULL AND s1.name_norm != ''
        
        UNION 
        
        SELECT s1.id as s1_id, s3.s3_id as match_id
        FROM s1 JOIN s3_name_buckets s3 ON s1.name_norm = s3.name_norm
        WHERE s1.name_norm IS NOT NULL AND s1.name_norm != ''
        
        UNION 
        
        SELECT s1.id as s1_id, s2.s2_id as match_id
        FROM s1 JOIN s2_addr_buckets s2 ON s1.address_norm = s2.address_norm
        WHERE s1.address_norm IS NOT NULL AND s1.address_norm != ''
        
        UNION 
        
        SELECT s1.id as s1_id, s3.s3_id as match_id
        FROM s1 JOIN s3_addr_buckets s3 ON s1.address_norm = s3.address_norm
        WHERE s1.address_norm IS NOT NULL AND s1.address_norm != ''
    """)

    print("Aggregating results...")
    con.execute("""
        CREATE TABLE final_output AS
        SELECT s1.id as source1_entity_id, string_agg(matches.match_id, ',') as matched_entity_ids
        FROM s1
        LEFT JOIN matches ON s1.id = matches.s1_id
        GROUP BY s1.id
    """)

    print(f"Writing to {out_path}...")
    # Add ORDER BY to ensure deterministic output for the final TSV
    con.execute(f"""
        COPY (
            SELECT source1_entity_id, coalesce(matched_entity_ids, '') as matched_entity_ids 
            FROM final_output
            ORDER BY source1_entity_id
        ) TO '{out_path}' (HEADER TRUE, DELIMITER '\t', QUOTE '');
    """)
    
    total_s1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    out_rows = con.execute("SELECT count(*) FROM final_output").fetchone()[0]
    with_matches = con.execute("SELECT count(*) FROM final_output WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != ''").fetchone()[0]
    zero_matches = total_s1 - with_matches
    total_matches = con.execute("SELECT count(*) FROM matches").fetchone()[0]

    print("--- OUTPUT STATS ---")
    print(f"Total test S1: {total_s1}")
    print(f"Output rows: {out_rows}")
    print(f"Missing S1: {total_s1 - out_rows}")
    print(f"Duplicate S1: 0")
    print(f"Extra S1: 0")
    print(f"S1 with matches: {with_matches}")
    print(f"S1 with zero matches: {zero_matches}")
    print(f"Total predicted matches: {total_matches}")
    print(f"Runtime: {time.time() - start:.2f}s")
    print(f"SHA-256: {get_file_sha256(out_path)}")

if __name__ == '__main__':
    run_baseline()
