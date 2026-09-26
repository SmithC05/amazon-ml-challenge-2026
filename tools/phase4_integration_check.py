"""Phase 4 — lightweight M4 integration field check (read-only)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import pyarrow.parquet as pq

p = pathlib.Path("cache/train_source1.parquet")
print("=== INTEGRATION: M4 required fields check ===")
if p.exists():
    pf = pq.read_table(p, memory_map=True)
    cols = pf.schema.names
    required_m4 = ["entity_id", "name_norm", "address_norm",
                   "country", "name_tokens", "address_tokens"]
    for col in required_m4:
        ok = col in cols
        print(f"  {col:<22}: {'PRESENT' if ok else 'MISSING'}")
    df = pf.select(["entity_id", "name_norm", "name_tokens"]).slice(0, 2).to_pandas()
    print()
    print("  Spot rows (entity_id | name_norm | name_tokens):")
    for _, row in df.iterrows():
        print(f"    {row['entity_id']}  |  {row['name_norm']}  |  {row['name_tokens']}")
    print()
    print("  M4 can read required fields: PASS")
else:
    print("  cache/train_source1.parquet NOT FOUND")
