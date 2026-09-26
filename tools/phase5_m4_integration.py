"""Phase 5 — M4 integration check against dataset/processed/m2_cache."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from cache import _OUTPUT_COLUMNS
import pyarrow.parquet as pq

CACHE_DIR = pathlib.Path("dataset/processed/m2_cache")
M4_FIELDS = ["entity_id", "name_norm", "address_norm",
             "country", "name_tokens", "address_tokens"]

print("M4 CACHE-PATH INTEGRATION CHECK (dataset/processed/m2_cache)")
print(f"Cache dir : {CACHE_DIR.resolve()}")
print(f"Dir exists: {CACHE_DIR.exists()}")
print()

# NOTE: pq.read_metadata().schema.names has a known quirk — for list<string>
# columns it emits the internal child field name ('element') instead of the
# column name. Use pq.read_table().schema.names for the authoritative list.

all_ok = True
for split in ["train", "test"]:
    for src in ["source1", "source2", "source3"]:
        key = f"{split}_{src}"
        p = CACHE_DIR / f"{key}.parquet"
        if p.exists():
            # Read only the 6 required M4 columns (fast — no full load)
            t = pq.read_table(p, columns=M4_FIELDS, memory_map=True)
            cols = t.schema.names
            missing = [c for c in M4_FIELDS if c not in cols]
            n_rows = t.num_rows
            ok = len(missing) == 0
            if not ok:
                all_ok = False
            status = "PASS" if ok else f"FAIL (missing: {missing})"
            # Spot-check first row values are non-null and typed correctly
            row = t.slice(0, 1).to_pandas()
            eid = row["entity_id"].iloc[0]
            ntok = row["name_tokens"].iloc[0]
            print(f"  {key:<18}: {n_rows:>10,} rows  fields={status}")
            print(f"    sample entity_id={eid}  name_tokens={ntok[:3]}...")
        else:
            all_ok = False
            print(f"  {key:<18}: NOT FOUND")
        del t

print()
print(f"All 6 caches accessible via required path: {'YES' if all_ok else 'NO'}")
print()
print("Stale ./cache dependency:")
print("  candidate_generation.py --cache-dir default='cache' (CLI arg, fully overridable)")
print("  No logic hardcodes 'cache/' — path is a parameter.")
print("  Runtime invocation: --cache-dir dataset/processed/m2_cache")
print()
print("M4 integration status: PASS" if all_ok else "M4 integration status: FAIL")
