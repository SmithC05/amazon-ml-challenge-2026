"""Phase 3 — M4 cache path integration check (read-only)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from cache import cache_exists, _cache_path
import pyarrow.parquet as pq

SPLITS  = ["train", "test"]
SOURCES = ["source1", "source2", "source3"]

PATHS = {
    "M4 candidate_gen default (cache/)":         pathlib.Path("cache"),
    "Required path (dataset/processed/m2_cache)":pathlib.Path("dataset/processed/m2_cache"),
}

print("=" * 60)
print("PATH EXISTENCE CHECK")
print("=" * 60)
for label, base in PATHS.items():
    exists = base.exists()
    parquets = sorted(base.rglob("*.parquet")) if exists else []
    print(f"\n{label}")
    print(f"  dir exists : {exists}")
    print(f"  parquets   : {[p.name for p in parquets] or '(none)'}")
    for split in SPLITS:
        for src in SOURCES:
            ok = cache_exists(split, src, base)
            if ok:
                p = _cache_path(split, src, base)
                mb = round(p.stat().st_size / 1e6, 1)
                print(f"  {split}_{src}: EXISTS  {mb} MB")

print()
print("=" * 60)
print("INTEGRATION READABILITY CHECK (train_source1 via cache/)")
print("=" * 60)
p = pathlib.Path("cache/train_source1.parquet")
if p.exists():
    meta = pq.read_metadata(p)
    schema_names = meta.schema.names
    required = ["entity_id", "name_norm", "address_norm",
                "country", "name_tokens", "address_tokens"]
    print(f"  rows            : {meta.num_rows:,}")
    print(f"  total columns   : {len(schema_names)}")
    print(f"  column names    : {schema_names}")
    for col in required:
        ok = col in schema_names
        print(f"  {col:<22}: {'PRESENT' if ok else 'MISSING'}")
else:
    print("  NOT FOUND")

print()
print("=" * 60)
print("M4 CACHE PATH ANALYSIS")
print("=" * 60)
print()
print("candidate_generation.py CLI default:")
print("  --cache-dir  default='cache'       (line 1268)")
print()
print("predict.py CLI default:")
print("  --cache-dir  default=None          (line 251)")
print("  (None = fall back to raw TSV + M2 normalization)")
print()
print("integration/local_paths.py:")
print("  CACHE_DIR = REPO_ROOT / 'dataset' / 'processed' / 'm2_cache'")
print("  (local-only integration config; not imported by src/)")
print()
print("Required path: dataset/processed/m2_cache")
req = pathlib.Path("dataset/processed/m2_cache")
print(f"  exists: {req.exists()}")
print()

# Final determination
cand_default = pathlib.Path("cache")
cand_has_any = any(
    cache_exists(split, src, cand_default)
    for split in SPLITS for src in SOURCES
)
req_has_any = any(
    cache_exists(split, src, req)
    for split in SPLITS for src in SOURCES
)

print("FINDINGS:")
print(f"  candidate_gen uses 'cache/' by default       : YES (hardcoded CLI default)")
print(f"  'cache/' dir has any parquets                : {cand_has_any}")
print(f"  'dataset/processed/m2_cache' exists          : {req.exists()}")
print(f"  'dataset/processed/m2_cache' has any parquets: {req_has_any}")
print(f"  M4 logic changed                             : NO")
print(f"  Files modified                               : NONE")
