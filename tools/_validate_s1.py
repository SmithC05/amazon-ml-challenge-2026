import sys, time
sys.path.insert(0, 'src')
sys.stdout.reconfigure(encoding='utf-8')
from cache import load_cache, validate_cache
from preprocess import normalize_name, normalize_address

# validate train_source1
vr = validate_cache('train', 'source1', 'cache')
print("validate_cache result:")
for k, v in vr.items():
    print(f"  {k}: {v}")

df = load_cache('train', 'source1', 'cache')
print(f"\nshape: {df.shape}")
print(f"columns: {list(df.columns)}")

# spot check
sample = df[df['business_name'].notna()].head(5)
print("\nSpot checks:")
for _, row in sample.iterrows():
    en = normalize_name(str(row.get('business_name') or ''))
    ea = normalize_address(str(row.get('business_address') or ''))
    nk = str(row['name_norm'] or '') == en
    ak = str(row['address_norm'] or '') == ea
    eid = row['entity_id']
    ntk = row['name_token_count']
    atk = row['address_token_count']
    nm  = 'OK' if nk else 'FAIL'
    am  = 'OK' if ak else 'FAIL'
    print(f"  {eid}  name={nm}  addr={am}  ntoks={ntk}  atoks={atk}")

# load timing
t0 = time.perf_counter()
df2 = load_cache('train', 'source1', 'cache')
lt = round(time.perf_counter() - t0, 3)
print(f"\nload_cache time: {lt}s  rows={len(df2):,}")
del df, df2
