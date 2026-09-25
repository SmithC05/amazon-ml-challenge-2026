# Preprocessing Cache

**Amazon ML Challenge 2026 — Preprocessing & Caching Reference**

This document describes the preprocessing-cache architecture, the cached schema,
and the API used by downstream team members to load preprocessed source records.

---

## Why Caching Is Needed

Source datasets (S1, S2, S3) must be normalised before any downstream step
— blocking, candidate generation, feature engineering, or model training — can
use them.

Without caching, the normalization pipeline runs on every notebook restart and
every experiment iteration, repeating identical work on the same records.

**The problem:**
```
Experiment A: reads TSV → normalises S1 (all rows) → uses result
Experiment B: reads TSV → normalises S1 (all rows) → uses result  ← wasted
Experiment C: reads TSV → normalises S1 (all rows) → uses result  ← wasted
```

**With caching:**
```
Once: reads TSV → normalises S1 → writes Parquet cache
Experiment A: load_cache("train", "source1") → instant, no normalisation
Experiment B: load_cache("train", "source1") → instant, no normalisation
Experiment C: load_cache("train", "source1") → instant, no normalisation
```

---

## Preprocessing Flow

```
RAW TSV  (e.g. /content/train_source1.tsv)
    │
    │  pd.read_csv(..., sep='\t', dtype=str)
    ▼
RAW DATAFRAME  (entity_id, business_name, business_address, country)
    │
    │  src.preprocess.preprocess_dataframe(df)
    │  ├─ normalize_name(business_name)    → name_norm
    │  ├─ normalize_address(business_address) → address_norm
    │  ├─ name_norm.str.split()            → name_tokens
    │  ├─ address_norm.str.split()         → address_tokens
    │  ├─ str.len() / digit count          → length/digit fields
    │  └─ raw columns preserved as-is
    ▼
PROCESSED DATAFRAME  (14 columns — see schema below)
    │
    │  src.cache.build_cache(...) → df.to_parquet(...)
    ▼
PARQUET CACHE  (cache/train_source1.parquet)
    │
    │  src.cache.load_cache(...)  → pd.read_parquet(...)
    ▼
DOWNSTREAM CONSUMER  (M4 blocking, M3 features, model)
```

**Important:** `preprocess_dataframe()` is called **once per source dataset**,
not once per candidate pair.  The cache prevents any re-normalisation.

---

## Cached Datasets

Six Parquet files are produced — one per (split, source) combination:

```
cache/
├── train_source1.parquet
├── train_source2.parquet
├── train_source3.parquet
├── test_source1.parquet
├── test_source2.parquet
└── test_source3.parquet
```

Ground truth (`train_ground_truth.tsv`) is **not** cached here; it contains
labels and is consumed directly by the feature-engineering / model notebooks.

---

## Cached Columns

Every Parquet file contains exactly **14 columns** in this fixed order:

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `entity_id` | str | Entity identifier (e.g. `S1-00001`) — raw, unchanged |
| 2 | `business_name` | str | Original business name — raw, unchanged |
| 3 | `business_address` | str | Original business address — raw, unchanged |
| 4 | `country` | str | Country string — raw, unchanged (free-form, no fixed list) |
| 5 | `name_norm` | str | `normalize_name(business_name)` — M2 validated normalization |
| 6 | `address_norm` | str | `normalize_address(business_address)` — M2 validated normalization |
| 7 | `name_tokens` | list[str] | Whitespace-split tokens of `name_norm` |
| 8 | `address_tokens` | list[str] | Whitespace-split tokens of `address_norm` |
| 9 | `name_token_count` | int | `len(name_tokens)` |
| 10 | `address_token_count` | int | `len(address_tokens)` |
| 11 | `name_length` | int | Character length of `name_norm` |
| 12 | `address_length` | int | Character length of `address_norm` |
| 13 | `name_digits` | int | Count of digit characters in `name_norm` |
| 14 | `address_digits` | int | Count of digit characters in `address_norm` |

**Notes:**
- `NaN` / `None` inputs produce `""` for normalized strings, `[]` for token
  lists, and `0` for all numeric derived fields.  No errors are raised.
- `name_tokens` and `address_tokens` are stored as pyarrow variable-length list
  arrays; they are read back as numpy arrays by default.  Call `.tolist()` if a
  Python list is needed.

---

## Building the Cache

```python
from src.cache import build_cache

# Build a single source (idempotent — skips if already built)
result = build_cache(
    split     = "train",       # "train" or "test"
    source    = "source1",     # "source1", "source2", or "source3"
    data_dir  = "/content",    # directory containing the TSV files
    cache_dir = "cache",       # output directory (created if absent)
    force     = False,         # True to force rebuild even if cached
)

print(result)
# {'split': 'train', 'source': 'source1',
#  'tsv_path': '/content/train_source1.tsv',
#  'cache_path': 'cache/train_source1.parquet',
#  'status': 'built',   # or 'skipped' or 'error'
#  'rows': 12345,
#  'error': None}

# Build all six sources in one pass
for split in ["train", "test"]:
    for source in ["source1", "source2", "source3"]:
        r = build_cache(split, source, data_dir="/content", cache_dir="cache")
        print(split, source, r["status"], r.get("rows"))
```

`build_cache()` returns a status dict rather than raising on missing files, so
a full-batch build loop can continue even if one source file is absent on the
current machine.

---

## Loading the Cache

```python
from src.cache import load_cache

df = load_cache(
    split     = "train",
    source    = "source1",
    cache_dir = "cache",
)

# df is a DataFrame with 14 columns, shape (N, 14)
print(df.shape)
print(df.columns.tolist())
print(df.head())
```

`load_cache()` raises `FileNotFoundError` if the cache has not been built.
It never rebuilds silently — the caller must explicitly call `build_cache()`
first.

---

## Cache Validation

### Existence check

```python
from src.cache import cache_exists

if cache_exists("train", "source1", "cache"):
    df = load_cache("train", "source1", "cache")
else:
    build_cache("train", "source1", data_dir="/content", cache_dir="cache")
    df = load_cache("train", "source1", "cache")
```

### Structural validation

```python
from src.cache import validate_cache

val = validate_cache("train", "source1", cache_dir="cache")

print(val["valid"])          # True / False
print(val["rows"])           # row count
print(val["missing_columns"])  # [] if all 14 cols present
for check, ok in val["checks"].items():
    print(check, "PASS" if ok else "FAIL")
```

The seven checks performed by `validate_cache()`:

| Check | Description |
|---|---|
| `file_exists` | Parquet file is on disk |
| `parquet_readable` | File can be read without errors |
| `all_14_columns` | All 14 required columns are present |
| `entity_id_present` | `entity_id` column exists |
| `raw_columns_present` | `business_name`, `business_address`, `country` present |
| `norm_columns_present` | `name_norm`, `address_norm` present |
| `non_empty` | At least one row |

---

## Downstream Usage

The preprocessing cache forms the boundary between the M2 preprocessing
deliverable and the downstream matching/modeling pipeline:

```
M2 (preprocessing):
    RAW TSV → preprocess_dataframe() → PARQUET CACHE

M4 (blocking / candidate generation):
    load_cache("train", "source1")  →  generate candidates
    load_cache("train", "source2")  ┘
    load_cache("train", "source3")  ┘

M3 (feature engineering + model):
    load_cache(...)  +  candidates  →  pair features  →  classifier  →  scores
```

**Contract for M4 and M3:**

- Always call `load_cache()` — do not re-read or re-normalise the raw TSV.
- Use `name_norm` / `address_norm` for all string comparisons.
- Use `name_tokens` / `address_tokens` for set-based similarity (Jaccard, overlap).
- Use `name_length` / `address_length` / `name_digits` / `address_digits` as
  numeric features without recomputing them.
- Do not modify or overwrite the Parquet files.

---

## Cache and Git

The `cache/` directory and all `*.parquet` files are listed in `.gitignore`
and **must not be committed to GitHub**.

**Reasons:**
- Source datasets are confidential competition data.
- Parquet cache files can be gigabytes in size.
- Each team member rebuilds the cache locally in their Colab environment.

```gitignore
cache/
*.parquet
```

To rebuild the cache in a fresh Colab session:

```python
from src.cache import build_cache

for split in ["train", "test"]:
    for source in ["source1", "source2", "source3"]:
        build_cache(split, source, data_dir="/content", cache_dir="cache")
```

---

## Normalization Reference

The normalization applied inside `preprocess_dataframe()` is the M2-validated
pipeline from `src/preprocess.py`.  Refer to `docs/dataset_dictionary.md` for
the full rule table.  Do not duplicate or override normalization logic in
downstream notebooks.

| Function | Applied to | Rule set |
|---|---|---|
| `normalize_text()` | base pipeline (both fields) | NFKC → lowercase → punctuation → whitespace |
| `normalize_name()` | `business_name` → `name_norm` | base + legal-suffix rules (pvt→private, ltd→limited, …) |
| `normalize_address()` | `business_address` → `address_norm` | base + street-type abbreviation rules (ave→avenue, rd→road, …) |
