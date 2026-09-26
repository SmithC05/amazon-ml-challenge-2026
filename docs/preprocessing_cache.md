# Preprocessing Cache — `src/cache.py`

**Amazon ML Challenge 2026 — Preprocessing & Caching Reference**

This document describes the preprocessing-cache architecture, the cached schema,
and the API used by downstream team members to load preprocessed source records.

---

## M2 Final Handoff Contract

> **Status: FROZEN** — M2 preprocessing and caching layer is complete.  
> All downstream components (M3, M4) must consume normalized data via `load_cache()`.  
> No component other than M2 may implement or modify normalization logic.

### Ownership

| Layer | Owns | Must NOT |
|---|---|---|
| **M2** | `src/preprocess.py`, `src/cache.py`, `cache/*.parquet` | — |
| **M3** | Feature engineering, model training | Re-implement normalization |
| **M4** | Candidate generation, blocking | Re-implement normalization |

### What M2 Provides

`build_cache()` preprocesses each source TSV once and saves a **14-column Parquet file**.  
All downstream components call `load_cache()` — they never read raw TSVs or re-normalize.

### Fields M4 Must Consume

```python
df = load_cache("train", "source1", cache_dir="cache")

# Required for blocking and candidate generation:
entity_id       # entity identifier
name_norm       # canonical normalized business name
address_norm    # canonical normalized business address
country         # free-form country string (raw, preserved)
name_tokens     # list[str] — whitespace-split tokens of name_norm
address_tokens  # list[str] — whitespace-split tokens of address_norm
```

### Fields M3 Must Consume

```python
df = load_cache("train", "source1", cache_dir="cache")

# Required for feature engineering and model training:
entity_id       # entity identifier
name_norm       # canonical normalized business name
address_norm    # canonical normalized business address
country         # free-form country string
# Optional numeric features (pre-computed):
name_token_count, address_token_count
name_length,     address_length
name_digits,     address_digits
```

### Normalization Contract

- **M2 owns normalization.**  `normalize_name()` and `normalize_address()` in
  `src/preprocess.py` are the **sole canonical implementations**.
- **M3 must not** call `normalize_name()`, `normalize_address()`, or any equivalent.
- **M4 must not** call `normalize_name()`, `normalize_address()`, or any equivalent.
- Both M3 and M4 must read `name_norm` / `address_norm` from the Parquet cache.
- Raw fields (`business_name`, `business_address`) are preserved in the cache for
  reference but must not be re-normalized downstream.

### Frozen API Surface

| Function | Signature | Status |
|---|---|---|
| `build_cache()` | `(split, source, data_dir, cache_dir, force, chunksize)` | FROZEN |
| `load_cache()` | `(split, source, cache_dir, columns)` | FROZEN |
| `load_all_cache()` | `(cache_dir, split)` | FROZEN |
| `cache_exists()` | `(split, source, cache_dir)` | FROZEN |
| `validate_cache()` | `(split, source, cache_dir)` | FROZEN |

### Frozen Schema (14 columns, exact order)

```
entity_id, business_name, business_address, country,
name_norm, address_norm, name_tokens, address_tokens,
name_token_count, address_token_count,
name_length, address_length, name_digits, address_digits
```

---

## Why Caching Is Needed

| Without cache | With cache |
|---|---|
| Each experiment re-normalizes all source records | Normalization runs once, result is saved |
| `normalize_name` + `normalize_address` called repeatedly per run | Sub-second Parquet column read |
| Subtle normalization inconsistencies possible across runs | Single canonical normalized form |

Without caching, the normalization pipeline runs on every notebook restart and
every experiment iteration, repeating identical work.

**With caching:**
```
Once: reads TSV → normalises S1 → writes Parquet cache
Experiment A: load_cache("train", "source1") → instant, no normalisation
Experiment B: load_cache("train", "source1") → instant, no normalisation
```

---

## Architecture

```
Raw TSVs  (train_source1/2/3.tsv, test_source1/2/3.tsv)
    │
    ▼  src/cache.py  ←  src/preprocess.py  (M2 normalization, unchanged)
Parquet cache  (cache/*.parquet)
    │
    ├──▶  M4  (blocking / candidate generation)  — load_cache() only
    └──▶  M3  (feature extraction + model)        — load_cache() only
```

**Important:** `preprocess_dataframe()` is called **once per source dataset**,
not once per candidate pair.  The cache prevents any re-normalisation.

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
    │  ├─ normalize_name(business_name)       → name_norm
    │  ├─ normalize_address(business_address) → address_norm
    │  ├─ name_norm.str.split()               → name_tokens
    │  ├─ address_norm.str.split()            → address_tokens
    │  ├─ str.len() / digit count             → length/digit fields
    │  └─ raw columns preserved as-is
    ▼
PROCESSED DATAFRAME  (14 columns — see schema below)
    │
    │  src.cache.build_cache(...) → df.to_parquet(...)
    ▼
PARQUET CACHE  (cache/train_source1.parquet, ...)
    │
    │  src.cache.load_cache(...)  → pd.read_parquet(...)
    ▼
DOWNSTREAM CONSUMER  (M4 blocking, M3 features/model)
```

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

> Cache files are excluded from git via `.gitignore`. Each team member builds
> their own local cache.

Ground truth (`train_ground_truth.tsv`) is **not** cached here.

---

## Cached Columns (Schema — 14 columns per file)

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

### Python API

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

# Build all six sources in one loop
for split in ["train", "test"]:
    for source in ["source1", "source2", "source3"]:
        r = build_cache(split, source, data_dir="/content", cache_dir="cache")
        print(split, source, r["status"], r.get("rows"))
```

### CLI

```bash
# Build all three train sources
python src/cache.py --data-dir /content --cache-dir cache --split train --source all

# Force rebuild of one source
python src/cache.py --data-dir /content --cache-dir cache --split train --source source1 --force
```

---

## Loading the Cache

### Load a single source

```python
from src.cache import load_cache

df = load_cache("train", "source1", cache_dir="cache")
# df is a DataFrame with 14 columns, shape (N, 14)
```

### Load all three sources at once

```python
from src.cache import load_all_cache

s1, s2, s3 = load_all_cache("cache", split="train")
```

### Load only selected columns (faster)

```python
df = load_cache("train", "source1", cache_dir="cache",
                columns=["entity_id", "name_norm", "address_norm"])
```

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

### Structural validation (7 checks)

```python
from src.cache import validate_cache

val = validate_cache("train", "source1", cache_dir="cache")
print(val["valid"])            # True / False
print(val["rows"])             # row count
print(val["missing_columns"])  # [] if all 14 cols present
for check, ok in val["checks"].items():
    print(check, "PASS" if ok else "FAIL")
```

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

```
M2 (preprocessing):
    RAW TSV → preprocess_dataframe() → PARQUET CACHE

M4 (blocking / candidate generation):
    load_cache("train", "source1")  → generate candidates
    load_cache("train", "source2")  ┘
    load_cache("train", "source3")  ┘

M3 (feature engineering + model):
    load_all_cache("cache", "train")  →  pair features  →  classifier  →  scores
```

**Contract for M4 and M3:**

- Always call `load_cache()` or `load_all_cache()` — do not re-read or re-normalise the raw TSV.
- Use `name_norm` / `address_norm` for all string comparisons.
- Use `name_tokens` / `address_tokens` for set-based similarity (Jaccard, overlap).
- Use `name_token_count`, `name_length`, `name_digits`, etc. as numeric features directly.
- Do not modify or overwrite the Parquet files.

---

## Phase 7 — Real Dataset Validation and Performance

> **Measured on 2026-09-26** against the actual competition dataset.
> Row counts for all 6 sources measured by exact line count.
> Build timing measured for `train_source1` (non-chunked); other
> sources projected from the measured rate of **4,315 rows/s**.

### Real Dataset Row Counts

| Split | Source | TSV Rows | TSV Size |
|---|---|---|---|
| train | source1 | 2,206,821 | 200.3 MB |
| train | source2 | 5,034,616 | 466.6 MB |
| train | source3 | 5,285,603 | 480.4 MB |
| test  | source1 | 1,732,544 | 166.9 MB |
| test  | source2 | 4,887,273 | 485.9 MB |
| test  | source3 | 5,082,316 | 482.6 MB |
| **Total** | | **24,229,173** | **2,363 MB** |

### Build Performance (non-chunked, actual machine)

| Split | Source | Rows | Build Time | Rows/s | Parquet Size | Peak RAM |
|---|---|---|---|---|---|---|
| train | source1 | 2,206,821 | **511s** | 4,315 | 343.9 MB | 2,298 MB |
| train | source2 | 5,034,616 | ~1,167s † | ~4,315 | ~785 MB | ~5,251 MB |
| train | source3 | 5,285,603 | ~1,225s † | ~4,315 | ~823 MB | ~5,513 MB |
| test  | source1 | 1,732,544 | ~402s †   | ~4,315 | ~270 MB | ~1,805 MB |
| test  | source2 | 4,887,273 | ~1,133s † | ~4,315 | ~762 MB | ~5,097 MB |
| test  | source3 | 5,082,316 | ~1,178s † | ~4,315 | ~792 MB | ~5,300 MB |

† Projected from measured 4,315 rows/s. Build time is CPU-bound (regex + token ops).

**Total estimated sequential build time: ~93 minutes.**

### Cache Validation (train_source1, fully validated)

All 7 structural checks passed:

```
file_exists:       PASS
parquet_readable:  PASS
all_14_columns:    PASS
entity_id_present: PASS
raw_cols_present:  PASS
norm_cols_present: PASS
non_empty:         PASS

rows:    2,206,821  (matches TSV exactly)
Parquet: 343.9 MB
Load:    5.566 s
```

Normalized content spot-check (5 real rows — all correct):
```
S1-925783039  name_norm=OK  addr_norm=OK
S1-773889195  name_norm=OK  addr_norm=OK
S1-377745466  name_norm=OK  addr_norm=OK
S1-133037285  name_norm=OK  addr_norm=OK
S1-755362802  name_norm=OK  addr_norm=OK
```

### Memory Observations

`preprocess_dataframe()` amplifies RAM approximately **11× the raw TSV file size**
because it materialises 10 derived columns simultaneously (token lists, norm strings,
counts, lengths, digit strings):

| Source | TSV | Peak RAM (non-chunked) | Approach |
|---|---|---|---|
| train_source1 | 200 MB | **2,298 MB** (measured) | OK on ≥ 4 GB |
| train/test source2–3 | 480 MB | **~5,300 MB** (est.) | Needs ≥ 8 GB |

> **Recommendation:** Use `chunksize=500_000` for source2 and source3 on machines
> with less than 8 GB RAM. Peak RAM per chunk is bounded to ~400–500 MB.

### Chunked vs Non-Chunked

| | Non-chunked | Chunked (500k) |
|---|---|---|
| Wall-clock time | 511s (source1) | ~511s (CPU-bound; no speedup) |
| Peak RAM | 2,298 MB | ~400–500 MB per chunk |
| Content identical | — | YES (Phase 6: 67/67 tests PASS) |
| Use when | RAM ≥ 8 GB | RAM < 8 GB, or source2/source3 |

### Rebuilding Remaining Caches

Use the CLI with `--chunksize` for memory-safe builds:

```bash
# Build all train caches (chunked — recommended for source2/source3)
python src/cache.py \
  --data-dir /path/to/dataset/train \
  --cache-dir cache \
  --split train \
  --source all \
  --chunksize 500000

# Build all test caches
python src/cache.py \
  --data-dir /path/to/dataset/test \
  --cache-dir cache \
  --split test \
  --source all \
  --chunksize 500000
```

Or in Python:

```python
from src.cache import build_cache

DATA = {"train": "/path/to/train", "test": "/path/to/test"}
for split, data_dir in DATA.items():
    for source in ["source1", "source2", "source3"]:
        r = build_cache(split, source,
                        data_dir=data_dir,
                        cache_dir="cache",
                        chunksize=500_000)   # memory-safe
        print(split, source, r["status"], r.get("rows"))
```

---

## Timing Benchmark

Reference timings measured locally on synthetic data (S1=15k, S2=20k, S3=18k rows):

| Operation | Time |
|---|---|
| Train cache build (all 3 sources) | ~2.13 s |
| Test cache build (all 3 sources) | ~2.12 s |
| Cache load (avg per source) | ~0.059 s |
| Speedup vs. TSV read + preprocess | **12.19×** |

Real production timings (train_source1, 2.2M rows): build=511s, load=5.6s.

---

## Cache and Git

The `cache/` directory and all `*.parquet` files are listed in `.gitignore`
and **must not be committed to GitHub**.

```gitignore
cache/
*.parquet
```

---

## Normalization Reference

Normalization applied inside `preprocess_dataframe()` is the M2-validated
pipeline from `src/preprocess.py`. Refer to `docs/dataset_dictionary.md` for
the full rule table.

| Function | Applied to | Rules |
|---|---|---|
| `normalize_text()` | base pipeline (both fields) | NFKC → lowercase → punctuation → whitespace |
| `normalize_name()` | `business_name` → `name_norm` | base + legal-suffix rules |
| `normalize_address()` | `business_address` → `address_norm` | base + street-type abbreviation rules |

---

## Design Constraints

- Normalization is **never reimplemented** in `cache.py`. Only `preprocess.py` functions are called.
- Raw columns (`business_name`, `business_address`) are preserved unchanged alongside normalized versions.
- Cache files are `.gitignore`d — never committed to the repo.
- `pyarrow` + snappy compression are used for fast columnar reads.
- Country is treated as a free-form string; no fixed country list is assumed.
