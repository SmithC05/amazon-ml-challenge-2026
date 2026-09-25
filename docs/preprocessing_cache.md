# Preprocessing Cache — `src/cache.py`
## Amazon ML Challenge 2026

---

## Purpose

`src/cache.py` builds a **one-time Parquet cache** of M2-normalized source records.

Every team member's code (M3 feature extraction, M4 candidate generation, your advanced experiments) loads from this cache instead of re-running the normalization pipeline on every experiment iteration.

---

## Why this matters

| Without cache | With cache |
|---|---|
| Each experiment re-normalizes 300k+ records | Normalization runs once, result is saved |
| `normalize_name` + `normalize_address` called millions of times per run | Sub-second Parquet column read |
| Normalization bugs silently produce different results each run | Single canonical normalized form |

---

## Architecture

```
Raw TSVs  (train/test_source1/2/3.tsv)
    │
    ▼  src/cache.py  ←  src/preprocess.py  (Member 2, unchanged)
Parquet cache  (cache/*.parquet)
    │
    ├──▶  M4  candidate generation / blocking
    └──▶  M3  feature extraction / training
              (src/train.py, src/predict.py)
```

---

## Cache layout

```
cache/
  train_source1.parquet
  train_source2.parquet
  train_source3.parquet
  test_source1.parquet     (written only when test data is present)
  test_source2.parquet
  test_source3.parquet
```

> Cache files are excluded from git via `.gitignore`. Each team member builds their own local cache.

---

## Schema (per Parquet file)

All original TSV columns are preserved, plus:

| Column | Type | Description |
|---|---|---|
| `entity_id` | str | Original entity ID |
| `business_name` | str | Raw business name (unchanged) |
| `business_address` | str | Raw address (unchanged) |
| `country` | str | Raw country (unchanged) |
| `business_name_norm` | str | `normalize_name(business_name)` |
| `business_address_norm` | str | `normalize_address(business_address)` |
| `name_tokens` | int | Token count of normalized name |
| `address_tokens` | int | Token count of normalized address |
| `address_is_empty` | int8 | 1 if normalized address == `""` |

---

## Usage

### Build cache (once per machine)

```python
from pathlib import Path
from src.cache import build_cache

build_cache(
    data_dir=Path("dataset/train"),
    cache_dir=Path("cache"),
    split="train",
    force=False,          # True to rebuild
)
```

Or via CLI:

```bash
python src/cache.py \
    --data-dir  dataset/train \
    --cache-dir cache \
    --split     train \
    --validate
```

### Load in M3 training

```python
from src.cache import load_all_cache
s1, s2, s3 = load_all_cache("cache", split="train")
```

Or pass `--cache-dir cache` to `src/train.py` — it will auto-detect and use the cache:

```bash
python src/train.py \
    --data-dir  dataset/train \
    --candidates output/candidate_pairs.tsv \
    --cache-dir cache
```

### Load in M4 blocking

```python
from src.cache import load_cache
s1 = load_cache("cache", "train", "source1")
s2 = load_cache("cache", "train", "source2")
s3 = load_cache("cache", "train", "source3")
```

### Validate cache integrity

```python
from src.cache import validate_cache
validate_cache("dataset/train", "cache", split="train", n_check=500)
```

Samples 500 rows, re-normalizes fresh, and asserts results match the cache.

---

## Timing benchmark

*(To be filled after first Colab run)*

| Operation | Time |
|---|---|
| Raw normalization (S1+S2+S3, ~330k rows) | TBD |
| Cache build (normalize + write Parquet) | TBD |
| Cache load (all three sources) | TBD |
| Speedup factor | TBD |

---

## Design constraints

- Normalization is **never reimplemented** in `cache.py`. Only `preprocess.py` functions are called.
- Raw columns (`business_name`, `business_address`) are preserved unchanged alongside normalized versions.
- Cache files are `.gitignore`d — never committed to the repo.
- `pyarrow` + snappy compression are used for fast columnar reads.
