# Dataset Dictionary — Amazon ML Challenge 2026

This document describes the schema and format of every training file used in the Amazon ML Challenge 2026 entity-matching task.

---

## Source 1

Source 1 is the primary ("left-hand side") entity dataset.  
Every record in the ground-truth file is anchored to a Source 1 entity.

| Column | Description | Type |
|---|---|---|
| `entity_id` | Unique identifier for the entity in Source 1 (e.g. `S1-00001`) | String |
| `business_name` | Original, unmodified business name as stored in the source | String |
| `business_address` | Original, unmodified business address as stored in the source | String |
| `country` | Country associated with the entity — treated as a free-form string; do **not** assume a fixed country list | String |

---

## Source 2

Source 2 is one of the secondary ("right-hand side") entity datasets to be matched against Source 1.

| Column | Description | Type |
|---|---|---|
| `entity_id` | Unique identifier for the entity in Source 2 (e.g. `S2-00001`) | String |
| `business_name` | Original, unmodified business name as stored in Source 2 | String |
| `business_address` | Original, unmodified business address as stored in Source 2 | String |
| `country` | Country associated with the entity — treated as a free-form string | String |

---

## Source 3

Source 3 is the second secondary entity dataset to be matched against Source 1.

| Column | Description | Type |
|---|---|---|
| `entity_id` | Unique identifier for the entity in Source 3 (e.g. `S3-00001`) | String |
| `business_name` | Original, unmodified business name as stored in Source 3 | String |
| `business_address` | Original, unmodified business address as stored in Source 3 | String |
| `country` | Country associated with the entity — treated as a free-form string | String |

---

## Ground Truth

The ground-truth file maps each Source 1 entity to zero, one, or multiple matching entities from Source 2 and/or Source 3.

| Column | Description | Type |
|---|---|---|
| `source1_entity_id` | The Source 1 entity identifier being resolved | String |
| `matched_entity_ids` | Comma-separated list of matching entity IDs from Source 2/Source 3; may be empty (zero matches), a single ID (one match), or multiple IDs (multiple matches) | String |

**Match cardinality:**

| Case | `matched_entity_ids` value |
|---|---|
| Zero matches | Empty string or NaN |
| One match | Single ID, e.g. `S2-00042` |
| Multiple matches | Comma-separated IDs, e.g. `S2-00042, S3-00017` |

---

## File Format

All four training files are **tab-separated values (TSV)** files with a header row.  
They should be loaded with:

```python
import pandas as pd
df = pd.read_csv("train_source1.tsv", sep="\t")
```

---

## Normalization Fields

The EDA notebook (`notebooks/01_eda.ipynb`) and preprocessing module (`src/preprocess.py`) add two derived columns per source dataframe.  
**Original raw fields are always preserved and are never overwritten.**

| Derived Column | Source Column | Description |
|---|---|---|
| `business_name_norm` | `business_name` | Normalized version of the business name |
| `business_address_norm` | `business_address` | Normalized version of the business address |

The derived columns are stored **alongside** the originals.  
All downstream matching stages must use the `_norm` columns; raw columns are kept for auditing, debugging, and human review.

---

## Normalization Strategy

Normalization is implemented in `src/preprocess.py` via three functions:

| Function | Purpose |
|---|---|
| `normalize_text(value)` | Core normalization pipeline shared by names and addresses |
| `normalize_name(value)` | Entry point for business-name normalization; delegates to `normalize_text` |
| `normalize_address(value)` | Entry point for address normalization; delegates to `normalize_text` |

**Pipeline steps (applied in order):**

1. **Null handling** — return `""` for `NaN` / `None` values.
2. **Unicode NFKC normalization** — converts compatibility characters (e.g. fullwidth Latin letters, ligatures) to their ASCII/canonical equivalents; composes combining characters.
3. **Lowercase conversion** — case-insensitive comparison.
4. **Punctuation replacement** — non-word, non-space characters are replaced with a single space.  
   This ensures `"St."` == `"St"`, but **address numbers and digits are preserved** (they are `\w` characters).
5. **Whitespace normalization** — multiple consecutive spaces are collapsed to one; leading and trailing whitespace is stripped.

**Design principle:** normalization aims to *reduce surface noise* without *discarding meaningful information*.  
Aggressive stemming, stop-word removal, or abbreviation expansion are **not** applied at this stage; those are left to the matching/feature-engineering stage.

---

## Country Handling

`country` is stored as a **free-form string** in all three source files.

**The pipeline does not assume a fixed country list.**  
Any hardcoded enumeration of allowed countries (e.g. `["US", "IN"]`) is incorrect and must not be introduced at any stage.  
The matching model must treat country as an open-set categorical feature or use it only for blocking/filtering in a country-agnostic way.
