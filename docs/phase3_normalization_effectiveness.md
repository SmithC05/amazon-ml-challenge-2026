# Phase 3 — Normalization Effectiveness Report

**Task:** M2 Preprocessing Finalization & Data-Quality Audit  
**Phase:** 3 — Full Training Normalization Effectiveness  
**Date:** 2026-09-26  
**Data:** Full training split (train_source1/2/3.tsv + train_ground_truth.tsv)

---

## 1. Methodology

### Data Sources
| File | Rows |
|------|------|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `train_ground_truth.tsv` | 2,206,821 |

### Ground-Truth Parsing

The ground-truth file was parsed with the following rules:

- **Denominator:** Only true labeled pairs are counted. S1 entities with missing/empty `matched_entity_ids` are excluded from all percentages.
- **Pair expansion:** Each comma-separated `matched_entity_id` in the ground truth creates one labeled pair `(s1_entity_id, matched_entity_id)`.
- **Source label:** determined by entity ID prefix (`S2-...` = Source 2, `S3-...` = Source 3).

### Normalization

The **canonical M2 normalization functions** from [`src/preprocess.py`](../src/preprocess.py) are used:

- `normalize_name(value)` — applied to `business_name` fields
- `normalize_address(value)` — applied to `business_address` fields

**No new rules were added.** The Phase 2 dead-key fix (`"pvt. ltd."` removed) is the only change applied before this analysis.

### Comparison Logic

- **Raw exact match:** both fields are non-empty AND `s1_value == target_value` (case-sensitive, punctuation included)
- **Normalized exact match:** `normalize_name/address(s1_value) == normalize_name/address(target_value)`
- **Address null rule:** two empty/null address values are **not** counted as an exact match
- **Denominator for all percentages:** total true labeled pairs resolved (7,638,365)

---

## 2. Ground-Truth Statistics

| Metric | Count |
|--------|-------|
| Total ground-truth rows | 2,206,821 |
| S1 rows WITH known matches | 2,083,574 |
| S1 rows with missing/empty GT | 123,247 |
| **Total true labeled pairs** | **7,638,365** |
| S1→S2 pairs | 3,693,619 |
| S1→S3 pairs | 3,944,746 |

> **Note:** 123,247 S1 entities (5.59%) have no ground-truth match. These are correctly excluded from all effectiveness percentages.

---

## 3. Name Normalization Effectiveness

| Metric | Count | Percentage |
|--------|------:|----------:|
| Raw exact name matches | 354,118 | 4.64% |
| Normalized exact name matches | 1,872,588 | 24.52% |
| **Name newly exact** | **1,518,470** | **19.88%** |

**Improvement: +19.88 percentage points**

### What the improvement captures

The 5.3× increase (from 4.64% to 24.52%) reflects normalization successfully reconciling:

1. **Case differences** — e.g., `"CALDEON NOVA"` vs `"Caldeon Nova"` → both become `"caldeon nova"`
2. **Punctuation differences** — e.g., `"Obsidian, LLC"` vs `"Obsidian,-LLC"` → both become `"obsidian llc"`
3. **Bracket/symbol noise** — e.g., `"Obsidian, [[LLC]]"` → `"obsidian llc"`
4. **Whitespace differences** — e.g., `"Caldeon  Nova"` (double space) → `"caldeon nova"`
5. **Legal suffix abbreviations** — e.g., `"Obsidian, LLC"` vs `"obsidian, llc"` → `"obsidian llc"` (both; punctuation removal handles LLC here, suffix map handles pvt/ltd/inc/corp/llp patterns)

### Name Newly-Exact Examples

```
S1-274126313  →  S2-422093843 [S2]
  raw S1:   'Obsidian, LLC'
  raw tgt:  'Obsidian,-LLC'
  norm:     'obsidian llc'  ==  'obsidian llc'  ✓

S1-274126313  →  S2-680265918 [S2]
  raw S1:   'Obsidian, LLC'
  raw tgt:  'obsidian, llc'
  norm:     'obsidian llc'  ==  'obsidian llc'  ✓

S1-274126313  →  S3-925631694 [S3]
  raw S1:   'Obsidian, LLC'
  raw tgt:  'Obsidian, Llc'
  norm:     'obsidian llc'  ==  'obsidian llc'  ✓

S1-274126313  →  S3-461175723 [S3]
  raw S1:   'Obsidian, LLC'
  raw tgt:  'Obsidian, [[LLC]]'
  norm:     'obsidian llc'  ==  'obsidian llc'  ✓

S1-65263544  →  S2-881122703 [S2]
  raw S1:   'Caldeon Nova'
  raw tgt:  'Caldeon  Nova'   (double space)
  norm:     'caldeon nova'  ==  'caldeon nova'  ✓

S1-65263544  →  S2-180463458 [S2]
  raw S1:   'Caldeon Nova'
  raw tgt:  'CALDEON NOVA'
  norm:     'caldeon nova'  ==  'caldeon nova'  ✓

S1-65263544  →  S3-102573485 [S3]
  raw S1:   'Caldeon Nova'
  raw tgt:  '>> caldeon nova'
  norm:     'caldeon nova'  ==  'caldeon nova'  ✓
```

---

## 4. Address Normalization Effectiveness

| Metric | Count | Percentage |
|--------|------:|----------:|
| Raw exact address matches | 170,074 | 2.23% |
| Normalized exact address matches | 835,576 | 10.94% |
| **Address newly exact** | **665,502** | **8.71%** |

**Improvement: +8.71 percentage points**

> **Note:** The address overall match rate (10.94%) is lower than name match rate (24.52%) because addresses are longer, more structured, and more prone to partial representation (missing suites, floor numbers, etc.) — normalization cannot bridge semantic differences like `"Floor 3"` vs `"3rd Floor"`.

### What the improvement captures

The 4.9× increase (from 2.23% to 10.94%) reflects:

1. **Case normalization** — e.g., `"6(29), C.I.T. Colony"` vs `"6(29), C.I.T. COLONY"` → both lowercase
2. **Punctuation removal** — e.g., commas, dots, parentheses removed → `"6 29 c i t colony ..."`
3. **Street abbreviation expansion** — e.g., `"8706 KENTUCKY DERBY DR"` → `"8706 kentucky derby drive"` matching `"8706 Kentucky Derby Drive"`
4. **Mixed-case normalization** — e.g., `"20085 Us 23"` vs `"20085 US 23"` → `"20085 us 23"`

### Address Newly-Exact Examples

```
S1-55344266  →  S2-249013014 [S2]
  raw S1:   '6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu'
  raw tgt:  '6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu'
  norm:     '6 29 c i t colony 2nd main road mylapore chennai tamil nadu'  ✓

S1-546142636  →  S2-487600131 [S2]
  raw S1:   '8706 Kentucky Derby Drive, Waxhaw, NC'
  raw tgt:  '8706 KENTUCKY DERBY DR, WAXHAW, NC'
  norm:     '8706 kentucky derby drive waxhaw nc'  ✓   (DR → drive)

S1-546142636  →  S2-392804085 [S2]
  raw S1:   '8706 Kentucky Derby Drive, Waxhaw, NC'
  raw tgt:  '8706 KENTUCKY DERBY DRIVE, WAXHAW, NC'
  norm:     '8706 kentucky derby drive waxhaw nc'  ✓

S1-503957000  →  S2-994658326 [S2]
  raw S1:   '20085 Us 23, Circleville, OH'
  raw tgt:  '20085 US 23, CIRCLEVILLE, OH'
  norm:     '20085 us 23 circleville oh'  ✓
```

---

## 5. Complete Results Table

| Metric | Count | Percentage |
|--------|------:|----------:|
| Raw exact name | 354,118 | 4.64% |
| Normalized exact name | 1,872,588 | 24.52% |
| Name newly exact | 1,518,470 | 19.88% |
| Raw exact address | 170,074 | 2.23% |
| Normalized exact address | 835,576 | 10.94% |
| Address newly exact | 665,502 | 8.71% |

**Denominator for all percentages: 7,638,365 true labeled pairs**

---

## 6. Key Findings

1. **Normalization is highly effective for name matching**: it increases exact match rate by 5.3× (4.64% → 24.52%), creating 1,518,470 newly-exact name pairs.

2. **Normalization is effective for address matching**: it increases exact match rate by 4.9× (2.23% → 10.94%), creating 665,502 newly-exact address pairs.

3. **The dominant gains are from lowercasing + punctuation removal** — the core `normalize_text()` pipeline. The legal-suffix and address-abbreviation token maps contribute additional improvements on top.

4. **Address matching remains harder** — even after normalization, only 10.94% of true pairs have exactly matching addresses, confirming that address representations vary significantly beyond what normalization can resolve. Fuzzy/token matching at the feature-engineering stage is needed.

5. **No new normalization rules are warranted** from this analysis — the current M2 rules are already validated. The remaining non-exact pairs require downstream fuzzy/ML matching.

---

## 7. Scope Confirmation

| Check | Status |
|-------|--------|
| Normalization rules changed | NO |
| New normalization rules added | NO |
| Blocking performed | NO |
| Candidate generation performed | NO |
| Fuzzy matching performed | NO |
| ML model trained | NO |
| cache.py modified | NO |
| Feature engineering performed | NO |
| Denominator: true labeled pairs only | YES |
| Missing GT excluded from denominator | YES |
| Two null addresses counted as match | NO |

---

*Script: [`tools/phase3_effectiveness.py`](../tools/phase3_effectiveness.py)*  
*Normalization: [`src/preprocess.py`](../src/preprocess.py) — canonical M2 functions, unchanged*
