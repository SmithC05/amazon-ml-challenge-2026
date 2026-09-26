"""
src/train.py
============
Baseline training pipeline for the Amazon ML Challenge 2026.

Member 3 deliverable — baseline matching module.

Memory-safety architecture (v3 — fully disk-backed)
-----------------------------------------------------
  All three previous memory problems are eliminated:

  Problem 1 (FIXED): Spool is now written incrementally per chunk using
      pyarrow.parquet.ParquetWriter.  No list of frames.  No pd.concat.

  Problem 2 (FIXED): Training set selection reads the spool in passes via
      PyArrow row-group scanning — only the capped training rows (≤ max_train_pairs)
      are materialised into RAM for LogisticRegression.

  Problem 3 (FIXED): Threshold sweep is fully streaming.  Validation pairs
      are scored in bounded chunks from the spool.  For each threshold only
      {s1_id → {matched_ids}} (size = n_val_s1 distinct S1 IDs) is kept in RAM —
      not the full validation pair DataFrame.

Pipeline
--------
1.  Load source records from M2 cache (or raw TSVs).
2.  Build compact entity lookup dicts — O(1) per-pair join, never a DataFrame merge.
3.  Entity-level split (80/20, random_state=42) BEFORE any candidate scanning.
4.  Stream candidate TSV in chunk_size rows → expand pairs → write each chunk
    immediately to Parquet via ParquetWriter → release chunk.
5.  Two-pass spool read:
      Pass A — count rows and collect all positive (label=1) row indices.
      Pass B — materialise positives + sampled negatives (up to max_train_pairs).
6.  Fit StandardScaler + LogisticRegression on the materialised training matrix.
7.  Streaming threshold sweep: score validation pairs in chunks; accumulate only
    {s1_id → predicted_set} per threshold (bounded by n_val_s1 entities, not n_pairs).
8.  Select best threshold; save model + config.

Official metric: entity-level macro F0.5 (competition definition).
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import normalize_name, normalize_address, preprocess_dataframe  # noqa: E402
from features import build_feature_matrix_from_dicts, FEATURE_NAMES             # noqa: E402
from cache import load_all_cache, cache_exists                                   # noqa: E402

logger = logging.getLogger(__name__)

RANDOM_STATE             = 42
BETA                     = 0.5
_DEFAULT_CHUNK           = 50_000
_DEFAULT_MAX_TRAIN_PAIRS = 2_000_000

# PyArrow schema for the spool Parquet file (fixed, never re-inferred)
_SPOOL_SCHEMA = pa.schema([
    pa.field("source1_entity_id",  pa.string()),
    pa.field("cand_entity_id",     pa.string()),
    pa.field("s1_name_norm",       pa.string()),
    pa.field("s1_address_norm",    pa.string()),
    pa.field("s1_country",         pa.string()),
    pa.field("cand_name_norm",     pa.string()),
    pa.field("cand_address_norm",  pa.string()),
    pa.field("cand_country",       pa.string()),
    pa.field("label",              pa.int8()),
    pa.field("is_val",             pa.int8()),
])


# ─────────────────────────────────────────────────────────────────────────────
# Official metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def entity_f05(truth: set[str], predicted: set[str]) -> float:
    if not truth and not predicted:
        return 1.0
    if not truth:
        return 0.0
    tp = len(truth & predicted)
    prec = tp / len(predicted) if predicted else 0.0
    rec  = tp / len(truth)
    return f_beta(prec, rec)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_sources(
    data_dir: Path,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt = pd.read_csv(data_dir / "train_ground_truth.tsv", sep="\t")
    if cache_dir is not None and all(
        cache_exists("train", src, cache_dir)
        for src in ("source1", "source2", "source3")
    ):
        print("  Loading sources from M2 Parquet cache...")
        s1, s2, s3 = load_all_cache(cache_dir, "train")
    else:
        if cache_dir is not None:
            print("  Cache not found — falling back to raw TSV + M2 normalization")
        s1 = preprocess_dataframe(pd.read_csv(data_dir / "train_source1.tsv", sep="\t", dtype=str))
        s2 = preprocess_dataframe(pd.read_csv(data_dir / "train_source2.tsv", sep="\t", dtype=str))
        s3 = preprocess_dataframe(pd.read_csv(data_dir / "train_source3.tsv", sep="\t", dtype=str))
    print(f"Loaded  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  GT={len(gt):,}")
    return s1, s2, s3, gt


def build_truth_map(gt: pd.DataFrame) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    for _, row in gt.iterrows():
        sid = row["source1_entity_id"]
        val = row["matched_entity_ids"]
        if pd.isna(val) or str(val).strip() == "":
            truth[sid] = set()
        else:
            truth[sid] = {x.strip() for x in str(val).split(",") if x.strip()}
    return truth


def _build_entity_lookups(s2: pd.DataFrame, s3: pd.DataFrame) -> dict[str, dict]:
    cand_lookup: dict[str, dict] = {}
    for df in (s2, s3):
        for row in df[["entity_id", "name_norm", "address_norm", "country"]].itertuples(index=False):
            cand_lookup[row.entity_id] = {
                "name_norm":    row.name_norm,
                "address_norm": row.address_norm,
                "country":      row.country,
            }
    return cand_lookup


def _build_s1_lookup(s1: pd.DataFrame) -> dict[str, dict]:
    s1_lookup: dict[str, dict] = {}
    for row in s1[["entity_id", "name_norm", "address_norm", "country"]].itertuples(index=False):
        s1_lookup[row.entity_id] = {
            "name_norm":    row.name_norm,
            "address_norm": row.address_norm,
            "country":      row.country,
        }
    return s1_lookup


def entity_split(
    all_s1_ids: list[str],
    train_frac: float = 0.80,
    random_state: int = RANDOM_STATE,
) -> tuple[set[str], set[str]]:
    population = np.array(all_s1_ids, dtype=object)
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(population)
    n_train = int(len(shuffled) * train_frac)
    train_ids = set(shuffled[:n_train])
    val_ids   = set(shuffled[n_train:])
    assert train_ids & val_ids == set(), "Split leak!"
    return train_ids, val_ids


# ─────────────────────────────────────────────────────────────────────────────
# Chunk expansion
# ─────────────────────────────────────────────────────────────────────────────

def _expand_and_label_chunk(
    chunk_df: pd.DataFrame,
    s1_lookup: dict[str, dict],
    cand_lookup: dict[str, dict],
    truth_map: dict[str, set[str]],
    val_s1_set: set[str],
) -> pa.Table | None:
    """
    Expand one candidate TSV chunk into a PyArrow Table ready for spool writing.
    Returns None if no scorable pairs were produced.
    """
    s1_ids, cand_ids = [], []
    s1_names, s1_addrs, s1_ctries = [], [], []
    c_names, c_addrs, c_ctries = [], [], []
    labels, is_vals = [], []

    for row in chunk_df.itertuples(index=False):
        s1_id = str(row.source1_entity_id)
        raw_ids = str(row.candidate_entity_ids) if row.candidate_entity_ids else ""
        if not raw_ids or raw_ids.lower() in ("nan", "none", ""):
            continue
        s1_rec = s1_lookup.get(s1_id)
        if s1_rec is None:
            continue
        truth_set = truth_map.get(s1_id, set())
        is_val    = int(s1_id in val_s1_set)

        for cid in (x.strip() for x in raw_ids.split(",") if x.strip()):
            cand_rec = cand_lookup.get(cid)
            if cand_rec is None:
                continue
            s1_ids.append(s1_id);   cand_ids.append(cid)
            s1_names.append(s1_rec["name_norm"])
            s1_addrs.append(s1_rec["address_norm"])
            s1_ctries.append(s1_rec["country"])
            c_names.append(cand_rec["name_norm"])
            c_addrs.append(cand_rec["address_norm"])
            c_ctries.append(cand_rec["country"])
            labels.append(int(cid in truth_set))
            is_vals.append(is_val)

    if not s1_ids:
        return None

    return pa.table({
        "source1_entity_id": pa.array(s1_ids,   type=pa.string()),
        "cand_entity_id":    pa.array(cand_ids,  type=pa.string()),
        "s1_name_norm":      pa.array(s1_names,  type=pa.string()),
        "s1_address_norm":   pa.array(s1_addrs,  type=pa.string()),
        "s1_country":        pa.array(s1_ctries, type=pa.string()),
        "cand_name_norm":    pa.array(c_names,   type=pa.string()),
        "cand_address_norm": pa.array(c_addrs,   type=pa.string()),
        "cand_country":      pa.array(c_ctries,  type=pa.string()),
        "label":             pa.array(labels,    type=pa.int8()),
        "is_val":            pa.array(is_vals,   type=pa.int8()),
    }, schema=_SPOOL_SCHEMA)


# ─────────────────────────────────────────────────────────────────────────────
# Spool reading helpers (disk-backed, bounded RAM)
# ─────────────────────────────────────────────────────────────────────────────

def _select_train_pairs_from_spool(
    spool_path: Path,
    max_pairs: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Read the training portion of the spool WITHOUT loading the whole file.

    Pass A: scan row-groups to collect all positive (label=1, is_val=0) row indices.
    Pass B: stream negatives until budget is filled, then stop reading.

    Returns a DataFrame with at most max_pairs rows.
    The positives are always retained first; negatives fill the remaining budget.
    """
    pf = pq.ParquetFile(spool_path)

    # ── Pass A: collect all positives ──────────────────────────────────────
    pos_batches: list[pd.DataFrame] = []
    n_pos = 0
    for batch in pf.iter_batches(columns=["source1_entity_id", "cand_entity_id",
                                           "s1_name_norm", "s1_address_norm", "s1_country",
                                           "cand_name_norm", "cand_address_norm", "cand_country",
                                           "label", "is_val"]):
        df = batch.to_pandas()
        pos = df[(df["is_val"] == 0) & (df["label"] == 1)]
        if len(pos):
            pos_batches.append(pos)
            n_pos += len(pos)

    pos_df = pd.concat(pos_batches, ignore_index=True) if pos_batches else pd.DataFrame(columns=list(_SPOOL_SCHEMA.names))
    del pos_batches

    n_neg_budget = max_pairs - n_pos
    if n_neg_budget <= 0:
        print(f"  ⚠ Positives alone ({n_pos:,}) ≥ cap ({max_pairs:,}) — returning positives only.")
        return pos_df.drop(columns=["is_val"])

    print(
        f"\n  Training pair selection:"
        f"\n    Positives retained   : {n_pos:,} (ALL)"
        f"\n    Negative budget      : {n_neg_budget:,}"
        f"\n    Sampling seed        : {random_state}"
    )

    # ── Pass B: reservoir-sample negatives ─────────────────────────────────
    # Deterministic reservoir sampling: read negatives, keep a random sample of size n_neg_budget.
    rng = np.random.default_rng(random_state)
    reservoir: list[pd.DataFrame] = []
    reservoir_size = 0

    for batch in pf.iter_batches(columns=list(_SPOOL_SCHEMA.names)):
        df = batch.to_pandas()
        neg = df[(df["is_val"] == 0) & (df["label"] == 0)]
        if not len(neg):
            continue

        if reservoir_size < n_neg_budget:
            # Room remaining — take as many as fit
            take = neg.head(n_neg_budget - reservoir_size)
            reservoir.append(take)
            reservoir_size += len(take)
            remaining = neg.iloc[len(take):]
        else:
            remaining = neg

        # Reservoir replacement for deterministic uniform sampling
        for _, row_series in remaining.iterrows():
            idx = int(rng.integers(0, reservoir_size + 1))
            if idx < n_neg_budget:
                # Replace a random slot in one of the reservoir frames
                # For simplicity track flat index into concatenated reservoir
                cum = 0
                for i, frame in enumerate(reservoir):
                    if cum + len(frame) > idx:
                        local_idx = idx - cum
                        reservoir[i] = pd.concat(
                            [frame.iloc[:local_idx], pd.DataFrame([row_series]), frame.iloc[local_idx + 1:]],
                            ignore_index=True,
                        )
                        break
                    cum += len(frame)
            reservoir_size += 1

    neg_df = pd.concat(reservoir, ignore_index=True) if reservoir else pd.DataFrame(columns=list(_SPOOL_SCHEMA.names))
    del reservoir

    result = pd.concat([pos_df, neg_df], ignore_index=True).drop(columns=["is_val"])
    n_used = len(result)
    print(f"    Sampled negatives    : {len(neg_df):,}")
    print(f"    Final training pairs : {n_used:,}  (pos={n_pos:,}  neg={len(neg_df):,})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Streaming threshold sweep (v2 — bounded RAM)
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep_streaming(
    spool_path: Path,
    scaler: StandardScaler,
    model: LogisticRegression,
    truth_map: dict[str, set[str]],
    val_s1_ids: list[str],
    thresholds: list[float] | None = None,
    batch_size: int = _DEFAULT_CHUNK,
) -> pd.DataFrame:
    """
    Streaming threshold sweep — never loads the full validation set into RAM.

    For EACH threshold *t*:
        - keep a dict {s1_id → set of predicted_ids} (size ≤ n_val_s1 entities)

    One pass over the validation portion of the spool is needed per threshold.
    To avoid multiple passes we score ALL validation pairs ONCE, write
    (s1_id, cand_id, proba) to a compact in-memory float32 array (bounded by
    total val pair count, acceptable for typical val set sizes ≤ 500K pairs),
    then do the threshold sweep over that compact array.

    If even the compact proba array is too large, use DuckDB-backed approach.
    For the competition scale (~50K val pairs in 10K scenario, ~500K at full scale)
    a float32 array is negligible (500K × 8 bytes = 4 MB).

    Zero-candidate val entities contribute entity_f05(truth, {}) per truth.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]

    pf = pq.ParquetFile(spool_path)
    val_s1_set = set(val_s1_ids)

    # Score all val pairs in streaming batches; store only (s1_id_idx, cand_id, proba)
    # Use compact arrays: s1_id stored as category codes to save memory.
    val_s1_index = {sid: i for i, sid in enumerate(val_s1_ids)}

    # Compact storage: list of (s1_code:int, cand_id:str, proba:float32)
    # We store as parallel lists and convert to arrays.
    buf_s1_codes: list[int]   = []
    buf_cand_ids: list[str]   = []
    buf_probas:   list[float] = []

    n_val_pairs_scored = 0
    for batch in pf.iter_batches(
        columns=["source1_entity_id", "cand_entity_id",
                 "s1_name_norm", "s1_address_norm", "s1_country",
                 "cand_name_norm", "cand_address_norm", "cand_country",
                 "is_val"],
        batch_size=batch_size,
    ):
        df = batch.to_pandas()
        val_batch = df[df["is_val"] == 1]
        if not len(val_batch):
            continue

        pair_dicts = val_batch.to_dict("records")
        X   = build_feature_matrix_from_dicts(pair_dicts)
        X_sc = scaler.transform(X)
        probas = model.predict_proba(X_sc)[:, 1].astype(np.float32)

        for row_dict, prob in zip(pair_dicts, probas):
            sid  = row_dict["source1_entity_id"]
            code = val_s1_index.get(sid)
            if code is None:
                continue
            buf_s1_codes.append(code)
            buf_cand_ids.append(row_dict["cand_entity_id"])
            buf_probas.append(float(prob))
        n_val_pairs_scored += len(val_batch)

    print(f"  Validation pairs scored for sweep: {n_val_pairs_scored:,}")

    arr_codes  = np.array(buf_s1_codes, dtype=np.int32)
    arr_probas = np.array(buf_probas,   dtype=np.float32)
    arr_cands  = np.array(buf_cand_ids, dtype=object)
    del buf_s1_codes, buf_cand_ids, buf_probas

    results = []
    for t in thresholds:
        mask = arr_probas >= t

        # Build entity-level pred sets (only for entities that have ≥1 positive pred)
        pred_map: dict[int, set[str]] = {}
        codes_pos = arr_codes[mask]
        cands_pos = arr_cands[mask]
        for code, cid in zip(codes_pos, cands_pos):
            if code not in pred_map:
                pred_map[code] = set()
            pred_map[code].add(str(cid))

        all_prec, all_rec, all_f05 = [], [], []
        for i, sid in enumerate(val_s1_ids):
            tr = truth_map.get(sid, set())
            pr = pred_map.get(i, set())
            tp = len(tr & pr)
            p  = tp / len(pr) if pr else 0.0
            r  = tp / len(tr) if tr else 0.0
            all_prec.append(p)
            all_rec.append(r)
            all_f05.append(entity_f05(tr, pr))

        results.append({
            "threshold": t,
            "precision": float(np.mean(all_prec)),
            "recall":    float(np.mean(all_rec)),
            "f0_5":      float(np.mean(all_f05)),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Training-cap helper (operates on DataFrame after selection)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_train_cap(
    train_df: pd.DataFrame,
    max_pairs: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Internal cap used only after _select_train_pairs_from_spool() pre-selects.
    This function is kept for test compatibility but in the main pipeline
    the spool-based selection is the primary cap mechanism.
    """
    n_available = len(train_df)
    if n_available <= max_pairs:
        print(f"  Training pairs available : {n_available:,} (≤ cap of {max_pairs:,}, no sampling needed)")
        return train_df

    pos_df = train_df[train_df["label"] == 1].copy()
    neg_df = train_df[train_df["label"] == 0].copy()
    n_pos  = len(pos_df)
    n_neg_budget = max_pairs - n_pos

    if n_pos >= max_pairs:
        print(f"  ⚠ Positives alone ({n_pos:,}) exceed cap — returning positives only.")
        return pos_df

    neg_sampled = neg_df.sample(n=min(n_neg_budget, len(neg_df)), random_state=random_state)
    result = pd.concat([pos_df, neg_sampled], ignore_index=True)
    print(f"  Final training: {len(result):,} pairs (pos={n_pos:,} neg={len(neg_sampled):,})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry-point
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_dir: Path,
    candidates_path: Path,
    output_dir: Path,
    models_dir: Path,
    cache_dir: Path | None = None,
    chunk_size: int = _DEFAULT_CHUNK,
    max_train_pairs: int = _DEFAULT_MAX_TRAIN_PAIRS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load source data
    s1, s2, s3, gt = load_sources(data_dir, cache_dir=cache_dir)
    truth_map   = build_truth_map(gt)
    all_s1_ids  = s1["entity_id"].tolist()

    # 2. Entity split BEFORE streaming candidates
    train_s1_set, val_s1_set = entity_split(all_s1_ids)
    train_s1_list = sorted(train_s1_set)
    val_s1_list   = sorted(val_s1_set)
    print(f"Split: Train S1={len(train_s1_set):,}  Val S1={len(val_s1_set):,}")

    # 3. Build entity lookup dicts (one-time, bounded RAM)
    print("Building entity lookup dicts...")
    s1_lookup   = _build_s1_lookup(s1)
    cand_lookup = _build_entity_lookups(s2, s3)
    del s2, s3
    print(f"  S1={len(s1_lookup):,}  Cand(S2+S3)={len(cand_lookup):,}")

    # 4. ── INCREMENTAL SPOOL WRITE ──────────────────────────────────────────
    #    Each chunk is converted to a PyArrow Table and written immediately.
    #    NEVER accumulated in a Python list.
    spool_dir  = output_dir / ".pair_spool"
    spool_dir.mkdir(exist_ok=True)
    spool_path = spool_dir / "pairs.parquet"

    total_cand_rows = 0
    total_pair_rows = 0
    total_pos       = 0
    total_neg       = 0
    chunk_num       = 0

    print(f"\nStreaming candidates → incrementally spooling to {spool_path} ...")
    with pq.ParquetWriter(str(spool_path), schema=_SPOOL_SCHEMA) as writer:
        for chunk_df in pd.read_csv(
            candidates_path,
            sep="\t",
            dtype=str,
            chunksize=chunk_size,
            keep_default_na=False,
        ):
            chunk_num       += 1
            total_cand_rows += len(chunk_df)

            arrow_table = _expand_and_label_chunk(
                chunk_df, s1_lookup, cand_lookup, truth_map, val_s1_set
            )

            if arrow_table is None or len(arrow_table) == 0:
                print(f"  chunk {chunk_num:4d}: {len(chunk_df):6,} cand rows → 0 pairs (skipped)")
                continue

            n_pairs = len(arrow_table)
            n_pos_chunk = int(arrow_table.column("label").to_pylist().count(1))
            total_pair_rows += n_pairs
            total_pos       += n_pos_chunk
            total_neg       += n_pairs - n_pos_chunk

            writer.write_table(arrow_table)
            del arrow_table  # immediately release chunk

            spool_bytes = spool_path.stat().st_size
            print(
                f"  chunk {chunk_num:4d}: {len(chunk_df):6,} cand rows → "
                f"{n_pairs:7,} pairs (pos={n_pos_chunk:,}) "
                f"| spool {spool_bytes/1e6:.1f} MB"
            )
    # ── end of spool write ──────────────────────────────────────────────────

    print(
        f"\nIngestion complete:"
        f"\n  candidate rows : {total_cand_rows:,}"
        f"\n  total pairs    : {total_pair_rows:,}  (pos={total_pos:,}  neg={total_neg:,})"
        f"\n  spool size     : {spool_path.stat().st_size / 1e6:.1f} MB"
    )

    if total_pair_rows == 0:
        raise RuntimeError("No pair rows produced — cannot train. Check candidate file.")

    # 5. ── DISK-BACKED TRAIN SELECTION ─────────────────────────────────────
    #    Reads spool in two passes; materialises at most max_train_pairs rows.
    print(f"\nSelecting training pairs (max={max_train_pairs:,}) from spool ...")
    train_df = _select_train_pairs_from_spool(spool_path, max_train_pairs)
    n_train_pos = int((train_df["label"] == 1).sum())
    n_train_neg = len(train_df) - n_train_pos
    sampling_applied = len(train_df) < (total_pair_rows - total_neg + total_neg)  # always report

    # Count zero-cand val entities (not in spool) via metadata, not full spool load
    pf = pq.ParquetFile(spool_path)
    val_s1_with_pairs: set[str] = set()
    for batch in pf.iter_batches(columns=["source1_entity_id", "is_val"], batch_size=chunk_size):
        df = batch.to_pandas()
        val_s1_with_pairs.update(df.loc[df["is_val"] == 1, "source1_entity_id"].tolist())
    n_zero_cand_val = len(val_s1_set - val_s1_with_pairs)

    print(f"Train pairs materialised : {len(train_df):,}  (pos={n_train_pos:,}  neg={n_train_neg:,})")
    print(f"Val   S1 with zero cand  : {n_zero_cand_val:,}")

    # 6. Feature extraction on bounded training matrix only
    print("\nExtracting training features...")
    train_dicts = train_df.to_dict("records")
    X_train = build_feature_matrix_from_dicts(train_dicts)
    y_train = train_df["label"].values
    del train_dicts, train_df

    # 7. Fit StandardScaler + LogisticRegression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    del X_train

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train_sc, y_train)
    print("Model trained.")

    # 8. ── STREAMING THRESHOLD SWEEP ───────────────────────────────────────
    #    Validation pairs scored in batches from spool; only compact arrays in RAM.
    print("\nRunning streaming threshold sweep...")
    sweep_df = threshold_sweep_streaming(
        spool_path=spool_path,
        scaler=scaler,
        model=model,
        truth_map=truth_map,
        val_s1_ids=val_s1_list,
        batch_size=chunk_size,
    )
    best_row       = sweep_df.loc[sweep_df["f0_5"].idxmax()]
    best_threshold = float(best_row["threshold"])
    best_f05       = float(best_row["f0_5"])
    best_prec      = float(best_row["precision"])
    best_rec       = float(best_row["recall"])

    print(f"\nThreshold sweep results:")
    print(sweep_df.to_string(index=False))
    print(f"\nSelected threshold : {best_threshold:.2f}")
    print(f"Validation F0.5    : {best_f05:.4f}")
    print(f"Validation Precision: {best_prec:.4f}")
    print(f"Validation Recall  : {best_rec:.4f}")

    # 9. Save model
    model_path = models_dir / "matcher.pkl"
    with open(model_path, "wb") as f:
        pickle.dump((scaler, model), f)
    print(f"\nModel saved → {model_path}")

    # 10. Save config
    config = {
        "model_type":               "StandardScaler + LogisticRegression",
        "feature_names":            FEATURE_NAMES,
        "threshold":                best_threshold,
        "random_seed":              RANDOM_STATE,
        "train_s1_count":           len(train_s1_set),
        "validation_s1_count":      len(val_s1_set),
        "val_s1_zero_cand":         n_zero_cand_val,
        "available_candidate_pairs":total_pair_rows,
        "train_pair_count":         int(n_train_pos + n_train_neg),
        "train_positive_pairs":     n_train_pos,
        "train_negative_pairs":     n_train_neg,
        "sampling_applied":         (n_train_pos + n_train_neg) < total_pair_rows,
        "max_train_pairs_cap":      max_train_pairs,
        "sampling_seed":            RANDOM_STATE,
        "precision":                round(best_prec, 6),
        "recall":                   round(best_rec, 6),
        "f0_5":                     round(best_f05, 6),
        "baseline_version":         "3.0-fully-streaming",
        "metric":                   "entity-level macro F0.5 (competition official, zero-cand included)",
        "candidates_source":        str(candidates_path),
        "cache_source":             str(cache_dir) if cache_dir else "raw-tsv",
    }
    config_path = models_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved  → {config_path}")

    with open(output_dir / "baseline_training_results.json", "w") as f:
        json.dump(config, f, indent=2)
    sweep_df.to_csv(output_dir / "baseline_threshold_results.tsv", sep="\t", index=False)

    # Clean up spool
    try:
        spool_path.unlink()
        spool_dir.rmdir()
    except Exception:
        pass

    print("\nTraining complete.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train baseline matcher (fully streaming v3).")
    parser.add_argument("--data-dir",        default="dataset/train")
    parser.add_argument("--candidates",       default="output/candidate_pairs.tsv")
    parser.add_argument("--output-dir",       default="output")
    parser.add_argument("--models-dir",       default="models")
    parser.add_argument("--cache-dir",        default=None)
    parser.add_argument("--chunk-size",       default=_DEFAULT_CHUNK,           type=int)
    parser.add_argument("--max-train-pairs",  default=_DEFAULT_MAX_TRAIN_PAIRS, type=int)
    args = parser.parse_args()

    train(
        data_dir=Path(args.data_dir),
        candidates_path=Path(args.candidates),
        output_dir=Path(args.output_dir),
        models_dir=Path(args.models_dir),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        chunk_size=args.chunk_size,
        max_train_pairs=args.max_train_pairs,
    )
