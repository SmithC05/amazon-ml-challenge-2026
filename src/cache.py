"""
src/cache.py
============
One-time preprocessing cache for the Amazon ML Challenge 2026.

Member 2 deliverable — preprocessing + cache layer.

Purpose
-------
Apply M2 normalization exactly ONCE per source record and persist the result
as Parquet so that M3 feature extraction and M4 candidate generation never
re-normalize raw text during experiments.

Cache layout
------------
cache/
  train_source1.parquet
  train_source2.parquet
  train_source3.parquet
  test_source1.parquet     (written only when test data is present)
  test_source2.parquet
  test_source3.parquet

Each Parquet file contains all original columns plus:
  business_name_norm       — output of normalize_name(business_name)
  business_address_norm    — output of normalize_address(business_address)
  name_tokens              — frozenset-style whitespace-token count (int, for quick stats)
  address_tokens           — whitespace-token count (int)
  address_is_empty         — 1 if normalized address == "" else 0

Public API
----------
  build_cache(data_dir, cache_dir, split)   — normalize and write Parquet
  load_cache(cache_dir, split, source)      — load one Parquet into DataFrame
  load_all_cache(cache_dir, split)          — load S1, S2, S3 as a tuple

Design constraints
------------------
  • Normalization logic is NEVER reimplemented here.  Only preprocess.py is used.
  • Cache files are excluded from git (see .gitignore).
  • Parquet is chosen for columnar compression and fast partial reads.
  • pyarrow is the only new dependency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pandas as pd

# M2 normalization — single source of truth
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import normalize_name, normalize_address   # noqa: E402


Split  = Literal["train", "test"]
Source = Literal["source1", "source2", "source3"]

_SOURCE_FILES = {
    "train": {
        "source1": "train_source1.tsv",
        "source2": "train_source2.tsv",
        "source3": "train_source3.tsv",
    },
    "test": {
        "source1": "test_source1.tsv",
        "source2": "test_source2.tsv",
        "source3": "test_source3.tsv",
    },
}

_CACHE_NAMES = {
    ("train", "source1"): "train_source1.parquet",
    ("train", "source2"): "train_source2.parquet",
    ("train", "source3"): "train_source3.parquet",
    ("test",  "source1"): "test_source1.parquet",
    ("test",  "source2"): "test_source2.parquet",
    ("test",  "source3"): "test_source3.parquet",
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply M2 normalization to a source DataFrame and add derived columns.
    Returns a new DataFrame — never modifies in place.
    """
    out = df.copy()

    # Core M2 normalization — never reimplemented here
    out["business_name_norm"]    = out["business_name"].apply(normalize_name)
    out["business_address_norm"] = out["business_address"].apply(normalize_address)

    # Cheap derived fields useful for blocking and feature engineering
    out["name_tokens"]     = out["business_name_norm"].str.split().apply(len)
    out["address_tokens"]  = out["business_address_norm"].str.split().apply(
        lambda t: len(t) if isinstance(t, list) else 0
    )
    out["address_is_empty"] = (out["business_address_norm"] == "").astype("int8")

    return out


def _cache_path(cache_dir: Path, split: Split, source: Source) -> Path:
    return cache_dir / _CACHE_NAMES[(split, source)]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_cache(
    data_dir: Path | str,
    cache_dir: Path | str,
    split: Split = "train",
    force: bool = False,
) -> dict[str, float]:
    """
    Normalize every source record and write to Parquet.

    Parameters
    ----------
    data_dir  : directory containing the raw TSV files
    cache_dir : directory where Parquet files are written (created if absent)
    split     : "train" or "test"
    force     : if True, overwrite existing cache files

    Returns
    -------
    dict with timing info:
      {source: elapsed_seconds, "total": total_seconds}
    """
    data_dir  = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    for source in ("source1", "source2", "source3"):
        tsv_name = _SOURCE_FILES[split].get(source)
        tsv_path = data_dir / tsv_name
        out_path = _cache_path(cache_dir, split, source)   # type: ignore[arg-type]

        if not tsv_path.exists():
            print(f"  SKIP {tsv_path.name} — file not found")
            continue

        if out_path.exists() and not force:
            print(f"  SKIP {out_path.name} — already cached (use force=True to rebuild)")
            continue

        t0 = time.perf_counter()
        df_raw = pd.read_csv(tsv_path, sep="\t")
        df_norm = _normalize_df(df_raw)
        df_norm.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        elapsed = time.perf_counter() - t0

        timings[source] = elapsed
        print(f"  BUILT {out_path.name}  ({len(df_norm):,} rows, {elapsed:.2f}s)")

    timings["total"] = time.perf_counter() - total_start
    print(f"  Cache build total: {timings['total']:.2f}s")
    return timings


def load_cache(
    cache_dir: Path | str,
    split: Split,
    source: Source,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load a cached Parquet file into a DataFrame.

    Parameters
    ----------
    cache_dir : directory where Parquet files live
    split     : "train" or "test"
    source    : "source1", "source2", or "source3"
    columns   : optional list of column names to load (loads all if None)

    Returns
    -------
    DataFrame with at minimum: entity_id, business_name_norm,
    business_address_norm, country.

    Raises
    ------
    FileNotFoundError if the Parquet file does not exist.
    """
    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, split, source)  # type: ignore[arg-type]

    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}\n"
            f"Run build_cache(data_dir, cache_dir, split='{split}') first."
        )

    t0 = time.perf_counter()
    df = pd.read_parquet(path, engine="pyarrow", columns=columns)
    elapsed = time.perf_counter() - t0
    print(f"  LOAD  {path.name}  ({len(df):,} rows, {elapsed:.3f}s)")
    return df


def load_all_cache(
    cache_dir: Path | str,
    split: Split = "train",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load source1, source2, source3 caches for *split*.

    Returns
    -------
    (s1, s2, s3) DataFrames, each with normalized + derived columns.
    """
    cache_dir = Path(cache_dir)
    s1 = load_cache(cache_dir, split, "source1")
    s2 = load_cache(cache_dir, split, "source2")
    s3 = load_cache(cache_dir, split, "source3")
    return s1, s2, s3


def cache_exists(cache_dir: Path | str, split: Split) -> bool:
    """Return True if all three source caches for *split* are present."""
    cache_dir = Path(cache_dir)
    return all(
        _cache_path(cache_dir, split, src).exists()  # type: ignore[arg-type]
        for src in ("source1", "source2", "source3")
    )


def validate_cache(
    data_dir: Path | str,
    cache_dir: Path | str,
    split: Split = "train",
    n_check: int = 500,
) -> bool:
    """
    Spot-check that cached normalized values match fresh M2 output.

    Samples *n_check* rows from each source and re-normalizes them,
    comparing against the cached values.  Raises AssertionError on mismatch.
    """
    import numpy as np

    data_dir  = Path(data_dir)
    cache_dir = Path(cache_dir)
    all_ok = True

    for source in ("source1", "source2", "source3"):
        tsv_path  = data_dir / _SOURCE_FILES[split][source]
        if not tsv_path.exists():
            print(f"  SKIP validation for {source} — raw file not found")
            continue

        cached = load_cache(cache_dir, split, source)   # type: ignore[arg-type]
        raw    = pd.read_csv(tsv_path, sep="\t")

        # Sample rows present in both
        sample_ids = raw["entity_id"].sample(
            min(n_check, len(raw)), random_state=42
        ).tolist()

        raw_sample    = raw[raw["entity_id"].isin(sample_ids)].set_index("entity_id")
        cached_sample = cached[cached["entity_id"].isin(sample_ids)].set_index("entity_id")

        # Re-normalize and compare
        fresh_name = raw_sample["business_name"].apply(normalize_name)
        fresh_addr = raw_sample["business_address"].apply(normalize_address)

        name_match = (fresh_name == cached_sample["business_name_norm"]).all()
        addr_match = (fresh_addr == cached_sample["business_address_norm"]).all()

        status = "OK" if (name_match and addr_match) else "MISMATCH"
        print(f"  VALIDATE {split}/{source}: name={name_match} addr={addr_match} → {status}")

        if not (name_match and addr_match):
            all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build M2 preprocessing cache.")
    parser.add_argument("--data-dir",   required=True, help="Directory with raw TSV files")
    parser.add_argument("--cache-dir",  default="cache", help="Output directory for Parquet files")
    parser.add_argument("--split",      default="train", choices=["train", "test"])
    parser.add_argument("--force",      action="store_true", help="Overwrite existing cache")
    parser.add_argument("--validate",   action="store_true", help="Validate cache after building")
    args = parser.parse_args()

    print(f"Building cache: split={args.split}  data_dir={args.data_dir}")
    timings = build_cache(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        split=args.split,
        force=args.force,
    )

    if args.validate:
        print("\nValidating cache...")
        ok = validate_cache(args.data_dir, args.cache_dir, split=args.split)
        if not ok:
            raise SystemExit("Cache validation failed.")
        print("Validation passed.")
