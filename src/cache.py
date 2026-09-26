"""
src/cache.py
============
Parquet-based preprocessing cache for the Amazon ML Challenge 2026 pipeline.

All source records (S1, S2, S3 for both train and test splits) are preprocessed
exactly once using src.preprocess.preprocess_dataframe() and saved to disk as
Parquet files in the project-level cache/ directory.

Downstream notebooks and scripts load a preprocessed source with load_cache()
instead of re-running normalization on every run.

Public API
----------
    build_cache(split, source, data_dir, cache_dir, force=False)
    load_cache(split, source, cache_dir)
    load_all_cache(cache_dir, split)
    cache_exists(split, source, cache_dir)
    validate_cache(split, source, cache_dir)

Naming convention
-----------------
    TSV  :  <data_dir>/<split>_<source>.tsv     e.g.  /content/train_source1.tsv
    Cache:  <cache_dir>/<split>_<source>.parquet e.g.  cache/train_source1.parquet

Cached schema (14 columns)
--------------------------
    entity_id, business_name, business_address, country
    name_norm, address_norm
    name_tokens, address_tokens
    name_token_count, address_token_count
    name_length, address_length
    name_digits, address_digits

Design principles
-----------------
    • Normalization is delegated entirely to preprocess_dataframe(); no
      normalization logic is duplicated here.
    • Original TSV files are never modified.
    • The cache directory is created automatically if it does not exist.
    • build_cache() is idempotent by default: it skips rebuilding if the cache
      file already exists unless force=True is passed.
    • load_cache() never rebuilds; it raises FileNotFoundError if the cache is
      absent.
    • No candidate-pair generation, fuzzy matching, blocking, ML training,
      feature engineering, or submission logic is present in this module.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Literal, Union

import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
_sys.path.insert(0, str(Path(__file__).resolve().parent))         # src/

try:
    from src.preprocess import preprocess_dataframe, _OUTPUT_COLUMNS  # package import
except ModuleNotFoundError:
    from preprocess import preprocess_dataframe, _OUTPUT_COLUMNS       # script import


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

Split  = Literal["train", "test"]
Source = Literal["source1", "source2", "source3"]

_VALID_SPLITS  = {"train", "test"}
_VALID_SOURCES = {"source1", "source2", "source3"}

# Exactly the 14 columns that preprocess_dataframe() guarantees.
_REQUIRED_COLUMNS = set(_OUTPUT_COLUMNS)

_SOURCE_FILES: dict[str, dict[str, str]] = {
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


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_args(split: str, source: str) -> None:
    """Raise ValueError if split or source is not recognised."""
    if split not in _VALID_SPLITS:
        raise ValueError(
            f"split must be one of {_VALID_SPLITS!r}, got {split!r}"
        )
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"source must be one of {_VALID_SOURCES!r}, got {source!r}"
        )


def _cache_path(split: str, source: str, cache_dir: Union[str, Path]) -> Path:
    """Return the expected Parquet file path for (split, source)."""
    return Path(cache_dir) / f"{split}_{source}.parquet"


def _tsv_path(split: str, source: str, data_dir: Union[str, Path]) -> Path:
    """Return the expected TSV file path for (split, source)."""
    return Path(data_dir) / _SOURCE_FILES[split][source]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def cache_exists(
    split: str,
    source: str,
    cache_dir: Union[str, Path] = "cache",
) -> bool:
    """
    Return True if the Parquet cache file for (split, source) exists on disk.

    Args:
        split:     "train" or "test".
        source:    "source1", "source2", or "source3".
        cache_dir: Directory that contains the Parquet files. Defaults to
                   "cache" (relative to the current working directory).

    Returns:
        True if the expected Parquet file is present; False otherwise.
    """
    _validate_args(split, source)
    return _cache_path(split, source, cache_dir).is_file()


def build_cache(
    split: str,
    source: str,
    data_dir: Union[str, Path] = "/content",
    cache_dir: Union[str, Path] = "cache",
    force: bool = False,
    chunksize: int | None = None,
) -> dict:
    """
    Preprocess a source TSV and save it as a Parquet cache file.

    The function:
      1. Validates arguments.
      2. Skips building if the cache already exists and force=False.
      3. Reads the source TSV from data_dir.
      4. Passes the DataFrame through preprocess_dataframe().
      5. Saves the result as Parquet in cache_dir.
      6. Returns a status dictionary.

    The original TSV is never modified.  Normalization is performed exactly
    once per source record by delegating to preprocess_dataframe().

    Args:
        split:     "train" or "test".
        source:    "source1", "source2", or "source3".
        data_dir:  Directory containing the source TSV files.
                   Defaults to "/content" (Google Colab mount point).
        cache_dir: Directory in which to write the Parquet file.
                   Created automatically if it does not exist.
        force:     If True, rebuild even if the cache file already exists.
        chunksize: Optional integer.  When None (default), the entire TSV is
                   read into memory at once before preprocessing — this is the
                   original behavior and is unchanged.

                   When set to a positive integer (e.g. 500_000), the TSV is
                   read and preprocessed one chunk at a time.  Each preprocessed
                   chunk is written to Parquet incrementally using PyArrow's
                   ParquetWriter so that peak RAM is bounded to one chunk rather
                   than the entire file.  The final Parquet file is byte-for-byte
                   equivalent in schema and semantically equivalent in content to
                   the non-chunked output.  Downstream load_cache() and
                   validate_cache() calls require no changes.

                   Recommended value for the competition sources: 500_000.

    Returns:
        A dict with keys:
            split, source, cache_path, tsv_path,
            status ("built" | "skipped" | "error"),
            rows (int, or None on skip/error),
            error (str, or None on success).
    """
    _validate_args(split, source)

    tsv   = _tsv_path(split, source, data_dir)
    cache = _cache_path(split, source, cache_dir)

    result: dict = {
        "split":      split,
        "source":     source,
        "tsv_path":   str(tsv),
        "cache_path": str(cache),
        "status":     None,
        "rows":       None,
        "error":      None,
    }

    # Skip if cache already present and not forced.
    if cache.is_file() and not force:
        result["status"] = "skipped"
        return result

    try:
        # Ensure cache directory exists.
        cache.parent.mkdir(parents=True, exist_ok=True)

        if chunksize is None:
            # ── Original whole-file path (unchanged behavior) ──────────────
            df_raw = pd.read_csv(tsv, sep="\t", dtype=str)
            df_processed = preprocess_dataframe(df_raw)
            df_processed.to_parquet(
                cache, index=False, engine="pyarrow", compression="snappy"
            )
            result["rows"] = len(df_processed)

        else:
            # ── Chunked path: bounded peak RAM ─────────────────────────────
            # Uses PyArrow's ParquetWriter so we never hold more than one
            # preprocessed chunk in memory simultaneously.
            import pyarrow as pa
            import pyarrow.parquet as pq

            writer = None
            total_rows = 0
            try:
                reader = pd.read_csv(
                    tsv, sep="\t", dtype=str, chunksize=chunksize
                )
                for chunk_raw in reader:
                    chunk_proc = preprocess_dataframe(chunk_raw)
                    table = pa.Table.from_pandas(chunk_proc, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            cache,
                            table.schema,
                            compression="snappy",
                        )
                    writer.write_table(table)
                    total_rows += len(chunk_proc)
            finally:
                if writer is not None:
                    writer.close()

            result["rows"] = total_rows

        result["status"] = "built"

    except Exception as exc:                   # noqa: BLE001
        result["status"] = "error"
        result["error"]  = str(exc)

    return result


def load_cache(
    split: str,
    source: str,
    cache_dir: Union[str, Path] = "cache",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load a preprocessed source from its Parquet cache file.

    This function never rebuilds the cache.  Call build_cache() first if the
    cache has not yet been created.

    Args:
        split:     "train" or "test".
        source:    "source1", "source2", or "source3".
        cache_dir: Directory that contains the Parquet files.
        columns:   Optional list of column names to load (loads all 14 if None).

    Returns:
        A DataFrame with the 14 columns produced by preprocess_dataframe()
        (or a subset if columns= is specified).

    Raises:
        FileNotFoundError: if the expected cache file does not exist.
        ValueError:        if split or source is invalid.
    """
    _validate_args(split, source)

    cache = _cache_path(split, source, cache_dir)

    if not cache.is_file():
        raise FileNotFoundError(
            f"Cache file not found: {cache}\n"
            f"Run build_cache('{split}', '{source}', ...) to create it."
        )

    t0 = time.perf_counter()
    df = pd.read_parquet(cache, engine="pyarrow", columns=columns)
    elapsed = time.perf_counter() - t0
    print(f"  LOAD  {cache.name}  ({len(df):,} rows, {elapsed:.3f}s)")
    return df


def load_all_cache(
    cache_dir: Union[str, Path] = "cache",
    split: str = "train",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function: load source1, source2, source3 caches for *split*.

    Args:
        cache_dir: Directory that contains the Parquet files.
        split:     "train" or "test".
        columns:   Optional list of column names to load.

    Returns:
        (s1, s2, s3) — three DataFrames, one per source.
    """
    s1 = load_cache(split, "source1", cache_dir, columns=columns)
    s2 = load_cache(split, "source2", cache_dir, columns=columns)
    s3 = load_cache(split, "source3", cache_dir, columns=columns)
    return s1, s2, s3


def validate_cache(
    split: str,
    source: str,
    cache_dir: Union[str, Path] = "cache",
) -> dict:
    """
    Validate a Parquet cache file for (split, source).

    Checks (in order):
      1. File exists on disk.
      2. Parquet file can be read without error.
      3. All 14 required columns are present.
      4. entity_id column is present.
      5. Raw columns (business_name, business_address, country) are present.
      6. Normalized columns (name_norm, address_norm) are present.
      7. No completely-empty DataFrame (at least one row).

    Args:
        split:     "train" or "test".
        source:    "source1", "source2", or "source3".
        cache_dir: Directory that contains the Parquet files.

    Returns:
        A dict with keys:
            valid (bool),
            cache_path (str),
            rows (int or None),
            columns (list or None),
            checks (dict mapping check_name -> bool),
            missing_columns (list),
            error (str or None).
    """
    _validate_args(split, source)

    cache = _cache_path(split, source, cache_dir)

    checks: dict[str, bool] = {
        "file_exists":          False,
        "parquet_readable":     False,
        "all_14_columns":       False,
        "entity_id_present":    False,
        "raw_columns_present":  False,
        "norm_columns_present": False,
        "non_empty":            False,
    }

    result: dict = {
        "valid":           False,
        "cache_path":      str(cache),
        "rows":            None,
        "columns":         None,
        "checks":          checks,
        "missing_columns": [],
        "error":           None,
    }

    # 1. File exists
    checks["file_exists"] = cache.is_file()
    if not checks["file_exists"]:
        result["error"] = f"File not found: {cache}"
        return result

    # 2. Readable
    try:
        df = pd.read_parquet(cache, engine="pyarrow")
        checks["parquet_readable"] = True
    except Exception as exc:                   # noqa: BLE001
        result["error"] = f"Cannot read Parquet: {exc}"
        return result

    cols = set(df.columns)
    result["rows"]    = len(df)
    result["columns"] = list(df.columns)

    # 3. All 14 required columns
    missing = _REQUIRED_COLUMNS - cols
    result["missing_columns"]   = sorted(missing)
    checks["all_14_columns"]    = len(missing) == 0

    # 4. entity_id
    checks["entity_id_present"] = "entity_id" in cols

    # 5. Raw columns
    checks["raw_columns_present"] = {
        "business_name", "business_address", "country"
    }.issubset(cols)

    # 6. Normalized columns
    checks["norm_columns_present"] = {
        "name_norm", "address_norm"
    }.issubset(cols)

    # 7. Non-empty
    checks["non_empty"] = len(df) > 0

    result["valid"] = all(checks.values())
    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the M2 preprocessing cache for one or all sources."
    )
    parser.add_argument("--data-dir",  required=True,
                        help="Directory containing raw TSV files")
    parser.add_argument("--cache-dir", default="cache",
                        help="Output directory for Parquet files (default: cache/)")
    parser.add_argument("--split",     default="train",
                        choices=["train", "test"],
                        help="Which split to build (default: train)")
    parser.add_argument("--source",    default="all",
                        choices=["source1", "source2", "source3", "all"],
                        help="Which source to build, or 'all' (default: all)")
    parser.add_argument("--force",     action="store_true",
                        help="Overwrite existing cache files")
    parser.add_argument("--chunksize", type=int, default=None,
                        help="Rows per chunk for memory-safe build "
                             "(e.g. 500000). Default: whole-file (None).")
    args = parser.parse_args()

    sources = (
        ["source1", "source2", "source3"]
        if args.source == "all"
        else [args.source]
    )

    chunk_msg = f"  chunksize={args.chunksize:,}" if args.chunksize else "  chunksize=None (whole-file)"
    print(f"Building cache: split={args.split}  data_dir={args.data_dir}{chunk_msg}")
    for src in sources:
        r = build_cache(
            split     = args.split,
            source    = src,
            data_dir  = args.data_dir,
            cache_dir = args.cache_dir,
            force     = args.force,
            chunksize = args.chunksize,
        )
        rows_str = f"{r['rows']:,}" if r["rows"] else "n/a"
        print(f"  {args.split}_{src}: {r['status']}  rows={rows_str}  {r['error'] or ''}")
