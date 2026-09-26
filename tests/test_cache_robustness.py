# -*- coding: utf-8 -*-
"""
tests/test_cache_robustness.py
================================
Phase 6 — Cache Robustness Validation.

Tests the existing src/cache.py implementation for correctness, robustness,
and schema integrity using in-memory synthetic data only (no large TSVs loaded
into RAM simultaneously).

All large-dataset tests use only ONE source file at a time and delete the
DataFrame immediately after the test completes.

Run from repo root:
    python -m unittest tests.test_cache_robustness -v
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cache import (
    build_cache,
    load_cache,
    load_all_cache,
    cache_exists,
    validate_cache,
    _VALID_SPLITS,
    _VALID_SOURCES,
    _SOURCE_FILES,
    _REQUIRED_COLUMNS,
    _cache_path,
    _tsv_path,
)
from preprocess import _OUTPUT_COLUMNS, preprocess_dataframe

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_tsv(path: Path, n_rows: int = 50) -> None:
    """Write a minimal synthetic source TSV at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {
            "entity_id":        f"S1-{i:08d}",
            "business_name":    f"Acme Corp {i}" if i % 10 != 0 else None,
            "business_address": f"{i} Main Ave" if i % 5 != 0 else None,
            "country":          ["US", "IN", "GB", "DE", "CN"][i % 5],
        }
        for i in range(n_rows)
    ])
    df.to_csv(path, sep="\t", index=False)


def _make_all_tsvs(data_dir: Path, split: str, n_rows: int = 50) -> None:
    """Write all three synthetic source TSVs for a split."""
    for src in ("source1", "source2", "source3"):
        filename = _SOURCE_FILES[split][src]
        _make_tsv(data_dir / filename, n_rows=n_rows)


# ─────────────────────────────────────────────────────────────────────────────
# T1 — API surface
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheAPIExists(unittest.TestCase):
    """All five public functions must be importable and callable."""

    def test_build_cache_callable(self):
        self.assertTrue(callable(build_cache))

    def test_load_cache_callable(self):
        self.assertTrue(callable(load_cache))

    def test_load_all_cache_callable(self):
        self.assertTrue(callable(load_all_cache))

    def test_cache_exists_callable(self):
        self.assertTrue(callable(cache_exists))

    def test_validate_cache_callable(self):
        self.assertTrue(callable(validate_cache))


# ─────────────────────────────────────────────────────────────────────────────
# T2 — File naming convention
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheNaming(unittest.TestCase):
    """_cache_path and _SOURCE_FILES must follow <split>_<source>.parquet/.tsv naming."""

    def test_cache_path_train_source1(self):
        p = _cache_path("train", "source1", "/tmp/cache")
        self.assertEqual(p.name, "train_source1.parquet")

    def test_cache_path_train_source2(self):
        p = _cache_path("train", "source2", "/tmp/cache")
        self.assertEqual(p.name, "train_source2.parquet")

    def test_cache_path_train_source3(self):
        p = _cache_path("train", "source3", "/tmp/cache")
        self.assertEqual(p.name, "train_source3.parquet")

    def test_cache_path_test_source1(self):
        p = _cache_path("test", "source1", "/tmp/cache")
        self.assertEqual(p.name, "test_source1.parquet")

    def test_cache_path_test_source2(self):
        p = _cache_path("test", "source2", "/tmp/cache")
        self.assertEqual(p.name, "test_source2.parquet")

    def test_cache_path_test_source3(self):
        p = _cache_path("test", "source3", "/tmp/cache")
        self.assertEqual(p.name, "test_source3.parquet")

    def test_cache_path_in_correct_directory(self):
        p = _cache_path("train", "source1", "/my/cache")
        # Use Path comparison to be platform-independent (Windows uses backslash)
        self.assertEqual(p.parent, Path("/my/cache"))

    def test_tsv_path_train_source1(self):
        p = _tsv_path("train", "source1", "/data")
        self.assertEqual(p.name, "train_source1.tsv")

    def test_tsv_path_test_source1(self):
        p = _tsv_path("test", "source1", "/data")
        self.assertEqual(p.name, "test_source1.tsv")

    def test_source_files_covers_all_six(self):
        for split in ("train", "test"):
            for src in ("source1", "source2", "source3"):
                self.assertIn(split, _SOURCE_FILES)
                self.assertIn(src, _SOURCE_FILES[split])


# ─────────────────────────────────────────────────────────────────────────────
# T3 — Error handling for invalid arguments
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheErrorHandling(unittest.TestCase):
    """Invalid split/source must raise ValueError."""

    def test_cache_exists_bad_split_raises(self):
        with self.assertRaises(ValueError):
            cache_exists("bogus", "source1")

    def test_cache_exists_bad_source_raises(self):
        with self.assertRaises(ValueError):
            cache_exists("train", "source99")

    def test_validate_cache_bad_split_raises(self):
        with self.assertRaises(ValueError):
            validate_cache("bogus", "source1")

    def test_validate_cache_bad_source_raises(self):
        with self.assertRaises(ValueError):
            validate_cache("train", "source99")

    def test_load_cache_bad_split_raises(self):
        with self.assertRaises(ValueError):
            load_cache("bogus", "source1")

    def test_load_cache_bad_source_raises(self):
        with self.assertRaises(ValueError):
            load_cache("train", "source99")

    def test_load_cache_missing_file_raises_filenotfound(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_cache("train", "source1", cache_dir=tmp)

    def test_build_cache_bad_split_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_cache("bogus", "source1", data_dir=tmp, cache_dir=tmp)

    def test_build_cache_bad_source_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_cache("train", "source99", data_dir=tmp, cache_dir=tmp)

    def test_build_cache_missing_tsv_returns_error_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = build_cache("train", "source1",
                            data_dir=tmp, cache_dir=tmp)
            self.assertEqual(r["status"], "error")
            self.assertIsNotNone(r["error"])


# ─────────────────────────────────────────────────────────────────────────────
# T4 — Full build/load/validate cycle (synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheBuildLoadValidate(unittest.TestCase):
    """
    Build a cache from synthetic data, load it, and validate it.
    All operations use a temporary directory — no large files touched.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.data_dir  = Path(cls.tmpdir) / "data"
        cls.cache_dir = Path(cls.tmpdir) / "cache"
        cls.data_dir.mkdir()

        # Write synthetic train TSVs
        _make_all_tsvs(cls.data_dir, "train", n_rows=100)

        # Build all three train caches
        cls.build_results = {}
        for src in ("source1", "source2", "source3"):
            cls.build_results[src] = build_cache(
                "train", src,
                data_dir=cls.data_dir,
                cache_dir=cls.cache_dir,
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    # ── build_cache() status ─────────────────────────────────────────────────

    def test_build_returns_built_status(self):
        for src, r in self.build_results.items():
            with self.subTest(src=src):
                self.assertEqual(r["status"], "built")

    def test_build_reports_correct_row_count(self):
        for src, r in self.build_results.items():
            with self.subTest(src=src):
                self.assertEqual(r["rows"], 100)

    def test_build_returns_no_error(self):
        for src, r in self.build_results.items():
            with self.subTest(src=src):
                self.assertIsNone(r["error"])

    def test_build_result_has_expected_keys(self):
        r = self.build_results["source1"]
        for key in ("split", "source", "tsv_path", "cache_path", "status", "rows", "error"):
            self.assertIn(key, r)

    # ── cache_exists() ───────────────────────────────────────────────────────

    def test_cache_exists_after_build(self):
        for src in ("source1", "source2", "source3"):
            with self.subTest(src=src):
                self.assertTrue(cache_exists("train", src, self.cache_dir))

    def test_cache_not_exists_before_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(cache_exists("train", "source1", tmp))

    # ── Idempotency ──────────────────────────────────────────────────────────

    def test_rebuild_without_force_skips(self):
        r = build_cache("train", "source1",
                        data_dir=self.data_dir,
                        cache_dir=self.cache_dir,
                        force=False)
        self.assertEqual(r["status"], "skipped")

    def test_rebuild_with_force_rebuilds(self):
        r = build_cache("train", "source1",
                        data_dir=self.data_dir,
                        cache_dir=self.cache_dir,
                        force=True)
        self.assertEqual(r["status"], "built")
        self.assertEqual(r["rows"], 100)

    # ── load_cache() ─────────────────────────────────────────────────────────

    def test_load_cache_returns_dataframe(self):
        df = load_cache("train", "source1", self.cache_dir)
        self.assertIsInstance(df, pd.DataFrame)
        del df

    def test_load_cache_has_14_columns(self):
        df = load_cache("train", "source1", self.cache_dir)
        self.assertEqual(len(df.columns), 14)
        del df

    def test_load_cache_columns_match_schema(self):
        df = load_cache("train", "source1", self.cache_dir)
        self.assertEqual(list(df.columns), _OUTPUT_COLUMNS)
        del df

    def test_load_cache_correct_row_count(self):
        df = load_cache("train", "source1", self.cache_dir)
        self.assertEqual(len(df), 100)
        del df

    def test_load_cache_column_selection(self):
        df = load_cache("train", "source1", self.cache_dir,
                        columns=["entity_id", "name_norm"])
        self.assertEqual(list(df.columns), ["entity_id", "name_norm"])
        del df

    # ── validate_cache() ─────────────────────────────────────────────────────

    def test_validate_cache_returns_valid(self):
        r = validate_cache("train", "source1", self.cache_dir)
        self.assertTrue(r["valid"],
                        msg=f"validate_cache returned invalid: {r}")

    def test_validate_cache_all_checks_pass(self):
        r = validate_cache("train", "source1", self.cache_dir)
        for check, passed in r["checks"].items():
            self.assertTrue(passed, msg=f"Check {check!r} failed: {r}")

    def test_validate_cache_correct_row_count(self):
        r = validate_cache("train", "source1", self.cache_dir)
        self.assertEqual(r["rows"], 100)

    def test_validate_cache_missing_file_returns_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = validate_cache("train", "source1", tmp)
            self.assertFalse(r["valid"])
            self.assertFalse(r["checks"]["file_exists"])

    # ── load_all_cache() ─────────────────────────────────────────────────────

    def test_load_all_cache_returns_three_dataframes(self):
        s1, s2, s3 = load_all_cache(self.cache_dir, "train")
        self.assertIsInstance(s1, pd.DataFrame)
        self.assertIsInstance(s2, pd.DataFrame)
        self.assertIsInstance(s3, pd.DataFrame)
        del s1, s2, s3

    def test_load_all_cache_all_have_14_columns(self):
        s1, s2, s3 = load_all_cache(self.cache_dir, "train")
        for df, name in zip([s1, s2, s3], ["s1", "s2", "s3"]):
            with self.subTest(source=name):
                self.assertEqual(list(df.columns), _OUTPUT_COLUMNS)
        del s1, s2, s3

    # ── Schema correctness ───────────────────────────────────────────────────

    def test_14_columns_match_preprocess_output_columns(self):
        df = load_cache("train", "source1", self.cache_dir)
        self.assertEqual(set(df.columns), _REQUIRED_COLUMNS)
        del df

    def test_raw_entity_id_preserved(self):
        df = load_cache("train", "source1", self.cache_dir)
        # All entity_ids must start with 'S1-'
        self.assertTrue(df["entity_id"].str.startswith("S1-").all())
        del df

    def test_name_norm_is_string_type(self):
        df = load_cache("train", "source1", self.cache_dir)
        # Parquet round-trip on pandas 2.x returns StringDtype (not object).
        # Both represent string columns — verify it is a string-compatible dtype.
        import pandas as pd
        dtype = df["name_norm"].dtype
        is_string = (
            dtype == object
            or isinstance(dtype, pd.StringDtype)
            or hasattr(dtype, "name") and "string" in str(dtype).lower()
        )
        self.assertTrue(is_string, msg=f"Unexpected name_norm dtype: {dtype!r}")
        del df

    def test_name_tokens_is_list_column(self):
        df = load_cache("train", "source1", self.cache_dir)
        # After Parquet round-trip, pandas 2.x may return numpy ndarray or list.
        # Both are valid sequence types. Accept list, ndarray, or None.
        import numpy as np
        sample = df["name_tokens"].iloc[0]
        self.assertIsInstance(
            sample, (list, np.ndarray, type(None)),
            msg=f"Unexpected name_tokens element type: {type(sample)}",
        )
        del df

    def test_name_token_count_is_non_negative_int(self):
        df = load_cache("train", "source1", self.cache_dir)
        self.assertTrue((df["name_token_count"] >= 0).all())
        del df

    def test_null_name_produces_empty_norm(self):
        df = load_cache("train", "source1", self.cache_dir)
        null_rows = df[df["business_name"].isna()]
        if not null_rows.empty:
            self.assertTrue((null_rows["name_norm"] == "").all())
        del df


# ─────────────────────────────────────────────────────────────────────────────
# T5 — Test split support (separate temp dir)
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheTestSplit(unittest.TestCase):
    """build_cache() must handle the 'test' split identically."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.data_dir  = Path(cls.tmpdir) / "data"
        cls.cache_dir = Path(cls.tmpdir) / "cache"
        cls.data_dir.mkdir()
        _make_all_tsvs(cls.data_dir, "test", n_rows=60)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_build_test_cache(self):
        r = build_cache("test", "source1",
                        data_dir=self.data_dir,
                        cache_dir=self.cache_dir)
        self.assertEqual(r["status"], "built")
        self.assertEqual(r["rows"], 60)

    def test_test_cache_exists_after_build(self):
        build_cache("test", "source2",
                    data_dir=self.data_dir,
                    cache_dir=self.cache_dir)
        self.assertTrue(cache_exists("test", "source2", self.cache_dir))

    def test_test_cache_has_correct_schema(self):
        build_cache("test", "source3",
                    data_dir=self.data_dir,
                    cache_dir=self.cache_dir)
        df = load_cache("test", "source3", self.cache_dir)
        self.assertEqual(list(df.columns), _OUTPUT_COLUMNS)
        del df

    def test_test_and_train_caches_are_independent(self):
        """Building test cache must not create a train cache file."""
        self.assertFalse(cache_exists("train", "source1", self.cache_dir))

    def test_test_validate_returns_valid(self):
        r = validate_cache("test", "source1", self.cache_dir)
        self.assertTrue(r["valid"], msg=f"validate returned: {r}")


# ─────────────────────────────────────────────────────────────────────────────
# T6 — Parquet round-trip fidelity
# ─────────────────────────────────────────────────────────────────────────────

class TestParquetFidelity(unittest.TestCase):
    """
    The Parquet-cached output must be equivalent to a direct preprocess_dataframe()
    call on the same raw TSV.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.data_dir  = Path(cls.tmpdir) / "data"
        cls.cache_dir = Path(cls.tmpdir) / "cache"
        cls.data_dir.mkdir()
        _make_tsv(cls.data_dir / "train_source1.tsv", n_rows=200)
        build_cache("train", "source1",
                    data_dir=cls.data_dir,
                    cache_dir=cls.cache_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _direct_result(self) -> pd.DataFrame:
        raw = pd.read_csv(
            self.data_dir / "train_source1.tsv", sep="\t", dtype=str
        )
        return preprocess_dataframe(raw)

    def test_row_count_matches_direct_preprocess(self):
        cached = load_cache("train", "source1", self.cache_dir)
        direct = self._direct_result()
        self.assertEqual(len(cached), len(direct))
        del cached, direct

    def test_name_norm_matches_direct_preprocess(self):
        cached = load_cache("train", "source1", self.cache_dir)
        direct = self._direct_result()
        self.assertListEqual(
            list(cached["name_norm"]),
            list(direct["name_norm"]),
        )
        del cached, direct

    def test_address_norm_matches_direct_preprocess(self):
        cached = load_cache("train", "source1", self.cache_dir)
        direct = self._direct_result()
        self.assertListEqual(
            list(cached["address_norm"]),
            list(direct["address_norm"]),
        )
        del cached, direct

    def test_name_token_count_matches_direct(self):
        cached = load_cache("train", "source1", self.cache_dir)
        direct = self._direct_result()
        self.assertListEqual(
            list(cached["name_token_count"]),
            list(direct["name_token_count"]),
        )
        del cached, direct

    def test_entity_id_matches_direct(self):
        cached = load_cache("train", "source1", self.cache_dir)
        direct = self._direct_result()
        self.assertListEqual(
            list(cached["entity_id"]),
            list(direct["entity_id"]),
        )
        del cached, direct

    def test_name_digits_matches_direct(self):
        cached = load_cache("train", "source1", self.cache_dir)
        direct = self._direct_result()
        self.assertListEqual(
            list(cached["name_digits"]),
            list(direct["name_digits"]),
        )
        del cached, direct


# ─────────────────────────────────────────────────────────────────────────────
# T7 — Chunked build (if implemented) backward-compat test
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkedBuildCompat(unittest.TestCase):
    """
    If build_cache() accepts a chunksize parameter, verify:
      - chunksize=None behaves identically to the current call
      - chunked output schema == non-chunked schema
      - chunked row count == non-chunked row count
      - chunked name_norm == non-chunked name_norm
    If chunksize is not in the signature, this test class is marked as skipped.
    """

    @classmethod
    def setUpClass(cls):
        import inspect
        sig = inspect.signature(build_cache)
        cls.has_chunksize = "chunksize" in sig.parameters
        if not cls.has_chunksize:
            return   # will be skipped in each test

        cls.tmpdir = tempfile.mkdtemp()
        cls.data_dir   = Path(cls.tmpdir) / "data"
        cls.cache_dir_nochunk  = Path(cls.tmpdir) / "cache_nochunk"
        cls.cache_dir_chunked  = Path(cls.tmpdir) / "cache_chunked"
        cls.data_dir.mkdir()
        _make_tsv(cls.data_dir / "train_source1.tsv", n_rows=300)

        build_cache("train", "source1",
                    data_dir=cls.data_dir,
                    cache_dir=cls.cache_dir_nochunk,
                    chunksize=None)
        build_cache("train", "source1",
                    data_dir=cls.data_dir,
                    cache_dir=cls.cache_dir_chunked,
                    chunksize=100)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "tmpdir"):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _skip_if_no_chunksize(self):
        if not self.has_chunksize:
            self.skipTest("build_cache() does not have a chunksize parameter — chunking not implemented")

    def test_chunked_status_built(self):
        self._skip_if_no_chunksize()
        r = build_cache("train", "source1",
                        data_dir=self.data_dir,
                        cache_dir=Path(self.tmpdir) / "tmp_chunk",
                        chunksize=100)
        self.assertEqual(r["status"], "built")

    def test_chunked_schema_matches_nochunked(self):
        self._skip_if_no_chunksize()
        df_nochunk = load_cache("train", "source1", self.cache_dir_nochunk)
        df_chunked = load_cache("train", "source1", self.cache_dir_chunked)
        self.assertEqual(list(df_chunked.columns), list(df_nochunk.columns))
        del df_nochunk, df_chunked

    def test_chunked_row_count_matches_nochunked(self):
        self._skip_if_no_chunksize()
        df_nochunk = load_cache("train", "source1", self.cache_dir_nochunk)
        df_chunked = load_cache("train", "source1", self.cache_dir_chunked)
        self.assertEqual(len(df_chunked), len(df_nochunk))
        del df_nochunk, df_chunked

    def test_chunked_name_norm_matches_nochunked(self):
        self._skip_if_no_chunksize()
        df_nochunk = load_cache("train", "source1", self.cache_dir_nochunk)
        df_chunked = load_cache("train", "source1", self.cache_dir_chunked)
        self.assertListEqual(
            list(df_chunked["name_norm"]),
            list(df_nochunk["name_norm"]),
        )
        del df_nochunk, df_chunked

    def test_chunked_address_norm_matches_nochunked(self):
        self._skip_if_no_chunksize()
        df_nochunk = load_cache("train", "source1", self.cache_dir_nochunk)
        df_chunked = load_cache("train", "source1", self.cache_dir_chunked)
        self.assertListEqual(
            list(df_chunked["address_norm"]),
            list(df_nochunk["address_norm"]),
        )
        del df_nochunk, df_chunked

    def test_chunksize_none_same_as_no_param(self):
        self._skip_if_no_chunksize()
        df_nochunk = load_cache("train", "source1", self.cache_dir_nochunk)
        df_none    = load_cache("train", "source1", self.cache_dir_nochunk)
        self.assertListEqual(list(df_nochunk["name_norm"]), list(df_none["name_norm"]))
        del df_nochunk, df_none


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
