# -*- coding: utf-8 -*-
"""
tests/test_train_test_consistency.py
=====================================
Phase 5 — Train/Test Consistency Audit.

Verifies that the canonical M2 preprocessing functions are used identically
for TRAIN and TEST source datasets — no split-specific normalization exists.

Tests:
  T1  — canonical import (same functions, single source of truth)
  T2  — cache.py split-neutrality (train & test use same code path)
  T3  — train.py normalization usage (preprocess_dataframe only, no reimpl)
  T4  — predict.py normalization usage (same as train.py)
  T5  — features.py reads _norm columns, never calls raw normalization
  T6  — no rogue normalization in non-preprocess modules
  T7  — required normalized fields (same definition for train & test)
  T8  — raw field preservation (entity_id, business_name, business_address, country)
  T9  — country handling: open-set string, no hard-coded list
  T10 — M3/M4 downstream field consumption (name_norm, address_norm, country)
  T11 — train/test output parity (same function → same output for same input)
  T12 — schema consistency (14-column _OUTPUT_COLUMNS identical for both splits)

Run from repo root:
    python -m unittest tests.test_train_test_consistency -v
"""

import ast
import importlib
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Canonical M2 module
import preprocess as _preprocess_mod
from preprocess import (
    normalize_name,
    normalize_address,
    preprocess_dataframe,
    _OUTPUT_COLUMNS,
    _LEGAL_SUFFIX_MAP,
    _ADDRESS_ABBREV_MAP,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_imports(filepath: Path) -> list[str]:
    """Return a list of dotted import names used in *filepath*."""
    source = filepath.read_text(encoding="utf-8", errors="replace")
    tree   = ast.parse(source)
    names  = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    names.append(f"{node.module}.{alias.name}")
    return names


def _source_text(filepath: Path) -> str:
    return filepath.read_text(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# T1 — Canonical import: single source of truth in src/preprocess.py
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalImport(unittest.TestCase):
    """
    The canonical normalization functions are defined ONLY in src/preprocess.py.
    All other modules must import from there — not redefine.
    """

    def test_normalize_name_defined_in_preprocess_only(self):
        """normalize_name must originate from src/preprocess.py."""
        self.assertEqual(
            normalize_name.__module__,
            "preprocess",
            msg="normalize_name is not defined in src/preprocess.py",
        )

    def test_normalize_address_defined_in_preprocess_only(self):
        self.assertEqual(
            normalize_address.__module__,
            "preprocess",
            msg="normalize_address is not defined in src/preprocess.py",
        )

    def test_preprocess_dataframe_defined_in_preprocess_only(self):
        self.assertEqual(
            preprocess_dataframe.__module__,
            "preprocess",
            msg="preprocess_dataframe is not defined in src/preprocess.py",
        )

    def test_output_columns_exported(self):
        """_OUTPUT_COLUMNS must be importable from preprocess and have 14 items."""
        self.assertIsInstance(_OUTPUT_COLUMNS, list)
        self.assertEqual(len(_OUTPUT_COLUMNS), 14)

    def test_legal_suffix_map_exported(self):
        self.assertIsInstance(_LEGAL_SUFFIX_MAP, dict)
        self.assertGreater(len(_LEGAL_SUFFIX_MAP), 0)

    def test_address_abbrev_map_exported(self):
        self.assertIsInstance(_ADDRESS_ABBREV_MAP, dict)
        self.assertGreater(len(_ADDRESS_ABBREV_MAP), 0)


# ─────────────────────────────────────────────────────────────────────────────
# T2 — cache.py split-neutrality
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheSplitNeutral(unittest.TestCase):
    """
    cache.py must use the same preprocess_dataframe() for both train and test.
    It must register both 'train' and 'test' as valid splits.
    It must NOT contain any split-conditional normalization.
    """

    @classmethod
    def setUpClass(cls):
        import cache as _cache_mod
        cls.cache = _cache_mod
        cls.cache_src = _source_text(ROOT / "src" / "cache.py")

    def test_cache_imports_preprocess_dataframe(self):
        src = self.cache_src
        self.assertIn("preprocess_dataframe", src)

    def test_cache_valid_splits_include_train(self):
        self.assertIn("train", self.cache._VALID_SPLITS)

    def test_cache_valid_splits_include_test(self):
        self.assertIn("test", self.cache._VALID_SPLITS)

    def test_cache_source_files_include_train(self):
        self.assertIn("train", self.cache._SOURCE_FILES)
        self.assertEqual(
            set(self.cache._SOURCE_FILES["train"].keys()),
            {"source1", "source2", "source3"},
        )

    def test_cache_source_files_include_test(self):
        self.assertIn("test", self.cache._SOURCE_FILES)
        self.assertEqual(
            set(self.cache._SOURCE_FILES["test"].keys()),
            {"source1", "source2", "source3"},
        )

    def test_cache_source_files_train_basenames(self):
        sf = self.cache._SOURCE_FILES["train"]
        self.assertEqual(sf["source1"], "train_source1.tsv")
        self.assertEqual(sf["source2"], "train_source2.tsv")
        self.assertEqual(sf["source3"], "train_source3.tsv")

    def test_cache_source_files_test_basenames(self):
        sf = self.cache._SOURCE_FILES["test"]
        self.assertEqual(sf["source1"], "test_source1.tsv")
        self.assertEqual(sf["source2"], "test_source2.tsv")
        self.assertEqual(sf["source3"], "test_source3.tsv")

    def test_cache_no_train_only_normalization(self):
        # No code path that calls normalize_name/address only for 'train'
        src = self.cache_src
        self.assertNotIn("normalize_name", src)
        self.assertNotIn("normalize_address", src)

    def test_cache_no_test_only_normalization(self):
        src = self.cache_src
        # cache must not implement its own normalization at all
        self.assertNotIn("unicodedata", src)
        self.assertNotIn("NFKC", src)

    def test_build_cache_delegates_to_preprocess_dataframe(self):
        import inspect
        src = inspect.getsource(self.cache.build_cache)
        self.assertIn("preprocess_dataframe", src)

    def test_validate_cache_checks_same_14_columns_for_both_splits(self):
        import inspect
        src = inspect.getsource(self.cache.validate_cache)
        self.assertIn("_REQUIRED_COLUMNS", src)


# ─────────────────────────────────────────────────────────────────────────────
# T3 — train.py normalization usage
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainNormalizationUsage(unittest.TestCase):
    """
    train.py must import normalize_name, normalize_address, preprocess_dataframe
    from preprocess — not reimplement them.  No inline unicodedata/re.sub
    normalization is permitted.
    """

    @classmethod
    def setUpClass(cls):
        cls.train_src = _source_text(ROOT / "src" / "train.py")

    def test_train_imports_preprocess_dataframe(self):
        self.assertIn("preprocess_dataframe", self.train_src)

    def test_train_imports_normalize_name(self):
        self.assertIn("normalize_name", self.train_src)

    def test_train_imports_normalize_address(self):
        self.assertIn("normalize_address", self.train_src)

    def test_train_imports_from_preprocess_module(self):
        self.assertIn("from preprocess import", self.train_src)

    def test_train_no_unicodedata(self):
        self.assertNotIn("unicodedata", self.train_src)

    def test_train_no_nfkc(self):
        self.assertNotIn("NFKC", self.train_src)

    def test_train_uses_m2_norm_columns(self):
        # train.py joins on name_norm / address_norm from preprocess output
        self.assertIn("name_norm", self.train_src)
        self.assertIn("address_norm", self.train_src)

    def test_train_load_sources_uses_preprocess_dataframe(self):
        import inspect
        import train as _train_mod
        src = inspect.getsource(_train_mod.load_sources)
        self.assertIn("preprocess_dataframe", src)


# ─────────────────────────────────────────────────────────────────────────────
# T4 — predict.py normalization usage
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictNormalizationUsage(unittest.TestCase):
    """
    predict.py must use the same preprocess_dataframe() as train.py.
    No test-specific normalization code is permitted.
    """

    @classmethod
    def setUpClass(cls):
        cls.predict_src = _source_text(ROOT / "src" / "predict.py")

    def test_predict_imports_preprocess_dataframe(self):
        self.assertIn("preprocess_dataframe", self.predict_src)

    def test_predict_imports_normalize_name(self):
        self.assertIn("normalize_name", self.predict_src)

    def test_predict_imports_normalize_address(self):
        self.assertIn("normalize_address", self.predict_src)

    def test_predict_imports_from_preprocess_module(self):
        self.assertIn("from preprocess import", self.predict_src)

    def test_predict_no_unicodedata(self):
        self.assertNotIn("unicodedata", self.predict_src)

    def test_predict_no_nfkc(self):
        self.assertNotIn("NFKC", self.predict_src)

    def test_predict_uses_m2_norm_columns(self):
        self.assertIn("name_norm", self.predict_src)
        self.assertIn("address_norm", self.predict_src)

    def test_predict_load_sources_accepts_split_parameter(self):
        import inspect
        import predict as _predict_mod
        sig = inspect.signature(_predict_mod.load_sources)
        self.assertIn("split", sig.parameters)

    def test_predict_load_sources_uses_preprocess_dataframe(self):
        import inspect
        import predict as _predict_mod
        src = inspect.getsource(_predict_mod.load_sources)
        self.assertIn("preprocess_dataframe", src)

    def test_predict_load_sources_parametric_split(self):
        """predict.load_sources must use the split parameter to form filenames,
        not hardcode 'test_' or 'train_'."""
        import inspect
        import predict as _predict_mod
        src = inspect.getsource(_predict_mod.load_sources)
        # The filename is formed dynamically with the split variable
        self.assertIn("{split}", src)


# ─────────────────────────────────────────────────────────────────────────────
# T5 — features.py reads _norm columns, never calls raw normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestFeaturesPipelineUsage(unittest.TestCase):
    """
    features.py (M3) must consume pre-computed _norm columns from M2.
    It must not import or call normalize_name/normalize_address/unicodedata.
    """

    @classmethod
    def setUpClass(cls):
        cls.feat_src = _source_text(ROOT / "src" / "features.py")

    def test_features_no_normalize_name(self):
        # features.py docstrings legitimately mention 'normalize_name' by name
        # (describing delegation). The check is that it is never imported or called.
        import re
        # No 'import normalize_name' or 'normalize_name(' call
        self.assertIsNone(
            re.search(r'(?:^from\s+preprocess|^import\s+preprocess|normalize_name\s*\()',
                      self.feat_src, re.MULTILINE),
            msg="features.py imports or calls normalize_name()",
        )

    def test_features_no_normalize_address(self):
        # Same: docstrings may reference the name. Check for actual import/call.
        import re
        self.assertIsNone(
            re.search(r'(?:^from\s+preprocess|^import\s+preprocess|normalize_address\s*\()',
                      self.feat_src, re.MULTILINE),
            msg="features.py imports or calls normalize_address()",
        )

    def test_features_no_preprocess_import(self):
        self.assertNotIn("from preprocess", self.feat_src)
        self.assertNotIn("import preprocess", self.feat_src)

    def test_features_no_unicodedata(self):
        self.assertNotIn("unicodedata", self.feat_src)

    def test_features_no_nfkc(self):
        self.assertNotIn("NFKC", self.feat_src)

    def test_features_reads_name_norm_column(self):
        self.assertIn("name_norm", self.feat_src)

    def test_features_reads_address_norm_column(self):
        self.assertIn("address_norm", self.feat_src)

    def test_features_reads_country_column(self):
        self.assertIn("country", self.feat_src)

    def test_features_no_inline_regex_normalization(self):
        import re
        # Detect any re.sub calls with \w or [^\w that would imply normalization
        matches = re.findall(r're\.sub\s*\(.*?\\\\w', self.feat_src)
        self.assertEqual(
            matches, [],
            msg=f"features.py contains inline regex normalization: {matches}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# T6 — No rogue normalization in non-preprocess modules
# ─────────────────────────────────────────────────────────────────────────────

class TestNoRogueNormalization(unittest.TestCase):
    """
    Scan all src/*.py files (except preprocess.py itself and the test file)
    for signs of reimplemented normalization logic.
    """

    ROGUE_PATTERNS = [
        "unicodedata.normalize",
        "unicodedata.normalize(\"NFKC\"",
        "unicodedata.normalize('NFKC'",
        ".lower().strip()",   # inline ad-hoc lowercasing pipeline
    ]

    NON_PREPROCESS_FILES = [
        ROOT / "src" / "cache.py",
        ROOT / "src" / "train.py",
        ROOT / "src" / "predict.py",
        ROOT / "src" / "features.py",
        ROOT / "src" / "candidates.py",
    ]

    def test_no_unicodedata_normalize_in_cache(self):
        src = _source_text(ROOT / "src" / "cache.py")
        self.assertNotIn("unicodedata.normalize", src)

    def test_no_unicodedata_normalize_in_train(self):
        src = _source_text(ROOT / "src" / "train.py")
        self.assertNotIn("unicodedata.normalize", src)

    def test_no_unicodedata_normalize_in_predict(self):
        src = _source_text(ROOT / "src" / "predict.py")
        self.assertNotIn("unicodedata.normalize", src)

    def test_no_unicodedata_normalize_in_features(self):
        src = _source_text(ROOT / "src" / "features.py")
        self.assertNotIn("unicodedata.normalize", src)

    def test_no_unicodedata_normalize_in_candidates(self):
        src = _source_text(ROOT / "src" / "candidates.py")
        self.assertNotIn("unicodedata.normalize", src)

    def test_no_train_specific_normalize_function(self):
        """No function named normalize_* should exist in train.py."""
        import re
        src = _source_text(ROOT / "src" / "train.py")
        matches = re.findall(r"def\s+normalize_\w+\s*\(", src)
        self.assertEqual(matches, [], msg=f"train.py defines: {matches}")

    def test_no_test_specific_normalize_function(self):
        """No function named normalize_* should exist in predict.py."""
        import re
        src = _source_text(ROOT / "src" / "predict.py")
        matches = re.findall(r"def\s+normalize_\w+\s*\(", src)
        self.assertEqual(matches, [], msg=f"predict.py defines: {matches}")


# ─────────────────────────────────────────────────────────────────────────────
# T7 — Required normalized fields: same definition for train and test
# ─────────────────────────────────────────────────────────────────────────────

class TestRequiredNormalizedFields(unittest.TestCase):
    """
    preprocess_dataframe() always returns the same 14 columns in _OUTPUT_COLUMNS
    regardless of whether the input data came from train or test TSVs.
    """

    REQUIRED_NORM_FIELDS = [
        "name_norm",
        "address_norm",
        "name_tokens",
        "address_tokens",
        "name_token_count",
        "address_token_count",
        "name_length",
        "address_length",
        "name_digits",
        "address_digits",
    ]

    REQUIRED_ALL = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ] + REQUIRED_NORM_FIELDS

    def _make_df(self, prefix: str) -> pd.DataFrame:
        """Build a minimal 3-row DataFrame mimicking a source TSV."""
        return pd.DataFrame([
            {
                "entity_id":         f"{prefix}-00001",
                "business_name":     "Acme Corp",
                "business_address":  "123 Main Ave",
                "country":           "US",
            },
            {
                "entity_id":         f"{prefix}-00002",
                "business_name":     None,
                "business_address":  None,
                "country":           "IN",
            },
            {
                "entity_id":         f"{prefix}-00003",
                "business_name":     "Global Ltd",
                "business_address":  "456 Oak Blvd",
                "country":           "GB",
            },
        ])

    def test_train_df_has_all_14_columns(self):
        df = preprocess_dataframe(self._make_df("S1"))
        self.assertEqual(list(df.columns), _OUTPUT_COLUMNS)

    def test_test_df_has_all_14_columns(self):
        df = preprocess_dataframe(self._make_df("T1"))
        self.assertEqual(list(df.columns), _OUTPUT_COLUMNS)

    def test_train_and_test_output_same_columns(self):
        train_df = preprocess_dataframe(self._make_df("S1"))
        test_df  = preprocess_dataframe(self._make_df("T1"))
        self.assertEqual(list(train_df.columns), list(test_df.columns))

    def test_all_required_norm_fields_in_output(self):
        df = preprocess_dataframe(self._make_df("S2"))
        for field in self.REQUIRED_NORM_FIELDS:
            self.assertIn(field, df.columns, msg=f"Missing field: {field!r}")

    def test_all_14_fields_in_output(self):
        df = preprocess_dataframe(self._make_df("S3"))
        for field in self.REQUIRED_ALL:
            self.assertIn(field, df.columns, msg=f"Missing field: {field!r}")

    def test_name_norm_definition_same_for_train_and_test(self):
        """The same input must produce the same name_norm regardless of split label."""
        train_df = preprocess_dataframe(self._make_df("S1"))
        test_df  = preprocess_dataframe(self._make_df("T1"))
        # name_norm column values should be identical (same input, same function)
        self.assertListEqual(
            list(train_df["name_norm"]),
            list(test_df["name_norm"]),
        )

    def test_address_norm_definition_same_for_train_and_test(self):
        train_df = preprocess_dataframe(self._make_df("S1"))
        test_df  = preprocess_dataframe(self._make_df("T1"))
        self.assertListEqual(
            list(train_df["address_norm"]),
            list(test_df["address_norm"]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# T8 — Raw field preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestRawFieldPreservation(unittest.TestCase):
    """
    preprocess_dataframe() must preserve the raw fields
    entity_id, business_name, business_address, country unchanged.
    """

    RAW_COLS = ["entity_id", "business_name", "business_address", "country"]

    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Acme, Inc.",
             "business_address": "123 Main Ave", "country": "United States"},
            {"entity_id": "S2-001", "business_name": "CALDEON NOVA",
             "business_address": "456 Oak Blvd.", "country": "India"},
        ])

    def test_entity_id_preserved(self):
        raw = self._sample_df()
        out = preprocess_dataframe(raw)
        self.assertListEqual(list(out["entity_id"]), list(raw["entity_id"]))

    def test_business_name_not_overwritten(self):
        raw = self._sample_df()
        out = preprocess_dataframe(raw)
        self.assertListEqual(
            list(out["business_name"]),
            list(raw["business_name"]),
        )

    def test_business_address_not_overwritten(self):
        raw = self._sample_df()
        out = preprocess_dataframe(raw)
        self.assertListEqual(
            list(out["business_address"]),
            list(raw["business_address"]),
        )

    def test_country_not_overwritten(self):
        raw = self._sample_df()
        out = preprocess_dataframe(raw)
        self.assertListEqual(list(out["country"]), list(raw["country"]))

    def test_name_norm_differs_from_business_name(self):
        """Proof that a separate derived column exists, not an in-place override."""
        raw = self._sample_df()
        out = preprocess_dataframe(raw)
        # "Acme, Inc." must NOT equal its normalized form "acme incorporated"
        self.assertNotEqual(out.iloc[0]["business_name"], out.iloc[0]["name_norm"])

    def test_all_raw_cols_in_output(self):
        out = preprocess_dataframe(self._sample_df())
        for col in self.RAW_COLS:
            self.assertIn(col, out.columns)


# ─────────────────────────────────────────────────────────────────────────────
# T9 — Country handling: open-set string, no hard-coded list
# ─────────────────────────────────────────────────────────────────────────────

class TestCountryHandling(unittest.TestCase):
    """
    Country must be treated as a free-form string with no fixed country list.
    normalize_name / normalize_address must not be applied to country values.
    preprocess_dataframe() must pass country through unchanged.
    """

    UNEXPECTED_COUNTRY_LISTS = [
        "US", "IN", "GB", "AU", "DE", "CN", "JP", "FR", "CA", "BR",
    ]

    def test_preprocess_passes_country_through(self):
        df = pd.DataFrame([
            {"entity_id": "X1", "business_name": "A", "business_address": "B",
             "country": "Fantasia"},
            {"entity_id": "X2", "business_name": "C", "business_address": "D",
             "country": ""},
            {"entity_id": "X3", "business_name": "E", "business_address": "F",
             "country": "Unknown Territory"},
        ])
        out = preprocess_dataframe(df)
        self.assertEqual(out.iloc[0]["country"], "Fantasia")
        self.assertEqual(out.iloc[1]["country"], "")
        self.assertEqual(out.iloc[2]["country"], "Unknown Territory")

    def test_preprocess_py_has_no_hardcoded_country_list(self):
        """preprocess.py must not enumerate specific country codes/names."""
        src = _source_text(ROOT / "src" / "preprocess.py")
        # Check that no country-specific string is used in conditional logic
        import re
        country_conditionals = re.findall(
            r'(?:if|elif|==|in)\s+["\'](?:US|IN|GB|DE|CN|JP|FR|CA|BR)["\']',
            src
        )
        self.assertEqual(
            country_conditionals, [],
            msg=f"preprocess.py contains hardcoded country checks: {country_conditionals}",
        )

    def test_cache_py_has_no_hardcoded_country_list(self):
        src = _source_text(ROOT / "src" / "cache.py")
        import re
        matches = re.findall(
            r'(?:if|elif|==|in)\s+["\'](?:US|IN|GB|DE|CN|JP|FR|CA|BR)["\']',
            src
        )
        self.assertEqual(matches, [], msg=f"cache.py country checks: {matches}")

    def test_country_docstring_says_open_set(self):
        """The preprocess.py docstring must state that country is open-set."""
        src = _source_text(ROOT / "src" / "preprocess.py")
        self.assertIn("open-set", src.lower(),
                      msg="preprocess.py does not document that country is open-set")

    def test_country_not_normalized_by_name_pipeline(self):
        """country='US' must remain 'US' after preprocess_dataframe — not lowercased."""
        df = pd.DataFrame([{
            "entity_id": "X1", "business_name": "Acme",
            "business_address": "1 Main St", "country": "US"
        }])
        out = preprocess_dataframe(df)
        self.assertEqual(out.iloc[0]["country"], "US")


# ─────────────────────────────────────────────────────────────────────────────
# T10 — M3/M4 downstream field consumption
# ─────────────────────────────────────────────────────────────────────────────

class TestDownstreamFieldConsumption(unittest.TestCase):
    """
    M3 (features.py, train.py, predict.py) and M4 (candidates.py)
    must consume the M2 _norm fields — not raw fields or reimplemented ones.
    Expected fields: entity_id, name_norm, address_norm, country, name_tokens, address_tokens.
    """

    def test_features_reads_s1_name_norm(self):
        src = _source_text(ROOT / "src" / "features.py")
        self.assertIn("s1_name_norm", src)

    def test_features_reads_cand_name_norm(self):
        src = _source_text(ROOT / "src" / "features.py")
        self.assertIn("cand_name_norm", src)

    def test_features_reads_s1_address_norm(self):
        src = _source_text(ROOT / "src" / "features.py")
        self.assertIn("s1_address_norm", src)

    def test_features_reads_cand_address_norm(self):
        src = _source_text(ROOT / "src" / "features.py")
        self.assertIn("cand_address_norm", src)

    def test_features_reads_country(self):
        src = _source_text(ROOT / "src" / "features.py")
        self.assertIn("country", src)

    def test_train_joins_on_name_norm(self):
        src = _source_text(ROOT / "src" / "train.py")
        self.assertIn("name_norm", src)

    def test_train_joins_on_address_norm(self):
        src = _source_text(ROOT / "src" / "train.py")
        self.assertIn("address_norm", src)

    def test_train_joins_on_country(self):
        src = _source_text(ROOT / "src" / "train.py")
        self.assertIn("country", src)

    def test_predict_joins_on_name_norm(self):
        src = _source_text(ROOT / "src" / "predict.py")
        self.assertIn("name_norm", src)

    def test_predict_joins_on_address_norm(self):
        src = _source_text(ROOT / "src" / "predict.py")
        self.assertIn("address_norm", src)

    def test_train_does_not_read_raw_business_name_for_matching(self):
        """train.py joins on name_norm not business_name for candidate scoring."""
        import train as _train_mod
        import inspect
        src = inspect.getsource(_train_mod.build_pair_rows)
        # Must use _norm columns in lookups, not raw business_name
        self.assertIn("name_norm", src)
        self.assertNotIn('"business_name"', src.replace("'", '"'))

    def test_predict_does_not_read_raw_business_name_for_matching(self):
        import predict as _predict_mod
        import inspect
        src = inspect.getsource(_predict_mod.build_pair_rows)
        self.assertIn("name_norm", src)
        self.assertNotIn('"business_name"', src.replace("'", '"'))


# ─────────────────────────────────────────────────────────────────────────────
# T11 — Train/test output parity (same function → same output for same input)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainTestOutputParity(unittest.TestCase):
    """
    The same raw input row must produce identical normalized output
    regardless of whether it is labelled as a 'train' or 'test' entity.
    """

    REPRESENTATIVE_PAIRS = [
        # (business_name, business_address, country)
        ("Acme Corp",          "123 Main Ave",         "US"),
        ("CALDEON NOVA",       "8706 KENTUCKY DERBY DR","US"),
        ("Acme Pvt. Ltd.",     "456 Oak Blvd, Suite 10","IN"),
        (None,                 None,                    "GB"),
        ("7-Eleven Stores Inc","P.O. Box 1234",         "US"),
        ("ABC 中华 Corp",       "北京 100020",            "CN"),
        ("Müller GmbH",        "Café Str. 5",           "DE"),
    ]

    def _make_row(self, prefix: str, name, addr, country) -> pd.DataFrame:
        return pd.DataFrame([{
            "entity_id":        f"{prefix}-00001",
            "business_name":    name,
            "business_address": addr,
            "country":          country,
        }])

    def _normalize_scalar(self, name, addr):
        return normalize_name(name), normalize_address(addr)

    def test_name_norm_identical_for_train_and_test_rows(self):
        for name, addr, country in self.REPRESENTATIVE_PAIRS:
            with self.subTest(name=name, addr=addr):
                train_out = preprocess_dataframe(self._make_row("S1", name, addr, country))
                test_out  = preprocess_dataframe(self._make_row("T1", name, addr, country))
                self.assertEqual(
                    train_out.iloc[0]["name_norm"],
                    test_out.iloc[0]["name_norm"],
                    msg=f"name_norm differs for input {name!r}",
                )

    def test_address_norm_identical_for_train_and_test_rows(self):
        for name, addr, country in self.REPRESENTATIVE_PAIRS:
            with self.subTest(name=name, addr=addr):
                train_out = preprocess_dataframe(self._make_row("S1", name, addr, country))
                test_out  = preprocess_dataframe(self._make_row("T1", name, addr, country))
                self.assertEqual(
                    train_out.iloc[0]["address_norm"],
                    test_out.iloc[0]["address_norm"],
                    msg=f"address_norm differs for input {addr!r}",
                )

    def test_all_14_columns_identical_for_train_and_test(self):
        for name, addr, country in self.REPRESENTATIVE_PAIRS:
            with self.subTest(name=name):
                train_out = preprocess_dataframe(self._make_row("S1", name, addr, country))
                test_out  = preprocess_dataframe(self._make_row("T1", name, addr, country))
                for col in _OUTPUT_COLUMNS:
                    if col == "entity_id":
                        continue  # entity_id differs by design (S1 vs T1)
                    t_val = train_out.iloc[0][col]
                    p_val = test_out.iloc[0][col]
                    self.assertEqual(
                        t_val, p_val,
                        msg=f"Column {col!r} differs for name={name!r}: "
                            f"train={t_val!r}, test={p_val!r}",
                    )

    def test_normalize_name_same_function_in_train_and_test(self):
        """Verify the function object imported by train.py is the M2 canonical one."""
        import train as _train_mod
        # train.py imports normalize_name from preprocess
        self.assertIs(
            getattr(_train_mod, "normalize_name", None),
            normalize_name,
            msg="train.py's normalize_name is not the same object as M2 canonical",
        )

    def test_normalize_address_same_function_in_train_and_test(self):
        import train as _train_mod
        self.assertIs(
            getattr(_train_mod, "normalize_address", None),
            normalize_address,
            msg="train.py's normalize_address is not the M2 canonical function",
        )

    def test_normalize_name_same_function_in_predict(self):
        import predict as _predict_mod
        self.assertIs(
            getattr(_predict_mod, "normalize_name", None),
            normalize_name,
            msg="predict.py's normalize_name is not the M2 canonical function",
        )

    def test_normalize_address_same_function_in_predict(self):
        import predict as _predict_mod
        self.assertIs(
            getattr(_predict_mod, "normalize_address", None),
            normalize_address,
            msg="predict.py's normalize_address is not the M2 canonical function",
        )


# ─────────────────────────────────────────────────────────────────────────────
# T12 — Schema consistency: _OUTPUT_COLUMNS is the single authoritative list
# ─────────────────────────────────────────────────────────────────────────────

class TestSchemaConsistency(unittest.TestCase):
    """
    _OUTPUT_COLUMNS in preprocess.py is the single schema authority.
    cache.py's _REQUIRED_COLUMNS must be derived from it.
    The 14-column list must include the required raw + norm fields in order.
    """

    EXPECTED_SCHEMA = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "name_norm",
        "address_norm",
        "name_tokens",
        "address_tokens",
        "name_token_count",
        "address_token_count",
        "name_length",
        "address_length",
        "name_digits",
        "address_digits",
    ]

    def test_output_columns_matches_expected_schema(self):
        self.assertEqual(_OUTPUT_COLUMNS, self.EXPECTED_SCHEMA)

    def test_cache_required_columns_derived_from_output_columns(self):
        import cache as _cache_mod
        self.assertEqual(
            _cache_mod._REQUIRED_COLUMNS,
            set(_OUTPUT_COLUMNS),
        )

    def test_schema_has_exactly_14_columns(self):
        self.assertEqual(len(_OUTPUT_COLUMNS), 14)

    def test_schema_has_4_raw_fields(self):
        raw = [c for c in _OUTPUT_COLUMNS
               if c in ("entity_id", "business_name", "business_address", "country")]
        self.assertEqual(len(raw), 4)

    def test_schema_has_norm_fields(self):
        self.assertIn("name_norm", _OUTPUT_COLUMNS)
        self.assertIn("address_norm", _OUTPUT_COLUMNS)

    def test_schema_has_token_fields(self):
        self.assertIn("name_tokens", _OUTPUT_COLUMNS)
        self.assertIn("address_tokens", _OUTPUT_COLUMNS)

    def test_schema_has_count_fields(self):
        self.assertIn("name_token_count", _OUTPUT_COLUMNS)
        self.assertIn("address_token_count", _OUTPUT_COLUMNS)

    def test_schema_has_length_fields(self):
        self.assertIn("name_length", _OUTPUT_COLUMNS)
        self.assertIn("address_length", _OUTPUT_COLUMNS)

    def test_schema_has_digit_fields(self):
        self.assertIn("name_digits", _OUTPUT_COLUMNS)
        self.assertIn("address_digits", _OUTPUT_COLUMNS)

    def test_no_extra_columns_in_preprocess_output(self):
        """preprocess_dataframe() must return exactly 14 columns, no extras."""
        df = pd.DataFrame([{
            "entity_id": "X1", "business_name": "A",
            "business_address": "B", "country": "US"
        }])
        out = preprocess_dataframe(df)
        self.assertEqual(len(out.columns), 14)
        self.assertEqual(list(out.columns), _OUTPUT_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
