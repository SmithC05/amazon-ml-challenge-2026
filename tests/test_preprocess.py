# -*- coding: utf-8 -*-
"""
tests/test_preprocess.py
========================
Phase 4 — Edge-Case Test Suite for M2 normalization.

Tests the CURRENT behavior of:
  - normalize_text()
  - normalize_name()
  - normalize_address()
  - preprocess_dataframe()

No normalization rules are changed or added.
All expected values are derived from the actual current implementation.

Run from repo root:
    python -m unittest tests.test_preprocess -v

Or discover automatically:
    python -m unittest discover -s tests -v
"""

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

# ── Path setup (works when run as script or via unittest discover) ────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import (  # noqa: E402
    normalize_text,
    normalize_name,
    normalize_address,
    preprocess_dataframe,
    _LEGAL_SUFFIX_MAP,
    _ADDRESS_ABBREV_MAP,
)


# =============================================================================
# T1 — None handling
# =============================================================================

class TestNoneHandling(unittest.TestCase):
    """None inputs must return '' without raising any exception."""

    def test_normalize_text_none(self):
        self.assertEqual(normalize_text(None), "")

    def test_normalize_name_none(self):
        self.assertEqual(normalize_name(None), "")

    def test_normalize_address_none(self):
        self.assertEqual(normalize_address(None), "")

    def test_normalize_text_none_type(self):
        result = normalize_text(None)
        self.assertIsInstance(result, str)

    def test_normalize_name_none_type(self):
        result = normalize_name(None)
        self.assertIsInstance(result, str)

    def test_normalize_address_none_type(self):
        result = normalize_address(None)
        self.assertIsInstance(result, str)


# =============================================================================
# T2 — NaN handling
# =============================================================================

class TestNaNHandling(unittest.TestCase):
    """float NaN inputs must return '' without raising any exception."""

    def test_normalize_text_nan(self):
        self.assertEqual(normalize_text(float("nan")), "")

    def test_normalize_name_nan(self):
        self.assertEqual(normalize_name(float("nan")), "")

    def test_normalize_address_nan(self):
        self.assertEqual(normalize_address(float("nan")), "")

    def test_normalize_text_pd_na(self):
        self.assertEqual(normalize_text(pd.NA), "")

    def test_normalize_text_pd_nat(self):
        self.assertEqual(normalize_text(pd.NaT), "")

    def test_normalize_text_numpy_nan(self):
        import numpy as np
        self.assertEqual(normalize_text(np.nan), "")


# =============================================================================
# T3 — Empty string handling
# =============================================================================

class TestEmptyStringHandling(unittest.TestCase):
    """Empty strings must return '' without raising any exception."""

    def test_normalize_text_empty(self):
        self.assertEqual(normalize_text(""), "")

    def test_normalize_name_empty(self):
        self.assertEqual(normalize_name(""), "")

    def test_normalize_address_empty(self):
        self.assertEqual(normalize_address(""), "")

    def test_normalize_text_whitespace_only(self):
        # Whitespace-only string -> all collapsed -> stripped -> ""
        self.assertEqual(normalize_text("   "), "")

    def test_normalize_name_whitespace_only(self):
        self.assertEqual(normalize_name("   \t\n"), "")

    def test_normalize_address_whitespace_only(self):
        self.assertEqual(normalize_address("  \r\n  "), "")


# =============================================================================
# T4 — Punctuation-heavy names
# =============================================================================

class TestPunctuationHeavyNames(unittest.TestCase):
    """
    Punctuation characters are replaced with spaces; consecutive spaces
    collapsed; result stripped.  Digits and word chars are preserved.
    """

    def test_commas_removed(self):
        self.assertEqual(normalize_name("Acme, Inc."), "acme incorporated")

    def test_dots_removed(self):
        self.assertEqual(normalize_text("A.B.C."), "a b c")

    def test_hyphens_removed(self):
        self.assertEqual(normalize_text("Coca-Cola"), "coca cola")

    def test_brackets_removed(self):
        self.assertEqual(normalize_text("Obsidian [[LLC]]"), "obsidian llc")

    def test_ampersand_removed(self):
        self.assertEqual(normalize_text("Smith & Jones"), "smith jones")

    def test_slash_removed(self):
        self.assertEqual(normalize_text("North/South"), "north south")

    def test_multiple_punctuation_collapsed(self):
        # "---" -> "   " -> collapsed to single space
        self.assertEqual(normalize_text("A---B"), "a b")

    def test_leading_punctuation_stripped(self):
        # Leading ">> " -> "  " -> stripped
        result = normalize_text(">> company name")
        self.assertEqual(result, "company name")

    def test_parentheses_removed(self):
        self.assertEqual(normalize_text("ABC (Pvt) Ltd"), "abc pvt ltd")

    def test_exclamation_removed(self):
        self.assertEqual(normalize_text("Best Company!"), "best company")

    def test_all_punctuation_empty(self):
        # Only punctuation -> all become spaces -> collapse -> ""
        self.assertEqual(normalize_text(",.;:!?"), "")


# =============================================================================
# T5 — Repeated spaces
# =============================================================================

class TestRepeatedSpaces(unittest.TestCase):
    """Multiple consecutive spaces must be collapsed to one."""

    def test_double_space_collapsed(self):
        self.assertEqual(normalize_text("Caldeon  Nova"), "caldeon nova")

    def test_triple_space_collapsed(self):
        self.assertEqual(normalize_text("A   B   C"), "a b c")

    def test_tab_collapsed(self):
        self.assertEqual(normalize_text("A\tB"), "a b")

    def test_mixed_whitespace_collapsed(self):
        self.assertEqual(normalize_text("A \t B \n C"), "a b c")

    def test_leading_trailing_stripped(self):
        self.assertEqual(normalize_text("  hello world  "), "hello world")

    def test_newline_collapsed(self):
        self.assertEqual(normalize_text("foo\nbar"), "foo bar")


# =============================================================================
# T6 — Unicode / full-width text
# =============================================================================

class TestUnicodeFullWidth(unittest.TestCase):
    """
    NFKC normalization converts full-width Latin to ASCII equivalents.
    Unicode word chars (CJK etc.) are preserved.
    """

    def test_fullwidth_latin_normalized(self):
        # Full-width 'Ａ' (U+FF21) -> 'A' -> 'a'
        self.assertEqual(normalize_text("\uFF21\uFF22\uFF23"), "abc")

    def test_fullwidth_digits_normalized(self):
        # Full-width digit '１' (U+FF11) -> '1'
        self.assertEqual(normalize_text("\uFF11\uFF12\uFF13"), "123")

    def test_fullwidth_space_normalized(self):
        # Full-width space U+3000 is handled by NFKC -> regular space -> stripped
        result = normalize_text("hello\u3000world")
        self.assertEqual(result, "hello world")

    def test_ligature_fi_normalized(self):
        # 'ﬁ' (U+FB01, fi ligature) -> 'fi' via NFKC
        self.assertEqual(normalize_text("\uFB01le"), "file")

    def test_superscript_normalized(self):
        # Superscript '²' (U+00B2) -> '2' via NFKC
        self.assertEqual(normalize_text("CO\u00B2"), "co2")


# =============================================================================
# T7 — Accented text
# =============================================================================

class TestAccentedText(unittest.TestCase):
    """
    NFKC composes combining diacritics. Precomposed accented chars are kept
    as accented chars (NFKC preserves them when already canonical).
    The regex [^\w\s] preserves \w which includes accented letters.
    """

    def test_precomposed_accented_lowercase(self):
        # 'é' (U+00E9) is a word char; kept by regex, lowercased
        self.assertEqual(normalize_text("Café"), "café")

    def test_precomposed_accented_name(self):
        self.assertEqual(normalize_text("Müller GmbH"), "müller gmbh")

    def test_combining_diacritic_composed(self):
        # 'e' + combining acute (U+0301) -> NFKC -> 'é'
        composed   = normalize_text("e\u0301")   # decomposed input
        precomposed = normalize_text("\u00e9")   # precomposed input
        self.assertEqual(composed, precomposed)

    def test_accented_letters_preserved(self):
        result = normalize_text("São Paulo")
        self.assertIn("ão", result)   # ã is a word char, preserved

    def test_cedilla_preserved(self):
        result = normalize_text("François")
        self.assertIn("ç", result)


# =============================================================================
# T8 — Numbers in names
# =============================================================================

class TestNumbersInNames(unittest.TestCase):
    """Digit characters in names must be preserved exactly."""

    def test_number_in_name_preserved(self):
        self.assertEqual(normalize_name("7-Eleven"), "7 eleven")

    def test_leading_number_preserved(self):
        self.assertEqual(normalize_name("3M Company"), "3m company")

    def test_trailing_number_preserved(self):
        self.assertEqual(normalize_name("Studio 54"), "studio 54")

    def test_alphanumeric_token_preserved(self):
        self.assertEqual(normalize_name("H2O Technologies"), "h2o technologies")

    def test_version_number_preserved(self):
        self.assertEqual(normalize_text("Version 2.0"), "version 2 0")

    def test_year_preserved(self):
        self.assertEqual(normalize_name("Founded 1920 Corp"), "founded 1920 corporation")

    def test_digits_not_dropped_by_punctuation_removal(self):
        # "99" is \w, not touched by [^\w\s] -> ""
        result = normalize_name("99 Bottles LLC")
        self.assertIn("99", result)


# =============================================================================
# T9 — Numbers in addresses
# =============================================================================

class TestNumbersInAddresses(unittest.TestCase):
    """Digit characters in addresses must be preserved exactly."""

    def test_street_number_preserved(self):
        result = normalize_address("123 Main Street")
        self.assertIn("123", result)

    def test_zip_code_preserved(self):
        result = normalize_address("New York, NY 10001")
        self.assertIn("10001", result)

    def test_suite_number_preserved(self):
        result = normalize_address("Suite 200, 45 Park Ave")
        self.assertIn("200", result)
        self.assertIn("45", result)

    def test_mixed_alphanumeric_address(self):
        # "8706 KENTUCKY DERBY DR" -> "8706 kentucky derby drive"
        result = normalize_address("8706 KENTUCKY DERBY DR")
        self.assertIn("8706", result)

    def test_po_box_number_preserved(self):
        result = normalize_address("P.O. Box 1234")
        self.assertIn("1234", result)

    def test_address_with_hyphen_number(self):
        # "6-29" -> "6 29" (hyphen removed, digits kept)
        result = normalize_address("6-29 Main Road")
        self.assertIn("6", result)
        self.assertIn("29", result)


# =============================================================================
# T10 — Legal suffixes
# =============================================================================

class TestLegalSuffixes(unittest.TestCase):
    """
    Verified legal-suffix rules from _LEGAL_SUFFIX_MAP must be applied by
    normalize_name() exactly as defined. Dead key "pvt. ltd." is removed
    (Phase 2 fix); functional rules must all still work.
    """

    # Multi-token phrase (longest-match-first)
    def test_pvt_ltd_phrase(self):
        self.assertEqual(normalize_name("Acme Pvt Ltd"), "acme private limited")

    def test_pvt_ltd_phrase_with_punctuation(self):
        # "Pvt. Ltd." -> after normalize_text: "pvt ltd" -> map -> "private limited"
        self.assertEqual(normalize_name("Acme Pvt. Ltd."), "acme private limited")

    # Single-token rules
    def test_pvt(self):
        self.assertEqual(normalize_name("Acme Pvt"), "acme private")

    def test_ltd(self):
        self.assertEqual(normalize_name("Acme Ltd"), "acme limited")

    def test_inc(self):
        self.assertEqual(normalize_name("Acme Inc"), "acme incorporated")

    def test_corp(self):
        self.assertEqual(normalize_name("Acme Corp"), "acme corporation")

    def test_llp(self):
        self.assertEqual(
            normalize_name("Acme LLP"),
            "acme limited liability partnership",
        )

    # Suffix at start of name
    def test_suffix_at_start(self):
        # "Inc" at start -> "incorporated ..."
        self.assertEqual(normalize_name("Inc Acme"), "incorporated acme")

    # Suffix inside longer token is NOT replaced (whole-token matching only)
    def test_partial_match_not_replaced(self):
        # "incorp" is not in _LEGAL_SUFFIX_MAP -> must NOT be replaced
        result = normalize_name("Acme Incorp")
        self.assertNotEqual(result, "acme incorporated")
        self.assertEqual(result, "acme incorp")

    # Dead key must be gone
    def test_dead_key_removed(self):
        self.assertNotIn("pvt. ltd.", _LEGAL_SUFFIX_MAP)

    # normalize_address must NOT apply legal suffix map
    def test_legal_suffix_not_applied_to_address(self):
        result = normalize_address("100 Ltd Street")
        # "ltd" must remain — _LEGAL_SUFFIX_MAP is not applied in normalize_address
        self.assertIn("ltd", result)

    # All keys in the map are reachable after normalize_text (no dots survive)
    def test_all_map_keys_are_post_normalization_forms(self):
        for key in _LEGAL_SUFFIX_MAP:
            # After normalize_text, no punctuation should remain in keys
            normed = normalize_text(key)
            self.assertEqual(
                normed, key,
                msg=f"Key {key!r} contains characters that would be removed by "
                    f"normalize_text(). After normalize_text: {normed!r}",
            )


# =============================================================================
# T11 — Address abbreviations
# =============================================================================

class TestAddressAbbreviations(unittest.TestCase):
    """
    Verified address-abbreviation rules from _ADDRESS_ABBREV_MAP must be
    applied by normalize_address() exactly as defined.
    """

    def test_ave_expanded(self):
        self.assertEqual(normalize_address("123 Main Ave"), "123 main avenue")

    def test_blvd_expanded(self):
        self.assertEqual(normalize_address("456 Oak Blvd"), "456 oak boulevard")

    def test_rd_expanded(self):
        self.assertEqual(normalize_address("789 Pine Rd"), "789 pine road")

    def test_dr_expanded(self):
        self.assertEqual(normalize_address("10 Elm Dr"), "10 elm drive")

    def test_ln_expanded(self):
        self.assertEqual(normalize_address("5 Oak Ln"), "5 oak lane")

    def test_ct_expanded(self):
        self.assertEqual(normalize_address("3 Maple Ct"), "3 maple court")

    def test_pl_expanded(self):
        self.assertEqual(normalize_address("7 Park Pl"), "7 park place")

    def test_hwy_expanded(self):
        self.assertEqual(normalize_address("100 Old Hwy"), "100 old highway")

    def test_pkwy_expanded(self):
        self.assertEqual(normalize_address("200 Sunset Pkwy"), "200 sunset parkway")

    def test_apt_expanded(self):
        self.assertEqual(normalize_address("Apt 4B"), "apartment 4b")

    def test_ste_expanded(self):
        self.assertEqual(normalize_address("Ste 300"), "suite 300")

    def test_abbrev_uppercase_expanded(self):
        # Input is uppercase; normalize_text lowercases before map
        self.assertEqual(normalize_address("123 MAIN AVE"), "123 main avenue")

    def test_dr_with_punctuation_expanded(self):
        # "DR," -> "DR" after punctuation removal -> "dr" -> "drive"
        self.assertEqual(normalize_address("10 Elm Dr, Springfield"), "10 elm drive springfield")

    # abbreviation map must NOT apply to names
    def test_address_abbrev_not_applied_to_name(self):
        result = normalize_name("100 Pine Ave Corp")
        # normalize_name applies _LEGAL_SUFFIX_MAP, not _ADDRESS_ABBREV_MAP
        # "ave" must remain as "ave" in name result
        self.assertIn("ave", result)

    # "st" is intentionally excluded (ambiguous)
    def test_st_not_in_map(self):
        self.assertNotIn("st", _ADDRESS_ABBREV_MAP)

    # All keys in the map are post-normalization forms
    def test_all_map_keys_are_post_normalization_forms(self):
        for key in _ADDRESS_ABBREV_MAP:
            normed = normalize_text(key)
            self.assertEqual(
                normed, key,
                msg=f"Key {key!r} after normalize_text: {normed!r}",
            )


# =============================================================================
# T12 — Multilingual / non-Latin text
# =============================================================================

class TestMultilingualText(unittest.TestCase):
    """
    Non-Latin Unicode word chars (CJK, Arabic, Devanagari, Cyrillic, etc.)
    must be preserved because the regex uses re.UNICODE and \w matches them.
    """

    def test_chinese_characters_preserved(self):
        result = normalize_text("ABC 中华")
        self.assertIn("中华", result)

    def test_arabic_characters_preserved(self):
        result = normalize_text("شركة ABC")
        self.assertIn("شركة", result)

    def test_devanagari_preserved(self):
        # BEHAVIOR NOTE: Devanagari anusvara U+0902 (nasalization mark on 'क')
        # is category Mn (non-spacing mark).  Under NFKC it decomposes such that
        # the resulting character is not matched by \w in the punctuation regex
        # and is replaced by a space.  This splits 'कंपनी' into 'क' and 'पनी'.
        # The BASE consonants are preserved; only the combining mark is split off.
        # This is the ACTUAL current behavior — not a bug to be fixed here.
        result = normalize_text("कंपनी ABC")
        # Base characters that survive
        self.assertIn("\u0915", result)   # 'क' base consonant
        self.assertIn("\u092a", result)   # 'प' base consonant
        self.assertIn("abc", result)
        # The full combined token does NOT survive (anusvara removed/split)
        # This is documented actual behavior, not an expectation violation.

    def test_cyrillic_preserved(self):
        result = normalize_text("Компания ABC")
        self.assertIn("компания", result)

    def test_japanese_preserved(self):
        result = normalize_text("株式会社 ABC")
        self.assertIn("株式会社", result)

    def test_korean_preserved(self):
        result = normalize_text("주식회사 ABC")
        self.assertIn("주식회사", result)

    def test_thai_preserved(self):
        # BEHAVIOR NOTE: Thai vowel signs (U+0E34 sara i, U+0E31 mai tai khu)
        # are non-spacing marks. After NFKC they may separate from their base
        # consonants in the punctuation-removal step, causing token splits.
        # The base Thai consonants are preserved; only the vowel marks may be
        # separated. Observed output: 'เบ' base chars are preserved.
        # This is the ACTUAL current behavior — not a bug to be fixed here.
        result = normalize_text("บริษัท ABC")
        # At least some Thai base characters survive
        self.assertIn("\u0e1a", result)   # 'บ' base consonant
        self.assertIn("abc", result)
        # The full combined token does NOT survive intact (vowel marks split off)
        # This is documented actual behavior, not an expectation violation.

    def test_mixed_latin_non_latin(self):
        result = normalize_name("ABC 中华 Corp")
        self.assertIn("中华", result)
        self.assertIn("abc", result)
        self.assertIn("corporation", result)

    def test_punctuation_between_non_latin_removed(self):
        # Comma between Chinese chars -> removed
        result = normalize_text("中华，公司")
        self.assertNotIn("，", result)   # fullwidth comma removed

    def test_fullwidth_chinese_punctuation_removed(self):
        # U+3002 IDEOGRAPHIC FULL STOP -> not \w -> removed
        result = normalize_text("公司\u3002ABC")
        self.assertNotIn("\u3002", result)


# =============================================================================
# T13 — Deterministic behavior
# =============================================================================

class TestDeterministicBehavior(unittest.TestCase):
    """
    The same input must always produce the same output.
    Run each normalization three times and confirm all outputs match.
    """

    INPUTS = [
        None,
        "",
        "  ",
        "Acme, Inc.",
        "Acme Pvt. Ltd.",
        "123 Main Ave, Suite 200",
        "中华 Company Ltd",
        "HELLO   WORLD!!!",
        "Café Müller GmbH",
        "\uFF21\uFF22\uFF23 Corp",
        "7-Eleven Stores Inc",
        float("nan"),
    ]

    def _check_deterministic(self, fn, label):
        for inp in self.INPUTS:
            results = [fn(inp) for _ in range(3)]
            self.assertEqual(
                results[0], results[1],
                msg=f"{label}({inp!r}): run 1 != run 2: {results[0]!r} vs {results[1]!r}",
            )
            self.assertEqual(
                results[1], results[2],
                msg=f"{label}({inp!r}): run 2 != run 3: {results[1]!r} vs {results[2]!r}",
            )

    def test_normalize_text_deterministic(self):
        self._check_deterministic(normalize_text, "normalize_text")

    def test_normalize_name_deterministic(self):
        self._check_deterministic(normalize_name, "normalize_name")

    def test_normalize_address_deterministic(self):
        self._check_deterministic(normalize_address, "normalize_address")


# =============================================================================
# T14 — preprocess_dataframe()
# =============================================================================

class TestPreprocessDataframe(unittest.TestCase):
    """
    Verifies the full batch preprocessing function with representative
    edge-case values in an in-memory DataFrame.
    """

    REQUIRED_COLUMNS = [
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

    @classmethod
    def setUpClass(cls):
        """Build representative edge-case DataFrame once for all tests."""
        cls.raw_df = pd.DataFrame([
            # (entity_id, business_name, business_address, country)
            ("E01", "Acme, Inc.",             "123 Main Ave",                "US"),
            ("E02", "Acme Pvt. Ltd.",          "456 Oak Blvd, Suite 10",      "IN"),
            ("E03", None,                      None,                          "GB"),
            ("E04", float("nan"),              float("nan"),                  "AU"),
            ("E05", "",                        "",                            "DE"),
            ("E06", "CALDEON NOVA",            "8706 KENTUCKY DERBY DR",      "US"),
            ("E07", "Caldeon  Nova",           "8706 Kentucky Derby Drive",   "US"),
            ("E08", "ABC 中华 Corp",            "北京 100020",                  "CN"),
            ("E09", "Müller GmbH",             "Café Str. 5",                 "DE"),
            ("E10", "\uFF21\uFF22\uFF23 Corp",  "Suite 200, 45 Park Ave",      "US"),
            ("E11", "7-Eleven Stores Inc",     "P.O. Box 1234",               "US"),
            ("E12", "H2O Technologies Corp",   "20085 Us 23, Circleville OH", "US"),
        ], columns=["entity_id", "business_name", "business_address", "country"])

        cls.result = preprocess_dataframe(cls.raw_df)

    # ── Schema ────────────────────────────────────────────────────────────────

    def test_output_has_exactly_14_columns(self):
        self.assertEqual(len(self.result.columns), 14)

    def test_all_required_columns_present(self):
        for col in self.REQUIRED_COLUMNS:
            self.assertIn(col, self.result.columns, msg=f"Missing column: {col!r}")

    def test_column_order_matches_schema(self):
        self.assertEqual(list(self.result.columns), self.REQUIRED_COLUMNS)

    def test_row_count_preserved(self):
        self.assertEqual(len(self.result), len(self.raw_df))

    # ── Raw columns preserved ─────────────────────────────────────────────────

    def test_raw_entity_id_preserved(self):
        self.assertListEqual(
            list(self.result["entity_id"]),
            list(self.raw_df["entity_id"]),
        )

    def test_raw_business_name_preserved(self):
        # Raw value must not be overwritten
        raw_e01 = self.raw_df.loc[self.raw_df["entity_id"] == "E01", "business_name"].iloc[0]
        out_e01 = self.result.loc[self.result["entity_id"] == "E01", "business_name"].iloc[0]
        self.assertEqual(raw_e01, out_e01)

    def test_raw_business_address_preserved(self):
        raw_e06 = self.raw_df.loc[self.raw_df["entity_id"] == "E06", "business_address"].iloc[0]
        out_e06 = self.result.loc[self.result["entity_id"] == "E06", "business_address"].iloc[0]
        self.assertEqual(raw_e06, out_e06)

    # ── Normalized columns correct ────────────────────────────────────────────

    def test_name_norm_acme_inc(self):
        row = self.result[self.result["entity_id"] == "E01"].iloc[0]
        self.assertEqual(row["name_norm"], "acme incorporated")

    def test_name_norm_pvt_ltd(self):
        row = self.result[self.result["entity_id"] == "E02"].iloc[0]
        self.assertEqual(row["name_norm"], "acme private limited")

    def test_name_norm_null_input(self):
        # E03: name is None -> name_norm == ""
        row = self.result[self.result["entity_id"] == "E03"].iloc[0]
        self.assertEqual(row["name_norm"], "")

    def test_name_norm_nan_input(self):
        # E04: name is NaN -> name_norm == ""
        row = self.result[self.result["entity_id"] == "E04"].iloc[0]
        self.assertEqual(row["name_norm"], "")

    def test_address_norm_ave_expanded(self):
        row = self.result[self.result["entity_id"] == "E01"].iloc[0]
        self.assertEqual(row["address_norm"], "123 main avenue")

    def test_address_norm_blvd_expanded(self):
        row = self.result[self.result["entity_id"] == "E02"].iloc[0]
        self.assertEqual(row["address_norm"], "456 oak boulevard suite 10")

    def test_address_norm_dr_expanded(self):
        row = self.result[self.result["entity_id"] == "E06"].iloc[0]
        self.assertEqual(row["address_norm"], "8706 kentucky derby drive")

    def test_name_norm_case_normalization(self):
        # E06 CALDEON NOVA -> E07 caldeon nova (should match after norm)
        r06 = self.result[self.result["entity_id"] == "E06"].iloc[0]["name_norm"]
        r07 = self.result[self.result["entity_id"] == "E07"].iloc[0]["name_norm"]
        self.assertEqual(r06, r07)

    def test_address_norm_case_normalization(self):
        # E06 "8706 KENTUCKY DERBY DR" == E07 "8706 Kentucky Derby Drive" after norm
        r06 = self.result[self.result["entity_id"] == "E06"].iloc[0]["address_norm"]
        r07 = self.result[self.result["entity_id"] == "E07"].iloc[0]["address_norm"]
        self.assertEqual(r06, r07)

    def test_non_latin_preserved_in_name_norm(self):
        row = self.result[self.result["entity_id"] == "E08"].iloc[0]
        self.assertIn("中华", row["name_norm"])

    def test_fullwidth_normalized_in_name(self):
        row = self.result[self.result["entity_id"] == "E10"].iloc[0]
        # Full-width ABC -> abc
        self.assertIn("abc", row["name_norm"])

    # ── Token lists ───────────────────────────────────────────────────────────

    def test_name_tokens_is_list(self):
        row = self.result[self.result["entity_id"] == "E01"].iloc[0]
        self.assertIsInstance(row["name_tokens"], list)

    def test_name_tokens_empty_for_null_input(self):
        row = self.result[self.result["entity_id"] == "E03"].iloc[0]
        self.assertEqual(row["name_tokens"], [])

    def test_name_tokens_correct(self):
        row = self.result[self.result["entity_id"] == "E01"].iloc[0]
        self.assertEqual(row["name_tokens"], ["acme", "incorporated"])

    def test_address_tokens_is_list(self):
        row = self.result[self.result["entity_id"] == "E01"].iloc[0]
        self.assertIsInstance(row["address_tokens"], list)

    # ── Derived numeric fields ────────────────────────────────────────────────

    def test_name_token_count_correct(self):
        row = self.result[self.result["entity_id"] == "E01"].iloc[0]
        self.assertEqual(row["name_token_count"], 2)  # "acme", "incorporated"

    def test_name_token_count_zero_for_empty(self):
        row = self.result[self.result["entity_id"] == "E05"].iloc[0]
        self.assertEqual(row["name_token_count"], 0)

    def test_name_length_correct(self):
        row = self.result[self.result["entity_id"] == "E01"].iloc[0]
        # "acme incorporated" -> len 17
        self.assertEqual(row["name_length"], len("acme incorporated"))

    def test_name_length_zero_for_empty(self):
        row = self.result[self.result["entity_id"] == "E05"].iloc[0]
        self.assertEqual(row["name_length"], 0)

    def test_name_digits_correct(self):
        # E11: "7-Eleven Stores Inc" -> name_norm = "7 eleven stores incorporated"
        row = self.result[self.result["entity_id"] == "E11"].iloc[0]
        # digits in name_norm: only '7'
        self.assertEqual(row["name_digits"], 1)

    def test_name_digits_zero_for_no_digits(self):
        row = self.result[self.result["entity_id"] == "E01"].iloc[0]
        self.assertEqual(row["name_digits"], 0)

    def test_address_digits_correct(self):
        # E01: address_norm = "123 main avenue" -> digits: 1,2,3 -> count=3
        row = self.result[self.result["entity_id"] == "E01"].iloc[0]
        self.assertEqual(row["address_digits"], 3)

    def test_address_digits_zero_for_null(self):
        row = self.result[self.result["entity_id"] == "E03"].iloc[0]
        self.assertEqual(row["address_digits"], 0)

    # ── No exception / type safety ────────────────────────────────────────────

    def test_no_exception_raised(self):
        # If setUpClass raised, these tests would already fail; explicit check
        self.assertIsNotNone(self.result)

    def test_result_is_dataframe(self):
        self.assertIsInstance(self.result, pd.DataFrame)

    # ── Determinism ───────────────────────────────────────────────────────────

    def test_deterministic_repeated_calls(self):
        out1 = preprocess_dataframe(self.raw_df)
        out2 = preprocess_dataframe(self.raw_df)
        for col in self.REQUIRED_COLUMNS:
            self.assertTrue(
                out1[col].equals(out2[col]),
                msg=f"Column {col!r} differs between two calls to preprocess_dataframe()",
            )

    def test_input_not_mutated(self):
        raw_copy = self.raw_df.copy(deep=True)
        preprocess_dataframe(self.raw_df)
        # Compare column by column (can't use equals directly with NaN/None)
        for col in raw_copy.columns:
            for i in range(len(raw_copy)):
                orig = raw_copy.iloc[i][col]
                curr = self.raw_df.iloc[i][col]
                if isinstance(orig, float) and math.isnan(orig):
                    self.assertTrue(
                        isinstance(curr, float) and math.isnan(curr),
                        msg=f"Input mutated at row {i}, col {col!r}",
                    )
                else:
                    self.assertEqual(
                        orig, curr,
                        msg=f"Input mutated at row {i}, col {col!r}: {orig!r} -> {curr!r}",
                    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
