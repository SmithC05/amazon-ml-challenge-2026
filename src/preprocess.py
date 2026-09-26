"""
src/preprocess.py
=================
Reusable normalization functions for the Amazon ML Challenge 2026
entity-matching pipeline.

Member 2 deliverable — validated via notebooks/02_normalization_effectiveness.ipynb

Normalization pipeline
----------------------
Phase 1 (normalize_text):
  1. Return "" for NaN/None.
  2. Unicode NFKC normalization.
  3. Lowercase.
  4. Replace punctuation with spaces  ([^\\w\\s] -> " ").
  5. Collapse whitespace.

Phase 2 — validated token-level rules applied in normalize_name / normalize_address:
  • Legal-suffix rules (name only): standardize abbreviated legal forms to their
    canonical long forms (e.g. "pvt" → "private", "ltd" → "limited").
    Rules accepted in Phase 4 of the normalization-effectiveness study.
  • Address-abbreviation rules (address only): expand common street-type
    abbreviations to full words (e.g. "ave" → "avenue", "blvd" → "boulevard").
    Rules accepted in Phase 4 of the normalization-effectiveness study.

Design principles
-----------------
  • Raw field values are NEVER overwritten; callers store normalized output
    alongside the original.
  • Unicode-safe throughout (re.UNICODE flag, unicodedata.normalize NFKC).
  • No aggressive information loss: digits, address numbers, and non-Latin
    scripts are preserved by normalize_text.
  • No fuzzy matching, blocking, ML, or candidate generation.
  • Country is treated as an open-set string — no fixed country list is assumed.
"""

import re
import unicodedata
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Internal token-map helpers
# ──────────────────────────────────────────────────────────────────────────────

def _apply_token_map(text: str, token_map: dict) -> str:
    """
    Apply a dictionary of token (or phrase) replacements to a normalized string.

    Replacement is whole-token only (never substring).  Multi-token keys are
    matched before overlapping single-token keys (longest-match-first), so
    "pvt ltd" is replaced as a unit before "pvt" or "ltd" individually.

    Args:
        text:      Normalized string (already lowercase, punctuation removed).
        token_map: Mapping from source phrase → replacement phrase.
                   Use an empty-string replacement to delete a token.

    Returns:
        String with applicable tokens replaced, whitespace re-collapsed.
    """
    if not text or not isinstance(text, str):
        return text

    tokens = text.split()
    result = []
    i = 0
    max_phrase_len = max((len(k.split()) for k in token_map), default=1)

    while i < len(tokens):
        matched = False
        for n in range(min(max_phrase_len, len(tokens) - i), 0, -1):
            phrase = " ".join(tokens[i: i + n])
            if phrase in token_map:
                replacement = token_map[phrase]
                if replacement:                    # non-empty: insert tokens
                    result.extend(replacement.split())
                i += n
                matched = True
                break
        if not matched:
            result.append(tokens[i])
            i += 1

    return " ".join(result)


# ──────────────────────────────────────────────────────────────────────────────
# Validated legal-suffix token map  (name normalization only)
# Validated in: notebooks/02_normalization_effectiveness.ipynb Phase 4
# Rules accepted: new_exact >= 1 AND lost_exact == 0
# ──────────────────────────────────────────────────────────────────────────────
# Standardise abbreviated legal forms to their canonical long form.
# Multi-token keys must appear before their single-token sub-keys so that the
# longest-match-first logic fires correctly.
_LEGAL_SUFFIX_MAP: dict = {
    # Multi-token first
    "pvt ltd":                  "private limited",
    # Single-token
    "pvt":                      "private",
    "ltd":                      "limited",
    "inc":                      "incorporated",
    "corp":                     "corporation",
    "llp":                      "limited liability partnership",
}


# ──────────────────────────────────────────────────────────────────────────────
# Validated address-abbreviation token map  (address normalization only)
# Validated in: notebooks/02_normalization_effectiveness.ipynb Phase 4
# Rules accepted: new_exact >= 1 AND lost_exact == 0
# ──────────────────────────────────────────────────────────────────────────────
# Only unambiguous, high-frequency abbreviations are included.
# Directional single-letter abbreviations (n/s/e/w) are deliberately excluded
# because they are too ambiguous (appear in many non-directional contexts).
_ADDRESS_ABBREV_MAP: dict = {
    "ave":  "avenue",
    "blvd": "boulevard",
    "rd":   "road",
    "dr":   "drive",
    "ln":   "lane",
    "ct":   "court",
    "pl":   "place",
    "hwy":  "highway",
    "pkwy": "parkway",
    "apt":  "apartment",
    "ste":  "suite",
    # "st" intentionally omitted: ambiguous (street vs. saint vs. state suffix)
    # "n/s/e/w" intentionally omitted: too ambiguous
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text(value) -> str:
    """
    Core normalization pipeline shared by all field types.

    Steps:
      1. Return "" for NaN / None.
      2. Unicode NFKC normalization — converts compatibility characters
         (fullwidth Latin, ligatures) to canonical equivalents and
         composes combining diacritics.
      3. Lowercase.
      4. Replace punctuation with a single space  ([^\\w\\s] -> " ").
         Address numbers and digits are preserved (they are word-character matches).
      5. Collapse consecutive whitespace; strip leading/trailing space.

    Args:
        value: Raw field value (str, float NaN, or None).

    Returns:
        Normalized string, or "" if the input was null/empty.
    """
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = value.lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value) -> str:
    """
    Normalize a business name field.

    Applies the base normalize_text() pipeline followed by the validated
    legal-suffix token map (_LEGAL_SUFFIX_MAP).

    The legal-suffix rules standardize abbreviated legal forms to their
    canonical long forms (e.g. "pvt" → "private", "ltd" → "limited",
    "pvt ltd" → "private limited") so that names that differ only in their
    legal-suffix abbreviation style compare as equal.

    Rules were validated in notebooks/02_normalization_effectiveness.ipynb
    Phase 4: each accepted rule creates at least one new exact match in the
    true labeled pairs without breaking any existing exact matches.

    Args:
        value: Raw business name (str, float NaN, or None).

    Returns:
        Normalized business name string.
    """
    text = normalize_text(value)
    if not text:
        return text
    return _apply_token_map(text, _LEGAL_SUFFIX_MAP)


def normalize_address(value) -> str:
    """
    Normalize a business address field.

    Applies the base normalize_text() pipeline followed by the validated
    address-abbreviation token map (_ADDRESS_ABBREV_MAP).

    The address-abbreviation rules expand unambiguous street-type tokens to
    their full forms (e.g. "ave" → "avenue", "blvd" → "boulevard", "rd" → "road")
    so that addresses that differ only in street-type abbreviation style compare
    as equal.

    Ambiguous abbreviations ("st" = street vs. saint; single-letter directionals
    "n", "s", "e", "w") are intentionally excluded to avoid over-normalization.

    Rules were validated in notebooks/02_normalization_effectiveness.ipynb
    Phase 4: each accepted rule creates at least one new exact match in the
    true labeled pairs without breaking any existing exact matches.

    Args:
        value: Raw business address (str, float NaN, or None).

    Returns:
        Normalized business address string.
    """
    text = normalize_text(value)
    if not text:
        return text
    return _apply_token_map(text, _ADDRESS_ABBREV_MAP)


# ──────────────────────────────────────────────────────────────────────────────
# Batch preprocessing — DataFrame-level entry point
# ──────────────────────────────────────────────────────────────────────────────

_OUTPUT_COLUMNS = [
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


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to a source DataFrame in one batch pass.

    Each source record (S1, S2, or S3) is processed exactly once.  The returned
    DataFrame is a new object — the input is never mutated.

    Args:
        df: DataFrame with at least these columns:
              entity_id, business_name, business_address, country.
            Extra columns are silently ignored.

    Returns:
        A new DataFrame with exactly 14 columns in this fixed order:

          entity_id         – preserved from input (no change)
          business_name     – preserved from input (no change)
          business_address  – preserved from input (no change)
          country           – preserved from input (no change)
          name_norm         – normalize_name(business_name)
          address_norm      – normalize_address(business_address)
          name_tokens       – whitespace-split tokens of name_norm (list)
          address_tokens    – whitespace-split tokens of address_norm (list)
          name_token_count  – len(name_tokens)
          address_token_count – len(address_tokens)
          name_length       – character length of name_norm
          address_length    – character length of address_norm
          name_digits       – count of digit characters in name_norm
          address_digits    – count of digit characters in address_norm

    Notes:
        • NaN / None in business_name or business_address are handled safely by
          normalize_name() / normalize_address(), which return "" for null inputs.
        • name_tokens / address_tokens are Python lists ([] for empty strings).
        • This function does not perform candidate-pair generation, fuzzy
          matching, blocking, or feature engineering beyond the fields listed
          above.
        • Country is treated as a free-form string; no country list is assumed.
    """
    # Work on a copy of only the four raw columns so the input is never mutated.
    out = df[["entity_id", "business_name", "business_address", "country"]].copy()

    # ── Normalized strings (one call per record via map) ──────────────────────
    # normalize_name / normalize_address are scalar functions validated in M2.
    # .map() is used in preference to .apply() for scalar string → string
    # transforms: it avoids the overhead of constructing a Series per call.
    out["name_norm"]    = out["business_name"].map(normalize_name)
    out["address_norm"] = out["business_address"].map(normalize_address)

    # ── Token lists ───────────────────────────────────────────────────────────
    # normalize_name / normalize_address always return str (never NaN), so
    # str.split() always returns a list ([] for empty strings).
    out["name_tokens"]    = out["name_norm"].str.split()
    out["address_tokens"] = out["address_norm"].str.split()

    # ── Token counts ─────────────────────────────────────────────────────────
    out["name_token_count"]    = out["name_tokens"].str.len()
    out["address_token_count"] = out["address_tokens"].str.len()

    # ── Character lengths ────────────────────────────────────────────────────
    out["name_length"]    = out["name_norm"].str.len()
    out["address_length"] = out["address_norm"].str.len()

    # ── Digit counts ─────────────────────────────────────────────────────────
    # Count digit characters in the normalized string.
    # sum(c.isdigit() for c in "") == 0, so empty strings are safe.
    out["name_digits"]    = out["name_norm"].apply(
        lambda s: sum(c.isdigit() for c in s)
    )
    out["address_digits"] = out["address_norm"].apply(
        lambda s: sum(c.isdigit() for c in s)
    )

    # Return columns in the required fixed order.
    return out[_OUTPUT_COLUMNS]
