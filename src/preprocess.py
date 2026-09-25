import re
import unicodedata
import pandas as pd


def normalize_text(value):
    """
    Normalize a raw text field for entity matching.

    Steps:
      1. Return empty string for null/NaN values.
      2. Apply Unicode NFKC normalization (e.g. fullwidth → ASCII,
         composed accents → precomposed forms).
      3. Lowercase the string.
      4. Replace punctuation characters with a single space so that
         tokens such as 'st.' and 'st' compare equal, without
         discarding address numbers or other meaningful digits.
      5. Collapse consecutive whitespace to a single space and strip
         leading/trailing whitespace.

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


def normalize_name(value):
    """
    Normalize a business name field.

    Delegates to normalize_text().  Kept as a separate entry point so
    that name-specific rules (e.g. expanding legal suffixes) can be
    added here without touching address normalization.

    Args:
        value: Raw business name (str, float NaN, or None).

    Returns:
        Normalized business name string.
    """
    return normalize_text(value)


def normalize_address(value):
    """
    Normalize a business address field.

    Delegates to normalize_text().  Kept as a separate entry point so
    that address-specific rules (e.g. unit/suite token handling,
    directional abbreviations) can be added here without touching name
    normalization.

    Args:
        value: Raw business address (str, float NaN, or None).

    Returns:
        Normalized business address string.
    """
    return normalize_text(value)
