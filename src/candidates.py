"""
src/candidates.py
=================
Candidate-pair format contract for the Amazon ML Challenge 2026.

Member 4 deliverable — candidate generation / blocking.

This module owns the conversion between M4's internal (row-per-pair)
representation and the official submission format (one row per S1 entity,
comma-separated candidate IDs).

Official candidate_pairs.tsv format
-------------------------------------
source1_entity_id    candidate_entity_ids
S1-00001             S2-1,S2-2,S3-4       ← comma-separated, no spaces
S1-00002             S3-8
S1-00003                                  ← empty when no candidates

Rules (from validate_submission.py)
--------------------------------------
  • Exactly two tab-separated columns.
  • One row per S1 entity (duplicates → error).
  • candidate_entity_ids: comma-separated S2-/S3- IDs, no duplicates within a row.
  • No S1- IDs in the candidate list.
  • Every test S1 entity must have a row (even if empty).

M4 internal representation (flexible)
--------------------------------------
M4 may use any intermediate format during blocking, e.g.:

    source1_entity_id   candidate_entity_id
    S1-001              S2-010
    S1-001              S2-020
    S1-001              S3-030
    S1-002              S3-008

Call to_official_format() to convert to the submission format before writing.

Validation / statistics helpers
---------------------------------
  candidate_stats(df)         — coverage, avg candidates, reduction ratio
  validate_format(df)         — check against submission rules
"""

from __future__ import annotations

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Format conversion
# ─────────────────────────────────────────────────────────────────────────────

def to_official_format(
    internal_df: pd.DataFrame,
    all_s1_ids: list[str] | pd.Series | None = None,
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
) -> pd.DataFrame:
    """
    Convert M4 internal (row-per-pair) candidate DataFrame to the official
    one-row-per-S1-entity submission format.

    Parameters
    ----------
    internal_df  : DataFrame with at least two columns: *s1_col*, *cand_col*.
    all_s1_ids   : Optional complete list of S1 entity IDs.  When provided,
                   every S1 ID gets a row (empty candidate list if no candidates).
                   When None, only S1 IDs present in internal_df are written.
    s1_col       : Name of the S1 entity ID column in internal_df.
    cand_col     : Name of the candidate entity ID column in internal_df.

    Returns
    -------
    DataFrame with columns ["source1_entity_id", "candidate_entity_ids"].
    """
    # Aggregate: group by S1, collect unique candidate IDs, join with comma
    grouped = (
        internal_df.groupby(s1_col)[cand_col]
        .apply(lambda ids: ",".join(sorted(set(str(i) for i in ids if str(i).strip()))))
        .reset_index()
        .rename(columns={s1_col: "source1_entity_id", cand_col: "candidate_entity_ids"})
    )

    if all_s1_ids is not None:
        # Ensure every S1 entity has a row, even with no candidates
        all_s1_df = pd.DataFrame({"source1_entity_id": list(all_s1_ids)})
        grouped = all_s1_df.merge(grouped, on="source1_entity_id", how="left")
        grouped["candidate_entity_ids"] = grouped["candidate_entity_ids"].fillna("")

    return grouped[["source1_entity_id", "candidate_entity_ids"]].reset_index(drop=True)


def write_candidates(
    official_df: pd.DataFrame,
    path: str,
    encoding: str = "utf-8",
) -> None:
    """Write the official candidate DataFrame to a TSV file."""
    official_df.to_csv(path, sep="\t", index=False, encoding=encoding)
    n_total = len(official_df)
    n_empty = (official_df["candidate_entity_ids"] == "").sum()
    print(f"Written {path}: {n_total:,} S1 rows  ({n_empty:,} empty)")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def candidate_stats(
    official_df: pd.DataFrame,
    truth_map: dict[str, set[str]] | None = None,
) -> dict:
    """
    Compute coverage and quality statistics for the official candidate file.

    Parameters
    ----------
    official_df : output of to_official_format()
    truth_map   : optional {s1_id → set of true matched IDs} for recall stats

    Returns
    -------
    dict with keys: n_s1, n_with_candidates, n_empty, total_candidate_ids,
                    avg_per_s1, max_per_s1, candidate_recall (if truth_map),
                    true_matches_lost (if truth_map), reduction_ratio (if truth_map)
    """
    counts = official_df["candidate_entity_ids"].apply(
        lambda x: len(x.split(",")) if x.strip() else 0
    )

    stats: dict = {
        "n_s1":               len(official_df),
        "n_with_candidates":  int((counts > 0).sum()),
        "n_empty":            int((counts == 0).sum()),
        "total_candidate_ids": int(counts.sum()),
        "avg_per_s1":         round(float(counts.mean()), 2),
        "max_per_s1":         int(counts.max()),
    }

    if truth_map is not None:
        # Recall: fraction of true matches that appear in the candidate set
        found = missed = 0
        for _, row in official_df.iterrows():
            sid  = row["source1_entity_id"]
            cids = set(row["candidate_entity_ids"].split(",")) if row["candidate_entity_ids"].strip() else set()
            true_matches = truth_map.get(sid, set())
            found  += len(true_matches & cids)
            missed += len(true_matches - cids)

        total_true = found + missed
        stats["candidate_recall"]   = round(found / total_true, 6) if total_true else None
        stats["true_matches_lost"]  = missed
        # Reduction ratio: how much of the cross-product is avoided
        n_s2_s3 = stats["total_candidate_ids"]       # rough proxy
        stats["reduction_ratio"] = None              # needs S2/S3 total IDs externally

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_format(official_df: pd.DataFrame) -> list[str]:
    """
    Check the official candidate DataFrame against submission rules.
    Returns a list of error strings (empty = all clear).
    """
    errors: list[str] = []

    # 1. Required columns
    for col in ("source1_entity_id", "candidate_entity_ids"):
        if col not in official_df.columns:
            errors.append(f"Missing column: {col}")

    if errors:
        return errors

    # 2. No duplicate S1 IDs
    dupes = official_df["source1_entity_id"][official_df["source1_entity_id"].duplicated()].tolist()
    if dupes:
        errors.append(f"Duplicate source1_entity_id rows: {dupes[:5]}")

    # 3. Per-row checks
    for _, row in official_df.iterrows():
        sid  = row["source1_entity_id"]
        raw  = row["candidate_entity_ids"]
        if not isinstance(raw, str) or raw.strip() == "":
            continue
        ids = [x.strip() for x in raw.split(",")]
        # Duplicate within row
        if len(ids) != len(set(ids)):
            errors.append(f"{sid}: duplicate IDs in candidate list")
        # Wrong prefix
        bad = [x for x in ids if not x.startswith(("S2-", "S3-"))]
        if bad:
            errors.append(f"{sid}: non-S2/S3 candidate IDs: {bad[:3]}")

    return errors
