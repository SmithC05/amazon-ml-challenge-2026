"""
tests/test_m3_streaming.py
==========================
Unit and integration tests for the M3 streaming pipeline refactor.

Tests
-----
A. 10K candidate smoke test (using existing output/local_10k/candidate_pairs_10k.tsv).
B. Two or more candidate chunks for the same S1 merge correctly.
C. Zero-candidate S1 is preserved in output.
D. Duplicate candidate IDs do not create duplicate predictions.
E. Feature order is exactly the canonical 16-feature order.
F. Prediction does not rely on fitting the model/scaler.
G. Model threshold comes from model_config.json.
H. Large synthetic candidate input is processed chunk-wise without giant DataFrame.
I. Output format is exactly: source1_entity_id<TAB>matched_entity_ids
"""

from __future__ import annotations

import io
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Bootstrap sys.path so tests can import src.* without installation ──────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from features import (
    FEATURE_NAMES,
    build_feature_matrix_from_dicts,
    extract_features,
)
from predict import (
    _build_entity_lookups,
    _build_s1_lookup,
    _expand_chunk,
    load_model,
    predict,
)
from train import (
    build_truth_map,
    entity_split,
    _apply_train_cap,
    entity_f05,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny synthetic fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_s1(ids):
    rows = [
        {
            "entity_id":    sid,
            "name_norm":    f"name {sid}",
            "address_norm": f"addr {sid}",
            "country":      "us",
        }
        for sid in ids
    ]
    return pd.DataFrame(rows)


def _make_s2(ids):
    rows = [
        {
            "entity_id":    cid,
            "name_norm":    f"name {cid}",
            "address_norm": f"addr {cid}",
            "country":      "us",
        }
        for cid in ids
    ]
    return pd.DataFrame(rows)


def _make_cand_tsv(pairs: list[tuple[str, list[str]]]) -> str:
    """Return a TSV string from list of (s1_id, [cand_ids])."""
    lines = ["source1_entity_id\tcandidate_entity_ids"]
    for s1_id, cids in pairs:
        lines.append(f"{s1_id}\t{','.join(cids)}")
    return "\n".join(lines) + "\n"


def _make_stub_model(n_features: int = 16):
    """
    Return a (scaler, model, threshold, config) stub where the model
    always predicts 0.99 (match) for every pair.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    # Fit scaler on a tiny dummy matrix so it is callable
    scaler.fit(np.zeros((2, n_features)))

    model = LogisticRegression()
    # Fit model on tiny dummy data: both classes present
    model.fit(
        np.vstack([np.zeros(n_features), np.ones(n_features)]),
        [0, 1],
    )
    # Force coefficients so model always returns ~1.0 probability
    model.coef_ = np.ones((1, n_features)) * 100.0
    model.intercept_ = np.array([0.0])

    threshold = 0.5
    config = {
        "model_type":   "stub",
        "feature_names": FEATURE_NAMES,
        "threshold":    threshold,
    }
    return scaler, model, threshold, config


def _write_model(tmpdir: Path, scaler, model, config: dict) -> Path:
    model_dir = tmpdir / "models"
    model_dir.mkdir()
    with open(model_dir / "matcher.pkl", "wb") as f:
        pickle.dump((scaler, model), f)
    with open(model_dir / "model_config.json", "w") as f:
        json.dump(config, f)
    return model_dir


# ─────────────────────────────────────────────────────────────────────────────
# E — Feature order
# ─────────────────────────────────────────────────────────────────────────────

def test_E_feature_order_is_canonical():
    """build_feature_matrix_from_dicts returns columns in exactly FEATURE_NAMES order."""
    pair_dict = {
        "s1_name_norm":    "acme corp",
        "s1_address_norm": "123 main st",
        "s1_country":      "us",
        "cand_name_norm":  "acme corporation",
        "cand_address_norm":"123 main st",
        "cand_country":    "us",
        "cand_entity_id":  "S2-001",
    }
    arr = build_feature_matrix_from_dicts([pair_dict])
    assert arr.shape == (1, 16), f"Expected shape (1,16), got {arr.shape}"
    # Verify each feature can be extracted individually and matches position
    feats = extract_features(pd.Series(pair_dict))
    for i, name in enumerate(FEATURE_NAMES):
        assert abs(arr[0, i] - feats[name]) < 1e-9, (
            f"Feature {name} at position {i}: array={arr[0,i]}, dict={feats[name]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# D — Duplicate candidates → no duplicate predictions
# ─────────────────────────────────────────────────────────────────────────────

def test_D_duplicate_candidate_ids_no_duplicate_predictions():
    """
    If the same candidate ID appears twice for an S1, the prediction set
    must contain it only once.
    """
    s1  = _make_s1(["S1-001"])
    s2  = _make_s2(["S2-100"])
    s1_lookup   = _build_s1_lookup(s1)
    cand_lookup = _build_entity_lookups(s2, pd.DataFrame(columns=["entity_id","name_norm","address_norm","country"]))

    # Deliberately duplicate S2-100 in the candidate list
    cand_tsv = _make_cand_tsv([("S1-001", ["S2-100", "S2-100", "S2-100"])])
    chunk_df = pd.read_csv(io.StringIO(cand_tsv), sep="\t", dtype=str)

    pairs = _expand_chunk(chunk_df, s1_lookup, cand_lookup)
    # All three tuples are produced (dedup happens in pred_map accumulation)
    assert len(pairs) == 3

    # Simulate predict loop: accumulate into set → dedup guaranteed
    pred_map: dict[str, set] = {}
    for p in pairs:
        pred_map.setdefault(p["source1_entity_id"], set()).add(p["cand_entity_id"])
    assert pred_map["S1-001"] == {"S2-100"}


# ─────────────────────────────────────────────────────────────────────────────
# B — Multi-chunk same S1 merges correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_B_multi_chunk_same_s1_merges_correctly(tmp_path):
    """
    When candidates for S1-001 span two chunks (chunk_size=1),
    the final pred_map should contain all matches from both chunks.
    """
    s1  = _make_s1(["S1-001"])
    s2  = _make_s2(["S2-100", "S2-200"])
    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)

    # Two rows — each chunk will be size 1
    cand_tsv = _make_cand_tsv([
        ("S1-001", ["S2-100"]),
        ("S1-001", ["S2-200"]),
    ])
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text(cand_tsv, encoding="utf-8")

    # Build lookup dicts (simulate what predict() does)
    s1_lookup   = _build_s1_lookup(s1)
    cand_lookup = _build_entity_lookups(s2, pd.DataFrame(columns=["entity_id","name_norm","address_norm","country"]))

    pred_map: dict[str, set] = {}
    for chunk_df in pd.read_csv(cand_path, sep="\t", dtype=str, chunksize=1, keep_default_na=False):
        pairs = _expand_chunk(chunk_df, s1_lookup, cand_lookup)
        X = build_feature_matrix_from_dicts(pairs)
        X_sc = scaler.transform(X)
        proba = model.predict_proba(X_sc)[:, 1]
        for i, d in enumerate(pairs):
            if proba[i] >= threshold:
                pred_map.setdefault(d["source1_entity_id"], set()).add(d["cand_entity_id"])

    assert "S1-001" in pred_map
    assert "S2-100" in pred_map["S1-001"]
    assert "S2-200" in pred_map["S1-001"]


# ─────────────────────────────────────────────────────────────────────────────
# C — Zero-candidate S1 preserved
# ─────────────────────────────────────────────────────────────────────────────

def test_C_zero_candidate_s1_preserved(tmp_path):
    """
    S1 entities with no candidates must appear in output with empty matched_entity_ids.
    """
    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)

    # S1-002 has no candidates
    s1  = _make_s1(["S1-001", "S1-002"])
    s2  = _make_s2(["S2-100"])

    cand_tsv = _make_cand_tsv([("S1-001", ["S2-100"])])
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text(cand_tsv, encoding="utf-8")

    # Write minimal source TSVs (no cache)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for src, df in [("test_source1", s1), ("test_source2", s2)]:
        df.rename(columns={
            "name_norm": "business_name",
            "address_norm": "business_address",
        }, inplace=False).assign(
            business_name=df["name_norm"],
            business_address=df["address_norm"],
        )[["entity_id","business_name","business_address","country"]].to_csv(
            data_dir / f"{src}.tsv", sep="\t", index=False
        )
    # Create empty source3
    pd.DataFrame(columns=["entity_id","business_name","business_address","country"]).to_csv(
        data_dir / "test_source3.tsv", sep="\t", index=False
    )

    out_dir = tmp_path / "output"
    predict(
        data_dir=data_dir,
        candidates_path=cand_path,
        model_dir=model_dir,
        output_dir=out_dir,
        cache_dir=None,
        split="test",
        chunk_size=100,
    )

    result = pd.read_csv(out_dir / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert len(result) == 2, f"Expected 2 rows, got {len(result)}"
    s1_002_row = result[result["source1_entity_id"] == "S1-002"]
    assert len(s1_002_row) == 1
    assert s1_002_row["matched_entity_ids"].iloc[0] == ""


# ─────────────────────────────────────────────────────────────────────────────
# F — Prediction does not re-fit model/scaler
# ─────────────────────────────────────────────────────────────────────────────

def test_F_predict_does_not_refit(tmp_path, monkeypatch):
    """
    Verify the prediction pipeline calls .transform() only (never .fit_transform()).
    Strategy: write a clean (picklable) model to disk, then monkeypatch
    sklearn.preprocessing.StandardScaler.fit_transform to raise AFTER the
    model file is written but BEFORE predict() is called.
    """
    from sklearn.preprocessing import StandardScaler

    scaler, model, threshold, config = _make_stub_model()
    # Write model BEFORE patching (so pickle.dump never sees the lambda)
    model_dir = _write_model(tmp_path, scaler, model, config)

    # Now patch: any call to fit_transform on a StandardScaler instance raises
    def _no_fit_transform(self, X, y=None):
        raise AssertionError("fit_transform was called during prediction — model was RE-FIT!")

    monkeypatch.setattr(StandardScaler, "fit_transform", _no_fit_transform)

    s1 = _make_s1(["S1-001"])
    s2 = _make_s2(["S2-100"])
    cand_tsv = _make_cand_tsv([("S1-001", ["S2-100"])])
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text(cand_tsv, encoding="utf-8")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for src, df in [("test_source1", s1), ("test_source2", s2)]:
        df.assign(
            business_name=df["name_norm"],
            business_address=df["address_norm"],
        )[["entity_id","business_name","business_address","country"]].to_csv(
            data_dir / f"{src}.tsv", sep="\t", index=False
        )
    pd.DataFrame(columns=["entity_id","business_name","business_address","country"]).to_csv(
        data_dir / "test_source3.tsv", sep="\t", index=False
    )

    out_dir = tmp_path / "output"
    # Should not raise — fit_transform is patched to raise if called
    predict(
        data_dir=data_dir,
        candidates_path=cand_path,
        model_dir=model_dir,
        output_dir=out_dir,
        chunk_size=100,
    )


# ─────────────────────────────────────────────────────────────────────────────
# G — Threshold comes from model_config.json
# ─────────────────────────────────────────────────────────────────────────────

def test_G_threshold_from_config(tmp_path):
    """load_model() reads threshold from model_config.json."""
    scaler, model, _, config = _make_stub_model()
    config["threshold"] = 0.77  # override
    model_dir = _write_model(tmp_path, scaler, model, config)

    _, _, threshold, _ = load_model(model_dir)
    assert abs(threshold - 0.77) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# I — Output format
# ─────────────────────────────────────────────────────────────────────────────

def test_I_output_format(tmp_path):
    """The output TSV has exactly the two correct tab-separated columns."""
    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)

    s1 = _make_s1(["S1-001"])
    s2 = _make_s2(["S2-100"])
    cand_tsv = _make_cand_tsv([("S1-001", ["S2-100"])])
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text(cand_tsv, encoding="utf-8")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for src, df in [("test_source1", s1), ("test_source2", s2)]:
        df.assign(
            business_name=df["name_norm"],
            business_address=df["address_norm"],
        )[["entity_id","business_name","business_address","country"]].to_csv(
            data_dir / f"{src}.tsv", sep="\t", index=False
        )
    pd.DataFrame(columns=["entity_id","business_name","business_address","country"]).to_csv(
        data_dir / "test_source3.tsv", sep="\t", index=False
    )

    out_dir = tmp_path / "output"
    predict(
        data_dir=data_dir,
        candidates_path=cand_path,
        model_dir=model_dir,
        output_dir=out_dir,
        chunk_size=100,
    )

    out_path = out_dir / "matching_results.tsv"
    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Header
    assert lines[0].rstrip("\n") == "source1_entity_id\tmatched_entity_ids"
    # Each data line has exactly one tab
    for line in lines[1:]:
        parts = line.rstrip("\n").split("\t")
        assert len(parts) == 2, f"Expected 2 columns, got: {parts}"
        assert parts[0].startswith("S1-")


# ─────────────────────────────────────────────────────────────────────────────
# H — Large synthetic input processed chunk-wise (no giant DataFrame)
# ─────────────────────────────────────────────────────────────────────────────

def test_H_chunked_processing_no_giant_dataframe(tmp_path):
    """
    Generate 500 S1 entities with 10 candidates each (5000 candidate pairs).
    Process in chunk_size=50.  Verify output has all 500 S1 rows.
    """
    n_s1   = 500
    n_cand = 10
    s1_ids = [f"S1-{i:04d}" for i in range(n_s1)]
    s2_ids = [f"S2-{i:04d}" for i in range(n_s1 * n_cand)]

    s1 = _make_s1(s1_ids)
    s2 = _make_s2(s2_ids)

    lines = ["source1_entity_id\tcandidate_entity_ids"]
    for i, sid in enumerate(s1_ids):
        cids = s2_ids[i * n_cand : (i + 1) * n_cand]
        lines.append(f"{sid}\t{','.join(cids)}")
    cand_tsv = "\n".join(lines) + "\n"

    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text(cand_tsv, encoding="utf-8")

    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    s1.assign(business_name=s1["name_norm"], business_address=s1["address_norm"])[
        ["entity_id","business_name","business_address","country"]
    ].to_csv(data_dir / "test_source1.tsv", sep="\t", index=False)
    s2.assign(business_name=s2["name_norm"], business_address=s2["address_norm"])[
        ["entity_id","business_name","business_address","country"]
    ].to_csv(data_dir / "test_source2.tsv", sep="\t", index=False)
    pd.DataFrame(columns=["entity_id","business_name","business_address","country"]).to_csv(
        data_dir / "test_source3.tsv", sep="\t", index=False
    )

    out_dir = tmp_path / "output"
    predict(
        data_dir=data_dir,
        candidates_path=cand_path,
        model_dir=model_dir,
        output_dir=out_dir,
        chunk_size=50,  # small chunks
    )

    result = pd.read_csv(out_dir / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert len(result) == n_s1, f"Expected {n_s1} rows, got {len(result)}"
    assert set(result["source1_entity_id"]) == set(s1_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Train helpers: entity_split, _apply_train_cap
# ─────────────────────────────────────────────────────────────────────────────

def test_entity_split_no_overlap():
    """entity_split: train and val sets must be disjoint."""
    ids = [f"S1-{i}" for i in range(100)]
    train_ids, val_ids = entity_split(ids)
    assert train_ids & val_ids == set()
    assert train_ids | val_ids == set(ids)


def test_entity_split_deterministic():
    """entity_split is deterministic for the same seed."""
    ids = [f"S1-{i}" for i in range(200)]
    a_train, a_val = entity_split(ids, random_state=42)
    b_train, b_val = entity_split(ids, random_state=42)
    assert a_train == b_train
    assert a_val   == b_val


def test_apply_train_cap_retains_all_positives():
    """_apply_train_cap: all positives must always be retained."""
    n = 1000
    df = pd.DataFrame({
        "source1_entity_id": [f"S1-{i}" for i in range(n)],
        "cand_entity_id":    [f"S2-{i}" for i in range(n)],
        "label": [1 if i < 100 else 0 for i in range(n)],
    })
    capped = _apply_train_cap(df, max_pairs=200)
    assert (capped["label"] == 1).sum() == 100, "All positives must be retained"
    assert len(capped) <= 200


def test_apply_train_cap_no_sampling_under_limit():
    """_apply_train_cap: does not modify DataFrame when under the cap."""
    df = pd.DataFrame({
        "source1_entity_id": [f"S1-{i}" for i in range(10)],
        "cand_entity_id":    [f"S2-{i}" for i in range(10)],
        "label": [0] * 10,
    })
    capped = _apply_train_cap(df, max_pairs=1000)
    assert len(capped) == 10


# ─────────────────────────────────────────────────────────────────────────────
# A — 10K smoke test (requires local 10K candidate file)
# ─────────────────────────────────────────────────────────────────────────────

CAND_10K_PATH = REPO_ROOT / "output" / "local_10k" / "candidate_pairs_10k.tsv"
MODEL_PKL     = REPO_ROOT / "models" / "matcher.pkl"
MODEL_CONFIG  = REPO_ROOT / "models" / "model_config.json"

@pytest.mark.skipif(
    not CAND_10K_PATH.exists() or not MODEL_PKL.exists(),
    reason="10K candidate file or model not present (skipping smoke test)"
)
def test_A_10k_smoke_test(tmp_path):
    """
    Run predict() on the 10K candidate file with the frozen model.
    Verify output has the expected number of rows and correct format.
    """
    # Determine which cache / data_dir to use
    cache_dir = REPO_ROOT / "cache"
    if not (cache_dir / "test_source1.parquet").exists():
        cache_dir = None

    data_dir = REPO_ROOT / "dataset" / "test"
    if not data_dir.exists():
        pytest.skip("Test data directory not present")

    out_dir = tmp_path / "output"
    predict(
        data_dir=data_dir,
        candidates_path=CAND_10K_PATH,
        model_dir=REPO_ROOT / "models",
        output_dir=out_dir,
        cache_dir=cache_dir,
        split="test",
        chunk_size=5000,
    )

    out_path = out_dir / "matching_results.tsv"
    assert out_path.exists()
    result = pd.read_csv(out_path, sep="\t", dtype=str, keep_default_na=False)

    # Header check
    assert list(result.columns) == ["source1_entity_id", "matched_entity_ids"]

    # No duplicates
    assert result["source1_entity_id"].duplicated().sum() == 0

    # All S1 IDs start with "S1-"
    assert result["source1_entity_id"].str.startswith("S1-").all()

    # All non-empty matched IDs start with S2- or S3-
    for _, row in result[result["matched_entity_ids"] != ""].iterrows():
        for mid in row["matched_entity_ids"].split(","):
            assert mid.startswith(("S2-", "S3-")), f"Bad match ID: {mid}"

    print(f"\n10K smoke test PASS — {len(result):,} S1 rows in output")
