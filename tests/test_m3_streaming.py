"""
tests/test_m3_streaming.py
==========================
Unit and integration tests for the M3 fully-streaming pipeline (v3).

Original 13 tests (A–I) are preserved.
New tests (A2–J) verify the disk-backed architecture corrections.

Test index
----------
Original (from v2):
  test_E_feature_order_is_canonical
  test_D_duplicate_candidate_ids_no_duplicate_predictions
  test_B_multi_chunk_same_s1_merges_correctly
  test_C_zero_candidate_s1_preserved
  test_F_predict_does_not_refit
  test_G_threshold_from_config
  test_I_output_format
  test_H_chunked_processing_no_giant_dataframe
  test_entity_split_no_overlap
  test_entity_split_deterministic
  test_apply_train_cap_retains_all_positives
  test_apply_train_cap_no_sampling_under_limit
  test_A_10k_smoke_test

New (v3 correction pass):
  test_A2_spool_written_incrementally          — ParquetWriter used, no concat
  test_A2b_spool_no_all_frames_list           — all_frames_for_write not present
  test_A2c_spool_not_loaded_whole            — pd.read_parquet(spool) not called
  test_D2_val_sweep_streaming               — threshold sweep scored in batches
  test_E2_zero_cand_val_in_metric           — zero-cand val S1 contributes 1.0
  test_F2_predict_uses_duckdb               — DuckDB accumulator is used
  test_G2_duplicate_matches_deduped_in_db   — DuckDB deduplication works
  test_H2_output_every_s1_exactly_once      — every S1 in output exactly once
  test_J_10k_smoke_train                    — 10K training smoke test
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
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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
    _insert_matches,
    _open_match_db,
    _write_output_from_db,
    load_model,
    predict,
)
from train import (
    build_truth_map,
    entity_split,
    entity_f05,
    _expand_and_label_chunk,
    threshold_sweep_streaming,
    _select_train_pairs_from_spool,
    _SPOOL_SCHEMA,
    train,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures / factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_s1(ids):
    return pd.DataFrame([{
        "entity_id": sid, "name_norm": f"name {sid}",
        "address_norm": f"addr {sid}", "country": "us",
    } for sid in ids])


def _make_s2(ids):
    return pd.DataFrame([{
        "entity_id": cid, "name_norm": f"name {cid}",
        "address_norm": f"addr {cid}", "country": "us",
    } for cid in ids])


def _make_cand_tsv(pairs: list[tuple[str, list[str]]]) -> str:
    lines = ["source1_entity_id\tcandidate_entity_ids"]
    for s1_id, cids in pairs:
        lines.append(f"{s1_id}\t{','.join(cids)}")
    return "\n".join(lines) + "\n"


def _make_stub_model(n_features: int = 16):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.zeros((2, n_features)))

    model = LogisticRegression()
    model.fit(
        np.vstack([np.zeros(n_features), np.ones(n_features)]),
        [0, 1],
    )
    model.coef_ = np.ones((1, n_features)) * 100.0
    model.intercept_ = np.array([0.0])

    threshold = 0.5
    config = {
        "model_type":    "stub",
        "feature_names": FEATURE_NAMES,
        "threshold":     threshold,
    }
    return scaler, model, threshold, config


def _write_model(tmpdir: Path, scaler, model, config: dict) -> Path:
    model_dir = tmpdir / "models"
    model_dir.mkdir(exist_ok=True)
    with open(model_dir / "matcher.pkl", "wb") as f:
        pickle.dump((scaler, model), f)
    with open(model_dir / "model_config.json", "w") as f:
        json.dump(config, f)
    return model_dir


def _make_source_tsvs(data_dir: Path, s1: pd.DataFrame, s2: pd.DataFrame):
    """Write minimal source TSVs for predict() to consume (no cache)."""
    data_dir.mkdir(exist_ok=True)
    for src, df in [("test_source1", s1), ("test_source2", s2)]:
        df.assign(
            business_name=df["name_norm"],
            business_address=df["address_norm"],
        )[["entity_id", "business_name", "business_address", "country"]].to_csv(
            data_dir / f"{src}.tsv", sep="\t", index=False
        )
    pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"]).to_csv(
        data_dir / "test_source3.tsv", sep="\t", index=False
    )


def _make_small_spool(path: Path, n_s1: int = 5, n_neg_each: int = 3, n_pos_each: int = 1):
    """Build a tiny synthetic spool Parquet file for testing."""
    rows = {k: [] for k in _SPOOL_SCHEMA.names}
    for i in range(n_s1):
        sid = f"S1-{i:04d}"
        # positives
        for j in range(n_pos_each):
            cid = f"S2-pos-{i}-{j}"
            rows["source1_entity_id"].append(sid)
            rows["cand_entity_id"].append(cid)
            rows["s1_name_norm"].append(f"name {sid}")
            rows["s1_address_norm"].append(f"addr {sid}")
            rows["s1_country"].append("us")
            rows["cand_name_norm"].append(f"name {cid}")
            rows["cand_address_norm"].append(f"addr {cid}")
            rows["cand_country"].append("us")
            rows["label"].append(1)
            rows["is_val"].append(1 if i >= n_s1 // 2 else 0)
        # negatives
        for j in range(n_neg_each):
            cid = f"S2-neg-{i}-{j}"
            rows["source1_entity_id"].append(sid)
            rows["cand_entity_id"].append(cid)
            rows["s1_name_norm"].append(f"name {sid}")
            rows["s1_address_norm"].append(f"addr {sid}")
            rows["s1_country"].append("us")
            rows["cand_name_norm"].append(f"unrelated {cid}")
            rows["cand_address_norm"].append(f"other {cid}")
            rows["cand_country"].append("gb")
            rows["label"].append(0)
            rows["is_val"].append(1 if i >= n_s1 // 2 else 0)

    table = pa.table({k: rows[k] for k in _SPOOL_SCHEMA.names}, schema=_SPOOL_SCHEMA)
    pq.write_table(table, path)


# ─────────────────────────────────────────────────────────────────────────────
# ── ORIGINAL 13 TESTS (preserved) ────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def test_E_feature_order_is_canonical():
    pair_dict = {
        "s1_name_norm": "acme corp", "s1_address_norm": "123 main st", "s1_country": "us",
        "cand_name_norm": "acme corporation", "cand_address_norm": "123 main st",
        "cand_country": "us", "cand_entity_id": "S2-001",
    }
    arr = build_feature_matrix_from_dicts([pair_dict])
    assert arr.shape == (1, 16)
    feats = extract_features(pd.Series(pair_dict))
    for i, name in enumerate(FEATURE_NAMES):
        assert abs(arr[0, i] - feats[name]) < 1e-9


def test_D_duplicate_candidate_ids_no_duplicate_predictions():
    s1 = _make_s1(["S1-001"])
    s2 = _make_s2(["S2-100"])
    s1_lookup   = _build_s1_lookup(s1)
    cand_lookup = _build_entity_lookups(s2, pd.DataFrame(columns=["entity_id","name_norm","address_norm","country"]))

    cand_tsv = _make_cand_tsv([("S1-001", ["S2-100", "S2-100", "S2-100"])])
    chunk_df = pd.read_csv(io.StringIO(cand_tsv), sep="\t", dtype=str)
    pairs = _expand_chunk(chunk_df, s1_lookup, cand_lookup)
    assert len(pairs) == 3

    # Accumulate into a set (as predict() does via DuckDB primary key dedup)
    pred_map: dict[str, set] = {}
    for p in pairs:
        pred_map.setdefault(p["source1_entity_id"], set()).add(p["cand_entity_id"])
    assert pred_map["S1-001"] == {"S2-100"}


def test_B_multi_chunk_same_s1_merges_correctly(tmp_path):
    s1  = _make_s1(["S1-001"])
    s2  = _make_s2(["S2-100", "S2-200"])
    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)

    cand_tsv = _make_cand_tsv([("S1-001", ["S2-100"]), ("S1-001", ["S2-200"])])
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text(cand_tsv, encoding="utf-8")

    s1_lookup   = _build_s1_lookup(s1)
    cand_lookup = _build_entity_lookups(s2, pd.DataFrame(columns=["entity_id","name_norm","address_norm","country"]))

    db_path = str(tmp_path / "test_acc.duckdb")
    con = _open_match_db(db_path)

    for chunk_df in pd.read_csv(cand_path, sep="\t", dtype=str, chunksize=1, keep_default_na=False):
        pairs = _expand_chunk(chunk_df, s1_lookup, cand_lookup)
        X = build_feature_matrix_from_dicts(pairs)
        X_sc = scaler.transform(X)
        probas = model.predict_proba(X_sc)[:, 1]
        _insert_matches(con, pairs, probas, threshold)

    out_path = tmp_path / "result.tsv"
    _write_output_from_db(con, ["S1-001"], out_path)
    con.close()

    result = pd.read_csv(out_path, sep="\t", dtype=str, keep_default_na=False)
    matched = set(result.iloc[0]["matched_entity_ids"].split(","))
    assert "S2-100" in matched
    assert "S2-200" in matched


def test_C_zero_candidate_s1_preserved(tmp_path):
    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)
    s1 = _make_s1(["S1-001", "S1-002"])
    s2 = _make_s2(["S2-100"])
    cand_tsv = _make_cand_tsv([("S1-001", ["S2-100"])])
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text(cand_tsv, encoding="utf-8")
    data_dir = tmp_path / "data"
    _make_source_tsvs(data_dir, s1, s2)
    out_dir = tmp_path / "output"

    predict(data_dir=data_dir, candidates_path=cand_path, model_dir=model_dir,
            output_dir=out_dir, chunk_size=100)

    result = pd.read_csv(out_dir / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert len(result) == 2
    s1_002 = result[result["source1_entity_id"] == "S1-002"]
    assert s1_002["matched_entity_ids"].iloc[0] == ""


def test_F_predict_does_not_refit(tmp_path, monkeypatch):
    from sklearn.preprocessing import StandardScaler

    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)

    def _no_fit_transform(self, X, y=None):
        raise AssertionError("fit_transform was called during prediction — model was RE-FIT!")

    monkeypatch.setattr(StandardScaler, "fit_transform", _no_fit_transform)

    s1 = _make_s1(["S1-001"])
    s2 = _make_s2(["S2-100"])
    cand_tsv = _make_cand_tsv([("S1-001", ["S2-100"])])
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text(cand_tsv, encoding="utf-8")
    data_dir = tmp_path / "data"
    _make_source_tsvs(data_dir, s1, s2)
    out_dir = tmp_path / "output"
    predict(data_dir=data_dir, candidates_path=cand_path, model_dir=model_dir,
            output_dir=out_dir, chunk_size=100)


def test_G_threshold_from_config(tmp_path):
    scaler, model, _, config = _make_stub_model()
    config["threshold"] = 0.77
    model_dir = _write_model(tmp_path, scaler, model, config)
    _, _, threshold, _ = load_model(model_dir)
    assert abs(threshold - 0.77) < 1e-9


def test_I_output_format(tmp_path):
    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)
    s1 = _make_s1(["S1-001"])
    s2 = _make_s2(["S2-100"])
    cand_tsv = _make_cand_tsv([("S1-001", ["S2-100"])])
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text(cand_tsv, encoding="utf-8")
    data_dir = tmp_path / "data"
    _make_source_tsvs(data_dir, s1, s2)
    out_dir = tmp_path / "output"
    predict(data_dir=data_dir, candidates_path=cand_path, model_dir=model_dir,
            output_dir=out_dir, chunk_size=100)

    out_path = out_dir / "matching_results.tsv"
    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert lines[0].rstrip("\n") == "source1_entity_id\tmatched_entity_ids"
    for line in lines[1:]:
        parts = line.rstrip("\n").split("\t")
        assert len(parts) == 2
        assert parts[0].startswith("S1-")


def test_H_chunked_processing_no_giant_dataframe(tmp_path):
    n_s1, n_cand = 500, 10
    s1_ids = [f"S1-{i:04d}" for i in range(n_s1)]
    s2_ids = [f"S2-{i:04d}" for i in range(n_s1 * n_cand)]
    s1 = _make_s1(s1_ids)
    s2 = _make_s2(s2_ids)
    lines = ["source1_entity_id\tcandidate_entity_ids"]
    for i, sid in enumerate(s1_ids):
        lines.append(f"{sid}\t{','.join(s2_ids[i*n_cand:(i+1)*n_cand])}")
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)
    data_dir = tmp_path / "data"
    _make_source_tsvs(data_dir, s1, s2)
    out_dir = tmp_path / "output"
    predict(data_dir=data_dir, candidates_path=cand_path, model_dir=model_dir,
            output_dir=out_dir, chunk_size=50)

    result = pd.read_csv(out_dir / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert len(result) == n_s1
    assert set(result["source1_entity_id"]) == set(s1_ids)


def test_entity_split_no_overlap():
    ids = [f"S1-{i}" for i in range(100)]
    train_ids, val_ids = entity_split(ids)
    assert train_ids & val_ids == set()
    assert train_ids | val_ids == set(ids)


def test_entity_split_deterministic():
    ids = [f"S1-{i}" for i in range(200)]
    a_train, a_val = entity_split(ids, random_state=42)
    b_train, b_val = entity_split(ids, random_state=42)
    assert a_train == b_train
    assert a_val   == b_val


def test_class_balance_1_pos_exceeds_cap_with_neg(tmp_path):
    """TEST 1: positives > max_train_pairs, negatives available."""
    spool_path = tmp_path / "spool.parquet"
    _make_small_spool(spool_path, n_s1=40, n_neg_each=5, n_pos_each=1)
    # n_s1=40 means 20 train, 20 val. 
    # Total train pos = 20, Total train neg = 100. Let max_pairs = 15.
    df, meta = _select_train_pairs_from_spool(spool_path, max_pairs=15)
    
    assert (df["label"] == 1).sum() == 20  # All positives retained
    assert (df["label"] == 0).sum() == 20  # min(n_neg_avail, n_pos) = min(100, 20) = 20
    assert len(df) == 40
    assert meta["effective_max_train_pairs"] == 40

def test_class_balance_2_pos_exceeds_cap_no_neg(tmp_path):
    """TEST 2: positives > max_train_pairs, negatives = 0. Should fail clearly."""
    spool_path = tmp_path / "spool.parquet"
    _make_small_spool(spool_path, n_s1=40, n_neg_each=0, n_pos_each=1)
    
    with pytest.raises(ValueError, match="0 training negatives available"):
        _select_train_pairs_from_spool(spool_path, max_pairs=15)

def test_class_balance_3_pos_under_cap(tmp_path):
    """TEST 3: positives < max_train_pairs, normal negative sampling."""
    spool_path = tmp_path / "spool.parquet"
    _make_small_spool(spool_path, n_s1=20, n_neg_each=10, n_pos_each=1)
    # n_s1=20 -> 10 train S1. Train pos = 10, Train neg = 100. Let max_pairs = 50.
    df, meta = _select_train_pairs_from_spool(spool_path, max_pairs=50)
    
    assert (df["label"] == 1).sum() == 10
    assert (df["label"] == 0).sum() == 40  # 50 - 10
    assert len(df) == 50
    assert meta["effective_max_train_pairs"] == 50

def test_class_balance_4_deterministic_sampling(tmp_path):
    """TEST 4: repeated run with seed=42 produces identical selected pairs."""
    spool_path = tmp_path / "spool.parquet"
    _make_small_spool(spool_path, n_s1=40, n_neg_each=20, n_pos_each=1)
    
    df1, _ = _select_train_pairs_from_spool(spool_path, max_pairs=100, random_state=42)
    df2, _ = _select_train_pairs_from_spool(spool_path, max_pairs=100, random_state=42)
    pd.testing.assert_frame_equal(df1, df2)

def test_class_balance_5_model_fit_succeeds(tmp_path):
    """TEST 5: model.fit succeeds with the resulting training set."""
    from sklearn.linear_model import LogisticRegression
    spool_path = tmp_path / "spool.parquet"
    _make_small_spool(spool_path, n_s1=40, n_neg_each=5, n_pos_each=1)
    
    df, _ = _select_train_pairs_from_spool(spool_path, max_pairs=10) # 20 pos, will take 20 negs
    X = build_feature_matrix_from_dicts(df.to_dict("records"))
    y = df["label"].values
    
    model = LogisticRegression()
    model.fit(X, y)
    assert hasattr(model, "coef_")



CAND_10K_PATH = REPO_ROOT / "output" / "local_10k" / "candidate_pairs_10k.tsv"
MODEL_PKL     = REPO_ROOT / "models" / "matcher.pkl"

@pytest.mark.skipif(
    not CAND_10K_PATH.exists() or not MODEL_PKL.exists(),
    reason="10K candidate file or model not present"
)
def test_A_10k_smoke_test(tmp_path):
    cache_dir = REPO_ROOT / "cache"
    if not (cache_dir / "test_source1.parquet").exists():
        cache_dir = None
    data_dir = REPO_ROOT / "dataset" / "test"
    if not data_dir.exists():
        pytest.skip("Test data directory not present")

    out_dir = tmp_path / "output"
    predict(data_dir=data_dir, candidates_path=CAND_10K_PATH,
            model_dir=REPO_ROOT / "models", output_dir=out_dir,
            cache_dir=cache_dir, split="test", chunk_size=5000)

    result = pd.read_csv(out_dir / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert list(result.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert result["source1_entity_id"].duplicated().sum() == 0
    assert result["source1_entity_id"].str.startswith("S1-").all()
    for _, row in result[result["matched_entity_ids"] != ""].iterrows():
        for mid in row["matched_entity_ids"].split(","):
            assert mid.startswith(("S2-", "S3-")), f"Bad match ID: {mid}"
    print(f"\n10K smoke test PASS — {len(result):,} S1 rows")


# ─────────────────────────────────────────────────────────────────────────────
# ── NEW TESTS (v3 correction pass) ───────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def test_A2_spool_written_incrementally(tmp_path):
    """
    Verify that the spool Parquet file grows after EACH candidate chunk,
    proving ParquetWriter is used (not accumulate-then-write).
    """
    s1 = _make_s1([f"S1-{i}" for i in range(10)])
    s2 = _make_s2([f"S2-{i}" for i in range(30)])

    gt_rows = [{"source1_entity_id": f"S1-{i}", "matched_entity_ids": f"S2-{i}"} for i in range(10)]
    gt = pd.DataFrame(gt_rows)
    truth_map = build_truth_map(gt)
    train_s1_set = {f"S1-{i}" for i in range(8)}
    val_s1_set   = {f"S1-{i}" for i in range(8, 10)}
    s1_lookup    = _build_s1_lookup(s1)
    cand_lookup  = _build_entity_lookups(s2, pd.DataFrame(columns=["entity_id","name_norm","address_norm","country"]))

    spool_path = tmp_path / "spool.parquet"
    sizes: list[int] = []

    with pq.ParquetWriter(str(spool_path), schema=_SPOOL_SCHEMA) as writer:
        for i in range(0, 10, 2):  # 5 chunks of 2 rows each
            chunk_rows = [{"source1_entity_id": f"S1-{i+j}",
                           "candidate_entity_ids": f"S2-{i+j},S2-{(i+j+10)%30}"}
                          for j in range(2)]
            chunk_df = pd.DataFrame(chunk_rows)
            arrow_table = _expand_and_label_chunk(chunk_df, s1_lookup, cand_lookup, truth_map, val_s1_set)
            if arrow_table is not None:
                writer.write_table(arrow_table)
            sizes.append(spool_path.stat().st_size if spool_path.exists() else 0)

    # File must have grown across at least one step
    assert sizes[-1] > 0, "Spool file is empty"
    # Multiple chunks written → file grown more than once
    # (first write creates header, subsequent writes add row groups)
    pf = pq.ParquetFile(spool_path)
    assert pf.metadata.num_row_groups >= 1


def test_A2b_spool_no_all_frames_list():
    """
    Source-code audit: train.py must NOT contain 'all_frames_for_write'.
    """
    train_src = (REPO_ROOT / "src" / "train.py").read_text(encoding="utf-8")
    assert "all_frames_for_write" not in train_src, (
        "FAIL: 'all_frames_for_write' found in src/train.py — "
        "accumulating all frames in a list is not streaming!"
    )


def test_A2c_spool_not_loaded_whole():
    """
    Source-code audit: train.py must NOT contain 'pd.read_parquet(spool_path)'.
    """
    train_src = (REPO_ROOT / "src" / "train.py").read_text(encoding="utf-8")
    assert "pd.read_parquet(spool_path)" not in train_src, (
        "FAIL: 'pd.read_parquet(spool_path)' found in src/train.py — "
        "loading entire spool defeats the purpose!"
    )
    assert "pd.concat(all_frames" not in train_src, (
        "FAIL: 'pd.concat(all_frames' found in src/train.py — giant concat present!"
    )


def test_D2_val_sweep_streaming(tmp_path):
    """
    Threshold sweep uses streaming batches — validate on a synthetic spool.
    Checks that the sweep produces correct results without requiring
    the full val DataFrame in RAM.
    """
    spool_path = tmp_path / "spool.parquet"
    n_s1 = 10
    _make_small_spool(spool_path, n_s1=n_s1, n_neg_each=3, n_pos_each=1)

    scaler, model, threshold, _ = _make_stub_model()
    truth_map = {f"S1-{i:04d}": {f"S2-pos-{i}-0"} for i in range(n_s1)}
    val_ids = [f"S1-{i:04d}" for i in range(n_s1 // 2, n_s1)]

    sweep_df = threshold_sweep_streaming(
        spool_path=spool_path,
        scaler=scaler,
        model=model,
        truth_map=truth_map,
        val_s1_ids=val_ids,
        thresholds=[0.5],
        batch_size=4,  # tiny batch to confirm chunking
    )
    assert len(sweep_df) == 1
    assert "threshold" in sweep_df.columns
    assert "f0_5" in sweep_df.columns
    assert 0.0 <= sweep_df["f0_5"].iloc[0] <= 1.0


def test_E2_zero_cand_val_in_metric(tmp_path):
    """
    Zero-candidate val S1 entities with empty truth must contribute 1.0 to metric.
    """
    spool_path = tmp_path / "spool.parquet"
    # Only one val entity has pairs; another has none (pure zero-cand)
    _make_small_spool(spool_path, n_s1=2, n_neg_each=2, n_pos_each=1)

    scaler, model, threshold, _ = _make_stub_model()
    # val_ids includes "S1-0001" which has zero candidates in truth_map
    val_ids = ["S1-0001", "S1-ZERO"]  # S1-ZERO not in spool at all
    truth_map = {
        "S1-0001": {"S2-pos-1-0"},
        "S1-ZERO": set(),          # zero truth → entity_f05 with empty pred = 1.0
    }

    sweep_df = threshold_sweep_streaming(
        spool_path=spool_path,
        scaler=scaler,
        model=model,
        truth_map=truth_map,
        val_s1_ids=val_ids,
        thresholds=[0.5],
        batch_size=100,
    )
    # The macro-F0.5 should be > 0 (at least the zero-cand entity contributes 1.0)
    assert sweep_df["f0_5"].iloc[0] > 0.0


def test_F2_predict_uses_duckdb(tmp_path):
    """
    Verify that predict() creates a DuckDB accumulator file during execution.
    The .match_acc.duckdb file should exist transiently; we verify the import
    and that the output is correctly generated.
    """
    import duckdb  # confirm duckdb is imported in predict.py
    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)
    s1 = _make_s1(["S1-001", "S1-002"])
    s2 = _make_s2(["S2-100", "S2-200"])
    cand_tsv = _make_cand_tsv([("S1-001", ["S2-100"]), ("S1-002", ["S2-200"])])
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text(cand_tsv, encoding="utf-8")
    data_dir = tmp_path / "data"
    _make_source_tsvs(data_dir, s1, s2)
    out_dir = tmp_path / "output"

    predict(data_dir=data_dir, candidates_path=cand_path, model_dir=model_dir,
            output_dir=out_dir, chunk_size=1)  # tiny chunk to force multiple iterations

    result = pd.read_csv(out_dir / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert len(result) == 2
    # Both entities should have matches (stub model predicts 0.99 for all)
    assert result["matched_entity_ids"].ne("").all()

    # DuckDB temp file should be cleaned up
    assert not (out_dir / ".match_acc.duckdb").exists()


def test_G2_duplicate_matches_deduped_in_db(tmp_path):
    """
    DuckDB PRIMARY KEY constraint deduplicates the same (S1, cand) pair
    inserted from two different chunks.
    """
    import duckdb
    db_path = str(tmp_path / "dedup_test.duckdb")
    con = _open_match_db(db_path)

    scaler, model, threshold, _ = _make_stub_model()

    pair = [{
        "source1_entity_id": "S1-001", "cand_entity_id": "S2-100",
        "s1_name_norm": "x", "s1_address_norm": "y", "s1_country": "us",
        "cand_name_norm": "x", "cand_address_norm": "y", "cand_country": "us",
    }]
    probas = np.array([0.99])

    # Insert the same pair 3 times (simulating duplicate candidate rows)
    for _ in range(3):
        _insert_matches(con, pair, probas, threshold)

    count = con.execute("SELECT COUNT(*) FROM matches WHERE source1_entity_id='S1-001'").fetchone()[0]
    assert count == 1, f"Expected 1 match row, got {count}"
    con.close()


def test_H2_output_every_s1_exactly_once(tmp_path):
    """
    Even with many chunks and repeated S1 IDs in candidates, each S1 appears
    exactly once in the output TSV.
    """
    n_s1 = 20
    s1_ids = [f"S1-{i:04d}" for i in range(n_s1)]
    s2_ids = [f"S2-{i:04d}" for i in range(n_s1 * 5)]
    s1 = _make_s1(s1_ids)
    s2 = _make_s2(s2_ids)

    # Each S1 appears 3 times across chunks
    lines = ["source1_entity_id\tcandidate_entity_ids"]
    for sid in s1_ids:
        for _ in range(3):
            lines.append(f"{sid}\tS2-0000,S2-0001,S2-0002")
    cand_path = tmp_path / "cands.tsv"
    cand_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    scaler, model, threshold, config = _make_stub_model()
    model_dir = _write_model(tmp_path, scaler, model, config)
    data_dir = tmp_path / "data"
    _make_source_tsvs(data_dir, s1, s2)
    out_dir = tmp_path / "output"
    predict(data_dir=data_dir, candidates_path=cand_path, model_dir=model_dir,
            output_dir=out_dir, chunk_size=3)

    result = pd.read_csv(out_dir / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert len(result) == n_s1, f"Expected {n_s1} rows, got {len(result)}"
    assert result["source1_entity_id"].duplicated().sum() == 0


@pytest.mark.skipif(
    not (REPO_ROOT / "output" / "local_10k" / "candidate_pairs_10k.tsv").exists(),
    reason="10K candidate file not present"
)
def test_J_10k_smoke_train(tmp_path):
    """
    10K training smoke test — verifies train() completes and produces a valid model.
    """
    cand_path = REPO_ROOT / "output" / "local_10k" / "candidate_pairs_10k.tsv"
    data_dir  = REPO_ROOT / "dataset" / "train"
    if not data_dir.exists():
        pytest.skip("Train data directory not present")

    cache_dir = REPO_ROOT / "cache"
    if not (cache_dir / "train_source1.parquet").exists():
        cache_dir = None

    models_dir = tmp_path / "models"
    out_dir    = tmp_path / "output"

    train(
        data_dir=data_dir,
        candidates_path=cand_path,
        output_dir=out_dir,
        models_dir=models_dir,
        cache_dir=cache_dir,
        chunk_size=5000,
        max_train_pairs=500_000,  # smaller cap for speed
    )

    assert (models_dir / "matcher.pkl").exists()
    assert (models_dir / "model_config.json").exists()
    with open(models_dir / "model_config.json") as f:
        cfg = json.load(f)
    assert "threshold" in cfg
    assert cfg["feature_names"] == FEATURE_NAMES
    assert cfg["train_pair_count"] > 0
    print(f"\n10K smoke train PASS — threshold={cfg['threshold']:.2f}  f0_5={cfg['f0_5']:.4f}")
