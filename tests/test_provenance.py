"""
Tests for input fingerprinting and staleness detection.

The behaviour that matters most: a result whose inputs changed is caught, a
result whose inputs are unchanged is reused, and a result saved before
fingerprinting existed is never invalidated just for being old.

Run: pytest tests/test_provenance.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from multimodalva.utils.provenance import (
    StaleReport,
    check_inputs,
    file_fingerprint,
    guard_inputs,
    record_inputs,
    resolve_stale,
)


def _oof(tmp_path, seed=0):
    """A stand-in for the out-of-fold artifacts Stage 2 consumes."""
    rng = np.random.default_rng(seed)
    x = tmp_path / "oof_meta_X.npy"
    y = tmp_path / "oof_y.npy"
    np.save(x, rng.random((50, 12)))
    np.save(y, rng.integers(0, 4, 50))
    return {"oof_meta_X": x, "oof_y": y}


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def test_fingerprint_follows_content_not_timestamp(tmp_path):
    f = tmp_path / "a.npy"
    np.save(f, np.arange(10))
    first = file_fingerprint(f)

    f.touch()                                  # new mtime, same bytes
    assert file_fingerprint(f)["sha256"] == first["sha256"]

    np.save(f, np.arange(11))                  # different bytes
    assert file_fingerprint(f)["sha256"] != first["sha256"]


def test_missing_file_is_recorded_not_raised(tmp_path):
    meta = record_inputs({}, {"gone": tmp_path / "nope.npy"})
    assert meta["inputs"]["files"]["gone"]["missing"] is True


# ---------------------------------------------------------------------------
# Detecting staleness
# ---------------------------------------------------------------------------

def test_unchanged_inputs_are_reusable(tmp_path):
    inputs = _oof(tmp_path)
    meta = record_inputs({"best_meta_name": "logistic_regression"}, inputs)
    report = check_inputs(meta, inputs)
    assert report.status == "match"
    assert not report.stale


def test_rerunning_upstream_with_identical_output_does_not_invalidate(tmp_path):
    """Re-running a base model that produces the same numbers is not a change."""
    inputs = _oof(tmp_path, seed=3)
    meta = record_inputs({}, inputs)
    _oof(tmp_path, seed=3)                     # rewrite, same content
    assert check_inputs(meta, inputs).status == "match"


def test_changed_oof_matrix_is_detected(tmp_path):
    inputs = _oof(tmp_path, seed=0)
    meta = record_inputs({}, inputs)
    _oof(tmp_path, seed=1)                     # base model re-run, new numbers

    report = check_inputs(meta, inputs)
    assert report.status == "stale"
    assert "oof_meta_X" in report.changed
    assert "changed" in report.message("The saved meta-learner")


def test_deleted_input_is_detected(tmp_path):
    inputs = _oof(tmp_path)
    meta = record_inputs({}, inputs)
    inputs["oof_meta_X"].unlink()
    report = check_inputs(meta, inputs)
    assert report.status == "missing"
    assert report.stale


# ---------------------------------------------------------------------------
# Backward compatibility: existing results must keep working
# ---------------------------------------------------------------------------

def test_result_without_fingerprints_is_unknown_not_stale(tmp_path):
    """Results saved before input tracking existed are reused, not invalidated."""
    legacy = {"best_meta_name": "logistic_regression", "meta_scores": {}}
    report = check_inputs(legacy, _oof(tmp_path))
    assert report.status == "unknown"
    assert not report.stale


def test_legacy_result_passes_the_guard(tmp_path):
    report = check_inputs({"best_meta_name": "lightgbm"}, _oof(tmp_path))
    guard_inputs(report, "The saved meta-learner", "re-run stage 2")   # must not raise


def test_missing_metadata_file_is_unknown(tmp_path):
    assert check_inputs(None, _oof(tmp_path)).status == "unknown"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def test_guard_stops_by_default_and_names_the_fix(tmp_path):
    inputs = _oof(tmp_path, seed=0)
    meta = record_inputs({}, inputs)
    _oof(tmp_path, seed=1)
    report = check_inputs(meta, inputs)

    with pytest.raises(ValueError, match="re-run train_meta_learner_stage"):
        guard_inputs(report, "The saved meta-learner", "re-run train_meta_learner_stage()")


@pytest.mark.parametrize("policy", ["warn", "ignore"])
def test_guard_can_be_overridden(tmp_path, policy):
    inputs = _oof(tmp_path, seed=0)
    meta = record_inputs({}, inputs)
    _oof(tmp_path, seed=1)
    report = check_inputs(meta, inputs)
    guard_inputs(report, "x", "re-run stage 2", on_stale=policy)       # must not raise


def test_cheap_stages_rebuild_and_expensive_ones_stop():
    stale = StaleReport(status="stale", changed=["oof_meta_X"],
                        details=["oof_meta_X changed"])
    assert resolve_stale(stale, "meta-learner", cheap=True) is True
    with pytest.raises(ValueError, match="expensive"):
        resolve_stale(stale, "base models", cheap=False)


def test_force_rebuilds_and_fresh_results_are_reused():
    fresh = StaleReport(status="match")
    assert resolve_stale(fresh, "x") is False
    assert resolve_stale(fresh, "x", force=True) is True


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="on_stale must be"):
        resolve_stale(StaleReport(status="match"), "x", on_stale="maybe")


# ---------------------------------------------------------------------------
# The fingerprint survives a JSON round trip, as it must in metadata files
# ---------------------------------------------------------------------------

def test_fingerprint_survives_json(tmp_path):
    inputs = _oof(tmp_path)
    meta = record_inputs({"n_models": 3}, inputs)
    path = tmp_path / "meta_learner_metadata.json"
    path.write_text(json.dumps(meta, indent=2, default=str))

    reloaded = json.loads(path.read_text())
    assert check_inputs(reloaded, inputs).status == "match"
    _oof(tmp_path, seed=9)
    assert check_inputs(reloaded, inputs).status == "stale"
