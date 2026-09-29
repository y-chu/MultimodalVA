"""Parallel fused-text generation must be efficient and fail transparently."""

from __future__ import annotations

import pandas as pd
import pytest

from multimodalva.ensemble.data_fusion import build_fused_text


def _frame(n=8):
    return pd.DataFrame({
        "narrative": [f"case {i}" for i in range(n)],
        "fever": [i % 2 for i in range(n)],
    })


def test_worker_error_is_not_swallowed_and_repeated_sequentially(monkeypatch):
    import joblib

    class BrokenParallel:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, jobs):
            raise ValueError("worker rendering failed")

    monkeypatch.setattr(joblib, "Parallel", BrokenParallel)
    with pytest.raises(ValueError, match="worker rendering failed"):
        build_fused_text(
            _frame(), text_col="narrative", feature_cols=["fever"],
            qdesc=pd.DataFrame(), n_jobs=2,
        )


def test_auto_parallelism_stays_serial_for_small_data(monkeypatch):
    import joblib

    def should_not_run(*args, **kwargs):
        raise AssertionError("small automatic jobs should not launch workers")

    monkeypatch.setenv("MULTIMODALVA_FUSION_N_JOBS", "8")
    monkeypatch.setattr(joblib, "Parallel", should_not_run)
    out = build_fused_text(
        _frame(), text_col="narrative", feature_cols=["fever"],
        qdesc=pd.DataFrame(), n_jobs=None,
    )
    assert len(out) == 8
