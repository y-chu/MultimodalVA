"""Storage-efficiency regression tests for text HPO + training.

These guard the invariants from the storage-efficiency work without running real
model training (no torch/transformers compute needed beyond import):

  - train() defaults: cleanup_checkpoints=True, save_total_limit=2, report_to="none".
  - _remove_checkpoint_dirs() deletes only checkpoint-*/ and keeps the final model
    + tokenizer + label maps that predict() reloads.
  - _find_latest_checkpoint() still resolves the latest checkpoint (resume needs it
    to survive DURING a run — cleanup happens only after completion).
  - HPO trials leave no weights: _trial_workdir(None) is a tempdir deleted on exit;
    _trial_workdir(path) persists. optimize()/optimize_ray() default
    export_best_trial=False, cleanup_trials=True.
  - data() example loaders work and reproduce committed sample CSVs.

Run: pytest tests/test_storage_efficiency.py -q
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest


# --- helpers ---------------------------------------------------------------
def _make_fake_run_dir(root: Path) -> Path:
    """Create a dir that looks like a completed train() output_dir."""
    root.mkdir(parents=True, exist_ok=True)
    # final model artifacts at root (what predict() reloads)
    (root / "config.json").write_text("{}")
    (root / "model.safetensors").write_bytes(b"\x00" * 16)
    (root / "tokenizer_config.json").write_text("{}")
    (root / "vocab.txt").write_text("[PAD]\n")
    (root / "label2id.json").write_text(json.dumps({"A": 0, "B": 1}))
    (root / "id2label.json").write_text(json.dumps({"0": "A", "1": "B"}))
    # intermediate checkpoints (the bloat)
    for step in (50, 100):
        ck = root / f"checkpoint-{step}"
        ck.mkdir()
        (ck / "model.safetensors").write_bytes(b"\x00" * 32)
        (ck / "optimizer.pt").write_bytes(b"\x00" * 64)
    # a non-checkpoint dir that must NOT be touched
    (root / "predictions").mkdir()
    (root / "predictions" / "predictions_top1.csv").write_text("true_label\nA\n")
    return root


# --- train.py defaults -----------------------------------------------------
def test_train_storage_defaults():
    from multimodalva.text.train import train

    params = inspect.signature(train).parameters
    assert params["cleanup_checkpoints"].default is True
    assert params["save_total_limit"].default == 2
    assert params["report_to"].default == "none"


# --- checkpoint cleanup keeps the reloadable final model -------------------
def test_remove_checkpoint_dirs_keeps_final_model(tmp_path):
    from multimodalva.text.train import _remove_checkpoint_dirs

    run = _make_fake_run_dir(tmp_path / "final")
    removed = _remove_checkpoint_dirs(run)

    assert removed == 2
    # checkpoints gone
    assert not list(run.glob("checkpoint-*"))
    # everything predict() needs survives
    for keep in ("config.json", "model.safetensors", "tokenizer_config.json",
                 "vocab.txt", "label2id.json", "id2label.json"):
        assert (run / keep).exists(), keep
    # unrelated dirs untouched
    assert (run / "predictions" / "predictions_top1.csv").exists()


def test_remove_checkpoint_dirs_missing_dir_is_noop(tmp_path):
    from multimodalva.text.train import _remove_checkpoint_dirs
    assert _remove_checkpoint_dirs(tmp_path / "does_not_exist") == 0


# --- resume invariant: latest checkpoint resolvable before cleanup ---------
def test_find_latest_checkpoint_then_cleanup(tmp_path):
    from multimodalva.text.train import _find_latest_checkpoint, _remove_checkpoint_dirs

    run = _make_fake_run_dir(tmp_path / "final")
    # During an interrupted run the latest checkpoint must be found (resume).
    latest = _find_latest_checkpoint(run)
    assert latest is not None and latest.name == "checkpoint-100"
    # After a successful run cleanup removes them; resume then finds none.
    _remove_checkpoint_dirs(run)
    assert _find_latest_checkpoint(run) is None


# --- HPO trial workdir: tempdir deleted, persistent kept -------------------
def test_trial_workdir_tempdir_is_deleted():
    from multimodalva.text.hpo import _trial_workdir

    seen: Path | None = None
    with _trial_workdir(None) as d:
        seen = d
        (d / "model.safetensors").write_bytes(b"\x00")
        assert d.exists()
    # tempdir (and any weights written inside) removed on exit
    assert seen is not None and not seen.exists()


def test_trial_workdir_persistent_is_kept(tmp_path):
    from multimodalva.text.hpo import _trial_workdir

    target = tmp_path / "trial_0" / "fold_0"
    with _trial_workdir(target) as d:
        assert d == target and d.exists()
    assert target.exists()  # persisted for best-trial export


# --- HPO signature defaults ------------------------------------------------
def test_optimize_storage_defaults():
    from multimodalva.text.hpo import optimize, optimize_ray

    op = inspect.signature(optimize).parameters
    assert op["export_best_trial"].default is False
    assert op["cleanup_trials"].default is True
    rp = inspect.signature(optimize_ray).parameters
    assert rp["export_best_trial"].default is False
    assert rp["cleanup_trials"].default is True


# --- example data loaders --------------------------------------------------
def test_data_loader_known_and_unknown():
    from multimodalva import data, list_datasets

    names = list_datasets()
    assert {"va_sample", "va_who2016", "va_sample_text_only", "va_demo"} <= set(names)

    df = data("va_sample")
    assert {"id", "cause_of_death", "narrative"} <= set(df.columns)
    assert (df["i019a"].isin(["y", "n"])).all()  # i-code y/n coding

    who = data("va_who2016")
    assert "Id10147" in who.columns and (who["Id10147"].isin(["yes", "no"])).all()

    assert list(data("va_sample_text_only").columns) == ["id", "cause_of_death", "narrative"]

    with pytest.raises(ValueError):
        data("not_a_dataset")


def test_data_loader_reproduces_committed_csv():
    from multimodalva import data

    committed_path = Path(__file__).parent / "sample_data" / "va_sample.csv"
    if not committed_path.exists():
        pytest.skip("committed va_sample.csv not present")
    committed = pd.read_csv(committed_path)
    # committed CSV is generated with n_per_class=6 (see demo_hub sample-data default)
    regenerated = data("va_sample", n_per_class=6)
    pd.testing.assert_frame_equal(committed, regenerated)


def test_import_multimodalva_is_light():
    """`import multimodalva` (for data()) must not pull the transformer stack."""
    import subprocess
    import sys

    code = (
        "import sys, multimodalva; "
        "from multimodalva import data; "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules; "
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
