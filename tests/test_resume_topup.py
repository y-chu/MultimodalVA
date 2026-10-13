"""Raising ``n_trials`` on a finished run tops its search up.

The resume manifest used to record ``Optimize.n_trials``, so asking a finished
run for more trials changed its signature and was refused -- by ``resume=True``
and ``resume="adopt"`` alike. The trial budget says how far a search has got,
not what it searches, and the default Optuna study name already leaves it out.

What makes dropping it safe differs per pipeline, so each test pins one side:
pipelines that refit after every search ignore the budget; pipelines that keep
work built from the best hyperparameters still refuse it; a manifest written
while the budget was recorded is still accepted; a real difference is still
refused; and a final-fit checkpoint trained under other hyperparameters is not
resumed under the new ones.
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

import multimodalva as mv
from multimodalva import Optimize
from multimodalva.utils.provenance import (
    TOPUP_PIPELINES, _jsonable, _sign, enforce_resume_manifest,
    make_resume_manifest,
)

LABEL = "cause_of_death"


def _manifest(pipeline, n_trials, **config):
    frame = pd.DataFrame({"a": [1, 2, 3]})
    return make_resume_manifest(
        dataframes={"train": frame},
        config={"pipeline": pipeline, "max_length": 512,
                "hyperparams": Optimize(n_trials=n_trials), **config},
    )


@pytest.mark.parametrize("pipeline", sorted(TOPUP_PIPELINES))
def test_trial_budget_is_not_part_of_a_refitting_pipelines_signature(pipeline):
    assert _manifest(pipeline, 15)["signature"] == _manifest(pipeline, 29)["signature"]


@pytest.mark.parametrize("pipeline", ["feature_fusion", "stacking_stage1", "voting"])
def test_trial_budget_still_counts_where_work_is_cached_downstream(pipeline):
    assert _manifest(pipeline, 15)["signature"] != _manifest(pipeline, 29)["signature"]


def _legacy(manifest, n_trials):
    """``manifest`` as written 2026-09-28..10-02, with the budget recorded."""
    config = manifest["config"]
    config = {**config, "hyperparams": {**config["hyperparams"], "n_trials": n_trials}}
    core = {"schema": manifest["schema"], "dataframes": manifest["dataframes"],
            "config": config}
    return {**core, "signature": _sign(core), "recorded_at": "2026-09-30T00:00:00+00:00"}


@pytest.mark.parametrize("resume", [True, "adopt"])
def test_a_manifest_that_recorded_the_budget_can_be_topped_up(tmp_path, resume):
    path = tmp_path / "resume_manifest.json"
    path.write_text(json.dumps(_legacy(_manifest("text", 15), 15)))
    expected = _manifest("text", 29)
    enforce_resume_manifest(path, expected, resume=resume,
                            artifacts_exist=True, what="test")
    assert json.loads(path.read_text())["signature"] == expected["signature"]


def test_a_real_difference_in_a_legacy_manifest_is_still_refused(tmp_path):
    path = tmp_path / "resume_manifest.json"
    path.write_text(json.dumps(_legacy(_manifest("text", 15, max_length=1024), 15)))
    for resume in (True, "adopt"):
        with pytest.raises(RuntimeError, match="configuration differs"):
            enforce_resume_manifest(path, _manifest("text", 29), resume=resume,
                                    artifacts_exist=True, what="test")


def test_fixed_hyperparams_are_compared_in_full():
    """Only an ``Optimize`` is budget-blind; a fixed dict is a setting."""
    frame = pd.DataFrame({"a": [1]})
    sig = lambda hp: make_resume_manifest(   # noqa: E731
        dataframes={"t": frame}, config={"pipeline": "text", "hyperparams": hp},
    )["signature"]
    assert sig({"epochs": 3, "n_trials": 1}) != sig({"epochs": 3, "n_trials": 2})


def test_a_finished_tabular_run_tops_its_search_up(tmp_path):
    """End to end: a second call with a bigger budget adds trials, nothing else."""
    df = mv.data("va_sample", n_per_class=10)
    feats = [c for c in df.columns if c not in (LABEL, "narrative", "id")]
    cfg = dict(task="tabular", data=df, label_col=LABEL, features=feats,
               model="naive_bayes", output_dir=tmp_path)
    mv.run(hyperparams=Optimize(n_trials=2, cv=False), **cfg)
    mv.run(hyperparams=Optimize(n_trials=4, cv=False), **cfg)
    trials = pd.read_csv(next(tmp_path.rglob("hpo_trials.csv")))
    assert (trials["state"] == "COMPLETE").sum() == 4


# ---------------------------------------------------------------------------
# The final fit's checkpoints
# ---------------------------------------------------------------------------
TINY = "prajjwal1/bert-tiny"


@pytest.fixture(scope="module")
def tiny_dataset():
    pytest.importorskip("transformers")
    from huggingface_hub import try_to_load_from_cache
    if not isinstance(try_to_load_from_cache(TINY, "config.json"), str):
        pytest.skip(f"{TINY} is not in the local Hugging Face cache")
    from multimodalva.text.dataset import prepare_text_dataset
    df = mv.data("va_sample", n_per_class=4)
    return prepare_text_dataset(df, df.head(4), text_col="narrative",
                                label_col=LABEL, model_name=TINY, max_length=32)


def _fit(dataset, out, hp, resume):
    from multimodalva.text.train import train_text
    train, _, label2id, id2label = dataset
    train_text(train_dataset=train, label2id=label2id, id2label=id2label,
               model_name=TINY, output_dir=out, val_size=None, resume=resume,
               cleanup_checkpoints=False,
               hyperparams={"epochs": 1, "batch_size": 8, **hp})


def test_a_checkpoint_resumes_under_the_hyperparameters_it_was_trained_with(
        tiny_dataset, tmp_path, caplog):
    _fit(tiny_dataset, tmp_path, {"learning_rate": 1e-4}, resume=False)
    assert list(tmp_path.glob("checkpoint-*"))
    with caplog.at_level(logging.INFO):
        _fit(tiny_dataset, tmp_path, {"learning_rate": 1e-4}, resume=True)
    assert "Resuming from checkpoint" in caplog.text
    assert not list(tmp_path.glob("stale_checkpoints_*"))


def test_a_checkpoint_from_other_hyperparameters_is_set_aside(
        tiny_dataset, tmp_path, caplog):
    _fit(tiny_dataset, tmp_path, {"learning_rate": 1e-4}, resume=False)
    before = sorted(p.name for p in tmp_path.glob("checkpoint-*"))
    with caplog.at_level(logging.INFO):
        _fit(tiny_dataset, tmp_path, {"learning_rate": 3e-4}, resume=True)
    assert "Resuming from checkpoint" not in caplog.text
    (aside,) = tmp_path.glob("stale_checkpoints_*")
    assert sorted(p.name for p in aside.iterdir()) == before


def test_a_checkpoint_with_no_record_is_not_trusted(tiny_dataset, tmp_path, caplog):
    _fit(tiny_dataset, tmp_path, {"learning_rate": 1e-4}, resume=False)
    (tmp_path / "checkpoint_hyperparams.json").unlink()   # a pre-guard checkpoint
    with caplog.at_level(logging.INFO):
        _fit(tiny_dataset, tmp_path, {"learning_rate": 1e-4}, resume=True)
    assert "Resuming from checkpoint" not in caplog.text
    assert list(tmp_path.glob("stale_checkpoints_*"))
