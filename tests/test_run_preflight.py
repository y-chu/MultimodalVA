"""What ``preflight()`` catches, and what it deliberately does not.

A preflight exists for one reason: on a cluster a mistyped column name costs a
queue wait, a job start and a pipeline import before anything says so. These
tests pin down each check, and — more importantly — pin the two directions that
keep the checker honest:

* a config ``preflight`` accepts really does get past the same point in
  ``run()`` (:func:`test_a_clean_preflight_agrees_with_a_real_run`), and
* a config it rejects really does fail there
  (:func:`test_a_rejected_config_really_does_fail`).

Without that pair a preflight can drift into either uselessness (accepts
everything) or noise (rejects things that work), and neither shows up in a
per-check test.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest import mock

import pytest

import multimodalva as mv
from multimodalva import Optimize
from multimodalva.runner import SUPPORTED_TASKS, _kwarg_targets, run

LABEL = "cause_of_death"
TEXT = "narrative"


@pytest.fixture(scope="module")
def df():
    return mv.data("va_sample", n_per_class=8)


@pytest.fixture(scope="module")
def csv(df, tmp_path_factory):
    path = tmp_path_factory.mktemp("pf") / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def _pf(**kw):
    """A tabular preflight with the required arguments already filled in."""
    kw.setdefault("task", "tabular")
    kw.setdefault("label_col", LABEL)
    return mv.preflight(**kw)


def _problems(report) -> str:
    return "\n".join(report["problems"])


def _notes(report) -> str:
    return "\n".join(report["notes"])


# ---------------------------------------------------------------------------
# A clean config
# ---------------------------------------------------------------------------
def test_a_clean_tabular_config_passes_and_describes_the_run(csv):
    r = _pf(data=csv, model="lightgbm", features="auto")
    assert r["ok"], _problems(r)
    assert r["task"] == "tabular"
    assert r["facts"]["pipeline"] == "TabularClassifier"
    assert "88" in r["facts"]["rows"]          # 11 causes x 8 rows
    assert r["facts"]["classes"] == "11"
    assert "test_size=0.2" in r["facts"]["split"]
    assert "resolved" in r["facts"]["feature columns"]


def test_a_clean_text_config_passes(csv):
    r = mv.preflight(task="text", data=csv, label_col=LABEL, text_col=TEXT,
                     model="bluebert")
    assert r["ok"], _problems(r)
    assert r["facts"]["pipeline"] == "TextClassifier"
    assert "bionlp/bluebert" in r["facts"]["model"]


def test_an_in_memory_frame_works_as_well_as_a_path(df, csv):
    from_frame = _pf(data=df, features="auto")
    from_path = _pf(data=csv, features="auto")
    assert from_frame["ok"] and from_path["ok"]
    assert from_frame["facts"]["rows"] == from_path["facts"]["rows"]


def test_preflight_writes_nothing(csv, tmp_path):
    out = tmp_path / "never"
    r = _pf(data=csv, features="auto", output_dir=out)
    assert r["ok"], _problems(r)
    assert not out.exists(), "a dry run created its output directory"


def test_preflight_does_not_touch_the_callers_frame(df):
    before = df.copy()
    _pf(data=df, features="auto", split=None)
    assert list(df.columns) == list(before.columns)
    assert df.equals(before)


# ---------------------------------------------------------------------------
# The task and the arguments
# ---------------------------------------------------------------------------
def test_an_unknown_task_is_reported_and_stops_the_rest(csv):
    r = _pf(task="tabluar", data=csv)
    assert not r["ok"]
    assert "Unknown task" in _problems(r)
    assert r["task"] is None


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
def test_every_supported_task_resolves_a_destination_class(task):
    """A task added to SUPPORTED_TASKS must be reachable by the keyword check.

    Otherwise the check silently stops applying to it and the typos it exists to
    catch go through again.
    """
    from multimodalva.runner import _normalize_task

    ctor, run_params, name = _kwarg_targets(_normalize_task(task))
    assert name, task
    assert ctor is not None or run_params is not None, (
        f"{name} accepts **kwargs on both __init__ and run, so no keyword can "
        "be checked for task={task!r}"
    )


def test_a_misspelt_keyword_is_named_with_a_suggestion(csv):
    r = mv.preflight(task="text", data=csv, label_col=LABEL, text_col=TEXT,
                     max_lenght=512)
    assert not r["ok"]
    assert "max_lenght= is not accepted by TextClassifier" in _problems(r)
    assert "Did you mean max_length=?" in _problems(r)


def test_a_keyword_run_supplies_itself_is_reported_as_a_collision(csv):
    r = mv.preflight(task="text", data=csv, label_col=LABEL, text_col=TEXT,
                     model_name="bluebert")
    assert not r["ok"]
    assert "model_name= is computed by run() itself" in _problems(r)


@pytest.mark.parametrize("name", sorted(
    set(inspect.signature(run).parameters) - {"run_kwargs", "task"}))
def test_no_named_argument_of_run_is_called_unknown(csv, name):
    """Every parameter run() names must pass the keyword check.

    The check reads run()'s signature rather than a copied list, so this test is
    what proves the two cannot disagree after someone adds an argument.
    """
    cfg = {"task": "tabular", "data": csv, "label_col": LABEL, name: None}
    r = mv.preflight(**cfg)
    assert "is not accepted by" not in _problems(r), (
        f"run() names {name}= but preflight calls it unknown")


def test_a_removed_seed_argument_names_its_replacement(csv):
    r = _pf(data=csv, random_state=1)
    assert not r["ok"]
    assert "split_seed" in _problems(r) and "train_seed" in _problems(r)


def test_resume_training_names_its_replacement(csv):
    r = _pf(data=csv, resume_training=True)
    assert not r["ok"]
    assert "resume_training= is now resume=" in _problems(r)


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
def test_a_missing_label_column_is_reported(csv):
    r = _pf(data=csv, label_col="cause")
    assert not r["ok"]
    assert "label column 'cause' not in DataFrame" in _problems(r)


def test_a_missing_text_column_is_reported(csv):
    """The gap this closes: the missing-row drop only looks at text_col when it
    is present, so an absent one reaches the pipeline before failing."""
    r = mv.preflight(task="text", data=csv, label_col=LABEL, text_col="narrativ")
    assert not r["ok"]
    assert "text column 'narrativ' is not in the data" in _problems(r)


def test_a_missing_data_file_is_reported():
    r = _pf(data="/nonexistent/nowhere.csv")
    assert not r["ok"]
    assert "data file not found" in _problems(r)


def test_data_and_label_col_are_required(csv):
    r = mv.preflight(task="tabular")
    assert not r["ok"]
    assert "needs data= and label_col=" in _problems(r)


def test_stacking_stage_two_needs_no_data(tmp_path):
    prior = tmp_path / "stage1"
    prior.mkdir()
    r = mv.preflight(task="stacking", oof_from=str(prior))
    assert "needs data= and label_col=" not in _problems(r)
    assert "rows, split and seeds come from" in r["facts"]["stage 2 only"]


def test_a_missing_oof_from_is_reported(tmp_path):
    r = mv.preflight(task="stacking", oof_from=str(tmp_path / "absent"))
    assert not r["ok"]
    assert "does not exist" in _problems(r)


def test_a_bad_filter_column_is_reported(csv):
    r = _pf(data=csv, filters={"site": "A"})
    assert not r["ok"]
    assert "filter column 'site' not in DataFrame" in _problems(r)


def test_a_filter_value_that_matches_nothing_is_named(csv):
    """"No rows remain" names the step that noticed, not the value that emptied
    the frame. The note is what makes the failure actionable."""
    r = _pf(data=csv, filters={LABEL: "Malaria"})
    assert not r["ok"]
    assert "match no row" in _notes(r)
    assert "Assault" in _notes(r), "the note should show what the column holds"


def test_the_filter_row_count_is_captured_from_the_runs_own_log(csv):
    r = _pf(data=csv, features="auto", filters={LABEL: ["Assault", "Diabetes"]})
    assert r["ok"], _problems(r)
    assert any("Filters" in line and "16 rows" in line for line in r["log"]), r["log"]


def test_a_supplied_split_is_reported_by_its_two_sides(df, tmp_path):
    train = df.iloc[: len(df) // 2]
    test = df.iloc[len(df) // 2 :]
    r = _pf(data=(train, test), features="auto")
    assert r["ok"], _problems(r)
    assert r["facts"]["split"] == f"supplied: {len(train)} train / {len(test)} test"


# ---------------------------------------------------------------------------
# The features
# ---------------------------------------------------------------------------
def test_a_regex_matching_no_column_is_reported(csv):
    r = _pf(data=csv, features="re:^zzz")
    assert not r["ok"]
    assert "matched no columns" in _problems(r)


def test_a_named_feature_column_that_is_absent_is_reported(csv):
    r = _pf(data=csv, features=["i019a", "not_a_column"])
    assert not r["ok"]
    assert "not_a_column" in _problems(r)


def test_the_id_column_being_a_feature_is_reported(csv):
    r = _pf(data=csv, features=["id", "i019a"], id_col="id")
    assert not r["ok"]
    assert "carries no information" in _problems(r)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def test_an_unknown_tabular_model_is_named_with_a_suggestion(csv):
    r = _pf(data=csv, model="lightgmb")
    assert not r["ok"]
    assert "is not a tabular model" in _problems(r)
    assert "Did you mean 'lightgbm'?" in _problems(r)


def test_an_ambiguous_text_model_is_reported(csv):
    r = mv.preflight(task="text", data=csv, label_col=LABEL, text_col=TEXT,
                     model="bioroberta")
    assert not r["ok"]
    assert "ambiguous" in _problems(r)


def test_a_hub_id_is_accepted_without_network_access(csv):
    r = mv.preflight(task="text", data=csv, label_col=LABEL, text_col=TEXT,
                     model="emilyalsentzer/Bio_ClinicalBERT")
    assert r["ok"], _problems(r)
    assert "cannot be verified" in r["facts"]["model"]


def test_a_remote_model_is_reported_as_a_pending_download_not_fetched(csv):
    with mock.patch("multimodalva.text.models.download_model") as download:
        r = mv.preflight(task="text", data=csv, label_col=LABEL, text_col=TEXT,
                         model="roberta-pm")
    assert r["ok"], _problems(r)
    # A preflight that pulls half a gigabyte is no longer a preflight.
    download.assert_not_called()
    assert "downloaded to" in r["facts"]["model"]


def test_a_near_miss_text_model_in_a_base_spec_is_flagged_as_a_note(csv):
    """A bare unknown name cannot be an error — bert-base-uncased is a real Hub
    id with no slash — but a close alias is worth saying out loud."""
    r = mv.preflight(task="voting", data=csv, label_col=LABEL, text_col=TEXT,
                     text_models=[{"model_name": "blubert"}],
                     tabular_models=[{"model_name": "lightgbm"}])
    assert "did you mean 'bluebert'?" in _notes(r)


def test_an_unknown_base_model_is_reported(csv):
    r = mv.preflight(task="voting", data=csv, label_col=LABEL, text_col=TEXT,
                     text_models=[{"model_name": "bert"}],
                     tabular_models=[{"model_name": "catbost"}])
    assert not r["ok"]
    assert "tabular_models:" in _problems(r)
    assert "Did you mean 'catboost'?" in _problems(r)


def test_a_base_spec_without_a_model_name_is_reported(csv):
    r = mv.preflight(task="voting", data=csv, label_col=LABEL, text_col=TEXT,
                     text_models=[{"hyperparams": {}}],
                     tabular_models=[{"model_name": "lightgbm"}])
    assert not r["ok"]
    assert "has no model_name" in _problems(r)


def test_an_unknown_base_spec_key_is_rejected_instead_of_ignored(csv):
    r = mv.preflight(
        task="voting", data=csv, label_col=LABEL, features="auto",
        tabular_models=[
            {"model_name": "lightgbm", "max_dept": 3},
            {"model_name": "naive_bayes"},
        ],
    )
    assert not r["ok"]
    assert "max_dept" in _problems(r)
    assert "would otherwise have no effect" in _problems(r)


def test_voting_requires_two_base_models(csv):
    r = mv.preflight(
        task="voting", data=csv, label_col=LABEL, features="auto",
        tabular_models=[{"model_name": "lightgbm"}],
    )
    assert not r["ok"]
    assert "at least 2 base models" in _problems(r)


def test_stacking_rejects_the_unfinished_insilicova_hook(csv):
    r = mv.preflight(
        task="stacking", data=csv, label_col=LABEL, features="auto",
        tabular_models=[{"model_name": "insilicova"}],
    )
    assert not r["ok"]
    assert "insilicova" in _problems(r)


def test_a_text_task_without_text_col_is_reported(csv):
    r = mv.preflight(task="data_fusion", data=csv, label_col=LABEL)
    assert not r["ok"]
    assert "requires text_col" in _problems(r)


# ---------------------------------------------------------------------------
# init_kwargs
# ---------------------------------------------------------------------------
def test_a_misspelt_init_kwarg_is_named_with_a_suggestion(csv):
    r = mv.preflight(task="stacking", data=csv, label_col=LABEL, text_col=TEXT,
                     tabular_models=[{"model_name": "lightgbm"}],
                     init_kwargs={"n_fold": 5})
    assert not r["ok"]
    assert "init_kwargs['n_fold'] is not accepted by StackingClassifier" in _problems(r)
    assert "Did you mean 'n_folds'?" in _problems(r)


def test_init_kwargs_silently_overriding_a_named_argument_is_a_note(csv):
    r = mv.preflight(task="stacking", data=csv, label_col=LABEL, text_col=TEXT,
                     text_models=[{"model_name": "bert"}],
                     tabular_models=[{"model_name": "lightgbm"}],
                     init_kwargs={"text_models": [{"model_name": "biobert"}]})
    assert "init_kwargs wins" in _notes(r)


def test_a_valid_init_kwarg_passes(csv):
    r = mv.preflight(task="feature_fusion", data=csv, label_col=LABEL,
                     text_col=TEXT, features="auto",
                     init_kwargs={"fusion_strategy": "concat"})
    assert r["ok"], _problems(r)


def test_constructor_only_setting_must_be_inside_init_kwargs(csv):
    r = mv.preflight(
        task="stacking", data=csv, label_col=LABEL, features="auto",
        tabular_models=[{"model_name": "lightgbm"}], n_folds=3,
    )
    assert not r["ok"]
    assert "constructor setting" in _problems(r)
    assert "init_kwargs={'n_folds': ...}" in _problems(r)


def test_feature_fusion_single_modality_strategies_need_only_their_inputs(csv):
    tabular = mv.preflight(
        task="feature_fusion", data=csv, label_col=LABEL, features="auto",
        init_kwargs={"fusion_strategy": "tabular_only"},
    )
    assert tabular["ok"], _problems(tabular)

    text = mv.preflight(
        task="feature_fusion", data=csv, label_col=LABEL, text_col=TEXT,
        init_kwargs={"fusion_strategy": "text_only"},
    )
    assert text["ok"], _problems(text)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
def test_no_search_is_said_plainly(csv):
    assert "no search" in _pf(data=csv, features="auto")["facts"]["hyperparameters"]
    fixed = _pf(data=csv, features="auto", hyperparams={"n_estimators": 10})
    assert "fixed" in fixed["facts"]["hyperparameters"]
    assert "no search" in fixed["facts"]["hyperparameters"]


def test_a_search_reports_its_metric_budget_and_split(csv):
    r = _pf(data=csv, features="auto",
            hyperparams=Optimize(n_trials=7, metric="csmf_accuracy", cv_folds=4))
    assert r["ok"], _problems(r)
    assert "metric=csmf_accuracy" in r["facts"]["hyperparameters"]
    assert "n_trials=7" in r["facts"]["hyperparameters"]
    assert r["facts"]["search split"] == "4-fold cross-validation"


def test_a_partial_search_space_says_the_rest_is_still_searched(csv):
    """The behaviour a user must not have to guess: keys they gave are used as
    passed, and the adaptive space supplies the rest."""
    r = _pf(data=csv, features="auto", model="lightgbm",
            hyperparams=Optimize(n_trials=3,
                                 space={"learning_rate": ("float_log", 1e-3, 0.3)}))
    space = r["facts"]["search space"]
    assert "1 key(s) given and used exactly as passed" in space
    assert "learning_rate" in space
    assert "every other key of the adaptive space" in space


def test_an_empty_search_space_says_the_default_is_used(csv):
    r = _pf(data=csv, features="auto", hyperparams=Optimize(n_trials=3))
    assert r["facts"]["search space"] == "the package default, adapted to this data"


def test_more_folds_than_the_rarest_training_class_is_an_error(csv):
    """Preflight reports this through the searches' own resolver, so the two
    cannot phrase — or decide — it differently."""
    r = _pf(data=csv, features="auto", hyperparams=Optimize(n_trials=2, cv_folds=20))
    assert not r["ok"]
    problems = _problems(r)
    assert "n_cv_folds=20 is greater than the minimum class count" in problems
    assert 'cv_folds="auto"' in problems, "the error should name the way out"


def test_a_single_holdout_search_with_a_singleton_class_is_an_error(df, tmp_path):
    """The outer stratified train/test split cannot place one row on both sides."""
    thin = df.copy()
    thin.loc[thin.index[0], LABEL] = "a cause seen once"
    r = _pf(data=thin, features="auto", hyperparams=Optimize(n_trials=2, cv=False))
    assert not r["ok"]
    assert "at least 2 rows per class" in _problems(r)


def test_an_explicit_ray_backend_without_ray_is_an_error(csv):
    """An explicit backend='ray' is never downgraded — it raises. Better here
    than after the job has queued."""
    with mock.patch("multimodalva.utils.optimize_config.ray_is_available",
                    return_value=False):
        r = _pf(data=csv, features="auto",
                hyperparams=Optimize(n_trials=2, backend="ray"))
    assert not r["ok"]
    assert "never downgraded" in _problems(r)


def test_the_resolved_backend_is_reported_alongside_what_was_asked_for(csv):
    r = _pf(data=csv, features="auto",
            hyperparams=Optimize(n_trials=2, backend="optuna"))
    assert r["facts"]["hpo backend"] == "optuna"


# ---------------------------------------------------------------------------
# Several problems at once
# ---------------------------------------------------------------------------
def test_one_report_names_every_problem(csv):
    """A checker that stops at the first error makes the user resubmit once per
    mistake, which is the cost it exists to avoid."""
    r = mv.preflight(task="text", data=csv, label_col=LABEL,
                     text_col="narrativ", model="bioroberta", max_lenght=512)
    assert not r["ok"]
    assert len(r["problems"]) >= 3, r["problems"]
    joined = _problems(r)
    assert "max_lenght=" in joined          # the keyword
    assert "ambiguous" in joined            # the model
    assert "narrativ" in joined             # the column


def test_a_stage_whose_input_is_broken_does_not_report_downstream_noise(csv):
    """With no usable label column there is no frame to check columns against,
    so the column checks stay quiet instead of guessing."""
    r = mv.preflight(task="text", data=csv, label_col="cause", text_col="narrativ")
    assert not r["ok"]
    assert "label column 'cause' not in DataFrame" in _problems(r)
    assert "text column" not in _problems(r)


# ---------------------------------------------------------------------------
# Preflight and run() must agree
# ---------------------------------------------------------------------------
def test_a_clean_preflight_agrees_with_a_real_run(df, tmp_path):
    cfg = dict(task="tabular", data=df, label_col=LABEL, features="auto",
               model="naive_bayes", hyperparams={})
    assert mv.preflight(**cfg)["ok"]
    result = mv.run(output_dir=tmp_path / "real", **cfg)
    assert result["predictions"] is not None


def test_a_rejected_config_really_does_fail(df, tmp_path):
    cfg = dict(task="tabular", data=df, label_col=LABEL, features="re:^zzz",
               model="naive_bayes")
    assert not mv.preflight(**cfg)["ok"]
    with pytest.raises(ValueError, match="matched no columns"):
        mv.run(output_dir=tmp_path / "real", **cfg)


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------
def _cli(*argv) -> int:
    from multimodalva.cli import main

    return main(["run", *argv])


def test_the_cli_dry_run_exits_zero_and_writes_nothing(csv, tmp_path, capsys):
    out = tmp_path / "never"
    assert _cli("--task", "tabular", "--data", csv, "--label-col", LABEL,
                "--model", "lightgbm", "--output-dir", str(out), "--dry-run") == 0
    assert not out.exists()
    printed = capsys.readouterr().out
    assert "Dry run" in printed
    assert "does not promise" in printed, "the report must state its own limits"


def test_the_cli_dry_run_exits_two_on_a_bad_config(csv, tmp_path, capsys):
    assert _cli("--task", "tabular", "--data", csv, "--label-col", "cause",
                "--output-dir", str(tmp_path / "never"), "--dry-run") == 2
    assert "error:" in capsys.readouterr().err


def test_the_dry_run_flag_does_not_reach_run(csv):
    """--dry-run is a CLI concern; run() has no such argument, so the flag must
    not survive the translation from the command line into a run() config."""
    from multimodalva.cli import _build_parser, _cli_overrides

    assert "dry_run" not in inspect.signature(run).parameters
    args = _build_parser().parse_args(
        ["run", "--task", "tabular", "--data", csv, "--label-col", LABEL,
         "--dry-run"])
    assert args.dry_run is True
    assert "dry_run" not in _cli_overrides(args)


def test_cli_stacking_oof_from_needs_no_data_or_label(tmp_path, monkeypatch):
    """The CLI must expose the same stage-two-only contract as the Python API."""
    source = tmp_path / "finished-stage-one"
    source.mkdir()
    received = {}

    def fake_run(**kwargs):
        received.update(kwargs)

    monkeypatch.setattr("multimodalva.runner.run", fake_run)
    assert _cli(
        "--task", "stacking",
        "--oof-from", str(source),
        "--output-dir", str(tmp_path / "stage-two"),
    ) == 0
    assert received["oof_from"] == str(source)
    assert "data" not in received and "label_col" not in received


# ---------------------------------------------------------------------------
# Keywords run() routes to the constructor itself
# ---------------------------------------------------------------------------
def test_stacking_accepts_a_flat_resume_keyword(df, tmp_path):
    """``resume=`` is the documented spelling for every task, stacking included.

    Stacking is the one pipeline that keeps ``resume`` on the object rather than
    on ``run()``, so ``_dispatch`` routes it to the constructor. That routing used
    to happen *after* ``run_args`` was built from ``**run_kwargs``, which left a
    copy behind and made ``run(task="stacking", resume=False)`` — and the CLI's
    ``--no-resume`` — die with "unexpected keyword argument 'resume'". Preflight
    surfaced it; this pins both halves.
    """
    cfg = dict(task="stacking", data=df, label_col=LABEL, features="auto",
               tabular_models=[{"model_name": "naive_bayes"},
                               {"model_name": "random_forest"}],
               resume=False)
    report = mv.preflight(**cfg)
    assert report["ok"], _problems(report)
    assert "init_kwargs" not in _problems(report)
    result = mv.run(output_dir=tmp_path / "stack", init_kwargs={"n_folds": 2}, **cfg)
    assert result["predictions"] is not None


def test_the_no_resume_flag_survives_the_cli_for_stacking(csv, tmp_path, capsys):
    assert _cli("--task", "stacking", "--data", csv, "--label-col", LABEL,
                "--output-dir", str(tmp_path / "never"), "--no-resume",
                "--dry-run") == 2, "expected only the missing-base-models error"
    err = capsys.readouterr().err
    assert "resume=" not in err, f"--no-resume was reported as a mistake:\n{err}"


def test_a_genuinely_constructor_only_keyword_is_still_reported(csv):
    """The routing table must not turn the whole check off."""
    r = mv.preflight(task="stacking", data=csv, label_col=LABEL,
                     tabular_models=[{"model_name": "naive_bayes"},
                                     {"model_name": "random_forest"}],
                     n_folds=5)
    assert not r["ok"]
    assert "constructor setting of StackingClassifier" in _problems(r)
    assert "init_kwargs={'n_folds': ...}" in _problems(r)


# ---------------------------------------------------------------------------
# The shipped example configs
# ---------------------------------------------------------------------------
EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("config_*.y*ml"),
                  key=lambda p: p.name)


@pytest.mark.parametrize("config", EXAMPLES, ids=lambda p: p.name)
def test_every_shipped_config_passes_its_own_dry_run(config, capsys):
    """A config that drifts from the API is a config a new user copies.

    These run on ``data: demo``, so the check is free and catches a renamed
    argument in the files we hand people before they do.
    """
    assert _cli(str(config), "--dry-run") == 0, capsys.readouterr().err
