r"""
Command-line entry point for MultimodalVA.

Installed as the ``multimodalva`` console script (see ``[project.scripts]`` in
``pyproject.toml``). It is a thin shell over :func:`multimodalva.runner.run` and
:func:`multimodalva.inference.api.predict_from_pretrained` — every flag maps 1:1
onto an argument of those, so the CLI, the YAML config, and the Python functions
all drive the same implementation and produce the same outputs.

Examples
--------
    # Unimodal text, one line — suitable for an sbatch script
    multimodalva run --task text --data clean.csv --label-col cause \
        --text-col narrative --model bluebert --output-dir runs/bluebert --optimize

    # Unimodal tabular with a regex feature selector
    multimodalva run --task tabular --data clean.csv --label-col cause \
        --features "re:^i\d{3}[a-zA-Z]$" --model lightgbm --output-dir runs/lgbm

    # Everything from a declarative config file (YAML or JSON)
    multimodalva run experiment.yaml

    # Predict on new data with an already-trained model — one command for
    # single models and ensembles alike; what the run is is read from it
    multimodalva predict --source runs/bluebert --data new.csv \
        --text-col narrative --output-dir preds/

    # Labelled new data: performance is reported as well
    multimodalva predict --source runs/stacking --data new.csv \
        --text-col narrative --label-col cause --output-dir preds/

    # Check a config against the data before submitting it: nothing is trained
    multimodalva run experiment.yaml --dry-run

    # Check a submitted job before it spends an hour: nothing is scored
    multimodalva predict --source runs/stacking --data new.csv --dry-run

    # Helpers
    multimodalva list-datasets
    multimodalva list-models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .inference.backends import PRETRAINED_MISSING_FEATURE_METHODS
from .utils.optimize_config import BACKENDS, Optimize, optimize_from_flags


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``multimodalva`` command.

    Args:
        argv: Argument list to parse. ``None`` reads ``sys.argv``.

    Returns:
        Process exit status: 0 on success, non-zero on a usage or run error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "list-datasets":
        return _cmd_list_datasets()
    if args.command == "list-models":
        return _cmd_list_models()
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "predict":
        return _cmd_predict(args)

    parser.print_help()
    return 1


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimodalva",
        description="Cause-of-death classification from verbal autopsy data.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser(
        "run",
        help="Run a pipeline end-to-end (text/tabular/ensemble).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_p.add_argument(
        "config", nargs="?", default=None,
        help="Optional YAML/JSON config file. CLI flags override its values.",
    )
    run_p.add_argument("--task", help="text | tabular | data_fusion | "
                       "feature_fusion | voting | stacking")
    run_p.add_argument("--data", help="CSV/Parquet path (or 'demo' for the "
                       "built-in synthetic dataset).")
    run_p.add_argument("--label-col", dest="label_col",
                       help="Label (cause-of-death) column.")
    run_p.add_argument("--output-dir", dest="output_dir", help="Artifacts root.")
    run_p.add_argument("--text-col", dest="text_col", help="Narrative column.")
    run_p.add_argument("--features", help="Comma-separated feature list, "
                       "'re:PATTERN', or 'auto'.")
    run_p.add_argument("--model", help="Model name/alias/checkpoint.")
    run_p.add_argument("--filters", help="Row filter, e.g. 'age_grp3=adult,site=A'.")
    run_p.add_argument("--optimize", dest="optimize", action="store_true",
                       default=None,
                       help="Enable HPO before final training.")
    run_p.add_argument("--n-trials", dest="n_trials", type=int, help="HPO trials.")
    run_p.add_argument("--optimize-metric", dest="optimize_metric",
                       help="HPO metric (default f1_macro).")
    run_p.add_argument("--hpo-backend", dest="hpo_backend", choices=list(BACKENDS),
                       help="Search engine: 'ray' runs trials in parallel across "
                            "the GPUs or cluster nodes Ray can see, 'optuna' runs "
                            "them in one process. Default 'auto' — Ray on a "
                            "CUDA/SLURM machine, Optuna otherwise.")
    run_p.add_argument("--test-size", dest="test_size", type=float,
                       help="Random-split test fraction (default 0.2).")
    run_p.add_argument("--split-seed", dest="split_seed", type=int,
                       help="Seed for the train/test split, search folds and "
                            "early-stopping slice. Vary to measure sampling "
                            "uncertainty. Default 42.")
    run_p.add_argument("--train-seed", dest="train_seed", type=int,
                       help="Seed for model training and the search sampler. "
                            "Vary (with fixed hyperparams) to measure model "
                            "stochasticity. Default 42.")
    run_p.add_argument("--deterministic", dest="deterministic",
                       action="store_true",
                       help="Demand bit-for-bit repeatable kernels. Slower; an "
                            "op with no deterministic kernel raises.")
    run_p.add_argument("--split", help="Fixed-split column name or path to a "
                       "splits JSON ({train_ids,test_ids}).")
    run_p.add_argument("--id-col", dest="id_col", help="Row-id column for "
                       "id-based --split.")
    run_p.add_argument("--top-k", dest="top_k", type=int, help="Top-K to report.")
    run_p.add_argument("--combiner", dest="combiner",
                       help="Stacking only: how base models are combined — "
                            "meta_learner (default), simple_average, "
                            "class_aware_voting, ensemble_selection, a "
                            "meta-learner model name (e.g. lightgbm), several "
                            "comma-separated (all from one set of out-of-fold "
                            "predictions), or 'best' to choose one on training "
                            "data only. Combiner options go in a config file "
                            "(meta_learners, class_voter_kwargs, "
                            "ensemble_selection_kwargs).")
    run_p.add_argument("--no-resume", dest="no_resume", action="store_true",
                       help="Start over instead of continuing an interrupted "
                            "run in the same output directory.")
    run_p.add_argument("--oof-from", dest="oof_from",
                       help="Stacking only: output_dir of a finished stacking "
                            "run whose stage 1 to reuse; nothing is retrained.")
    run_p.add_argument("--oof-only", dest="oof_only", action="store_true",
                       default=None,
                       help="Stacking only: stop after stage 1 (the "
                            "out-of-fold predictions), before any combiner is "
                            "fitted. Continue later with --oof-from pointing "
                            "at this run.")
    run_p.add_argument("--resume-adopt", dest="resume_adopt", action="store_true",
                       help="Reuse artifacts already in --output-dir that carry "
                            "no resume signature, instead of refusing them. For "
                            "runs made before resume manifests existed: nothing "
                            "verifies they came from this data and configuration, "
                            "so this is your word, not a check. The signature of "
                            "this call is then recorded, so later runs are "
                            "checked normally.")
    run_p.add_argument("--dry-run", dest="dry_run", action="store_true",
                       help="Check the configuration against the data and stop: "
                            "load, filter, split and resolve the feature columns, "
                            "report what the run would do, train nothing. Exits 2 "
                            "if anything would stop the run, so a job script can "
                            "gate on it.")
    run_p.add_argument("--push-to-hub", dest="push_to_hub", action="store_true",
                       default=None, help="Publish the trained model to the HF Hub "
                       "(text / data_fusion).")
    run_p.add_argument("--hub-repo-id", dest="hub_repo_id",
                       help="Target HF Hub repo id (e.g. user/model).")

    predict_p = sub.add_parser(
        "predict",
        help="Predict on new data with an already-trained model or ensemble.",
        description=(
            "Score new records with a model that is already trained — a local "
            "run directory, a Hugging Face Hub repo id, or a Hub URL. What kind "
            "of run it is is read from the artifact, so one command serves "
            "single models and ensembles alike; --dry-run reports what would "
            "happen without scoring anything."
        ),
    )
    predict_p.add_argument("--source", required=True,
                           help="Run directory, Hub repo id ('org/name'), or Hub URL.")
    predict_p.add_argument("--data", required=True,
                           help="CSV/Parquet of the records to score.")
    predict_p.add_argument("--text-col", dest="text_col",
                           help="Narrative column (text, data-fusion and "
                                "feature-fusion models).")
    predict_p.add_argument("--label-col", dest="label_col",
                           help="True-label column, when the data is labelled. "
                                "Adds true_label to the outputs and reports "
                                "performance. Omit for unlabelled data.")
    predict_p.add_argument("--id-col", dest="id_col",
                           help="Row-id column, copied onto every output table.")
    predict_p.add_argument("--feature-cols", dest="feature_cols",
                           help="Comma-separated column names, in training "
                                "order. Only for a tabular model that does not "
                                "record its own.")
    predict_p.add_argument("--missing-feature-method", dest="missing_feature_method",
                           choices=list(PRETRAINED_MISSING_FEATURE_METHODS),
                           default="error",
                           help="What to do when a trained-on column is absent: "
                                "error (default) or fill_na.")
    predict_p.add_argument("--combiner",
                           help="Stacking only: which fitted stage-2 combiner to "
                                "apply (meta_learner, simple_average, "
                                "class_aware_voting, ensemble_selection, or a "
                                "meta-learner name). Default: the one the run "
                                "delivered.")
    predict_p.add_argument("--task",
                           help="Override what kind of run this is. Detection "
                                "still runs: contradicting the run's own "
                                "metadata is an error (see --force), while "
                                "overriding a guess from the files present only "
                                "warns.")
    predict_p.add_argument("--force", action="store_true",
                           help="Proceed with --task even against the pipeline "
                                "the run recorded. Only when you know that "
                                "metadata is wrong.")
    predict_p.add_argument("--batch-size", dest="batch_size", type=int, default=32,
                           help="Inference batch size for text models (default 32).")
    predict_p.add_argument("--top-k", dest="top_k", type=int, default=3,
                           help="Classes listed in the topk table (default 3).")
    predict_p.add_argument("--output-dir", dest="output_dir",
                           help="Where to write predictions_{top1,full,topk}.csv. "
                                "Omitted: nothing is written, only the summary "
                                "is printed.")
    predict_p.add_argument("--dry-run", dest="dry_run", action="store_true",
                           help="Report what would be done — kind of run, "
                                "combiner, columns present and missing — then "
                                "stop without scoring anything.")

    sub.add_parser("list-datasets", help="List built-in synthetic datasets.")
    sub.add_parser("list-models", help="List supported text model aliases.")
    return parser


# ---------------------------------------------------------------------------
# sub-commands
# ---------------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace) -> int:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg.update(_load_config(args.config))
    cfg.update(_cli_overrides(args))

    if "task" not in cfg:
        print("error: --task is required (via flags or config file).",
              file=sys.stderr)
        return 2
    # Stage-2/3 stacking deliberately reads rows, labels and seeds from the
    # completed OOF run. The Python API already supports that contract; the CLI
    # used to reject the same valid call before it reached run().
    from .runner import _normalize_task
    try:
        reuses_oof = (
            _normalize_task(cfg["task"]) == "stacking"
            and cfg.get("oof_from") is not None
        )
    except ValueError:
        reuses_oof = False  # preflight/run will report the unknown task itself
    if not reuses_oof and ("data" not in cfg or "label_col" not in cfg):
        print("error: --data and --label-col are required (via flags or config "
              "file), except for stacking with --oof-from.", file=sys.stderr)
        return 2
    cfg.setdefault("output_dir", "runs/mmva")

    # Convenience: 'demo' loads the built-in synthetic dataset in-memory.
    # n_per_class=30 keeps every cause large enough for a stratified train/test
    # split; the registry default (6) is too small for the full cause list.
    if cfg.get("data") == "demo":
        from .datasets import data as _data
        cfg["data"] = _data("va_sample", n_per_class=30)

    if getattr(args, "dry_run", False):
        return _run_dry_run(cfg)

    from .runner import run
    run(**cfg)
    print(f"Done. Artifacts in: {cfg['output_dir']}")
    return 0


def _run_dry_run(cfg: dict[str, Any]) -> int:
    """Report what a run would do, without building a model.

    Exits 2 when the configuration would not get past the checks, so a job
    script can gate on it::

        multimodalva run job.yaml --dry-run || exit 1
        multimodalva run job.yaml
    """
    from .runner import preflight

    report = preflight(**cfg)
    width = max((len(k) for k in report["facts"]), default=0)
    for key, value in report["facts"].items():
        print(f"{key:{width}s}  {value}")
    for line in report["log"]:
        print(f"{'':{width}s}  log: {line}")
    for note in report["notes"]:
        print(f"\nnote: {note}")
    if report["problems"]:
        # The facts above are the context for the errors below; without this the
        # two streams interleave and the report reads out of order.
        sys.stdout.flush()
        print(file=sys.stderr)
        for problem in report["problems"]:
            print(f"error: {problem}", file=sys.stderr)
        print(f"\nDry run: {len(report['problems'])} problem(s); the run would "
              "not get past them.", file=sys.stderr)
        return 2
    print("\nDry run: the configuration is consistent with the data. Nothing was "
          "trained or written.\nA clean dry run does not promise the run will "
          "succeed — nothing here trains, so it cannot\nsee a batch size that "
          "will not fit or a trial budget too small for the search space.")
    return 0


# Tasks whose artifact holds several models plus the rule combining them; they
# are served by predict_ensemble_from_pretrained() rather than by the
# single-model entry point. The task itself is read from the artifact.
_ENSEMBLE_TASKS = ("voting", "stacking")

# Printed after a labelled prediction: the four a user checks first. The rest
# are one call away in results.performance_leaderboard().
_PREDICT_REPORT_METRICS = ("accuracy", "balanced_accuracy", "f1_macro",
                           "csmf_accuracy")


def _cmd_predict(args: argparse.Namespace) -> int:
    from .inference.api import (
        _find_artifact_dir, _read_training_metadata, _resolve_pretrained_source,
        descend_to_pretrained_run, predict_from_pretrained, resolve_pretrained_task,
    )
    from .inference.ensemble_api import predict_ensemble_from_pretrained

    # Say what this is before anything expensive happens: reading the task is
    # a handful of file checks, while an inference pass down the wrong pipeline
    # costs a whole job and still produces plausible-looking predictions.
    try:
        root = descend_to_pretrained_run(_resolve_pretrained_source(args.source, {}))
        artifact_dir = _find_artifact_dir(root)
        meta = _read_training_metadata(root, artifact_dir)
        task, task_source, conflict = resolve_pretrained_task(
            meta, artifact_dir, args.task, force=args.force)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Detected: {task} (from {_SOURCE_WORDING.get(task_source, task_source)})",
          flush=True)
    if conflict:
        print(f"warning: {conflict}", file=sys.stderr)
    if task == "data_fusion":
        print("note: a data-fusion model expects the fused long text; --text-col "
              "must already hold it (the DataFrame → long-text converter is not "
              "released yet).", flush=True)

    # task is already settled above, conflicts included, so the call itself has
    # nothing left to reconcile — force keeps it from raising on the same
    # disagreement twice.
    common = dict(data=args.data, text_col=args.text_col, label_col=args.label_col,
                  id_col=args.id_col, task=task, force=True,
                  missing_feature_method=args.missing_feature_method,
                  batch_size=args.batch_size, top_k=args.top_k,
                  save_dir=args.output_dir)
    if task in _ENSEMBLE_TASKS:
        if args.feature_cols:
            print("error: --feature-cols does not apply to an ensemble; each "
                  "base model carries its own columns.", file=sys.stderr)
            return 2
        call, extra = predict_ensemble_from_pretrained, {"combiner": args.combiner}
    else:
        if args.combiner:
            print(f"error: --combiner applies to a stacking run, and this is a "
                  f"{task} run.", file=sys.stderr)
            return 2
        call, extra = predict_from_pretrained, {
            "feature_cols": _parse_features(args.feature_cols) if args.feature_cols
            else None,
        }

    if args.dry_run:
        return _predict_dry_run(args, task, root, meta)

    try:
        res = call(args.source, **common, **extra)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_dir:
        print(f"Wrote {len(res.top1)} predictions to {args.output_dir}")
    else:
        print(f"Scored {len(res.top1)} records (--output-dir writes them out).")
    if res.checks.combiner:
        print(f"Combined {len(res.checks.base_models)} base models with "
              f"{res.checks.combiner}.")
    if args.label_col:
        from .utils.metrics import score_predictions
        print()
        for metric in _PREDICT_REPORT_METRICS:
            print(f"{metric:22s} {score_predictions(res.top1, metric):.4f}")
        print("\n(the full set of metrics: multimodalva.results."
              "performance_leaderboard)")
    return 0


def _predict_dry_run(args: argparse.Namespace, task: str, root: Path,
                     meta: dict) -> int:
    """Report what a real call would do, without scoring anything.

    The models themselves are loaded — that is what turns "the column names look
    fine" into "this artifact can actually be read" — but no inference runs, so
    this returns in seconds rather than a job's worth of GPU time.
    """
    from .inference.api import _find_artifact_dir, _load_pretrained_input

    try:
        df = _load_pretrained_input(args.data)
        print(f"Records: {len(df)}")
        if task in _ENSEMBLE_TASKS:
            from .inference.ensemble_api import (
                _resolve_combiner, _stacking_base_dirs, _voting_base_dirs,
            )
            from .inference.api import build_pretrained_backend
            from .inference.checks import PretrainedChecks
            bases = (_voting_base_dirs(root, meta) if task == "voting"
                     else _stacking_base_dirs(root))
            kinds = {}
            for _, base_task in bases:
                kinds[base_task] = kinds.get(base_task, 0) + 1
            print("Base models: %d (%s)" % (
                len(bases), ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))))
            if task == "stacking":
                print("Combiner:", _resolve_combiner(root, meta, args.combiner))
            # Match the real ensemble prediction path: load every base artifact
            # and prepare the input, but stop before predict_proba(). Merely
            # counting directories allowed corrupt/unloadable models to pass a
            # dry run even though the command promised the models were loaded.
            expected_cols: list[str] = []
            first_labels = None
            for i, (base_dir, base_task) in enumerate(bases):
                checks = PretrainedChecks(
                    source=str(base_dir), artifact_dir=str(base_dir),
                    task=base_task, task_source="ensemble_metadata",
                )
                backend = build_pretrained_backend(
                    _find_artifact_dir(base_dir), base_task, checks,
                    text_col=args.text_col,
                    missing_feature_method=args.missing_feature_method,
                    batch_size=args.batch_size,
                )
                backend.prepare_inputs(df)
                if first_labels is None:
                    first_labels = backend.id2label
                elif backend.id2label != first_labels:
                    raise ValueError(
                        f"Base model {base_dir} has a different label map from "
                        "the first base model."
                    )
                for col in getattr(backend, "feature_cols", []) or []:
                    if col not in expected_cols:
                        expected_cols.append(col)
                print(f"Loaded base model {i + 1}: {base_task} ({base_dir})")
            cols = expected_cols
        else:
            from .inference.api import build_pretrained_backend
            from .inference.checks import PretrainedChecks
            checks = PretrainedChecks(source=str(args.source),
                                      artifact_dir=str(root), task=task,
                                      task_source="caller")
            backend = build_pretrained_backend(
                _find_artifact_dir(root), task, checks,
                text_col=args.text_col,
                feature_cols=_parse_features(args.feature_cols) if args.feature_cols
                else None,
                missing_feature_method=args.missing_feature_method,
                batch_size=args.batch_size)
            print(f"Classes: {len(backend.id2label)}")
            cols = getattr(backend, "feature_cols", None)
        if args.text_col:
            present = args.text_col in df.columns
            print(f"Text column {args.text_col!r}: "
                  f"{'present' if present else 'MISSING'}")
        if cols:
            missing = [c for c in cols if c not in df.columns]
            print(f"Feature columns: {len(cols)} expected, "
                  f"{len(cols) - len(missing)} present, {len(missing)} missing"
                  + (f" → {missing[:10]}" if missing else ""))
        if args.label_col:
            print(f"Label column {args.label_col!r}: "
                  + ("present, performance will be reported"
                     if args.label_col in df.columns else "MISSING"))
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("Dry run: nothing was scored or written.")
    return 0


# How each task_source reads in a one-line report.
_SOURCE_WORDING = {
    "training_metadata": "the run's own metadata",
    "structure": "the files present — not recorded, so check it",
    "caller": "--task",
}


def _cmd_list_datasets() -> int:
    from .datasets import list_datasets

    for name, desc in list_datasets().items():
        print(f"{name:24s} {desc}")
    return 0


def _cmd_list_models() -> int:
    from .text.models import MODEL_DESCRIPTIONS, REMOTE_MODELS, TEXT_MODELS

    print("# Text models (Hugging Face Hub)")
    for alias, checkpoint in TEXT_MODELS.items():
        print(f"{alias:20s} {checkpoint}")
        print(f"{'':20s}   {MODEL_DESCRIPTIONS.get(alias, '')}")
    if REMOTE_MODELS:
        print("\n# Text models not on the Hub (downloaded once to ~/.cache/multimodalva/)")
        for alias, url in REMOTE_MODELS.items():
            print(f"{alias:20s} {url}")
            print(f"{'':20s}   {MODEL_DESCRIPTIONS.get(alias, '')}")
    print("\nNote: 'biomedroberta' (BioMed-RoBERTa) and 'roberta-pm' (RoBERTa-PM) "
          "are different models.")
    print("\n# Tabular models")
    print("catboost, lightgbm, gbdt, xgboost, mlp, random_forest, "
          "naive_bayes, knn, svm")
    return 0


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------
def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # declared in the core dependencies; guarded anyway
        except ImportError as exc:
            raise SystemExit(
                "PyYAML is required to read YAML configs. Install it "
                "('pip install pyyaml') or use a JSON config file instead."
            ) from exc
        cfg = yaml.safe_load(text) or {}
    else:
        cfg = json.loads(text)
    return _config_hyperparams(cfg)


def _config_hyperparams(cfg: dict) -> dict:
    """Turn a config file's ``hyperparams:`` block into what ``run()`` expects.

    A config file is flat text, so a search is written as a nested mapping and
    converted here::

        hyperparams:            # search, with settings
          optimize:
            metric: csmf_accuracy
            n_trials: 50

        hyperparams:            # fixed values
          learning_rate: 3.0e-5

        hyperparams: default    # the model library's own
    """
    hp = cfg.get("hyperparams")
    if isinstance(hp, dict) and "optimize" in hp:
        if len(hp) > 1:
            raise SystemExit(
                "hyperparams: an 'optimize:' block cannot be mixed with fixed "
                f"values. Remove: {sorted(k for k in hp if k != 'optimize')}."
            )
        opts = hp["optimize"] or {}
        if not isinstance(opts, dict):
            raise SystemExit("hyperparams.optimize: must be a mapping of settings.")
        known = {f for f in Optimize.__dataclass_fields__}
        unknown = sorted(set(opts) - known)
        if unknown:
            raise SystemExit(
                f"hyperparams.optimize: unknown setting(s) {unknown}. "
                f"Valid: {sorted(known)}."
            )
        cfg["hyperparams"] = Optimize(**opts)
    return cfg


def _cli_overrides(args: argparse.Namespace) -> dict:
    """Collect only the flags the user actually set, mapped to run() kwargs."""
    out: dict[str, Any] = {}
    simple = [
        "task", "data", "label_col", "output_dir", "text_col", "model",
        "test_size", "split_seed", "train_seed", "split", "id_col",
        "top_k", "hub_repo_id",
    ]
    for key in simple:
        val = getattr(args, key, None)
        if val is not None:
            out[key] = val
    # A command line is flat, so the search is asked for with flags and turned
    # into the Optimize that run() expects.
    search = optimize_from_flags(
        args.optimize,
        n_trials=args.n_trials,
        optimize_metric=args.optimize_metric,
        backend=args.hpo_backend,
    )
    if search is not None:
        out["hyperparams"] = search
    elif (args.n_trials is not None or args.optimize_metric is not None
          or args.hpo_backend is not None):
        raise SystemExit(
            "--n-trials, --optimize-metric and --hpo-backend only mean something "
            "with --optimize. Add --optimize, or drop them."
        )
    if args.push_to_hub:
        out["push_to_hub"] = True
    if args.deterministic:
        out["deterministic"] = True
    if args.combiner is not None:
        combiners = [m.strip() for m in args.combiner.split(",") if m.strip()]
        out["combiner"] = combiners[0] if len(combiners) == 1 else combiners
    if args.oof_from is not None:
        out["oof_from"] = args.oof_from
    if args.oof_only:
        out["oof_only"] = True
    if args.no_resume and getattr(args, "resume_adopt", False):
        raise SystemExit(
            "error: --no-resume and --resume-adopt contradict each other. "
            "--no-resume rebuilds the artifacts; --resume-adopt reuses them."
        )
    if args.no_resume:
        out["resume"] = False
    elif getattr(args, "resume_adopt", False):
        out["resume"] = "adopt"
    if args.features is not None:
        out["features"] = _parse_features(args.features)
    if args.filters is not None:
        out["filters"] = _parse_filters(args.filters)
    return out


def _parse_features(raw: str):
    """A regex/auto selector stays a string; a comma list becomes a list."""
    if raw.startswith("re:") or raw == "auto":
        return raw
    return [c.strip() for c in raw.split(",") if c.strip()]


def _parse_filters(raw: str) -> dict:
    """Parse 'col=val,col2=a|b' into {col: val, col2: [a, b]}."""
    filters: dict[str, Any] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"bad --filters entry {pair!r}; expected col=value.")
        col, val = pair.split("=", 1)
        col, val = col.strip(), val.strip()
        filters[col] = [v.strip() for v in val.split("|")] if "|" in val else val
    return filters


if __name__ == "__main__":
    raise SystemExit(main())
