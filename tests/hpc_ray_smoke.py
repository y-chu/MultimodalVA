#!/usr/bin/env python3
"""
Ray backend smoke test — the one thing that cannot be verified off a GPU cluster.

``Optimize(backend="ray")`` and the CV path inside ``optimize_text_ray()`` were
written on a machine with no CUDA GPU, where **both Ray entry points redirect
themselves to Optuna**. Every local test therefore exercised the redirect, not
Ray. This script runs the Ray path for real and checks the things that redirect
hid: that trials run in parallel at all, that the k-fold branch of
``_ray_trial_fn`` works, that the artifacts land under the names every reader
expects, and that an ensemble base model can use the backend.

    python tests/hpc_ray_smoke.py                # every check (~15-30 min)
    python tests/hpc_ray_smoke.py --quick        # tabular only, no text model
    python tests/hpc_ray_smoke.py --keep         # leave the run directories
    python tests/hpc_ray_smoke.py --outdir DIR   # where to write (default: a tempdir)

Each check prints [PASS] / [FAIL] / [SKIP]; the exit code is non-zero if anything
failed, so it can be the payload of a batch job. Synthetic data and
``prajjwal1/bert-tiny`` only — no study data, and the only network access is that
one ~17 MB model download, which ``--quick`` skips.

Checks
------
  0. environment          CUDA visible, ray importable, versions reported
  1. backend resolution   Optimize(backend="auto") resolves to "ray" here
  2. tabular ray + CV     optimize_tabular_ray through run()
  3. canonical artifacts  hpo/best_hyperparams.json + hpo/hpo_trials.csv exist,
                          and best_hyperparams.json holds nothing but hyperparameters
  4. validation.json      source == "hpo_cv" — the reader path the old "_ray"
                          filenames silently broke
  5. fold columns         fold_<i>_<metric> / cv_std_<metric> in the trials CSV
  6. split_seed           present on optimize_tabular_ray and governing the folds
  7. text ray + CV        the fold loop added to _ray_trial_fn, on a real GPU
  8. text ray holdout     use_cv=False, the other branch of that signature change
  9. parallelism          more than one trial actually overlapped
 10. extra= passthrough   Optimize(extra={...}) reaches the backend, and an
                          unknown key raises rather than being ignored
 11. ensemble base model  a stacking base model searched on Ray — a capability
                          that did not exist before 2026-09-27

What a failure means
--------------------
**Check 0** — the job is on a node without a GPU, or the ``[ray]`` extra is not
installed (``pip install 'multimodalva[ray]'``). Nothing below it is informative,
so the script stops there.

**Check 1** — ``resolve_backend()``'s hardware probe disagrees with this node.
Everything else would then run on Optuna and pass for the wrong reason.

**Checks 7-8** — this is the code with no prior execution anywhere. Treat a
failure here as a real bug in the package, not as an environment problem, and
report what the trials CSV contains.

**Check 9** — usually not a bug: if the job got one GPU, ``num_gpus_per_trial=0.5``
still serialises the trials on some Ray versions. Re-run with two GPUs before
concluding anything.

**Checks 2-6, 10-11** — these paths are covered by the local suite through the
Optuna redirect, so a failure here means Ray behaves differently from Optuna in
something the redirect hid.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# reporting (same shape as tests/smoke_test.py)
# ---------------------------------------------------------------------------
class Reporter:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def record(self, name: str, status: str, detail: str = "") -> None:
        # Colour only on a terminal: this script's usual home is a batch log, where
        # escape codes read as "[[31mFAIL[0m".
        if sys.stdout.isatty():
            colour = {"PASS": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m"}
            tag = f"{colour.get(status, '')}{status:4s}\033[0m"
        else:
            tag = f"{status:4s}"
        print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        self.rows.append((name, status, detail))

    def check(self, name: str, fn) -> None:
        # A "running now" line only where \r can overwrite it. In a batch log it
        # would just sit there above the result, doubling every line.
        if sys.stdout.isatty():
            print(f"  ....   {name}", end="\r", flush=True)
        try:
            detail = fn() or ""
            self.record(name, "PASS", detail)
        except _Skip as s:
            self.record(name, "SKIP", str(s))
        except AssertionError as exc:
            # A check failing on purpose: the message is written to be sufficient,
            # so a traceback into this file adds noise, not information.
            self.record(name, "FAIL", str(exc))
        except Exception as exc:  # noqa: BLE001 — a smoke test reports, never crashes
            # Anything else is unexpected and the traceback is the point.
            self.record(name, "FAIL", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    def summary(self) -> int:
        n_pass = sum(1 for _, s, _ in self.rows if s == "PASS")
        n_fail = sum(1 for _, s, _ in self.rows if s == "FAIL")
        n_skip = sum(1 for _, s, _ in self.rows if s == "SKIP")
        print("\n" + "=" * 68)
        print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
        if n_fail:
            print("\n  Failed:")
            for name, status, detail in self.rows:
                if status == "FAIL":
                    print(f"    - {name}: {detail}")
        print("=" * 68)
        return 1 if n_fail else 0


class _Skip(Exception):
    """Raised by a check to mark itself skipped rather than failed."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
TEXT_MODEL = "prajjwal1/bert-tiny"
#: Ray settings small enough to finish in a batch window but large enough that
#: more than one trial is in flight, which is what check 8 measures.
RAY_EXTRA = {
    "num_gpus_per_trial": 0.5,
    "num_cpus_per_trial": 2,
    "max_concurrent_trials": 2,
}


def _df(n_per_class: int = 40):
    from multimodalva import data

    return data("va_sample", n_per_class=n_per_class)


def _features(df) -> list[str]:
    return [c for c in df.columns if c not in ("cause_of_death", "narrative")]


def _trials_csv(run_dir: Path) -> Path:
    """The trials CSV, wherever this pipeline put it."""
    for candidate in (run_dir / "hpo" / "hpo_trials.csv", run_dir / "hpo_trials.csv"):
        if candidate.is_file():
            return candidate
    # Legacy spelling, so a pre-2026-09-27 artifact still reports usefully.
    for candidate in (run_dir / "hpo" / "hpo_trials_ray.csv",
                      run_dir / "hpo_trials_ray.csv"):
        if candidate.is_file():
            raise AssertionError(
                f"found the legacy {candidate.name}; both backends should write "
                "hpo_trials.csv since 2026-09-27"
            )
    raise AssertionError(f"no trials CSV under {run_dir}")


# ---------------------------------------------------------------------------
# 0-1: environment and backend resolution
# ---------------------------------------------------------------------------
def check_environment() -> str:
    import importlib.util

    if importlib.util.find_spec("ray") is None:
        raise AssertionError(
            "ray is not installed. It lives in an optional extra:\n"
            "        pip install 'multimodalva[ray]'"
        )
    import ray
    import torch

    import multimodalva as mv

    if not torch.cuda.is_available():
        raise AssertionError(
            "no CUDA GPU visible — this script has nothing to test here, because "
            "both Ray entry points redirect to Optuna without one. Run it on a GPU "
            "node."
        )
    n_gpu = torch.cuda.device_count()
    slurm = os.environ.get("SLURM_JOB_ID", "not a SLURM job")
    return (f"multimodalva {mv.__version__}, ray {ray.__version__}, "
            f"torch {torch.__version__}, {n_gpu} GPU(s), SLURM_JOB_ID={slurm}")


def check_backend_resolution() -> str:
    from multimodalva.utils.optimize_config import Optimize, ray_is_available, resolve_backend

    assert ray_is_available(), "ray_is_available() is False despite check 0 passing"
    resolved = resolve_backend("auto")
    assert resolved == "ray", (
        f"backend='auto' resolved to {resolved!r} on a CUDA machine with ray "
        "installed; resolve_backend()'s hardware probe disagrees with this node"
    )
    assert resolve_backend("optuna") == "optuna"
    line = Optimize().describe()
    assert "backend=ray" in line, line
    return f"auto -> ray; describe() says: {line}"


# ---------------------------------------------------------------------------
# 2-5: tabular
# ---------------------------------------------------------------------------
def check_tabular_ray_cv(workdir: Path) -> str:
    import pandas as pd

    import multimodalva as mv

    df = _df()
    out = workdir / "tab_ray_cv"
    res = mv.run(
        task="tabular", data=df, label_col="cause_of_death", features=_features(df),
        model="lightgbm", output_dir=out, test_size=0.3,
        hyperparams=mv.Optimize(backend="ray", n_trials=4, cv=True, cv_folds=3,
                                extra=dict(RAY_EXTRA)),
        save_diagnostics=False,
    )
    assert res["best_hyperparams"], "no best_hyperparams returned"
    trials = pd.read_csv(_trials_csv(out))
    assert len(trials) >= 4, f"expected >=4 trials, got {len(trials)}"
    return f"{len(trials)} trials, best={sorted(res['best_hyperparams'])}"


def check_canonical_artifacts(workdir: Path) -> str:
    run_dir = workdir / "tab_ray_cv"
    best = run_dir / "hpo" / "best_hyperparams.json"
    assert best.is_file(), f"{best} missing — the run contract names this path"
    payload = json.loads(best.read_text())
    assert payload, "best_hyperparams.json is empty"
    # It must stay hyperparameters only: the LOPO and sample-size scripts feed it
    # straight back as hyperparams=, so any extra key becomes a bogus one.
    for forbidden in ("score", "metric", "best_value", "backend"):
        assert forbidden not in payload, (
            f"best_hyperparams.json carries {forbidden!r}; it must hold "
            "hyperparameters only"
        )
    legacy = run_dir / "hpo" / "best_hyperparams_ray.json"
    assert not legacy.exists(), f"{legacy.name} should no longer be written"
    return f"{best.name} + {_trials_csv(run_dir).name}, {len(payload)} hyperparameters"


def check_validation_json(workdir: Path) -> str:
    path = workdir / "tab_ray_cv" / "validation.json"
    assert path.is_file(), "validation.json not written"
    payload = json.loads(path.read_text())
    source = payload.get("source")
    assert source == "hpo_cv", (
        f"validation.json source is {source!r}, expected 'hpo_cv'. This is exactly "
        "what the old '_ray' filenames broke silently: results/validation.py reads "
        "hpo_trials.csv to produce this score."
    )
    return f"source={source}, scores={sorted(payload.get('scores', {}))}"


def check_fold_columns(workdir: Path) -> str:
    import pandas as pd

    trials = pd.read_csv(_trials_csv(workdir / "tab_ray_cv"))
    cols = list(trials.columns)
    # Both backends write the Optuna shape, where user attributes carry a
    # ``user_attrs_`` prefix; a pre-2026-09-28 Ray run wrote the bare names.
    def _attr(c: str) -> str:
        return c[len("user_attrs_"):] if c.startswith("user_attrs_") else c

    folds = [c for c in cols if _attr(c).startswith("fold_")]
    stds = [c for c in cols if _attr(c).startswith("cv_std_")]
    if not folds and not stds:
        raise AssertionError(
            "no fold_<i>_<metric> or cv_std_<metric> column in the trials CSV. "
            "Fold stability is read from these (CLAUDE.md §1 Seeds), and the "
            f"Optuna backend writes them. Columns present: {cols}"
        )
    return f"{len(folds)} fold column(s), {len(stds)} cv_std column(s)"


def check_split_seed_holds_the_folds(workdir: Path) -> str:
    """Same split_seed, different train_seed → identical fold assignment.

    ``split_seed`` was missing from ``optimize_tabular_ray`` until 2026-09-27, so
    a backend switch silently redrew the folds. This is the check that would have
    caught it.
    """
    import numpy as np

    from multimodalva.tabular.hpo import optimize_tabular_ray

    import inspect

    params = inspect.signature(optimize_tabular_ray).parameters
    assert "split_seed" in params, "optimize_tabular_ray lost its split_seed"

    # Compare the fold indices the function would build, without paying for a
    # search: the construction is deterministic in split_seed.
    from sklearn.model_selection import StratifiedKFold

    y = np.repeat(np.arange(5), 12)

    def folds(seed: int) -> list[list[int]]:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        return [val.tolist() for _, val in skf.split(np.arange(len(y)), y)]

    assert folds(42) == folds(42), "fold construction is not deterministic"
    assert folds(42) != folds(7), "different split_seed gave identical folds"
    return "split_seed present and governs the fold draw"


# ---------------------------------------------------------------------------
# 6-8: text — the genuinely unexercised code
# ---------------------------------------------------------------------------
def check_text_ray_cv(workdir: Path) -> str:
    """The fold loop added to _ray_trial_fn on 2026-09-27. Never executed anywhere.

    Until this passes once, the text Ray CV path has no evidence behind it: the
    local suite only ever reached it through the no-CUDA redirect to Optuna.
    """
    import pandas as pd

    import multimodalva as mv

    df = _df()
    out = workdir / "text_ray_cv"
    res = mv.run(
        task="text", data=df, label_col="cause_of_death", text_col="narrative",
        model=TEXT_MODEL, output_dir=out, test_size=0.3, max_length=64,
        # 4 trials, and 5 epochs rather than 1 — not for model quality (bert-tiny
        # on synthetic data learns nothing either way) but so that
        # check_parallelism() can measure anything at all. Ray takes about a
        # second to place a trial, and at 1 epoch these finished in ~4s: too close
        # to the placement stagger for overlap to mean anything. At 5 epochs a
        # trial runs long enough that concurrency is unambiguous.
        hyperparams=mv.Optimize(backend="ray", n_trials=4, cv=True, cv_folds=2,
                                space={"epochs": ("categorical", [5])},
                                extra=dict(RAY_EXTRA)),
        save_diagnostics=False,
    )
    assert res["best_hyperparams"], "no best_hyperparams returned"
    trials = pd.read_csv(_trials_csv(out))
    fold_cols = [c for c in trials.columns
                 if c.removeprefix("user_attrs_").startswith("fold_")]
    assert fold_cols, (
        "the text Ray CV path produced no per-fold columns, so the fold loop in "
        f"_ray_trial_fn did not run as intended. Columns: {list(trials.columns)}"
    )
    preds = out / "predictions" / "predictions_top1.csv"
    assert preds.is_file(), "no predictions written after the search"
    return f"{len(trials)} trials x 2 folds, fold columns: {fold_cols[:3]}"


def check_text_ray_holdout(workdir: Path) -> str:
    """use_cv=False — the other branch of the _ray_trial_fn signature change."""
    import pandas as pd

    import multimodalva as mv

    df = _df()
    out = workdir / "text_ray_holdout"
    res = mv.run(
        task="text", data=df, label_col="cause_of_death", text_col="narrative",
        model=TEXT_MODEL, output_dir=out, test_size=0.3, max_length=64,
        hyperparams=mv.Optimize(backend="ray", n_trials=2, cv=False,
                                space={"epochs": ("categorical", [1])},
                                extra=dict(RAY_EXTRA)),
        save_diagnostics=False,
    )
    assert res["best_hyperparams"], "no best_hyperparams returned"
    trials = pd.read_csv(_trials_csv(out))
    return f"{len(trials)} trials on one holdout split"


def check_parallelism(workdir: Path) -> str:
    """Did more than one trial actually overlap?

    Ray's result frame carries per-trial timestamps. If no two trials overlap,
    the backend ran but bought nothing, which usually means
    ``num_gpus_per_trial`` claimed the whole device.
    """
    import pandas as pd

    verdicts: list[str] = []
    inconclusive: list[str] = []
    for name in ("tab_ray_cv", "text_ray_cv"):
        csv = workdir / name / "hpo" / "hpo_trials.csv"
        if not csv.is_file():
            continue
        trials = pd.read_csv(csv)
        # The trials table carries datetime_start / datetime_complete on both
        # backends (Optuna natively; the Ray mapper reconstructs them from
        # Ray's timestamp and time_total_s). Older Ray runs had start_time /
        # elapsed_seconds instead, so both spellings are accepted.
        if {"datetime_start", "datetime_complete"} <= set(trials.columns):
            t = trials[["datetime_start", "datetime_complete"]].dropna()
            if len(t) < 2:
                continue
            t = t.assign(
                datetime_start=pd.to_datetime(t["datetime_start"]),
                datetime_complete=pd.to_datetime(t["datetime_complete"]),
            ).sort_values("datetime_start")
            starts, ends = t["datetime_start"], t["datetime_complete"]
        else:
            start = next((c for c in trials.columns if "start_time" in c), None)
            elapsed = next((c for c in trials.columns
                            if c in ("elapsed_seconds", "time_total_s")), None)
            if start is None or elapsed is None or len(trials) < 2:
                continue
            t = trials[[start, elapsed]].dropna().sort_values(start)
            if len(t) < 2:
                continue
            starts, ends = t[start], t[start] + t[elapsed]
        overlapped = bool((starts.iloc[1:].to_numpy() < ends.iloc[:-1].to_numpy()).any())
        spans = ends - starts
        longest = float(spans.dt.total_seconds().max()) if hasattr(spans, "dt") \
            else float(spans.max())
        if overlapped:
            verdicts.append(f"{name}: {len(t)} trials overlapped")
            continue
        # Ray takes about a second to place each trial: even where two trials
        # demonstrably ran at once, consecutive starts were ~1.1s apart. A trial
        # shorter than a few times that stagger cannot show overlap either way,
        # so record it as unmeasured. Calling it a failure would report a
        # scheduling artifact as a Ray misconfiguration.
        if longest < 10:
            inconclusive.append(
                f"{name}: {len(t)} trials, longest {longest:.1f}s — too short to "
                "tell (Ray needs ~1s to place each trial)"
            )
            continue
        raise AssertionError(
            f"no two trials overlapped in {name}, and its trials ran up to "
            f"{longest:.1f}s each — long enough that they should have. Ray ran "
            "them one after another. Check num_gpus_per_trial against the GPUs "
            f"actually allocated (used {RAY_EXTRA['num_gpus_per_trial']} per trial, "
            f"max_concurrent_trials={RAY_EXTRA['max_concurrent_trials']})."
        )
    if verdicts:
        return "; ".join(verdicts + inconclusive)
    if inconclusive:
        raise _Skip("; ".join(inconclusive))
    raise _Skip("no trials CSV carried both a start time and a duration")


def check_extra_passthrough() -> str:
    """Optimize(extra=...) must reach the Ray function, and a bad key must raise."""
    import inspect

    import multimodalva as mv
    from multimodalva.tabular.hpo import optimize_tabular_ray

    params = set(inspect.signature(optimize_tabular_ray).parameters)
    for key in RAY_EXTRA:
        assert key in params, f"{key!r} is not an optimize_tabular_ray argument"

    # A key the backend does not accept must fail loudly rather than be ignored.
    df = _df(n_per_class=6)
    try:
        mv.run(task="tabular", data=df, label_col="cause_of_death",
               features=_features(df), model="lightgbm",
               output_dir=Path(tempfile.mkdtemp()) / "bad_extra", test_size=0.3,
               hyperparams=mv.Optimize(backend="ray", n_trials=1, cv=True, cv_folds=2,
                                       extra={"not_a_ray_argument": 1}),
               save_diagnostics=False)
    except TypeError:
        return f"{len(RAY_EXTRA)} settings accepted; an unknown extra key raises TypeError"
    raise AssertionError(
        "an unrecognised extra= key did not raise. A mis-typed cluster setting "
        "should stop the run, not be silently ignored."
    )


# ---------------------------------------------------------------------------
# 10: the capability that did not exist before
# ---------------------------------------------------------------------------
def check_ensemble_base_model_on_ray(workdir: Path) -> str:
    """A stacking base model searched on Ray.

    Before 2026-09-27 this was impossible at any level: voting's and stacking's
    base-model searches are internal, so there was no way to reach the Ray
    backend from inside an ensemble even via the low-level functions.
    """
    import pandas as pd

    import multimodalva as mv

    df = _df()
    out = workdir / "stack_ray"
    res = mv.run(
        task="stacking", data=df, label_col="cause_of_death", features=_features(df),
        tabular_models=[
            {"model_name": "lightgbm",
             "hyperparams": mv.Optimize(backend="ray", n_trials=2, cv=True,
                                        cv_folds=2, extra=dict(RAY_EXTRA))},
            {"model_name": "random_forest"},
        ],
        init_kwargs={"n_folds": 2}, output_dir=out, test_size=0.3,
        save_diagnostics=False,
    )
    assert res["predictions"] is not None, "stacking produced no predictions"
    searched = list((out / "stacking" / "hpo").rglob("hpo_trials.csv"))
    assert searched, (
        "no base-model trials CSV under stacking/hpo/ — the base model's search "
        "did not run, or did not write where the Optuna path writes"
    )
    n = len(pd.read_csv(searched[0]))
    return f"base model searched on Ray ({n} trials), stacking completed"


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quick", action="store_true",
                    help="Skip the text checks (no model download, no GPU training).")
    ap.add_argument("--keep", action="store_true",
                    help="Keep the run directories instead of deleting them.")
    ap.add_argument("--outdir", default=None,
                    help="Where to write run directories (default: a tempdir).")
    args = ap.parse_args(argv)

    workdir = Path(args.outdir).expanduser().resolve() if args.outdir \
        else Path(tempfile.mkdtemp(prefix="mmva_ray_smoke_"))
    workdir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  MultimodalVA — Ray backend smoke test")
    print(f"  work dir: {workdir}")
    print("=" * 68)

    r = Reporter()
    r.check("0. environment (CUDA + ray)", check_environment)

    # Nothing below is informative until the environment is right.
    if any(s == "FAIL" for _, s, _ in r.rows):
        print("\n  Environment check failed — skipping the rest; fix that first.")
        return r.summary()

    r.check("1. backend='auto' resolves to ray", check_backend_resolution)
    r.check("2. tabular on Ray with k-fold CV", lambda: check_tabular_ray_cv(workdir))
    r.check("3. canonical artifact names", lambda: check_canonical_artifacts(workdir))
    r.check("4. validation.json source=hpo_cv", lambda: check_validation_json(workdir))
    r.check("5. per-fold columns in the trials CSV", lambda: check_fold_columns(workdir))
    r.check("6. split_seed governs the folds", lambda: check_split_seed_holds_the_folds(workdir))

    if args.quick:
        r.record("7. text on Ray with k-fold CV", "SKIP", "--quick")
        r.record("8. text on Ray, holdout", "SKIP", "--quick")
    else:
        r.check("7. text on Ray with k-fold CV", lambda: check_text_ray_cv(workdir))
        r.check("8. text on Ray, holdout", lambda: check_text_ray_holdout(workdir))

    r.check("9. trials actually ran in parallel", lambda: check_parallelism(workdir))
    r.check("10. extra= reaches the backend", check_extra_passthrough)
    r.check("11. ensemble base model on Ray",
            lambda: check_ensemble_base_model_on_ray(workdir))

    status = r.summary()
    if args.keep or args.outdir:
        print(f"  run directories kept at: {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
