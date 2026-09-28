#!/usr/bin/env python3
"""
MultimodalVA diagnostic matrix — run every pipeline and every model, and say
exactly which cell broke and where.

``smoke_test.py`` answers "is the package alive?" in a few seconds. This answers
"what still works?" across the full surface: all 6 pipelines and all 9 tabular
models, plus the text models on request.

    python tests/diagnose.py                  # offline: tabular models + tabular pipelines
    python tests/diagnose.py --with-text      # adds text pipelines (downloads a 17MB model)
    python tests/diagnose.py --group tabular-models
    python tests/diagnose.py --json report.json   # machine-readable, for tracking over time

Every check prints [PASS] / [FAIL] / [SKIP]. A failure prints the exception and
the last line *inside multimodalva* that ran, so there is a file:line to open
rather than a wall of framework traceback. Exit code is non-zero if anything
failed, so this can gate a release.

Optional dependencies (lightgbm, xgboost, catboost, autogluon) are reported as
SKIP with the install command, never as failures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""

TINY_TEXT_MODEL = "prajjwal1/bert-tiny"


@dataclass
class Result:
    group: str
    name: str
    status: str          # PASS | FAIL | SKIP
    detail: str = ""
    seconds: float = 0.0
    where: str = ""      # file:line inside multimodalva, for FAIL
    exc_type: str = ""
    traceback: str = field(default="", repr=False)


def _package_frame(exc: BaseException) -> str:
    """Return the deepest 'file:line in function' that is inside multimodalva."""
    import multimodalva

    root = str(Path(multimodalva.__file__).parent.resolve())
    hit = ""
    for frame in traceback.extract_tb(exc.__traceback__):
        if str(Path(frame.filename).resolve()).startswith(root):
            rel = Path(frame.filename).resolve().relative_to(Path(root).parent)
            hit = f"{rel}:{frame.lineno} in {frame.name}()"
    return hit


def _sample_df(n_per_class: int = 12):
    from multimodalva import data

    return data("va_sample", n_per_class=n_per_class)


def _feature_cols(df) -> list[str]:
    return [c for c in df.columns if c not in {"id", "cause_of_death", "narrative"}]


def run_check(group: str, name: str, fn) -> Result:
    """Run one check, converting any exception into a FAIL with a debug pointer."""
    t0 = time.perf_counter()
    try:
        detail = fn()
        if isinstance(detail, str) and detail.startswith("SKIP:"):
            return Result(group, name, "SKIP", detail[5:].strip(), time.perf_counter() - t0)
        return Result(group, name, "PASS", detail or "", time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic harness
        return Result(
            group, name, "FAIL",
            f"{type(exc).__name__}: {exc}"[:300],
            time.perf_counter() - t0,
            where=_package_frame(exc),
            exc_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_tabular_model(model: str, workdir: Path):
    """One tabular model, end to end through run()."""
    def _fn():
        from multimodalva.runner import run

        df = _sample_df()
        res = run(
            task="tabular", data=df, label_col="cause_of_death",
            features=_feature_cols(df), model=model,
            output_dir=workdir / f"tab_{model}", save_diagnostics=False,
        )
        preds = res["predictions"]
        n = len(preds.top1)
        assert n > 0, "no predictions returned"
        assert preds.full.shape[1] > 1, "probability columns missing"
        return f"{n} predictions, {preds.full.shape[1] - 1} classes"
    return _fn


def check_pipeline(task: str, workdir: Path, *, with_text: bool):
    """One pipeline end to end through run()."""
    def _fn():
        from multimodalva.runner import run

        df = _sample_df()
        feats = _feature_cols(df)
        kwargs: dict = dict(
            task=task, data=df, label_col="cause_of_death",
            output_dir=workdir / f"pipe_{task}", save_diagnostics=False,
        )
        needs_text = task in {"text", "data_fusion", "feature_fusion"}
        if needs_text and not with_text:
            return "SKIP: needs a text model — pass --with-text"

        if task == "tabular":
            kwargs.update(features=feats, model="random_forest")
        elif task == "text":
            kwargs.update(text_col="narrative", model=TINY_TEXT_MODEL,
                          hyperparams={"epochs": 1, "batch_size": 8})
        elif task == "data_fusion":
            kwargs.update(text_col="narrative", features=feats, model=TINY_TEXT_MODEL,
                          max_length=256, hyperparams={"epochs": 1, "batch_size": 8})
        elif task == "feature_fusion":
            try:
                import autogluon.multimodal  # noqa: F401
            except ImportError:
                return "SKIP: autogluon.multimodal not installed — pip install multimodalva[feature_fusion]"
            kwargs.update(text_col="narrative", features=feats, model=TINY_TEXT_MODEL,
                          time_limit=120)
        elif task in {"voting", "stacking"}:
            tab = [{"model_name": "random_forest"}, {"model_name": "gbdt"}]
            txt = ([{"model_name": TINY_TEXT_MODEL, "max_length": 128,
                     "hyperparams": {"epochs": 1, "batch_size": 8}}] if with_text else [])
            kwargs.update(features=feats, tabular_models=tab, text_models=txt)
            if txt:
                kwargs.update(text_col="narrative")
            if task == "stacking":
                # n_folds is a StackingClassifier constructor argument, not a
                # run() argument, so it travels via init_kwargs.
                kwargs.update(init_kwargs={"n_folds": 2})

        res = run(**kwargs)
        preds = res.get("predictions")
        assert preds is not None and len(preds.top1) > 0, "no predictions returned"
        extra = "" if not with_text or task == "tabular" else " (+text base)"
        return f"{len(preds.top1)} predictions{extra}"
    return _fn


def check_text_model(alias: str, workdir: Path):
    """Resolve a text model alias to its checkpoint (no download, no training)."""
    def _fn():
        from multimodalva.text.models import resolve_model_name

        ckpt = resolve_model_name(alias)
        assert isinstance(ckpt, str) and ckpt, "empty checkpoint"
        return ckpt
    return _fn


# --------------------------------------------------------------------------

GROUPS = ("tabular-models", "text-models", "pipelines")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=GROUPS, action="append",
                    help="Only run this group (repeatable). Default: all.")
    ap.add_argument("--with-text", action="store_true",
                    help=f"Also train text pipelines using {TINY_TEXT_MODEL} (~17MB download).")
    ap.add_argument("--json", type=Path, help="Write a machine-readable report here.")
    ap.add_argument("--keep", action="store_true", help="Keep the work directory.")
    args = ap.parse_args(argv)
    groups = set(args.group or GROUPS)

    from multimodalva.tabular.train import TABULAR_MODELS
    from multimodalva.text.models import TEXT_MODELS
    from multimodalva.runner import SUPPORTED_TASKS

    tmp = Path(tempfile.mkdtemp(prefix="mmva_diag_"))
    print(f"MultimodalVA diagnostic matrix{DIM}   workdir: {tmp}{RESET}")
    print("-" * 68)

    results: list[Result] = []
    plan: list[tuple[str, str, object]] = []
    if "tabular-models" in groups:
        plan += [("tabular-models", m, check_tabular_model(m, tmp)) for m in sorted(TABULAR_MODELS)]
    if "text-models" in groups:
        plan += [("text-models", m, check_text_model(m, tmp)) for m in sorted(TEXT_MODELS)]
    if "pipelines" in groups:
        plan += [("pipelines", t, check_pipeline(t, tmp, with_text=args.with_text))
                 for t in SUPPORTED_TASKS]

    current = None
    for group, name, fn in plan:
        if group != current:
            current = group
            print(f"\n{group}")
        r = run_check(group, name, fn)
        results.append(r)
        tag = {"PASS": f"{GREEN}PASS{RESET}", "FAIL": f"{RED}FAIL{RESET}",
               "SKIP": f"{YELLOW}SKIP{RESET}"}[r.status]
        print(f"  [{tag}] {name:<18} {r.detail[:70]}{DIM}  {r.seconds:.1f}s{RESET}")
        if r.status == "FAIL" and r.where:
            print(f"         {DIM}↳ last line in multimodalva: {r.where}{RESET}")

    n_pass = sum(r.status == "PASS" for r in results)
    n_fail = sum(r.status == "FAIL" for r in results)
    n_skip = sum(r.status == "SKIP" for r in results)

    print("\n" + "=" * 68)
    for g in GROUPS:
        rows = [r for r in results if r.group == g]
        if rows:
            p = sum(r.status == "PASS" for r in rows)
            print(f"  {g:<16} {p}/{len(rows)} passed"
                  + (f", {sum(r.status=='FAIL' for r in rows)} failed" if any(r.status == "FAIL" for r in rows) else "")
                  + (f", {sum(r.status=='SKIP' for r in rows)} skipped" if any(r.status == "SKIP" for r in rows) else ""))
    print(f"  {'TOTAL':<16} {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print("=" * 68)

    if n_fail:
        print(f"\n{RED}Failures to debug:{RESET}")
        for r in results:
            if r.status == "FAIL":
                print(f"  {r.group}/{r.name}: {r.exc_type}")
                print(f"    {r.detail[:200]}")
                if r.where:
                    print(f"    open: {r.where}")

    if args.json:
        args.json.write_text(json.dumps(
            {"passed": n_pass, "failed": n_fail, "skipped": n_skip,
             "with_text": args.with_text,
             "results": [{k: v for k, v in vars(r).items() if k != "traceback"}
                         for r in results]},
            indent=2))
        print(f"\nReport written to {args.json}")

    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
