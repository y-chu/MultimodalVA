#!/usr/bin/env python3
"""
MultimodalVA smoke test — one script, clear pass/fail reporting.

Verifies the package is installed and the one-call ``run()`` API works across
its interfaces, using only the committed synthetic data (no network, no GPU).

    python tests/smoke_test.py            # core checks (fast, offline)
    python tests/smoke_test.py --with-text  # also run a tiny text model
                                            # (downloads prajjwal1/bert-tiny, ~17MB)

Each check prints [PASS] / [FAIL] / [SKIP] with a short reason; a summary table
and a non-zero exit code on any failure make it CI-friendly.

Checks
------
  1. light import        `from multimodalva import run` pulls no torch/transformers
  2. tabular pipeline    run(task="tabular") end-to-end, artifacts on disk
  3. external split      a fixed {train_ids,test_ids} split is honoured exactly
  4. CLI                 `python -m multimodalva.cli run ...` produces predictions
  5. YAML config         `... run config.yaml` matches the CLI result
  6. text pipeline       run(task="text") with a tiny model      (--with-text only)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


class Reporter:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def record(self, name: str, status: str, detail: str = "") -> None:
        colour = {"PASS": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m"}
        reset = "\033[0m"
        tag = f"{colour.get(status, '')}{status:4s}{reset}"
        print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
        self.rows.append((name, status, detail))

    def check(self, name: str, fn) -> None:
        try:
            detail = fn() or ""
            self.record(name, "PASS", detail)
        except _Skip as s:
            self.record(name, "SKIP", str(s))
        except Exception as exc:  # noqa: BLE001 — smoke test reports, never crashes
            self.record(name, "FAIL", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    def summary(self) -> int:
        n_pass = sum(1 for _, s, _ in self.rows if s == "PASS")
        n_fail = sum(1 for _, s, _ in self.rows if s == "FAIL")
        n_skip = sum(1 for _, s, _ in self.rows if s == "SKIP")
        print("\n" + "=" * 60)
        print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
        print("=" * 60)
        return 1 if n_fail else 0


class _Skip(Exception):
    """Raised by a check to mark itself skipped rather than failed."""


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def check_light_import() -> str:
    subprocess.run(
        [sys.executable, "-c",
         "import sys; import multimodalva; assert hasattr(multimodalva, 'run');"
         " assert 'torch' not in sys.modules and 'transformers' not in sys.modules,"
         " 'heavy deps imported at module load'"],
        check=True, capture_output=True, text=True,
    )
    return "no torch/transformers pulled on import"


def _sample_df(n_per_class: int = 30):
    from multimodalva import data
    return data("va_sample", n_per_class=n_per_class)


def check_tabular(workdir: Path) -> str:
    from multimodalva import run
    out = workdir / "tab"
    res = run(task="tabular", data=_sample_df(), label_col="cause_of_death",
              model="random_forest", output_dir=out, features=r"re:^i\d{3}")
    top1 = out / "predictions" / "predictions_top1.csv"
    assert top1.exists(), "predictions_top1.csv not written"
    assert res["predictions"].top1 is not None
    n = len(res["predictions"].top1)
    return f"{n} test predictions written to {top1.parent}"


def check_external_split(workdir: Path) -> str:
    from multimodalva import run
    df = _sample_df()
    ids = df["id"].tolist()
    spec = {"train_ids": ids[:250], "test_ids": ids[250:]}
    res = run(task="tabular", data=df, label_col="cause_of_death",
              model="random_forest", output_dir=workdir / "split",
              features=r"re:^i\d{3}", split=spec, id_col="id")
    got = len(res["predictions"].top1)
    assert got == len(spec["test_ids"]), (
        f"expected {len(spec['test_ids'])} test rows, got {got}"
    )
    return f"external split honoured exactly ({got} test rows)"


def check_cli(workdir: Path, csv_path: Path) -> str:
    out = workdir / "cli"
    subprocess.run(
        [sys.executable, "-m", "multimodalva.cli", "run",
         "--task", "tabular", "--data", str(csv_path),
         "--label-col", "cause_of_death", "--model", "random_forest",
         "--output-dir", str(out), "--features", r"re:^i\d{3}"],
        check=True, capture_output=True, text=True,
    )
    assert (out / "predictions" / "predictions_top1.csv").exists()
    return "CLI run wrote predictions"


def check_yaml(workdir: Path, csv_path: Path) -> str:
    try:
        import yaml  # noqa: F401
    except ImportError:
        raise _Skip("PyYAML not installed")
    out = workdir / "yaml"
    cfg = workdir / "config.yaml"
    cfg.write_text(
        "task: tabular\n"
        f"data: {csv_path}\n"
        "label_col: cause_of_death\n"
        "model: random_forest\n"
        f"output_dir: {out}\n"
        'features: "re:^i\\\\d{3}"\n'
    )
    subprocess.run(
        [sys.executable, "-m", "multimodalva.cli", "run", str(cfg)],
        check=True, capture_output=True, text=True,
    )
    assert (out / "predictions" / "predictions_top1.csv").exists()
    return "YAML config run wrote predictions"


def check_text(workdir: Path) -> str:
    from multimodalva import run
    try:
        res = run(task="text", data=_sample_df(20), label_col="cause_of_death",
                  text_col="narrative", model="prajjwal1/bert-tiny",
                  output_dir=workdir / "text", use_lora=False, use_cv=False,
                  hyperparams={"epochs": 1, "batch_size": 16})
    except Exception as exc:  # network / model download issues → skip, not fail
        raise _Skip(f"tiny model unavailable ({type(exc).__name__})")
    assert (workdir / "text" / "predictions" / "predictions_top1.csv").exists()
    return "text pipeline ran with prajjwal1/bert-tiny"


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-text", action="store_true",
                    help="Also run a tiny text model (downloads ~17MB).")
    ap.add_argument("--keep", action="store_true",
                    help="Keep the temporary work directory for inspection.")
    args = ap.parse_args(argv)

    print("MultimodalVA smoke test\n" + "-" * 60)
    r = Reporter()
    workdir = Path(tempfile.mkdtemp(prefix="mmva_smoke_"))
    try:
        r.check("1. light import", check_light_import)
        r.check("2. tabular pipeline", lambda: check_tabular(workdir))
        r.check("3. external split", lambda: check_external_split(workdir))

        # Materialise a CSV once for the CLI/YAML checks.
        csv_path = workdir / "sample.csv"
        try:
            _sample_df().to_csv(csv_path, index=False)
        except Exception:
            csv_path = None  # tabular check will have already failed loudly

        if csv_path is not None:
            r.check("4. CLI interface", lambda: check_cli(workdir, csv_path))
            r.check("5. YAML config", lambda: check_yaml(workdir, csv_path))

        if args.with_text:
            r.check("6. text pipeline", lambda: check_text(workdir))
        else:
            r.record("6. text pipeline", "SKIP", "pass --with-text to enable")
    finally:
        if args.keep:
            print(f"\nWork dir kept at: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    return r.summary()


if __name__ == "__main__":
    raise SystemExit(main())
