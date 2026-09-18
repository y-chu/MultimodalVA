r"""
Command-line entry point for MultimodalVA.

Installed as the ``multimodalva`` console script (see ``[project.scripts]`` in
``pyproject.toml``). It is a thin shell over :func:`multimodalva.runner.run` —
every flag maps 1:1 onto a ``run()`` argument, so the CLI, the YAML config, and
the Python function all drive the same implementation and produce the same
outputs.

Examples
--------
    # Unimodal text, one line — suitable for an sbatch script
    multimodalva run --task text --data clean.csv --label cause \
        --text-col narrative --model bluebert --output-dir runs/bluebert --optimize

    # Unimodal tabular with a regex feature selector
    multimodalva run --task tabular --data clean.csv --label cause \
        --features "re:^i\d{3}[a-zA-Z]$" --model lightgbm --output-dir runs/lgbm

    # Everything from a declarative config file (YAML or JSON)
    multimodalva run experiment.yaml

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


def main(argv: list[str] | None = None) -> int:
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
    run_p.add_argument("--label", help="Label (cause-of-death) column.")
    run_p.add_argument("--output-dir", dest="output_dir", help="Artifacts root.")
    run_p.add_argument("--text-col", dest="text_col", help="Narrative column.")
    run_p.add_argument("--features", help="Comma-separated feature list, "
                       "'re:PATTERN', or 'auto'.")
    run_p.add_argument("--model", help="Model name/alias/checkpoint.")
    run_p.add_argument("--filters", help="Row filter, e.g. 'age_grp3=adult,site=A'.")
    run_p.add_argument("--optimize", action="store_true", default=None,
                       help="Enable HPO before final training.")
    run_p.add_argument("--n-trials", dest="n_trials", type=int, help="HPO trials.")
    run_p.add_argument("--metric", help="HPO metric (default f1_macro).")
    run_p.add_argument("--test-size", dest="test_size", type=float,
                       help="Random-split test fraction (default 0.2).")
    run_p.add_argument("--random-state", dest="random_state", type=int,
                       help="Seed (default 42).")
    run_p.add_argument("--split", help="Fixed-split column name or path to a "
                       "splits JSON ({train_ids,test_ids}).")
    run_p.add_argument("--id-col", dest="id_col", help="Row-id column for "
                       "id-based --split.")
    run_p.add_argument("--top-k", dest="top_k", type=int, help="Top-K to report.")
    run_p.add_argument("--push-to-hub", dest="push_to_hub", action="store_true",
                       default=None, help="Publish the trained model to the HF Hub "
                       "(text / data_fusion).")
    run_p.add_argument("--hub-repo-id", dest="hub_repo_id",
                       help="Target HF Hub repo id (e.g. user/model).")

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

    if "task" not in cfg or "data" not in cfg or "label" not in cfg:
        print("error: --task, --data and --label are required (via flags or "
              "config file).", file=sys.stderr)
        return 2
    cfg.setdefault("output_dir", "runs/mmva")

    # Convenience: 'demo' loads the built-in synthetic dataset in-memory.
    # n_per_class=30 keeps every cause large enough for a stratified train/test
    # split; the registry default (6) is too small for the full cause list.
    if cfg.get("data") == "demo":
        from .datasets import data as _data
        cfg["data"] = _data("va_sample", n_per_class=30)

    from .runner import run
    run(**cfg)
    print(f"Done. Artifacts in: {cfg['output_dir']}")
    return 0


def _cmd_list_datasets() -> int:
    from .datasets import list_datasets

    for name, desc in list_datasets().items():
        print(f"{name:24s} {desc}")
    return 0


def _cmd_list_models() -> int:
    from .text.models import MODEL_DESCRIPTIONS, REMOTE_MODELS, SUPPORTED_MODELS

    print("# Text models (Hugging Face Hub)")
    for alias, checkpoint in SUPPORTED_MODELS.items():
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
            import yaml  # optional; PyYAML is pulled in transitively by Ray
        except ImportError as exc:
            raise SystemExit(
                "PyYAML is required to read YAML configs. Install it "
                "('pip install pyyaml') or use a JSON config file instead."
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _cli_overrides(args: argparse.Namespace) -> dict:
    """Collect only the flags the user actually set, mapped to run() kwargs."""
    out: dict[str, Any] = {}
    simple = [
        "task", "data", "label", "output_dir", "text_col", "model",
        "n_trials", "metric", "test_size", "random_state", "split", "id_col",
        "top_k", "hub_repo_id",
    ]
    for key in simple:
        val = getattr(args, key, None)
        if val is not None:
            out[key] = val
    if args.optimize:
        out["optimize"] = True
    if args.push_to_hub:
        out["push_to_hub"] = True
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
