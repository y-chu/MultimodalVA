#!/usr/bin/env bash
# Interface 2 of 3 — the `multimodalva` command-line tool.
#
# Best for SLURM/sbatch jobs and non-coders: one line, no Python authoring.
# Installed as a console script by `pip install multimodalva`. If not installed,
# every `multimodalva` below can be replaced with `python -m multimodalva.cli`.
#
# Each flag maps 1:1 to a `multimodalva.run()` argument, so the CLI, the Python
# function, and a YAML config all drive the same pipeline and write the same
# outputs (final/ , predictions/ , hpo/ , diagnostics).
#
# The examples use a CSV exported from the built-in synthetic dataset so they
# run with no external data. Swap in your own CSV, --label, --text-col and
# --features. `set -e` stops at the first error.

set -euo pipefail

# --- Prepare a small synthetic CSV to run against ---------------------------
DATA=/tmp/mmva_cli_sample.csv
python -c "from multimodalva import data; data('va_sample', n_per_class=30).to_csv('${DATA}', index=False)"

# Discover what's available
multimodalva list-datasets
multimodalva list-models

# ---------------------------------------------------------------------------
# 1. Unimodal text
# ---------------------------------------------------------------------------
multimodalva run \
  --task text \
  --data "${DATA}" \
  --label cause_of_death \
  --text-col narrative \
  --model bluebert \
  --output-dir runs/cli/text_bluebert \
  --optimize --n-trials 20 --metric f1_macro

# ---------------------------------------------------------------------------
# 2. Unimodal tabular
# ---------------------------------------------------------------------------
multimodalva run \
  --task tabular \
  --data "${DATA}" \
  --label cause_of_death \
  --features 're:^i\d{3}[a-zA-Z]$' \
  --model lightgbm \
  --output-dir runs/cli/tabular_lightgbm \
  --optimize --n-trials 40 --metric f1_macro

# Row filtering (e.g. adults only) and a fixed external split both work from the
# CLI too:
#   --filters age_grp3=adult
#   --split /path/to/splits.json --id-col record_id

# ---------------------------------------------------------------------------
# 3. Ensemble — data fusion (the ensemble strategy that needs no model-spec lists)
# ---------------------------------------------------------------------------
multimodalva run \
  --task data_fusion \
  --data "${DATA}" \
  --label cause_of_death \
  --text-col narrative \
  --features 're:^i\d{3}[a-zA-Z]$' \
  --model clinicalbigbird \
  --output-dir runs/cli/data_fusion
# NOTE: voting and stacking need per-model spec lists (text_models / tabular_models).
#       Those are lists of dicts — express them in a YAML config (see
#       example_config_*.yaml) rather than on the command line.

# ---------------------------------------------------------------------------
# 4. Train a text model and publish it to the Hugging Face Hub
#    (text + data_fusion only). Requires `huggingface-cli login` first.
# ---------------------------------------------------------------------------
multimodalva run \
  --task text \
  --data "${DATA}" \
  --label cause_of_death \
  --text-col narrative \
  --model bluebert \
  --output-dir runs/cli/text_for_hub \
  --push-to-hub --hub-repo-id your-username/va-text-bluebert

echo "All CLI examples finished."
