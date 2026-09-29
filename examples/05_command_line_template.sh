#!/usr/bin/env bash
# Template 2 of 3 — the `multimodalva` command-line tool.
#
# Copy this file and edit it. Use it when you do not want to write Python at
# all, and for cluster jobs (SLURM/sbatch), where one command per job is the
# natural unit. `multimodalva` is installed alongside the package; if it is not
# on your PATH, replace every `multimodalva` below with
# `python -m multimodalva.cli`.
#
# Each flag is one `multimodalva.run()` argument, so this file, the Python
# template and a config file drive the same pipeline and write the same outputs
# (final/ , predictions/ , hpo/ , diagnostics).
#
# To run it on your own data, change four things in each command below:
#
#   --data        the path to your CSV
#   --label-col   your cause-of-death column
#   --text-col    your narrative column
#   --features    your indicator columns: a regex, or a comma-separated list
#
# Everything else already has a working default. `set -e` stops at the first
# error.
#
# Add --dry-run to any command below to check it against your data and stop
# before anything is trained: it reports the rows, classes, split and feature
# columns the run would use, names anything that would stop it, and exits
# non-zero so an sbatch script can gate on it. Worth its few seconds in front of
# a queued job.

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
  --label-col cause_of_death \
  --text-col narrative \
  --model bluebert \
  --output-dir runs/cli/text_bluebert \
  --optimize --n-trials 20 --optimize-metric f1_macro

# ---------------------------------------------------------------------------
# 2. Unimodal tabular
# ---------------------------------------------------------------------------
multimodalva run \
  --task tabular \
  --data "${DATA}" \
  --label-col cause_of_death \
  --features 're:^i\d{3}[a-zA-Z]$' \
  --model lightgbm \
  --output-dir runs/cli/tabular_lightgbm \
  --optimize --n-trials 40 --optimize-metric f1_macro

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
  --label-col cause_of_death \
  --text-col narrative \
  --features 're:^i\d{3}[a-zA-Z]$' \
  --model clinicalbigbird \
  --output-dir runs/cli/data_fusion
# NOTE: voting and stacking each combine a list of models, and a list of models
#       is awkward to type on a command line. Use a config file for those —
#       see config_ensemble_stacking.yaml.

# ---------------------------------------------------------------------------
# 4. Train a text model and publish it to the Hugging Face Hub
#    (text + data_fusion only). Requires `huggingface-cli login` first.
# ---------------------------------------------------------------------------
multimodalva run \
  --task text \
  --data "${DATA}" \
  --label-col cause_of_death \
  --text-col narrative \
  --model bluebert \
  --output-dir runs/cli/text_for_hub \
  --push-to-hub --hub-repo-id your-username/va-text-bluebert

echo "All CLI examples finished."
