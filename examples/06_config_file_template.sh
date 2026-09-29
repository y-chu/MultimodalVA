#!/usr/bin/env bash
# Template 3 of 3 — run a pipeline from a config file.
#
# Copy the config file you want and edit it. Use this route when the settings
# matter more than the code: the config file is the complete, readable record of
# what was run, so it can be version-controlled, attached to a paper, or sent to
# a colleague who can reproduce the run exactly.
#
# Each config below is one `multimodalva run <file>` command. Every key inside
# is one `multimodalva.run()` argument, and the files carry [CHANGE] / [KEEP] /
# [TUNE] tags saying which lines need your attention.
#
#   config_text_model.yaml         a transformer on the narrative
#   config_tabular_model.yaml      a tree/linear model on the symptom indicators
#   config_ensemble_stacking.yaml  several models combined (also shows voting,
#                                  data fusion and feature fusion)
#   config_tabular_model.json      the tabular config again, as JSON — both
#                                  formats are accepted; see examples/README.md
#
# The configs ship with `data: demo` (built-in synthetic data), so they run
# as-is. If `multimodalva` is not on PATH, use `python -m multimodalva.cli`.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# A config file's keys are passed straight through to run(), so a misspelt one
# ('lable_col', 'max_lenght') is only noticed once the pipeline is reached —
# after a cluster job has queued and started. --dry-run does everything up to
# that point and stops: it checks the keys against the pipeline's own arguments,
# loads the data, applies the filters, resolves the feature columns, and reports
# what the run would do without training anything. It exits non-zero if
# something would stop the run, so this gate is worth its few seconds in front
# of anything queued on a GPU.
for cfg in config_tabular_model.yaml config_text_model.yaml \
           config_ensemble_stacking.yaml; do
    multimodalva run "${HERE}/${cfg}" --dry-run
done

multimodalva run "${HERE}/config_tabular_model.yaml"
multimodalva run "${HERE}/config_text_model.yaml"
multimodalva run "${HERE}/config_ensemble_stacking.yaml"

echo "All config-file examples finished."
