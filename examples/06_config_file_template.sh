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

multimodalva run "${HERE}/config_tabular_model.yaml"
multimodalva run "${HERE}/config_text_model.yaml"
multimodalva run "${HERE}/config_ensemble_stacking.yaml"

echo "All config-file examples finished."
