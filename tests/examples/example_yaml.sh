#!/usr/bin/env bash
# Interface 3 of 3 — driver for the declarative YAML configs.
#
# The three pipelines live in sibling config files; each is one
# `multimodalva run <config>` invocation. YAML is the best fit for reproducible,
# shareable, version-controlled experiments — and the only interface that
# cleanly expresses the voting/stacking base-model spec lists.
#
#   example_config_text.yaml       unimodal text (+ commented Hub publishing)
#   example_config_tabular.yaml    unimodal tabular
#   example_config_ensemble.yaml   stacking (+ commented data/feature fusion)
#
# The configs use `data: demo` (built-in synthetic data), so they run as-is.
# If `multimodalva` is not on PATH, use `python -m multimodalva.cli` instead.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

multimodalva run "${HERE}/example_config_tabular.yaml"
multimodalva run "${HERE}/example_config_text.yaml"
multimodalva run "${HERE}/example_config_ensemble.yaml"

echo "All YAML examples finished."
