# Changelog

All notable changes to MultimodalVA are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-28

First public release.

MultimodalVA assigns a cause of death from verbal-autopsy records — a free-text
narrative, the questionnaire indicators, or both together — through six
pipelines that share one interface: `text`, `tabular`, `data_fusion`,
`feature_fusion`, `voting` and `stacking`.

What it does and how to install it is in [README.md](README.md); hardware, run
time, disk use, reproducibility, resuming and the common errors are in
[FAQ.md](FAQ.md); the API reference is at <https://y-chu.github.io/MultimodalVA/>.

### Known limitations

- Stacking records absolute paths in `model_sources`, so a stacking run cannot be
  moved between machines without editing them.
- Feature fusion needs `autogluon.multimodal`, which pins torch and transformers
  more tightly than the rest of the package. Install it as the
  `[feature_fusion]` extra into a clean environment.
- A fixed pair of seeds reproduces the predicted labels exactly, but not the
  probabilities to the last decimal, and nothing makes results bit-identical
  across GPU models or library versions. See the
  [FAQ](FAQ.md#how-do-i-make-a-run-reproducible).
- `--dry-run` checks the configuration against the data, not the run: it cannot
  see a batch size that will not fit or a trial budget too small for the search
  space.
- The Ray search backend has been tested on one cluster only. Off a CUDA machine
  both entry points fall back to Optuna by design, so the test suite never
  exercises it — run `tests/hpc_ray_smoke.py` on your own cluster before relying
  on Ray for a long search.
