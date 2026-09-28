# Contributing to MultimodalVA

Thanks for your interest in the project. Bug reports, questions and pull
requests are all welcome.

## Getting help or reporting a problem

Open an issue at <https://github.com/y-chu/MultimodalVA/issues>.

For a bug, please include:

- what you ran (the `run(...)` call, CLI command or YAML config),
- what happened, with the full error message and traceback,
- your Python version, operating system, and whether you are on CUDA, Apple
  Silicon (MPS) or CPU,
- the package version (`python -c "import multimodalva; print(multimodalva.__version__)"`).

If the problem involves your own data, please **do not attach real records**.
Reproduce it with the built-in synthetic data where you can —
`from multimodalva import data; df = data("va_sample")`.

For questions about how to use the package, open an issue as well; there is no
separate forum.

## Development setup

```bash
git clone https://github.com/y-chu/MultimodalVA.git
cd MultimodalVA
pip install -e ".[all,dev]"
```

Three parts, and they do different things:

- **`-e`** installs in editable mode, so your edits take effect without
  reinstalling. This is what makes the checkout editable — not the `dev` extra.
- **`dev`** is exactly `pytest`, the one thing the unit tests need that the
  package itself does not. CI installs the same extra, so there is a single
  declaration of what running the tests requires.
- **`all`** pulls in the optional LoRA, Ray and feature-fusion paths so the full
  test matrix can run.

`smoke_test.py` and `diagnose.py` need none of this — they use only the standard
library and the package itself, and both ship in the source distribution, so they
can be run straight after `pip install` to check an environment.

## Running the tests

Three layers. Only the first is collected by `pytest`; the other two are named so
that it skips them, because they train real models:

```bash
pytest -q                           # unit tests, ~4 min
python tests/smoke_test.py          # is it alive: offline, CPU-only, ~45 s
python tests/smoke_test.py --with-text   # also trains a tiny text model
python tests/diagnose.py            # every tabular model and the offline
                                    # pipelines, ~3 min
python tests/diagnose.py --with-text     # adds the text pipelines, ~10-15 min
```

Timings are from an Apple Silicon laptop with no CUDA GPU; a cluster node is
faster and a cold model download is slower.

There is a fourth, for the one thing the other three cannot reach. Off a CUDA
machine the Ray entry points hand the search back to Optuna by design, so the
suite exercises that redirect rather than Ray itself. `tests/hpc_ray_smoke.py`
runs the Ray path end to end on a GPU node, with `hpc_ray_smoke.sbatch` as a SLURM
wrapper whose cluster-specific lines are marked `EDIT`.

A green `pytest` run does **not** mean the pipelines work — run `diagnose.py`
before a release. It exits non-zero on failure and prints, for each failing cell,
the last line inside `multimodalva` that ran, so there is a file:line to open.

Please make sure `pytest` and `smoke_test.py` pass before opening a pull request.
Both run in CI on every push.

## Pull requests

- Open an issue first for anything substantial, so the approach can be discussed
  before you spend time on it.
- Keep the change focused; unrelated cleanups are easier to review separately.
- Match the surrounding style: type hints on public functions, Google-style
  docstrings, `logging` rather than `print`, and `ValueError` with an actionable
  message when validating inputs.
- Add or update a test when you fix a bug or add behaviour.
- Update the README or FAQ if you change something a user would notice.

## Adding a model

Text backbones live in `multimodalva/text/models.py` (`TEXT_MODELS`), and
tabular models in `multimodalva/tabular/train.py` (`TABULAR_MODELS`) with a
matching search space in `tabular/search_spaces.py`. Add the alias in both the
registry and the README table so `multimodalva list-models` and the docs agree.

## Code of conduct

Please be respectful and constructive. Harassment or personal attacks are not
acceptable in issues, pull requests or any other project space.

## License

By contributing you agree that your contributions are licensed under the
project's MIT license.
