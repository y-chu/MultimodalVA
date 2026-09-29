"""Ray Tune helpers shared by the text and tabular HPO backends.

Both backends need the same two things, and Ray's API has moved around between
2.x releases, so the version handling lives here rather than in each backend:

    to_ray_space(search_space)  — convert our search-space format to Ray's
    get_ray_trial_dir()         — find the current trial's artifact directory

Ray itself is imported inside the functions, so importing this module does not
require Ray to be installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def to_ray_space(search_space: dict) -> dict:
    """Convert our ``(type, *args)`` search space format to Ray Tune's format.

    Mapping::

        ("float_log", low, high)  →  tune.loguniform(low, high)
        ("float",     low, high)  →  tune.uniform(low, high)
        ("int",       low, high)  →  tune.randint(low, high)
        ("categorical", [vals])   →  tune.choice([vals])

    Raises:
        ImportError: If ``ray[tune]`` is not installed.
        ValueError:  If an unknown type string is encountered.
    """
    try:
        from ray import tune
    except ImportError as exc:
        raise ImportError(
            "Ray Tune is required for optimize_text_ray() / optimize_tabular_ray(). "
            "It lives in an optional extra: pip install 'multimodalva[ray]'"
        ) from exc

    ray_space: dict = {}
    for key, spec in search_space.items():
        if not isinstance(spec, (tuple, list)) or not spec:
            raise ValueError(
                f"Search space entry {key!r} has an invalid spec {spec!r}. "
                "Each entry must be a non-empty tuple: "
                "('float_log', low, high), ('float', low, high), "
                "('int', low, high), or ('categorical', [values]). "
                f"Got type {type(spec).__name__!r}."
            )
        kind, *args = spec
        if kind == "float_log":
            ray_space[key] = tune.loguniform(args[0], args[1])
        elif kind == "float":
            ray_space[key] = tune.uniform(args[0], args[1])
        elif kind == "int":
            ray_space[key] = tune.randint(args[0], args[1])
        elif kind == "categorical":
            ray_space[key] = tune.choice(args[0])
        else:
            raise ValueError(
                f"Search space entry {key!r} has unknown type {kind!r} "
                f"(full spec: {spec!r}). "
                "Valid types: 'float_log', 'float', 'int', 'categorical'. "
                "Example: ('categorical', [8, 16, 32]) or ('float_log', 1e-5, 1e-4)."
            )
    return ray_space


def get_ray_trial_dir() -> Path:
    """Return the current Ray Tune trial directory across Ray 2.x variants."""
    try:
        from ray import tune as _ray_tune

        if hasattr(_ray_tune, "get_context"):
            ctx = _ray_tune.get_context()
            if ctx is not None:
                return Path(ctx.get_trial_dir())
    except Exception:
        logger.debug("ray.tune.get_context() unavailable; trying legacy APIs.", exc_info=True)

    try:
        from ray.air import session as _air_session

        trial_dir = _air_session.get_trial_dir()
        if trial_dir:
            return Path(trial_dir)
    except Exception:
        logger.debug("ray.air.session.get_trial_dir() unavailable.", exc_info=True)

    cwd = Path.cwd()
    if cwd.exists():
        logger.warning(
            "Falling back to current working directory for Ray trial artifacts: %s",
            cwd,
        )
        return cwd

    raise RuntimeError(
        "Unable to resolve the Ray Tune trial directory. "
        "Install a supported Ray Tune version or update the compatibility shim."
    )


# Ray Tune bookkeeping columns: present in every ResultGrid dataframe, carry no
# information about the search itself, and are dropped when the trials table is
# written in the shape every reader expects.
_RAY_BOOKKEEPING = frozenset({
    "trial_id", "date", "timestamp", "pid", "hostname", "node_ip", "done",
    "training_iteration", "time_this_iter_s", "time_since_restore",
    "iterations_since_restore", "checkpoint_dir_name", "logdir",
    "should_checkpoint", "experiment_tag",
})


def ray_trials_to_contract(df, metric: str):
    """Map a Ray ``ResultGrid`` dataframe onto the trials-table contract.

    Both backends write ``hpo/hpo_trials.csv``, but until 2026-09-28 the Ray path
    wrote Ray's own dataframe into it: bare metric columns, ``config/<name>``,
    ``trial_id``, ``logdir``. Every reader of that file expects Optuna's shape —
    ``results/validation.py`` keys on ``value`` and ``user_attrs_*`` (and counts
    folds from ``user_attrs_fold_<i>_<metric>``), ``results/vis_train.py`` on
    ``value``/``duration``/``params_*``. Matching only the *file name* left
    ``validation.json`` reading ``source="none"`` for a Ray-searched run and no
    fold stability anywhere, which is what hpc_ray_smoke checks 4 and 5 caught on
    the first real GPU run.

    Args:
        df:     ``results.get_dataframe()`` from a finished ``Tuner.fit()``.
        metric: The objective, which becomes the ``value`` column.

    Returns:
        A new DataFrame; the input is not modified.
    """
    import pandas as pd

    out = df.copy()

    # Optuna's trials_dataframe() carries datetime_start / datetime_complete, and
    # readers use them to tell whether trials overlapped. Ray reports `timestamp`
    # (when the result came in) and `time_total_s`, so the pair is recoverable.
    _start = _complete = None
    if "timestamp" in out.columns:
        _complete = pd.to_datetime(out["timestamp"], unit="s", errors="coerce")
        if "time_total_s" in out.columns:
            _start = _complete - pd.to_timedelta(out["time_total_s"], unit="s",
                                                 errors="coerce")

    renamed = {}
    for col in out.columns:
        if col in _RAY_BOOKKEEPING:
            continue
        if col.startswith("config/"):
            renamed[col] = f"params_{col[len('config/'):]}"
        elif col == "time_total_s":
            renamed[col] = "duration"
        else:
            renamed[col] = f"user_attrs_{col}"
    out = out[list(renamed)].rename(columns=renamed)

    # `value` is the objective, as Optuna's trials_dataframe() spells it.
    if _complete is not None:
        out["datetime_complete"] = _complete
    if _start is not None:
        out["datetime_start"] = _start

    objective = f"user_attrs_{metric}"
    if objective in out.columns:
        out.insert(0, "value", out[objective])
    out.insert(0, "number", range(len(out)))
    return out
