"""Where a model's hyperparameters come from, as one argument.

A model can get its hyperparameters in three ways, and they used to be easy to
confuse. Now ``hyperparams=`` says which one, and nothing else does::

    hyperparams={"learning_rate": 3e-5}    use exactly these
    hyperparams="default"                  the model library's own defaults
    hyperparams=Optimize(...)              search for them, configured like this

``Optimize`` carries the search settings with it, rather than leaving them as
loose arguments that mean nothing unless a flag is set. That matters most for
ensembles: a base-model spec holds one ``hyperparams`` key instead of six, and a
pipeline forwarding base-model settings passes one object instead of re-exporting
each knob by hand — which is what had silently drifted before.

This replaces ``use_optimize=True`` with ``n_trials=``, ``optimize_metric=``
and ``search_space=`` alongside it; those arguments are gone from every
classifier and from ``run()``. The low-level :func:`optimize_text` and
:func:`optimize_tabular` keep those as ordinary arguments: searching is all they
do, so an object saying "please search" would add nothing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import MISSING, dataclass, field, fields, replace
from typing import Any

__all__ = [
    "Optimize",
    "DEFAULT",
    "BACKENDS",
    "resolve_hyperparams",
    "resolve_spec_hyperparams",
    "optimize_from_flags",
    "resolve_search_resume",
    "resolve_backend",
    "resolve_pruning",
    "ray_is_available",
]

logger = logging.getLogger(__name__)

#: Accepted values for ``Optimize(backend=...)``.
BACKENDS = ("auto", "optuna", "ray")

#: ``cv_folds`` value meaning "as many folds as this data supports".
CV_FOLDS_AUTO = "auto"

#: Folds a search uses when nothing else is said, and the ceiling ``"auto"``
#: works down from. Three balances cost (each trial trains k times) against a
#: steadier estimate.
CV_FOLDS_DEFAULT = 3


#: ``hyperparams=DEFAULT`` and ``hyperparams="default"`` are the same thing:
#: train with whatever the underlying library uses when told nothing.
DEFAULT = "default"


#: The four forms a search-space entry can take. ``sample_hyperparams()`` and
#: ``to_ray_space()`` both understand exactly these.
SEARCH_SPACE_KINDS = ("float", "float_log", "int", "categorical")


def validate_search_space(space: dict | None, *, where: str = "Optimize(space=...)") -> None:
    """Raise on a malformed search space, naming every problem at once.

    Worth doing at construction time rather than at sampling time. The searches
    run ``study.optimize(..., catch=(Exception,))`` so that one bad configuration
    cannot kill a long search — which means a malformed *space* does not fail
    loudly: every trial raises, every raise is swallowed, and the run reaches the
    end with nothing completed. On a cluster that is the queue wait plus the whole
    trial budget before anything says why. The errors that surfaced from inside
    the sampler were also unreadable: a NaN bound arrived as
    ``OverflowError: Range exceeds valid bounds`` out of NumPy, and a ``None``
    bound as ``TypeError: '>' not supported between instances of 'NoneType' and
    'float'``.

    Args:
        space: The caller's search space, or ``None``/empty (both fine).
        where: How to refer to it in the message.

    Raises:
        ValueError: Listing every malformed entry, with what was expected.
    """
    if not space:
        return

    problems: list[str] = []
    for name, spec in space.items():
        if not isinstance(spec, (tuple, list)) or len(spec) < 2:
            problems.append(
                f"  {name}: {spec!r} — expected a tuple, e.g. "
                f"('float', low, high) or ('categorical', [choices])"
            )
            continue
        kind = spec[0]
        if kind not in SEARCH_SPACE_KINDS:
            problems.append(
                f"  {name}: kind {kind!r} is not one of "
                f"{', '.join(repr(k) for k in SEARCH_SPACE_KINDS)}"
            )
            continue
        if kind == "categorical":
            choices = spec[1]
            if not isinstance(choices, (list, tuple)):
                problems.append(
                    f"  {name}: ('categorical', {choices!r}) — the second element "
                    "must be a list of choices"
                )
            elif len(choices) == 0:
                problems.append(
                    f"  {name}: ('categorical', []) — needs at least one choice. "
                    "A single choice pins the value, which is how a hyperparameter "
                    "is held fixed while others are searched."
                )
            continue
        # float / float_log / int: a numeric, finite, ordered pair
        if len(spec) != 3:
            problems.append(
                f"  {name}: ({kind!r}, ...) takes exactly a low and a high, got "
                f"{len(spec) - 1} value(s)"
            )
            continue
        low, high = spec[1], spec[2]
        for label, value in (("low", low), ("high", high)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"  {name}: {label}={value!r} is not a number")
            elif value != value or value in (float("inf"), float("-inf")):
                problems.append(
                    f"  {name}: {label}={value!r} is not finite; a NaN bound "
                    "reaches the sampler as an unreadable OverflowError"
                )
        if (isinstance(low, (int, float)) and isinstance(high, (int, float))
                and not isinstance(low, bool) and not isinstance(high, bool)
                and low == low and high == high and low > high):
            problems.append(f"  {name}: low={low!r} is greater than high={high!r}")

    if problems:
        raise ValueError(
            f"{where} is malformed:\n" + "\n".join(problems) + "\n"
            "Each entry is ('float'|'float_log'|'int', low, high) or "
            "('categorical', [choices]). Validated here rather than at sampling "
            "time, because a search swallows per-trial errors and would otherwise "
            "run its whole budget with every trial failing."
        )


@dataclass(frozen=True)
class Optimize:
    """Search for this model's hyperparameters, with these settings.

    Every field has a working default, so ``Optimize()`` on its own is a complete
    instruction: search this model with the package's adaptive space for the
    data's size and cause count.

    Two ``Optimize`` objects with the same settings compare equal. They are not
    hashable, because ``space`` and ``extra`` are dicts.

    Args:
        metric:   What the search maximises. ``"f1_macro"`` weighs every cause
                  equally; ``"csmf_accuracy"`` scores the cause distribution of
                  the population rather than individual deaths. Also accepts
                  ``"accuracy"``, ``"balanced_accuracy"``, ``"f1_weighted"``.
        n_trials: How many configurations to try. ``None`` uses the family
                  default — 30 for text, 50 for tabular, since one text trial
                  costs far more.
        space:    Overrides the adaptive default space, one key at a time. Keys
                  you leave out keep their adaptive value, and the run log
                  records which ones you replaced. Same spec format as
                  ``text/search_spaces.py``: ``("float_log", lo, hi)``,
                  ``("float", lo, hi)``, ``("int", lo, hi)``,
                  ``("categorical", [...])``.
        cv:       Score each configuration by cross-validation rather than one
                  validation split. Steadier, and costs ``cv_folds`` times more.
        cv_folds: Folds used when ``cv`` is True. An integer is taken literally
            and a search **raises** when the rarest class in the training rows
            has fewer rows than that, because stratified folds cannot then hold
            every class. ``"auto"`` instead uses as many folds as the data
            supports, up to :data:`CV_FOLDS_DEFAULT`, and says so in the log —
            for data whose rarest cause is not known in advance. Default 3.
        resume:   Reattach to an interrupted search in the same output
                  directory and carry on, instead of starting over. ``None``
                  (default) follows the run's own ``resume=``, so one switch
                  covers the whole run; set it only to treat the search
                  differently from the rest.
        pruning:  Stop unpromising configurations early. ``None`` (default) uses
                  the family default — on for text, off for tabular, because a
                  text trial costs minutes and a tree-model trial costs seconds.
                  On the Optuna backend this is Optuna's ``MedianPruner``,
                  comparing a configuration against the median of the
                  configurations finished so far. **Ignored by the Ray
                  backend**, which has no equivalent: use
                  ``Optimize(extra={"use_asha": True})`` there, which terminates
                  unpromising trials by successive halving instead.
        backend:  Which search engine runs the trials.

                  * ``"auto"`` (default) — Ray when a CUDA GPU is visible or
                    SLURM is detected, Optuna otherwise. This is the same
                    predicate the low-level functions already apply when they
                    redirect a GPU-less Ray call back to Optuna.
                  * ``"optuna"`` — always Optuna: one process, trials in
                    sequence, resumable through its journal file.
                  * ``"ray"`` — always Ray Tune: trials in parallel across the
                    CPUs, GPUs or cluster nodes Ray can see. Falls back to
                    Optuna with a warning when no CUDA GPU is available, since
                    a single-machine MPS/CPU run gains nothing from it.

                  Ray's cluster and resource settings have no fields of their
                  own; pass them through ``extra``, which reaches the underlying
                  function verbatim::

                      Optimize(backend="ray", extra={
                          "ray_address": "auto",
                          "num_gpus_per_trial": 0.5,
                          "max_concurrent_trials": 4,
                          "use_asha": True,
                      })
        space_profile: Tabular only. Which adaptive profile to start from;
                       ``"auto"`` picks one from the feature count.
        extra:         Anything else to hand the underlying optimize function.

    ``use_lora`` and ``use_focal`` are deliberately NOT here. They change how the
    model is trained whether or not a search happens, so they belong beside
    ``model`` — on ``run()``, or in the base-model spec — not inside an object
    that only exists when searching. A search picks them up from there and adds
    their hyperparameters to the space.
    """

    metric: str = "f1_macro"
    n_trials: int | None = None
    space: dict | None = None
    cv: bool = True
    cv_folds: "int | str" = 3
    resume: bool | None = None
    pruning: bool | None = None
    backend: str = "auto"

    # tabular-only
    space_profile: str = "auto"

    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Copy the dict fields. Freezing the dataclass stops the fields being
        # rebound, but not the dicts being edited in place, so without this a
        # caller could change a Optimize after handing it over — and in an
        # ensemble the same Optimize is often reused across base models.
        object.__setattr__(self, "space", dict(self.space) if self.space else self.space)
        object.__setattr__(self, "extra", dict(self.extra))

        validate_search_space(self.space)

        if not (self.cv_folds == CV_FOLDS_AUTO
                or (isinstance(self.cv_folds, int)
                    and not isinstance(self.cv_folds, bool)
                    and self.cv_folds >= 2)):
            raise ValueError(
                f"cv_folds={self.cv_folds!r} is not valid. Use an integer of 2 "
                f'or more, or {CV_FOLDS_AUTO!r} to use as many folds as the '
                f"rarest class supports (up to {CV_FOLDS_DEFAULT})."
            )

        if self.backend not in BACKENDS:
            raise ValueError(
                f"backend={self.backend!r} is not recognised. Use "
                f"{', '.join(repr(b) for b in BACKENDS)}. "
                '"auto" picks Ray on a CUDA/SLURM machine and Optuna otherwise.'
            )

    def with_defaults(self, **kwargs: Any) -> "Optimize":
        """Return a copy with only the unset fields filled in.

        Used where a caller's run-level setting should apply to a base model
        that did not state its own.
        """
        updates = {
            k: v for k, v in kwargs.items()
            if v is not None and getattr(self, k) == getattr(_UNSET_REFERENCE, k)
        }
        return replace(self, **updates) if updates else self

    def describe(self) -> str:
        """One line for the run log, naming what was actually configured."""
        parts = [f"metric={self.metric}"]
        parts.append(f"n_trials={self.n_trials if self.n_trials is not None else 'family default'}")
        parts.append(f"cv={self.cv}" + (f"/{self.cv_folds} folds" if self.cv else ""))
        # Log the resolved engine, not the literal "auto": the run log has to say
        # which backend actually ran the trials.
        _resolved = resolve_backend(self.backend, warn=False)
        parts.append(
            f"backend={_resolved}"
            + (f" (from {self.backend})" if self.backend != _resolved else "")
        )
        parts.append(
            f"pruning={self.pruning if self.pruning is not None else 'family default'}"
        )
        if self.space:
            parts.append(f"space overrides {len(self.space)} key(s): {', '.join(sorted(self.space))}")
        if self.space_profile != "auto":
            parts.append(f"profile={self.space_profile}")
        if self.resume is False:
            parts.append("resume=off")
        if self.extra:
            parts.append(f"extra: {', '.join(sorted(self.extra))}")
        return ", ".join(parts)


_UNSET_REFERENCE = Optimize()


def resolve_hyperparams(value: Any) -> tuple[str, dict | None, "Optimize | None"]:
    """Work out which of the three cases ``hyperparams=`` is.

    Args:
        value: What the caller passed: a dict of fixed values, the string
               ``"default"``, a :class:`Optimize`, or ``None``.

    Returns:
        ``(kind, fixed, search)`` where ``kind`` is ``"fixed"``, ``"default"``
        or ``"search"``. Exactly one of ``fixed`` / ``search`` is set.

    Raises:
        TypeError:  The value is not one of the accepted forms.
        ValueError: The value is a string other than ``"default"`` — most
                    likely ``"search"``, which has to be ``Optimize()`` because a
                    bare string cannot carry the search settings.
    """
    if value is None:
        return ("default", None, None)
    if isinstance(value, Optimize):
        return ("search", None, value)
    if isinstance(value, str):
        if value.lower() == DEFAULT:
            return ("default", None, None)
        if value.lower() == "search":
            raise ValueError(
                'hyperparams="search" is not accepted, because a search needs '
                "settings that a string cannot carry. Use Optimize() for the "
                "package defaults, or e.g. "
                'Optimize(metric="csmf_accuracy", n_trials=50).'
            )
        raise ValueError(
            f"hyperparams={value!r} is not a recognised keyword. Pass a dict of "
            'fixed values, "default" for the model library\'s own defaults, or '
            "Optimize(...) to search."
        )
    if isinstance(value, dict):
        return ("fixed", dict(value), None)
    raise TypeError(
        f"hyperparams must be a dict, \"default\", or Optimize(...), "
        f"got {type(value).__name__}."
    )


def ray_is_available() -> bool:
    """Whether the ``ray`` package can be imported.

    Ray lives in the ``[ray]`` extra, so a normal install does not have it. This
    is a spec lookup rather than an import: importing ray costs seconds and
    starts background machinery, and this is called just to resolve a string.
    """
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("ray") is not None


def resolve_backend(backend: str, *, warn: bool = True) -> str:
    """Turn ``"auto"`` into the engine that will actually run, ``"optuna"`` or ``"ray"``.

    ``"auto"`` means Ray where parallel trials pay off **and** Ray is installed:
    a visible CUDA GPU or a SLURM allocation, plus the ``[ray]`` extra. Optuna
    everywhere else. The hardware half is deliberately the same predicate the
    low-level Ray functions use when they redirect a GPU-less call back to
    Optuna, so ``"auto"`` never resolves to a backend that would only turn around
    and hand the work back.

    When the machine looks like one where Ray would have paid off but ray is not
    installed, this **warns** rather than failing. Falling back silently would be
    worse than an error: a submitted GPU job would run its trials one at a time
    and still produce correct results, so nobody would notice until the wall
    clock did.

    ``backend="ray"`` asked for explicitly is never downgraded here — the
    underlying function raises an ``ImportError`` naming the extra, because an
    explicit request deserves an explicit failure.

    ``warn=False`` resolves silently, for callers that are only formatting the
    value rather than about to act on it — :meth:`Optimize.describe` does this, so
    one search logs the warning once instead of once per mention.

    Importing torch is deferred: this is called while logging a run, and on the
    Optuna path torch may not be loaded yet.
    """
    if backend != "auto":
        return backend

    wants_ray = False
    if os.environ.get("SLURM_JOB_ID"):
        wants_ray = True
    else:
        try:
            import torch  # noqa: PLC0415

            wants_ray = torch.cuda.is_available()
        except Exception:  # torch missing, or a driver that raises on probe
            logger.debug("CUDA probe failed while resolving backend='auto'.", exc_info=True)

    if not wants_ray:
        return "optuna"
    if ray_is_available():
        return "ray"

    if not warn:
        return "optuna"

    logger.warning(
        "backend='auto' resolved to Optuna because Ray is not installed, on a "
        "machine where Ray would have run trials in parallel (%s). The search "
        "will run one trial at a time and the results will be correct — only "
        "slower. Install the extra to use it:  pip install 'multimodalva[ray]'. "
        "Pass backend='optuna' to silence this.",
        "SLURM allocation detected" if os.environ.get("SLURM_JOB_ID") else "CUDA GPU detected",
    )
    return "optuna"


def resolve_pruning(search: "Optimize", family_default: bool) -> bool:
    """Whether trial pruning is on, given the family's own default.

    ``Optimize(pruning=None)`` — the default — means "whatever this model family
    does normally": on for text, off for tabular. Anything else is the caller's
    explicit choice and wins.

    Args:
        search:         The search settings.
        family_default: ``enable_pruning``'s default for this family —
                        ``True`` for text, ``False`` for tabular.
    """
    return bool(family_default) if search.pruning is None else bool(search.pruning)


def resolve_cv_folds(
    n_cv_folds: "int | str",
    labels,
    *,
    where: str = "the search",
    log: "logging.Logger | None" = None,
) -> int:
    """Folds to cross-validate with, given what the rarest class supports.

    ``StratifiedKFold`` needs at least as many rows of every class as there are
    folds; below that a fold cannot contain the class at all, and the fold score
    silently stops meaning what it says. All four search paths therefore refuse
    an integer ``n_cv_folds`` the data cannot honour.

    ``"auto"`` is the way to say "fit the folds to the data": it takes
    ``min(CV_FOLDS_DEFAULT, rarest class)`` and logs the reduction. It still
    raises when the rarest class has a single row, because there is no valid
    fold count for that — group the rare causes, or use ``cv=False``.

    Args:
        n_cv_folds: An integer, or ``"auto"``.
        labels: Training labels, in any form ``numpy.unique`` accepts.
        where: Named in the messages ("the tabular search").
        log: Logger for the reduction notice. Defaults to this module's.

    Returns:
        The fold count to use.

    Raises:
        ValueError: An integer the data cannot honour, or a singleton class.
    """
    import numpy as np  # noqa: PLC0415

    log = log or logger
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    rarest = int(counts.min()) if counts.size else 0

    if n_cv_folds == CV_FOLDS_AUTO:
        folds = min(CV_FOLDS_DEFAULT, rarest)
        if folds < 2:
            raise ValueError(
                f'cv_folds="auto" cannot cross-validate {where}: the rarest '
                f"class in the training rows has {rarest} row(s), and stratified "
                "folds need at least 2. Group the rare causes, or pass "
                "Optimize(cv=False) to search on a single holdout instead."
            )
        if folds < CV_FOLDS_DEFAULT:
            log.warning(
                'cv_folds="auto": using %d folds for %s instead of the default '
                "%d, because the rarest class in the training rows has %d row(s) "
                "and a stratified fold cannot hold fewer. Trial scores average "
                "over %d folds, so they are noisier than a %d-fold search and "
                "are not comparable with one.",
                folds, where, CV_FOLDS_DEFAULT, rarest, folds, CV_FOLDS_DEFAULT,
            )
        return folds

    folds = int(n_cv_folds)
    if folds > rarest:
        raise ValueError(
            f"n_cv_folds={folds} is greater than the minimum class count "
            f"({rarest}) in the training rows for {where}, so a stratified fold "
            f"cannot contain every class. Lower cv_folds, pass "
            f'cv_folds="auto" to use the {min(CV_FOLDS_DEFAULT, rarest)} fold(s) '
            "this data supports, group the rare causes, or use "
            "Optimize(cv=False)."
        )
    return folds


def resolve_search_resume(search: "Optimize", run_resume: bool) -> bool:
    """Whether a search should reattach to an interrupted study.

    ``Optimize(resume=None)`` — the default — follows the run's ``resume=``, so
    ``resume=False`` on the run restarts everything, search included.
    """
    return bool(run_resume) if search.resume is None else bool(search.resume)


def optimize_from_flags(
    optimize: bool,
    n_trials: int | None = None,
    optimize_metric: str | None = None,
    search_space: dict | None = None,
    **kwargs: Any,
) -> "Optimize | None":
    """Build an :class:`Optimize` from flat flags.

    For the CLI, whose surface is flat and cannot carry an object. Returns
    ``None`` when ``optimize`` is false, so the caller falls through to whatever
    ``hyperparams=`` says.
    """
    if not optimize:
        return None
    fields = {k: v for k, v in kwargs.items() if v is not None}
    if n_trials is not None:
        fields["n_trials"] = n_trials
    if optimize_metric is not None:
        fields["metric"] = optimize_metric
    if search_space is not None:
        fields["space"] = search_space
    return Optimize(**fields)


#: Spec keys that used to configure a search. They now live on :class:`Optimize`,
#: and a spec still carrying one is an error rather than a silent no-op.
_REMOVED_SPEC_KEYS = {
    "use_optimize":         'hyperparams=Optimize(...)',
    "n_trials":             "Optimize(n_trials=...)",
    "optimize_metric":      "Optimize(metric=...)",
    "search_space":         "Optimize(space=...)",
    "use_cv":               "Optimize(cv=...)",
    "n_cv_folds":           "Optimize(cv_folds=...)",
    "resume_hpo":           "Optimize(resume=...)",
    "search_space_profile": "Optimize(space_profile=...)",
}


def resolve_spec_hyperparams(spec: dict, family_defaults: dict | None = None):
    """Work out where one ensemble base model's hyperparameters come from.

    One spec key says it::

        {"model_name": "lightgbm", "hyperparams": Optimize(n_trials=50)}
        {"model_name": "lightgbm", "hyperparams": {"n_estimators": 500}}
        {"model_name": "lightgbm"}                      # library defaults

    ``family_defaults`` fills only the ``Optimize`` fields the caller left
    alone, so an explicit ``Optimize(metric=...)`` is never overwritten.

    Args:
        spec:            One base-model spec dict.
        family_defaults: ``TEXT_SPEC_DEFAULTS`` or ``TABULAR_SPEC_DEFAULTS``,
                         used for fields neither spelling supplies.

    Returns:
        ``(kind, fixed, search)`` as :func:`resolve_hyperparams` returns it.
    """
    stale = sorted(k for k in _REMOVED_SPEC_KEYS if k in spec)
    if stale:
        raise ValueError(
            f"Base model spec for {spec.get('model_name', '?')!r} uses "
            f"{stale}, which no longer configure a search. Put them on "
            "Optimize instead: "
            + "; ".join(f"{k} -> {_REMOVED_SPEC_KEYS[k]}" for k in stale)
            + ". For example: "
            '{"model_name": "lightgbm", "hyperparams": Optimize(n_trials=50)}.'
        )

    kind, fixed, search = resolve_hyperparams(spec.get("hyperparams"))

    if kind == "search" and family_defaults:
        search = search.with_defaults(
            metric=family_defaults.get("optimize_metric"),
            n_trials=family_defaults.get("n_trials"),
            space_profile=family_defaults.get("search_space_profile"),
        )
    return kind, fixed, search


def validate_base_model_specs(
    specs: "list[dict] | None",
    allowed_keys: "set[str] | frozenset[str]",
    *,
    setting: str,
) -> None:
    """Reject malformed or silently ignored ensemble base-model settings.

    Base-model specs are dictionaries by design, so a misspelling otherwise
    survives construction and is ignored by every ``spec.get(...)`` call. This
    validator belongs beside :func:`resolve_spec_hyperparams`: both voting and
    stacking call it before any model or search is started.
    """
    for index, spec in enumerate(specs or []):
        if not isinstance(spec, dict):
            raise TypeError(
                f"{setting}[{index}] must be a dict, got "
                f"{type(spec).__name__}."
            )
        if not spec.get("model_name"):
            raise ValueError(f"{setting}[{index}] has no model_name.")
        # Retired search keys get their precise Optimize(...) replacement.
        resolve_spec_hyperparams(spec)
        unknown = sorted(set(spec) - set(allowed_keys))
        if unknown:
            raise ValueError(
                f"{setting}[{index}] ({spec['model_name']!r}) has unknown "
                f"setting(s) {unknown}. Accepted keys: {sorted(allowed_keys)}. "
                "Unknown keys are rejected because they would otherwise have "
                "no effect."
            )


def _log_fixed_or_default(logger_, model_name: str, fixed: dict | None) -> None:
    """Record where the hyperparameters came from on the no-search path.

    Two outcomes that look identical in a result file, so each gets its own
    line: the caller supplied values, or nobody did and the model library's own
    defaults apply.
    """
    if fixed:
        logger_.info(
            "Hyperparameters FROM CALLER — %d key(s): %s.",
            len(fixed), ", ".join(sorted(fixed)),
        )
    else:
        logger_.warning(
            "No hyperparameters given and no search requested — training %s "
            'with its library defaults. Pass hyperparams={...} to set them, or '
            "hyperparams=Optimize() to search.",
            model_name,
        )

def _log_hp_source(
    tag: str, model_name: str, hp: dict | None, logger_=None
) -> None:
    """Say where one ensemble base model's hyperparameters came from.

    Called on the no-search path. Three outcomes are possible and they are easy
    to confuse, so each gets its own line in the run log: the caller supplied
    them, or nothing did and the underlying library's own defaults apply.
    """
    logger_ = logger_ or logging.getLogger(__name__)
    if hp:
        logger_.info(
            "Base model %s (%s): hyperparameters FROM SPEC — %d key(s): %s.",
            tag, model_name, len(hp), ", ".join(sorted(hp)),
        )
    else:
        logger_.warning(
            "Base model %s (%s): NO hyperparameters given and no search "
            "requested — falling back to %s's own library defaults. Set "
            '"hyperparams" in its spec: a dict of values, or '
            "Optimize(...) to search.",
            tag, model_name, model_name,
        )


def warn_unused_settings(
    search: "Optimize",
    honoured: "set[str] | frozenset[str]",
    *,
    pipeline: str,
    note: str = "",
    log: "logging.Logger | None" = None,
) -> list[str]:
    """Warn about ``Optimize`` settings this pipeline will not act on.

    One ``Optimize`` object is accepted by every task, but not every task can
    honour every field: feature fusion, for instance, hands the search to
    AutoGluon AutoMM, which runs its own scheduler and selects on the
    classifier's ``eval_metric=`` — so ``backend=``, ``cv=``, ``pruning=`` and
    the rest never reach anything. Silently dropping a setting the caller
    deliberately passed is the hardest kind of problem to notice, so each one is
    named.

    Only fields that differ from their default are reported: a default the caller
    never touched is not a request, and warning about it would bury the real ones.

    Args:
        search:   The caller's settings.
        honoured: Field names this pipeline actually reads.
        pipeline: Named in the message, e.g. ``"feature_fusion"``.
        note:     One sentence on *why*, appended to the message.
        log:      Logger to warn on. Defaults to this module's.

    Returns:
        The field names warned about, in declaration order — so a caller (or a
        test) can see what was dropped without parsing the log line.
    """
    log = log or logger
    dropped = []
    for spec in fields(search):
        if spec.name in honoured:
            continue
        value = getattr(search, spec.name)
        if spec.default is not MISSING:
            if value == spec.default:
                continue
        elif spec.default_factory is not MISSING:     # type: ignore[misc]
            if value == spec.default_factory():       # type: ignore[misc]
                continue
        dropped.append(spec.name)

    if dropped:
        log.warning(
            "Optimize(%s) %s ignored by the %s pipeline.%s",
            ", ".join(f"{name}={getattr(search, name)!r}" for name in dropped),
            "is" if len(dropped) == 1 else "are",
            pipeline,
            f" {note}" if note else "",
        )
    return dropped
