"""The hyperparameters of a prompt, and the grid a selection ranges over.

``hyperparams=`` means the same three things here as for every other task:

* ``None`` / ``"default"`` — :data:`LLM_HYPERPARAMS_DEFAULT`;
* a dict — exactly these values, unnamed keys taking the defaults;
* ``Optimize(...)`` — a selection over a grid of prompt configurations, scored
  on the dev slice.

There is nothing to sample: the grid is enumerated in full, so the only
``Optimize`` fields that apply are ``metric``, ``space`` (categorical choices
for the generation settings) and ``extra["modules"]`` (which module grid).
Every other field that differs from its default is refused, not dropped.
"""

from __future__ import annotations

import itertools
from typing import Any

from ..utils.optimize_config import Optimize, resolve_hyperparams
from .backends import LLM_REASONING_EFFORTS
from .prompts import LIVE, SELECTABLE

__all__ = [
    "LLM_HYPERPARAMS_DEFAULT", "LLM_HYPERPARAM_KEYS", "LLM_SELECTION_GRIDS",
    "resolve_llm_hyperparams", "describe_llm_cell",
]

#: What a prompt run is configured by. ``modules`` is the prompt; the rest is
#: generation. ``reasoning_effort`` reaches only models that accept it.
LLM_HYPERPARAMS_DEFAULT: dict[str, Any] = {
    "modules": tuple(LIVE),
    "reasoning_effort": "low",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 2048,
}
LLM_HYPERPARAM_KEYS = tuple(LLM_HYPERPARAMS_DEFAULT)
#: Named module grids. ``leave_one_out``: the full set plus one cell per module
#: dropped. ``all_subsets``: every combination of the live modules.
LLM_SELECTION_GRIDS = ("leave_one_out", "all_subsets")
#: ``Optimize`` fields a prompt selection reads. The rest are refused when set.
LLM_OPTIMIZE_HONOURED = frozenset({"metric", "space", "extra"})


def _modules_tuple(value) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [v for v in value.replace(",", " ").split() if v]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"modules must be a list of module names, got {value!r}")
    cleaned = []
    for item in value:
        name = str(item).strip()
        canonical = {m.lower(): m for m in SELECTABLE}.get(name.lower())
        if canonical is None:
            raise ValueError(
                f"unknown module {name!r}; the modules are {list(SELECTABLE)}"
            )
        if canonical not in cleaned:
            cleaned.append(canonical)
    if {"M1", "M1b"} <= set(cleaned):
        raise ValueError("M1 and M1b are two wordings of the same block; pass one, not both")
    return tuple(sorted(cleaned, key=lambda m: list(SELECTABLE).index(m)))


def validate_llm_fixed(values: dict | None) -> dict:
    """A complete, checked configuration from a (possibly partial) dict."""
    values = dict(values or {})
    unknown = sorted(set(values) - set(LLM_HYPERPARAM_KEYS))
    if unknown:
        raise ValueError(
            f"hyperparams for task='llm' has unknown key(s) {unknown}. The keys are "
            f"{list(LLM_HYPERPARAM_KEYS)}."
        )
    out = dict(LLM_HYPERPARAMS_DEFAULT)
    out.update(values)
    out["modules"] = _modules_tuple(out["modules"])
    effort = out["reasoning_effort"]
    if effort is not None and effort not in LLM_REASONING_EFFORTS:
        raise ValueError(
            f"reasoning_effort={effort!r} is not one of {LLM_REASONING_EFFORTS} (or None)."
        )
    temperature = out["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError(f"temperature={temperature!r} must be a number >= 0.")
    out["temperature"] = float(temperature)
    top_p = out["top_p"]
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError(f"top_p={top_p!r} must be in (0, 1].")
    out["top_p"] = float(top_p)
    tokens = out["max_new_tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError(f"max_new_tokens={tokens!r} must be a positive integer.")
    return out


def _module_grid(spec) -> list[tuple[str, ...]]:
    if spec is None or spec == "leave_one_out":
        full = tuple(LIVE)
        return [full] + [tuple(m for m in full if m != dropped) for dropped in full]
    if spec == "all_subsets":
        full = tuple(LIVE)
        cells = []
        for k in range(len(full), -1, -1):
            cells.extend(itertools.combinations(full, k))
        return cells
    if isinstance(spec, (list, tuple)) and all(isinstance(c, (list, tuple, str)) for c in spec):
        cells = [_modules_tuple(c) for c in spec]
        if not cells:
            raise ValueError("extra['modules'] lists no cells")
        return cells
    raise ValueError(
        f"extra['modules']={spec!r} is not a module grid. Use "
        f"{', '.join(repr(g) for g in LLM_SELECTION_GRIDS)} or a list of module "
        "lists, e.g. [['M1','M2','M3','M4'], ['M2','M3']]."
    )


def validate_llm_search(search: Optimize) -> list[dict]:
    """The grid an ``Optimize`` describes for a prompt, or a ``ValueError``."""
    reference = Optimize()
    offending = []
    for name in ("n_trials", "cv", "cv_folds", "resume", "pruning", "backend", "space_profile"):
        if getattr(search, name) != getattr(reference, name):
            offending.append(f"{name}={getattr(search, name)!r}")
    if offending:
        raise ValueError(
            f"Optimize({', '.join(offending)}) does not apply to task='llm': the "
            "selection enumerates its whole grid on the dev slice, so there is no "
            "trial budget, no cross-validation, no pruning and no search backend. "
            "Use Optimize(metric=..., space={...}, extra={'modules': ...})."
        )
    unknown_extra = sorted(set(search.extra) - {"modules"})
    if unknown_extra:
        raise ValueError(
            f"Optimize(extra={{...}}) for task='llm' accepts only 'modules'; got "
            f"{unknown_extra}."
        )
    choices: dict[str, list] = {}
    for key, spec in (search.space or {}).items():
        if key not in LLM_HYPERPARAM_KEYS or key == "modules":
            raise ValueError(
                f"Optimize(space=...) key {key!r} is not a prompt setting. Use "
                f"{[k for k in LLM_HYPERPARAM_KEYS if k != 'modules']} as "
                "('categorical', [choices]); module grids go in extra={'modules': ...}."
            )
        if spec[0] != "categorical":
            raise ValueError(
                f"Optimize(space={{{key!r}: {spec!r}}}): a prompt selection is an "
                "exhaustive grid, so give the values to try as ('categorical', [...])."
            )
        choices[key] = list(spec[1])
    cells = []
    for modules in _module_grid(search.extra.get("modules")):
        keys = list(choices)
        for combo in itertools.product(*(choices[k] for k in keys)) if keys else [()]:
            cell = {"modules": modules, **dict(zip(keys, combo))}
            cells.append(validate_llm_fixed(cell))
    return cells


def resolve_llm_hyperparams(value) -> tuple[str, dict, "Optimize | None", "list[dict] | None"]:
    """``(kind, fixed, search, grid)`` for a prompt run.

    ``kind`` is ``"default"``, ``"fixed"`` or ``"search"``. ``fixed`` is the
    complete configuration for the first two; for a search it is the first
    cell of ``grid`` until the selection picks one.
    """
    kind, fixed, search = resolve_hyperparams(value)
    if kind == "search":
        grid = validate_llm_search(search)
        return kind, grid[0], search, grid
    return kind, validate_llm_fixed(fixed), None, None


def describe_llm_cell(cell: dict) -> str:
    """One short string naming a configuration, for logs and the trials table."""
    modules = "+".join(cell["modules"]) or "base"
    return (f"modules={modules} effort={cell['reasoning_effort']} "
            f"temperature={cell['temperature']} top_p={cell['top_p']} "
            f"max_new_tokens={cell['max_new_tokens']}")
