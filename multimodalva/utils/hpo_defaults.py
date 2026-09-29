"""Pieces the text and tabular HPO defaults must agree on.

Two things live here because both pipelines need them and they must not drift
apart:

* ``get_class_tier`` — where the cause-count tier boundaries fall. What a tier
  *does* to a search space differs by model family and is decided where the tier
  is used (``text/search_spaces.py``, ``tabular/search_spaces.py``).
* ``merge_search_space`` — how a caller-supplied ``search_space`` combines with
  the adaptive default, and the log line that records what it replaced.
* ``TEXT_SPEC_DEFAULTS`` / ``TABULAR_SPEC_DEFAULTS`` — what an ensemble
  base-model spec falls back to when it stays silent. These must equal the
  single-model defaults, or the same model would be trained differently
  depending on which pipeline reached it. They sit side by side here so a
  divergence is visible in one diff; ``tests/test_spec_defaults.py`` enforces it.
"""

from __future__ import annotations

__all__ = [
    "get_class_tier",
    "merge_search_space",
    "TEXT_SPEC_DEFAULTS",
    "TABULAR_SPEC_DEFAULTS",
]


# ---------------------------------------------------------------------------
# Ensemble base-model spec defaults
#
# A base model inside voting or stacking is the same model a single-model run
# trains, but those pipelines drive the low-level functions themselves: they
# need one split, then k folds, then a full-data refit, which
# ``TextClassifier.run`` / ``TabularClassifier.run`` cannot express because each
# owns its own split. So every knob has to be re-exported through the spec dict,
# and these are the values used when a spec does not mention one.
#
# Keep each value equal to the corresponding classifier ``run()`` default.
# They are not read from the signatures at import time on purpose: that would
# make importing the ensemble modules pull in torch, which the package avoids.
# ---------------------------------------------------------------------------

TEXT_SPEC_DEFAULTS: dict = {
    # Training knobs — they apply whether or not a search runs, so they must
    # match TextClassifier.run.
    "use_lora":        False,
    "use_focal":       False,
    "use_fast":        True,
    # Only the trial budget differs by family; everything else about a search
    # is an Optimize default. One text trial costs far more than a tabular one.
    "n_trials":        30,
    "optimize_metric": "f1_macro",
}

TABULAR_SPEC_DEFAULTS: dict = {
    "n_trials":             50,
    "optimize_metric":      "f1_macro",
    "search_space_profile": "auto",
}


def get_class_tier(n_classes: int) -> str:
    """Return the tier for a number of target classes.

    Calibrated for verbal-autopsy cause lists of roughly 5 to 100 causes:

        few       14 or fewer
        moderate  15 to 39
        many      40 or more

    Args:
        n_classes: Number of distinct causes in the training labels.

    Returns:
        ``"few"``, ``"moderate"`` or ``"many"``.
    """
    if n_classes <= 14:
        return "few"
    if n_classes <= 39:
        return "moderate"
    return "many"


def merge_search_space(active: dict, override: dict | None, logger_) -> dict:
    """Merge a caller's ``search_space`` over the adaptive one, in place.

    The caller wins per key: keys present in ``override`` replace the adaptive
    value **exactly as passed** — nothing narrows, rescales or re-profiles a range
    the caller wrote — and keys absent keep their adaptive value and are still
    searched. Both halves are logged, because the run record has to make it
    obvious that the space actually searched is the caller's where they spoke and
    the adaptive one where they did not.

    Args:
        active:   Adaptive search space, already including any LoRA/focal keys.
                  Modified in place.
        override: The caller's ``search_space`` argument, or ``None``.
        logger_:  Logger for the record.

    Returns:
        ``active``.
    """
    if not override:
        return active
    replaced = sorted(k for k in override if k in active and override[k] != active[k])
    added = sorted(k for k in override if k not in active)
    active.update(override)
    from_adaptive = sorted(k for k in active if k not in override)

    logger_.info(
        "search_space= gave %d key(s), used exactly as passed: %s.",
        len(override), ", ".join(sorted(override)),
    )
    if added:
        logger_.info("search_space= added %d key(s) the adaptive space does not "
                     "have: %s.", len(added), ", ".join(added))
    if from_adaptive:
        logger_.info(
            "The other %d key(s) come from the adaptive space for this data and "
            "are searched too: %s. Name them in search_space= to set them "
            "yourself.",
            len(from_adaptive), ", ".join(from_adaptive),
        )
    else:
        logger_.info("search_space= covers every key, so the adaptive space "
                     "contributes nothing to this run.")
    return active
