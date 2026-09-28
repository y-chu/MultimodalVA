"""``predict_ensemble_from_pretrained()``: score new data with a trained ensemble.

Voting and stacking runs hold several base models plus the rule that combines
them, so they get their own entry point rather than a mode of
``predict_from_pretrained()``: the arguments differ (a combiner to pick, one
text column shared by every text base model) and each base model is loaded
through the ordinary single-model machinery.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.numpy_compat import load_joblib_compat
from ..utils.predictions import assemble_predictions, resolve_test_ids, save_predictions
from ..utils.types import PredictionResult
from .api import (
    _find_artifact_dir,
    descend_to_pretrained_run,
    _load_pretrained_input,
    _read_training_metadata,
    _resolve_pretrained_source,
    build_pretrained_backend,
    resolve_pretrained_task,
)
from .backends import PRETRAINED_MISSING_FEATURE_METHODS
from .checks import PretrainedChecks, show_label_disclaimer

logger = logging.getLogger(__name__)

# How a stacking run's stage 2 combines its base models, and where each one's
# fitted combiner lives inside the run directory. ``simple_average`` fits
# nothing — equal weights — so it is recognised by its output directory.
# A meta-learner named in ``combiner=`` writes to ``meta_learner_<name>/``
# instead, and is resolved by _combiner_artifact().
PRETRAINED_COMBINER_SOURCES: dict[str, str] = {
    "meta_learner": "meta_learner/meta_learner.joblib",
    "simple_average": "simple_average",
    "class_aware_voting": "class_voter/class_weights.npy",
    "ensemble_selection": "ensemble_selection/ensemble_weights.npy",
}


def predict_ensemble_from_pretrained(
    source: str | Path,
    data: pd.DataFrame | str | Path,
    *,
    text_col: str | None = None,
    label_col: str | None = None,
    id_col: str | None = None,
    missing_feature_method: str = "error",
    task: str | None = None,
    force: bool = False,
    combiner: str | None = None,
    batch_size: int = 32,
    top_k: int = 3,
    save_dir: str | Path | None = None,
    backend_kwargs: dict | None = None,
) -> PredictionResult:
    """Predict with a trained soft-voting or stacking run.

    Every base model is loaded from the run directory and scored on ``data``,
    then combined exactly as the run combined them: the saved weights for soft
    voting, and for stacking the fitted meta-learner, class-aware voter or
    ensemble-selection weights.

    Args:
        source:         The ensemble run's ``output_dir`` (local). Stacking runs
                        record absolute paths to their base models; a run that
                        has moved is re-anchored to this directory.
        data:           New records: a DataFrame, or a CSV / Parquet path.
        text_col:       Text column, if the ensemble has text base models.
        label_col:      Ground-truth column, when the data is labelled.
        id_col:         Column copied onto every table as a leading ``id``.
        missing_feature_method: ``"error"`` or ``"fill_na"``, as in
                        :func:`~multimodalva.inference.api.predict_from_pretrained`.
        task:           ``"voting"`` or ``"stacking"``, for a run too old to
                        record its pipeline. Detection still runs and a
                        contradiction with the run's own metadata raises.
        force:          Proceed with ``task`` even against recorded metadata.
        combiner:       Stacking only — which of the run's fitted stage-2
                        combiners to apply: ``"meta_learner"``,
                        ``"simple_average"``, ``"class_aware_voting"``,
                        ``"ensemble_selection"``, or the name of a meta-learner
                        the run fitted under ``meta_learner_<name>/``. Only a
                        combiner the run actually fitted can be used. ``None``
                        (default) reuses the one the run delivered:
                        ``combiner_chosen`` when ``combiner="best"`` picked one,
                        otherwise the first combiner the run was asked for.
        batch_size:     Inference batch size for text base models.
        top_k:          Classes listed in ``topk``.
        save_dir:       Where to write the three tables, if anywhere.
        backend_kwargs: Passed to each base model's loader.

    Returns:
        ``PredictionResult`` whose ``checks`` lists every base model and the
        combiner used.
    """
    if missing_feature_method not in PRETRAINED_MISSING_FEATURE_METHODS:
        raise ValueError(
            f"missing_feature_method must be one of "
            f"{PRETRAINED_MISSING_FEATURE_METHODS}; got {missing_feature_method!r}."
        )
    root = descend_to_pretrained_run(_resolve_pretrained_source(source, {}))
    meta = _read_training_metadata(root, root)
    task, task_source, conflict = resolve_pretrained_task(
        meta, root, task, force=force)
    if task not in ("voting", "stacking"):
        raise ValueError(
            f"{source} is a {task!r} run, not an ensemble. Use "
            "predict_from_pretrained() for a single model."
        )
    checks = PretrainedChecks(
        source=str(source), artifact_dir=str(root), task=task,
        task_source=task_source,
        multimodalva_version=meta.get("multimodalva_version"),
    )
    if conflict:
        checks.warn(conflict)
    if checks.multimodalva_version is None:
        checks.warn("The artifact does not record which MultimodalVA version "
                    "trained it (a foreign or older run); continuing.")

    base_dirs = (_voting_base_dirs(root, meta) if task == "voting"
                 else _stacking_base_dirs(root))
    df = _load_pretrained_input(data)

    prob_list, id2label = [], None
    for base_dir, base_task in base_dirs:
        base_checks = PretrainedChecks(
            source=str(base_dir), artifact_dir=str(base_dir), task=base_task,
            task_source="ensemble_metadata",
            multimodalva_version=checks.multimodalva_version,
        )
        backend = build_pretrained_backend(
            _find_artifact_dir(base_dir), base_task, base_checks, text_col=text_col,
            missing_feature_method=missing_feature_method, batch_size=batch_size,
            backend_kwargs=backend_kwargs,
        )
        if id2label is None:
            id2label = backend.id2label
        elif backend.id2label != id2label:
            raise ValueError(
                f"Base model {base_dir} has a different label map from the "
                "first one. The base models of an ensemble must share it."
            )
        prob_list.append(backend.predict_proba(backend.prepare_inputs(df)))
        checks.warnings.extend(base_checks.warnings)
        checks.base_models.append({"dir": str(base_dir), "task": base_task,
                                   "id2label_source": base_checks.id2label_source})
    checks.id2label_source = checks.base_models[0]["id2label_source"]

    if task == "voting":
        from ..ensemble.voting import soft_vote  # noqa: PLC0415

        checks.combiner = "soft_voting"
        proba = soft_vote(prob_list, weights=meta.get("weights"))
    else:
        checks.combiner = _resolve_combiner(root, meta, combiner)
        proba = _combine_stacked_probs(root, checks.combiner, prob_list)
    logger.info("Combined %d base models with %s.", len(prob_list), checks.combiner)

    true_labels = None
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"label_col {label_col!r} not found in the input.")
        true_labels = df[label_col].astype(str).where(df[label_col].notna()).tolist()
        checks.labelled = True
    checks.n_rows = len(df)

    result = assemble_predictions(proba, id2label, true_labels=true_labels,
                                  ids=resolve_test_ids(df, id_col), top_k=top_k)
    if save_dir is not None:
        save_predictions(result, save_dir, "predictions")
    show_label_disclaimer()
    return result._replace(checks=checks)


def _voting_base_dirs(root: Path, meta: dict) -> list[tuple[Path, str]]:
    """``base_models/text_<i>`` then ``base_models/tabular_<i>`` — weight order."""
    dirs = []
    for kind, task in (("text", "text"), ("tabular", "tabular")):
        for i in range(int(meta.get(f"n_{kind}_models", 0))):
            d = root / "base_models" / f"{kind}_{i}"
            if not d.is_dir():
                raise FileNotFoundError(
                    f"Base model {d} is missing from this voting run."
                )
            dirs.append((d, task))
    if not dirs:
        raise ValueError(
            f"{root} keeps no base models under base_models/, so there is "
            "nothing to run new data through. A vote computed over predictions "
            "that already existed (vote_from_results() / soft_vote()) stores the "
            "vote only — which combiner was used is not the issue, the models "
            "are. Score each base model with predict_from_pretrained() and "
            "combine those results with vote_from_results(), or re-run the "
            "voting pipeline so it trains and keeps its base models."
        )
    return dirs


def _stacking_base_dirs(root: Path) -> list[tuple[Path, str]]:
    """Base models in the OOF column order the combiners were fitted on.

    ``model_sources`` stores the absolute path each base model had when stage 1
    ran, so a run opened on another machine — or from a synced folder — points
    at directories that no longer exist. Fall back to this run's own ``final/``.
    """
    oof_meta_path = root / "oof" / "oof_metadata.json"
    if not oof_meta_path.is_file():
        raise FileNotFoundError(
            f"No oof/oof_metadata.json in {root}; it records which base models "
            "the combiner was fitted on, in which order."
        )
    sources = json.loads(oof_meta_path.read_text()).get("model_sources") or []
    if not sources:
        raise ValueError(f"{oof_meta_path} records no model_sources.")

    dirs = []
    for src in sources:
        stored = Path(src["final_dir"])
        local = root / "final" / f"{src['type']}_{src['local_index']}"
        if stored.is_dir():
            d = stored
        elif local.is_dir():
            logger.warning("Recorded final_dir %s is gone; using %s.", stored, local)
            d = local
        else:
            raise FileNotFoundError(
                f"Base model {src['type']}_{src['local_index']} is at neither "
                f"{stored} nor {local}."
            )
        dirs.append((d, src["type"]))
    return dirs


def _combiner_artifact(root: Path, name: str) -> Path | None:
    """Where this run keeps ``name``'s fitted combiner, or None if it has none.

    ``simple_average`` fits nothing, so its own output directory stands in for
    an artifact. A name that is not a known combiner is taken to be a
    meta-learner listed in ``combiner=``, which writes to ``meta_learner_<name>/``.
    """
    rel = PRETRAINED_COMBINER_SOURCES.get(name)
    if rel is None:
        path = root / f"meta_learner_{name}" / "meta_learner.joblib"
        return path if path.is_file() else None
    path = root / rel
    if name == "simple_average":
        return path if path.is_dir() else None
    return path if path.is_file() else None


def _fitted_combiners(root: Path) -> list[str]:
    """Every combiner this run fitted, in the order they are preferred."""
    fitted = [m for m in PRETRAINED_COMBINER_SOURCES
              if _combiner_artifact(root, m) is not None]
    fitted += sorted(
        d.name[len("meta_learner_"):] for d in root.glob("meta_learner_*")
        if (d / "meta_learner.joblib").is_file()
    )
    return fitted


def _resolve_combiner(root: Path, meta: dict, requested: str | None) -> str:
    """Pick the combiner, defaulting to the one the run reported as its result."""
    fitted = _fitted_combiners(root)
    if not fitted:
        raise FileNotFoundError(
            f"{root} has no fitted stage-2 combiner (looked for "
            f"{', '.join(PRETRAINED_COMBINER_SOURCES.values())})."
        )
    if requested is None:
        # combiner="best" records its winner in combiner_chosen; that is the
        # model the run delivered, so it is what a prediction should reuse.
        recorded = meta.get("combiner_chosen") or meta.get("combiner")
        if isinstance(recorded, list):
            recorded = recorded[0] if recorded else None
        requested = recorded if recorded in fitted else fitted[0]
    if requested not in fitted:
        raise ValueError(
            f"combiner must be one this run fitted — {fitted} — or one of "
            f"{sorted(PRETRAINED_COMBINER_SOURCES)} if the run fitted it; "
            f"got {requested!r}."
        )
    return requested


def _combine_stacked_probs(root: Path, combiner: str,
                         prob_list: list) -> np.ndarray:
    """Apply the run's fitted stage-2 combiner to the base probabilities."""
    from ..ensemble.voting import soft_vote  # noqa: PLC0415

    if combiner == "simple_average":
        return soft_vote(prob_list)
    path = _combiner_artifact(root, combiner)
    if combiner == "meta_learner" or combiner not in PRETRAINED_COMBINER_SOURCES:
        meta_learner = load_joblib_compat(path)
        # The meta-learner reads the base models side by side, in the OOF
        # column layout it was fitted on.
        raw = np.asarray(meta_learner.predict_proba(np.hstack(prob_list)), dtype=float)
        n_classes = prob_list[0].shape[1]
        if raw.shape[1] == n_classes:
            return raw
        # A class absent from the OOF labels is missing from the meta-learner's
        # own classes_; put each column back where its class id belongs.
        proba = np.zeros((raw.shape[0], n_classes))
        proba[:, np.asarray(meta_learner.classes_, dtype=int)] = raw
        return proba
    if combiner == "class_aware_voting":
        from ..ensemble.stacking import _apply_class_voter  # noqa: PLC0415

        return _apply_class_voter(prob_list, np.load(path))

    return soft_vote(prob_list, weights=np.load(path).tolist())
