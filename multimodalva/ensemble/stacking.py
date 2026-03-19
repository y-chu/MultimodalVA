"""
Ensemble strategy 4 — Decision-level fusion via stacking (super learner).

Two-stage pipeline:

    Stage 1 — train_base_models():
        Split → k-fold OOF loop over text + tabular base models →
        retrain final base models on full training set.
        All artifacts (OOF probs, model weights) saved to disk; fully resumable.

    Stage 2 — train_meta_learner_stage():
        Load OOF meta-features → train one or more meta-learner candidates →
        select best by metric via stratified k-fold CV → save fitted meta-learner.

    Stage 3 — predict_test():
        Reload final base models → predict on hold-out test set →
        stack test probabilities → meta-learner.predict_proba() →
        assemble PredictionResult.

    Convenience: run() calls all three stages in sequence.

Output directory layout::

    output_dir/
    ├── data/
    │   ├── train_df.csv          — training split
    │   ├── test_df.csv           — held-out test split
    │   ├── X_test.npy            — preprocessed tabular test features
    │   └── y_test.npy            — integer test labels
    ├── oof/
    │   ├── fold_0/
    │   │   ├── text_0/           — per-fold model artifacts (if save_fold_models=True)
    │   │   │   └── oof_probs.npy — OOF probability matrix for fold 0, text model 0
    │   │   ├── text_1/
    │   │   └── tabular_0/
    │   ├── fold_1/ …
    │   ├── oof_meta_X.npy        — assembled OOF meta-feature matrix (n_train × n_models*n_classes)
    │   ├── oof_y.npy             — integer labels aligned with OOF rows
    │   ├── meta_feature_names.json
    │   └── oof_metadata.json
    ├── hpo/
    │   ├── text_0/               — Optuna HPO artifacts (if use_optimize=True)
    │   └── tabular_0/
    ├── final/
    │   ├── text_0/               — base model retrained on full training set
    │   ├── text_1/
    │   └── tabular_0/
    ├── meta_learner/
    │   ├── meta_learner.joblib   — fitted best meta-learner
    │   ├── meta_scores.json      — CV scores for all candidate meta-learners
    │   └── meta_learner_metadata.json
    ├── predictions/
    ├── label2id.json
    ├── id2label.json
    └── training_metadata.json

HPO per base model:
    Add ``"use_optimize": True`` to any model spec.  HPO runs ONCE on the full
    training set (before the OOF loop), and the best HP are reused for every
    fold and for the final full-data model.  This is the standard approach —
    per-fold HPO is too expensive and risks fold leakage.

Resume:
    Interrupted runs auto-resume by default (``resume=True``).
    Fold completion is detected by the presence of ``oof_probs.npy``; final
    model completion by ``training_metadata.json``.  Re-run the same stage
    call to continue from where training stopped.

Multi-GPU / MPS:
    Text models: HuggingFace Trainer handles device placement automatically
    (single GPU, multi-GPU DDP, Apple MPS).  No extra configuration needed.
    Tabular models: ``n_jobs=-1`` uses all CPU cores; GPU-capable models
    (lightgbm, catboost, xgboost) activate CUDA when ``use_gpu=True``.

InSilicoVA base model:
    Any tabular model spec may use ``model_name="insilicova"`` to include
    PyInSilicoVA as a base model.  Key differences from sklearn tabular models:

    * No training phase — InSilicoVA applies a fixed Bayesian symptom-cause
      probability database (WHO 2016 or PHMRC), so ``use_optimize`` is ignored.
    * Raw DataFrame required — ``train_df`` must be supplied to
      :func:`generate_oof_predictions` (and is passed automatically by
      :class:`StackingClassifier`).  InSilicoVA needs the original VA indicator
      columns, not the preprocessed numpy arrays.
    * Cause mapping — InSilicoVA's output uses its own cause-name strings
      (e.g. ``"HIV/AIDS related death"``).  Supply ``cause_map`` in the spec
      to map your label names to InSilicoVA column names::

          {"HIV/AIDS": "HIV/AIDS related death", "Malaria": "Malaria", ...}

      Omit ``cause_map`` for automatic case-insensitive matching (a warning is
      logged for any unmatched causes, which receive probability 0).
    * Columns — specify ``va_cols`` in the spec to restrict which indicator
      columns are passed to InSilicoVA; defaults to ``feature_cols``.

    Spec example::

        {"model_name": "insilicova",
         "data_type":  "WHO2016",          # "WHO2016" (default) or "PHMRC"
         "va_cols":    [...],               # optional; defaults to feature_cols
         "cause_map":  {"HIV/AIDS": "HIV/AIDS related death", ...},  # optional
         "hyperparams": {"n_sim": 4000, "burnin": 2000, "thin": 10}}

    Requires:  ``pip install pyinsilicova``

Public API:
    generate_oof_predictions(...)
        Standalone k-fold OOF function for advanced users.
    StackingClassifier
        End-to-end wrapper with explicit stage methods and run().
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..utils.types import PredictionResult

logger = logging.getLogger(__name__)

BaseModelSpec = dict

DEFAULT_META_LEARNER: dict = {
    "model_name":  "logistic_regression",
    "hyperparams": {"max_iter": 1000, "C": 1.0},
}

# Alias used to identify InSilicoVA specs throughout this module
_INSILICOVA = "insilicova"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _SubsetDataset:
    """Lightweight wrapper exposing a contiguous-index subset of a Dataset.

    Compatible with any ``torch.utils.data.Dataset`` via ``__len__`` /
    ``__getitem__``.  Used to feed fold-specific training / validation slices
    to ``text.train()`` and ``text.predict()`` without re-tokenising.
    """

    def __init__(self, full_ds, indices):
        self._full    = full_ds
        self._indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        return self._full[int(self._indices[i])]


def _extract_probs(result: PredictionResult, sorted_ids: list) -> np.ndarray:
    """Return (n_samples, n_classes) float array from a PredictionResult.full."""
    prob_cols = [f"prob_{cid}" for cid in sorted_ids]
    return result.full[prob_cols].to_numpy(dtype=float)


def _assemble_prediction_result(
    combined_proba: np.ndarray,
    true_labels: list,
    id2label: dict,
    top_k: int,
) -> PredictionResult:
    """Build PredictionResult from a (n_samples, n_classes) probability matrix."""
    n          = len(true_labels)
    sorted_ids = sorted(id2label.keys())
    n_classes  = len(sorted_ids)
    top_k      = min(top_k, n_classes)

    top1_pos = np.argmax(combined_proba, axis=1)
    top1_ids = [sorted_ids[p] for p in top1_pos]
    top1_df  = pd.DataFrame({
        "true_label":      true_labels,
        "predicted_label": [id2label[ci] for ci in top1_ids],
        "predicted_prob":  combined_proba[np.arange(n), top1_pos],
    })

    full_data = {"true_label": true_labels}
    for j, cid in enumerate(sorted_ids):
        full_data[f"prob_{cid}"] = combined_proba[:, j]
    full_df = pd.DataFrame(full_data)

    top_indices = np.argsort(combined_proba, axis=1)[:, ::-1][:, :top_k]
    topk_data   = {"true_label": true_labels}
    for j in range(top_k):
        col_pos   = top_indices[:, j]
        class_ids = [sorted_ids[p] for p in col_pos]
        topk_data[f"top{j+1}_label"] = [id2label[ci] for ci in class_ids]
        topk_data[f"top{j+1}_prob"]  = combined_proba[np.arange(n), col_pos]
    topk_df = pd.DataFrame(topk_data)

    return PredictionResult(top1=top1_df, full=full_df, topk=topk_df, id2label=id2label)


def _build_meta_feature_names(
    text_specs: list,
    tabular_specs: list,
    sorted_ids: list,
) -> list[str]:
    """Build a list of column names for the meta-feature matrix.

    Format: ``text_{i}_{model_name}_prob_{class_id}`` and
            ``tab_{i}_{model_name}_prob_{class_id}``.
    """
    names = []
    for i, spec in enumerate(text_specs):
        mn = spec["model_name"].replace("/", "_").replace("-", "_")
        for cid in sorted_ids:
            names.append(f"text_{i}_{mn}_prob_{cid}")
    for i, spec in enumerate(tabular_specs):
        mn = spec["model_name"].replace("/", "_").replace("-", "_")
        for cid in sorted_ids:
            names.append(f"tab_{i}_{mn}_prob_{cid}")
    return names


def _resolve_meta_learner(
    spec: dict,
    n_jobs: int = -1,
    random_state: int = 42,
    use_gpu: bool = False,
):
    """Instantiate a meta-learner from a spec dict.

    Special alias ``"logistic_regression"`` maps to
    ``sklearn.linear_model.LogisticRegression``.
    All tabular model aliases from ``tabular.train.SUPPORTED_MODELS`` are also
    accepted (lightgbm, catboost, xgboost, random_forest, mlp, …).

    Args:
        spec:         Dict with ``model_name`` and optional ``hyperparams``.
        n_jobs:       Parallelism for CPU-bound models.
        random_state: Seed for reproducibility.
        use_gpu:      GPU flag (catboost / lightgbm / xgboost).

    Returns:
        Instantiated sklearn-compatible classifier (unfitted).
    """
    model_name = spec["model_name"]
    hp         = dict(spec.get("hyperparams") or {})

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        hp.setdefault("max_iter",      1000)
        hp.setdefault("C",             1.0)
        hp.setdefault("n_jobs",        n_jobs)
        hp.setdefault("random_state",  random_state)
        return LogisticRegression(**hp)

    # All other models: delegate to tabular pipeline's SUPPORTED_MODELS registry
    from ..tabular.train import SUPPORTED_MODELS, _NJOBS_PARAM, _CUDA_PARAMS, _FORCED_PARAMS
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown meta-learner '{model_name}'.  "
            f"Use 'logistic_regression' or one of: {list(SUPPORTED_MODELS)}"
        )

    import importlib
    module_path, class_name = SUPPORTED_MODELS[model_name]
    cls = getattr(importlib.import_module(module_path), class_name)

    if model_name in _NJOBS_PARAM:
        hp.setdefault(_NJOBS_PARAM[model_name], n_jobs)
    if model_name in _FORCED_PARAMS:
        hp.update(_FORCED_PARAMS[model_name])
    if use_gpu and model_name in _CUDA_PARAMS:
        hp.update(_CUDA_PARAMS[model_name])

    # Silence verbose output in the meta-learner CV loop
    if model_name == "catboost":
        hp.setdefault("verbose", 0)
    elif model_name == "lightgbm":
        hp.setdefault("verbose", -1)
    elif model_name == "xgboost":
        hp.setdefault("verbosity", 0)

    if model_name not in {"catboost"}:
        hp.setdefault("random_state", random_state)

    return cls(**hp)


def _score_meta_candidate(
    meta_model,
    meta_X: np.ndarray,
    y_oof: np.ndarray,
    id2label: dict,
    metric: str,
    cv_folds: int,
    random_state: int,
) -> float:
    """Score a meta-learner via stratified k-fold CV on the OOF meta-features.

    Uses ``cross_val_predict`` to get second-level OOF predictions, then scores
    with the given metric.  All four metrics supported by ``score_predictions``
    are valid here.

    Args:
        meta_model:   Unfitted sklearn-compatible classifier.
        meta_X:       OOF meta-feature matrix (n_train × n_models*n_classes).
        y_oof:        Integer label array (n_train,).
        id2label:     Integer → label string mapping.
        metric:       One of ``accuracy``, ``f1_macro``, ``f1_weighted``,
                      ``csmf_accuracy``.
        cv_folds:     CV folds for meta-learner selection (default 3).
        random_state: Seed.

    Returns:
        Mean CV score (float).
    """
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from ..utils.metrics import score_predictions

    cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    oof_pred = cross_val_predict(meta_model, meta_X, y_oof, cv=cv, method="predict")

    sorted_ids = sorted(id2label.keys())
    true_str   = [id2label[int(y)] for y in y_oof]
    pred_str   = [id2label[sorted_ids[int(p)]] for p in oof_pred]

    top1_df = pd.DataFrame({"true_label": true_str, "predicted_label": pred_str})
    return score_predictions(top1_df, metric=metric)


# ---------------------------------------------------------------------------
# InSilicoVA integration helpers
# ---------------------------------------------------------------------------

def _insilicova_save_config(
    spec: dict,
    best_hp: dict | None,
    label2id: dict,
    id2label: dict,
    feature_cols: list[str] | None,
    output_dir: Path,
    label_col: str = "cause",
) -> None:
    """Persist InSilicoVA run configuration to *output_dir*.

    InSilicoVA uses ``data_type="customize"``: it learns symptom-cause
    probabilities from the training data at inference time.  The output
    ``indiv_prob`` columns are named directly by the user's cause labels
    from the training data — no cause mapping is needed.

    A dummy ``training_metadata.json`` is written so the standard
    resume-detection logic (which looks for that file) works unchanged.

    Args:
        spec:          Model spec dict (``model_name="insilicova"``).
        best_hp:       Hyperparams dict (``Nsim``, ``burnin``, ``thin``, …).
                       Falls back to ``spec["hyperparams"]`` when ``None``.
        label2id:      Shared label → integer map.
        id2label:      Shared integer → label map.
        feature_cols:  Fallback VA indicator columns when ``spec["va_cols"]``
                       is absent.
        output_dir:    Target directory (created if missing).
        label_col:     Column name in training DataFrame containing cause labels.
                       Passed to InSilicoVA as ``causes_train``. Default "cause".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hp = best_hp or spec.get("hyperparams") or {}

    config: dict = {
        # va_cols: raw VA indicator columns to pass to InSilicoVA (excludes label)
        "va_cols":     spec.get("va_cols", feature_cols),
        # label_col: cause column in training DataFrame (used as causes_train)
        "label_col":   spec.get("label_col", label_col),
        "hyperparams": hp,
        "id2label":    {str(k): v for k, v in id2label.items()},
        "label2id":    label2id,
    }
    with open(output_dir / "insilicova_config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    # Dummy training_metadata.json — enables resume detection
    with open(output_dir / "training_metadata.json", "w") as fh:
        json.dump({"model_name": _INSILICOVA, "hyperparams": hp}, fh, indent=2)

    logger.info("InSilicoVA config saved to %s", output_dir)


def _insilicova_predict_df(
    config_dir: Path,
    df: pd.DataFrame,
    sorted_ids: list,
    train_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Run InSilicoVA (customize mode) on *df* and return a probability matrix.

    Uses ``data_type="customize"``: passes *train_df* (with the cause column)
    as the training reference so InSilicoVA learns symptom-cause probabilities
    from the actual training data rather than the fixed WHO database.  The
    ``indiv_prob`` columns returned by InSilicoVA are then named by the user's
    own cause labels — no cause mapping is required.

    Equivalent R call::

        codeVA(data = df, data.type = "customize", model = "InSilicoVA",
               data.train = train_df, causes.train = label_col,
               Nchain = 3, Nsim = 10000, auto.length = TRUE)

    Args:
        config_dir:  Directory containing ``insilicova_config.json``.
        df:          VA DataFrame of cases to classify (symptom columns only,
                     no label).  Pass ``reset_index(drop=True)`` so row
                     positions align with InSilicoVA's output.
        sorted_ids:  Sorted integer class IDs (from ``sorted(id2label.keys())``).
        train_df:    Training DataFrame including the cause label column.
                     Required for ``data_type="customize"``.  For OOF, pass
                     the fold's training rows; for test prediction, pass the
                     full training set.

    Returns:
        np.ndarray of shape ``(len(df), len(sorted_ids))``.

    Raises:
        ImportError:       ``pyinsilicova`` is not installed.
        FileNotFoundError: ``insilicova_config.json`` missing in *config_dir*.
        ValueError:        *train_df* is None (required for customize mode).
    """
    try:
        from pyinsilicova.insilicova import InSilicoVA
    except ImportError as exc:
        raise ImportError(
            "pyinsilicova is required for 'insilicova' base models.  "
            "Install with:  pip install pyinsilicova"
        ) from exc

    with open(config_dir / "insilicova_config.json") as fh:
        config = json.load(fh)

    id2label  = {int(k): v for k, v in config["id2label"].items()}
    va_cols   = config.get("va_cols")
    label_col = config.get("label_col", "cause")
    hp        = {k: v for k, v in (config.get("hyperparams") or {}).items()}

    if train_df is None:
        raise ValueError(
            "train_df is required for InSilicoVA (data_type='customize').  "
            "Pass the fold training rows for OOF, or the full training set "
            "for test prediction."
        )

    # Extract VA indicator columns (exclude label from train_df)
    val_va   = df[va_cols].reset_index(drop=True)   if va_cols else df.reset_index(drop=True)
    train_va = train_df[va_cols + [label_col]].reset_index(drop=True) if va_cols else train_df.reset_index(drop=True)

    logger.info(
        "Running InSilicoVA (customize) on %d cases with %d training samples ...",
        len(val_va), len(train_va),
    )
    isva = InSilicoVA(
        val_va,
        data_type="customize",
        train_data=train_va,
        causes_train=label_col,
        **hp,
    )
    isva.run()
    indiv_probs: pd.DataFrame = isva.indiv_prob   # cols = user's cause labels

    # Direct column lookup — output column names match user's cause labels
    n         = len(df)
    n_classes = len(sorted_ids)
    proba     = np.zeros((n, n_classes), dtype=float)

    missing = []
    for j, cid in enumerate(sorted_ids):
        label = id2label[cid]
        if label in indiv_probs.columns:
            proba[:, j] = indiv_probs[label].to_numpy(dtype=float)
        else:
            missing.append(label)

    if missing:
        logger.warning(
            "InSilicoVA: %d cause(s) not found in indiv_prob output: %s.  "
            "These columns receive probability 0 and rows are renormalised.",
            len(missing), missing,
        )
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        proba   /= row_sums

    return proba


# ---------------------------------------------------------------------------
# Standalone OOF function
# ---------------------------------------------------------------------------

def generate_oof_predictions(
    text_model_specs: list[BaseModelSpec],
    tabular_model_specs: list[BaseModelSpec],
    train_text_dataset,
    X_train: np.ndarray,
    y_train: np.ndarray,
    label2id: dict,
    id2label: dict,
    n_folds: int = 5,
    random_state: int = 42,
    output_dir: str | Path = "runs/ensemble/stacking/oof",
    # text training
    val_size: float = 0.1,
    gradient_checkpointing: bool = False,
    early_stopping_patience: int | None = 3,
    batch_size: int = 32,
    # tabular training
    n_jobs: int = -1,
    use_gpu: bool | None = None,
    # resume / disk
    resume: bool = True,
    save_fold_models: bool = False,
    cleanup_fold_files: bool = True,
    # optional pre-computed HP (from HPO stage; one dict per model in spec order)
    text_best_hp: list[dict] | None = None,
    tabular_best_hp: list[dict] | None = None,
    # InSilicoVA — raw DataFrame needed for pyinsilicova (not numpy arrays)
    train_df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    label_col: str = "cause",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold probability predictions for all base models.

    For each fold and each base model, trains on the k-1 in-fold samples and
    predicts probabilities for the held-out fold.  OOF probabilities from all
    base models are concatenated into a meta-feature matrix covering the full
    training set.

    Meta-feature matrix column layout (one block of n_classes per model):
    ``[text_0_prob_0..C, text_1_prob_0..C, ..., tab_0_prob_0..C, ...]``

    Args:
        text_model_specs:     List of text base-model spec dicts.
        tabular_model_specs:  List of tabular base-model spec dicts.
        train_text_dataset:   Pre-tokenised ``ClassificationDataset`` for the
                              full training set.  Fold subsets are derived via
                              ``_SubsetDataset``.
        X_train:              Preprocessed tabular feature matrix (n_train×p).
        y_train:              Integer label array (n_train,).
        label2id:             Label → integer mapping.
        id2label:             Integer → label mapping.
        n_folds:              CV folds. Default 5.
        random_state:         Seed. Default 42.
        output_dir:           Root OOF directory.  Fold artifacts saved under
                              ``output_dir/fold_k/text_i/`` etc.
        val_size:             Internal val fraction for text fold training
                              (early stopping). Default 0.1.
        gradient_checkpointing: Enable gradient checkpointing for text folds.
        early_stopping_patience: Early stopping patience for text fold models.
        batch_size:           Text inference batch size.
        n_jobs:               CPU parallelism for tabular models.
        use_gpu:              GPU flag for tabular models. None = auto-detect.
        resume:               Skip completed fold/model combos. Default True.
        save_fold_models:     Keep model weights after extracting OOF probs.
                              ``False`` (default) saves disk space.
        text_best_hp:         Pre-computed HP dicts for text models (from HPO).
                              ``None`` → use ``spec["hyperparams"]`` as-is.
        tabular_best_hp:      Pre-computed HP dicts for tabular models.
        train_df:             Full training DataFrame (required when any spec
                              has ``model_name="insilicova"``).
        feature_cols:         VA indicator columns for InSilicoVA.
        label_col:            Cause label column name in *train_df*.
                              Passed to InSilicoVA as ``causes_train``.
                              Default "cause".

    Returns:
        Tuple ``(oof_meta_X, oof_y)`` where:

        ``oof_meta_X``:  np.ndarray shape (n_train, n_models × n_classes).
                         Row i holds the OOF probability predictions for
                         training sample i from every base model.
        ``oof_y``:       np.ndarray shape (n_train,) — original integer labels
                         in train-set order (identical to input ``y_train``).

    Note:
        When any ``tabular_model_specs`` entry has ``model_name="insilicova"``,
        both ``train_df`` and ``feature_cols`` must be supplied.  InSilicoVA
        operates on the raw VA DataFrame (not the preprocessed numpy arrays)
        and has no separate training phase — the fold val rows are passed
        directly to :func:`_insilicova_predict_df`.
    """
    from sklearn.model_selection import StratifiedKFold

    # Lazy pipeline imports (avoid circular)
    from ..text.train   import train   as text_train
    from ..text.predict import predict as text_predict
    from ..tabular.train   import train   as tabular_train
    from ..tabular.predict import predict as tabular_predict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_train     = len(y_train)
    sorted_ids  = sorted(id2label.keys())
    n_classes   = len(sorted_ids)
    n_text      = len(text_model_specs)
    n_tabular   = len(tabular_model_specs)
    n_total     = n_text + n_tabular

    oof_meta_X  = np.zeros((n_train, n_total * n_classes), dtype=float)
    oof_y       = y_train.copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    all_splits = list(skf.split(np.arange(n_train), y_train))

    for fold_idx, (train_idx, val_idx) in enumerate(all_splits):
        fold_dir = output_dir / f"fold_{fold_idx}"
        logger.info("=== Fold %d/%d  (train=%d, val=%d) ===",
                    fold_idx + 1, n_folds, len(train_idx), len(val_idx))

        # ---------- Text base models ----------------------------------------
        for i, spec in enumerate(text_model_specs):
            model_dir      = fold_dir / f"text_{i}"
            oof_probs_file = model_dir / "oof_probs.npy"

            if resume and oof_probs_file.exists():
                logger.info("  [text_%d fold_%d] Resuming — loading %s", i, fold_idx, oof_probs_file)
                fold_probs = np.load(oof_probs_file)
            else:
                model_dir.mkdir(parents=True, exist_ok=True)
                model_name = spec["model_name"]
                max_length = spec.get("max_length", 512)
                use_lora   = spec.get("use_lora", False)
                esp        = spec.get("early_stopping_patience", early_stopping_patience)
                gc         = spec.get("gradient_checkpointing", gradient_checkpointing)
                hp         = (text_best_hp[i] if text_best_hp else None) or spec.get("hyperparams") or {}

                logger.info("  [text_%d fold_%d] Training %s ...", i, fold_idx, model_name)
                fold_train_ds = _SubsetDataset(train_text_dataset, train_idx)
                fold_val_ds   = _SubsetDataset(train_text_dataset, val_idx)

                text_train(
                    train_dataset=fold_train_ds,
                    label2id=label2id, id2label=id2label,
                    model_name=model_name,
                    output_dir=model_dir / "weights",
                    hyperparams=hp,
                    val_size=val_size,
                    use_lora=use_lora,
                    gradient_checkpointing=gc,
                    early_stopping_patience=esp,
                    resume=False,          # never cross-contaminate fold checkpoints
                )

                result     = text_predict(
                    model_dir / "weights",
                    fold_val_ds,
                    batch_size=batch_size,
                    top_k=1,
                )
                fold_probs = _extract_probs(result, sorted_ids)  # (n_val, n_classes)
                np.save(oof_probs_file, fold_probs)
                logger.info("  [text_%d fold_%d] OOF probs saved (%d samples)", i, fold_idx, len(val_idx))

                if not save_fold_models:
                    shutil.rmtree(model_dir / "weights", ignore_errors=True)

            col_s = i * n_classes
            oof_meta_X[val_idx, col_s:col_s + n_classes] = fold_probs

        # ---------- Tabular base models -------------------------------------
        for i, spec in enumerate(tabular_model_specs):
            model_dir      = fold_dir / f"tabular_{i}"
            oof_probs_file = model_dir / "oof_probs.npy"

            if resume and oof_probs_file.exists():
                logger.info("  [tab_%d fold_%d] Resuming — loading %s", i, fold_idx, oof_probs_file)
                fold_probs = np.load(oof_probs_file)
            else:
                model_dir.mkdir(parents=True, exist_ok=True)
                model_name = spec["model_name"]
                logger.info("  [tab_%d fold_%d] Training %s ...", i, fold_idx, model_name)

                if model_name == _INSILICOVA:
                    # InSilicoVA (customize mode): learns symptom-cause probs
                    # from the fold training rows; predicts on fold val rows.
                    if train_df is None:
                        raise ValueError(
                            "train_df must be supplied to generate_oof_predictions() "
                            "when any tabular_model_spec has model_name='insilicova'."
                        )
                    hp = (tabular_best_hp[i] if tabular_best_hp else None) or spec.get("hyperparams") or {}
                    _insilicova_save_config(
                        spec, hp, label2id, id2label, feature_cols,
                        model_dir / "weights",
                        label_col=label_col,
                    )
                    fold_probs = _insilicova_predict_df(
                        model_dir / "weights",
                        train_df.iloc[val_idx].reset_index(drop=True),
                        sorted_ids,
                        train_df=train_df.iloc[train_idx].reset_index(drop=True),
                    )
                else:
                    hp = (tabular_best_hp[i] if tabular_best_hp else None) or spec.get("hyperparams")
                    tabular_train(
                        X_train=X_train[train_idx],
                        y_train=y_train[train_idx],
                        label2id=label2id, id2label=id2label,
                        model_name=model_name,
                        output_dir=model_dir / "weights",
                        hyperparams=hp,
                        random_state=random_state,
                        n_jobs=n_jobs, use_gpu=use_gpu,
                    )
                    result = tabular_predict(
                        model_dir / "weights",
                        X_test=X_train[val_idx],
                        y_test=y_train[val_idx],
                        top_k=1,
                    )
                    fold_probs = _extract_probs(result, sorted_ids)
                    if not save_fold_models:
                        shutil.rmtree(model_dir / "weights", ignore_errors=True)

                np.save(oof_probs_file, fold_probs)
                logger.info("  [tab_%d fold_%d] OOF probs saved (%d samples)", i, fold_idx, len(val_idx))

            col_s = (n_text + i) * n_classes
            oof_meta_X[val_idx, col_s:col_s + n_classes] = fold_probs

    # Persist assembled OOF matrix
    np.save(output_dir / "oof_meta_X.npy", oof_meta_X)
    np.save(output_dir / "oof_y.npy",      oof_y)

    meta_names = _build_meta_feature_names(text_model_specs, tabular_model_specs, sorted_ids)
    with open(output_dir / "meta_feature_names.json", "w") as fh:
        json.dump(meta_names, fh, indent=2)

    # Remove per-fold intermediate directories — only oof_meta_X.npy / oof_y.npy
    # are needed from this point on; fold dirs are resume checkpoints only.
    if cleanup_fold_files:
        for fold_dir in sorted(output_dir.glob("fold_*")):
            shutil.rmtree(fold_dir, ignore_errors=True)
        logger.info("Cleaned up fold directories from %s", output_dir)

    logger.info("OOF generation complete — meta_X shape: %s", oof_meta_X.shape)
    return oof_meta_X, oof_y


# ---------------------------------------------------------------------------
# StackingClassifier
# ---------------------------------------------------------------------------

class StackingClassifier:
    """End-to-end stacking (super learner) ensemble classifier.

    Implements a two-stage pipeline with explicit stage methods so that each
    stage can be run independently across separate Python sessions (resumable).

    Stage 1 — ``train_base_models(df, …)``
        Splits data, runs optional per-model HPO, executes the k-fold OOF loop,
        and retrains final base models on the full training set.

    Stage 2 — ``train_meta_learner_stage()``
        Loads OOF meta-features (from disk or in-memory), trains one or more
        meta-learner candidates, selects the best by CV metric, saves the winner.

    Stage 3 — ``predict_test()``
        Loads final base models, predicts on the hold-out test set, stacks
        probabilities, passes through the meta-learner, returns PredictionResult.

    ``run(df, …)`` calls all three stages in sequence.

    Attributes (populated after train_base_models())
    -------------------------------------------------
    train_df, test_df :   Split DataFrames.
    label2id, id2label :  Shared label maps.
    oof_meta_X :          OOF meta-feature matrix (n_train × n_models*n_classes).
    oof_y :               Integer label array for OOF rows.
    text_best_hp :        Best HP per text model (list of dicts).
    tabular_best_hp :     Best HP per tabular model (list of dicts).

    Attributes (populated after train_meta_learner_stage())
    --------------------------------------------------------
    meta_learner :        Fitted best meta-learner.
    meta_scores :         Dict {model_name: cv_score} for all candidates.

    Attributes (populated after predict_test() / run())
    ----------------------------------------------------
    predictions :         Final PredictionResult.
    """

    def __init__(
        self,
        text_models: list[BaseModelSpec],
        tabular_models: list[BaseModelSpec],
        output_dir: str | Path = "runs/ensemble/stacking",
        meta_learners: list[dict] | dict | None = None,
        meta_select_metric: str = "f1_macro",
        n_folds: int = 5,
        resume: bool = True,
        load_oof_from: str | Path | None = None,
    ):
        """Initialise StackingClassifier.

        Args:
            text_models:          List of text base-model spec dicts.
                                  Same format as SoftVotingClassifier.
                                  When ``load_oof_from`` is set, list **only
                                  the new models** to add — do not repeat
                                  models already in the source run.
            tabular_models:       List of tabular base-model spec dicts.
                                  Same rule as ``text_models``.
            output_dir:           Root output directory for this run.
            meta_learners:        One meta-learner spec dict, or a list of
                                  candidates (the best by CV score is chosen).
                                  Default: logistic regression (C=1.0).
            meta_select_metric:   Metric used to select among multiple
                                  meta-learner candidates.  Options:
                                  ``accuracy``, ``f1_macro`` (default),
                                  ``f1_weighted``, ``csmf_accuracy``.
            n_folds:              CV folds for the OOF loop. Default 5.
            resume:               Resume interrupted training. Default True.
            load_oof_from:        Path to an existing stacking ``output_dir``
                                  whose OOF predictions should be reused.
                                  When set:

                                  * The train/test split from that run is
                                    reused verbatim (row alignment is required).
                                  * OOF is only computed for the models in
                                    *this* ``text_models`` / ``tabular_models``
                                    (the newly added ones).
                                  * The combined OOF matrix
                                    ``[inherited | new]`` feeds Stage 2.
                                  * ``predict_test()`` draws final-model
                                    weights from the original run for inherited
                                    models and from ``output_dir/final/`` for
                                    new models.
                                  * Chaining is supported: the source run may
                                    itself have been created with
                                    ``load_oof_from``.

                                  Default ``None`` (standard full training).
        """
        if not text_models and not tabular_models:
            raise ValueError("At least one text or tabular model spec is required.")

        # Normalise meta_learners to a list
        if meta_learners is None:
            meta_learners = [dict(DEFAULT_META_LEARNER)]
        elif isinstance(meta_learners, dict):
            meta_learners = [meta_learners]

        self.text_models        = list(text_models)    if text_models    else []
        self.tabular_models     = list(tabular_models) if tabular_models else []
        self.output_dir         = Path(output_dir)
        self.meta_learner_specs = meta_learners
        self.meta_select_metric = meta_select_metric
        self.n_folds            = n_folds
        self.resume             = resume
        self.load_oof_from      = Path(load_oof_from) if load_oof_from else None

        # Populated after train_base_models()
        self.train_df:       pd.DataFrame | None = None
        self.test_df:        pd.DataFrame | None = None
        self.label2id:       dict | None          = None
        self.id2label:       dict | None          = None
        self.oof_meta_X:     np.ndarray | None    = None
        self.oof_y:          np.ndarray | None    = None
        self.text_best_hp:   list[dict]           = []
        self.tabular_best_hp: list[dict]          = []
        self._text_col:      str | None           = None
        self._feature_cols:  list[str] | None     = None
        self._label_col:     str | None           = None

        # Populated after train_meta_learner_stage()
        self.meta_learner: Any | None = None
        self.meta_scores:  dict       = {}

        # Populated after predict_test()
        self.predictions: PredictionResult | None = None

    # ------------------------------------------------------------------
    # Internal: load OOF state from disk (for cross-session Stage 2/3)
    # ------------------------------------------------------------------

    def _ensure_oof_loaded(self):
        """Load OOF artifacts from disk if not already in memory."""
        if self.oof_meta_X is not None:
            return

        oof_dir = self.output_dir / "oof"
        meta_X_file = oof_dir / "oof_meta_X.npy"
        if not meta_X_file.exists():
            raise RuntimeError(
                "OOF meta-features not found at %s.  "
                "Run train_base_models() first." % meta_X_file
            )

        self.oof_meta_X = np.load(meta_X_file)
        self.oof_y      = np.load(oof_dir / "oof_y.npy")

        lbl_path   = self.output_dir / "label2id.json"
        id2lbl_path = self.output_dir / "id2label.json"
        with open(lbl_path)   as f: self.label2id = json.load(f)
        with open(id2lbl_path) as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}

        meta_path = oof_dir / "oof_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._text_col    = meta.get("text_col")
            self._feature_cols = meta.get("feature_cols")
            self._label_col   = meta.get("label_col")
            self.text_best_hp  = meta.get("text_best_hp",    [{}] * len(self.text_models))
            self.tabular_best_hp = meta.get("tabular_best_hp", [{}] * len(self.tabular_models))

        logger.info("OOF state loaded from disk — meta_X shape: %s", self.oof_meta_X.shape)

    # ------------------------------------------------------------------
    # Stage 1: train base models + generate OOF predictions
    # ------------------------------------------------------------------

    def train_base_models(
        self,
        df: pd.DataFrame,
        label_col: str,
        text_col: str | None = None,
        feature_cols: list[str] | None = None,
        # --- split ---
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        # --- text training ---
        val_size: float = 0.1,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 3,
        batch_size: int = 32,
        # --- tabular training ---
        n_jobs: int = -1,
        use_gpu: bool | None = None,
        encode_categoricals: str | None = "ordinal",
        scale_numeric: bool = False,
        # --- fold model storage ---
        save_fold_models: bool = False,
        cleanup_fold_files: bool = True,
    ) -> dict:
        """Stage 1: split → (HPO) → k-fold OOF loop → final model training.

        After this stage all OOF artifacts and final base models are saved to
        disk.  The next stage (``train_meta_learner_stage``) can be run in a
        separate Python session.

        Args:
            df:                     Input DataFrame.
            label_col:              Cause-of-death label column.
            text_col:               Free-text narrative column.  Required when
                                    ``text_models`` is non-empty.
            feature_cols:           Tabular feature columns.  Required when
                                    ``tabular_models`` is non-empty.
            test_size:              Test fraction. Default 0.2.
            random_state:           Seed. Default 42.
            stratify:               Stratified split. Default True.
            val_size:               Internal val fraction for text fold models
                                    (early stopping).  Default 0.1.
            gradient_checkpointing: Enable gradient checkpointing. Default False.
            early_stopping_patience: Text early stopping patience. Default 3.
            batch_size:             Inference batch size for text. Default 32.
            n_jobs:                 CPU parallelism for tabular. Default -1.
            use_gpu:                GPU flag for tabular. None = auto-detect.
            encode_categoricals:    Categorical encoding. Default ``"ordinal"``.
            scale_numeric:          Apply StandardScaler. Default False.
            save_fold_models:       Keep fold model weights on disk. Default False.

        Returns:
            dict with ``oof_meta_X``, ``oof_y``, ``label2id``, ``id2label``,
            ``n_folds``, ``output_dir``.
        """
        if self.text_models and text_col is None:
            raise ValueError("text_col is required when text_models is non-empty.")
        if self.tabular_models and feature_cols is None:
            raise ValueError("feature_cols is required when tabular_models is non-empty.")

        # Lazy imports
        from ..utils.split            import split
        from ..text.dataset           import prepare_dataset as text_prepare
        from ..text.train             import train            as text_train
        from ..tabular.dataset        import prepare_dataset as tab_prepare
        from ..tabular.train          import train            as tab_train
        from ..text.hpo               import optimize         as text_optimize
        from ..tabular.hpo            import optimize         as tab_optimize

        self._text_col    = text_col
        self._feature_cols = feature_cols
        self._label_col   = label_col

        data_dir = self.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # --- Inherited OOF loading (load_oof_from) --------------------------
        # When load_oof_from is set:
        #   • Reuse the exact train/test split from the source run (row alignment).
        #   • Only run OOF for the NEW models listed in text_models/tabular_models.
        #   • Concatenate [inherited_oof | new_oof] column-wise.
        #   • Store absolute final_dir paths in model_sources so predict_test()
        #     can locate weights for both old and new models across sessions.
        inherited_oof_X: np.ndarray | None = None
        inherited_model_sources: list[dict] = []

        if self.load_oof_from:
            _src = self.load_oof_from
            _ioof_file = _src / "oof" / "oof_meta_X.npy"
            if not _ioof_file.exists():
                raise FileNotFoundError(
                    f"oof_meta_X.npy not found in {_src / 'oof'}.  "
                    "Run train_base_models() on the source stacking run first."
                )
            inherited_oof_X = np.load(_ioof_file)
            logger.info(
                "load_oof_from: loaded inherited OOF matrix %s from %s",
                inherited_oof_X.shape, _src,
            )

            # Reuse the source split — OOF row i must mean the same training
            # sample in both inherited and new matrices.
            _src_data = _src / "data"
            if not (_src_data / "train_df.csv").exists():
                raise FileNotFoundError(
                    f"train_df.csv not found in {_src_data}.  "
                    "The source run must have been completed with train_base_models()."
                )
            logger.info(
                "load_oof_from: reusing train/test split from %s to guarantee "
                "OOF row alignment.  The df argument is still used to prepare "
                "features for the new models.", _src,
            )
            self.train_df = pd.read_csv(_src_data / "train_df.csv")
            self.test_df  = pd.read_csv(_src_data / "test_df.csv")

            if inherited_oof_X.shape[0] != len(self.train_df):
                raise ValueError(
                    f"Inherited OOF has {inherited_oof_X.shape[0]} rows but "
                    f"train_df.csv has {len(self.train_df)} rows — shape mismatch.  "
                    "Ensure load_oof_from points to the correct source run."
                )

            # Load model_sources from source metadata (supports chaining)
            _src_meta_path = _src / "oof" / "oof_metadata.json"
            if _src_meta_path.exists():
                with open(_src_meta_path) as _f:
                    _src_meta = json.load(_f)
                if "model_sources" in _src_meta:
                    inherited_model_sources = _src_meta["model_sources"]
                else:
                    # Backward compat: construct from old flat metadata fields
                    _old_text    = _src_meta.get("text_specs", [])
                    _old_tabular = _src_meta.get("tabular_specs", [])
                    _old_nc      = _src_meta.get("n_classes", 1)
                    _col = 0
                    for _j, _spec in enumerate(_old_text):
                        inherited_model_sources.append({
                            "type": "text", "local_index": _j, "spec": _spec,
                            "final_dir": str((_src / "final" / f"text_{_j}").resolve()),
                            "col_start": _col, "n_cols": _old_nc,
                        })
                        _col += _old_nc
                    for _j, _spec in enumerate(_old_tabular):
                        inherited_model_sources.append({
                            "type": "tabular", "local_index": _j, "spec": _spec,
                            "final_dir": str((_src / "final" / f"tabular_{_j}").resolve()),
                            "col_start": _col, "n_cols": _old_nc,
                        })
                        _col += _old_nc
            else:
                logger.warning(
                    "No oof_metadata.json found in %s — cannot reconstruct "
                    "inherited model_sources.  predict_test() may fail for "
                    "inherited models.", _src / "oof",
                )
        else:
            # --- Step 1: split ----------------------------------------------
            self.train_df, self.test_df = split(
                df, label_col=label_col, text_col=text_col,
                test_size=test_size, random_state=random_state, stratify=stratify,
            )
            logger.info("Split: %d train / %d test", len(self.train_df), len(self.test_df))

        # Save (or re-save) splits for predict_test() in future sessions
        self.train_df.to_csv(data_dir / "train_df.csv", index=False)
        self.test_df.to_csv(data_dir  / "test_df.csv",  index=False)

        # --- Step 2: prepare datasets (once, shared label maps) -------------
        label2id = id2label = None

        if self.text_models:
            first_text_spec = self.text_models[0]
            train_text_ds, test_text_ds, label2id, id2label = text_prepare(
                self.train_df, self.test_df,
                text_col=text_col, label_col=label_col,
                model_name=first_text_spec["model_name"],
                max_length=first_text_spec.get("max_length", 512),
            )

        # Tabular preprocessing
        X_train = X_test = y_train = y_test = None
        preprocessor = feature_names = None

        if self.tabular_models:
            (X_train, X_test, y_train, y_test,
             preprocessor, lbl2id, id2lbl, feature_names) = tab_prepare(
                self.train_df, self.test_df,
                feature_cols=feature_cols, label_col=label_col,
                encode_categoricals=encode_categoricals,
                scale_numeric=scale_numeric,
            )
            if label2id is None:
                label2id, id2label = lbl2id, id2lbl

        self.label2id = label2id
        self.id2label = id2label
        sorted_ids    = sorted(id2label.keys())
        n_classes     = len(sorted_ids)

        # Persist label maps
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "label2id.json", "w") as fh:
            json.dump(label2id, fh, indent=2)
        with open(self.output_dir / "id2label.json", "w") as fh:
            json.dump({str(k): v for k, v in id2label.items()}, fh, indent=2)

        # Save tabular test arrays for predict_test()
        if X_test is not None:
            np.save(data_dir / "X_test.npy", X_test)
            np.save(data_dir / "y_test.npy", y_test)

        # --- Step 3: optional HPO (once, before OOF loop) -------------------
        hpo_dir = self.output_dir / "hpo"
        self.text_best_hp    = []
        self.tabular_best_hp = []

        for i, spec in enumerate(self.text_models):
            if spec.get("use_optimize", False):
                logger.info("Text model %d (%s): running HPO ...", i, spec["model_name"])
                best_hp, _ = text_optimize(
                    train_dataset=train_text_ds,
                    label2id=label2id, id2label=id2label,
                    model_name=spec["model_name"],
                    output_dir=hpo_dir / f"text_{i}",
                    n_trials=spec.get("n_trials", 20),
                    metric=spec.get("optimize_metric", "f1_macro"),
                    search_space=spec.get("search_space"),
                    random_state=random_state,
                    use_lora=spec.get("use_lora", False),
                    gradient_checkpointing=gradient_checkpointing,
                    early_stopping_patience=early_stopping_patience,
                )
                self.text_best_hp.append(best_hp)
            else:
                self.text_best_hp.append(spec.get("hyperparams") or {})

        for i, spec in enumerate(self.tabular_models):
            if spec.get("use_optimize", False):
                if spec["model_name"] == _INSILICOVA:
                    logger.info(
                        "Tabular model %d (insilicova): HPO not applicable — "
                        "InSilicoVA uses a fixed Bayesian cause database.  "
                        "Using spec hyperparams.", i
                    )
                    self.tabular_best_hp.append(spec.get("hyperparams") or {})
                else:
                    logger.info("Tabular model %d (%s): running HPO ...", i, spec["model_name"])
                    best_hp, _ = tab_optimize(
                        X_train=X_train, y_train=y_train,
                        label2id=label2id, id2label=id2label,
                        model_name=spec["model_name"],
                        output_dir=hpo_dir / f"tabular_{i}",
                        n_trials=spec.get("n_trials", 20),
                        metric=spec.get("optimize_metric", "f1_macro"),
                        search_space=spec.get("search_space"),
                        random_state=random_state,
                        n_jobs=n_jobs, use_gpu=use_gpu,
                    )
                    self.tabular_best_hp.append(best_hp)
            else:
                self.tabular_best_hp.append(spec.get("hyperparams"))

        # --- Step 4: k-fold OOF loop ----------------------------------------
        oof_dir = self.output_dir / "oof"
        n_inherited_cols = inherited_oof_X.shape[1] if inherited_oof_X is not None else 0
        n_new_models     = len(self.text_models) + len(self.tabular_models)
        expected_cols    = n_inherited_cols + n_new_models * n_classes

        # Resume check: skip loop only when assembled matrix has the right shape.
        # If load_oof_from was used and the file exists but has only new-model
        # columns (incomplete previous run), re-generate and re-combine.
        _oof_file = oof_dir / "oof_meta_X.npy"
        _skip_oof = (
            self.resume
            and _oof_file.exists()
            and np.load(_oof_file, mmap_mode="r").shape[1] == expected_cols
        )

        if _skip_oof:
            logger.info("OOF meta-features already assembled — skipping OOF loop.")
            self.oof_meta_X = np.load(_oof_file)
            self.oof_y      = np.load(oof_dir / "oof_y.npy")
        else:
            new_oof_X, self.oof_y = generate_oof_predictions(
                text_model_specs=self.text_models,
                tabular_model_specs=self.tabular_models,
                train_text_dataset=train_text_ds if self.text_models else None,
                X_train=X_train,
                y_train=y_train if y_train is not None else np.array([]),
                label2id=label2id,
                id2label=id2label,
                n_folds=self.n_folds,
                random_state=random_state,
                output_dir=oof_dir,
                val_size=val_size,
                gradient_checkpointing=gradient_checkpointing,
                early_stopping_patience=early_stopping_patience,
                batch_size=batch_size,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
                resume=self.resume,
                save_fold_models=save_fold_models,
                cleanup_fold_files=cleanup_fold_files,
                text_best_hp=self.text_best_hp    or None,
                tabular_best_hp=self.tabular_best_hp or None,
                train_df=self.train_df,
                feature_cols=feature_cols,
            )

            # Prepend inherited columns (if any) and overwrite the file.
            if inherited_oof_X is not None:
                self.oof_meta_X = np.concatenate([inherited_oof_X, new_oof_X], axis=1)
                logger.info(
                    "Combined inherited OOF %s + new OOF %s → %s",
                    inherited_oof_X.shape, new_oof_X.shape, self.oof_meta_X.shape,
                )
            else:
                self.oof_meta_X = new_oof_X

            # generate_oof_predictions() already saved new_oof_X; overwrite with
            # the combined matrix so future resume picks up the full matrix.
            oof_dir.mkdir(parents=True, exist_ok=True)
            np.save(_oof_file,            self.oof_meta_X)
            np.save(oof_dir / "oof_y.npy", self.oof_y)

        # Build model_sources — flat list of all models (inherited + new) in
        # column order.  Absolute final_dir paths survive across sessions and
        # across directories, enabling predict_test() to find weights for both
        # inherited models (in the source run) and new models (in output_dir).
        final_dir = self.output_dir / "final"
        col_offset = n_inherited_cols
        new_model_sources: list[dict] = []
        for i, spec in enumerate(self.text_models):
            new_model_sources.append({
                "type":        "text",
                "local_index": i,
                "spec":        spec,
                "final_dir":   str((final_dir / f"text_{i}").resolve()),
                "col_start":   col_offset + i * n_classes,
                "n_cols":      n_classes,
            })
        for i, spec in enumerate(self.tabular_models):
            new_model_sources.append({
                "type":        "tabular",
                "local_index": i,
                "spec":        spec,
                "final_dir":   str((final_dir / f"tabular_{i}").resolve()),
                "col_start":   col_offset + (len(self.text_models) + i) * n_classes,
                "n_cols":      n_classes,
            })
        all_model_sources = inherited_model_sources + new_model_sources

        # Save OOF metadata for cross-session handoff
        oof_meta_names = []
        for src in all_model_sources:
            mn = src["spec"]["model_name"].replace("/", "_").replace("-", "_")
            prefix = f"text_{src['local_index']}_{mn}" if src["type"] == "text" \
                     else f"tab_{src['local_index']}_{mn}"
            for cid in sorted_ids:
                oof_meta_names.append(f"{prefix}_prob_{cid}")

        oof_metadata = {
            "n_folds":          self.n_folds,
            "n_text_models":    len(self.text_models),
            "n_tabular_models": len(self.tabular_models),
            "n_classes":        n_classes,
            "n_train":          len(self.train_df),
            "n_test":           len(self.test_df),
            "text_col":         text_col,
            "feature_cols":     feature_cols,
            "label_col":        label_col,
            "label2id":         label2id,
            "id2label":         {str(k): v for k, v in id2label.items()},
            "text_specs":       self.text_models,
            "tabular_specs":    self.tabular_models,
            "model_sources":    all_model_sources,
            "meta_feature_names": oof_meta_names,
            "text_best_hp":     self.text_best_hp,
            "tabular_best_hp":  self.tabular_best_hp,
            "inherited_from":   str(self.load_oof_from.resolve()) if self.load_oof_from else None,
        }
        with open(oof_dir / "oof_metadata.json", "w") as fh:
            json.dump(oof_metadata, fh, indent=2, default=str)

        # --- Step 5: train final base models on full training set -----------
        final_dir = self.output_dir / "final"

        for i, spec in enumerate(self.text_models):
            model_dir  = final_dir / f"text_{i}"
            done_marker = model_dir / "training_metadata.json"

            if self.resume and done_marker.exists():
                logger.info("Final text model %d already trained — skipping.", i)
                continue

            logger.info("Training final text model %d/%d: %s",
                        i + 1, len(self.text_models), spec["model_name"])
            final_val = None if spec.get("use_optimize") else val_size

            text_train(
                train_dataset=train_text_ds,
                label2id=label2id, id2label=id2label,
                model_name=spec["model_name"],
                output_dir=model_dir,
                hyperparams=self.text_best_hp[i] or {},
                val_size=final_val,
                use_lora=spec.get("use_lora", False),
                gradient_checkpointing=gradient_checkpointing,
                early_stopping_patience=early_stopping_patience,
                resume=self.resume,
            )

        for i, spec in enumerate(self.tabular_models):
            model_dir   = final_dir / f"tabular_{i}"
            done_marker = model_dir / "training_metadata.json"

            if self.resume and done_marker.exists():
                logger.info("Final tabular model %d already trained — skipping.", i)
                continue

            logger.info("Training final tabular model %d/%d: %s",
                        i + 1, len(self.tabular_models), spec["model_name"])

            if spec["model_name"] == _INSILICOVA:
                _insilicova_save_config(
                    spec, self.tabular_best_hp[i], label2id, id2label,
                    feature_cols, model_dir, label_col=label_col,
                )
            else:
                tab_train(
                    X_train=X_train, y_train=y_train,
                    label2id=label2id, id2label=id2label,
                    model_name=spec["model_name"],
                    output_dir=model_dir,
                    hyperparams=self.tabular_best_hp[i],
                    preprocessor=preprocessor, feature_names=feature_names,
                    random_state=random_state, n_jobs=n_jobs, use_gpu=use_gpu,
                )

        logger.info("Stage 1 complete. OOF shape: %s", self.oof_meta_X.shape)
        return {
            "oof_meta_X": self.oof_meta_X,
            "oof_y":      self.oof_y,
            "label2id":   label2id,
            "id2label":   id2label,
            "n_folds":    self.n_folds,
            "output_dir": self.output_dir,
        }

    # ------------------------------------------------------------------
    # Stage 2: train meta-learner(s) + select best
    # ------------------------------------------------------------------

    def train_meta_learner_stage(
        self,
        meta_learners: list[dict] | dict | None = None,
        metric: str | None = None,
        meta_cv_folds: int = 3,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> dict:
        """Stage 2: train meta-learner candidates and select the best.

        Can be called after ``train_base_models()`` in the same session, or in
        a fresh session (OOF data loaded automatically from disk).

        When multiple meta-learner specs are provided, each is scored via
        stratified ``meta_cv_folds``-fold CV on the OOF meta-features.  The
        highest-scoring model is fitted on all OOF data and saved.

        Args:
            meta_learners:  Override ``self.meta_learner_specs``.  One dict or
                            a list of dicts.  ``None`` uses the specs from
                            ``__init__``.
            metric:         Selection metric.  Default: ``self.meta_select_metric``
                            (``f1_macro`` unless overridden at init).
            meta_cv_folds:  CV folds for meta-learner selection. Default 3.
            random_state:   Seed. Default 42.
            n_jobs:         CPU parallelism for meta-learner instantiation.

        Returns:
            dict with ``meta_learner``, ``meta_scores``, ``best_meta_name``,
            ``output_dir``.
        """
        self._ensure_oof_loaded()

        if meta_learners is not None:
            specs = [meta_learners] if isinstance(meta_learners, dict) else list(meta_learners)
        else:
            specs = self.meta_learner_specs

        metric = metric or self.meta_select_metric

        logger.info("Stage 2: training %d meta-learner candidate(s) ...", len(specs))

        meta_dir = self.output_dir / "meta_learner"
        meta_dir.mkdir(parents=True, exist_ok=True)

        scores = {}
        for spec in specs:
            name  = spec["model_name"]
            model = _resolve_meta_learner(spec, n_jobs=n_jobs, random_state=random_state)

            if len(specs) > 1:
                score = _score_meta_candidate(
                    model, self.oof_meta_X, self.oof_y,
                    self.id2label, metric, meta_cv_folds, random_state,
                )
                scores[name] = round(score, 6)
                logger.info("  %s  %s = %.4f", name, metric, score)
            else:
                scores[name] = None

        # Select best
        if len(specs) > 1:
            best_name = max(scores, key=lambda k: scores[k])
            logger.info("Best meta-learner: %s (%.4f)", best_name, scores[best_name])
        else:
            best_name = specs[0]["model_name"]

        best_spec  = next(s for s in specs if s["model_name"] == best_name)
        best_model = _resolve_meta_learner(best_spec, n_jobs=n_jobs, random_state=random_state)
        best_model.fit(self.oof_meta_X, self.oof_y)

        self.meta_learner = best_model
        self.meta_scores  = scores

        # Save
        joblib.dump(best_model, meta_dir / "meta_learner.joblib")
        with open(meta_dir / "meta_scores.json", "w") as fh:
            json.dump(scores, fh, indent=2)

        meta_metadata = {
            "best_meta_name":    best_name,
            "meta_select_metric": metric,
            "meta_cv_folds":     meta_cv_folds,
            "meta_scores":       scores,
            "meta_specs":        specs,
            "oof_meta_X_shape":  list(self.oof_meta_X.shape),
        }
        with open(meta_dir / "meta_learner_metadata.json", "w") as fh:
            json.dump(meta_metadata, fh, indent=2, default=str)

        logger.info("Stage 2 complete. Meta-learner saved to %s", meta_dir)
        return {
            "meta_learner":    best_model,
            "meta_scores":     scores,
            "best_meta_name":  best_name,
            "output_dir":      self.output_dir,
        }

    # ------------------------------------------------------------------
    # Stage 3: test prediction via meta-learner
    # ------------------------------------------------------------------

    def predict_test(
        self,
        top_k: int = 3,
        batch_size: int = 32,
    ) -> PredictionResult:
        """Stage 3: predict on the held-out test set using the meta-learner.

        Loads final base models and meta-learner from disk if not already in
        memory.  Stacks test probability matrices into a meta-feature vector
        for each test sample, then passes through the meta-learner.

        Can be called after ``train_meta_learner_stage()`` in the same session,
        or in a fresh session (all artifacts reloaded from disk).

        Args:
            top_k:      Number of top classes in the topk output. Default 3.
            batch_size: Text inference batch size. Default 32.

        Returns:
            :class:`~multimodalva.utils.types.PredictionResult`.
        """
        # Lazy imports
        from ..text.dataset    import prepare_dataset as text_prepare
        from ..text.predict    import predict          as text_predict
        from ..tabular.dataset import prepare_dataset as tab_prepare
        from ..tabular.predict import predict          as tab_predict

        # Ensure meta-learner is loaded
        if self.meta_learner is None:
            meta_model_path = self.output_dir / "meta_learner" / "meta_learner.joblib"
            if not meta_model_path.exists():
                raise RuntimeError(
                    "Meta-learner not found at %s.  "
                    "Run train_meta_learner_stage() first." % meta_model_path
                )
            self.meta_learner = joblib.load(meta_model_path)
            logger.info("Meta-learner loaded from %s", meta_model_path)

        # Ensure label maps are loaded
        self._ensure_oof_loaded()

        # Load test data
        data_dir = self.output_dir / "data"
        train_df = pd.read_csv(data_dir / "train_df.csv")
        test_df  = pd.read_csv(data_dir / "test_df.csv")

        sorted_ids = sorted(self.id2label.keys())
        n_classes  = len(sorted_ids)
        n_test     = len(test_df)

        # Load model_sources — flat ordered list of all base models
        # (inherited + new) with absolute final_dir paths and column ranges.
        # Falls back to constructing from self.text_models / self.tabular_models
        # for runs created before model_sources was introduced.
        _oof_meta_path = self.output_dir / "oof" / "oof_metadata.json"
        model_sources: list[dict] | None = None
        if _oof_meta_path.exists():
            with open(_oof_meta_path) as _f:
                _oof_meta = json.load(_f)
            model_sources = _oof_meta.get("model_sources")

        if not model_sources:
            # Backward-compat fallback: only current-run models, no inheritance
            final_dir_fb = self.output_dir / "final"
            model_sources = []
            for i, spec in enumerate(self.text_models):
                model_sources.append({
                    "type": "text", "local_index": i, "spec": spec,
                    "final_dir": str((final_dir_fb / f"text_{i}").resolve()),
                    "col_start": i * n_classes, "n_cols": n_classes,
                })
            for i, spec in enumerate(self.tabular_models):
                model_sources.append({
                    "type": "tabular", "local_index": i, "spec": spec,
                    "final_dir": str((final_dir_fb / f"tabular_{i}").resolve()),
                    "col_start": (len(self.text_models) + i) * n_classes,
                    "n_cols": n_classes,
                })

        n_total_cols = sum(s["n_cols"] for s in model_sources)
        meta_X_test  = np.zeros((n_test, n_total_cols), dtype=float)

        # Pre-load tabular test arrays once (shared by all sklearn tabular models).
        # All tabular models in model_sources share the same feature space, so
        # one preprocessed X_test works for all of them.
        X_test_npy = data_dir / "X_test.npy"
        y_test_npy = data_dir / "y_test.npy"
        _has_sklearn_tab = any(
            s["type"] == "tabular" and s["spec"].get("model_name") != _INSILICOVA
            for s in model_sources
        )
        X_test = y_test = None
        if _has_sklearn_tab and X_test_npy.exists():
            X_test = np.load(X_test_npy)
            y_test = np.load(y_test_npy)
        elif _has_sklearn_tab:
            (_, X_test, _, y_test, _, _, _, _) = tab_prepare(
                train_df, test_df,
                feature_cols=self._feature_cols,
                label_col=self._label_col,
                encode_categoricals="ordinal",
            )

        # --- Iterate over all model sources (inherited + new) ---------------
        for source in model_sources:
            model_dir  = Path(source["final_dir"])
            spec       = source["spec"]
            col_s      = source["col_start"]
            n_cols     = source["n_cols"]
            model_name = spec["model_name"]

            if source["type"] == "text":
                max_length = spec.get("max_length", 512)
                logger.info(
                    "Predicting test — text model %s (weights: %s) ...",
                    model_name, model_dir,
                )
                _, test_text_ds, _, _ = text_prepare(
                    train_df, test_df,
                    text_col=self._text_col, label_col=self._label_col,
                    model_name=model_name, max_length=max_length,
                )
                result     = text_predict(model_dir, test_text_ds, batch_size=batch_size, top_k=1)
                fold_probs = _extract_probs(result, sorted_ids)

            else:  # tabular
                logger.info(
                    "Predicting test — tabular model %s (weights: %s) ...",
                    model_name, model_dir,
                )
                if model_name == _INSILICOVA:
                    fold_probs = _insilicova_predict_df(
                        model_dir, test_df, sorted_ids, train_df=train_df,
                    )
                else:
                    result     = tab_predict(model_dir, X_test, y_test, top_k=1)
                    fold_probs = _extract_probs(result, sorted_ids)

            meta_X_test[:, col_s:col_s + n_cols] = fold_probs

        # --- Meta-learner final prediction ----------------------------------
        meta_proba = self.meta_learner.predict_proba(meta_X_test)  # (n_test, n_classes)

        # Reorder columns to canonical sorted_ids order
        classes   = list(self.meta_learner.classes_)
        col_order = [classes.index(cid) for cid in sorted_ids]
        meta_proba = meta_proba[:, col_order]

        true_labels = test_df[self._label_col].tolist()
        self.predictions = _assemble_prediction_result(
            meta_proba, true_labels, self.id2label, top_k
        )

        # Save predictions
        pred_dir = self.output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        self.predictions.top1.to_csv(pred_dir / "top1.csv", index=False)
        self.predictions.topk.to_csv(pred_dir / "topk.csv", index=False)
        self.predictions.full.to_csv(pred_dir / "full.csv", index=False)

        voted_acc = (
            self.predictions.top1["true_label"] == self.predictions.top1["predicted_label"]
        ).mean()
        logger.info("Stage 3 complete. Test accuracy: %.4f", voted_acc)

        return self.predictions

    # ------------------------------------------------------------------
    # Convenience: all stages in sequence
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        label_col: str,
        text_col: str | None = None,
        feature_cols: list[str] | None = None,
        # --- split ---
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        # --- text ---
        val_size: float = 0.1,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 3,
        batch_size: int = 32,
        # --- tabular ---
        n_jobs: int = -1,
        use_gpu: bool | None = None,
        encode_categoricals: str | None = "ordinal",
        scale_numeric: bool = False,
        # --- fold storage ---
        save_fold_models: bool = False,
        cleanup_fold_files: bool = True,
        # --- meta-learner ---
        meta_learners: list[dict] | dict | None = None,
        meta_select_metric: str | None = None,
        meta_cv_folds: int = 3,
        # --- inference ---
        top_k: int = 3,
    ) -> dict:
        """Convenience wrapper: Stage 1 → Stage 2 → Stage 3 in sequence.

        Equivalent to calling::

            clf.train_base_models(df, ...)
            clf.train_meta_learner_stage(...)
            predictions = clf.predict_test(...)

        Returns:
            dict with ``predictions``, ``oof_meta_X``, ``meta_learner``,
            ``meta_scores``, ``label2id``, ``id2label``, ``output_dir``.
        """
        self.train_base_models(
            df=df, label_col=label_col,
            text_col=text_col, feature_cols=feature_cols,
            test_size=test_size, random_state=random_state, stratify=stratify,
            val_size=val_size,
            gradient_checkpointing=gradient_checkpointing,
            early_stopping_patience=early_stopping_patience,
            batch_size=batch_size,
            n_jobs=n_jobs, use_gpu=use_gpu,
            encode_categoricals=encode_categoricals,
            scale_numeric=scale_numeric,
            save_fold_models=save_fold_models,
            cleanup_fold_files=cleanup_fold_files,
        )
        self.train_meta_learner_stage(
            meta_learners=meta_learners,
            metric=meta_select_metric,
            meta_cv_folds=meta_cv_folds,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        predictions = self.predict_test(top_k=top_k, batch_size=batch_size)

        # Save top-level metadata
        metadata = {
            "output_dir":       str(self.output_dir),
            "n_text_models":    len(self.text_models),
            "n_tabular_models": len(self.tabular_models),
            "n_folds":          self.n_folds,
            "meta_scores":      self.meta_scores,
            "label2id":         self.label2id,
            "id2label":         {str(k): v for k, v in self.id2label.items()},
            "n_train":          len(self.train_df),
            "n_test":           len(self.test_df),
            "n_classes":        len(self.label2id),
        }
        with open(self.output_dir / "training_metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)

        return {
            "predictions":  predictions,
            "oof_meta_X":   self.oof_meta_X,
            "meta_learner": self.meta_learner,
            "meta_scores":  self.meta_scores,
            "label2id":     self.label2id,
            "id2label":     self.id2label,
            "output_dir":   self.output_dir,
        }
