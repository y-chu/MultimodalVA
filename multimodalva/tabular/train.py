"""
Step 3 (tabular pipeline): Fit a sklearn-compatible classifier and save artifacts.

Input:  X_train, y_train from prepare_dataset(); hyperparams dict
Output: (model, metadata)
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import subprocess

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

# Supported model aliases: (module path, class name)
SUPPORTED_MODELS: dict[str, tuple[str, str]] = {
    "catboost":      ("catboost",                  "CatBoostClassifier"),
    "lightgbm":      ("lightgbm",                  "LGBMClassifier"),
    "gbdt":          ("sklearn.ensemble",           "GradientBoostingClassifier"),
    "xgboost":       ("xgboost",                   "XGBClassifier"),
    "mlp":           ("sklearn.neural_network",     "MLPClassifier"),
    "random_forest": ("sklearn.ensemble",           "RandomForestClassifier"),
    "naive_bayes":   ("sklearn.naive_bayes",        "GaussianNB"),
    "knn":           ("sklearn.neighbors",          "KNeighborsClassifier"),
    "svm":           ("sklearn.svm",                "SVC"),
}

DEFAULT_HYPERPARAMS: dict[str, dict] = {
    "catboost":      {"iterations": 300, "learning_rate": 0.05, "depth": 6, "verbose": 0},
    "lightgbm":      {"n_estimators": 300, "learning_rate": 0.05, "max_depth": -1, "verbose": -1},
    "gbdt":          {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
    "xgboost":       {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "verbosity": 0},
    "mlp":           {"hidden_layer_sizes": (128, 64), "max_iter": 300},
    "random_forest": {"n_estimators": 200, "max_depth": None},
    "naive_bayes":   {},
    "knn":           {"n_neighbors": 5, "weights": "uniform"},
    "svm":           {"probability": True, "decision_function_shape": "ovr"},
}

# Parameters that must always be set regardless of user-supplied hyperparams.
# Prevents accidental breakage (e.g. SVC needs probability=True for predict_proba).
_FORCED_PARAMS: dict[str, dict] = {
    "svm": {"probability": True},
}

# Maps model_name → constructor parameter name for CPU parallelism.
_NJOBS_PARAM: dict[str, str] = {
    "random_forest": "n_jobs",
    "knn":           "n_jobs",
    "xgboost":       "n_jobs",
    "lightgbm":      "n_jobs",
}

# Parameters injected when CUDA is available (XGBoost >= 2.0 API).
# MPS (Apple Silicon) is not supported by tree model GPU backends.
_CUDA_PARAMS: dict[str, dict] = {
    "xgboost":  {"device": "cuda"},
    "lightgbm": {"device_type": "gpu"},
    "catboost": {"task_type": "GPU"},
}


def _detect_cuda() -> bool:
    """Return True if CUDA is available.

    Tries torch first; falls back to nvidia-smi if torch is not installed.
    MPS (Apple Silicon) is not supported by tree model GPU backends.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        return subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        ).returncode == 0
    except Exception:
        return False


def _build_model(
    model_name: str,
    hyperparams: dict,
    random_state: int,
    n_jobs: int = -1,
    use_gpu: bool = False,
) -> Any:
    """Instantiate the model from its alias and hyperparams dict."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Choose from: {list(SUPPORTED_MODELS)}."
        )
    module_path, class_name = SUPPORTED_MODELS[model_name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    hp = dict(hyperparams)
    # Inject random_state only where the constructor accepts it
    try:
        sig = inspect.signature(cls.__init__)
        if "random_state" in sig.parameters:
            hp.setdefault("random_state", random_state)
    except (ValueError, TypeError):
        pass

    # CPU parallelism — setdefault so user-supplied values take precedence
    if model_name in _NJOBS_PARAM:
        hp.setdefault(_NJOBS_PARAM[model_name], n_jobs)

    # GPU acceleration — setdefault so user-supplied values take precedence
    if use_gpu and model_name in _CUDA_PARAMS:
        for k, v in _CUDA_PARAMS[model_name].items():
            hp.setdefault(k, v)

    # Override params that must always be set (e.g. SVC needs probability=True)
    hp.update(_FORCED_PARAMS.get(model_name, {}))

    return cls(**hp)


def _serialize_hyperparams(hp: dict) -> dict:
    """Convert hyperparams to JSON-safe types (e.g. tuples → lists)."""
    return json.loads(
        json.dumps(hp, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    )


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    hyperparams: dict | None = None,
    preprocessor: ColumnTransformer | None = None,
    feature_names: list[str] | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
    use_gpu: bool | None = None,
) -> tuple[Any, dict]:
    """Fit a tabular classifier and save all artifacts to output_dir.

    Saves:
        model.joblib          — {"model": fitted estimator, "preprocessor": ColumnTransformer}
        label2id.json
        id2label.json         — keys are strings (JSON requirement)
        hyperparams.json
        training_metadata.json

    Args:
        X_train:      Preprocessed feature matrix from prepare_dataset().
        y_train:      Integer label array from prepare_dataset().
        label2id:     Label-to-integer mapping from prepare_dataset().
        id2label:     Integer-to-label mapping from prepare_dataset().
        model_name:   Model alias — one of SUPPORTED_MODELS keys.
        output_dir:   Directory to save all artifacts.
        hyperparams:  Hyperparameter dict. Merged over DEFAULT_HYPERPARAMS[model_name].
                      None = use defaults only.
        preprocessor: Fitted ColumnTransformer from prepare_dataset(). Bundled
                      into model.joblib for later use on raw new data.
        random_state: Random seed. Default 42.
        n_jobs:       Parallel jobs for CPU-parallel models (random_forest, knn,
                      xgboost, lightgbm). -1 = all cores. Default -1.
        use_gpu:      Enable CUDA GPU acceleration for supported models (xgboost,
                      lightgbm, catboost). None = auto-detect. Default None.
                      MPS (Apple Silicon) is not supported by tree model GPU
                      backends — use None or False on Apple Silicon.

    Returns:
        model:    The fitted sklearn-compatible estimator.
        metadata: Dict with output_dir, model_name, hyperparams, label2id, id2label.

    Raises:
        ValueError: If model_name is not a supported alias.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge defaults and user hyperparams
    hp = dict(DEFAULT_HYPERPARAMS.get(model_name, {}))
    if hyperparams:
        hp.update(hyperparams)

    # Resolve GPU availability once (None = auto-detect CUDA)
    gpu = _detect_cuda() if use_gpu is None else use_gpu

    model = _build_model(model_name, hp, random_state, n_jobs=n_jobs, use_gpu=gpu)
    logger.info(
        "Fitting %s (n_jobs=%d, gpu=%s) with hyperparams: %s",
        model_name, n_jobs, gpu, hp,
    )
    model.fit(X_train, y_train)

    # Bundle model + preprocessor so predict() can reload both from one file.
    # The preprocessor is not used by predict() when X_test is already preprocessed,
    # but it is available for users who want to score raw new samples later.
    bundle = {"model": model, "preprocessor": preprocessor}
    joblib.dump(bundle, output_dir / "model.joblib")

    # Persist feature names (for SHAP and visualization)
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names or [], f, indent=2)

    # Persist label maps
    with open(output_dir / "label2id.json", "w") as f:
        json.dump(label2id, f, indent=2)
    with open(output_dir / "id2label.json", "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)

    hp_safe = _serialize_hyperparams(hp)
    with open(output_dir / "hyperparams.json", "w") as f:
        json.dump(hp_safe, f, indent=2)

    metadata = {
        "output_dir": str(output_dir),
        "model_name": model_name,
        "hyperparams": hp_safe,
        "label2id": label2id,
        "id2label": id2label,
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete. Artifacts saved to %s", output_dir)
    return model, metadata
