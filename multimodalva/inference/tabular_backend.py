"""Tabular backend for ``predict_from_pretrained()``.

Serves a MultimodalVA tabular ``final/`` directory (``model.joblib`` bundling the
estimator with its fitted preprocessor) or a foreign scikit-learn-compatible
estimator saved with joblib. Input columns are matched **by name**, never by
position: a silent misalignment gives confident nonsense.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.numpy_compat import load_joblib_compat
from .backends import _read_pretrained_id2label, reindex_pretrained_features
from .checks import PretrainedChecks, _is_placeholder_feature_cols

logger = logging.getLogger(__name__)


class TabularPretrainedBackend:
    """Estimator (+ optional preprocessor) loaded from ``model.joblib``."""

    def __init__(self, model, preprocessor, feature_cols, id2label, class_index,
                 checks, missing_feature_method):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_cols = feature_cols
        self.id2label = id2label
        self.class_index = class_index
        self.checks = checks
        self.missing_feature_method = missing_feature_method

    @classmethod
    def from_pretrained(cls, artifact_dir: Path, checks: PretrainedChecks, *,
                        feature_cols: list[str] | None = None,
                        missing_feature_method: str = "error",
                        id2label: dict | None = None, **_ignored):
        bundle = load_joblib_compat(artifact_dir / "model.joblib")
        if isinstance(bundle, dict) and "model" in bundle:
            model, preprocessor = bundle["model"], bundle.get("preprocessor")
        else:
            model, preprocessor = bundle, None

        cols = _align_pretrained_feature_cols(
            model, preprocessor, artifact_dir, checks, caller=feature_cols
        )

        classes = np.asarray(model.classes_)
        sidecar = artifact_dir / "id2label.json"
        if np.issubdtype(classes.dtype, np.integer):
            # MultimodalVA encodes causes as integer ids; the names live in the
            # sidecar, which also covers classes absent from the training split.
            native = None
            n_classes = (len(json.loads(sidecar.read_text()))
                         if sidecar.is_file() else int(classes.max()) + 1)
            class_index = classes.astype(int)
        else:
            native = {i: str(c) for i, c in enumerate(classes)}
            n_classes = len(classes)
            class_index = np.arange(len(classes))
        names = _read_pretrained_id2label(native, artifact_dir, checks,
                                          n_classes=n_classes, caller=id2label)
        return cls(model, preprocessor, cols, names, class_index, checks,
                   missing_feature_method)

    def prepare_inputs(self, df: pd.DataFrame):
        """Reindex *df* to the trained columns by name.

        Extra columns are dropped. A whole missing column raises (default) or is
        filled with NA (``missing_feature_method="fill_na"``). NA *values* in a
        present column are normal here and pass through without a warning.
        """
        X = reindex_pretrained_features(df, self.feature_cols,
                                        self.missing_feature_method, self.checks)
        if self.preprocessor is not None:
            return self.preprocessor.transform(X)
        return X.to_numpy()

    def predict_proba(self, inputs) -> np.ndarray:
        raw = np.asarray(self.model.predict_proba(inputs), dtype=float)
        proba = np.zeros((raw.shape[0], len(self.id2label)))
        proba[:, self.class_index] = raw
        return proba

    def check_artifact(self) -> None:
        if len(self.class_index) < len(self.id2label):
            self.checks.warn(
                f"The model saw {len(self.class_index)} of {len(self.id2label)} "
                "classes in training; the others get probability 0."
            )


def _align_pretrained_feature_cols(model, preprocessor, artifact_dir: Path,
                                   checks: PretrainedChecks,
                                   caller: list[str] | None) -> list[str]:
    """Find the raw input column names the model was trained on.

    Order: the fitted preprocessor's input names; the estimator's own
    (``feature_names_in_``, CatBoost ``feature_names_``); ``feature_names.json``
    when no preprocessor transformed the columns; finally the caller's
    ``feature_cols``. Positional placeholders count as missing.
    """
    candidates = []
    if preprocessor is not None:
        candidates.append(("preprocessor", getattr(preprocessor, "feature_names_in_", None)))
    else:
        candidates.append(("native", getattr(model, "feature_names_in_", None)))
        candidates.append(("native", getattr(model, "feature_names_", None)))
        sidecar = artifact_dir / "feature_names.json"
        if sidecar.is_file():
            candidates.append(("sidecar", json.loads(sidecar.read_text())))

    for source, cols in candidates:
        if cols is not None and not _is_placeholder_feature_cols(list(cols)):
            if caller is not None and list(caller) != list(cols):
                checks.warn(f"feature_cols was ignored: the artifact records its "
                            f"own column names ({source}).")
            checks.feature_cols_source = source
            return [str(c) for c in cols]

    if caller is None:
        raise ValueError(
            "This model does not record its feature column names, so the input "
            "cannot be aligned by name (aligning by position gives confident "
            "nonsense). Pass feature_cols=[...] in the order the model was "
            "trained on."
        )
    checks.feature_cols_source = "caller"
    return list(caller)
