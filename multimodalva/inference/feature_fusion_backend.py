"""Feature-fusion (AutoGluon MultiModalPredictor) backend for ``predict_from_pretrained()``.

The artifact is a feature-fusion run directory: ``automm_model/`` holding the
predictor, plus the run's ``training_metadata.json`` and ``id2label.json``. The
predictor takes the raw DataFrame — text column and tabular columns together —
so the columns it was trained on come from ``training_metadata["feature_cols"]``
(AutoMM does not expose them itself).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .backends import _read_pretrained_id2label, reindex_pretrained_features
from .checks import PretrainedChecks

logger = logging.getLogger(__name__)

_AUTOMM_SUBDIR = "automm_model"


class FeatureFusionPretrainedBackend:
    """AutoMM predictor loaded from a feature-fusion run directory."""

    def __init__(self, predictor, feature_cols, id2label, checks,
                 missing_feature_method):
        self.predictor = predictor
        self.feature_cols = feature_cols
        self.id2label = id2label
        self.checks = checks
        self.missing_feature_method = missing_feature_method

    @classmethod
    def from_pretrained(cls, artifact_dir: Path, checks: PretrainedChecks, *,
                        feature_cols: list[str] | None = None,
                        missing_feature_method: str = "error",
                        id2label: dict | None = None, **load_kwargs):
        try:
            from autogluon.multimodal import MultiModalPredictor  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Feature-fusion models need AutoGluon: "
                "pip install 'multimodalva[feature_fusion]'."
            ) from exc

        model_dir = artifact_dir / _AUTOMM_SUBDIR
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"No {_AUTOMM_SUBDIR}/ in {artifact_dir}; that is where a "
                "feature-fusion run keeps its predictor."
            )
        predictor = MultiModalPredictor.load(str(model_dir), **load_kwargs)

        meta_path = artifact_dir / "training_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        cols = meta.get("feature_cols") or feature_cols
        if not cols:
            raise ValueError(
                "This feature-fusion run does not record the columns it was "
                "trained on (no 'feature_cols' in training_metadata.json), so "
                "the input cannot be aligned by name. Pass feature_cols=[...]."
            )
        checks.feature_cols_source = "sidecar" if meta.get("feature_cols") else "caller"

        names = _read_pretrained_id2label(
            meta.get("id2label"), artifact_dir, checks,
            n_classes=len(meta.get("id2label") or predictor.class_labels),
            caller=id2label,
        )
        return cls(predictor, list(cols), names, checks, missing_feature_method)

    def prepare_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """AutoMM consumes the raw frame; line its columns up by name."""
        return reindex_pretrained_features(df, self.feature_cols,
                                           self.missing_feature_method, self.checks)

    def predict_proba(self, inputs) -> np.ndarray:
        from ..ensemble.feature_fusion import _align_automm_proba  # noqa: PLC0415

        proba_df = self.predictor.predict_proba(inputs)
        return _align_automm_proba(proba_df, self.id2label)

    def check_artifact(self) -> None:
        # class_labels is a NumPy array, so it must be tested for None rather
        # than for truthiness.
        labels = getattr(self.predictor, "class_labels", None)
        trained = set() if labels is None else {str(c) for c in labels}
        unseen = sorted(set(self.id2label.values()) - trained) if trained else []
        if unseen:
            self.checks.warn(
                f"{len(unseen)} class(es) are in the label map but not in the "
                f"predictor's own classes; they get probability 0: {unseen[:10]}"
            )
