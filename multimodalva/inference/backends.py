"""Backend protocol and the lazy registry for ``predict_from_pretrained()``.

Every backend implements the same four methods with the same input and output
types, so the entry point never branches on the task:

    backend = Backend.from_pretrained(artifact_dir, checks, **backend_kwargs)
    inputs  = backend.prepare_inputs(df)          # model-ready inputs, one per row
    proba   = backend.predict_proba(inputs)       # (n_rows, n_classes) ndarray
    backend.check_artifact()                      # fills the shared report

``predict_proba`` follows scikit-learn: column *j* is the class whose id is the
*j*-th smallest key of ``backend.id2label``. Backends are imported only when
needed, so ``import multimodalva`` stays free of torch and transformers.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .checks import PretrainedChecks, _is_placeholder_id2label

# task → "module:class". data_fusion artifacts are ordinary sequence
# classifiers over already-converted long text, so they share the text backend.
PRETRAINED_BACKENDS: dict[str, str] = {
    "text": "multimodalva.inference.text_backend:TextPretrainedBackend",
    "data_fusion": "multimodalva.inference.text_backend:TextPretrainedBackend",
    "tabular": "multimodalva.inference.tabular_backend:TabularPretrainedBackend",
    "feature_fusion":
        "multimodalva.inference.feature_fusion_backend:FeatureFusionPretrainedBackend",
}


# Values accepted by predict_from_pretrained(missing_feature_method=...).
PRETRAINED_MISSING_FEATURE_METHODS: tuple[str, ...] = ("error", "fill_na")


class PretrainedBackend(Protocol):
    """What every backend provides. See the module docstring."""

    id2label: dict

    @classmethod
    def from_pretrained(cls, artifact_dir: Path, checks: PretrainedChecks,
                        **backend_kwargs) -> "PretrainedBackend": ...

    def prepare_inputs(self, df: pd.DataFrame): ...

    def predict_proba(self, inputs) -> np.ndarray: ...

    def check_artifact(self) -> None: ...


def load_pretrained_backend(task: str) -> type:
    """Import and return the backend class registered for *task*."""
    try:
        target = PRETRAINED_BACKENDS[task]
    except KeyError:
        raise ValueError(
            f"predict_from_pretrained() does not support task {task!r} yet. "
            f"Supported: {sorted(PRETRAINED_BACKENDS)}. Feature fusion and the "
            "ensembles (voting, stacking) are planned; ensembles will get their "
            "own entry point."
        ) from None
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def reindex_pretrained_features(df: pd.DataFrame, feature_cols: list,
                                missing_feature_method: str,
                                checks: PretrainedChecks) -> pd.DataFrame:
    """Line the input up with the columns the model was trained on, by name.

    Extra columns are dropped. A whole missing column raises (default) or is
    filled with NA (``missing_feature_method="fill_na"``). NA *values* inside a
    column that is present are normal in this data and never warn — that is a
    different situation and is reported differently.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    checks.dropped_input_cols = [c for c in df.columns if c not in feature_cols]
    if missing:
        if missing_feature_method == "error":
            raise ValueError(
                f"{len(missing)} feature column(s) the model was trained on are "
                f"missing from the input: {missing}. Filling them with NA would "
                "give confident predictions from a pattern the model never saw. "
                "Pass missing_feature_method='fill_na' to do it anyway."
            )
        checks.missing_feature_cols = missing
        checks.warn(
            f"Filled {len(missing)} missing feature column(s) with NA: {missing}."
        )
    out = df.reindex(columns=list(feature_cols))
    if missing:
        # reindex fills a new column as float NaN, which categorical encoders
        # fitted on strings reject; object dtype holds NA for either kind.
        out[missing] = out[missing].astype(object)
    return out


def _read_pretrained_id2label(
    native: dict | None,
    artifact_dir: Path,
    checks: PretrainedChecks,
    n_classes: int,
    caller: dict | None = None,
) -> dict:
    """Resolve class names: the artifact's own, then the sidecar, then the caller.

    Placeholder names (``LABEL_0``, ...) count as missing. When nothing usable is
    found, returns ``class_<i>`` names and warns: an honest integer id beats a
    wrong cause name.
    """
    candidates = [("native", native)]
    sidecar = artifact_dir / "id2label.json"
    if sidecar.is_file():
        with open(sidecar) as fh:
            candidates.append(("sidecar", json.load(fh)))
    candidates.append(("caller", caller))

    for source, mapping in candidates:
        if mapping and not _is_placeholder_id2label(mapping):
            mapping = {int(k): str(v) for k, v in mapping.items()}
            if len(mapping) != n_classes:
                raise ValueError(
                    f"The {source} id2label has {len(mapping)} classes but the "
                    f"model outputs {n_classes}. They must come from the same run."
                )
            checks.id2label_source = source
            return mapping

    checks.id2label_source = "missing"
    checks.warn(
        "No class names found for this model (id2label is missing or only holds "
        "placeholders such as LABEL_0). Predictions use integer class ids "
        "(class_0, class_1, ...). Pass id2label={...} to name them."
    )
    return {i: f"class_{i}" for i in range(n_classes)}
