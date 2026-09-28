"""What ``predict_from_pretrained()`` found out about the artifact and the input.

The report travels on the result as ``result.checks`` so a caller can assert on
it. It is deliberately not called "validation": that word is reserved for
validation-set scores (``validation.json``, ``validation_leaderboard``).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

_LABEL_DISCLAIMER = (
    "Predictions are reported in the label vocabulary of the artifact you "
    "loaded. Whether a cause label from another site, instrument or coding "
    "round means the same thing as yours is a judgement only you can make; "
    "check the vocabulary against your own definitions before using or "
    "comparing the output."
)
_label_disclaimer_shown = False

_PLACEHOLDER_LABEL = re.compile(r"LABEL_\d+")


@dataclass
class PretrainedChecks:
    """Structured report of one ``predict_from_pretrained()`` call.

    Attributes:
        source:                 What the caller passed (Hub id, URL or path).
        artifact_dir:           Local directory the model was loaded from.
        task:                   Detected (or given) task, e.g. ``"text"``.
        task_source:            Where the task came from: ``"caller"``,
                                ``"training_metadata"`` or ``"structure"``.
        multimodalva_version:   Version that trained the artifact; ``None`` for
                                a foreign or older artifact.
        id2label_source:        ``"native"``, ``"sidecar"``, ``"caller"`` or
                                ``"missing"`` (integer class ids returned).
        feature_cols_source:    Tabular only — ``"preprocessor"``, ``"sidecar"``,
                                ``"native"`` or ``"caller"``.
        missing_feature_cols:   Expected columns absent from the input. Empty
                                unless ``missing_feature_method="fill_na"``.
        dropped_input_cols:     Input columns the model does not use.
        empty_text_rows:        Text only — rows whose text was missing/empty
                                and was scored as an empty string.
        base_models:            Ensembles only — one entry per base model:
                                ``{"dir", "task", "id2label_source"}``.
        combiner:           Ensembles only — how the base models were combined.
        labelled:               Whether a label column was supplied.
        n_rows:                 Rows scored.
        warnings:               Every warning raised during the call, in order.
    """

    source: str
    artifact_dir: str
    task: str
    task_source: str
    multimodalva_version: str | None = None
    id2label_source: str = "missing"
    feature_cols_source: str | None = None
    missing_feature_cols: list = field(default_factory=list)
    dropped_input_cols: list = field(default_factory=list)
    empty_text_rows: int = 0
    base_models: list = field(default_factory=list)
    combiner: str | None = None
    labelled: bool = False
    n_rows: int = 0
    warnings: list = field(default_factory=list)

    def warn(self, message: str) -> None:
        """Record *message* and raise it as a ``UserWarning``."""
        self.warnings.append(message)
        warnings.warn(message, UserWarning, stacklevel=3)


def _is_placeholder_id2label(id2label: dict | None) -> bool:
    """True when every name is Hugging Face's untouched ``LABEL_<i>`` default."""
    if not id2label:
        return True
    return all(_PLACEHOLDER_LABEL.fullmatch(str(v)) for v in id2label.values())


def _is_placeholder_feature_cols(cols) -> bool:
    """True when the names are just positions (``"0", "1", ...``).

    CatBoost fitted on a bare array records ``feature_names_`` this way; the
    names are present and meaningless.
    """
    if cols is None or len(cols) == 0:
        return True
    return [str(c) for c in cols] == [str(i) for i in range(len(cols))]


def show_label_disclaimer() -> None:
    """Warn once per process that label meaning is the caller's judgement."""
    global _label_disclaimer_shown
    if not _label_disclaimer_shown:
        _label_disclaimer_shown = True
        warnings.warn(_LABEL_DISCLAIMER, UserWarning, stacklevel=3)
