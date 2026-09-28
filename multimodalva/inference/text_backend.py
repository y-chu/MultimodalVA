"""Text (and data-fusion) backend for ``predict_from_pretrained()``.

Serves any Hugging Face sequence classifier: a MultimodalVA text or data-fusion
``final/`` directory, or a model trained elsewhere. The inference loop itself is
``text.predict.predict_text`` in its in-memory mode — not a second copy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .backends import _read_pretrained_id2label
from .checks import PretrainedChecks

logger = logging.getLogger(__name__)


class TextPretrainedBackend:
    """Sequence classifier loaded from a directory holding ``config.json``."""

    def __init__(self, model, tokenizer, id2label, artifact_dir, checks,
                 text_col, max_length, batch_size, use_fast):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.artifact_dir = artifact_dir
        self.checks = checks
        self.text_col = text_col
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fast = use_fast

    @classmethod
    def from_pretrained(cls, artifact_dir: Path, checks: PretrainedChecks, *,
                        text_col: str | None = None, batch_size: int = 32,
                        id2label: dict | None = None, max_length: int | None = None,
                        use_fast: bool | None = None, **model_kwargs):
        """Load model and tokenizer; ``max_length`` / ``use_fast`` default to the
        values the run recorded in ``training_metadata.json`` (else 512 / True)."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if text_col is None:
            raise ValueError("A text model needs text_col= naming the text column.")

        meta_path = artifact_dir / "training_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        if max_length is None:
            max_length = meta.get("max_length", 512)
        if use_fast is None:
            use_fast = meta.get("use_fast", True)

        tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir), use_fast=use_fast)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(artifact_dir), **model_kwargs
        )
        id2label = _read_pretrained_id2label(
            model.config.id2label, artifact_dir, checks,
            n_classes=model.config.num_labels, caller=id2label,
        )
        return cls(model, tokenizer, id2label, artifact_dir, checks,
                   text_col, max_length, batch_size, use_fast)

    def prepare_inputs(self, df: pd.DataFrame):
        """Tokenise ``text_col``. Missing or empty text is scored as ``""``
        (never dropped, so rows stay aligned with ids and labels) and counted."""
        from ..text.dataset import ClassificationDataset

        if self.text_col not in df.columns:
            raise ValueError(
                f"text_col {self.text_col!r} not found in the input. "
                f"Available columns: {list(df.columns)[:20]}"
            )
        texts = df[self.text_col]
        empty = texts.isna() | (texts.astype(str).str.strip() == "")
        if empty.any():
            self.checks.empty_text_rows = int(empty.sum())
            self.checks.warn(
                f"{int(empty.sum())} row(s) have missing or empty "
                f"{self.text_col!r}; they are scored as empty text."
            )
        texts = texts.where(~empty, "").astype(str).tolist()
        # Labels are unused at inference; zeros keep the dataset's shape.
        return ClassificationDataset(texts, [0] * len(texts), self.tokenizer,
                                     max_length=self.max_length)

    def predict_proba(self, inputs) -> np.ndarray:
        from ..text.predict import predict_text

        result = predict_text(
            None, inputs, batch_size=self.batch_size, use_fast=self.use_fast,
            model=self.model, tokenizer=self.tokenizer, id2label=self.id2label,
        )
        return result.full[[f"prob_{i}" for i in sorted(self.id2label)]].to_numpy()

    def check_artifact(self) -> None:
        if not (self.artifact_dir / "tokenizer_config.json").is_file():
            self.checks.warn("No tokenizer_config.json beside the model; the "
                             "tokenizer was loaded from its defaults.")
