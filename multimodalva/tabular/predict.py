"""
Step 4 (tabular pipeline): Predict class labels and probability distributions.

Input:  (A) output_dir from train_tabular() — loads model bundle from model.joblib
        (B) model + id2label in memory — skips disk I/O after train_tabular()
        X_test, y_test from prepare_tabular_dataset()
Output: PredictionResult(top1, full, topk, id2label) — identical format to predict_text()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..utils.types import PredictionResult
from ..utils.predictions import assemble_predictions, save_predictions
from ..utils.numpy_compat import load_joblib_compat

logger = logging.getLogger(__name__)


def predict_tabular(
    output_dir: str | Path | None,
    X_test: np.ndarray,
    y_test: np.ndarray,
    top_k: int = 3,
    save_dir: str | Path | None = None,
    save_prefix: str = "predictions",
    *,
    model: Any | None = None,
    id2label: dict | None = None,
    ids: "list | pd.Series | None" = None,
) -> PredictionResult:
    """Load a saved model and predict labels and probabilities on test data.

    Supports two modes:
      - Disk mode (default): loads model bundle and id2label from output_dir.
      - In-memory mode: pass model and id2label directly; set output_dir=None.

    X_test must be a preprocessed numpy array from prepare_tabular_dataset() — the
    preprocessor is NOT applied inside this function.

    Args:
        output_dir:   Path saved by train_tabular(). Contains model.joblib and id2label.json.
                      Pass None when using in-memory mode.
        X_test:       Preprocessed feature matrix from prepare_tabular_dataset().
        y_test:       Integer label array from prepare_tabular_dataset().
        top_k:        Number of top classes in topk output. Default 3.
        save_dir:     Directory to write CSV outputs. None = no files written.
        save_prefix:  Filename stem for saved CSVs. Default "predictions".
                      Three files: <save_dir>/<save_prefix>_top1/full/topk.csv
        model:        (keyword-only) Fitted estimator. Skip disk load when provided
                      together with id2label.
        id2label:     (keyword-only) Dict mapping integer IDs → label strings.
        ids:          (keyword-only) Row identifiers for the test rows, added as
                      a leading ``id`` column in all three output tables so
                      predictions can be joined back to the source records.
                      Must line up with the rows in ``X_test`` — note that
                      prepare_tabular_dataset() drops rows with a missing label.

    Returns:
        PredictionResult(top1, full, topk, id2label) — same format as predict_text().
            top1: true_label, predicted_label, predicted_prob
            full: true_label, prob_0, prob_1, ...  (integer class IDs as column names)
            topk: true_label, top1_label, top1_prob, ..., topK_label, topK_prob

    Raises:
        ValueError: If neither output_dir nor both (model, id2label) are provided.
    """
    in_memory = model is not None and id2label is not None

    if not in_memory:
        if output_dir is None:
            raise ValueError(
                "Either output_dir or both (model, id2label) must be provided."
            )
        output_dir = Path(output_dir)
        bundle = load_joblib_compat(output_dir / "model.joblib")
        model = bundle["model"]
        with open(output_dir / "id2label.json") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}

    # Normalise id2label keys to int in both paths
    id2label = {int(k): v for k, v in id2label.items()}

    # predict_proba: shape (n_samples, n_classes)
    probs = model.predict_proba(X_test)
    true_labels = [id2label[int(i)] for i in y_test]
    n_classes = len(id2label)
    k = min(top_k, n_classes)

    if ids is not None:
        ids = list(ids)
        if len(ids) != len(X_test):
            raise ValueError(
                f"Got {len(ids)} ids for {len(X_test)} test rows. Identifiers must "
                "come from the same rows that were scored — prepare_tabular_dataset() "
                "drops rows with a missing label (see valid_label_mask())."
            )

    result = assemble_predictions(
        probs, id2label, true_labels=true_labels, ids=ids, top_k=k
    )

    logger.info(
        "Prediction complete: %d samples, %d classes, top-%d output.",
        len(result.top1), n_classes, k,
    )

    if save_dir is not None:
        save_predictions(result, save_dir, save_prefix)

    return result
