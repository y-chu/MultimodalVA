"""
Step 4 (tabular pipeline): Predict class labels and probability distributions.

Input:  (A) output_dir from train() — loads model bundle from model.joblib
        (B) model + id2label in memory — skips disk I/O after train()
        X_test, y_test from prepare_dataset()
Output: PredictionResult(top1, full, topk, id2label) — identical format to text predict()
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
from ..utils.numpy_compat import load_joblib_compat

logger = logging.getLogger(__name__)


def predict(
    output_dir: str | Path | None,
    X_test: np.ndarray,
    y_test: np.ndarray,
    top_k: int = 3,
    save_dir: str | Path | None = None,
    save_prefix: str = "predictions",
    *,
    model: Any | None = None,
    id2label: dict | None = None,
) -> PredictionResult:
    """Load a saved model and predict labels and probabilities on test data.

    Supports two modes:
      - Disk mode (default): loads model bundle and id2label from output_dir.
      - In-memory mode: pass model and id2label directly; set output_dir=None.

    X_test must be a preprocessed numpy array from prepare_dataset() — the
    preprocessor is NOT applied inside this function.

    Args:
        output_dir:   Path saved by train(). Contains model.joblib and id2label.json.
                      Pass None when using in-memory mode.
        X_test:       Preprocessed feature matrix from prepare_dataset().
        y_test:       Integer label array from prepare_dataset().
        top_k:        Number of top classes in topk output. Default 3.
        save_dir:     Directory to write CSV outputs. None = no files written.
        save_prefix:  Filename stem for saved CSVs. Default "predictions".
                      Three files: <save_dir>/<save_prefix>_top1/full/topk.csv
        model:        (keyword-only) Fitted estimator. Skip disk load when provided
                      together with id2label.
        id2label:     (keyword-only) Dict mapping integer IDs → label strings.

    Returns:
        PredictionResult(top1, full, topk, id2label) — same format as text predict().
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

    # --- Output 1: top1 ---
    top1_idx = np.argmax(probs, axis=1)
    top1_probs = probs[np.arange(len(probs)), top1_idx]
    top1_df = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": [id2label[int(i)] for i in top1_idx],
        "predicted_prob": top1_probs,
    })

    # --- Output 2: full --- (integer IDs as column names — matches text pipeline)
    prob_cols = {f"prob_{i}": probs[:, i] for i in sorted(id2label)}
    full_df = pd.DataFrame({"true_label": true_labels, **prob_cols})

    # --- Output 3: topk ---
    topk_idx = np.argsort(probs, axis=1)[:, ::-1][:, :k]  # descending
    topk_probs = probs[np.arange(len(probs))[:, None], topk_idx]
    topk_data: dict[str, list] = {"true_label": true_labels}
    for rank in range(k):
        topk_data[f"top{rank + 1}_label"] = [id2label[int(i)] for i in topk_idx[:, rank]]
        topk_data[f"top{rank + 1}_prob"] = topk_probs[:, rank].tolist()
    topk_df = pd.DataFrame(topk_data)

    logger.info(
        "Prediction complete: %d samples, %d classes, top-%d output.",
        len(top1_df), n_classes, k,
    )

    result = PredictionResult(top1=top1_df, full=full_df, topk=topk_df, id2label=id2label)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        result.top1.to_csv(save_dir / f"{save_prefix}_top1.csv", index=False)
        result.full.to_csv(save_dir / f"{save_prefix}_full.csv", index=False)
        result.topk.to_csv(save_dir / f"{save_prefix}_topk.csv", index=False)
        logger.info(
            "Saved CSVs to %s: %s_top1.csv, %s_full.csv, %s_topk.csv",
            save_dir, save_prefix, save_prefix, save_prefix,
        )

    return result
