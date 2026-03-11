"""
Shared output types used across text, tabular, and ensemble pipelines.
"""

from typing import NamedTuple

import pandas as pd


class PredictionResult(NamedTuple):
    """Container for prediction outputs and the class ID→label mapping.

    All three DataFrames share the same row order (one row per test sample)
    and can be merged on their index if needed.

    Attributes:
        top1: Top predicted class and its probability.
              Columns: true_label, predicted_label, predicted_prob.
        full: Full probability distribution over all classes.
              Columns: true_label, prob_0, prob_1, ... (integer class IDs).
              Rename to label strings:
                  result.full.rename(columns={f"prob_{i}": f"prob_{v}"
                                              for i, v in result.id2label.items()})
        topk: Top-K predicted classes and their probabilities.
              Columns: true_label, top1_label, top1_prob, ..., topK_label, topK_prob.
        id2label: Dict mapping integer class ID → label string.
    """

    top1: pd.DataFrame
    full: pd.DataFrame
    topk: pd.DataFrame
    id2label: dict
