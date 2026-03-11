"""
Step 4: Predict cause of death labels and probability distributions.

Input:  (A) output_dir from train() — loads model, tokenizer, id2label from disk
        (B) model + tokenizer + id2label in memory — skips disk I/O after train()
        test_dataset from prepare_dataset()
Output: PredictionResult(top1, full, topk) — three DataFrames described below
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ..utils.types import PredictionResult
from .train import _get_dataset_labels, get_device

logger = logging.getLogger(__name__)


def predict(
    output_dir: str | Path | None,
    test_dataset: Dataset,
    batch_size: int = 32,
    top_k: int = 3,
    save_dir: str | Path | None = None,
    save_prefix: str = "predictions",
    *, # everything after here is keyword-only, e.g. MUST be passed as model=...
    model: AutoModelForSequenceClassification | None = None,
    tokenizer: AutoTokenizer | None = None,
    id2label: dict | None = None,
) -> PredictionResult:
    """Load a saved model and predict labels and probabilities on test data.

    Supports two modes:
      - Disk mode (default): loads model, tokenizer, and id2label from output_dir.
      - In-memory mode: pass model, tokenizer, and id2label directly to skip disk I/O.
        Useful for immediate prediction after train() without reloading from disk.
        Pass output_dir=None when using in-memory mode.

    Args:
        output_dir: Path to directory saved by train() — must contain the model,
                    tokenizer, and id2label.json. May be None when model, tokenizer,
                    and id2label are all provided directly.
        test_dataset: Tokenized ClassificationDataset (or Subset) from prepare_dataset().
        batch_size: Number of samples per inference batch. Default 32.
        top_k: Number of top classes to include in the topk output. Default 3.
               Capped at the total number of classes if top_k exceeds it.
        save_dir: Directory to save CSV outputs. If None, no files are written.
                  Created automatically if it does not exist.
        save_prefix: Filename stem for saved CSVs. Default "predictions".
                     Three files are written:
                         <save_dir>/<save_prefix>_top1.csv
                         <save_dir>/<save_prefix>_full.csv
                         <save_dir>/<save_prefix>_topk.csv
        model: (keyword-only) Fine-tuned model in memory. When all three of model,
               tokenizer, and id2label are provided, disk loading is skipped entirely.
        tokenizer: (keyword-only) Tokenizer in memory.
        id2label: (keyword-only) Dict mapping integer or string IDs to class label strings
                  (as returned by train() or loaded from id2label.json).

    Returns:
        PredictionResult with three DataFrames (one row per test sample each):
            top1: true_label, predicted_label, predicted_prob
            full: true_label, prob_0, prob_1, ... (integer class IDs)
            topk: true_label, top1_label, top1_prob, ..., topK_label, topK_prob

    Raises:
        ValueError: If neither output_dir nor all three of (model, tokenizer, id2label)
                    are provided.
    """
    in_memory = model is not None and tokenizer is not None and id2label is not None

    if not in_memory:
        if output_dir is None:
            raise ValueError(
                "Either output_dir or all three of (model, tokenizer, id2label) "
                "must be provided."
            )
        output_dir = Path(output_dir)

        with open(output_dir / "id2label.json") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}

        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(output_dir))

    # Normalise id2label keys to int in both paths
    # (in-memory keys are already int; disk keys come in as str from JSON)
    id2label = {int(k): v for k, v in id2label.items()}

    # Unwrap DataParallel / DistributedDataParallel if present.
    # trainer.model from a multi-GPU train() call may be wrapped; passing a wrapped
    # model to a new Trainer causes device and forward-pass errors.
    if hasattr(model, "module"):
        model = model.module

    # Move model to best available device (CUDA > MPS > CPU).
    # Covers both disk-loaded models and in-memory models from a CPU context.
    model.to(get_device())
    model.eval()

    # TrainingArguments requires an output_dir string.
    # Use a temp dir when predicting in-memory (the dir is never written to).
    args_output_dir = str(output_dir) if output_dir is not None else tempfile.mkdtemp()

    # Minimal TrainingArguments — only output_dir and batch size are meaningful here
    inference_args = TrainingArguments(
        output_dir=args_output_dir,
        per_device_eval_batch_size=batch_size,
        report_to="none",
        disable_tqdm=False,
    )
    # DataCollatorWithPadding matches the collator used during training,
    # padding each batch to its own longest sequence.
    trainer = Trainer(
        model=model,
        args=inference_args,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    test_predictions = trainer.predict(test_dataset)
    logits = test_predictions.predictions  # shape: (n_samples, n_classes)

    # Softmax probabilities (float32 cast avoids nan/inf on MPS with float16 logits)
    # Keep as tensor so torch.topk can be used directly for Output 3.
    softmax = torch.nn.Softmax(dim=-1)
    probs_tensor = softmax(torch.tensor(logits, dtype=torch.float32))  # (n_samples, n_classes)
    probs = probs_tensor.numpy()

    # True labels from the dataset (works for both ClassificationDataset and Subset)
    true_int_labels = _get_dataset_labels(test_dataset)
    true_labels = [id2label[i] for i in true_int_labels]

    # --- Output 1: top1 — predicted class + probability ---
    top1_probs_t, top1_idx_t = torch.topk(probs_tensor, k=1, dim=1)
    top1_idx = top1_idx_t.squeeze(1).numpy()   # shape: (n_samples,)
    top1_probs = top1_probs_t.squeeze(1).numpy()
    top1_df = pd.DataFrame(
        {
            "true_label": true_labels,
            "predicted_label": [id2label[i] for i in top1_idx],
            "predicted_prob": top1_probs,
        }
    )

    # --- Output 2: full — probability for every class (columns use integer IDs) ---
    # Integer IDs keep column names short regardless of label length.
    
    prob_cols = {f"prob_{i}": probs[:, i] for i in sorted(id2label)}
    full_df = pd.DataFrame({"true_label": true_labels, **prob_cols})

    # --- Output 3: topk — top-K classes + probabilities ---
    k = min(top_k, len(id2label))
    topk_probs_t, topk_idx_t = torch.topk(probs_tensor, k=k, dim=1)
    topk_probs = topk_probs_t.numpy()   # shape: (n_samples, k)
    topk_idx = topk_idx_t.numpy()       # shape: (n_samples, k)
    topk_data: dict[str, list] = {"true_label": true_labels}
    for rank in range(k):
        topk_data[f"top{rank + 1}_label"] = [id2label[i] for i in topk_idx[:, rank]]
        topk_data[f"top{rank + 1}_prob"] = topk_probs[:, rank]
    topk_df = pd.DataFrame(topk_data)

    logger.info(
        "Prediction complete: %d samples, %d classes, top-%d output.",
        len(top1_df), len(id2label), k,
    )

    result = PredictionResult(top1=top1_df, full=full_df, topk=topk_df, id2label=id2label)

    # --- Save to CSV if save_dir is specified ---
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

# full_df Rename to label strings via: result.full.rename(columns={f"prob_{i}": f"prob_{v}"
#                                                           for i, v in result.id2label.items()})