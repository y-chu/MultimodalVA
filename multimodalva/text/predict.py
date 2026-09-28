"""
Step 4: Predict cause of death labels and probability distributions.

Input:  (A) output_dir from train_text() — loads model, tokenizer, id2label from disk
        (B) model + tokenizer + id2label in memory — skips disk I/O after train_text()
        test_dataset from prepare_text_dataset()
Output: PredictionResult(top1, full, topk) — three DataFrames described below
"""

from __future__ import annotations

import json
from contextlib import nullcontext
import logging
import tempfile
import time
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

# from ..utils.types import PredictionResult
# from .train import _get_dataset_labels

from multimodalva.utils.types import PredictionResult
from multimodalva.utils.predictions import assemble_predictions, save_predictions
from multimodalva.utils.runtime import RuntimeTracker, get_device
from multimodalva.text.train import _auto_num_workers, _get_dataset_labels, _int_env


logger = logging.getLogger(__name__)


def _get_dataset_ids(dataset) -> list | None:
    """Row identifiers carried by a dataset, if it has any."""
    if getattr(dataset, "ids", None) is not None:
        return list(dataset.ids)
    inner = getattr(dataset, "dataset", None)
    indices = getattr(dataset, "indices", None)
    if inner is not None and indices is not None:
        inner_ids = getattr(inner, "ids", None)
        if inner_ids is not None:
            return [inner_ids[int(i)] for i in indices]
    return None


def predict_text(
    output_dir: str | Path | None,
    test_dataset: Dataset,
    batch_size: int = 32,
    top_k: int = 3,
    save_dir: str | Path | None = None,
    save_prefix: str = "predictions",
    use_fast: bool = True,
    *, # everything after here is keyword-only, e.g. MUST be passed as model=...
    model: AutoModelForSequenceClassification | None = None,
    tokenizer: AutoTokenizer | None = None,
    id2label: dict | None = None,
    ids: "list | pd.Series | None" = None,
) -> PredictionResult:
    """Load a saved model and predict labels and probabilities on test data.

    Supports two modes:
      - Disk mode (default): loads model, tokenizer, and id2label from output_dir.
      - In-memory mode: pass model, tokenizer, and id2label directly to skip disk I/O.
        Useful for immediate prediction after train_text() without reloading from disk.
        Pass output_dir=None when using in-memory mode.

    Args:
        output_dir: Path to directory saved by train_text() — must contain the model,
                    tokenizer, and id2label.json. May be None when model, tokenizer,
                    and id2label are all provided directly.
        test_dataset: Tokenized ClassificationDataset (or Subset) from prepare_text_dataset().
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
        use_fast: Use the HuggingFace fast (Rust) tokenizer. Default True.
                  Must match the value used in prepare_text_dataset() / train_text().
        model: (keyword-only) Fine-tuned model in memory. When all three of model,
               tokenizer, and id2label are provided, disk loading is skipped entirely.
        tokenizer: (keyword-only) Tokenizer in memory.
        id2label: (keyword-only) Dict mapping integer or string IDs to class label strings
                  (as returned by train_text() or loaded from id2label.json).

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

        tokenizer = AutoTokenizer.from_pretrained(str(output_dir), use_fast=use_fast)
        model = AutoModelForSequenceClassification.from_pretrained(str(output_dir))

    # Normalise id2label keys to int in both paths
    # (in-memory keys are already int; disk keys come in as str from JSON)
    id2label = {int(k): v for k, v in id2label.items()}

    # Unwrap DataParallel / DistributedDataParallel if present.
    # trainer.model from a multi-GPU train_text() call may be wrapped; passing a wrapped
    # model to a new Trainer causes device and forward-pass errors.
    if hasattr(model, "module"):
        model = model.module

    # Move model to best available device (CUDA > MPS > CPU).
    # Covers both disk-loaded models and in-memory models from a CPU context.
    _device = get_device()
    _is_cuda = _device.type == "cuda"
    _is_mps  = _device.type == "mps"
    _num_workers = _auto_num_workers()
    # On MPS (Apple Silicon), multi-process DataLoader workers add macOS spawn
    # overhead without the transfer benefit that justifies multiple workers on CUDA
    # (unified memory means batches are already accessible without a copy).
    # Use single-process loading unless the user explicitly requested workers.
    if _is_mps and _num_workers > 0 and _int_env("MULTIMODALVA_DATALOADER_WORKERS") is None:
        logger.info(
            "MPS device: overriding dataloader_num_workers %d → 0 "
            "(unified memory + macOS spawn overhead). "
            "Set MULTIMODALVA_DATALOADER_WORKERS=N to override.",
            _num_workers,
        )
        _num_workers = 0
    model.to(_device)
    model.eval()

    # TrainingArguments requires an output_dir string.
    # Use a TemporaryDirectory when predicting in-memory (the dir is never written to)
    # so it is cleaned up automatically instead of leaking across calls.
    _tmp_dir: tempfile.TemporaryDirectory | None = None
    row_ids = list(ids) if ids is not None else _get_dataset_ids(test_dataset)

    if output_dir is not None:
        args_output_dir = str(output_dir)
    else:
        _tmp_dir = tempfile.TemporaryDirectory()
        args_output_dir = _tmp_dir.name

    runtime_root = None
    if save_dir is not None:
        runtime_root = Path(save_dir)
    elif output_dir is not None:
        runtime_root = Path(output_dir)
    runtime_tracker = (
        RuntimeTracker(
            runtime_root,
            report_name="predict_runtime.json",
            metadata={
                "pipeline": "text_predict",
                "output_dir": str(output_dir) if output_dir is not None else None,
                "save_dir": str(save_dir) if save_dir is not None else None,
                "batch_size": batch_size,
                "top_k_requested": top_k,
                "use_fast": use_fast,
                "in_memory": in_memory,
            },
            logger_=logger,
        )
        if runtime_root is not None
        else None
    )

    try:
        # Minimal TrainingArguments — only output_dir and batch size are meaningful here
        inference_args = TrainingArguments(
            output_dir=args_output_dir,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=_num_workers,
            dataloader_pin_memory=_is_cuda,
            dataloader_persistent_workers=bool(_num_workers > 0),
            report_to="none",
            disable_tqdm=False,
        )
        # DataCollatorWithPadding matches the collator used during training,
        # padding each batch to its own longest sequence.
        trainer = Trainer(
            model=model,
            args=inference_args,
            data_collator=DataCollatorWithPadding(
                tokenizer,
                pad_to_multiple_of=(8 if (_is_cuda or _is_mps) else None),
            ),
        )

        if runtime_tracker is not None:
            with runtime_tracker.stage(
                "predict_loop",
                details={
                    "examples": len(test_dataset),
                    "num_workers": _num_workers,
                    "device": str(_device),
                },
                monitor_gpu=True,
                device=_device,
            ):
                test_predictions = trainer.predict(test_dataset)
        else:
            started = time.perf_counter()
            test_predictions = trainer.predict(test_dataset)
            logger.info(
                "predict_text(): inference completed in %.2fs on %d examples.",
                time.perf_counter() - started,
                len(test_dataset),
            )
    finally:
        if _tmp_dir is not None:
            _tmp_dir.cleanup()

    # Formatting the outputs is identical work whether or not it is being timed,
    # so the tracker wraps it instead of the block being written out twice.
    k = min(top_k, len(id2label))
    format_stage = (
        runtime_tracker.stage("format_outputs", details={"examples": len(test_dataset)})
        if runtime_tracker is not None
        else nullcontext()
    )
    with format_stage:
        # Softmax probabilities (float32 cast avoids nan/inf on MPS with float16
        # logits). torch.as_tensor() shares memory with the numpy array when the
        # dtype matches (HuggingFace Trainer returns float32) — avoids a copy.
        logits = test_predictions.predictions  # shape: (n_samples, n_classes)
        probs = torch.nn.functional.softmax(
            torch.as_tensor(logits).float(), dim=-1
        ).numpy()

        # True labels from the dataset (works for ClassificationDataset and Subset)
        true_labels = [id2label[i] for i in _get_dataset_labels(test_dataset)]

        result = assemble_predictions(
            probs, id2label, true_labels=true_labels, ids=row_ids, top_k=k
        )

    logger.info(
        "Prediction complete: %d samples, %d classes, top-%d output.",
        len(result.top1), len(id2label), k,
    )

    # --- Save to CSV if save_dir is specified ---
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_stage = (
            runtime_tracker.stage(
                "save_predictions",
                details={"save_prefix": save_prefix, "save_dir": str(save_dir)},
            )
            if runtime_tracker is not None
            else nullcontext()
        )
        with save_stage:
            save_predictions(result, save_dir, save_prefix)

    if runtime_tracker is not None:
        runtime_tracker.update_metadata(
            save_dir=str(save_dir) if save_dir is not None else None,
            prediction_rows=len(result.top1),
            top_k_actual=k,
            runtime_report=str(runtime_tracker.report_path),
            runtime_stage_csv=str(runtime_tracker.stage_csv_path),
        )

    return result

# full_df Rename to label strings via: result.full.rename(columns={f"prob_{i}": f"prob_{v}"
#                                                           for i, v in result.id2label.items()})
