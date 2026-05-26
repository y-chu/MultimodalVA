"""
Step 3: Fine-tune a BERT-family model for multiclass cause of death classification.

Input:  train_dataset, label2id, id2label  from prepare_dataset()
        hyperparams dict
Output: (model, tokenizer, metadata) — model and tokenizer ready for immediate
        prediction; all artifacts also saved to output_dir for later reuse
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split as _sklearn_val_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from multimodalva.utils.runtime import RuntimeTracker

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)


def _set_lightweight_seeds(seed: int) -> None:
    """Set lightweight RNG seeds for improved run-to-run reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _int_env(name: str, default: int | None = None) -> int | None:
    """Return an integer environment variable, or default on parse failure."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _auto_num_workers() -> int:
    """Infer a sensible DataLoader worker count for the current environment.

    Priority order:
      1) ``MULTIMODALVA_DATALOADER_WORKERS`` (explicit override)
      2) Derive from CPU allocation and distributed world size

    CPU allocation sources:
      - ``SLURM_CPUS_PER_TASK`` (preferred on OSC/SLURM)
      - ``SLURM_CPUS_ON_NODE`` (fallback when per-task value is absent)
      - ``os.cpu_count()`` (local/non-SLURM)

    The derived worker count is per process (not per node), so DDP runs divide
    the total allocation by ``WORLD_SIZE`` to avoid CPU over-subscription.
    """
    explicit = _int_env("MULTIMODALVA_DATALOADER_WORKERS")
    if explicit is not None:
        return max(0, explicit)

    total_cpus = (
        _int_env("SLURM_CPUS_PER_TASK")
        or _int_env("SLURM_CPUS_ON_NODE")
        or (os.cpu_count() or 4)
    )
    world_size = max(1, _int_env("WORLD_SIZE", 1) or 1)
    cpus_per_proc = max(1, total_cpus // world_size)

    reserve = 2 if cpus_per_proc > 4 else 1
    derived = max(1, cpus_per_proc - reserve)
    max_workers = max(1, _int_env("MULTIMODALVA_MAX_DATALOADER_WORKERS", 24) or 24)
    return min(max_workers, derived)


# Default training hyperparameters (keys match TrainingArguments where applicable)
DEFAULT_HYPERPARAMS: dict = {
    "learning_rate": 2e-5,             # AdamW LR; BERT fine-tuning: 1e-5–5e-5; larger models → lower end
    "batch_size": 16,                  # per-device; 8–32 typical; reduce if GPU OOM
    "epochs": 3,                       # 3–10; pair with early_stopping_patience for small datasets
    "weight_decay": 0.01,              # L2 regularisation on non-bias params; 0.01–0.1
    "warmup_ratio": 0.1,               # fraction of steps for LR warm-up; 0.06–0.1 typical
    "gradient_accumulation_steps": 1,  # multiply effective batch size; use 2–8 when GPU memory is limited
    # Regularisation
    "label_smoothing": 0.0,            # set 0.05–0.1 to reduce overconfidence; set 0.0 when using focal loss
    "max_grad_norm": 1.0,              # gradient clipping; lower to 0.5 if loss spikes on small data
    # Data loading — auto-scales to SLURM allocation on HPC; falls back to os.cpu_count()
    "dataloader_num_workers": _auto_num_workers(),  # parallel CPU workers per GPU
    "dataloader_prefetch_factor": 4,  # batches queued per worker (only when num_workers > 0)
    # Layer freezing — active by default for supported architectures (bert, roberta, longformer, bigbird, electra)
    # Skipped with a warning for unsupported architectures; set 0 to disable entirely
    "freeze_layers": 2,                # freeze embeddings + bottom N encoder layers; increase to 6 for very small datasets
    # Optional keys (not active by default — set as needed)
    # "loss_type": "focal"             # "cross_entropy" (default) | "focal" — use focal for severe class imbalance
    # "focal_gamma": 2.0               # focal loss focus strength; 2.0 is the standard (Lin et al. 2017)
    #                                  # higher γ (3–5) suppresses easy examples more aggressively
    # "class_weights": "balanced"      # "balanced" (sklearn), "effective_n" (Cui et al. 2019, recommended
    #                                  # for extreme imbalance), or list of floats ordered by class ID.
    #                                  # "effective_n" prevents weight blow-up when any class has < 5 samples.
    #                                  # Recommended combination for severe VA imbalance: loss_type="focal" +
    #                                  # class_weights="effective_n" + label_smoothing=0.0 + metric="f1_macro"
}

# LoRA-specific defaults (only applied when use_lora=True)
LORA_DEFAULTS: dict = {
    "lora_r": 16,        # rank of LoRA decomposition; 4–64; higher = more capacity, more params
    "lora_alpha": 64,    # scaling factor; typically 2–4× lora_r; higher = stronger adaptation
    "lora_dropout": 0.1, # dropout on LoRA layers; 0.05–0.1; increase for small datasets
}

# Reference map of well-known model families to canonical HuggingFace model IDs.
# model_name in train() / predict() / prepare_dataset() accepts any of:
#   - A value from this dict  (e.g. SUPPORTED_MODELS["biobert"])
#   - Any HuggingFace Hub ID  (e.g. "username/my-finetuned-bert")
#   - A local path to a saved model directory  (e.g. "/data/models/my_checkpoint")
# Architecture groups natively supported by freeze_model_layers():
#   BERT-family : bert, biobert, bioclinicalbert, bluebert, biomedbert, clinicalbert
#   RoBERTa     : biomedroberta
#   ELECTRA     : bioelectra
#   Long-range  : longformer, clinicallongformer, bigbird, clinicalbigbird
SUPPORTED_MODELS: dict[str, str] = {
    "bert":            "bert-base-uncased",
    "biobert":         "dmis-lab/biobert-base-cased-v1.2", #dmis-lab/biobert-v1.1
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "bluebert":        "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "biomedbert":      "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "clinicalbert":    "medicalai/ClinicalBERT",
    "biomedroberta":   "allenai/biomed_roberta_base",
    "bioelectra":      "kamalkraj/bioelectra-base-discriminator-pubmed",
    "longformer":      "allenai/longformer-base-4096",
    "clinicallongformer":      "yikuan8/Clinical-Longformer",
    "bigbird":         "google/bigbird-roberta-base",
    "clinicalbigbird":         "yikuan8/Clinical-BigBird",
}

# Models not on the HuggingFace Hub — must be fetched via download_model(key).
# download_model() extracts the archive to a temp directory and returns the
# local path, which can then be passed directly as model_name.
REMOTE_MODELS: dict[str, str] = {
    "roberta-pm": (
        "https://dl.fbaipublicfiles.com/biolm/"
        "RoBERTa-base-PM-M3-Voc-distill-hf.tar.gz"
    ),
}


def _looks_like_model_dir(path: Path) -> bool:
    """Return True when ``path`` looks like a HF-style local model directory."""
    if not path.is_dir():
        return False

    has_config = (path / "config.json").exists()
    has_weights = any(
        (path / filename).exists()
        for filename in (
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        )
    )
    return has_config and has_weights


def _find_extracted_model_dir(root: Path) -> Path | None:
    """Find the actual extracted model directory under ``root``.

    Some archives unpack directly into a single model directory, while others
    add an extra wrapper directory and place the HuggingFace files one level
    deeper. We return the shallowest directory that contains both
    ``config.json`` and model weights.
    """
    if _looks_like_model_dir(root):
        return root

    candidates = sorted(
        (
            path for path in root.rglob("*")
            if _looks_like_model_dir(path)
        ),
        key=lambda p: (len(p.relative_to(root).parts), str(p)),
    )
    return candidates[0] if candidates else None


def download_model(key: str, cache_dir: str | Path | None = None) -> str:
    """Download and extract a remote model checkpoint from REMOTE_MODELS.

    Downloads the archive to a temporary directory (or cache_dir), extracts
    it, removes the archive, and returns the local model directory path for
    use as model_name in train(), predict(), and prepare_dataset().

    Args:
        key: Key in REMOTE_MODELS (e.g. "roberta-pm").
        cache_dir: Directory to extract the model into.
                   Defaults to a new system temp directory (deleted on reboot).

    Returns:
        Absolute path to the extracted model directory.

    Raises:
        ValueError: If key is not in REMOTE_MODELS.

    Example:
        model_path = download_model("roberta-pm")
        train(..., model_name=model_path)
    """
    if key not in REMOTE_MODELS:
        raise ValueError(
            f"Unknown remote model key: '{key}'. "
            f"Available keys: {list(REMOTE_MODELS)}"
        )

    url = REMOTE_MODELS[key]
    archive_name = url.rsplit("/", 1)[-1]  # e.g. RoBERTa-base-PM-M3-Voc-distill-hf.tar.gz

    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="multimodalva_"))
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    existing_model_dir = _find_extracted_model_dir(cache_dir)
    if existing_model_dir is not None:
        logger.info("Reusing cached model at: %s", existing_model_dir)
        return str(existing_model_dir)

    archive_path = cache_dir / archive_name

    logger.info("Downloading %s ...", url)
    urllib.request.urlretrieve(url, archive_path)
    logger.info("Saved archive to %s", archive_path)

    logger.info("Extracting %s ...", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(cache_dir)
    archive_path.unlink()  # remove archive after extraction

    model_dir = _find_extracted_model_dir(cache_dir)
    if model_dir is None:
        raise FileNotFoundError(
            "Downloaded archive extracted successfully, but no HuggingFace-style "
            f"model directory was found under {cache_dir}."
        )

    logger.info("Model ready at: %s", model_dir)
    return str(model_dir)


def _get_dataset_labels(dataset) -> list[int]:
    """Extract integer labels from a ClassificationDataset or a Subset of one."""
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    if isinstance(dataset, Subset):
        return [dataset.dataset.labels[i] for i in dataset.indices]
    raise AttributeError(
        f"Cannot extract labels from dataset of type {type(dataset).__name__}. "
        "Expected ClassificationDataset or torch.utils.data.Subset."
    )


def _compute_eval_metrics(eval_pred) -> dict:
    """Compute accuracy and macro F1 from Trainer EvalPrediction.

    Passed as compute_metrics to the Trainer when has_eval=True.
    Both metrics are always present when logits are collected.
    EarlyStoppingCallback and load_best_model_at_end watch eval_macro_f1
    (more robust than eval_loss for imbalanced multiclass VA data).
    """
    from sklearn.metrics import f1_score  # noqa: PLC0415
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def _best_model_metric_name(has_eval: bool) -> str | None:
    """Return the best-model metric name in the format Trainer expects."""
    if not has_eval:
        return None
    # Use the explicit evaluation metric key for compatibility across Trainer
    # versions. Some versions accept an unprefixed name, but others look up the
    # exact key in the evaluation metrics dict and will fail the trial if only
    # ``macro_f1`` is provided here.
    return "eval_macro_f1"


def _val_split(dataset, val_size: float, random_state: int = 42) -> tuple[Subset, Subset]:
    """Carve a stratified validation subset from dataset, returning (train_subset, val_subset).

    Uses stratified splitting so class proportions are preserved in both subsets.
    The original dataset is not modified.
    """
    labels = _get_dataset_labels(dataset)
    indices = list(range(len(dataset)))
    train_idx, val_idx = _sklearn_val_split(
        indices,
        test_size=val_size,
        random_state=random_state,
        stratify=labels,
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the latest checkpoint subdirectory in output_dir, or None.

    HuggingFace Trainer saves checkpoints as ``checkpoint-{step}`` directories.
    Returns the one with the highest step number, or None if none exist.
    """
    checkpoints = [
        p for p in output_dir.iterdir()
        if p.is_dir() and p.name.startswith("checkpoint-") and p.name.split("-")[-1].isdigit()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))


def get_device() -> torch.device:
    """Return the best available device: CUDA, MPS, CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def freeze_model_layers(model, freeze_layers: int):
    """Freeze the embedding layer and the first N encoder layers.

    Supports BERT, RoBERTa, Longformer, BigBird, and ELECTRA architectures,
    covering all models in SUPPORTED_MODELS.

    Args:
        model: A loaded HuggingFace sequence classification model.
        freeze_layers: Number of encoder layers to freeze (from the bottom).
                       Set to 0 to freeze only embeddings; negative values are no-ops.

    Returns:
        model with the requested parameters frozen.

    Raises:
        ValueError: If the model architecture is not recognised.
    """
    encoder, embeddings = None, None
    for arch in ("bert", "roberta", "longformer", "bigbird", "electra"):
        backbone = getattr(model, arch, None)
        if backbone is not None:
            encoder = backbone.encoder
            embeddings = backbone.embeddings
            break

    if encoder is None:
        raise ValueError(
            f"Unsupported model architecture for layer freezing: {type(model).__name__}. "
            "Expected bert, roberta, longformer, bigbird, or electra."
        )

    if freeze_layers > 0:
        for param in embeddings.parameters():
            param.requires_grad = False

    n_layers = min(freeze_layers, len(encoder.layer))
    for i in range(n_layers):
        for param in encoder.layer[i].parameters():
            param.requires_grad = False

    frozen = n_layers + (1 if freeze_layers > 0 else 0)
    logger.info("Froze embeddings + %d encoder layers.", frozen - 1)
    return model


def _compute_effective_n_weights(
    labels_list: list[int],
    num_labels: int,
    beta: float = 0.9999,
) -> np.ndarray:
    """Compute per-class weights using the effective number of samples formula.

    Cui et al. (2019) "Class-Balanced Loss Based on Effective Number of Samples"
    (https://arxiv.org/abs/1901.05555).

    The effective number of samples for a class with n training examples is:
        E_n = (1 - beta^n) / (1 - beta)

    For small n, E_n ≈ n (no correction needed).  For large n, E_n saturates,
    preventing the weight for common classes from collapsing to near zero.

    Compared to sklearn's "balanced" (weight = N / (C * n_c)):
    - Prevents weight blow-up when any class has very few samples (< 5)
    - More numerically stable for the long-tailed VA cause distribution
    - beta=0.9999 is the standard value for datasets of 5 000–50 000 samples

    Args:
        labels_list: Integer class labels from the training set.
        num_labels:  Total number of classes (including unseen ones).
        beta:        Smoothing factor in (0, 1).  0.9999 suits most VA datasets.
                     Lower values (0.99) for very small datasets (< 500 samples).

    Returns:
        Array of shape (num_labels,) with per-class weights, normalised so the
        mean weight equals 1.0 (preserves the overall gradient scale).
    """
    counts = np.bincount(labels_list, minlength=num_labels).astype(float)
    counts = np.maximum(counts, 1.0)          # avoid division by zero for unseen classes
    effective_n = (1.0 - beta ** counts) / (1.0 - beta)
    weights = 1.0 / effective_n
    weights = weights / weights.mean()        # normalise: mean weight = 1.0
    return weights


class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class loss weights for class imbalance.

    Also re-applies label_smoothing_factor from TrainingArguments so that
    both class_weights and label_smoothing work correctly together.
    Used automatically by train() when hp["class_weights"] is set and
    hp["loss_type"] is not "focal".
    """

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            label_smoothing=self.args.label_smoothing_factor,
        )
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


class FocalLossTrainer(Trainer):
    """Trainer subclass implementing softmax focal loss for severe class imbalance.

    Focal loss (Lin et al. 2017, RetinaNet) down-weights easy, well-classified
    examples so the model focuses training budget on hard or rare cases:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where ``p_t`` is the softmax probability assigned to the true class,
    ``gamma`` controls the focus strength, and ``alpha_t`` is an optional
    per-class weight (same role as class_weights in WeightedTrainer).

    Compared to WeightedTrainer (weighted cross-entropy):
    - Re-weights within each class based on prediction confidence, not just
      across classes — benefits both rare and hard majority-class examples
    - Requires a good alpha initialisation (class_weights) to be most effective
    - Should not be combined with label_smoothing (set label_smoothing=0.0)

    Recommended configuration for severe VA imbalance:
        hp["loss_type"]     = "focal"
        hp["focal_gamma"]   = 2.0          # standard; tune 1.0–5.0 via HPO
        hp["class_weights"] = "effective_n" # Cui et al. 2019; better than "balanced"
                                            # for classes with < 5 training samples
        hp["label_smoothing"] = 0.0        # do not combine with focal loss
        optimize_metric       = "f1_macro" # or "balanced_accuracy" / "csmf_accuracy"

    Used automatically by train() when hp["loss_type"] == "focal".
    """

    def __init__(
        self,
        *args,
        focal_gamma: float = 2.0,
        class_weights: "torch.Tensor | None" = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.focal_gamma   = focal_gamma
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        num_labels  = logits.size(-1)
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        # Cast to float32 for numerical stability under fp16 training.
        logits_f = logits_flat.float()

        # p_t: softmax probability of the true class — used only for the
        # focal modulating factor.  Detached so gradients do not flow through
        # the weight term (standard focal loss practice).
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits_f, dim=-1)
            p_t   = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - p_t) ** self.focal_gamma   # shape: (batch,)

        # Per-sample CE (alpha applied here as class-level weight).
        # label_smoothing intentionally left at 0.0 when using focal loss;
        # the focal term already acts as a soft regulariser.
        alpha = (
            self.class_weights.to(logits_f.device)
            if self.class_weights is not None
            else None
        )
        ce_per_sample = torch.nn.functional.cross_entropy(
            logits_f,
            labels_flat,
            weight=alpha,
            reduction="none",
            label_smoothing=self.args.label_smoothing_factor,
        )

        loss = (focal_weight * ce_per_sample).mean()
        return (loss, outputs) if return_outputs else loss


def train(
    train_dataset: Dataset,
    label2id: dict,
    id2label: dict,
    model_name: str,
    output_dir: str | Path,
    hyperparams: dict | None = None,
    val_size: float | None = 0.1,
    use_lora: bool = False,
    gradient_checkpointing: bool = False,
    early_stopping_patience: int | None = None,
    resume: bool = True,
    cleanup_checkpoints: bool = False,
    use_fast: bool = True,
    eval_batch_size: int = 32,
    random_state: int = 42,
    use_compile: bool = False,
) -> tuple[Trainer, AutoTokenizer, dict]:
    """Fine-tune a BERT-family model for multiclass classification.

    Saves the following to output_dir:
        - Fine-tuned model weights and config
        - Tokenizer
        - label2id.json / id2label.json
        - hyperparams.json
        - training_metadata.json  (full log history + eval metrics)

    When use_lora=True, LoRA weights are merged into the base model before
    saving so that predict() can reload the model with the standard
    AutoModelForSequenceClassification.from_pretrained() call.

    Args:
        train_dataset: Tokenized ClassificationDataset (or Subset) from prepare_dataset().
        label2id: Label-to-integer mapping from prepare_dataset().
        id2label: Integer-to-label mapping from prepare_dataset().
        model_name: HuggingFace model name or local path (e.g. "bert-base-uncased").
        output_dir: Directory where all outputs are saved.
        hyperparams: Training hyperparameters. Merged over DEFAULT_HYPERPARAMS.
                     Core keys: learning_rate, batch_size, epochs, weight_decay,
                       warmup_ratio, gradient_accumulation_steps.
                     Regularisation: label_smoothing (0.0–0.1), max_grad_norm.
                     Hardware: dataloader_num_workers (increase for multi-core/GPU).
                     Imbalance: class_weights — "balanced" (auto from sklearn) or a
                       list of floats ordered by class ID.
                     Optional: freeze_layers (int), lora_r / lora_alpha / lora_dropout
                       (only used when use_lora=True).
        val_size: Fraction of train_dataset to hold out as a validation split for
                  in-training evaluation (loss monitoring, best-checkpoint selection,
                  early stopping). Stratified by class. Default 0.1.
                  Set to None to train on all training data with no in-training eval.
        use_lora: Apply LoRA adapters during training. Default False.
        gradient_checkpointing: Enable gradient checkpointing to reduce GPU memory.
                                Default False.
        early_stopping_patience: Stop training if eval_macro_f1 does not improve
                                  for this many epochs. Active whenever val_size > 0.
                                  Default None (disabled).
        resume: Resume training from the latest checkpoint in output_dir.
                Default True.
        cleanup_checkpoints: Delete intermediate ``checkpoint-*/`` subdirectories
                             from output_dir after training completes successfully.
                             These checkpoints are only needed to resume an interrupted
                             run; once training is done they consume significant disk
                             space (equal to the full model size per checkpoint, up to
                             ``save_total_limit=2``).  The final model weights saved
                             directly in output_dir are unaffected.  Default False.
        use_fast: Use the HuggingFace fast (Rust) tokenizer. Default True.
                  Set False for models that lack a fast tokenizer
                  (e.g. BlueBERT) to avoid a falling-back warning.
        eval_batch_size: Per-device batch size used exclusively for eval forward
                         passes. Decoupled from training batch_size. Default 32.
                         Increase further on GPUs with ample VRAM (e.g. 64 on A100).
                         Only relevant when val_size > 0.
        random_state: Lightweight seed used for RNG initialization plus
                      Hugging Face Trainer `seed` / `data_seed` settings.
                      Default 42.
        use_compile: Apply torch.compile() to the model before training.
                     Uses the "reduce-overhead" mode with backend="aot_eager" on MPS
                     (avoids unimplemented ops in the inductor backend) and
                     backend="inductor" on CUDA. First-call compilation takes 30–90 s;
                     subsequent calls are fast. Not recommended for HPO (compilation
                     overhead per trial). Default False.

    Returns:
        trainer:   HuggingFace Trainer with the fine-tuned model at trainer.model.
                   When val_size > 0, trainer.model holds the best checkpoint
                   (by eval_macro_f1) rather than the final epoch's weights.
        tokenizer: Tokenizer matching the model, ready for DataCollatorWithPadding.
        metadata:  Dict with keys output_dir, model_name, hyperparams, label2id,
                   id2label, log_history. Contains everything needed for in-memory
                   prediction without reloading from disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge hyperparams over defaults
    hp = {**DEFAULT_HYPERPARAMS}
    if use_lora:
        hp.update(LORA_DEFAULTS)
    if hyperparams:
        hp.update(hyperparams)

    runtime_tracker = RuntimeTracker(
        output_dir,
        report_name="train_runtime.json",
        metadata={
            "pipeline": "text_train",
            "model_name": model_name,
            "use_lora": use_lora,
            "gradient_checkpointing": gradient_checkpointing,
            "resume": resume,
            "random_state": random_state,
            "use_compile": use_compile,
        },
        logger_=logger,
    )

    _set_lightweight_seeds(random_state)
    num_labels = len(label2id)

    # --- Carve out validation split from training data ---
    # Test data must never enter train(); it is reserved exclusively for predict().
    if val_size is not None and val_size > 0.0:
        train_dataset, val_dataset = _val_split(
            train_dataset,
            val_size,
            random_state=random_state,
        )
        has_eval = True
        logger.info(
            "Validation split: %d train / %d val (val_size=%.2f).",
            len(train_dataset), len(val_dataset), val_size,
        )
    else:
        val_dataset = None
        has_eval = False

    # --- Compute class weights (if requested) ---
    # Uses the training subset (after val split) so weights reflect actual training labels.
    class_weights = None
    raw_cw = hp.get("class_weights")
    if raw_cw is not None:
        labels_list = _get_dataset_labels(train_dataset)
        if raw_cw == "balanced":
            weights = compute_class_weight(
                "balanced", classes=np.arange(num_labels), y=labels_list
            )
            class_weights = torch.tensor(weights, dtype=torch.float32)
        elif raw_cw == "effective_n":
            weights = _compute_effective_n_weights(labels_list, num_labels)
            class_weights = torch.tensor(weights, dtype=torch.float32)
        else:
            class_weights = torch.tensor(raw_cw, dtype=torch.float32)
        logger.info("Class weights (%s): %s", raw_cw, class_weights.tolist())

    # Warn if label_smoothing is set alongside focal loss — the combination is
    # not recommended since focal loss already acts as an implicit regulariser.
    loss_type = hp.get("loss_type", "cross_entropy")
    focal_gamma = float(hp.get("focal_gamma", 2.0))
    if loss_type == "focal" and hp.get("label_smoothing", 0.0) > 0.0:
        logger.warning(
            "label_smoothing=%.2f is set with loss_type='focal'.  "
            "Combining focal loss with label smoothing is not recommended — "
            "consider setting label_smoothing=0.0.",
            hp["label_smoothing"],
        )

    # --- Load base model ---
    # classifier_dropout overrides the model config's dropout on the classification head.
    # Set via hp["classifier_dropout"] (float, 0.0–0.5); None = use model default (~0.1).
    # Not all architectures expose this kwarg (e.g. Longformer uses hidden_dropout_prob).
    # Auto-detect support via the config class __init__ signature to avoid TypeError.
    _extra_model_kwargs = {}
    # When continuing fine-tuning from a checkpoint that already has a
    # classification head (e.g. a published MultimodalVA model reloaded as
    # `model_name`), the head size only matches if the new label set is identical.
    # Pass hyperparams={"ignore_mismatched_sizes": True} to reinitialise a new head
    # for a different cause set; it is a no-op when the head sizes already match.
    if hp.get("ignore_mismatched_sizes"):
        _extra_model_kwargs["ignore_mismatched_sizes"] = True
    if "classifier_dropout" in hp and hp["classifier_dropout"] is not None:
        import inspect
        from transformers import AutoConfig
        _cfg_cls = type(AutoConfig.from_pretrained(model_name))
        if "classifier_dropout" in inspect.signature(_cfg_cls.__init__).parameters:
            _extra_model_kwargs["classifier_dropout"] = hp["classifier_dropout"]
        else:
            logger.warning(
                "Model '%s' (%s) does not support 'classifier_dropout' — "
                "ignoring hp['classifier_dropout']=%.3f.  "
                "To control head dropout for this architecture, pass the appropriate "
                "config key (e.g. hidden_dropout_prob) via the hyperparams dict.",
                model_name, _cfg_cls.__name__, hp["classifier_dropout"],
            )

    # --- MPS architecture compatibility ---
    # Detect MPS early (before _is_mps is set further below) so we can configure
    # model kwargs at load time.
    _mps_at_load = torch.backends.mps.is_available() and not torch.cuda.is_available()
    if _mps_at_load:
        _model_lower = model_name.lower()
        if "longformer" in _model_lower:
            # LongformerSelfAttention uses scatter/gather ops that are not implemented
            # in Metal.  PYTORCH_ENABLE_MPS_FALLBACK=1 silently routes those ops to
            # CPU, but the resulting MPS↔CPU tensor copies frequently make the run
            # *slower* than pure CPU execution on M-chip Macs — not faster.
            # For local MPS development, BigBird with attention_type="original_full"
            # is the recommended drop-in replacement: same capacity, MPS-native ops.
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
                logger.warning(
                    "Longformer on MPS: LongformerSelfAttention uses scatter/gather "
                    "ops not natively supported by Metal.  Without "
                    "PYTORCH_ENABLE_MPS_FALLBACK=1 this will raise a RuntimeError.  "
                    "With the fallback enabled those ops run on CPU, which may be "
                    "slower than using --device cpu directly due to MPS↔CPU copies.  "
                    "Recommended MPS-native alternative: 'google/bigbird-roberta-base' "
                    "or 'yikuan8/Clinical-BigBird' (auto-configured to use "
                    "attention_type='original_full' on MPS)."
                )
            else:
                logger.warning(
                    "Longformer on MPS with PYTORCH_ENABLE_MPS_FALLBACK=1: "
                    "scatter/gather ops will execute on CPU.  This avoids crashes but "
                    "continuous MPS↔CPU tensor copies may make training slower than "
                    "running entirely on CPU.  For faster local runs consider "
                    "'google/bigbird-roberta-base' (MPS-native with original_full attention)."
                )
        elif "bigbird" in _model_lower:
            # BigBird's block-sparse attention (default) also uses custom CUDA ops
            # not available in Metal.  Setting attention_type="original_full" switches
            # to standard dense self-attention — same model weights, fully MPS-native.
            # The user can opt out by passing attention_type explicitly in hyperparams.
            if "attention_type" not in hp and "attention_type" not in _extra_model_kwargs:
                logger.info(
                    "BigBird on MPS: auto-setting attention_type='original_full' for "
                    "MPS-native computation.  Block-sparse attention requires custom "
                    "CUDA ops not available in Metal.  To use block_sparse anyway "
                    "(requires PYTORCH_ENABLE_MPS_FALLBACK=1), pass "
                    "hyperparams={'attention_type': 'block_sparse'}."
                )
                _extra_model_kwargs["attention_type"] = "original_full"
            elif hp.get("attention_type") == "block_sparse":
                logger.warning(
                    "BigBird on MPS with attention_type='block_sparse': this requires "
                    "PYTORCH_ENABLE_MPS_FALLBACK=1 and may be slower than CPU due to "
                    "MPS↔CPU tensor copies.  Remove attention_type from hyperparams to "
                    "auto-use 'original_full' instead."
                )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        **_extra_model_kwargs,
    )

    # --- Freeze layers (must happen before LoRA when use_lora=True) ---
    # freeze_model_layers() sets requires_grad=False on all parameters in the frozen
    # layers.  If called AFTER get_peft_model(), it silently freezes the LoRA adapters
    # (lora_A / lora_B) in those layers, defeating the purpose of LoRA.
    # Applying freeze to the base_model first ensures LoRA adapters remain trainable
    # on every layer while the base-model weights in bottom layers stay frozen.
    # When use_lora=False, freeze is applied to the model after assignment below.
    if hp.get("freeze_layers") and use_lora:
        try:
            base_model = freeze_model_layers(base_model, hp["freeze_layers"])
        except ValueError as e:
            logger.warning("Layer freezing skipped (unsupported architecture): %s", e)

    # --- Apply LoRA ---
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "peft is required for LoRA fine-tuning. "
                "Install it with: pip install 'multimodalva[lora]' or pip install peft>=0.7"
            ) from None

        # Required when gradient_checkpointing=True: ensures input embeddings retain
        # a grad_fn so gradients can flow through checkpointed activations.
        # Harmless when gradient_checkpointing=False.
        if gradient_checkpointing:
            base_model.enable_input_require_grads()

        # task_type is intentionally omitted.
        # TaskType.SEQ_CLS auto-populates modules_to_save=["classifier"] inside PEFT's
        # __post_init__ / inject_adapter.  With target_modules="all-linear", the classifier
        # head is already wrapped as a LoraLinear; PEFT >= 0.10 then raises
        #   TypeError: modules_to_save cannot be applied to modules of type LoraLinear
        # because it tries to double-wrap it.  Passing modules_to_save=[] does not reliably
        # prevent this — some PEFT versions override it when the list is falsy.
        # Omitting task_type entirely leaves modules_to_save=None (no auto-population)
        # and does not affect training: AutoModelForSequenceClassification already owns the
        # classification head; PEFT's task_type is only a hint for its module-wrapping logic.
        peft_config = LoraConfig(
            r=hp["lora_r"],
            lora_alpha=hp["lora_alpha"],
            lora_dropout=hp["lora_dropout"],
            target_modules="all-linear",  # works across all architectures + PEFT versions
            bias="none",
        )
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()
    else:
        model = base_model

    # --- Freeze layers (non-LoRA path) ---
    # When use_lora=False, freeze is applied here to the full model.
    # When use_lora=True, freeze was already applied to base_model above.
    if hp.get("freeze_layers") and not use_lora:
        try:
            model = freeze_model_layers(model, hp["freeze_layers"])
        except ValueError as e:
            logger.warning("Layer freezing skipped (unsupported architecture): %s", e)

    # --- Device capability flags ---
    _is_cuda = torch.cuda.is_available()
    _is_mps  = torch.backends.mps.is_available() and not _is_cuda
    _world_size = max(1, _int_env("WORLD_SIZE", 1) or 1)
    _rank = _int_env("RANK", 0) or 0

    # --- CUDA performance flags (no-ops on CPU / MPS) ---
    if _is_cuda:
        # TF32 cuts matmul latency ~2–3× on Ampere+ (A100, RTX 30xx) with negligible
        # accuracy loss.  Both flags must be set: one for matmuls, one for cuDNN convs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Use higher matmul precision kernels where available (CUDA + MPS).
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # bf16 (bfloat16) is the preferred mixed-precision format on Ampere+ (A100, H100).
    # It has the same dynamic range as fp32 (8-bit exponent) so it is numerically more
    # stable than fp16 (5-bit exponent) and does not require loss scaling.
    # Volta-class GPUs (V100) do NOT support native bf16 — fall back to fp16 there.
    _cuda_bf16 = _is_cuda and torch.cuda.is_bf16_supported()

    # Fused AdamW (PyTorch ≥ 2.0, CUDA only): kernel-fused optimizer step, ~10–20%
    # faster per step and lower peak memory than the unfused implementation.
    _torch_major = int(torch.__version__.split(".")[0])
    _optim = "adamw_torch_fused" if (_is_cuda and _torch_major >= 2) else "adamw_torch"

    # --- Device placement ---
    # Use LOCAL_RANK for DDP (torchrun): each worker must land on its own GPU.
    # Falls back to cuda:0 / mps / cpu for single-process runs.
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if _local_rank >= 0 and _is_cuda:
        torch.cuda.set_device(_local_rank)
        _device = torch.device("cuda", _local_rank)
    elif _local_rank >= 0 and not _is_cuda:
        logger.warning(
            "LOCAL_RANK=%d is set but CUDA is unavailable; falling back to %s.",
            _local_rank,
            "mps" if _is_mps else "cpu",
        )
        _device = torch.device("mps" if _is_mps else "cpu")
    elif _is_cuda:
        if torch.cuda.device_count() > 1:
            logger.warning(
                "Detected %d CUDA GPUs but LOCAL_RANK is unset. "
                "For efficient multi-GPU DDP on OSC, launch with torchrun "
                "(e.g. --nproc_per_node=%d).",
                torch.cuda.device_count(),
                torch.cuda.device_count(),
            )
        _device = torch.device("cuda")
    elif _is_mps:
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    model.to(_device)

    # --- Optional torch.compile() ---
    # "reduce-overhead" mode lowers Python dispatch overhead between ops (~10–30%
    # throughput gain on long training runs after the one-time compilation cost).
    # Use aot_eager on MPS — the default inductor backend has unimplemented ops for
    # many transformer architectures on Metal and raises RuntimeError at compile time.
    # Not recommended for HPO (compilation overhead is ~60 s per fresh trial start).
    if use_compile:
        _torch_ver = tuple(
            int(x) for x in torch.__version__.split(".")[:2]
            if x.isdigit()
        )
        if _torch_ver >= (2, 0):
            try:
                _compile_backend = "aot_eager" if _is_mps else "inductor"
                model = torch.compile(
                    model, backend=_compile_backend, mode="reduce-overhead"
                )
                logger.info(
                    "torch.compile() applied: backend=%s, mode=reduce-overhead.",
                    _compile_backend,
                )
            except Exception as exc:
                logger.warning(
                    "torch.compile() failed (backend=%s): %s — continuing in eager mode.",
                    "aot_eager" if _is_mps else "inductor",
                    exc,
                )
        else:
            logger.warning(
                "torch.compile() requires PyTorch >= 2.0 (found %s) — skipping.",
                torch.__version__,
            )

    # --- Training arguments ---
    # dataloader_pin_memory: True speeds up CUDA host→device transfers but is
    #   unsupported on MPS. Explicitly False on MPS/CPU to suppress UserWarning.
    # fp16 / bf16: mutually exclusive. bf16 preferred on Ampere+ (A100/H100);
    #   fp16 for Volta (V100); neither on MPS (uses bf16 natively, no flag needed).
    # group_by_length: group similarly-lengthed sequences into the same batch.
    #   Reduces intra-batch padding significantly for VA narratives whose lengths
    #   vary widely. Typical throughput gain: 15–35% depending on length distribution.
    # ddp_find_unused_parameters: set False when layers are frozen — frozen params
    #   are "unused" in the DDP backward pass, and the default True setting adds a
    #   traversal cost that is wasted when we already know which params are inactive.
    # optim: fused AdamW (PyTorch ≥ 2.0, CUDA) fuses the optimizer update kernel,
    #   reducing CUDA kernel launch overhead ~10–20% per step.
    # label_names: force Trainer to treat "labels" as supervised targets even when
    #   the model is wrapped by PEFT/LoRA across transformers versions.
    # gradient_checkpointing_kwargs use_reentrant=False: non-reentrant autograd
    #   checkpointing (PyTorch ≥ 2.1) does not require any input to have requires_grad,
    #   eliminating the "Gradients will be None" warning with frozen layers.
    metric_for_best_model = _best_model_metric_name(has_eval)
    num_workers = int(hp["dataloader_num_workers"])
    # On MPS (Apple Silicon), multiple DataLoader workers hurt rather than help:
    # (1) macOS uses the "spawn" start method — each worker costs ~0.5 s to start.
    # (2) Unified memory means batches don't need to be copied from CPU RAM to GPU;
    #     multiprocessing just adds IPC overhead on top of shared physical memory.
    # Override to 0 unless the user explicitly set MULTIMODALVA_DATALOADER_WORKERS.
    if _is_mps and num_workers > 0 and _int_env("MULTIMODALVA_DATALOADER_WORKERS") is None:
        logger.info(
            "MPS device: overriding dataloader_num_workers %d → 0 "
            "(unified memory + macOS spawn overhead; single-process loading is faster). "
            "Set MULTIMODALVA_DATALOADER_WORKERS=N to override.",
            num_workers,
        )
        num_workers = 0
    prefetch_factor = int(hp.get("dataloader_prefetch_factor", 4))
    _has_frozen = bool(hp.get("freeze_layers", 0))
    runtime_tracker.update_metadata(
        num_labels=num_labels,
        has_eval=has_eval,
        train_examples=len(train_dataset),
        val_examples=(len(val_dataset) if val_dataset is not None else 0),
        dataloader_num_workers=num_workers,
        dataloader_prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        hyperparams=hp,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=random_state,
        data_seed=random_state,
        learning_rate=hp["learning_rate"],
        per_device_train_batch_size=hp["batch_size"],
        # eval batch size is decoupled from training batch size to prevent eval OOM
        # with LoRA + large gradient_accumulation. Default 32 is safe for all
        # BERT-family models; increase to 64+ on GPUs with ≥ 40 GB VRAM.
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=hp["epochs"],
        warmup_ratio=hp["warmup_ratio"],
        weight_decay=hp["weight_decay"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        label_smoothing_factor=hp["label_smoothing"],
        max_grad_norm=hp["max_grad_norm"],
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True if has_eval else None,
        logging_steps=50,
        report_to="none",
        # Mixed precision — bf16 preferred on Ampere+, fp16 fallback for Volta
        fp16=(_is_cuda and not _cuda_bf16),
        bf16=_cuda_bf16,
        # Optimizer — fused kernel on PyTorch ≥ 2.0 + CUDA
        optim=_optim,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=_is_cuda,
        dataloader_persistent_workers=bool(num_workers > 0),
        dataloader_prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        # Group sequences of similar length to minimise intra-batch padding waste
        group_by_length=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        # DDP: skip unused-parameter traversal when layers are frozen
        ddp_find_unused_parameters=(False if _has_frozen else None),
        disable_tqdm=False,
    )

    # --- Load tokenizer for DataCollator (needed before Trainer is created) ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    # EarlyStoppingCallback watches eval_macro_f1 (via metric_for_best_model).
    # load_best_model_at_end=has_eval is always True when eval is available —
    # EarlyStoppingCallback requires load_best_model_at_end=True.
    # Active whenever val_size > 0 and early_stopping_patience is set.
    callbacks = []
    if has_eval and early_stopping_patience is not None:
        from transformers import EarlyStoppingCallback  # noqa: PLC0415
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    # --- Create trainer ---
    # DataCollatorWithPadding pads each batch to its own longest sequence,
    # so train and test batches are handled independently with no length mismatch.
    #
    # Dispatch rules:
    #   loss_type="focal"        → FocalLossTrainer (focal loss + optional alpha weights)
    #   class_weights set        → WeightedTrainer  (weighted cross-entropy + label smoothing)
    #   otherwise                → standard Trainer (cross-entropy)
    _trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(
            tokenizer,
            # Padding to multiples of 8 improves memory alignment and throughput
            # for CUDA (tensor core alignment) and MPS (Metal memory alignment).
            pad_to_multiple_of=(8 if (_is_cuda or _is_mps) else None),
        ),
        compute_metrics=_compute_eval_metrics if has_eval else None,
        callbacks=callbacks or None,
    )
    if loss_type == "focal":
        logger.info(
            "Using FocalLossTrainer  gamma=%.1f  class_weights=%s",
            focal_gamma, raw_cw,
        )
        trainer = FocalLossTrainer(
            **_trainer_kwargs,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
        )
    elif class_weights is not None:
        trainer = WeightedTrainer(**_trainer_kwargs, class_weights=class_weights)
    else:
        trainer = Trainer(**_trainer_kwargs)

    # --- Strip HuggingFace's auto-injected Ray Tune callback ---
    # When a Ray Tune trial is active, transformers.Trainer.__init__ automatically
    # adds RayTuneCallback, which calls ray.tune.report() after every eval epoch.
    # Those intermediate reports trigger OptunaSearch.on_trial_result(), which
    # raises KeyError because the trial ID isn't registered for intermediate results.
    # Our _ray_trial_fn returns a final metrics dict directly — no intermediate
    # reporting needed — so removing this callback is safe in all contexts.
    try:
        from transformers.integrations import RayTuneCallback  # noqa: PLC0415
        trainer.remove_callback(RayTuneCallback)
    except (ImportError, Exception):
        pass

    # --- Resolve resume checkpoint ---
    resume_checkpoint = None
    if resume:
        resume_checkpoint = _find_latest_checkpoint(output_dir)
        if resume_checkpoint is not None:
            logger.info("Resuming from checkpoint: %s", resume_checkpoint)
        else:
            logger.warning(
                "resume=True but no checkpoints found in %s — starting from scratch.", output_dir
            )
    else:
        existing = _find_latest_checkpoint(output_dir)
        if existing is not None:
            logger.warning(
                "Existing checkpoint found at %s. Pass resume=True to continue from it "
                "instead of starting from scratch.",
                existing,
            )

    # --- Train ---
    with runtime_tracker.stage(
        "trainer_train",
        details={
            "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "has_eval": has_eval,
            "batch_size": hp["batch_size"],
            "epochs": hp["epochs"],
        },
        monitor_gpu=True,
        device=_device,
    ):
        trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Release any MPS memory that the Metal runtime is holding but no longer
    # needs.  Without an explicit cache flush, fragmented memory from the forward
    # + backward passes can accumulate across multiple train() calls (e.g. HPO
    # trials), eventually causing OOM on machines with limited unified memory.
    if _is_mps:
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    is_world_zero = trainer.is_world_process_zero() if hasattr(trainer, "is_world_process_zero") else True

    # --- Merge LoRA weights before saving for inference compatibility ---
    if use_lora and is_world_zero:
        model = model.merge_and_unload()
        trainer.model = model

    with runtime_tracker.stage(
        "save_artifacts",
        details={
            "cleanup_checkpoints": cleanup_checkpoints,
            "is_world_process_zero": is_world_zero,
        },
    ):
        if is_world_zero:
            # --- Save model and tokenizer ---
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            # --- Optionally remove intermediate checkpoints ---
            # checkpoint-* dirs are only needed to resume an interrupted training run.
            # After successful completion the final weights are in output_dir; the
            # checkpoint copies are redundant and can be several hundred MB each.
            if cleanup_checkpoints:
                for ckpt in output_dir.iterdir():
                    if (
                        ckpt.is_dir()
                        and ckpt.name.startswith("checkpoint-")
                        and ckpt.name.split("-")[-1].isdigit()
                    ):
                        shutil.rmtree(ckpt)
                        logger.info("Removed checkpoint: %s", ckpt)

            # --- Save label maps and hyperparams ---
            with open(output_dir / "label2id.json", "w") as f:
                json.dump(label2id, f, indent=2)
            with open(output_dir / "id2label.json", "w") as f:
                json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
            with open(output_dir / "hyperparams.json", "w") as f:
                json.dump(hp, f, indent=2)

    metadata = {
        "output_dir": str(output_dir),
        "model_name": model_name,
        "hyperparams": hp,
        "label2id": label2id,
        "id2label": id2label,
        "log_history": trainer.state.log_history,
        "runtime_report": str(runtime_tracker.report_path),
        "runtime_stage_csv": str(runtime_tracker.stage_csv_path),
    }
    if is_world_zero:
        with open(output_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Training complete. Artifacts saved to %s", output_dir)
    else:
        logger.info(
            "Training complete on rank %d/%d. Skipping artifact writes on non-zero rank.",
            _rank,
            _world_size,
        )
    runtime_tracker.update_metadata(
        rank=_rank,
        world_size=_world_size,
        completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        training_metadata_path=str(output_dir / "training_metadata.json") if is_world_zero else None,
    )
    return trainer, tokenizer, metadata
