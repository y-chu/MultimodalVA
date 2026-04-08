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
import shutil
import tarfile
import tempfile
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)

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
    # Data loading
    "dataloader_num_workers": 2,       # parallel CPU workers per GPU; 4–8 on multi-core systems
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
    use_fast: bool = False,
    eval_batch_size: int = 4,
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
        use_fast: Use the HuggingFace fast (Rust) tokenizer. Default False.
                  Set False for models that lack a fast tokenizer
                  (e.g. BlueBERT) to avoid a falling-back warning.
        eval_batch_size: Per-device batch size used exclusively for eval forward
                         passes. Decoupled from training batch_size to prevent
                         eval OOM when training uses large batches. Default 4.
                         Only relevant when val_size > 0.

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

    num_labels = len(label2id)

    # --- Carve out validation split from training data ---
    # Test data must never enter train(); it is reserved exclusively for predict().
    if val_size is not None and val_size > 0.0:
        train_dataset, val_dataset = _val_split(train_dataset, val_size)
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

    model.to(get_device())

    # --- Device capability flags ---
    _is_cuda = torch.cuda.is_available()
    _is_mps  = torch.backends.mps.is_available() and not _is_cuda

    # --- Training arguments ---
    # dataloader_pin_memory: True speeds up CUDA transfers but is unsupported on MPS.
    #   HuggingFace TrainingArguments defaults to True, which triggers a UserWarning on
    #   Apple Silicon. Explicitly set to False on MPS (and CPU) to suppress it.
    # fp16: only safe on CUDA; MPS uses bfloat16 natively so fp16=False is correct.
    # label_names: force Trainer to treat "labels" as supervised targets even when
    #   the model is wrapped by PEFT/LoRA. Some transformers versions cannot infer
    #   label fields from wrapped forward signatures during evaluation, which causes
    #   eval to emit only timing metrics and breaks metric_for_best_model.
    # gradient_checkpointing_kwargs use_reentrant=False: the reentrant implementation
    #   requires at least one input tensor to have requires_grad=True.  When layers are
    #   frozen (freeze_layers > 0) the early checkpointed segments receive all-frozen
    #   inputs, causing "None of the inputs have requires_grad=True. Gradients will be
    #   None."  use_reentrant=False (non-reentrant autograd checkpointing, PyTorch >= 2.1)
    #   removes this requirement and is the recommended modern default.
    metric_for_best_model = _best_model_metric_name(has_eval)
    num_workers = int(hp["dataloader_num_workers"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=hp["learning_rate"],
        per_device_train_batch_size=hp["batch_size"],
        # eval_batch_size is intentionally decoupled from training batch_size.
        # Using training batch_size for eval can OOM (especially with LoRA +
        # gradient_accumulation where effective batch is already large), leaving
        # eval_loss absent and causing KeyError in _determine_best_metric.
        # A small fixed value (default 4) guarantees eval never OOMs.
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
        # Use the explicit evaluation metric key to match the metrics dict that
        # Trainer produces at eval time across transformers versions.
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True if has_eval else None,
        logging_steps=50,
        report_to="none",
        fp16=_is_cuda,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=_is_cuda,
        dataloader_persistent_workers=bool(num_workers > 0),
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
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
        data_collator=DataCollatorWithPadding(tokenizer),
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
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # --- Merge LoRA weights before saving for inference compatibility ---
    if use_lora:
        model = model.merge_and_unload()
        trainer.model = model

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
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete. Artifacts saved to %s", output_dir)
    return trainer, tokenizer, metadata
