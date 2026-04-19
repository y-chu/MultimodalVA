"""
Search space definitions for text (transformer) HPO.

This module holds all parameter ranges and adaptive-defaults logic so that
hyperparameter values are easy to find, read, and tune without wading through
the HPO orchestration code in hpo.py.

Spec format (used by both Optuna and Ray Tune backends in hpo.py):
    ("float_log", low, high)   — log-uniform float  (best for learning rates)
    ("float",     low, high)   — uniform float
    ("int",       low, high)   — uniform int
    ("categorical", [values])  — discrete choices
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Balanced reference space
# Calibrated for ~20 classes, 3 000–5 000 samples.
# get_default_search_space() copies this dict and overrides specific keys
# based on actual n_samples and n_classes — do not hardcode these values
# in analysis scripts; call get_default_search_space() instead.
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE: dict = {
    # AdamW learning rate — most influential hyperparameter for BERT fine-tuning.
    # BERT paper recommends 2e-5 to 5e-5; log-uniform covers the relevant scale.
    # Wider range (1e-5, 1e-4) if the default range consistently hits a boundary.
    "learning_rate": ("float_log", 8e-6, 4e-5),

    # Per-device training batch size.
    # 16 is the standard for 16 GB GPUs; use 8 if hitting OOM, or 32 for larger GPUs.
    # Effective batch size = batch_size × gradient_accumulation_steps × n_GPUs.
    "batch_size": ("categorical", [8, 16, 32]),

    # Number of full passes over the training data.
    # 3–5 epochs is typical for BERT fine-tuning; small datasets may benefit from more.
    # Always included so that optimize() returns an epoch count for the final train() call.
    "epochs": ("categorical", [3, 5, 7]),

    # L2 regularisation on non-bias / non-LayerNorm parameters (AdamW decoupled decay).
    # 0.01 is the HuggingFace default; uniform [0.0, 0.1] covers the actionable range.
    # Log-uniform is unnecessary — values below ~0.001 are functionally identical to 0.
    "weight_decay": ("float", 0.0, 0.1),

    # Fraction of total training steps used for linear LR warmup.
    # BERT paper uses 0.1 (10%); 0.0–0.1 is common in practice.
    # Larger values (0.1–0.2) can help stabilize training on noisy or imbalanced data.
    "warmup_ratio": ("float", 0.0, 0.1),

    # Number of gradient steps to accumulate before an optimizer update.
    # Simulates a larger effective batch size without extra GPU memory.
    # Use 4 or 8 when batch_size is forced low by memory constraints.
    "gradient_accumulation_steps": ("categorical", [1, 2]),

    # Number of encoder layers to freeze from the bottom (embeddings + N layers).
    # Freezing reduces trainable parameters and acts as regularisation.
    # 0  — full fine-tuning; best when data is large or domain differs from pre-training.
    # 2  — light freeze (default); good starting point for most VA datasets.
    # 4  — moderate; useful when the dataset is small (<2 000 samples).
    # 6  — aggressive (half of BERT-base's 12 layers); use when heavily overfitting.
    "freeze_layers": ("categorical", [2, 4, 6]),

    # Classifier dropout — regularizes the randomly-initialised classification head.
    # Kept narrow [0.1, 0.2]: values above 0.2 destroy signal from rare class examples
    # (5–15 samples) by randomly dropping 20–40 % of head features per forward pass.
    # The pretrained default (~0.1) is already sensible; this range allows modest tuning.
    "classifier_dropout": ("float", 0.1, 0.2),

    # label_smoothing intentionally omitted:
    # - Reduces the loss contribution from every training example, including scarce
    #   minority classes that already have limited learning signal.
    # - No meaningful benefit for BERT fine-tuning on small, imbalanced VA datasets
    #   (was popularised on ImageNet with millions of examples and 1 000 classes).
    # - Use class_weights="effective_n" or loss_type="focal" for imbalance instead.
}


# ---------------------------------------------------------------------------
# Optional add-on spaces (merged in by optimize() / optimize_ray())
# ---------------------------------------------------------------------------

# Focal loss — merged when use_focal=True.
# loss_type is fixed to "focal"; only the tunable focal parameters are searched.
# Recommended metric: "f1_macro", "balanced_accuracy", or "csmf_accuracy".
# Avoid "log_loss" with focal loss (distorts calibration).
FOCAL_SEARCH_SPACE: dict = {
    # Focus strength γ — how aggressively easy examples are down-weighted.
    # γ=0 reduces to standard CE.  γ=2 is the RetinaNet default.
    # γ>3 is rarely beneficial and can cause gradient instability on tiny classes.
    "focal_gamma": ("float", 1.0, 4.0),

    # Per-class alpha weights (α) — rebalances gradient contribution across classes.
    # "effective_n" (Cui et al. 2019) recommended when any class has <~10 samples.
    # "balanced" (sklearn) adequate for moderate imbalance.
    "class_weights": ("categorical", ["balanced", "effective_n"]),
}

# LoRA adapter parameters — merged when use_lora=True.
LORA_SEARCH_SPACE: dict = {
    # Rank of the LoRA low-rank decomposition — controls adapter capacity.
    # Higher rank = more trainable parameters; r=8 is the LoRA paper default.
    "lora_r": ("categorical", [4, 8, 16]),

    # LoRA scaling factor (output *= lora_alpha / lora_r).
    # Common heuristic: lora_alpha = 2 × lora_r (e.g. r=8 → alpha=16).
    "lora_alpha": ("categorical", [8, 16, 32, 64]),

    # Dropout inside LoRA adapters.
    # 0.05 is the paper default; 0.0 often works well for small adapters.
    "lora_dropout": ("float", 0.0, 0.1),
}


# ---------------------------------------------------------------------------
# Adaptive default search space  (n_samples × n_classes)
# ---------------------------------------------------------------------------

def _sample_tier(n_samples: int) -> str:
    """Coarse tier from training-set size (calibrated for 500–15 000 VA datasets).

    small    < 1 500  — few updates per epoch; lean toward stronger regularisation
    moderate 1 500–6 000  — typical VA dataset; balanced defaults
    large    > 6 000  — enough data to justify higher capacity / relaxed constraints
    """
    if n_samples < 1500:
        return "small"
    if n_samples < 6000:
        return "moderate"
    return "large"


def _class_tier(n_classes: int) -> str:
    """Coarse tier from number of target classes (calibrated for 5–100 VA causes).

    few      ≤ 14  — compact output head; standard regularisation
    moderate  15–39  — moderate multi-class complexity
    many     ≥ 40  — large output head; needs more epochs and lower layer freezing
    """
    if n_classes <= 14:
        return "few"
    if n_classes <= 39:
        return "moderate"
    return "many"


def get_default_search_space(n_samples: int, n_classes: int) -> dict:
    """Return a default HPO search space adapted to dataset scale.

    Starts from ``DEFAULT_SEARCH_SPACE`` (the balanced reference) and overrides
    specific parameters based on training-set size and number of target classes.
    Any explicit ``search_space`` passed by the caller is merged on top — caller
    overrides always win.

    Adaptations:
    - ``epochs``: small data / many classes → more epochs.
    - ``batch_size``: small data → smaller batches (more gradient updates per epoch).
    - ``freeze_layers``: many classes → less freezing (need more transformer capacity).
    - ``classifier_dropout``: small data or many classes → higher dropout range.
    - ``weight_decay``: small data → stronger L2 regularisation.
    - ``gradient_accumulation_steps``: small data → more accumulation steps.
    - ``label_smoothing``: many classes → slightly wider smoothing range.
    """
    st = _sample_tier(n_samples)
    ct = _class_tier(n_classes)
    space = dict(DEFAULT_SEARCH_SPACE)  # copy the balanced reference

    # ── epochs ──────────────────────────────────────────────────────────────
    # Small data and many classes both push toward more training.
    _epoch_map = {
        ("small",    "few"):      ("categorical", [5, 8, 10]),
        ("small",    "moderate"): ("categorical", [5, 8, 12]),
        ("small",    "many"):     ("categorical", [7, 10, 15]),
        ("moderate", "few"):      ("categorical", [3, 5, 7]),
        ("moderate", "moderate"): ("categorical", [3, 5, 8]),
        ("moderate", "many"):     ("categorical", [5, 7, 10]),
        ("large",    "few"):      ("categorical", [2, 3, 5]),
        ("large",    "moderate"): ("categorical", [3, 5, 7]),
        ("large",    "many"):     ("categorical", [3, 5, 8]),
    }
    space["epochs"] = _epoch_map[(st, ct)]

    # ── batch_size ───────────────────────────────────────────────────────────
    # Small data: small batches → more gradient updates per epoch.
    # Large data: larger batches → stable gradients.
    if st == "small":
        space["batch_size"] = ("categorical", [4, 8, 16])
    elif st == "large":
        space["batch_size"] = ("categorical", [16, 32, 64])
    # moderate: keep [8, 16, 32]

    # ── freeze_layers ────────────────────────────────────────────────────────
    # Many classes demand full transformer capacity → prefer less freezing.
    # Small data → more freezing acts as regularisation.
    if st == "small" and ct == "few":
        space["freeze_layers"] = ("categorical", [2, 4, 6])
    elif st == "small" and ct == "moderate":
        space["freeze_layers"] = ("categorical", [2, 4])
    elif st == "small" and ct == "many":
        space["freeze_layers"] = ("categorical", [0, 2, 4])
    elif st == "moderate" and ct == "few":
        space["freeze_layers"] = ("categorical", [2, 4, 6])
    elif st == "moderate" and ct in ("moderate", "many"):
        space["freeze_layers"] = ("categorical", [0, 2, 4])
    else:  # large (any class tier)
        space["freeze_layers"] = ("categorical", [0, 2, 4])

    # ── classifier_dropout ───────────────────────────────────────────────────
    # Kept bounded: high dropout destroys signal from rare class examples.
    if st == "small" or ct == "many":
        space["classifier_dropout"] = ("float", 0.1, 0.25)  # slightly wider for more regularization need
    elif st == "large" and ct == "few":
        space["classifier_dropout"] = ("float", 0.0, 0.15)  # less regularization needed
    # else: default [0.1, 0.2]

    # ── weight_decay ─────────────────────────────────────────────────────────
    # Scaled proportionally with the new narrower baseline [0.0, 0.1].
    if st == "small":
        space["weight_decay"] = ("float", 0.0, 0.15)   # slightly wider for regularization pressure
    elif st == "large":
        space["weight_decay"] = ("float", 0.0, 0.05)   # large data needs less L2
    # moderate: default [0.0, 0.1]

    # ── gradient_accumulation_steps ──────────────────────────────────────────
    # Small batches benefit from accumulation to maintain a reasonable effective size.
    if st == "small":
        space["gradient_accumulation_steps"] = ("categorical", [2, 4])
    # moderate/large: default [1, 2]

    # label_smoothing is intentionally not searched — see DEFAULT_SEARCH_SPACE comment.

    return space
