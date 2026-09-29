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

Precedence, lowest to highest: TEXT_DEFAULT_SEARCH_SPACE →
get_text_default_search_space(n_samples, n_classes) → LORA/FOCAL when enabled →
the caller's ``search_space=``, which overrides per key. The run log records
which keys the caller replaced. See FAQ.md, "How do I set the hyperparameter
search space?".
"""

from __future__ import annotations

from ..utils.hpo_defaults import get_class_tier

# ---------------------------------------------------------------------------
# Balanced reference space
# Calibrated for ~20 classes, 3 000–5 000 samples.
# get_text_default_search_space() copies this dict and overrides specific keys
# based on actual n_samples and n_classes — do not hardcode these values
# in analysis scripts; call get_text_default_search_space() instead.
# ---------------------------------------------------------------------------

TEXT_DEFAULT_SEARCH_SPACE: dict = {
    # AdamW learning rate. Log-uniform, since what matters is the order of magnitude.
    # LoRA adapters usually want a higher rate than full fine-tuning.
    "learning_rate": ("float_log", 8e-6, 1e-4),

    # Per-device training batch size. Effective batch size is
    # batch_size × gradient_accumulation_steps × n_GPUs.
    "batch_size": ("categorical", [8, 16, 32]),

    # Number of full passes over the training data.
    # 3–5 epochs is typical for BERT fine-tuning; small datasets may benefit from more.
    # Always included so that optimize_text() returns an epoch count for the final train_text() call.
    "epochs": ("categorical", [3, 5, 7]),

    # L2 regularisation on non-bias / non-LayerNorm parameters (AdamW decoupled decay).
    # 0.01 is the HuggingFace default; uniform [0.0, 0.1] covers the actionable range.
    # Log-uniform is unnecessary — values below ~0.001 are functionally identical to 0.
    "weight_decay": ("float", 0.0, 0.1),

    # Fraction of total training steps used for linear LR warmup.
    "warmup_ratio": ("float", 0.0, 0.2),

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

    # Dropout on the randomly-initialised classification head. Higher values
    # regularise more, at the cost of signal from classes with few examples.
    "classifier_dropout": ("float", 0.1, 0.2),

    # label_smoothing is not searched by default — it softens the target for every
    # example, including classes that already have few. To search it anyway, pass
    # e.g. {"label_smoothing": ("float", 0.0, 0.15)} in search_space=; 0.0 turns it
    # off. For imbalance, class_weights="effective_n" and loss_type="focal" are the
    # other levers.
}


# ---------------------------------------------------------------------------
# Optional add-on spaces (merged in by optimize_text() / optimize_text_ray(), i.e.
# by either backend search_text() dispatches to)
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
    # 0.05 is the LoRA paper's default; 0.0 often works well for small adapters.
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


def get_text_default_search_space(n_samples: int, n_classes: int) -> dict:
    """Return a default HPO search space adapted to dataset scale.

    Starts from ``TEXT_DEFAULT_SEARCH_SPACE`` (the balanced reference) and overrides
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
    ``label_smoothing`` is deliberately absent from every tier — see the comment
    on TEXT_DEFAULT_SEARCH_SPACE. Pass it in ``search_space=`` to search it.
    """
    st = _sample_tier(n_samples)
    # few: compact output head, standard regularisation.
    # many: large output head, needs more epochs and less layer freezing.
    ct = get_class_tier(n_classes)
    space = dict(TEXT_DEFAULT_SEARCH_SPACE)  # copy the balanced reference

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
    # Smaller batches give more gradient updates per epoch; larger ones give
    # steadier gradients.
    if st == "small":
        space["batch_size"] = ("categorical", [4, 8, 16])
    elif st == "moderate":
        space["batch_size"] = ("categorical", [4, 8, 16, 32])
    else:  # large
        space["batch_size"] = ("categorical", [16, 32, 64])

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
    if st == "moderate":
        space["classifier_dropout"] = ("float", 0.1, 0.5) if ct == "many" else ("float", 0.1, 0.4)
    elif st == "small" or ct == "many":
        space["classifier_dropout"] = ("float", 0.1, 0.25)
    elif st == "large" and ct == "few":
        space["classifier_dropout"] = ("float", 0.0, 0.15)
    # else: default [0.1, 0.2]

    # ── weight_decay ─────────────────────────────────────────────────────────
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

    # label_smoothing is intentionally not searched — see TEXT_DEFAULT_SEARCH_SPACE comment.

    return space
