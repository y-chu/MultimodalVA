"""
Search space definitions for tabular (sklearn/tree) HPO.

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

import numpy as np

# ---------------------------------------------------------------------------
# Baseline search spaces per model alias.
# Spec format mirrors text/hpo.py:
#   ("float_log", low, high)   — log-uniform float (best for learning rates)
#   ("float",     low, high)   — uniform float
#   ("int",       low, high)   — uniform int
#   ("categorical", [values])  — discrete choices
#
# These serve as a balanced middle-ground profile. `get_default_search_space()`
# adapts them for small / wide / large tabular problems using X_train.shape.
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACES: dict[str, dict] = {
    "catboost": {
        "iterations":    ("int",         100, 1000),    # number of trees; more = higher capacity but slower training
        "learning_rate": ("float_log",   1e-3, 0.3),   # shrinkage per tree; lower = better generalization but needs more iterations
        "depth":         ("int",         4, 10),        # tree depth; deeper captures more interactions but risks overfitting; >10 rarely helps
        "l2_leaf_reg":   ("float_log",   1e-2, 10.0),  # L2 penalty on leaf weights; higher = smoother predictions, less overfitting
        "boosting_type": ("categorical", ["Ordered", "Plain"]),  # Ordered = CatBoost permutation-based (better for small data); Plain = standard GBDT
    },
    "lightgbm": {
        "n_estimators":      ("int",         100, 1000),           # number of boosting rounds; more = better fit; too many = overfit without early stopping
        "learning_rate":     ("float_log",   1e-3, 0.3),          # smaller = more robust but needs proportionally more n_estimators
        "max_depth":         ("categorical", [-1, 6, 8, 10]),      # -1 = no limit (complexity governed by num_leaves instead)
        "num_leaves":        ("int",         20, 150),             # primary complexity knob for leaf-wise growth; more leaves = finer partitions, higher variance
        "min_child_samples": ("int",         5, 100),              # larger = coarser leaves, less overfitting on rare causes; critical for imbalanced VA data
        "subsample":         ("float",       0.6, 1.0),            # row sampling per tree; useful regularizer and often important on medium/large data
        "colsample_bytree":  ("float",       0.5, 1.0),            # feature sampling per tree; especially important once feature count grows
        "reg_alpha":         ("float_log",   1e-4, 10.0),          # L1 penalty; helps prune noisy splits in wide sparse-ish tabular data
        "reg_lambda":        ("float_log",   1e-4, 20.0),          # L2 penalty; stabilizes leaf weights and probability estimates
    },
    "gbdt": {
        "n_estimators":      ("int",       50, 500),    # number of sequential trees; more = lower bias, risk of overfitting
        "learning_rate":     ("float_log", 1e-3, 0.3), # smaller = each tree contributes less, needs more trees but generalizes better
        "max_depth":         ("int",       2, 8),       # shallow trees (2-4) often best for boosting; deep trees overfit and slow training
        "min_samples_split": ("int",       2, 20),      # larger = fewer splits made, simpler trees, less overfitting to small subgroups
        "min_samples_leaf":  ("int",       1, 10),      # larger = smoother leaf predictions; prevents fitting leaves from very few samples
        "subsample":         ("float",     0.5, 1.0),   # row subsampling per tree; <1 adds randomness like bagging, reduces variance
    },
    "xgboost": {
        "n_estimators":      ("int",       100, 1000),  # number of trees; combine with learning_rate (lower lr → more trees needed)
        "learning_rate":     ("float_log", 1e-3, 0.3), # eta; smaller = more conservative updates, better generalization
        "max_depth":         ("int",       3, 10),      # deeper trees capture more complex patterns but overfit and are slower
        "subsample":         ("float",     0.5, 1.0),   # fraction of rows per tree; <1 introduces stochasticity, reduces overfitting
        "colsample_bytree":  ("float",     0.5, 1.0),   # fraction of features per tree; lower = more diverse trees, similar to random forest effect
        "gamma":             ("float",     0.0, 1.0),   # min loss reduction required to split a node; larger = fewer splits, more conservative trees
        "min_child_weight":  ("int",       1, 10),      # larger = prevents learning from small leaf groups; important for rare cause classes
        "reg_alpha":         ("float_log", 1e-4, 10.0), # L1 regularization; helps when feature space is wide or noisy
        "reg_lambda":        ("float_log", 1e-4, 20.0), # L2 regularization; smooths leaf weights and combats overfit
    },
    "mlp": {
        "hidden_layer_sizes": ("categorical", [(64,), (128,), (128, 64), (256, 128), (256, 128, 64)]),  # network depth/width; wider or deeper = more capacity, more data needed to avoid overfitting
        "activation":         ("categorical", ["tanh", "relu"]),   # relu = sparse, fast, better for deep nets; tanh = smoother, bounded, better for shallow nets
        "learning_rate_init": ("float_log",   1e-4, 1e-1),        # too large = unstable training; too small = slow convergence
        "alpha":              ("float_log",   1e-5, 1e-1),         # L2 weight decay; larger = stronger regularization, smaller weights, better generalization
        "batch_size":         ("categorical", [32, 64, 128, 256]), # smaller = noisier gradients (can escape local minima); larger = faster but may overfit
    },
    "random_forest": {
        "n_estimators":      ("int",         50, 500),                   # more trees = more stable predictions; gains plateau around 200-300
        "max_depth":         ("categorical", [None, 5, 10, 20, 30]),     # None = fully grown (may overfit); limiting depth regularizes leaf predictions
        "min_samples_split": ("int",         2, 20),                     # larger = more conservative node splits; reduces overfitting on small subgroups
        "min_samples_leaf":  ("int",         1, 10),                     # larger = smoother predicted probabilities; prevents leaves from very few samples
        "max_features":      ("categorical", ["sqrt", "log2", 0.5]),     # fewer features per split = more diverse trees, stronger ensemble effect; sqrt is RF default for classification
    },
    "naive_bayes": {
        "var_smoothing": ("float_log", 1e-9, 1e-1),  # variance floor per feature; larger = more shrinkage toward uniform, useful when features have near-zero variance
    },
    "knn": {
        "n_neighbors": ("int",         3, 15),                                    # larger k = smoother boundary, less sensitive to noise; smaller k = complex boundary, overfits
        "weights":     ("categorical", ["uniform", "distance"]),                  # distance weighting emphasizes closest neighbors; useful when local structure matters
        "metric":      ("categorical", ["euclidean", "manhattan", "chebyshev"]),  # euclidean = L2 (sensitive to scale); manhattan = L1 (robust to outliers); chebyshev = max coordinate difference
    },
    "svm": {
        "C":      ("float_log",  1e-3, 1e3),                         # larger C = smaller margin, fits training data tightly (may overfit); smaller C = wider margin, more regularized
        "kernel": ("categorical", ["linear", "rbf", "poly"]),        # linear = fast for high-dim data; rbf = flexible nonlinear boundary; poly = polynomial feature interactions
        "gamma":  ("categorical", ["scale", "auto"]),                 # rbf bandwidth; "scale" = 1/(n_features * X.var()), adapts to feature spread; "auto" = 1/n_features
    },
}

SEARCH_SPACE_PROFILES = {"auto", "small", "balanced", "wide", "large"}


# ---------------------------------------------------------------------------
# Spec builder helpers (keep bounds sane after arithmetic adjustments)
# ---------------------------------------------------------------------------

def _int_spec(low: int | float, high: int | float) -> tuple[str, int, int]:
    low_i = max(1, int(round(low)))
    high_i = max(low_i, int(round(high)))
    return ("int", low_i, high_i)


def _float_spec(low: float, high: float) -> tuple[str, float, float]:
    low_f = float(low)
    high_f = max(low_f, float(high))
    return ("float", low_f, high_f)


def _float_log_spec(low: float, high: float) -> tuple[str, float, float]:
    low_f = max(1e-12, float(low))
    high_f = max(low_f, float(high))
    return ("float_log", low_f, high_f)


def _categorical_spec(values: list) -> tuple[str, list]:
    deduped = list(dict.fromkeys(values))
    return ("categorical", deduped)


# ---------------------------------------------------------------------------
# Class-count tier
# ---------------------------------------------------------------------------

def _class_tier(n_classes: int) -> str:
    """Coarse tier from number of target classes (calibrated for 5–100 VA causes).

    few      ≤ 14  — compact output; standard tree depth and leaf settings
    moderate  15–39  — moderate multi-class complexity
    many     ≥ 40  — large output; needs more trees, leaves, and MLP width
    """
    if n_classes <= 14:
        return "few"
    if n_classes <= 39:
        return "moderate"
    return "many"


# ---------------------------------------------------------------------------
# Class-count adjustments applied on top of the profile space
# ---------------------------------------------------------------------------

def _apply_nclasses_adjustments(
    space: dict,
    model_name: str,
    n_classes: int,
    ct: str,
) -> dict:
    """Adjust a profile-based search space for the number of target classes.

    More classes generally means:
    - Tree models: more trees, more leaves/depth, smaller per-leaf thresholds
      (each class has fewer training samples in a one-vs-rest or softmax setting).
    - MLP: wider/deeper networks to express a larger output space.
    - kNN: slightly more neighbours for stability with more classes.
    - SVM / Naive Bayes: parameters not meaningfully class-count sensitive.
    """
    if ct == "few":
        return space  # balanced profile already calibrated for few classes

    mult = 1.5 if ct == "many" else 1.2  # scale factor: many vs moderate

    if model_name in {"lightgbm", "xgboost", "catboost", "gbdt", "random_forest"}:
        # ── n_estimators / iterations: more classes → more trees ────────────
        for key in ("n_estimators", "iterations"):
            if key in space and space[key][0] == "int":
                lo, hi = space[key][1], space[key][2]
                space[key] = _int_spec(lo, min(int(hi * mult), 3000))

        # ── LightGBM num_leaves: more classes → finer partitions ────────────
        if "num_leaves" in space and space["num_leaves"][0] == "int":
            lo, hi = space["num_leaves"][1], space["num_leaves"][2]
            space["num_leaves"] = _int_spec(lo, min(int(hi * mult), 511))

        # ── min_child_samples / min_child_weight: scale down for many classes
        # With more classes the per-class sample count is lower, so leaf
        # thresholds should be smaller to avoid under-splitting.
        if "min_child_samples" in space and space["min_child_samples"][0] == "int":
            lo, hi = space["min_child_samples"][1], space["min_child_samples"][2]
            shrink = max(1, n_classes // 20)
            space["min_child_samples"] = _int_spec(max(2, lo // shrink), max(lo + 1, hi // shrink))

        if "min_child_weight" in space and space["min_child_weight"][0] == "int":
            lo, hi = space["min_child_weight"][1], space["min_child_weight"][2]
            space["min_child_weight"] = _int_spec(1, max(2, hi // 2))

        # ── CatBoost / GBDT depth: more classes → allow deeper trees ────────
        if "depth" in space and space["depth"][0] == "int":
            lo, hi = space["depth"][1], space["depth"][2]
            space["depth"] = _int_spec(lo, min(hi + (2 if ct == "many" else 1), 12))

        # ── XGBoost / GBDT max_depth (int spec) ─────────────────────────────
        if "max_depth" in space and space["max_depth"][0] == "int":
            lo, hi = space["max_depth"][1], space["max_depth"][2]
            space["max_depth"] = _int_spec(lo, min(hi + (2 if ct == "many" else 1), 12))

        # ── Random Forest max_depth (categorical spec) ──────────────────────
        if "max_depth" in space and space["max_depth"][0] == "categorical" and ct == "many":
            vals = list(space["max_depth"][1])
            numeric = [v for v in vals if v is not None]
            if numeric:
                new_val = min(max(numeric) + 10, 60)
                if new_val not in vals:
                    vals.append(new_val)
                    space["max_depth"] = _categorical_spec(vals)

        # ── min_samples_leaf / min_samples_split (sklearn GBDT / RF) ────────
        if "min_samples_leaf" in space and space["min_samples_leaf"][0] == "int":
            lo, hi = space["min_samples_leaf"][1], space["min_samples_leaf"][2]
            shrink = max(1, n_classes // 20)
            space["min_samples_leaf"] = _int_spec(max(1, lo // shrink), max(lo + 1, hi // shrink))

    elif model_name == "mlp":
        if "hidden_layer_sizes" in space and ct == "many":
            sizes = list(space["hidden_layer_sizes"][1])
            for extra in [(512, 256, 128), (1024, 512)]:
                if extra not in sizes:
                    sizes.append(extra)
            space["hidden_layer_sizes"] = _categorical_spec(sizes)

    elif model_name == "knn":
        if "n_neighbors" in space and space["n_neighbors"][0] == "int" and ct == "many":
            lo, hi = space["n_neighbors"][1], space["n_neighbors"][2]
            space["n_neighbors"] = _int_spec(lo, min(hi + 10, 60))

    return space


# ---------------------------------------------------------------------------
# Sample/feature profile inference
# ---------------------------------------------------------------------------

def infer_search_space_profile(X_train: np.ndarray) -> str:
    """Infer a coarse HPO profile from feature matrix shape.

    Heuristic goals:
    - ``small``: few rows, so bias toward stronger regularization / smaller models.
    - ``wide``: many features or low samples-per-feature ratio, so search stronger
      feature subsampling / regularization and avoid underpowered defaults.
    - ``large``: enough rows to justify broader-capacity spaces.
    - ``balanced``: reasonable middle ground for the common case.
    """
    n_samples, n_features = X_train.shape
    samples_per_feature = n_samples / max(n_features, 1)

    if n_features >= 200 or samples_per_feature < 8:
        return "wide"
    if n_samples <= 1500:
        return "small"
    if n_samples >= 20000 and samples_per_feature >= 20:
        return "large"
    return "balanced"


def _resolve_search_space_profile(
    X_train: np.ndarray,
    search_space_profile: str,
) -> str:
    if search_space_profile not in SEARCH_SPACE_PROFILES:
        raise ValueError(
            f"Unknown search_space_profile {search_space_profile!r}. "
            f"Valid values: {sorted(SEARCH_SPACE_PROFILES)}"
        )
    return infer_search_space_profile(X_train) if search_space_profile == "auto" else search_space_profile


# ---------------------------------------------------------------------------
# Random Forest max_features choices (data-shape aware)
# ---------------------------------------------------------------------------

def _max_features_choices(n_features: int, profile: str) -> list:
    choices = ["sqrt", "log2"]
    if profile == "wide":
        choices.extend([0.1, 0.2, 0.5])
    elif n_features >= 100:
        choices.extend([0.2, 0.5])
    else:
        choices.extend([0.5, None])
    return list(dict.fromkeys(choices))


# ---------------------------------------------------------------------------
# Profile-based space builder (no class adjustments)
# ---------------------------------------------------------------------------

def _build_profile_space(
    model_name: str,
    X_train: np.ndarray,
    search_space_profile: str = "auto",
) -> dict:
    """Build a sample/feature-count-aware search space (no class-count adjustments).

    Internal helper — call ``get_default_search_space()`` from outside code.
    """
    profile = _resolve_search_space_profile(X_train, search_space_profile)
    n_samples, n_features = X_train.shape
    max_neighbor_hi = min(50, max(10, n_samples // 20))
    min_leaf_hi = min(20, max(5, n_samples // 50))
    min_split_hi = min(50, max(10, n_samples // 25))

    if model_name == "lightgbm":
        if profile == "small":
            return {
                "n_estimators": _int_spec(100, 700),
                "learning_rate": _float_log_spec(5e-3, 0.15),
                "max_depth": _categorical_spec([-1, 4, 6, 8]),
                "num_leaves": _int_spec(15, 96),
                "min_child_samples": _int_spec(10, min(80, max(20, n_samples // 10))),
                "subsample": _float_spec(0.7, 1.0),
                "colsample_bytree": _float_spec(0.5 if n_features >= 40 else 0.7, 1.0),
                "reg_alpha": _float_log_spec(1e-4, 10.0),
                "reg_lambda": _float_log_spec(1e-4, 20.0),
            }
        if profile == "wide":
            return {
                "n_estimators": _int_spec(150, 1200),
                "learning_rate": _float_log_spec(5e-3, 0.15),
                "max_depth": _categorical_spec([-1, 4, 6, 8, 10]),
                "num_leaves": _int_spec(31, 127),
                "min_child_samples": _int_spec(10, min(150, max(40, n_samples // 20))),
                "subsample": _float_spec(0.6, 1.0),
                "colsample_bytree": _float_spec(0.3, 0.9),
                "reg_alpha": _float_log_spec(1e-4, 20.0),
                "reg_lambda": _float_log_spec(1e-4, 30.0),
            }
        if profile == "large":
            return {
                "n_estimators": _int_spec(200, 2000),
                "learning_rate": _float_log_spec(1e-2, 0.2),
                "max_depth": _categorical_spec([-1, 6, 8, 10, 12]),
                "num_leaves": _int_spec(31, 255),
                "min_child_samples": _int_spec(5, min(200, max(60, n_samples // 50))),
                "subsample": _float_spec(0.6, 1.0),
                "colsample_bytree": _float_spec(0.5, 1.0),
                "reg_alpha": _float_log_spec(1e-4, 10.0),
                "reg_lambda": _float_log_spec(1e-4, 20.0),
            }
        return {
            "n_estimators": _int_spec(150, 1500),
            "learning_rate": _float_log_spec(5e-3, 0.2),
            "max_depth": _categorical_spec([-1, 5, 7, 9, 12]),
            "num_leaves": _int_spec(20, 160),
            "min_child_samples": _int_spec(5, min(120, max(30, n_samples // 30))),
            "subsample": _float_spec(0.6, 1.0),
            "colsample_bytree": _float_spec(0.5, 1.0),
            "reg_alpha": _float_log_spec(1e-4, 10.0),
            "reg_lambda": _float_log_spec(1e-4, 20.0),
        }

    if model_name == "xgboost":
        if profile == "small":
            return {
                "n_estimators": _int_spec(100, 700),
                "learning_rate": _float_log_spec(5e-3, 0.15),
                "max_depth": _int_spec(3, 8),
                "subsample": _float_spec(0.7, 1.0),
                "colsample_bytree": _float_spec(0.5, 1.0),
                "gamma": _float_spec(0.0, 3.0),
                "min_child_weight": _int_spec(1, 12),
                "reg_alpha": _float_log_spec(1e-4, 10.0),
                "reg_lambda": _float_log_spec(1e-4, 20.0),
            }
        if profile == "wide":
            return {
                "n_estimators": _int_spec(150, 1200),
                "learning_rate": _float_log_spec(5e-3, 0.15),
                "max_depth": _int_spec(3, 7),
                "subsample": _float_spec(0.6, 1.0),
                "colsample_bytree": _float_spec(0.3, 0.9),
                "gamma": _float_spec(0.0, 5.0),
                "min_child_weight": _int_spec(2, 16),
                "reg_alpha": _float_log_spec(1e-4, 20.0),
                "reg_lambda": _float_log_spec(1e-4, 30.0),
            }
        if profile == "large":
            return {
                "n_estimators": _int_spec(200, 1500),
                "learning_rate": _float_log_spec(1e-2, 0.2),
                "max_depth": _int_spec(3, 10),
                "subsample": _float_spec(0.6, 1.0),
                "colsample_bytree": _float_spec(0.5, 1.0),
                "gamma": _float_spec(0.0, 3.0),
                "min_child_weight": _int_spec(1, 12),
                "reg_alpha": _float_log_spec(1e-4, 10.0),
                "reg_lambda": _float_log_spec(1e-4, 20.0),
            }
        return {
            "n_estimators": _int_spec(150, 1200),
            "learning_rate": _float_log_spec(5e-3, 0.2),
            "max_depth": _int_spec(3, 9),
            "subsample": _float_spec(0.6, 1.0),
            "colsample_bytree": _float_spec(0.5, 1.0),
            "gamma": _float_spec(0.0, 3.0),
            "min_child_weight": _int_spec(1, 12),
            "reg_alpha": _float_log_spec(1e-4, 10.0),
            "reg_lambda": _float_log_spec(1e-4, 20.0),
        }

    if model_name == "catboost":
        if profile == "small":
            return {
                "iterations": _int_spec(100, 800),
                "learning_rate": _float_log_spec(5e-3, 0.2),
                "depth": _int_spec(4, 8),
                "l2_leaf_reg": _float_log_spec(1e-2, 20.0),
                "boosting_type": _categorical_spec(["Ordered", "Plain"]),
            }
        if profile == "large":
            return {
                "iterations": _int_spec(200, 1500),
                "learning_rate": _float_log_spec(5e-3, 0.2),
                "depth": _int_spec(4, 10),
                "l2_leaf_reg": _float_log_spec(1e-2, 20.0),
                "boosting_type": _categorical_spec(["Ordered", "Plain"]),
            }
        if profile == "wide":
            return {
                "iterations": _int_spec(150, 1200),
                "learning_rate": _float_log_spec(5e-3, 0.15),
                "depth": _int_spec(4, 8),
                "l2_leaf_reg": _float_log_spec(1e-2, 30.0),
                "boosting_type": _categorical_spec(["Ordered", "Plain"]),
            }
        return dict(DEFAULT_SEARCH_SPACES["catboost"])

    if model_name == "gbdt":
        if profile == "small":
            return {
                "n_estimators": _int_spec(50, 400),
                "learning_rate": _float_log_spec(5e-3, 0.2),
                "max_depth": _int_spec(2, 5),
                "min_samples_split": _int_spec(2, min_split_hi),
                "min_samples_leaf": _int_spec(1, min_leaf_hi),
                "subsample": _float_spec(0.7, 1.0),
            }
        if profile == "large":
            return {
                "n_estimators": _int_spec(100, 1000),
                "learning_rate": _float_log_spec(5e-3, 0.2),
                "max_depth": _int_spec(2, 6),
                "min_samples_split": _int_spec(2, min_split_hi),
                "min_samples_leaf": _int_spec(1, min_leaf_hi),
                "subsample": _float_spec(0.5, 1.0),
            }
        if profile == "wide":
            return {
                "n_estimators": _int_spec(100, 700),
                "learning_rate": _float_log_spec(5e-3, 0.15),
                "max_depth": _int_spec(2, 5),
                "min_samples_split": _int_spec(2, min_split_hi),
                "min_samples_leaf": _int_spec(1, min_leaf_hi),
                "subsample": _float_spec(0.5, 1.0),
            }
        return {
            "n_estimators": _int_spec(50, 700),
            "learning_rate": _float_log_spec(5e-3, 0.2),
            "max_depth": _int_spec(2, 6),
            "min_samples_split": _int_spec(2, min_split_hi),
            "min_samples_leaf": _int_spec(1, min_leaf_hi),
            "subsample": _float_spec(0.5, 1.0),
        }

    if model_name == "random_forest":
        if profile == "small":
            return {
                "n_estimators": _int_spec(100, 400),
                "max_depth": _categorical_spec([None, 5, 10, 20]),
                "min_samples_split": _int_spec(2, min_split_hi),
                "min_samples_leaf": _int_spec(1, min_leaf_hi),
                "max_features": _categorical_spec(_max_features_choices(n_features, profile)),
            }
        if profile == "large":
            return {
                "n_estimators": _int_spec(200, 1000),
                "max_depth": _categorical_spec([None, 10, 20, 30, 40]),
                "min_samples_split": _int_spec(2, min_split_hi),
                "min_samples_leaf": _int_spec(1, min(30, max(10, n_samples // 80))),
                "max_features": _categorical_spec(_max_features_choices(n_features, profile)),
            }
        if profile == "wide":
            return {
                "n_estimators": _int_spec(150, 800),
                "max_depth": _categorical_spec([None, 10, 20, 30]),
                "min_samples_split": _int_spec(2, min_split_hi),
                "min_samples_leaf": _int_spec(1, min_leaf_hi),
                "max_features": _categorical_spec(_max_features_choices(n_features, profile)),
            }
        return {
            "n_estimators": _int_spec(100, 800),
            "max_depth": _categorical_spec([None, 5, 10, 20, 30]),
            "min_samples_split": _int_spec(2, min_split_hi),
            "min_samples_leaf": _int_spec(1, min_leaf_hi),
            "max_features": _categorical_spec(_max_features_choices(n_features, profile)),
        }

    if model_name == "mlp":
        if profile == "small":
            hidden_sizes = [(64,), (128,), (128, 64), (256, 128)]
            batch_sizes = [16, 32, 64, 128]
        elif profile == "wide":
            hidden_sizes = [(128,), (256,), (256, 128), (512, 256), (256, 128, 64)]
            batch_sizes = [32, 64, 128, 256]
        elif profile == "large":
            hidden_sizes = [(128,), (256,), (256, 128), (512, 256), (512, 256, 128)]
            batch_sizes = [64, 128, 256, 512]
        else:
            hidden_sizes = [(64,), (128,), (128, 64), (256, 128), (256, 128, 64), (512, 256)]
            batch_sizes = [32, 64, 128, 256]
        return {
            "hidden_layer_sizes": _categorical_spec(hidden_sizes),
            "activation": _categorical_spec(["tanh", "relu"]),
            "learning_rate_init": _float_log_spec(1e-4, 5e-2),
            "alpha": _float_log_spec(1e-6, 1e-1),
            "batch_size": _categorical_spec(batch_sizes),
        }

    if model_name == "knn":
        return {
            "n_neighbors": _int_spec(3, max_neighbor_hi),
            "weights": _categorical_spec(["uniform", "distance"]),
            "metric": _categorical_spec(["euclidean", "manhattan", "chebyshev"]),
        }

    if model_name == "svm":
        kernels = ["linear", "rbf"] if profile in {"wide", "large"} else ["linear", "rbf", "poly"]
        return {
            "C": _float_log_spec(1e-3, 1e3),
            "kernel": _categorical_spec(kernels),
            "gamma": _categorical_spec(["scale", "auto"]),
        }

    return dict(DEFAULT_SEARCH_SPACES.get(model_name, {}))


# ---------------------------------------------------------------------------
# Public entry point  (n_samples × n_features × n_classes)
# ---------------------------------------------------------------------------

def get_default_search_space(
    model_name: str,
    X_train: np.ndarray,
    search_space_profile: str = "auto",
    n_classes: int = 1,
) -> dict:
    """Return the default HPO search space adapted to data shape and label count.

    Builds a profile-based space from ``X_train.shape`` (n_samples, n_features)
    then applies ``_apply_nclasses_adjustments()`` to scale tree depth/leaves/
    estimators and MLP width for moderate or many-class problems.

    The returned space is a baseline.  Callers merge explicit ``search_space``
    overrides on top — caller values always win.
    """
    space = _build_profile_space(model_name, X_train, search_space_profile)
    ct = _class_tier(n_classes)
    return _apply_nclasses_adjustments(space, model_name, n_classes, ct)
