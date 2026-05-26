"""Shared helpers for the small demo scripts in ``tests/``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CAUSES = [
    "HIV/AIDS",
    "Pneumonia",
    "Traffic / transport accident",
    "Diabetes",
]


def _build_va_row(cause: str, rng: np.random.Generator) -> dict:
    """Build a single synthetic VA record for ``cause`` (shared schema)."""
    age = int(rng.integers(18, 85))
    sex = rng.choice(["female", "male"])
    fever = int(cause in {"HIV/AIDS", "Pneumonia"} or rng.random() < 0.08)
    cough = int(cause == "Pneumonia" or rng.random() < 0.10)
    weight_loss = int(cause == "HIV/AIDS" or rng.random() < 0.06)
    injury = int(cause == "Traffic / transport accident" or rng.random() < 0.03)
    polyuria = int(cause == "Diabetes" or rng.random() < 0.05)
    chest_pain = int(cause == "Traffic / transport accident" or rng.random() < 0.07)
    symptom_days = int(rng.integers(1, 40))

    if cause == "HIV/AIDS":
        narrative = (
            f"{sex} adult with prolonged fever, weight loss, weakness, and recurrent illness "
            f"for {symptom_days} days."
        )
    elif cause == "Pneumonia":
        narrative = (
            f"{sex} adult with cough, fast breathing, fever, and chest symptoms "
            f"for {symptom_days} days."
        )
    elif cause == "Traffic / transport accident":
        narrative = (
            f"{sex} adult involved in a road traffic injury with sudden collapse and chest pain."
        )
    else:
        narrative = (
            f"{sex} adult with excessive urination, thirst, weakness, and gradual decline "
            f"over {symptom_days} days."
        )

    return {
        "narrative": narrative,
        "age": age,
        "sex": sex,
        "fever": fever,
        "cough": cough,
        "weight_loss": weight_loss,
        "injury": injury,
        "polyuria": polyuria,
        "chest_pain": chest_pain,
        "symptom_days": symptom_days,
        "cause": cause,
    }


def make_demo_va_df(n_samples: int = 120, seed: int = 7) -> pd.DataFrame:
    """Create a small multimodal verbal-autopsy-like DataFrame."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(CAUSES, size=n_samples, p=[0.35, 0.25, 0.20, 0.20])
    return pd.DataFrame([_build_va_row(cause, rng) for cause in labels])


def make_sample_va_df(n_per_class: int = 4, seed: int = 11) -> pd.DataFrame:
    """Tiny, class-balanced synthetic VA sample for smoke tests.

    Unlike :func:`make_demo_va_df` (random, probability-weighted), this guarantees
    exactly ``n_per_class`` rows for every cause, so even a very small sample
    stratifies cleanly into a train/test split. Same column schema as the package
    expects:

        narrative   free-text VA narrative (text + data-fusion pipelines)
        age, sex    demographics
        fever ...   binary (0/1) symptom indicators (tabular features)
        symptom_days numeric feature
        cause       the label (cause of death)
    """
    rng = np.random.default_rng(seed)
    rows = [_build_va_row(cause, rng) for cause in CAUSES for _ in range(n_per_class)]
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def feature_cols() -> list[str]:
    """Return the tabular feature columns for the synthetic demo dataset."""
    return [
        "age",
        "sex",
        "fever",
        "cough",
        "weight_loss",
        "injury",
        "polyuria",
        "chest_pain",
        "symptom_days",
    ]


def resolve_feature_cols(
    df: pd.DataFrame,
    label_col: str,
    text_col: str | None = None,
    exclude_cols: list[str] | None = None,
    features_arg: str | None = None,
) -> list[str]:
    """Return the feature column list for training.

    Default rule (no ``features_arg``):
        features = ALL columns EXCEPT label_col + text_col + exclude_cols
        i.e. every column that is not the label, not the text narrative,
        and not explicitly listed in --id-col is treated as a feature.

    ``features_arg`` overrides this (advanced use only):
    - ``"age,sex,fever"``  → explicit comma-separated names
    - ``"features.txt"``   → path to a text file (one name per line or comma-sep)
    - ``"all"``            → same as default (kept for backward compat)
    """
    # Columns to never use as features:
    #   - label_col  (the target variable)
    #   - text_col   (raw narrative text — handled separately by the text pipeline)
    #   - exclude_cols  (IDs, dates, or any other non-predictive columns passed via --id-col)
    exclude: set[str] = {label_col}
    if text_col:
        exclude.add(text_col)
    if exclude_cols:
        exclude.update(exclude_cols)

    if features_arg is None or features_arg == "all":
        # Default: every remaining column is a feature.
        return [c for c in df.columns if c not in exclude]

    # File path: --features features.txt
    import os
    if os.path.isfile(features_arg):
        content = Path(features_arg).read_text().strip()
        if "\n" in content:
            return [ln.strip() for ln in content.splitlines() if ln.strip()]
        return [c.strip() for c in content.split(",") if c.strip()]

    # Explicit comma-separated list
    return [f.strip() for f in features_arg.split(",") if f.strip()]


def load_data(
    data: str,
    label_col: str,
    text_col: str | None = None,
    exclude_cols: list[str] | None = None,
    features_arg: str | None = None,
    n_samples: int = 120,
    seed: int = 7,
) -> tuple[pd.DataFrame, list[str]]:
    """Load data and return ``(df, feat_cols)``.

    ``data="demo"`` generates a synthetic VA dataset; any other value is
    treated as a path to a CSV file.

    Default feature selection (when ``features_arg`` is not given):
        All columns except ``label_col``, ``text_col``, and ``exclude_cols``.
        This is the recommended workflow for real data — just specify
        ``--label`` and optionally ``--id-col``.

    ``features_arg`` overrides the default (rarely needed):
        - ``"age,sex,fever"``  explicit comma-separated names
        - ``"features.txt"``   path to a text file (one name per line or comma-sep)
    """
    if data == "demo":
        df = make_demo_va_df(n_samples=n_samples, seed=seed)
        # For demo data with no overrides, return the curated default list
        # so the demo still works even if the DataFrame has extra columns.
        if features_arg is None and not exclude_cols:
            return df, feature_cols()
        return df, resolve_feature_cols(df, label_col, text_col, exclude_cols, features_arg)

    df = pd.read_csv(data)
    return df, resolve_feature_cols(df, label_col, text_col, exclude_cols, features_arg)


def parse_exclude_cols(exclude_cols_arg: str | None) -> list[str] | None:
    """Parse ``--exclude-cols`` CLI arg (comma-separated) into a list or ``None``."""
    if exclude_cols_arg is None:
        return None
    return [c.strip() for c in exclude_cols_arg.split(",") if c.strip()]


def parse_cat_cols(cat_cols_arg: str | None, data: str) -> list[str] | None:
    """Parse ``--cat-cols`` CLI arg into a list or ``None``."""
    if cat_cols_arg is not None:
        return [c.strip() for c in cat_cols_arg.split(",")]
    if data == "demo":
        return ["sex"]
    return None


def print_prediction_summary(result, title: str) -> None:
    """Print a short, consistent summary for a PredictionResult."""
    top1 = result.top1
    accuracy = (top1["true_label"] == top1["predicted_label"]).mean()
    print(f"\n{title}")
    print(f"Top-1 accuracy: {accuracy:.3f}")
    print(top1.head().to_string(index=False))


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
