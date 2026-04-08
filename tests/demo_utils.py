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


def make_demo_va_df(n_samples: int = 120, seed: int = 7) -> pd.DataFrame:
    """Create a small multimodal verbal-autopsy-like DataFrame."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(CAUSES, size=n_samples, p=[0.35, 0.25, 0.20, 0.20])

    rows: list[dict] = []
    for cause in labels:
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

        rows.append(
            {
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
        )

    return pd.DataFrame(rows)


def feature_cols() -> list[str]:
    """Return the tabular feature columns used throughout the demos."""
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
