"""
Tabular-only classification subpackage.

Sub-modules:
    - dataset:            Feature engineering — imputation, encoding, scaling (step 2)
    - train:              Model fitting and artifact saving (step 3)
    - predict:            predict_proba → PredictionResult (step 4)
    - hpo:                Optuna hyperparameter optimization (step 5)
    - tabular_classifier: TabularClassifier wrapper — user-facing API (step 6)

Step 1 (split) is shared from multimodalva.utils.split.
"""

from .tabular_classifier import TabularClassifier

__all__ = ["TabularClassifier"]
