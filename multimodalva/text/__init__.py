"""
Text-only classification subpackage.

Sub-modules:
    - split:      Train-test split (step 1)
    - dataset:    Tokenized HuggingFace Dataset preparation (step 2)
    - train:      BERT fine-tuning and artifact saving (step 3)
    - predict:    Label and probability prediction (step 4)
    - hpo:        Optuna hyperparameter optimization (step 5)
    - text_classifier: TextClassifier wrapper — user-facing API (step 6)
"""

from .text_classifier import TextClassifier

__all__ = ["TextClassifier"]
