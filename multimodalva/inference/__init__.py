"""Predict on new data with a model that is already trained.

``predict_from_pretrained()`` loads a MultimodalVA run, a Hugging Face Hub
model, or a compatible model trained elsewhere, and returns the usual
``PredictionResult``. Importing this package does not import torch; the text
backend loads it only when a text model is used.
"""

from .api import predict_from_pretrained
from .checks import PretrainedChecks
from .ensemble_api import predict_ensemble_from_pretrained

__all__ = ["predict_from_pretrained", "predict_ensemble_from_pretrained",
           "PretrainedChecks"]
