"""
MultimodalVA: Cause of Death Classification using Text and Tabular Data.

Core subpackages:
    - text: Text-only classification
    - tabular: Tabular-only classification
    - ensemble: Ensemble model combining text and tabular
    - results: Training diagnostics and prediction visualization
"""

__version__ = "0.1.0"

# Synthetic example datasets — preview the input shape the package expects:
#   from multimodalva import data, list_datasets
#   list_datasets()            # {name: description}
#   df = data("va_sample")     # a ready-to-use DataFrame
# Lightweight (pandas/numpy only) — importing this does not pull the transformer stack.
from .datasets import data, list_datasets  # noqa: E402

__all__ = ["data", "list_datasets"]
