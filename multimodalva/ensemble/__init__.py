"""Ensemble classification subpackage with lazy imports.

This keeps lightweight entry points, such as synthetic voting demos, from
pulling in transformer and AutoMM dependencies until they are actually used.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "EnsembleClassifier",
    "DataFusionClassifier",
    "FeatureFusionClassifier",
    "SoftVotingClassifier",
    "StackingClassifier",
    "qdesc_feature_overlap",
    # Voting over results that already exist — the no-training path. Exported
    # here because it is a first-class way to use the package, not an internal
    # detail of the voting module.
    "vote_from_results",
    "soft_vote",
]


_EXPORTS = {
    "EnsembleClassifier": ("multimodalva.ensemble.ensemble_classifier", "EnsembleClassifier"),
    "DataFusionClassifier": ("multimodalva.ensemble.data_fusion_classifier", "DataFusionClassifier"),
    "FeatureFusionClassifier": ("multimodalva.ensemble.feature_fusion", "FeatureFusionClassifier"),
    "SoftVotingClassifier": ("multimodalva.ensemble.voting", "SoftVotingClassifier"),
    "StackingClassifier": ("multimodalva.ensemble.stacking", "StackingClassifier"),
    "qdesc_feature_overlap": ("multimodalva.ensemble.data_fusion", "qdesc_feature_overlap"),
    "vote_from_results": ("multimodalva.ensemble.voting", "vote_from_results"),
    "soft_vote": ("multimodalva.ensemble.voting", "soft_vote"),
}


def __getattr__(name: str):
    """Resolve exported symbols on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the lazy exports in tab completion and help()."""
    return sorted(set(globals()) | set(__all__))
