"""
Ensemble classification subpackage — combines text and tabular modalities.

Four fusion strategies are available, ranging from shallow data combination
to deep learnable ensembles:

    1. Data-level fusion    (DataFusionClassifier)
       Convert tabular features to natural language, concatenate with the free-text
       narrative, and fine-tune a long-context language model (Longformer, BigBird,
       Clinical Longformer) on the combined text.  One unified model, no explicit
       ensemble; the LM learns cross-modal interactions in the attention layers.

    2. Feature-level fusion (FeatureFusionClassifier)
       Feed raw text and tabular features jointly to AutoGluon's MultiModalPredictor
       (AutoMM), which handles tokenisation, embedding, and fusion internally.
       Best when you want a strong automated baseline with minimal setup.

    3. Decision-level fusion — soft voting  (SoftVotingClassifier)
       Train a set of text base-models and a set of tabular base-models independently
       on the full training set.  At inference time, average (or weight-average) the
       predicted probability distributions from all base-models and take the argmax.
       Fast and interpretable; no meta-learner training required.

    4. Decision-level fusion — stacking / super learner  (StackingClassifier)
       Train base-models with k-fold cross-validation and collect out-of-fold (OOF)
       probability predictions.  Concatenate OOF predictions into a meta-feature matrix
       and train a meta-learner (e.g. logistic regression, LightGBM) to map them to
       final class labels.  Final base-models are retrained on the full training set;
       test predictions pass through the same meta-learner.

Usage quick-start::

    # Option A — use the unified wrapper (recommended for most users)
    from multimodalva.ensemble import EnsembleClassifier

    clf = EnsembleClassifier(method="soft_voting", output_dir="runs/ensemble")
    results = clf.run(df, text_col="narrative", feature_cols=[...], label_col="cause")

    # Option B — use a specific classifier directly (more control)
    from multimodalva.ensemble import SoftVotingClassifier

    clf = SoftVotingClassifier(
        text_models=[{"model_name": "bert-base-uncased", "hyperparams": {...}}],
        tabular_models=[{"model_name": "lightgbm", "hyperparams": {...}}],
        output_dir="runs/ensemble/voting",
    )
    results = clf.run(df, text_col="narrative", feature_cols=[...], label_col="cause")
"""

from .data_fusion import DataFusionClassifier
from .feature_fusion import FeatureFusionClassifier
from .voting import SoftVotingClassifier
from .stacking import StackingClassifier
from .ensemble_classifier import EnsembleClassifier

__all__ = [
    "EnsembleClassifier",
    "DataFusionClassifier",
    "FeatureFusionClassifier",
    "SoftVotingClassifier",
    "StackingClassifier",
]
