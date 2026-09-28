"""
Unified entry point for all four ensemble strategies.

EnsembleClassifier dispatches to the appropriate strategy class based on the
``method`` argument, providing a single consistent interface for all fusion
approaches.  Power users who need fine-grained control can import and instantiate
the strategy classes directly.

Strategy dispatch:
    "data_fusion"    → DataFusionClassifier    (ensemble/data_fusion.py)
    "feature_fusion" → FeatureFusionClassifier (ensemble/feature_fusion.py)
    "soft_voting"    → SoftVotingClassifier    (ensemble/voting.py)
    "stacking"       → StackingClassifier      (ensemble/stacking.py)

End-to-end usage::

    from multimodalva.ensemble import EnsembleClassifier

    # --- Data-level fusion ---
    clf = EnsembleClassifier(method="data_fusion", output_dir="runs/ensemble")
    results = clf.run(
        df=df, text_col="narrative", feature_cols=[...], label_col="cause",
        model_name="allenai/longformer-base-4096",
    )

    # --- Soft voting ---
    clf = EnsembleClassifier(
        method="soft_voting",
        output_dir="runs/ensemble",
        text_models=[{"model_name": "bert-base-uncased", "hyperparams": {...}}],
        tabular_models=[{"model_name": "lightgbm", "hyperparams": {...}}],
    )
    results = clf.run(
        df=df, text_col="narrative", feature_cols=[...], label_col="cause",
    )

Stage-wise stacking (cross-session)::

    # Session 1 — OOF loop
    clf = EnsembleClassifier(
        method="stacking",
        output_dir="runs/ensemble",
        text_models=[...],
        tabular_models=[...],
        n_folds=5,
    )
    clf.train_base_models(
        df=df, label_col="cause", text_col="narrative", feature_cols=[...],
    )

    # Session 2 — meta-learner selection
    clf = EnsembleClassifier(
        method="stacking",
        output_dir="runs/ensemble",
        text_models=[...],
        tabular_models=[...],
    )
    clf.train_meta_learner_stage(meta_cv_folds=3)

    # Session 3 — test prediction
    clf = EnsembleClassifier(
        method="stacking",
        output_dir="runs/ensemble",
        text_models=[...],
        tabular_models=[...],
    )
    predictions = clf.predict_test()

    # Stage 2/3 alternatives on the same Stage 1 output
    clf.compare_combiners_stage(n_combiner_folds=5) # nested-CV table of combiners
    clf.train_ensemble_selection_stage(ensemble_size=100)
    predictions = clf.predict_test_ensemble_selection()

    # Or end-to-end: several combiners from one Stage 1, the first is the main one
    clf.run(df=df, ..., combiner=["meta_learner", "ensemble_selection"])
    clf.run(df=df, ..., combiner="best")   # chosen on training data only

Accessing strategy-specific attributes::

    clf.classifier        # underlying strategy instance
    clf.oof_meta_X        # shorthand — delegated via __getattr__
    clf.meta_scores       # shorthand — delegated via __getattr__
    clf.base_predictions  # shorthand — delegated via __getattr__
    clf.study             # shorthand — delegated via __getattr__
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .data_fusion_classifier import DataFusionClassifier
from .feature_fusion import FeatureFusionClassifier
from .voting import SoftVotingClassifier
from .stacking import StackingClassifier

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = {
    "data_fusion":    DataFusionClassifier,
    "feature_fusion": FeatureFusionClassifier,
    "soft_voting":    SoftVotingClassifier,
    "stacking":       StackingClassifier,
}


class EnsembleClassifier:
    """Unified wrapper for all four multimodal ensemble strategies.

    Instantiate with a ``method`` string and any constructor keyword arguments
    for the underlying strategy class.  Call ``run()`` for end-to-end execution,
    or use the stage methods (``train_base_models``, ``train_meta_learner_stage``,
    ``predict_test``) for cross-session stacking workflows.

    Strategy-specific attributes (e.g. ``oof_meta_X``, ``meta_scores``) are
    accessible directly on the wrapper via transparent delegation to
    ``self.classifier``.

    Supported methods
    -----------------
    +------------------+-------------------------------------------+
    | method           | description                               |
    +==================+===========================================+
    | "data_fusion"    | tabular→text concat + LM fine-tuning      |
    | "feature_fusion" | AutoGluon AutoMM text+tabular             |
    | "soft_voting"    | independent base models + soft vote       |
    | "stacking"       | OOF k-fold + Stage 2 combiner: meta-      |
    |                  | learner, class-aware voter or ensemble    |
    |                  | selection, simple average (``combiner=``, |
    |                  | a list, or ``"best"``)                    |
    +------------------+-------------------------------------------+
    """

    def __init__(
        self,
        method: str,
        output_dir: str | Path = "runs/ensemble",
        **kwargs,
    ):
        """Initialise EnsembleClassifier and construct the strategy instance.

        Args:
            method:     Ensemble strategy.  One of: "data_fusion", "feature_fusion",
                        "soft_voting", "stacking".
            output_dir: Root directory passed to the strategy class.  The strategy
                        instance receives ``output_dir / method`` as its own root.
            **kwargs:   Keyword arguments forwarded verbatim to the strategy class
                        constructor.

                        data_fusion:
                            model_name (str)  — LM name or path; default Longformer

                        feature_fusion:
                            preset (str)  — AutoMM quality preset; default "best_quality"
                            fusion_strategy (str) — "default" | "concat" | "attention" | "attention_ft" | "text_only" | "tabular_only"; default "default"

                        soft_voting:
                            text_models    (list[dict])  — required
                            tabular_models (list[dict])  — required
                            weights        (list[float]) — optional per-model weights

                        stacking:
                            text_models       (list[dict])  — required
                            tabular_models    (list[dict])  — required
                            meta_learners     (list[dict] | dict) — default logistic regression
                            meta_select_metric (str)        — default "f1_macro"
                            n_folds           (int)         — default 5
                            resume            (bool)        — default True

        Raises:
            ValueError: If method is not one of the supported strategies.
        """
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: {list(SUPPORTED_METHODS)}."
            )

        self.method = method
        self.output_dir = Path(output_dir)

        # Instantiate the underlying strategy
        self.classifier = SUPPORTED_METHODS[method](
            output_dir=self.output_dir / method,
            **kwargs,
        )

        # Populated after run() / predict_test()
        self.predictions = None
        self.results: dict | None = None

    # ------------------------------------------------------------------
    # Transparent delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute lookup to the underlying strategy instance.

        Allows direct access to strategy-specific attributes without going
        through ``self.classifier``::

            clf.oof_meta_X   # same as clf.classifier.oof_meta_X  (stacking)
            clf.meta_scores  # same as clf.classifier.meta_scores (stacking)
            clf.study        # same as clf.classifier.study        (all strategies)

        Called only when the attribute is not found on ``EnsembleClassifier``
        itself, so ``clf.method``, ``clf.classifier``, and ``clf.run`` always
        resolve to the wrapper's own attrs/methods first.
        """
        try:
            classifier = object.__getattribute__(self, "classifier")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        try:
            return getattr(classifier, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object (method='{self.method}') "
                f"has no attribute '{name}'"
            )

    # ------------------------------------------------------------------
    # End-to-end pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        text_col: str,
        feature_cols: list[str],
        label_col: str,
        **kwargs,
    ) -> dict:
        """Run the chosen ensemble pipeline end-to-end.

        All keyword arguments are forwarded verbatim to the underlying strategy's
        ``run()`` method.  See the strategy class docstrings for the full list of
        accepted arguments.

        Common args (accepted by all strategies):
            test_size      (float) — test fraction; default 0.2
            split_seed     (int)   — row-partitioning seed; default 42
            train_seed     (int)   — model-stochasticity seed; default 42
            deterministic  (bool)  — bit-for-bit kernels; default False
            stratify       (bool)  — stratified split; default True
            top_k          (int)   — top-K classes in topk output; default 3

        Strategy-specific args (examples):
            data_fusion:    max_length, use_lora, hyperparams, ...
            feature_fusion: time_limit, hyperparams, hpo_scheduler, ...
            soft_voting:    n_jobs, use_gpu, batch_size, weights, ...
            stacking:       n_jobs, use_gpu, encode_categoricals,
                            save_fold_models, cleanup_fold_files,
                            meta_cv_folds, combiner (str, list or
                            "best"), n_combiner_folds, oof_from,
                            class_voter_kwargs, ensemble_selection_kwargs, ...

        Args:
            df:           Input DataFrame.
            text_col:     Column containing the free-text narrative.
            feature_cols: Tabular feature columns.
            label_col:    Column containing cause-of-death labels.
            **kwargs:     Strategy-specific keyword arguments.

        Returns:
            Strategy-specific results dict.  All strategies include at minimum:
                "predictions":  PredictionResult(top1, full, topk, id2label)
                "label2id":     label encoding map
                "id2label":     reverse label encoding map
                "output_dir":   Path to the root output directory

        Raises:
            NotImplementedError: Forwarded from the strategy's run() until implemented.
        """
        logger.info("EnsembleClassifier.run() — method='%s'", self.method)
        self.results = self.classifier.run(
            df=df,
            text_col=text_col,
            feature_cols=feature_cols,
            label_col=label_col,
            **kwargs,
        )
        self.predictions = self.results.get("predictions")
        return self.results

    # ------------------------------------------------------------------
    # Stage-wise stacking API
    # ------------------------------------------------------------------

    def train_base_models(
        self,
        df: pd.DataFrame,
        label_col: str,
        text_col: str | None = None,
        feature_cols: list[str] | None = None,
        **kwargs,
    ) -> dict:
        """Stage 1 of stacking: split → (HPO) → OOF loop → final model training.

        Requires ``method="stacking"``.  Delegates to
        :meth:`~multimodalva.ensemble.stacking.StackingClassifier.train_base_models`.

        After completion all OOF artifacts and final base models are saved under
        ``output_dir/stacking/``.  Stage 2 (``train_meta_learner_stage``) can be
        run in a separate Python session by constructing a new ``EnsembleClassifier``
        with the same ``output_dir`` and model specs.

        Args:
            df:           Full input DataFrame (split performed internally).
            label_col:    Label column name.
            text_col:     Text/narrative column name (required for text models).
            feature_cols: Tabular feature column names (required for tabular models).
            **kwargs:     Forwarded to
                          :meth:`~multimodalva.ensemble.stacking.StackingClassifier.train_base_models`.
                          Key options:
                          test_size (float), split_seed (int), train_seed (int),
                          val_size (float),
                          n_jobs (int), use_gpu (bool | None),
                          encode_categoricals (str | None), scale_numeric (bool),
                          save_fold_models (bool), cleanup_fold_files (bool).

        Returns:
            dict with ``oof_meta_X``, ``oof_y``, ``label2id``, ``id2label``,
            ``output_dir``.

        Raises:
            ValueError: If ``method`` is not ``"stacking"``.
        """
        self._require_method("stacking", "train_base_models")
        return self.classifier.train_base_models(
            df=df,
            label_col=label_col,
            text_col=text_col,
            feature_cols=feature_cols,
            **kwargs,
        )

    def train_meta_learner_stage(self, **kwargs) -> dict:
        """Stage 2 of stacking: train meta-learner candidates and select the best.

        Requires ``method="stacking"``.  Delegates to
        :meth:`~multimodalva.ensemble.stacking.StackingClassifier.train_meta_learner_stage`.

        Can be called immediately after ``train_base_models()`` in the same
        session, or in a new session (OOF artifacts reloaded from disk).

        Args:
            **kwargs: Forwarded to
                      :meth:`~multimodalva.ensemble.stacking.StackingClassifier.train_meta_learner_stage`.
                      Key options:
                      meta_learners (list[dict] | dict | None),
                      metric (str | None), meta_cv_folds (int),
                      split_seed (int | None), train_seed (int | None) —
                      ``None`` inherits the seeds stage 1 ran with —
                      n_jobs (int).

        Returns:
            dict with ``meta_learner``, ``meta_scores``, ``best_meta_name``,
            ``output_dir``.

        Raises:
            ValueError: If ``method`` is not ``"stacking"``.
        """
        self._require_method("stacking", "train_meta_learner_stage")
        return self.classifier.train_meta_learner_stage(**kwargs)

    def train_class_voter_stage(self, **kwargs) -> dict:
        """Stage 2 alternative for stacking: learn class-aware voting weights.

        Requires ``method="stacking"``. Delegates to
        :meth:`~multimodalva.ensemble.stacking.StackingClassifier.train_class_voter_stage`.

        Uses the Stage 1 OOF predictions to learn a per-model × per-class weight
        matrix for an interpretable class-aware voter.

        Args:
            **kwargs: Forwarded to
                      :meth:`~multimodalva.ensemble.stacking.StackingClassifier.train_class_voter_stage`.
                      Key options:
                      metric (str), shrinkage (float),
                      min_support_for_trust (float).

        Returns:
            dict with class-voter artifacts and metadata.

        Raises:
            ValueError: If ``method`` is not ``"stacking"``.
        """
        self._require_method("stacking", "train_class_voter_stage")
        return self.classifier.train_class_voter_stage(**kwargs)

    def predict_test(self, **kwargs):
        """Stage 3 of stacking: predict the test set via the trained meta-learner.

        Requires ``method="stacking"``.  Delegates to
        :meth:`~multimodalva.ensemble.stacking.StackingClassifier.predict_test`.

        Reloads all final base models and the meta-learner from disk if not
        already in memory.  Stacks test probability matrices and passes them
        through the meta-learner.

        Can be run in a new session after completing both Stage 1 and Stage 2.

        Args:
            **kwargs: Forwarded to
                      :meth:`~multimodalva.ensemble.stacking.StackingClassifier.predict_test`.
                      Key options:
                      top_k (int), batch_size (int).

        Returns:
            :class:`~multimodalva.utils.types.PredictionResult`.

        Raises:
            ValueError: If ``method`` is not ``"stacking"``.
        """
        self._require_method("stacking", "predict_test")
        result = self.classifier.predict_test(**kwargs)
        self.predictions = result
        return result

    def predict_test_class_voter(self, **kwargs):
        """Stage 3 alternative for stacking: predict test data via class-aware voting.

        Requires ``method="stacking"``. Delegates to
        :meth:`~multimodalva.ensemble.stacking.StackingClassifier.predict_test_class_voter`.

        Reloads final base models and the saved class-aware weight matrix from
        disk if needed, then combines base-model probabilities with the learned
        per-class weights.

        Args:
            **kwargs: Forwarded to
                      :meth:`~multimodalva.ensemble.stacking.StackingClassifier.predict_test_class_voter`.
                      Key options:
                      top_k (int), batch_size (int).

        Returns:
            :class:`~multimodalva.utils.types.PredictionResult`.

        Raises:
            ValueError: If ``method`` is not ``"stacking"``.
        """
        self._require_method("stacking", "predict_test_class_voter")
        result = self.classifier.predict_test_class_voter(**kwargs)
        self.predictions = result
        return result

    def train_ensemble_selection_stage(self, **kwargs) -> dict:
        """Stage 2 alternative for stacking: greedy ensemble selection.

        Requires ``method="stacking"``. Delegates to
        :meth:`~multimodalva.ensemble.stacking.StackingClassifier.train_ensemble_selection_stage`.

        Learns one non-negative weight per base model (summing to 1) from the
        Stage 1 OOF predictions (Caruana et al. 2004).

        Args:
            **kwargs: Forwarded. Key options: metric (str), ensemble_size (int),
                      use_best_in_trajectory (bool), sorted_init (int),
                      n_bags (int), bag_fraction (float), train_seed (int | None —
                      None inherits stage 1's).

        Returns:
            dict with ``weights`` (per model), ``scores``, ``oof_score``,
            ``best_single_score``, ``simple_average_score``, ``output_dir``.

        Raises:
            ValueError: If ``method`` is not ``"stacking"``.
        """
        self._require_method("stacking", "train_ensemble_selection_stage")
        return self.classifier.train_ensemble_selection_stage(**kwargs)

    def predict_test_ensemble_selection(self, **kwargs):
        """Stage 3 alternative for stacking: combine test probabilities with the
        saved ensemble-selection weights.

        Requires ``method="stacking"``. Delegates to
        :meth:`~multimodalva.ensemble.stacking.StackingClassifier.predict_test_ensemble_selection`.

        Args:
            **kwargs: Forwarded. Key options: top_k (int), batch_size (int).

        Returns:
            :class:`~multimodalva.utils.types.PredictionResult`.

        Raises:
            ValueError: If ``method`` is not ``"stacking"``.
        """
        self._require_method("stacking", "predict_test_ensemble_selection")
        result = self.classifier.predict_test_ensemble_selection(**kwargs)
        self.predictions = result
        return result

    def predict_test_simple_average(self, **kwargs):
        """Stage 3 for stacking: equal-weight average of the final base models.

        Requires ``method="stacking"``. Delegates to
        :meth:`~multimodalva.ensemble.stacking.StackingClassifier.predict_test_simple_average`.
        Nothing is trained, so there is no Stage 2 for this combiner.

        Args:
            **kwargs: Forwarded. Key options: top_k (int), batch_size (int).

        Returns:
            :class:`~multimodalva.utils.types.PredictionResult`.

        Raises:
            ValueError: If ``method`` is not ``"stacking"``.
        """
        self._require_method("stacking", "predict_test_simple_average")
        result = self.classifier.predict_test_simple_average(**kwargs)
        self.predictions = result
        return result

    def compare_combiners_stage(self, **kwargs):
        """Compare the Stage 2 combiners by nested CV over the OOF rows.

        Requires ``method="stacking"``. Delegates to
        :meth:`~multimodalva.ensemble.stacking.StackingClassifier.compare_combiners_stage`.
        Uses training data only; ``run(combiner="best")`` chooses with it.

        Args:
            **kwargs: Forwarded. Key options: metric (str), n_combiner_folds (int),
                      split_seed / train_seed (int | None — None inherits stage 1's),
                      meta_learners (list[dict]), class_voter_kwargs (dict),
                      ensemble_selection_kwargs (dict).

        Returns:
            DataFrame with one row per combiner (``simple_average``,
            ``class_aware_voting``, ``ensemble_selection``, then one per
            meta-learner, named by its ``model_name``).

        Raises:
            ValueError: If ``method`` is not ``"stacking"``.
        """
        self._require_method("stacking", "compare_combiners_stage")
        return self.classifier.compare_combiners_stage(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_method(self, required: str, fn_name: str) -> None:
        """Raise ValueError if the current method is not *required*."""
        if self.method != required:
            raise ValueError(
                f"{fn_name}() is only available for method='{required}', "
                f"but this instance uses method='{self.method}'."
            )

    def __repr__(self) -> str:
        return (
            f"EnsembleClassifier("
            f"method={self.method!r}, "
            f"output_dir={str(self.output_dir)!r})"
        )
