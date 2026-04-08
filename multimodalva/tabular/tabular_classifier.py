"""
Step 6 (tabular pipeline): TabularClassifier — user-facing wrapper.

Chains: split → prepare_dataset → [optimize →] train → predict
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.split import split
from ..utils.types import PredictionResult
from .dataset import prepare_dataset
from .predict import predict
from .train import train

logger = logging.getLogger(__name__)


class TabularClassifier:
    """End-to-end tabular cause-of-death classifier.

    Supported model_name values:
        "catboost", "lightgbm", "gbdt", "xgboost", "mlp", "random_forest"

    Typical usage (fixed hyperparams)::

        clf = TabularClassifier(model_name="random_forest", output_dir="runs/tabular")
        results = clf.run(df, feature_cols=["age", "sex", ...], label_col="cause")

    Typical usage (with HPO)::

        clf = TabularClassifier(model_name="lightgbm", output_dir="runs/tabular")
        results = clf.run(
            df, feature_cols=[...], label_col="cause",
            use_optimize=True, n_trials=50, optimize_metric="csmf_accuracy",
        )
    """

    def __init__(
        self,
        model_name: str = "random_forest",
        output_dir: str | Path = "runs/tabular",
    ):
        """Initialize the TabularClassifier.

        Args:
            model_name: Model alias. One of:
                        catboost, lightgbm, gbdt, xgboost, mlp, random_forest.
            output_dir: Root directory for all outputs.
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)

        # Populated after run()
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.preprocessor = None
        self.feature_names: list[str] | None = None
        self.label2id: dict | None = None
        self.id2label: dict | None = None
        self.train_metadata: dict | None = None
        self.predictions: PredictionResult | None = None
        self.best_hyperparams: dict | None = None
        self.study = None  # Optuna study, set when use_optimize=True

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        cat_cols: list[str] | None = None,
        num_cols: list[str] | None = None,
        drop_missing_label: bool = True,
        encode_categoricals: str | None = "ordinal",
        scale_numeric: bool = False,
        hyperparams: dict | None = None,
        use_optimize: bool = False,
        n_trials: int = 50,
        optimize_metric: str = "accuracy",
        search_space: dict | None = None,
        top_k: int = 3,
        n_jobs: int = -1,
        use_gpu: bool | None = None,
        search_space_profile: str = "auto",
    ) -> dict:
        """Run the full tabular classification pipeline.

        Steps executed in order:
            1. split()           — stratified train/test split
            2. prepare_dataset() — imputation, encoding, scaling; label encoding
            3. optimize()        — (optional) Optuna HPO on internal val split
            4. train()           — fit model with best/given hyperparams
            5. predict()         — generate labels and probabilities on test set

        All intermediate outputs are stored on self for inspection after run().

        Args:
            df:               Input DataFrame.
            feature_cols:     Columns to use as features.
            label_col:        Name of the label column.
            test_size:        Fraction held out for testing. Default 0.2.
            random_state:     Random seed. Default 42.
            stratify:         Stratified split. Default True.
            cat_cols:         Categorical feature columns. Auto-detected if None.
            num_cols:         Numeric feature columns. Auto-detected if None.
            hyperparams:      Fixed hyperparameter dict. Ignored when use_optimize=True.
            use_optimize:     Run Optuna HPO before final training. Default False.
            n_trials:         Optuna trial count (use_optimize=True only). Default 50.
            optimize_metric:  Metric to optimise during HPO. Default "accuracy".
                              Options: "accuracy", "balanced_accuracy",
                              "f1_macro", "f1_weighted", "csmf_accuracy", "log_loss".
            search_space:     Custom Optuna search space dict (use_optimize=True only).
                              Merged over the adaptive default space chosen from
                              X_train.shape.
            search_space_profile:
                              One of "auto", "small", "balanced", "wide", "large".
                              "auto" infers a profile from the prepared feature
                              matrix shape before HPO.
            top_k:            Number of top classes in the topk output. Default 3.

        Returns:
            dict with keys:
                "predictions":      PredictionResult(top1, full, topk, id2label)
                "train_metadata":   metadata dict from train()
                "best_hyperparams": hyperparams used for final training
                "label2id":         label encoding map
                "id2label":         reverse label encoding map
                "output_dir":       Path to the root output directory
        """
        # --- Step 1: split ---
        self.train_df, self.test_df = split(
            df,
            label_col=label_col,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        logger.info("Split: %d train, %d test.", len(self.train_df), len(self.test_df))

        # --- Step 2: prepare datasets ---
        (
            self.X_train, self.X_test,
            self.y_train, self.y_test,
            self.preprocessor,
            self.label2id, self.id2label,
            self.feature_names,
        ) = prepare_dataset(
            self.train_df,
            self.test_df,
            feature_cols=feature_cols,
            label_col=label_col,
            cat_cols=cat_cols,
            num_cols=num_cols,
            drop_missing_label=drop_missing_label,
            encode_categoricals=encode_categoricals,
            scale_numeric=scale_numeric,
        )
        logger.info("Prepared datasets: %d classes.", len(self.label2id))

        # --- Step 3: optional HPO ---
        if use_optimize:
            from multimodalva.tabular.hpo import optimize  # noqa: PLC0415
            self.best_hyperparams, self.study = optimize(
                X_train=self.X_train,
                y_train=self.y_train,
                label2id=self.label2id,
                id2label=self.id2label,
                model_name=self.model_name,
                output_dir=self.output_dir / "hpo",
                n_trials=n_trials,
                metric=optimize_metric,
                search_space=search_space,
                search_space_profile=search_space_profile,
                random_state=random_state,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
            )
            final_hyperparams = self.best_hyperparams
        else:
            self.best_hyperparams = hyperparams
            final_hyperparams = hyperparams

        # --- Step 4: final training ---
        _, self.train_metadata = train(
            X_train=self.X_train,
            y_train=self.y_train,
            label2id=self.label2id,
            id2label=self.id2label,
            model_name=self.model_name,
            output_dir=self.output_dir / "final",
            hyperparams=final_hyperparams,
            preprocessor=self.preprocessor,
            feature_names=self.feature_names,
            random_state=random_state,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
        )

        # --- Step 5: predict on test set ---
        self.predictions = predict(
            output_dir=self.output_dir / "final",
            X_test=self.X_test,
            y_test=self.y_test,
            top_k=top_k,
            save_dir=self.output_dir / "predictions",
        )

        return {
            "predictions": self.predictions,
            "train_metadata": self.train_metadata,
            "best_hyperparams": final_hyperparams,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "output_dir": self.output_dir,
        }
