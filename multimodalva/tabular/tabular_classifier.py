"""
Step 6 (tabular pipeline): TabularClassifier — user-facing wrapper.

Chains: split → prepare_tabular_dataset → [optimize →] train → predict
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.runtime import track_run
from ..utils.seeds import seed_everything, set_determinism
from ..utils.split import split
from ..utils.types import PredictionResult
from .dataset import prepare_tabular_dataset, valid_label_mask
from .predict import predict_tabular
from .train import train_tabular

logger = logging.getLogger(__name__)

from ..utils.hpo_defaults import TABULAR_SPEC_DEFAULTS  # noqa: E402
from ..utils.optimize_config import (  # noqa: E402
    resolve_search_resume,
    Optimize,
    _log_fixed_or_default,
    resolve_hyperparams,
)


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
            hyperparams=Optimize(n_trials=50, metric="csmf_accuracy"),
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
        self.study = None  # Optuna study, set when hyperparams=Optimize(...)

    @track_run("tabular")
    def run(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        test_size: float = 0.2,
        split_seed: int = 42,
        train_seed: int = 42,
        deterministic: bool = False,
        stratify: bool = True,
        split_col: str | None = None,
        cat_cols: list[str] | None = None,
        num_cols: list[str] | None = None,
        drop_missing_label: bool = True,
        encode_categoricals: str | None = "ordinal",
        scale_numeric: bool = False,
        hyperparams: "dict | str | Optimize | None" = None,
        top_k: int = 3,
        n_jobs: int = -1,
        use_gpu: bool | None = None,
        id_col: str | None = None,
        resume: bool = True,
    ) -> dict:
        """Run the full tabular classification pipeline.

        Steps executed in order:
            1. split()           — stratified train/test split
            2. prepare_tabular_dataset() — imputation, encoding, scaling; label encoding
            3. search_tabular()          — (optional) HPO by k-fold CV or an internal
                                           val split; Optuna or Ray Tune per Optimize(backend=)
            4. train_tabular()           — fit model with best/given hyperparams
            5. predict_tabular()         — generate labels and probabilities on test set

        All intermediate outputs are stored on self for inspection after run().

        Args:
            df:               Input DataFrame.
            feature_cols:     Columns to use as features.
            label_col:        Name of the label column.
            id_col:           Optional column holding a row identifier. When
                              given, the identifiers appear as a leading ``id``
                              column in the prediction tables, so results can be
                              joined back to the source records.
            resume:           Continue an interrupted search in the same
                              output_dir from its finished trials. The final
                              refit always reruns — it takes seconds. Default
                              True.
            test_size:        Fraction held out for testing. Default 0.2.
            split_seed:       Seed for every row-partitioning decision — the
                              train/test split and the search's folds. Vary it
                              to measure sampling uncertainty. Default 42.
            train_seed:       Seed for the estimator's own ``random_state`` and
                              the Optuna sampler. Vary it, with
                              ``hyperparams`` fixed, to measure model
                              stochasticity. Default 42.
            deterministic:    Accepted so every task takes the same arguments.
                              The tabular models here train on CPU and are
                              already repeatable for a fixed ``train_seed``, so
                              this changes nothing unless a model uses Torch.
                              Default False.
            stratify:         Stratified split. Default True.
            cat_cols:         Categorical feature columns. Auto-detected if None.
            num_cols:         Numeric feature columns. Auto-detected if None.
            hyperparams:      Where the hyperparameters come from: a dict to use
                              exactly those values, ``Optimize(...)`` to search
                              for them, or ``"default"`` / omitted for the model
                              library's own defaults.
                              Options: "accuracy", "balanced_accuracy",
                              "f1_macro", "f1_weighted", "csmf_accuracy", "log_loss".
            top_k:            Number of top classes in the topk output. Default 3.

        Returns:
            dict with keys:
                "predictions":      PredictionResult(top1, full, topk, id2label)
                "train_metadata":   metadata dict from train_tabular()
                "best_hyperparams": hyperparams used for final training
                "label2id":         label encoding map
                "id2label":         reverse label encoding map
                "output_dir":       Path to the root output directory
        """
        # --- Step 1: split ---
        set_determinism(deterministic)
        seed_everything(train_seed)
        self.train_df, self.test_df = split(
            df,
            label_col=label_col,
            test_size=test_size,
            random_state=split_seed,
            stratify=stratify,
            split_col=split_col,
        )
        logger.info("Split: %d train, %d test.", len(self.train_df), len(self.test_df))

        # --- Step 2: prepare datasets ---
        (
            self.X_train, self.X_test,
            self.y_train, self.y_test,
            self.preprocessor,
            self.label2id, self.id2label,
            self.feature_names,
        ) = prepare_tabular_dataset(
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

        # --- Step 3: hyperparameters — searched, given, or the library's own ---
        hp_kind, hp_fixed, hp_search = resolve_hyperparams(hyperparams)
        if hp_kind == "search":
            hp_search = hp_search.with_defaults(
                n_trials=TABULAR_SPEC_DEFAULTS["n_trials"],
                metric=TABULAR_SPEC_DEFAULTS["optimize_metric"],
            )
            logger.info("Hyperparameters FROM SEARCH — %s.", hp_search.describe())
            from multimodalva.tabular.hpo import search_tabular  # noqa: PLC0415
            self.best_hyperparams, self.study = search_tabular(
                hp_search,
                resume=resolve_search_resume(hp_search, resume),
                X_train=self.X_train,
                y_train=self.y_train,
                label2id=self.label2id,
                id2label=self.id2label,
                model_name=self.model_name,
                output_dir=self.output_dir / "hpo",
                random_state=train_seed,
                split_seed=split_seed,
                n_jobs=n_jobs,
                use_gpu=use_gpu,
            )
            final_hyperparams = self.best_hyperparams
        else:
            _log_fixed_or_default(logger, self.model_name, hp_fixed)
            self.best_hyperparams = hp_fixed
            final_hyperparams = hp_fixed

        # --- Step 4: final training ---
        _, self.train_metadata = train_tabular(
            X_train=self.X_train,
            y_train=self.y_train,
            label2id=self.label2id,
            id2label=self.id2label,
            model_name=self.model_name,
            output_dir=self.output_dir / "final",
            hyperparams=final_hyperparams,
            preprocessor=self.preprocessor,
            feature_names=self.feature_names,
            random_state=train_seed,
            n_jobs=n_jobs,
            use_gpu=use_gpu,
        )

        # --- Step 5: predict on test set ---
        # Identifiers for the rows that survive prepare_tabular_dataset()'s label drop,
        # so they line up with the scored rows.
        test_ids = None
        if id_col is not None:
            if id_col not in self.test_df.columns:
                raise ValueError(f"id_col {id_col!r} not found in the DataFrame.")
            kept = self.test_df
            if drop_missing_label:
                kept = kept[valid_label_mask(kept, label_col)]
            test_ids = kept[id_col].tolist()

        self.predictions = predict_tabular(
            output_dir=self.output_dir / "final",
            X_test=self.X_test,
            y_test=self.y_test,
            top_k=top_k,
            save_dir=self.output_dir / "predictions",
            ids=test_ids,
        )

        return {
            "predictions": self.predictions,
            "train_metadata": self.train_metadata,
            "best_hyperparams": final_hyperparams,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "output_dir": self.output_dir,
        }
