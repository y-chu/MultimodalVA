"""
Step 6: TextClassifier — user-facing wrapper for the full text classification pipeline.

Chains: split → prepare_dataset → [hpo →] train → predict
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# from ..utils.split import split
# from .dataset import prepare_dataset
# from .train import train
# from .predict import predict, PredictionResult
# from .hpo import optimize

from multimodalva.utils.split import split
from multimodalva.text.dataset import prepare_dataset
from multimodalva.text.train import train
from multimodalva.text.predict import predict, PredictionResult

logger = logging.getLogger(__name__)


class TextClassifier:
    """End-to-end text-only cause of death classifier.

    Typical usage (with HPO)::

        clf = TextClassifier(model_name="bert-base-uncased", output_dir="runs/text")
        results = clf.run(df, text_col="narrative", label_col="cause_of_death",
                          use_optimize=True, n_trials=20)

    Typical usage (fixed hyperparams)::

        clf = TextClassifier(model_name="bert-base-uncased", output_dir="runs/text")
        results = clf.run(df, text_col="narrative", label_col="cause_of_death",
                          hyperparams={"learning_rate": 3e-5, "epochs": 4})
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dir: str | Path = "runs/text",
    ):
        """Initialize the TextClassifier.

        Args:
            model_name: HuggingFace model name or local path.
            output_dir: Root directory for saving all outputs.
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)

        # Populated after run()
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.train_dataset = None
        self.test_dataset = None
        self.label2id: dict | None = None
        self.id2label: dict | None = None
        self.train_metadata: dict | None = None
        self.predictions: PredictionResult | None = None
        self.best_hyperparams: dict | None = None
        self.study = None  # Optuna study, set when use_optimize=True

    def run(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        max_length: int = 512,
        hyperparams: dict | None = None,
        use_optimize: bool = False,
        n_trials: int = 20,
        optimize_metric: str = "accuracy",
        search_space: dict | None = None,
        batch_size: int = 32,
    ) -> dict:
        """Run the full text classification pipeline.

        Steps executed in order:
            1. split()           — stratified train/test split
            2. prepare_dataset() — tokenize and encode labels
            3. optimize()        — (optional) Optuna HPO on 80/20 sub-split of train_ds
            4. train()           — fine-tune model with best/given hyperparams
            5. predict()         — generate labels and probabilities on test set

        All intermediate outputs are stored on self for inspection.

        When use_optimize=True:
            - optimize() determines best hyperparams (always includes epochs).
            - final train() uses val_size=None to train on all training data —
              epoch count already known from HPO, no internal eval split needed.
        When use_optimize=False:
            - train() uses val_size=0.1 (default) for early stopping / best-checkpoint
              selection. hyperparams param is passed directly.

        Args:
            df: Input DataFrame with text and label columns.
            text_col: Name of the text column.
            label_col: Name of the label column.
            test_size: Fraction of data held out for testing. Default 0.2.
            random_state: Random seed for splitting and HPO. Default 42.
            stratify: Stratified split. Default True.
            max_length: Max token length for the tokenizer. Default 512.
            hyperparams: Fixed hyperparameter dict. Ignored when use_optimize=True.
                         Keys use short names: learning_rate, batch_size, epochs,
                         weight_decay, warmup_ratio, gradient_accumulation_steps,
                         freeze_layers. See train() for the full list.
            use_optimize: Run Optuna HPO before final training. Default False.
            n_trials: Number of Optuna trials (only used when use_optimize=True).
            optimize_metric: Metric to maximize during HPO — "accuracy", "f1_macro",
                             "f1_weighted", "csmf_accuracy". Default "accuracy".
            search_space: Custom Optuna search space dict (only used when
                          use_optimize=True). Merged over DEFAULT_SEARCH_SPACE.
            batch_size: Inference batch size for predict(). Default 32.

        Returns:
            results: Dict with keys:
                - "predictions": PredictionResult(top1, full, topk, id2label) from predict()
                - "train_metadata": metadata dict from train() — includes log_history,
                                    hyperparams, label2id, id2label, output_dir
                - "best_hyperparams": hyperparams used for final training
                - "label2id": label encoding map
                - "id2label": reverse label encoding map
                - "output_dir": Path to the root output directory
        """
        # --- Step 1: split ---
        self.train_df, self.test_df = split(
            df,
            text_col=text_col,
            label_col=label_col,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        logger.info(
            "Split: %d train, %d test samples.",
            len(self.train_df), len(self.test_df),
        )

        # --- Step 2: prepare datasets ---
        self.train_dataset, self.test_dataset, self.label2id, self.id2label = (
            prepare_dataset(
                self.train_df,
                self.test_df,
                text_col=text_col,
                label_col=label_col,
                model_name=self.model_name,
                max_length=max_length,
            )
        )
        logger.info("Prepared datasets: %d classes.", len(self.label2id))

        # --- Step 3: optional HPO ---
        if use_optimize:
            from multimodalva.text.hpo import optimize  # noqa: PLC0415
            self.best_hyperparams, self.study = optimize(
                train_dataset=self.train_dataset,
                label2id=self.label2id,
                id2label=self.id2label,
                model_name=self.model_name,
                output_dir=self.output_dir / "hpo",
                n_trials=n_trials,
                metric=optimize_metric,
                search_space=search_space,
                random_state=random_state,
            )
            final_hyperparams = self.best_hyperparams
            # HPO already determined epochs; train on full training data.
            final_val_size = None
        else:
            self.best_hyperparams = hyperparams
            final_hyperparams = hyperparams
            # train() carves an internal val split for early stopping.
            final_val_size = 0.1

        # --- Step 4: final training ---
        _, _, self.train_metadata = train(
            train_dataset=self.train_dataset,
            label2id=self.label2id,
            id2label=self.id2label,
            model_name=self.model_name,
            output_dir=self.output_dir / "final",
            hyperparams=final_hyperparams,
            val_size=final_val_size,
        )

        # --- Step 5: predict on test set ---
        self.predictions = predict(
            output_dir=self.output_dir / "final",
            test_dataset=self.test_dataset,
            batch_size=batch_size,
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
