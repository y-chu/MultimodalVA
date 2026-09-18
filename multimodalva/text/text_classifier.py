"""
Step 6: TextClassifier — user-facing wrapper for the full text classification pipeline.

Chains: split → prepare_dataset → [hpo →] train → predict
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

# from ..utils.split import split
# from .dataset import prepare_dataset
# from .train import train
# from .predict import predict, PredictionResult
# from .hpo import optimize

from multimodalva.utils.split import split
from multimodalva.utils.runtime import RuntimeTracker, distributed_state, resolve_seed
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
        set_seed: int | None = None,
        stratify: bool = True,
        split_col: str | None = None,
        max_length: int = 512,
        hyperparams: dict | None = None,
        use_optimize: bool = False,
        n_trials: int = 30,
        optimize_metric: str = "accuracy",
        search_space: dict | None = None,
        batch_size: int = 32,
        top_k: int = 3,
        use_lora: bool = True,
        use_focal: bool = False,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 4,
        resume_hpo: bool = True,
        resume_training: bool = True,
        use_fast: bool = True,
        id_col: str | None = None,
        use_cv: bool = True,
        n_cv_folds: int = 3,
        use_compile: bool = False,
        push_to_hub: bool = False,
        hub_repo_id: str | None = None,
        hub_private: bool = True,
        hub_token: str | None = None,
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
            id_col: Optional column holding a row identifier. When given, the
                identifiers appear as a leading ``id`` column in the prediction
                tables, so results can be joined back to the source records.
            set_seed: Optional alias for a single run-level seed. When provided,
                      overrides ``random_state`` so one value controls split, HPO,
                      and training seeds end-to-end.
            stratify: Stratified split. Default True.
            max_length: Max token length for the tokenizer. Default 512.
            hyperparams: Fixed hyperparameter dict. Ignored when use_optimize=True.
                         Keys use short names: learning_rate, batch_size, epochs,
                         weight_decay, warmup_ratio, gradient_accumulation_steps,
                         freeze_layers. See train() for the full list.
            use_optimize: Run Optuna HPO before final training. Default False.
            n_trials: Number of Optuna trials (only used when use_optimize=True).
                      Default 30.
            optimize_metric: Metric to maximize during HPO — "accuracy", "f1_macro",
                             "f1_weighted", "csmf_accuracy". Default "accuracy".
            search_space: Custom Optuna search space dict (only used when
                          use_optimize=True). Merged over DEFAULT_SEARCH_SPACE.
            batch_size: Inference batch size for predict(). Default 32.
            top_k: Number of top classes in topk output. Default 3.
            use_lora: Apply LoRA adapters for parameter-efficient fine-tuning.
                      Reduces GPU memory and improves generalisation on small datasets.
                      When True, LoRA hyperparameters (lora_r, lora_alpha, lora_dropout)
                      are also searched during HPO. Default True.
            use_focal: Use focal loss during HPO trials (use_optimize=True only).
                       Merges FOCAL_SEARCH_SPACE and injects loss_type="focal" into every
                       trial. For non-HPO focal loss, pass hyperparams={"loss_type":
                       "focal", "focal_gamma": 2.0}. Default False.
            gradient_checkpointing: Enable gradient checkpointing to reduce GPU memory
                                    at the cost of slightly slower training. Default False.
            early_stopping_patience: Stop training early if eval metric does not improve
                                     for this many epochs. None = disabled. Default 4.
            resume_hpo: Resume an existing Optuna study if one exists at
                        output_dir/hpo/ (load_if_exists). Default True.
            resume_training: Resume final training from the latest checkpoint in
                             output_dir/final/ if one exists. Default True.
            use_fast: Use the HuggingFace fast (Rust) tokenizer. Default True.
                      Set False for models that lack a fast tokenizer (e.g. BlueBERT).
            use_cv: Use k-fold CV scoring in HPO trials (only when use_optimize=True).
                    Default True.
            n_cv_folds: Number of CV folds. Default 3.
            use_compile: Apply torch.compile() to the model for the final train() call.
                         Not forwarded to HPO trials (compilation overhead per trial is
                         counterproductive). Uses aot_eager backend on MPS, inductor on
                         CUDA. Default False.
            push_to_hub: After training, publish the final model to the Hugging Face
                         Hub. Requires hub_repo_id. Rank-0 only in DDP runs. Default False.
            hub_repo_id: Target Hub repo id (e.g. "your-org/va-bert"). Required when
                         push_to_hub=True.
            hub_private: Create/keep the Hub repo private. Default True.
            hub_token:   HF auth token. None uses the cached `huggingface-cli login`
                         credential or HF_TOKEN env var.

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
        runtime_tracker = RuntimeTracker(
            self.output_dir,
            report_name="pipeline_runtime.json",
            metadata={
                "pipeline": "text_classifier",
                "model_name": self.model_name,
                "text_col": text_col,
                "label_col": label_col,
                "use_optimize": use_optimize,
            },
            logger_=logger,
        )
        effective_seed = resolve_seed(random_state=random_state, set_seed=set_seed)
        # --- Step 1: split ---
        with runtime_tracker.stage(
            "split",
            details={
                "rows": len(df),
                "test_size": test_size,
                "random_state": effective_seed,
                "stratify": stratify,
            },
        ):
            self.train_df, self.test_df = split(
                df,
                text_col=text_col,
                label_col=label_col,
                test_size=test_size,
                random_state=effective_seed,
                stratify=stratify,
                split_col=split_col,
            )
            logger.info(
                "Split: %d train, %d test samples.",
                len(self.train_df), len(self.test_df),
            )

        # --- Step 2: prepare datasets ---
        with runtime_tracker.stage(
            "prepare_dataset",
            details={"max_length": max_length, "use_fast": use_fast},
        ):
            self.train_dataset, self.test_dataset, self.label2id, self.id2label = (
                prepare_dataset(
                    self.train_df,
                    self.test_df,
                    text_col=text_col,
                    label_col=label_col,
                    model_name=self.model_name,
                    max_length=max_length,
                    use_fast=use_fast,
                    id_col=id_col,
                )
            )
            logger.info("Prepared datasets: %d classes.", len(self.label2id))

        # --- Step 3: optional HPO ---
        if use_optimize:
            from multimodalva.text.hpo import optimize  # noqa: PLC0415
            with runtime_tracker.stage(
                "hpo",
                details={"n_trials": n_trials, "metric": optimize_metric},
            ):
                self.best_hyperparams, self.study = optimize(
                    train_dataset=self.train_dataset,
                    label2id=self.label2id,
                    id2label=self.id2label,
                    model_name=self.model_name,
                    output_dir=self.output_dir / "hpo",
                    n_trials=n_trials,
                    metric=optimize_metric,
                    search_space=search_space,
                    random_state=effective_seed,
                    use_lora=use_lora,
                    use_focal=use_focal,
                    gradient_checkpointing=gradient_checkpointing,
                    early_stopping_patience=early_stopping_patience,
                    load_if_exists=resume_hpo,
                    use_fast=use_fast,
                    use_cv=use_cv,
                    n_cv_folds=n_cv_folds,
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
        with runtime_tracker.stage(
            "train",
            details={"resume_training": resume_training, "val_size": final_val_size},
        ):
            _, _, self.train_metadata = train(
                train_dataset=self.train_dataset,
                label2id=self.label2id,
                id2label=self.id2label,
                model_name=self.model_name,
                output_dir=self.output_dir / "final",
                hyperparams=final_hyperparams,
                val_size=final_val_size,
                use_lora=use_lora,
                gradient_checkpointing=gradient_checkpointing,
                early_stopping_patience=early_stopping_patience,
                resume=resume_training,
                use_fast=use_fast,
                random_state=effective_seed,
                use_compile=use_compile,  # NOT forwarded to HPO — compile overhead per trial is counterproductive
            )

        rank, world_size = distributed_state()
        if world_size > 1 and rank != 0:
            logger.info(
                "Skipping predict() on non-zero rank %d/%d in DDP run.",
                rank, world_size,
            )
            self.predictions = None
            return {
                "predictions": None,
                "train_metadata": self.train_metadata,
                "best_hyperparams": final_hyperparams,
                "label2id": self.label2id,
                "id2label": self.id2label,
                "output_dir": self.output_dir,
                "runtime_report": runtime_tracker.report_path,
                "runtime_stage_csv": runtime_tracker.stage_csv_path,
            }

        # --- Step 5: predict on test set ---
        with runtime_tracker.stage(
            "predict",
            details={"batch_size": batch_size, "top_k": top_k},
        ):
            self.predictions = predict(
                output_dir=self.output_dir / "final",
                test_dataset=self.test_dataset,
                batch_size=batch_size,
                top_k=top_k,
                save_dir=self.output_dir / "predictions",
                use_fast=use_fast,
            )

        hub_url = self._maybe_push_to_hub(
            push_to_hub, hub_repo_id, hub_private, hub_token, max_length,
            model_kind="text",
        )

        return {
            "predictions": self.predictions,
            "train_metadata": self.train_metadata,
            "best_hyperparams": final_hyperparams,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "output_dir": self.output_dir,
            "hub_url": hub_url,
            "runtime_report": runtime_tracker.report_path,
            "runtime_stage_csv": runtime_tracker.stage_csv_path,
        }

    def _maybe_push_to_hub(
        self,
        push: bool,
        hub_repo_id: str | None,
        hub_private: bool,
        hub_token: str | None,
        max_length: int,
        model_kind: str = "text",
    ) -> str | None:
        """Publish the final model to the Hugging Face Hub if requested.

        Runs only on DDP rank 0. Computes held-out test metrics from
        ``self.predictions`` (when available) to embed in the model card.
        Returns the repo URL, or None if not pushed.
        """
        if not push:
            return None
        rank, _ = distributed_state()
        if rank != 0:
            return None
        if not hub_repo_id:
            raise ValueError(
                "push_to_hub=True requires hub_repo_id (e.g. 'your-org/va-bert')."
            )

        from multimodalva.utils.hub import push_to_hub as _push  # noqa: PLC0415

        metrics = None
        if self.predictions is not None:
            from multimodalva.utils.metrics import score_predictions, CV_METRICS  # noqa: PLC0415

            top1 = self.predictions.top1
            metrics = {
                m: float(score_predictions(top1, m))
                for m in CV_METRICS
            }
        return _push(
            self.output_dir / "final",
            hub_repo_id,
            private=hub_private,
            token=hub_token,
            model_kind=model_kind,
            base_model=self.model_name,
            metrics=metrics,
            max_length=max_length,
        )

    def _run_from_datasets(
        self,
        label_col: str,
        hyperparams: dict | None = None,
        use_optimize: bool = False,
        n_trials: int = 30,
        optimize_metric: str = "accuracy",
        search_space: dict | None = None,
        batch_size: int = 32,
        top_k: int = 3,
        use_lora: bool = True,
        use_focal: bool = False,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 4,
        resume_hpo: bool = True,
        resume_training: bool = True,
        random_state: int = 42,
        set_seed: int | None = None,
        use_fast: bool = True,
        use_cv: bool = True,
        n_cv_folds: int = 3,
        use_compile: bool = False,
        _runtime_tracker: RuntimeTracker | None = None,
    ) -> dict:
        """Run Steps 3–5 (HPO → train → predict) on pre-loaded datasets.

        Intended for callers (e.g. DataFusionClassifier) that have already
        populated ``self.train_dataset``, ``self.test_dataset``,
        ``self.label2id``, and ``self.id2label`` before calling this method.
        Skips the split and prepare_dataset steps.

        All parameters mirror ``run()``.
        """
        if self.train_dataset is None or self.label2id is None:
            raise RuntimeError(
                "_run_from_datasets() requires train_dataset, test_dataset, "
                "label2id, and id2label to be set on self before calling."
            )

        runtime_tracker = _runtime_tracker or RuntimeTracker(
            self.output_dir,
            report_name="pipeline_runtime.json",
            metadata={
                "pipeline": "text_classifier_from_datasets",
                "model_name": self.model_name,
                "label_col": label_col,
                "use_optimize": use_optimize,
            },
            logger_=logger,
        )
        effective_seed = resolve_seed(random_state=random_state, set_seed=set_seed)

        # --- Step 3: optional HPO ---
        if use_optimize:
            from multimodalva.text.hpo import optimize  # noqa: PLC0415
            with runtime_tracker.stage(
                "hpo",
                details={"n_trials": n_trials, "metric": optimize_metric},
            ):
                self.best_hyperparams, self.study = optimize(
                    train_dataset=self.train_dataset,
                    label2id=self.label2id,
                    id2label=self.id2label,
                    model_name=self.model_name,
                    output_dir=self.output_dir / "hpo",
                    n_trials=n_trials,
                    metric=optimize_metric,
                    search_space=search_space,
                    random_state=effective_seed,
                    use_lora=use_lora,
                    use_focal=use_focal,
                    gradient_checkpointing=gradient_checkpointing,
                    early_stopping_patience=early_stopping_patience,
                    load_if_exists=resume_hpo,
                    use_fast=use_fast,
                    use_cv=use_cv,
                    n_cv_folds=n_cv_folds,
                )
            final_hyperparams = self.best_hyperparams
            final_val_size = None
        else:
            self.best_hyperparams = hyperparams
            final_hyperparams = hyperparams
            final_val_size = 0.1

        # --- Step 4: final training ---
        with runtime_tracker.stage(
            "train",
            details={"resume_training": resume_training, "val_size": final_val_size},
        ):
            _, _, self.train_metadata = train(
                train_dataset=self.train_dataset,
                label2id=self.label2id,
                id2label=self.id2label,
                model_name=self.model_name,
                output_dir=self.output_dir / "final",
                hyperparams=final_hyperparams,
                val_size=final_val_size,
                use_lora=use_lora,
                gradient_checkpointing=gradient_checkpointing,
                early_stopping_patience=early_stopping_patience,
                resume=resume_training,
                use_fast=use_fast,
                random_state=effective_seed,
                use_compile=use_compile,  # NOT forwarded to HPO — compile overhead per trial is counterproductive
            )

        rank, world_size = distributed_state()
        if world_size > 1 and rank != 0:
            logger.info(
                "Skipping predict() on non-zero rank %d/%d in DDP run.",
                rank, world_size,
            )
            self.predictions = None
            return {
                "predictions": None,
                "train_metadata": self.train_metadata,
                "best_hyperparams": final_hyperparams,
                "label2id": self.label2id,
                "id2label": self.id2label,
                "output_dir": self.output_dir,
                "runtime_report": runtime_tracker.report_path,
                "runtime_stage_csv": runtime_tracker.stage_csv_path,
            }

        # --- Step 5: predict on test set ---
        with runtime_tracker.stage(
            "predict",
            details={"batch_size": batch_size, "top_k": top_k},
        ):
            self.predictions = predict(
                output_dir=self.output_dir / "final",
                test_dataset=self.test_dataset,
                batch_size=batch_size,
                top_k=top_k,
                save_dir=self.output_dir / "predictions",
                use_fast=use_fast,
            )

        return {
            "predictions": self.predictions,
            "train_metadata": self.train_metadata,
            "best_hyperparams": final_hyperparams,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "output_dir": self.output_dir,
            "runtime_report": runtime_tracker.report_path,
            "runtime_stage_csv": runtime_tracker.stage_csv_path,
        }
