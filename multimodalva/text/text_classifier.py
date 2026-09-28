"""
Step 6: TextClassifier — user-facing wrapper for the full text classification pipeline.

Chains: split → prepare_text_dataset → [hpo →] train → predict
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

# from ..utils.split import split
# from .dataset import prepare_text_dataset
# from .train import train
# from .predict import predict, PredictionResult
# from .hpo import optimize

from multimodalva.utils.split import split
from multimodalva.utils.runtime import (
    RuntimeTracker, distributed_state, apply_dataloader_workers,
)
from multimodalva.utils.runtime import track_run
from multimodalva.utils.seeds import seed_everything, set_determinism
from multimodalva.text.dataset import prepare_text_dataset
from multimodalva.text.train import train_text
from multimodalva.text.predict import predict_text, PredictionResult

logger = logging.getLogger(__name__)

from ..utils.hpo_defaults import TEXT_SPEC_DEFAULTS  # noqa: E402
from ..utils.optimize_config import (  # noqa: E402
    resolve_backend,
    resolve_search_resume,
    Optimize,
    _log_fixed_or_default,
    resolve_hyperparams,
)


def _record_tokenization(
    final_dir: Path, metadata: dict | None, max_length: int, use_fast: bool
) -> None:
    """Add the tokenisation settings to a finished run's ``training_metadata.json``.

    ``train_text()`` receives already-tokenised datasets, so ``max_length`` is not part of
    the metadata it writes. Anything that reads a saved run later — notably
    ``push_to_hub()`` building the usage snippet — then has no way to know the
    truncation length that was actually used. Recording it here keeps that
    recoverable. Failures are logged and ignored: the run itself has succeeded by
    this point and must not be lost over a metadata write.
    """
    import json  # noqa: PLC0415

    extra = {"max_length": max_length, "use_fast": use_fast}
    if metadata is not None:
        metadata.update(extra)

    meta_path = final_dir / "training_metadata.json"
    if not meta_path.exists():
        return
    try:
        with open(meta_path) as f:
            saved = json.load(f)
        saved.update(extra)
        with open(meta_path, "w") as f:
            json.dump(saved, f, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not record max_length in %s: %s", meta_path, exc)


class TextClassifier:
    """End-to-end text-only cause of death classifier.

    Typical usage (with HPO)::

        clf = TextClassifier(model_name="bert-base-uncased", output_dir="runs/text")
        results = clf.run(df, text_col="narrative", label_col="cause_of_death",
                          hyperparams=Optimize(n_trials=20))

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
        self.study = None  # Optuna study, set when hyperparams=Optimize(...)

    @track_run("text", track=False)
    @apply_dataloader_workers
    def run(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        test_size: float = 0.2,
        split_seed: int = 42,
        train_seed: int = 42,
        deterministic: bool = False,
        stratify: bool = True,
        split_col: str | None = None,
        max_length: int = 512,
        hyperparams: "dict | str | Optimize | None" = None,
        batch_size: int = 32,
        top_k: int = 3,
        use_lora: bool = False,
        use_focal: bool = False,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 4,
        val_size: float | None = 0.1,
        n_jobs: int | None = None,
        resume: bool = True,
        use_fast: bool = True,
        id_col: str | None = None,
        use_compile: bool = False,
        push_to_hub: bool = False,
        hub_repo_id: str | None = None,
        hub_private: bool = True,
        hub_token: str | None = None,
    ) -> dict:
        """Run the full text classification pipeline.

        Steps executed in order:
            1. split()           — stratified train/test split
            2. prepare_text_dataset() — tokenize and encode labels
            3. search_text()          — (optional) HPO on a sub-split of train_ds, or
                                        by k-fold CV over it; Optuna or Ray Tune
                                        per Optimize(backend=)
            4. train_text()           — fine-tune model with best/given hyperparams
            5. predict_text()         — generate labels and probabilities on test set

        All intermediate outputs are stored on self for inspection.

        When hyperparams=Optimize(...):
            - search_text() determines best hyperparams (always includes epochs).
            - final train_text() uses val_size=None to train on all training data —
              epoch count already known from HPO, no internal eval split needed.
        Otherwise:
            - train_text() uses val_size=0.1 (default) for early stopping / best-checkpoint
              selection. hyperparams param is passed directly.

        Args:
            df: Input DataFrame with text and label columns.
            text_col: Name of the text column.
            label_col: Name of the label column.
            test_size: Fraction of data held out for testing. Default 0.2.
            id_col: Optional column holding a row identifier. When given, the
                identifiers appear as a leading ``id`` column in the prediction
                tables, so results can be joined back to the source records.
            split_seed: Seed for every row-partitioning decision — the train/test
                split, the search's cross-validation folds and the early-stopping
                validation slice. Vary it to measure sampling uncertainty.
                Default 42.
            train_seed: Seed for everything the model does with its rows —
                initialisation, dropout, batch order, the Optuna sampler and the
                Hugging Face Trainer. Vary it, with ``hyperparams`` fixed, to
                measure model stochasticity. Default 42.
            deterministic: Demand bit-for-bit repeatable kernels. Slower, and
                an operation with no deterministic implementation raises instead
                of falling back. See :func:`multimodalva.utils.seeds.set_determinism`.
                Default False.
            stratify: Stratified split. Default True.
            max_length: Max token length for the tokenizer. Default 512.
            hyperparams: Where the hyperparameters come from. One of:

                         * a dict — use exactly these. Keys use short names:
                           learning_rate, batch_size, epochs, weight_decay,
                           warmup_ratio, gradient_accumulation_steps,
                           freeze_layers. See train_text() for the full list.
                         * ``Optimize(...)`` — search for them. ``Optimize()``
                           searches the package's adaptive space; its arguments
                           set the metric, trial count, a per-key space override
                           and the cross-validation settings.
                         * ``"default"`` or omitted — train with the model
                           library's own defaults, with no search.
            batch_size: Inference batch size for predict_text(). Default 32.
            top_k: Number of top classes in topk output. Default 3.
            use_lora: Train LoRA adapters instead of the full model. Far fewer
                      parameters carry gradients, so the optimizer state shrinks;
                      the forward pass is unchanged, so wall-clock time may not
                      drop much. During a search, lora_r, lora_alpha and
                      lora_dropout join the space. Default False.
            use_focal: Use focal loss. During a search it also adds focal_gamma
                       and class_weights to the space. Without a search, set it
                       through hyperparams={"loss_type": "focal",
                       "focal_gamma": 2.0} instead. Default False.
            gradient_checkpointing: Enable gradient checkpointing to reduce GPU memory
                                    at the cost of slightly slower training. Default False.
            early_stopping_patience: Stop training early if eval metric does not improve
                                     for this many epochs. None = disabled. Default 4.
            val_size: Share of the training split held out for early stopping
                      and choosing the best checkpoint, when the hyperparameters
                      are fixed or the library defaults. Ignored after a search,
                      which has already settled the number of epochs, so the
                      final model trains on the whole training split. Cut with
                      ``split_seed``. ``None`` trains on everything with no
                      early stopping. Default 0.1.
            n_jobs: CPU workers for loading text batches, in training, the
                    search and prediction. ``None`` (default) picks one per
                    spare core, and none on Apple Silicon, where worker
                    processes cost more than they save. Same name as the
                    tabular pipelines' CPU setting.
            resume: Pick up where an interrupted run in the same output_dir
                    stopped — the search's finished trials and the latest
                    training checkpoint. ``False`` starts over. Default True.
            use_fast: Use the HuggingFace fast (Rust) tokenizer. Default True.
                      Set False for models that lack a fast tokenizer (e.g. BlueBERT).
                    Default True.
            use_compile: Apply torch.compile() to the model for the final train_text() call.
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
                - "predictions": PredictionResult(top1, full, topk, id2label) from predict_text()
                - "train_metadata": metadata dict from train_text() — includes log_history,
                                    hyperparams, label2id, id2label, output_dir
                - "best_hyperparams": hyperparams used for final training
                - "label2id": label encoding map
                - "id2label": reverse label encoding map
                - "output_dir": Path to the root output directory
        """
        hp_kind, hp_fixed, hp_search = resolve_hyperparams(hyperparams)
        runtime_tracker = RuntimeTracker(
            self.output_dir,
            report_name="pipeline_runtime.json",
            metadata={
                "pipeline": "text_classifier",
                "model_name": self.model_name,
                "text_col": text_col,
                "label_col": label_col,
                "hyperparams_from": hp_kind,
            },
            logger_=logger,
        )
        set_determinism(deterministic)
        seed_everything(train_seed)
        # --- Step 1: split ---
        with runtime_tracker.stage(
            "split",
            details={
                "rows": len(df),
                "test_size": test_size,
                "split_seed": split_seed,
                "stratify": stratify,
            },
        ):
            self.train_df, self.test_df = split(
                df,
                text_col=text_col,
                label_col=label_col,
                test_size=test_size,
                random_state=split_seed,
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
                prepare_text_dataset(
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

        # --- Step 3: hyperparameters — searched, given, or the library's own ---
        if hp_kind == "search":
            hp_search = hp_search.with_defaults(
                n_trials=TEXT_SPEC_DEFAULTS["n_trials"],
                metric=TEXT_SPEC_DEFAULTS["optimize_metric"],
            )
            logger.info("Hyperparameters FROM SEARCH — %s.", hp_search.describe())
            from multimodalva.text.hpo import search_text  # noqa: PLC0415
            with runtime_tracker.stage(
                "hpo",
                details={
                    "n_trials": hp_search.n_trials,
                    "metric": hp_search.metric,
                    "backend": resolve_backend(hp_search.backend),
                },
            ):
                self.best_hyperparams, self.study = search_text(
                    hp_search,
                    resume=resolve_search_resume(hp_search, resume),
                    train_dataset=self.train_dataset,
                    label2id=self.label2id,
                    id2label=self.id2label,
                    model_name=self.model_name,
                    output_dir=self.output_dir / "hpo",
                    random_state=train_seed,
                    split_seed=split_seed,
                    use_lora=use_lora,
                    use_focal=use_focal,
                    gradient_checkpointing=gradient_checkpointing,
                    early_stopping_patience=early_stopping_patience,
                    use_fast=use_fast,
                )
            final_hyperparams = self.best_hyperparams
            # HPO already determined epochs; train on full training data.
            final_val_size = None
        else:
            _log_fixed_or_default(logger, self.model_name, hp_fixed)
            self.best_hyperparams = hp_fixed
            final_hyperparams = hp_fixed
            # train_text() carves an internal val split for early stopping.
            final_val_size = val_size

        # --- Step 4: final training ---
        with runtime_tracker.stage(
            "train",
            details={"resume": resume, "val_size": final_val_size},
        ):
            _, _, self.train_metadata = train_text(
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
                resume=resume,
                use_fast=use_fast,
                random_state=train_seed,
                split_seed=split_seed,
                use_compile=use_compile,  # NOT forwarded to HPO — compile overhead per trial is counterproductive
            )

        # train_text() never sees max_length (tokenisation happens upstream), so record it
        # here. Without it a later standalone push_to_hub() has nothing to read and the
        # model card would print a default that silently disagrees with the model.
        _record_tokenization(
            self.output_dir / "final", self.train_metadata, max_length, use_fast
        )

        rank, world_size = distributed_state()
        if world_size > 1 and rank != 0:
            logger.info(
                "Skipping predict_text() on non-zero rank %d/%d in DDP run.",
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
            self.predictions = predict_text(
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
        hyperparams: "dict | str | Optimize | None" = None,
        batch_size: int = 32,
        top_k: int = 3,
        use_lora: bool = False,
        use_focal: bool = False,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int | None = 4,
        val_size: float | None = 0.1,
        n_jobs: int | None = None,
        resume: bool = True,
        split_seed: int = 42,
        train_seed: int = 42,
        use_fast: bool = True,
        use_compile: bool = False,
        _runtime_tracker: RuntimeTracker | None = None,
    ) -> dict:
        """Run Steps 3–5 (HPO → train → predict) on pre-loaded datasets.

        Intended for callers (e.g. DataFusionClassifier) that have already
        populated ``self.train_dataset``, ``self.test_dataset``,
        ``self.label2id``, and ``self.id2label`` before calling this method.
        Skips the split and prepare_text_dataset steps.

        All parameters mirror ``run()``.
        """
        if self.train_dataset is None or self.label2id is None:
            raise RuntimeError(
                "_run_from_datasets() requires train_dataset, test_dataset, "
                "label2id, and id2label to be set on self before calling."
            )

        hp_kind, hp_fixed, hp_search = resolve_hyperparams(hyperparams)
        runtime_tracker = _runtime_tracker or RuntimeTracker(
            self.output_dir,
            report_name="pipeline_runtime.json",
            metadata={
                "pipeline": "text_classifier_from_datasets",
                "model_name": self.model_name,
                "label_col": label_col,
                "hyperparams_from": hp_kind,
            },
            logger_=logger,
        )
        # --- Step 3: hyperparameters — searched, given, or the library's own ---
        if hp_kind == "search":
            hp_search = hp_search.with_defaults(
                n_trials=TEXT_SPEC_DEFAULTS["n_trials"],
                metric=TEXT_SPEC_DEFAULTS["optimize_metric"],
            )
            logger.info("Hyperparameters FROM SEARCH — %s.", hp_search.describe())
            from multimodalva.text.hpo import search_text  # noqa: PLC0415
            with runtime_tracker.stage(
                "hpo",
                details={
                    "n_trials": hp_search.n_trials,
                    "metric": hp_search.metric,
                    "backend": resolve_backend(hp_search.backend),
                },
            ):
                self.best_hyperparams, self.study = search_text(
                    hp_search,
                    resume=resolve_search_resume(hp_search, resume),
                    train_dataset=self.train_dataset,
                    label2id=self.label2id,
                    id2label=self.id2label,
                    model_name=self.model_name,
                    output_dir=self.output_dir / "hpo",
                    random_state=train_seed,
                    split_seed=split_seed,
                    use_lora=use_lora,
                    use_focal=use_focal,
                    gradient_checkpointing=gradient_checkpointing,
                    early_stopping_patience=early_stopping_patience,
                    use_fast=use_fast,
                )
            final_hyperparams = self.best_hyperparams
            final_val_size = None
        else:
            _log_fixed_or_default(logger, self.model_name, hp_fixed)
            self.best_hyperparams = hp_fixed
            final_hyperparams = hp_fixed
            final_val_size = val_size

        # --- Step 4: final training ---
        with runtime_tracker.stage(
            "train",
            details={"resume": resume, "val_size": final_val_size},
        ):
            _, _, self.train_metadata = train_text(
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
                resume=resume,
                use_fast=use_fast,
                random_state=train_seed,
                split_seed=split_seed,
                use_compile=use_compile,  # NOT forwarded to HPO — compile overhead per trial is counterproductive
            )

        rank, world_size = distributed_state()
        if world_size > 1 and rank != 0:
            logger.info(
                "Skipping predict_text() on non-zero rank %d/%d in DDP run.",
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
            self.predictions = predict_text(
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
