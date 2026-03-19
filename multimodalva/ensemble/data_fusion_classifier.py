"""
Ensemble strategy 1 — Data-level fusion — classifier.

Receives the fused text produced by ``build_fused_text()`` (``data_fusion.py``)
and runs the standard text classification pipeline:
    split → prepare_dataset → [optimize →] train → predict

Long-context models are required because the combined tabular description and
free-text narrative routinely exceeds 512 tokens:

    "allenai/longformer-base-4096"   — general-purpose; up to 4 096 tokens
    "yikuan8/Clinical-Longformer"    — domain-adapted on clinical notes
    "google/bigbird-roberta-base"    — block-sparse attention; up to 4 096 tokens

Public API:
    DataFusionClassifier
        — end-to-end wrapper: fuse → split → tokenise → [hpo →] train → predict
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .data_fusion import build_fused_text, DEFAULT_SEPARATOR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default HPO search space for long-context models
# ---------------------------------------------------------------------------

# Passed to text.hpo.optimize() as the search_space argument when use_optimize=True
# and no custom search_space is provided.  Calibrated for Longformer / BigBird /
# Clinical-Longformer (sequence lengths 1 024–4 096).
#
# Key differences from the BERT default space (text/hpo.py):
#   learning_rate — tighter upper bound; long-context models destabilise above ~2e-5
#   batch_size    — 2 or 4; 1 024-token sequences require much more GPU memory
#   epochs        — up to 8; longer sequences often need more passes to converge
#   warmup_ratio  — wider range (0.05–0.2); longer warmup stabilises large-span attention
#   gradient_accumulation_steps — 4 or 8 to reach an effective batch of ~16–32
#
# These values override the corresponding entries in text/hpo.py's DEFAULT_SEARCH_SPACE.
# LoRA parameters (lora_r, lora_alpha) are merged in automatically by optimize()
# when use_lora=True.
DEFAULT_SEARCH_SPACE: dict = {
    # Long-context models are more sensitive to LR than BERT-base.
    # Upper bound capped at 2e-5 to avoid instability with 1024+ token sequences.
    "learning_rate": ("float_log", 5e-6, 2e-5),

    # Small physical batch sizes are typical for long sequences.
    # Effective batch = batch_size × gradient_accumulation_steps; target ~16.
    "batch_size": ("categorical", [2, 4]),

    # Long-context models often need more epochs than BERT on VA data.
    "epochs": ("categorical", [3, 5, 8]),

    "weight_decay": ("float", 0.0, 0.1),

    # Wider warmup stabilises training when attention spans the full 1 024–4 096 tokens.
    "warmup_ratio": ("float", 0.05, 0.2),

    # Accumulate gradients to compensate for small physical batch sizes.
    "gradient_accumulation_steps": ("categorical", [4, 8]),

    # Longformer / BigBird: 12 encoder layers.
    # Domain-adapted checkpoints (Clinical-Longformer) benefit from moderate freeze.
    # Generic checkpoints may need full fine-tuning (freeze_layers=0).
    "freeze_layers": ("categorical", [0, 2, 4]),
}


# ---------------------------------------------------------------------------
# DataFusionClassifier
# ---------------------------------------------------------------------------

class DataFusionClassifier:
    """End-to-end data-level fusion classifier.

    Fuses tabular features into the free-text narrative via
    ``build_fused_text()``, then runs the standard text classification
    pipeline (split → prepare_dataset → [optimize →] train → predict).

    Recommended models:
        "allenai/longformer-base-4096"   — general-purpose long-context
        "yikuan8/Clinical-Longformer"    — domain-adapted on clinical notes
        "google/bigbird-roberta-base"    — block-sparse attention

    Typical usage::

        from multimodalva.ensemble import DataFusionClassifier
        from multimodalva.ensemble.data_fusion import load_qdesc

        clf = DataFusionClassifier(
            model_name="allenai/longformer-base-4096",
            output_dir="runs/ensemble/data_fusion",
        )
        results = clf.run(
            df=df,
            text_col="open_narrative",
            feature_cols=[c for c in df.columns if c.startswith("i0")],
            label_col="cause",
            with_neg=True,
        )
    """

    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        output_dir: str | Path = "runs/ensemble/data_fusion",
    ):
        """Initialise DataFusionClassifier.

        Args:
            model_name: HuggingFace model name or local path.
                        Long-context models (Longformer, BigBird, Clinical-Longformer)
                        are required to accommodate the additional tabular text.
            output_dir: Root directory for all outputs.
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)

        # Populated after run()
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.label2id: dict | None = None
        self.id2label: dict | None = None
        self.predictions = None
        self.best_hyperparams: dict | None = None
        self.study = None   # Optuna study, set when use_optimize=True

    def run(
        self,
        df: pd.DataFrame,
        text_col: str,
        feature_cols: list[str],
        label_col: str,
        # --- tabular-to-text options ---
        qdesc: pd.DataFrame | None = None,
        templates: dict[str, str] | None = None,
        binary_map: dict | None = None,
        with_neg: bool = True,
        prefix_cols: dict[str, str] | None = None,
        yes_no_map: dict[str, str] | None = None,
        separator: str = DEFAULT_SEPARATOR,
        group_symptoms: bool = True,
        fused_col: str = "fused_text",
        # --- split ---
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        # --- tokenisation ---
        max_length: int = 1024,
        # --- training ---
        use_optimize: bool = False,
        n_trials: int = 30,
        optimize_metric: str = "csmf_accuracy",
        search_space: dict | None = None,
        hyperparams: dict | None = None,
        use_lora: bool = True,
        use_focal: bool = False,
        gradient_checkpointing: bool = True,
        early_stopping_patience: int | None = 2,
        resume_hpo: bool = True,
        resume_training: bool = True,
        # --- inference ---
        batch_size: int = 16,
        top_k: int = 3,
    ) -> dict:
        """Run the full data-level fusion pipeline.

        Steps executed in order:
            1. build_fused_text()    — tabular features → text, concat with narrative
            2. split()               — stratified train/test split on fused DataFrame
            3. prepare_dataset()     — tokenise for the chosen LM
            4. [optimize()]          — optional Optuna HPO
            5. train()               — fine-tune on the full training set
            6. predict()             — generate labels and probabilities on test set

        Args:
            df:            Input DataFrame containing both text and tabular columns.
            text_col:      Column with the free-text narrative.
            feature_cols:  Tabular columns to convert and prepend to the narrative.
            label_col:     Column containing cause-of-death labels.
            qdesc:         Question-description DataFrame (from load_qdesc()).
                           If None, auto-loaded from ``utils/qdesc.csv`` (cached).
                           Load manually: ``from multimodalva.ensemble.data_fusion import load_qdesc``
            templates:     Column-level format strings for tabular_to_text().
                           Example: {"age": "Patient age: {value} years."}
            binary_map:    Mapping for binary 0/1 indicators. Default {0: "no", 1: "yes"}.
            with_neg:      Include negative symptom phrases. Default True.
            prefix_cols:   Custom prefix column mapping (see tabular_to_text).
            yes_no_map:    Custom positive→negative verb map (see tabular_to_text).
            separator:     Text inserted between tabular description and narrative.
                           Default "\\n\\n".
            group_symptoms: Group symptoms sharing the same verb into one sentence.
                           Default True (~40–50% fewer tokens).
            fused_col:     Name of the temporary fused text column. Default "fused_text".
            test_size:     Fraction held out for testing. Default 0.2.
            random_state:  Random seed. Default 42.
            stratify:      Stratified split. Default True.
            max_length:    Tokeniser max length. Default 1 024 (Longformer/BigBird).
                           Use 4 096 for very long documents.
            use_optimize:  Run Optuna HPO before final training. Default False.
            n_trials:      Optuna trial count (use_optimize=True only). Default 30.
            optimize_metric: Metric to maximise during HPO. Default "csmf_accuracy".
            search_space:  Custom Optuna search space dict.
                           None → DEFAULT_SEARCH_SPACE (long-context calibrated defaults).
            hyperparams:   Fixed hyperparameter dict. Used when use_optimize=False.
            use_lora:      Apply LoRA adapters. Default True.
            use_focal:     Use focal loss during HPO trials (use_optimize=True only).
                           Merges FOCAL_SEARCH_SPACE (focal_gamma, class_weights) and
                           injects loss_type="focal" into every trial. Default False.
                           For non-HPO focal loss, pass hyperparams={"loss_type": "focal",
                           "focal_gamma": 2.0, "class_weights": "effective_n"}.
            gradient_checkpointing: Enable gradient checkpointing. Default True
                           (strongly recommended for long-context models).
            early_stopping_patience: Early stopping patience. Default 2.
            resume_hpo:    Resume an existing Optuna study if present (load_if_exists).
                           Default True.
            resume_training: Resume final training from the latest checkpoint in
                           output_dir/final/ if one exists. Default True.
            batch_size:    Inference batch size. Default 16.
            top_k:         Number of top classes in topk output. Default 3.

        Returns:
            dict with keys:
                "predictions":      PredictionResult(top1, full, topk, id2label)
                "train_metadata":   metadata dict from train()
                "best_hyperparams": hyperparams used for final training
                "label2id":         label encoding map
                "id2label":         reverse label encoding map
                "output_dir":       Path to the root output directory
        """
        # ------------------------------------------------------------------
        # Step 1 — Build fused text
        # ------------------------------------------------------------------
        logger.info(
            "DataFusionClassifier: fusing %d tabular columns into narrative text.",
            len(feature_cols),
        )
        fused_series = build_fused_text(
            df,
            text_col=text_col,
            feature_cols=feature_cols,
            qdesc=qdesc,
            templates=templates,
            binary_map=binary_map,
            with_neg=with_neg,
            prefix_cols=prefix_cols,
            yes_no_map=yes_no_map,
            separator=separator,
            group_symptoms=group_symptoms,
        )

        # Build a minimal DataFrame: fused text + label only.
        fused_df = df[[label_col]].copy()
        fused_df[fused_col] = fused_series.values

        # ------------------------------------------------------------------
        # Step 2 — Split
        # ------------------------------------------------------------------
        from ..utils.split import split as _split

        train_df, test_df = _split(
            fused_df,
            label_col=label_col,
            text_col=fused_col,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        self.train_df = train_df
        self.test_df  = test_df
        logger.info("Split: %d train / %d test rows.", len(train_df), len(test_df))

        # ------------------------------------------------------------------
        # Step 3 — Tokenise
        # ------------------------------------------------------------------
        from ..text.dataset import prepare_dataset

        train_ds, test_ds, label2id, id2label = prepare_dataset(
            train_df,
            test_df,
            text_col=fused_col,
            label_col=label_col,
            model_name=self.model_name,
            max_length=max_length,
        )
        self.label2id = label2id
        self.id2label = id2label

        # ------------------------------------------------------------------
        # Step 4 — HPO (optional)
        # ------------------------------------------------------------------
        if use_optimize:
            from ..text.hpo import optimize

            # Use long-context defaults when no custom space is provided.
            effective_search_space = search_space if search_space is not None else DEFAULT_SEARCH_SPACE

            logger.info(
                "DataFusionClassifier: running Optuna HPO — %d trials, metric=%s.",
                n_trials, optimize_metric,
            )
            best_hyperparams, study = optimize(
                train_ds,
                label2id=label2id,
                id2label=id2label,
                model_name=self.model_name,
                output_dir=self.output_dir / "hpo",
                n_trials=n_trials,
                metric=optimize_metric,
                search_space=effective_search_space,
                random_state=random_state,
                use_lora=use_lora,
                use_focal=use_focal,
                gradient_checkpointing=gradient_checkpointing,
                early_stopping_patience=early_stopping_patience,
                load_if_exists=resume_hpo,
            )
            self.best_hyperparams = best_hyperparams
            self.study = study
            val_size = None   # epochs already determined by HPO; train on all data
        else:
            best_hyperparams = hyperparams or {}
            self.best_hyperparams = best_hyperparams
            val_size = 0.1    # carve internal val split for early stopping / eval

        # ------------------------------------------------------------------
        # Step 5 — Train
        # ------------------------------------------------------------------
        from ..text.train import train

        logger.info("DataFusionClassifier: training final model.")
        trainer, tokenizer, metadata = train(
            train_ds,
            label2id=label2id,
            id2label=id2label,
            model_name=self.model_name,
            output_dir=self.output_dir / "final",
            hyperparams=best_hyperparams,
            val_size=val_size,
            use_lora=use_lora,
            gradient_checkpointing=gradient_checkpointing,
            early_stopping_patience=early_stopping_patience,
            resume=resume_training,
        )

        # ------------------------------------------------------------------
        # Step 6 — Predict
        # ------------------------------------------------------------------
        from ..text.predict import predict

        logger.info("DataFusionClassifier: generating predictions on test set.")
        result = predict(
            output_dir=None,      # in-memory mode — no disk reload needed
            test_dataset=test_ds,
            batch_size=batch_size,
            top_k=top_k,
            save_dir=self.output_dir / "predictions",
            model=trainer.model,
            tokenizer=tokenizer,
            id2label=id2label,
        )
        self.predictions = result

        return {
            "predictions":      result,
            "train_metadata":   metadata,
            "best_hyperparams": best_hyperparams,
            "label2id":         label2id,
            "id2label":         id2label,
            "output_dir":       self.output_dir,
        }
