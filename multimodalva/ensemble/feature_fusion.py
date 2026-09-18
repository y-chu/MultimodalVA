"""
Ensemble strategy 2 — Feature-level fusion via AutoGluon AutoMM.

Feeds raw text and tabular features jointly to AutoGluon's MultiModalPredictor
(AutoMM), which handles tokenisation, embedding, tabular encoding, and
cross-modal fusion in one unified model.

Pipeline
--------
DataFrame
  → split()                        — stratified train/test split
  → MultiModalPredictor.fit()      — joint text+tabular training
  → MultiModalPredictor.predict_proba() — probability predictions
  → _assemble_prediction_result()  — same PredictionResult as text/tabular

Fusion strategies (``fusion_strategy``)
---------------------------------------
    "default"       AutoMM preset picks the roster.  With preset="best_quality"
                    it selected ft_transformer + fusion_mlp here — MLP fusion,
                    no cross-modal attention.
    "concat"        hf_text + numerical_mlp + categorical_mlp + fusion_mlp.
    "attention"     hf_text + numerical_mlp + categorical_mlp +
                    fusion_transformer — self-attention over the joint
                    text-CLS + tabular token sequence.
    "attention_ft"  hf_text + ft_transformer + fusion_transformer — 2x2 cell
                    complement to "attention" and "default".
    "text_only"     hf_text only (ablation).
    "tabular_only"  numerical_mlp + categorical_mlp + fusion_mlp (ablation).

AutoMM drops branches for data types not present, so the fitted roster in
automm_model/config.yaml can be shorter than listed above.

Text backbone
-------------
``model_name`` is resolved by :func:`multimodalva.text.models.resolve_model_name`,
the same resolver used by the text pipeline: a package alias
(e.g. ``"bioclinicalbert"``, ``"biomedroberta"``), a remote key
(``"roberta-pm"``), a Hugging Face Hub ID, or a local model directory.
BioMed-RoBERTa (``"biomedroberta"``) and RoBERTa-PM (``"roberta-pm"``) are
different models; see :mod:`multimodalva.text.models`.

If ``hyperparameters`` passed to :meth:`FeatureFusionClassifier.run` contains
``"model.hf_text.checkpoint_name"``, that checkpoint is used instead of
``model_name``. The checkpoint actually used is written to
``training_metadata.json`` as ``checkpoint_name``.

Saving and reloading
--------------------
AutoGluon AutoMM saves to a structured directory (PyTorch weights + config +
tokenizer) — NOT a pickle/joblib file.  Artifacts are written to
``output_dir/automm_model/`` automatically during training.  Reload with::

    from autogluon.multimodal import MultiModalPredictor
    predictor = MultiModalPredictor.load("runs/ensemble/feature_fusion/automm_model")

Dependencies
------------
    pip install autogluon.multimodal

Public API
----------
    FUSION_STRATEGIES         — registry of supported strategy names
    DEFAULT_HPO_SPACE         — default search space for use_hpo=True (tuple format)
    TEXT_BACKBONE_MODELS      — shorthand → HuggingFace ID map (from text pipeline)
    FeatureFusionClassifier   — end-to-end wrapper
"""

from __future__ import annotations

import json
import logging
import os
import platform
import ssl
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.split import split
from ..utils.types import PredictionResult
from ..text.models import SUPPORTED_MODELS as TEXT_BACKBONE_MODELS, resolve_model_name

logger = logging.getLogger(__name__)


def _patch_automm_gpu_logging() -> None:
    """Make AutoMM GPU logging safe on non-NVIDIA machines.

    AutoGluon AutoMM logs GPU memory via ``nvidia_smi.nvmlInit()`` inside
    ``autogluon.multimodal.utils.log.get_gpu_message``. On Apple Silicon or
    any CPU-only machine where NVML is unavailable, that logging step can
    raise and abort training even though the model is not using CUDA.

    We patch both the canonical utility function and the copy imported into
    ``autogluon.multimodal.learners.base`` so AutoMM falls back to a minimal
    GPU-count message instead of crashing.
    """
    try:
        from autogluon.multimodal.learners import base as ag_base
        from autogluon.multimodal.utils import log as ag_log
    except ImportError:
        return

    original = ag_log.get_gpu_message
    if getattr(original, "_multimodalva_safe", False):
        return

    def _fallback_message(detected_num_gpus: int, used_num_gpus: int) -> str:
        return (
            f"GPU Count: {detected_num_gpus}\n"
            f"GPU Count to be Used: {used_num_gpus}\n"
            "GPU details unavailable (non-NVIDIA or NVML unavailable)\n"
        )

    def _safe_get_gpu_message(detected_num_gpus: int, used_num_gpus: int, strategy: str) -> str:
        try:
            return original(detected_num_gpus, used_num_gpus, strategy)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AutoMM GPU logging skipped: %s", exc)
            return _fallback_message(detected_num_gpus, used_num_gpus)

    _safe_get_gpu_message._multimodalva_safe = True  # type: ignore[attr-defined]
    ag_log.get_gpu_message = _safe_get_gpu_message
    ag_base.get_gpu_message = _safe_get_gpu_message


# ---------------------------------------------------------------------------
# Fusion strategy registry
# ---------------------------------------------------------------------------

#: Maps strategy name → ``model.names`` list passed to AutoMM.
#: ``None`` means "let the preset decide" (recommended default).
FUSION_STRATEGIES: dict[str, list[str] | None] = {
    # AutoMM preset decides the roster; best_quality picked fusion_mlp here.
    "default":      None,
    # MLP over concatenated text CLS + tabular embeddings.
    "concat":       ["hf_text", "numerical_mlp", "categorical_mlp", "fusion_mlp"],
    # Self-attention over the joint modality token sequence.
    "attention":    ["hf_text", "numerical_mlp", "categorical_mlp", "fusion_transformer"],
    # ft_transformer tabular encoder + fusion_transformer — 2x2 cell complement to "attention" and "default".
    "attention_ft": ["hf_text", "ft_transformer", "fusion_transformer"],
    # Ablation — text only (tabular features ignored).
    "text_only":    ["hf_text"],
    # Ablation — tabular only (narrative ignored).
    "tabular_only": ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
}

#: Default HPO search space for ``use_hpo=True``.
#: Uses the same ``(type, *args)`` tuple format as the text and tabular pipelines.
#: Pass ``hpo_search_space`` to ``run()`` to override or extend individual keys.
DEFAULT_HPO_SPACE: dict = {
    # Learning rate — most impactful; log-scale between 1e-5 and 1e-3.
    "optimization.learning_rate":        ("float_log", 1e-5, 1e-3),
    # Training epochs — controls overfitting; 3–15 covers most VA dataset sizes.
    "optimization.max_epochs":           ("int",        3,   15),
    # Batch size — stability vs. speed tradeoff.
    "env.batch_size":                    ("categorical", [8, 16, 32]),
    # Weight decay — L2 regularisation; useful for small clinical datasets.
    "optimization.weight_decay":         ("float_log", 1e-6, 1e-2),
    # Checkpoint averaging strategy.
    "optimization.top_k_average_method": ("categorical", ["best", "greedy_soup"]),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint(model_name: str, cache_dir: str | Path | None = None) -> str:
    """Return a loadable Hugging Face model ID or local path for ``model_name``.

    Thin wrapper around :func:`multimodalva.text.models.resolve_model_name`, so
    feature fusion and the text pipeline interpret model names identically.
    """
    return resolve_model_name(model_name, cache_dir=cache_dir)


def _build_automm_hyperparameters(
    checkpoint_name: str,
    fusion_strategy: str,
    extra: dict | None,
) -> dict:
    """Build the AutoMM hyperparameters dict.

    Sets ``model.hf_text.checkpoint_name`` and overrides ``model.names`` when
    ``fusion_strategy`` is not ``"default"``.  ``extra`` is merged last and
    takes precedence over all resolved values.

    When ``fusion_strategy`` excludes the text branch (``"tabular_only"``),
    the ``model.hf_text.*`` override is skipped: AutoGluon strips the
    ``hf_text`` block from the active config in that case, and overriding a
    missing block raises a config-merge KeyError.
    """
    model_names = FUSION_STRATEGIES[fusion_strategy]
    text_in_use = model_names is None or "hf_text" in model_names

    hp: dict = {}
    if text_in_use:
        hp["model.hf_text.checkpoint_name"] = checkpoint_name
    if model_names is not None:
        hp["model.names"] = model_names
    if extra:
        hp.update(extra)
    return hp


def _to_ag_space(search_space: dict) -> dict:
    """Convert ``(type, *args)`` tuple specs to ``autogluon.core.space.*`` objects.

    Uses the same tuple format as the text and tabular HPO pipelines::

        ("float",       lo, hi)       → ag.space.Real(lo, hi)
        ("float_log",   lo, hi)       → ag.space.Real(lo, hi, log=True)
        ("int",         lo, hi)       → ag.space.Int(lo, hi)
        ("int_log",     lo, hi)       → ag.space.Int(lo, hi, log=True)
        ("categorical", [v1, v2, ...])→ ag.space.Categorical(v1, v2, ...)

    Args:
        search_space: Dict mapping AutoMM hyperparameter key → tuple spec.

    Returns:
        Dict with same keys but ``ag.space.*`` values, ready to be merged into
        the AutoMM ``hyperparameters`` dict passed to ``fit()``.

    Raises:
        ValueError: If a spec tuple has an unknown type string.
        ImportError: If ``autogluon.core`` is not installed.
    """
    import autogluon.core as ag  # noqa: PLC0415

    result: dict = {}
    for key, spec in search_space.items():
        if not isinstance(spec, tuple) or len(spec) < 2:
            raise ValueError(
                f"Invalid HPO spec for key {key!r}: expected a tuple, got {spec!r}."
            )
        kind, *args = spec
        if kind == "float":
            result[key] = ag.space.Real(*args)
        elif kind == "float_log":
            result[key] = ag.space.Real(*args, log=True)
        elif kind == "int":
            result[key] = ag.space.Int(*args)
        elif kind == "int_log":
            result[key] = ag.space.Int(*args, log=True)
        elif kind == "categorical":
            # args[0] is the list of choices
            result[key] = ag.space.Categorical(*args[0])
        else:
            raise ValueError(
                f"Unknown HPO spec type {kind!r} for key {key!r}. "
                "Supported: 'float', 'float_log', 'int', 'int_log', 'categorical'."
            )
    return result


def _assemble_prediction_result(
    proba_df: pd.DataFrame,
    true_labels: list,
    id2label: dict,
    top_k: int,
) -> PredictionResult:
    """Build a PredictionResult from AutoGluon ``predict_proba()`` output.

    Args:
        proba_df:    DataFrame (rows=samples, columns=class label strings).
        true_labels: Ground-truth label strings, one per row.
        id2label:    Integer ID → label string mapping.
        top_k:       Number of top classes for the topk DataFrame.

    Returns:
        PredictionResult(top1, full, topk, id2label)
    """
    n_classes = len(id2label)

    # Build prob matrix in canonical class-ID order.
    # proba_df columns are label strings; AutoGluon's column order may differ
    # from our label2id ordering, so we re-index explicitly.
    prob_cols = []
    for i in range(n_classes):
        lbl = id2label[i]
        if lbl in proba_df.columns:
            prob_cols.append(proba_df[lbl].values.astype(float))
        else:
            logger.warning(
                "Class %r (id=%d) missing from predict_proba output; filling with 0.0",
                lbl, i,
            )
            prob_cols.append(np.zeros(len(proba_df), dtype=float))

    prob_matrix = np.column_stack(prob_cols)  # shape: (n_samples, n_classes)

    # ---- top1 ---------------------------------------------------------------
    top1_idx   = np.argmax(prob_matrix, axis=1)
    top1_probs = prob_matrix[np.arange(len(prob_matrix)), top1_idx]
    top1_df = pd.DataFrame({
        "true_label":      true_labels,
        "predicted_label": [id2label[i] for i in top1_idx],
        "predicted_prob":  top1_probs,
    })

    # ---- full ---------------------------------------------------------------
    full_data: dict = {"true_label": true_labels}
    for i in range(n_classes):
        full_data[f"prob_{i}"] = prob_matrix[:, i]
    full_df = pd.DataFrame(full_data)

    # ---- topk ---------------------------------------------------------------
    k = min(top_k, n_classes)
    top_indices = np.argsort(prob_matrix, axis=1)[:, ::-1][:, :k]
    topk_data: dict = {"true_label": true_labels}
    for j in range(k):
        col_idx = top_indices[:, j]
        topk_data[f"top{j + 1}_label"] = [id2label[i] for i in col_idx]
        topk_data[f"top{j + 1}_prob"]  = prob_matrix[np.arange(len(prob_matrix)), col_idx]
    topk_df = pd.DataFrame(topk_data)

    return PredictionResult(top1=top1_df, full=full_df, topk=topk_df, id2label=id2label)


# ---------------------------------------------------------------------------
# Environment setup helpers (SSL + NLTK — required by AutoGluon AutoMM)
# ---------------------------------------------------------------------------

def _ensure_nltk_deps() -> None:
    """Fix macOS SSL certificate errors and ensure NLTK data required by AutoMM.

    AutoGluon AutoMM depends on NLTK corpora (wordnet, omw-1.4) and tokenizers
    (punkt / punkt_tab).  On macOS the system Python SSL bundle may not include
    the root certificates needed to reach the NLTK data server; this function
    patches the default HTTPS context before attempting any downloads.

    Safe to call on Linux / Windows — the SSL patch is skipped on non-macOS
    platforms and all NLTK downloads are no-ops when the data is already present.
    """
    # --- macOS SSL patch ---------------------------------------------------
    # Python installed via python.org ships without the macOS keychain certs.
    # Patching ssl._create_default_https_context is the standard workaround
    # (also applied by the Install Certificates.command bundled with python.org).
    if platform.system() == "Darwin":
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            logger.debug("macOS SSL: patched default HTTPS context to unverified.")
        except AttributeError:
            pass  # ssl module doesn't support this on this build — skip silently

    # --- NLTK data ---------------------------------------------------------
    try:
        import nltk  # noqa: PLC0415 — optional dep, only needed for AutoMM
    except ImportError:
        return  # nltk not installed; AutoMM will surface its own error if needed

    # Map dataset name → nltk.data.find() path prefix.
    _needed = {
        "wordnet":  "corpora/wordnet",
        "omw-1.4":  "corpora/omw-1.4",
        "punkt":    "tokenizers/punkt",
    }
    # punkt_tab exists only in NLTK >= 3.9; older find() mangles the path and
    # raises OSError instead of LookupError.
    _ver = tuple(int(x) for x in nltk.__version__.split(".")[:2] if x.isdigit())
    if _ver >= (3, 9):
        _needed["punkt_tab"] = "tokenizers/punkt_tab"
    for name, find_path in _needed.items():
        try:
            nltk.data.find(find_path)
        except (LookupError, OSError):
            logger.info("Downloading NLTK data: %s", name)
            nltk.download(name, quiet=True)


def _load_saved_prediction_result(output_dir: Path) -> PredictionResult:
    """Load saved prediction artifacts from ``output_dir``.

    Expects the standard files written by :meth:`FeatureFusionClassifier.run`:
    ``predictions_top1.csv``, ``predictions_full.csv``, ``predictions_topk.csv``,
    and ``id2label.json``.
    """
    top1 = pd.read_csv(output_dir / "predictions_top1.csv")
    full = pd.read_csv(output_dir / "predictions_full.csv")
    topk = pd.read_csv(output_dir / "predictions_topk.csv")
    with open(output_dir / "id2label.json") as fh:
        raw_id2label = json.load(fh)
    id2label = {int(k): v for k, v in raw_id2label.items()}
    return PredictionResult(top1=top1, full=full, topk=topk, id2label=id2label)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class FeatureFusionClassifier:
    """End-to-end feature-level fusion classifier via AutoGluon AutoMM.

    AutoMM internally tokenises text, encodes tabular features, and fuses
    both modalities through the chosen fusion architecture — no manual feature
    engineering or two-stage training required.

    Typical usage::

        clf = FeatureFusionClassifier(
            model_name="bioclinicalbert",
            fusion_strategy="attention",
            output_dir="runs/ensemble/feature_fusion",
        )
        results = clf.run(
            df=df,
            text_col="narrative",
            feature_cols=["age_group", "sex", "fever", "cough", "weight_loss"],
            label_col="cause",
            time_limit=3600,
        )
        print(results["predictions"].top1.head())

        # Reload the trained model later (AutoGluon native format, not pickle):
        from autogluon.multimodal import MultiModalPredictor
        predictor = MultiModalPredictor.load(
            "runs/ensemble/feature_fusion/automm_model"
        )
        proba = predictor.predict_proba(new_df)

    Fusion strategy guide for VA data
    ----------------------------------
    ``"attention"``
        fusion_transformer: self-attention over the joint sequence of the
        text CLS token + tabular embeddings.

    ``"concat"``
        fusion_mlp over concatenated embeddings.  Faster to train.

    ``"default"``
        AutoMM preset decides.  ``"best_quality"`` picked ft_transformer +
        fusion_mlp here — not the same as ``"attention"``.

    ``"attention_ft"``
        ft_transformer tabular encoder + fusion_transformer — 2x2 cell
        complement to ``"attention"`` and ``"default"``.

    ``"text_only"`` / ``"tabular_only"``
        Ablation: disable one modality to isolate its contribution.

    Text backbone options (same as text-only pipeline)
    ---------------------------------------------------
    Shorthand           HuggingFace checkpoint
    ----------------    -----------------------------------------------
    ``"bioclinicalbert"`` emilyalsentzer/Bio_ClinicalBERT        ← default
    ``"bert"``            bert-base-uncased
    ``"biobert"``         dmis-lab/biobert-base-cased-v1.2
    ``"bluebert"``        bionlp/bluebert_pubmed_mimic_uncased_...
    ``"biomedbert"``      microsoft/BiomedNLP-BiomedBERT-base-...
    ``"clinicalbert"``    medicalai/ClinicalBERT
    ``"biomedroberta"``   allenai/biomed_roberta_base (BioMed-RoBERTa)
    ``"roberta-pm"``      RoBERTa-base-PM-M3-Voc-distill-hf (RoBERTa-PM;
                          downloaded, not on the Hub)
    ``"bioelectra"``      kamalkraj/bioelectra-base-discriminator-pubmed
    ``"longformer"``      allenai/longformer-base-4096
    ``"bigbird"``         google/bigbird-roberta-base

    BioMed-RoBERTa and RoBERTa-PM are different models with different
    vocabularies; see :mod:`multimodalva.text.models`.  Any full Hugging Face
    Hub ID or local model directory can also be passed directly.

    Attributes (populated after run())
    -----------------------------------
    predictor :        Fitted ``MultiModalPredictor`` instance.
    train_df :         Training split DataFrame.
    test_df :          Test split DataFrame.
    label2id :         Label → integer ID mapping.
    id2label :         Integer ID → label mapping.
    predictions :      ``PredictionResult`` from the test set.
    best_hpo_config :  Best hyperparameter config found during HPO
                       (``None`` when ``use_hpo=False``).
    """

    def __init__(
        self,
        output_dir: str | Path = "runs/ensemble/feature_fusion",
        model_name: str = "bioclinicalbert",
        fusion_strategy: str = "default",
        preset: str = "best_quality",
        eval_metric: str = "f1_macro",
    ):
        """Initialise FeatureFusionClassifier.

        Args:
            output_dir:       Root directory for AutoMM checkpoints and outputs.
                              Model saved to ``output_dir/automm_model/``.
            model_name:       Text backbone: package alias, remote key, Hugging
                              Face Hub ID, or local model directory.
                              Default ``"bioclinicalbert"``
                              (emilyalsentzer/Bio_ClinicalBERT).  Ignored when
                              ``run(hyperparameters=...)`` sets
                              ``"model.hf_text.checkpoint_name"``.
            fusion_strategy:  Cross-modal fusion architecture.  One of:
                              ``"default"``, ``"concat"``, ``"attention"``,
                              ``"attention_ft"``, ``"text_only"``,
                              ``"tabular_only"``.
                              Default ``"default"`` (preset decides).
            preset:           AutoMM quality preset:
                              ``"medium_quality"`` — fastest;
                              ``"high_quality"``   — balanced;
                              ``"best_quality"``   — most accurate (default).
            eval_metric:      Metric used to select the best checkpoint.
                              Options: ``"accuracy"``, ``"f1_macro"`` (default),
                              ``"f1_weighted"``, ``"log_loss"``,
                              ``"roc_auc_ovo_macro"``.
                              ``"f1_macro"`` recommended for imbalanced VA data.
        """
        if fusion_strategy not in FUSION_STRATEGIES:
            raise ValueError(
                f"Unknown fusion_strategy {fusion_strategy!r}. "
                f"Valid options: {list(FUSION_STRATEGIES)}."
            )

        self.output_dir      = Path(output_dir)
        self.model_name      = model_name
        self.fusion_strategy = fusion_strategy
        self.preset          = preset
        self.eval_metric     = eval_metric

        # Populated after run()
        self.train_df:        pd.DataFrame | None      = None
        self.test_df:         pd.DataFrame | None      = None
        self.predictor                                  = None
        self.label2id:        dict | None               = None
        self.id2label:        dict | None               = None
        self.predictions:     PredictionResult | None   = None
        self.best_hpo_config: dict | None               = None

    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        text_col: str,
        feature_cols: list[str],
        label_col: str,
        # --- split ---
        test_size: float = 0.2,
        random_state: int = 42,
        automm_seed: int | None = None,
        stratify: bool = True,
        split_col: str | None = None,
        # --- AutoMM training ---
        time_limit: int = 3600,
        hyperparameters: dict | None = None,
        # --- built-in HPO ---
        use_hpo: bool = False,
        n_hpo_trials: int = 10,
        hpo_search_space: dict | None = None,
        hpo_scheduler: str = "local",
        hpo_searcher: str = "bayes",
        resume: bool = True,
        # --- inference ---
        top_k: int = 3,
    ) -> dict:
        """Run the full feature-level fusion pipeline.

        Steps:
            1. ``split()``                  — stratified train/test split
            2. Build label maps             — union of train + test labels
            3. ``MultiModalPredictor.fit()``— joint text+tabular training
            4. ``predict_proba()``          — probability predictions
            5. Assemble ``PredictionResult``— same format as other pipelines
            6. Save label maps + metadata  — JSON files in ``output_dir``

        AutoMM automatically saves checkpoints to ``output_dir/automm_model/``
        during training.

        Args:
            df:              Input DataFrame with text, tabular, and label columns.
            text_col:        Column containing free-text narratives.
            feature_cols:    Columns used as features (may include ``text_col``).
                             AutoMM detects column types from dtype automatically:
                             object/str → text or categorical, numeric → numerical.
            label_col:       Column containing cause-of-death labels.
            test_size:       Fraction held out for testing.  Default 0.2.
            random_state:    Split seed.  Default 42.
            automm_seed:     AutoMM trainer seed passed to fit(); None uses
                             AutoMM's default (0). Varies training only.
            stratify:        Stratified split.  Default True.
            time_limit:      Training time budget in seconds.  Default 3 600
                             (1 hour).  Increase to 3–6 hours for
                             ``"best_quality"`` or large datasets.
            hyperparameters: AutoMM hyperparameters merged over the resolved
                             backbone + fusion settings.  Scalar values only
                             (fixed training run).  A
                             ``"model.hf_text.checkpoint_name"`` entry replaces
                             the backbone given by ``model_name``.  Examples::

                                 # Longer context for verbose narratives
                                 {"model.hf_text.max_text_len": 512}

                                 # Custom learning rate and epochs
                                 {"optimization.learning_rate": 1e-4,
                                  "optimization.max_epochs": 10}

                             ``None`` = resolved defaults only.
                             Ignored for keys that ``hpo_search_space`` overrides
                             when ``use_hpo=True``.
            use_hpo:         Run AutoMM's built-in HPO (Ray Tune + Bayes by
                             default) over ``DEFAULT_HPO_SPACE``.
                             ``False`` (default) = single fixed training run.
            n_hpo_trials:    Number of HPO trials.  Default 10.  Only used
                             when ``use_hpo=True``.
            hpo_search_space: Override or extend ``DEFAULT_HPO_SPACE`` with
                             additional keys.  Uses the same tuple format::

                                 {"optimization.max_epochs": ("int", 5, 20),
                                  "env.batch_size": ("categorical", [8, 16])}

                             Merged over ``DEFAULT_HPO_SPACE``; caller's keys win.
                             ``None`` = use ``DEFAULT_HPO_SPACE`` unchanged.
                             Only used when ``use_hpo=True``.
            hpo_scheduler:   Ray Tune scheduler.  ``"local"`` (default, single
                             machine) or ``"ray"`` (distributed cluster).
                             Only used when ``use_hpo=True``.
            hpo_searcher:    Search algorithm.  ``"bayes"`` (default, Bayesian
                             optimisation), ``"random"``, or ``"grid"``.
                             Only used when ``use_hpo=True``.
            resume:          If ``True`` (default), reuse an existing completed
                             AutoMM run in ``output_dir`` when the saved
                             predictions match the current deterministic split.
                             If a checkpoint exists but prediction CSVs are
                             missing, load the checkpoint and regenerate the
                             downstream outputs instead of calling ``fit()``.
            top_k:           Number of top classes in topk output.  Default 3.

        Returns:
            dict with keys:

            ``"predictions"``
                :class:`~multimodalva.utils.types.PredictionResult`
                ``(top1, full, topk, id2label)`` on the test set.
            ``"label2id"``
                Label → integer ID mapping.
            ``"id2label"``
                Integer ID → label mapping.
            ``"output_dir"``
                :class:`pathlib.Path` to the root output directory.
            ``"best_hpo_config"``
                Best hyperparameter config from HPO (``None`` when
                ``use_hpo=False``).

        Raises:
            ImportError: If ``autogluon.multimodal`` is not installed.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self.output_dir / "automm_model"

        # --- Step 1: split -----------------------------------------------
        self.train_df, self.test_df = split(
            df,
            label_col=label_col,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            split_col=split_col,
        )
        logger.info(
            "Split: %d train / %d test samples.",
            len(self.train_df), len(self.test_df),
        )

        # --- Step 2: build label maps (union of train + test) ------------
        all_labels = sorted(
            set(self.train_df[label_col]) | set(self.test_df[label_col])
        )
        label2id: dict = {lbl: i for i, lbl in enumerate(all_labels)}
        id2label: dict = {i: lbl for i, lbl in enumerate(all_labels)}
        self.label2id = label2id
        self.id2label = id2label
        logger.info("Label map: %d classes.", len(label2id))

        # --- Step 3: prepare input DataFrames ----------------------------
        # Deduplicate feature_cols preserving order; always include text_col.
        seen: set = set()
        ordered_cols: list[str] = []
        for c in ([text_col] + list(feature_cols)):
            if c not in seen and c != label_col:
                seen.add(c)
                ordered_cols.append(c)

        input_train = self.train_df[ordered_cols + [label_col]].copy()
        input_test  = self.test_df[ordered_cols + [label_col]].copy()

        # --- Step 4: build AutoMM hyperparameters ------------------------
        # An explicit model.hf_text.checkpoint_name takes precedence over
        # model_name, and model_name is then not resolved (nor downloaded).
        explicit_checkpoint = (hyperparameters or {}).get("model.hf_text.checkpoint_name")
        if explicit_checkpoint:
            checkpoint_name = resolve_model_name(str(explicit_checkpoint))
            if checkpoint_name != explicit_checkpoint:
                hyperparameters = {**hyperparameters, "model.hf_text.checkpoint_name": checkpoint_name}
            logger.info(
                "Using hyperparameters['model.hf_text.checkpoint_name']=%s as the "
                "text backbone (model_name=%r is not used).",
                checkpoint_name, self.model_name,
            )
        else:
            checkpoint_name = _resolve_checkpoint(self.model_name)
        automm_hp = _build_automm_hyperparameters(
            checkpoint_name, self.fusion_strategy, hyperparameters
        )
        logger.info(
            "AutoMM config — backbone: %s  fusion: %s  preset: %s  "
            "eval_metric: %s  time_limit: %ds",
            checkpoint_name, self.fusion_strategy, self.preset,
            self.eval_metric, time_limit,
        )
        logger.debug("AutoMM hyperparameters: %s", automm_hp)

        saved_top1_path = self.output_dir / "predictions_top1.csv"
        saved_full_path = self.output_dir / "predictions_full.csv"
        saved_topk_path = self.output_dir / "predictions_topk.csv"
        saved_id2label_path = self.output_dir / "id2label.json"
        metadata_path = self.output_dir / "training_metadata.json"
        has_saved_predictions = all(
            p.exists()
            for p in [saved_top1_path, saved_full_path, saved_topk_path, saved_id2label_path]
        )
        expected_true_labels = input_test[label_col].tolist()

        if resume and has_saved_predictions:
            try:
                saved_predictions = _load_saved_prediction_result(self.output_dir)
                saved_true_labels = saved_predictions.top1["true_label"].tolist()
                if saved_true_labels == expected_true_labels:
                    self.predictions = saved_predictions
                    self.id2label = saved_predictions.id2label
                    self.label2id = {lbl: idx for idx, lbl in saved_predictions.id2label.items()}
                    if metadata_path.exists():
                        with open(metadata_path) as fh:
                            saved_meta = json.load(fh)
                        self.best_hpo_config = saved_meta.get("best_hpo_config")
                    logger.info(
                        "Existing completed AutoMM run detected at %s — "
                        "reusing saved predictions and skipping fit().",
                        self.output_dir,
                    )
                    return {
                        "predictions": self.predictions,
                        "label2id": self.label2id,
                        "id2label": self.id2label,
                        "output_dir": self.output_dir,
                        "best_hpo_config": self.best_hpo_config,
                    }
                logger.warning(
                    "Existing predictions found at %s, but their true-label order "
                    "does not match the current split. They will not be reused.",
                    self.output_dir,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load saved predictions from %s; falling back to checkpoint "
                    "or fresh fit. Error: %s",
                    self.output_dir, exc,
                )

        try:
            from autogluon.multimodal import MultiModalPredictor
        except ImportError as exc:
            raise ImportError(
                "autogluon.multimodal is required for FeatureFusionClassifier. "
                "Install with:  pip install autogluon.multimodal"
            ) from exc

        _patch_automm_gpu_logging()

        # Fix macOS SSL cert errors and ensure NLTK corpora required by AutoMM.
        _ensure_nltk_deps()

        reuse_loaded_predictor = False
        if resume and model_dir.exists():
            try:
                predictor = MultiModalPredictor.load(str(model_dir))
                reuse_loaded_predictor = True
                logger.info(
                    "Existing AutoMM checkpoint detected at %s — loading it and "
                    "regenerating downstream outputs without fit().",
                    model_dir,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Existing AutoMM checkpoint at %s could not be loaded; "
                    "a fresh fit will be attempted. Error: %s",
                    model_dir, exc,
                )

        # --- Step 5: train -----------------------------------------------
        # path= directs AutoMM to write all checkpoints to model_dir.
        if not reuse_loaded_predictor:
            if model_dir.exists():
                raise RuntimeError(
                    "Existing AutoMM model directory could not be safely reused: "
                    f"{model_dir}. If you want a fresh fit, use a different output "
                    "directory/run suffix or remove the stale AutoMM directory first."
                )
            predictor = MultiModalPredictor(
                label=label_col,
                problem_type="multiclass",
                eval_metric=self.eval_metric,
                path=str(model_dir),
            )

            if use_hpo:
                # Build search space: DEFAULT_HPO_SPACE merged with caller overrides.
                space: dict = {**DEFAULT_HPO_SPACE}
                if hpo_search_space:
                    space.update(hpo_search_space)
                ag_space = _to_ag_space(space)
                # Merge ag.space objects into the fixed hyperparameters dict;
                # search distributions take precedence over fixed scalars for the
                # same key.
                automm_hp.update(ag_space)
                logger.info(
                    "HPO enabled — %d trials, scheduler=%s, searcher=%s, "
                    "search keys: %s",
                    n_hpo_trials, hpo_scheduler, hpo_searcher,
                    list(ag_space),
                )
                predictor.fit(
                    train_data=input_train,
                    hyperparameters=automm_hp,
                    presets=self.preset,
                    time_limit=time_limit,
                    hyperparameter_tune_kwargs={
                        "num_trials": n_hpo_trials,
                        "scheduler":  hpo_scheduler,
                        "searcher":   hpo_searcher,
                    },
                )
                # Retrieve best config from fit summary; not all AutoMM versions
                # expose this, so we fall back gracefully.
                try:
                    summary = predictor.fit_summary()
                    self.best_hpo_config = summary.get("best_config")
                except Exception:  # noqa: BLE001
                    self.best_hpo_config = None
                if self.best_hpo_config:
                    _hpo_cfg_path = self.output_dir / "best_hpo_config.json"
                    with open(_hpo_cfg_path, "w") as fh:
                        json.dump(self.best_hpo_config, fh, indent=2, default=str)
                    logger.info("Best HPO config saved to %s", _hpo_cfg_path)
                else:
                    logger.info(
                        "Best HPO config not available via fit_summary(); "
                        "inspect the AutoMM model directory for trial results."
                    )
            else:
                predictor.fit(
                    train_data=input_train,
                    hyperparameters=automm_hp,
                    presets=self.preset,
                    time_limit=time_limit,
                    **({"seed": automm_seed} if automm_seed is not None else {}),
                )
                self.best_hpo_config = None
        elif metadata_path.exists():
            with open(metadata_path) as fh:
                saved_meta = json.load(fh)
            self.best_hpo_config = saved_meta.get("best_hpo_config")
        else:
            self.best_hpo_config = None

        self.predictor = predictor
        logger.info(
            "Training complete.  Model saved to %s\n"
            "  Reload: MultiModalPredictor.load('%s')",
            model_dir, model_dir,
        )

        # --- Step 6: predict ---------------------------------------------
        true_labels = input_test[label_col].tolist()
        # Drop label col before predict_proba so AutoGluon does not confuse it
        # with a feature (safe even if AutoGluon would ignore it automatically).
        proba_df = predictor.predict_proba(input_test.drop(columns=[label_col]))

        self.predictions = _assemble_prediction_result(
            proba_df, true_labels, id2label, top_k
        )

        # --- Step 7: save metadata ---------------------------------------
        with open(self.output_dir / "label2id.json", "w") as fh:
            json.dump(label2id, fh, indent=2)
        with open(self.output_dir / "id2label.json", "w") as fh:
            json.dump({str(k): v for k, v in id2label.items()}, fh, indent=2)

        metadata = {
            "output_dir":       str(self.output_dir),
            "model_name":       self.model_name,
            "checkpoint_name":  checkpoint_name,
            "fusion_strategy":  self.fusion_strategy,
            "preset":           self.preset,
            "eval_metric":      self.eval_metric,
            "time_limit":       time_limit,
            "use_hpo":          use_hpo,
            "n_hpo_trials":     n_hpo_trials if use_hpo else None,
            "hpo_scheduler":    hpo_scheduler if use_hpo else None,
            "hpo_searcher":     hpo_searcher  if use_hpo else None,
            "best_hpo_config":  self.best_hpo_config,
            "label2id":         label2id,
            "id2label":         {str(k): v for k, v in id2label.items()},
            "n_train":          len(self.train_df),
            "n_test":           len(self.test_df),
            "n_classes":        len(label2id),
            "feature_cols":     ordered_cols,
        }
        with open(self.output_dir / "training_metadata.json", "w") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info("Metadata saved to %s", self.output_dir)

        return {
            "predictions":    self.predictions,
            "label2id":       label2id,
            "id2label":       id2label,
            "output_dir":     self.output_dir,
            "best_hpo_config": self.best_hpo_config,
        }
