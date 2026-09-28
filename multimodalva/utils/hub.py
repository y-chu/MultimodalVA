"""
Publish trained model weights to the Hugging Face Hub for re-use.

The text and data-fusion pipelines save models in standard HuggingFace format
(``trainer.save_model()`` + ``tokenizer.save_pretrained()``), and LoRA adapters
are merged via ``merge_and_unload()`` before saving (see ``text/train.py``).  The
saved directory is therefore a plain ``AutoModelForSequenceClassification`` with
``id2label`` / ``label2id`` already written into ``config.json`` — directly
loadable by anyone and shown with real cause names in the Hub inference widget.

This module turns that saved directory into a published, documented model repo:

    from multimodalva.utils.hub import push_to_hub
    push_to_hub("runs/text_bert/final", "your-org/va-bert-cause-of-death")

``push_to_hub()`` works on **any** saved run directory — including past runs — so
it is not tied to a live classifier instance.  ``TextClassifier.run()`` and
``DataFusionClassifier.run()`` also expose ``push_to_hub=True`` / ``hub_repo_id=...``
to publish automatically at the end of a run; both paths funnel through here.

``huggingface_hub`` ships as a transitive dependency of ``transformers``, so no
extra install is normally required.  Authenticate once with ``huggingface-cli
login`` (or pass ``token=...``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from .metrics import CV_METRICS

# Metrics summarised on the model card when a top1 predictions CSV is available.
_CARD_METRICS = list(CV_METRICS)

# Files that should never be pushed — checkpoints, runtime telemetry, HPO logs.
_IGNORE_PATTERNS = [
    "checkpoint-*/*",
    "checkpoint-*",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth",
    "*.log",
    "*_runtime.json",
    "stage_timings.csv",
    "gpu_usage.csv",
    "runtime_report.json",
]

_TEXT_USAGE = """\
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo = "{repo_id}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

narrative = "The deceased was a 45 year old male who had fever and cough for two weeks ..."
inputs = tok(narrative, return_tensors="pt", truncation=True{max_length_arg})
with torch.no_grad():
    probs = model(**inputs).logits.softmax(-1)[0]
cause = model.config.id2label[int(probs.argmax())]
print(cause, float(probs.max()))
```
"""

_DOWNSTREAM_USAGE = """\
### Re-use as a pre-trained encoder

The fine-tuned encoder transfers to other tasks; the classification head is
specific to this cause list and does **not** transfer.

```python
# Feature extractor / warm-start encoder (head dropped):
from transformers import AutoModel
encoder = AutoModel.from_pretrained("{repo_id}")

# New classification task (encoder kept, fresh head):
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "{repo_id}", num_labels=NEW_NUM_CLASSES, ignore_mismatched_sizes=True,
)
```
"""

_LABEL_MEANING_NOTE = """\
> **On the labels.** Predictions use this model's cause vocabulary. Whether a
> cause label from another site, instrument or coding round means the same
> thing as yours is a judgement only you can make; check the list above against
> your own definitions before using or comparing the output.
"""

_FUSION_CAVEAT = """\
> **Note on training inputs.** This is an ordinary text classifier — it accepts
> any string and returns a cause. It was *fine-tuned* on **fused** text: the
> free-text narrative followed by structured (tabular) verbal-autopsy fields
> rendered into sentences (the MultimodalVA data-fusion recipe). You can feed it
> any text; for the closest match to its training distribution, if you also have
> the structured fields, build the same fused string with
> `multimodalva.ensemble.data_fusion.build_fused_text(...)`. If you only have a
> narrative, pass it directly.
"""


def _load_metadata(model_dir: Path) -> dict:
    """Load ``training_metadata.json`` if present, else assemble from JSON sidecars."""
    meta_path = model_dir / "training_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)

    meta: dict = {}
    for key, fname in (("label2id", "label2id.json"), ("id2label", "id2label.json"),
                       ("hyperparams", "hyperparams.json")):
        p = model_dir / fname
        if p.exists():
            with open(p) as f:
                meta[key] = json.load(f)
    return meta


def _validation_metrics(metadata: dict) -> dict:
    """Pull the last eval row from ``log_history`` (validation-set metrics)."""
    history = metadata.get("log_history") or []
    last_eval = {}
    for row in history:
        if any(k.startswith("eval_") for k in row):
            last_eval = {k: v for k, v in row.items() if k.startswith("eval_")}
    return last_eval


def _discover_test_metrics(model_dir: Path) -> dict | None:
    """Compute test metrics from a sibling ``predictions/*_top1.csv`` if one exists.

    ``run()`` saves the model under ``<root>/final`` and predictions under
    ``<root>/predictions``; for a standalone push of a past run we look there so
    the card carries real held-out test metrics without the caller supplying them.
    Analysis scripts save ``predictions_top1.csv`` directly in ``<root>/``, so
    that layout is checked as well.
    """
    import pandas as pd  # noqa: PLC0415

    candidates = list((model_dir.parent / "predictions").glob("*_top1.csv"))
    candidates += list(model_dir.parent.glob("*_top1.csv"))
    candidates += list(model_dir.glob("*_top1.csv"))
    if not candidates:
        return None
    try:
        top1 = pd.read_csv(candidates[0])
        if "predicted_label" not in top1 and "top1_label" in top1:
            top1 = top1.rename(columns={"top1_label": "predicted_label"})
        if "true_label" not in top1 or "predicted_label" not in top1:
            return None
        from multimodalva.utils.metrics import score_predictions  # noqa: PLC0415

        return {m: float(score_predictions(top1, m)) for m in _CARD_METRICS}
    except Exception as exc:  # pragma: no cover - best-effort card enrichment
        logger.warning("Could not compute test metrics from %s: %s", candidates[0], exc)
        return None


def _metrics_table(metrics: dict) -> str:
    rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())
    return "| Metric | Value |\n|---|---|\n" + rows


def _base_model_info(name: str) -> tuple[str | None, str]:
    """Return ``(hub_id, label)`` for the model a checkpoint was fine-tuned from.

    ``hub_id`` is written to the model card's ``base_model`` field and is
    ``None`` when the base model is not on the Hugging Face Hub (for example
    RoBERTa-PM, which is downloaded from its original release, or any local
    directory). ``label`` is the human-readable name used in the card text.
    """
    from multimodalva.text.models import REMOTE_MODELS, TEXT_MODELS

    if name in TEXT_MODELS:
        return TEXT_MODELS[name], TEXT_MODELS[name]
    if name in REMOTE_MODELS or "RoBERTa-base-PM-M3-Voc" in name:
        return None, "RoBERTa-PM (RoBERTa-base-PM-M3-Voc-distill-hf)"
    if Path(name).expanduser().is_absolute() or Path(name).expanduser().is_dir():
        return None, Path(name).name
    return name, name


def build_model_card(
    model_dir: str | Path,
    repo_id: str,
    *,
    model_kind: str = "text",
    base_model: str | None = None,
    metrics: dict | None = None,
    max_length: int | None = None,
    license: str = "mit",
) -> str:
    """Build a HuggingFace model-card (README.md) string with metrics and caveats.

    Args:
        model_dir:   Saved model directory (contains ``config.json``).
        repo_id:     Target Hub repo id (``org/name``) — used in usage snippets.
        model_kind:  ``"text"`` (raw-narrative model) or ``"data_fusion"``
                     (fused narrative+tabular model — adds the fused-input caveat).
        base_model:  Base checkpoint the model was fine-tuned from. Falls back to
                     ``model_name`` in ``training_metadata.json``.
        metrics:     Optional held-out test metrics to embed. If None, the card
                     tries a sibling ``predictions/*_top1.csv`` and falls back to
                     validation metrics from ``log_history``.
        max_length:  Tokenizer ``max_length`` shown in the usage snippet. When
                     None, it is read from ``max_length`` in
                     ``training_metadata.json``. If that is absent too (models
                     trained before it was recorded), the snippet omits the
                     argument and lets the tokenizer use the length saved in its
                     own config, rather than printing a length that may be wrong.
        license:     SPDX license id for the card frontmatter.

    Returns:
        Markdown string suitable for writing to ``README.md``.
    """
    model_dir = Path(model_dir)
    metadata = _load_metadata(model_dir)
    base_model = base_model or metadata.get("model_name", "unknown")
    base_model_hub_id, base_model_label = _base_model_info(base_model)
    id2label = metadata.get("id2label", {}) or {}
    causes = [id2label[k] for k in sorted(id2label, key=lambda x: int(x))] if id2label else []
    hyperparams = metadata.get("hyperparams", {}) or {}

    test_metrics = metrics or _discover_test_metrics(model_dir)
    val_metrics = _validation_metrics(metadata)

    is_fusion = model_kind == "data_fusion"
    title = "Verbal Autopsy cause-of-death classifier"
    if is_fusion:
        title += " (multimodal data fusion)"

    tags = ["text-classification", "verbal-autopsy", "cause-of-death", "medical"]
    if is_fusion:
        tags.append("multimodal")

    parts: list[str] = []
    # --- YAML frontmatter ---
    parts.append("---")
    parts.append(f"license: {license}")
    parts.append("library_name: transformers")
    parts.append("pipeline_tag: text-classification")
    if base_model_hub_id:
        parts.append(f"base_model: {base_model_hub_id}")
    parts.append("tags:")
    parts.extend(f"  - {t}" for t in tags)
    parts.append("---\n")

    # --- Header ---
    parts.append(f"# {title}\n")
    parts.append(
        "Trained with [MultimodalVA](https://github.com/y-chu/MultimodalVA) — a "
        "package for cause-of-death classification from verbal autopsy data. "
        f"Fine-tuned from `{base_model_label}` "
        f"over **{len(causes)} cause categories**.\n"
    )
    if is_fusion:
        parts.append(_FUSION_CAVEAT + "\n")

    # --- Metrics ---
    if test_metrics:
        parts.append("## Held-out test performance\n")
        parts.append(_metrics_table(test_metrics) + "\n")
    if val_metrics:
        parts.append("## Validation metrics (final epoch)\n")
        parts.append(_metrics_table(val_metrics) + "\n")
    if not test_metrics and not val_metrics:
        parts.append(
            "## Performance\n\n_No metrics were recorded with this run._\n"
        )

    # --- Usage ---
    # Show the truncation length this model was actually trained with. Falling
    # back to a literal default would print a number that silently disagrees
    # with the model for anyone who trained at a different length, so when the
    # value is unknown the argument is left out instead.
    if max_length is None:
        max_length = metadata.get("max_length")
    max_length_arg = f", max_length={max_length}" if max_length else ""

    parts.append("## Usage\n")
    parts.append(_TEXT_USAGE.format(repo_id=repo_id, max_length_arg=max_length_arg))
    parts.append("\n" + _DOWNSTREAM_USAGE.format(repo_id=repo_id) + "\n")

    # --- Causes ---
    if causes:
        parts.append("## Cause categories\n")
        parts.append("<details><summary>Show all categories</summary>\n")
        parts.append("\n".join(f"- {c}" for c in causes))
        parts.append("\n</details>\n")
        parts.append(_LABEL_MEANING_NOTE)

    # --- Hyperparameters ---
    if hyperparams:
        parts.append("## Training hyperparameters\n")
        parts.append("```json")
        parts.append(json.dumps(hyperparams, indent=2))
        parts.append("```\n")

    parts.append(
        "## Citation\n\nIf you use this model, please cite the MultimodalVA "
        "package: https://github.com/y-chu/MultimodalVA\n"
    )
    return "\n".join(parts)


def push_to_hub(
    model_dir: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    token: str | None = None,
    commit_message: str | None = None,
    model_kind: str = "text",
    base_model: str | None = None,
    metrics: dict | None = None,
    max_length: int | None = None,
    license: str = "mit",
    generate_card: bool = True,
    create_pr: bool = False,
) -> str:
    """Publish a saved model directory to the Hugging Face Hub.

    Works on any directory containing a saved HuggingFace model (``config.json`` +
    weights + tokenizer), including past runs — point it at the run's ``final/``
    directory.

    Args:
        model_dir:      Directory with the saved model (must contain ``config.json``).
        repo_id:        Target Hub repo id, e.g. ``"your-org/va-bert"``. Created if
                        it does not exist.
        private:        Create/keep the repo private. Default True.
        token:          HF auth token. If None, uses the cached login
                        (``huggingface-cli login``) or ``HF_TOKEN`` env var.
        commit_message: Commit message for the upload.
        model_kind:     ``"text"`` or ``"data_fusion"`` — controls the card caveat.
        base_model:     Base checkpoint name (defaults to metadata ``model_name``).
        metrics:        Optional held-out test metrics dict to embed in the card.
        max_length:     Truncation length shown in the card usage snippet. When
                        None (default), it is read from the run's
                        ``training_metadata.json``.
        license:        SPDX license id for the card frontmatter. Default ``"mit"``.
        generate_card:  Write a ``README.md`` model card before uploading. Default True.
        create_pr:      Open a PR instead of committing to main. Default False.

    Returns:
        The URL of the published model repo.

    Raises:
        FileNotFoundError: If ``model_dir`` has no ``config.json``.
        ImportError:       If ``huggingface_hub`` is not installed.
    """
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"No 'config.json' in {model_dir}. Point model_dir at the saved model "
            "directory (e.g. '<run>/final'), not the pipeline root."
        )

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to push to the Hub. It normally ships "
            "with transformers; install explicitly with `pip install huggingface_hub`, "
            "then authenticate with `huggingface-cli login`."
        ) from exc

    if generate_card:
        card = build_model_card(
            model_dir,
            repo_id,
            model_kind=model_kind,
            base_model=base_model,
            metrics=metrics,
            max_length=max_length,
            license=license,
        )
        (model_dir / "README.md").write_text(card)
        logger.info("Wrote model card to %s", model_dir / "README.md")

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message or "Upload MultimodalVA cause-of-death model",
        ignore_patterns=_IGNORE_PATTERNS,
        create_pr=create_pr,
    )
    url = f"https://huggingface.co/{repo_id}"
    logger.info("Pushed model from %s to %s (private=%s)", model_dir, url, private)
    return url
