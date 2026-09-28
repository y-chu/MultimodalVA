"""
Text model registry and model-name resolution.

Every text-based pipeline (text classification, data fusion, feature fusion,
soft voting, stacking, and the command-line interface) resolves the
``model_name`` it receives through :func:`resolve_model_name`, so a given name
always loads the same checkpoint regardless of the entry point.

Accepted model names
--------------------
1. A package alias from :data:`TEXT_MODELS` (e.g. ``"bluebert"``),
   resolved to its Hugging Face Hub ID.
2. A remote key from :data:`REMOTE_MODELS` (e.g. ``"roberta-pm"``) for models
   that are not on the Hugging Face Hub. The archive is downloaded once and
   cached under ``~/.cache/multimodalva/<key>/``.
3. A local directory containing a Hugging Face model (``config.json`` plus
   weights), used as-is.
4. Any other string, treated as a Hugging Face Hub ID and used as-is
   (e.g. ``"emilyalsentzer/Bio_ClinicalBERT"``).

BioMed-RoBERTa and RoBERTa-PM are different models
--------------------------------------------------
Both are biomedical RoBERTa-base models, but they were pretrained on different
corpora with different vocabularies, so their weights are not interchangeable.
Each has exactly one name in this package.

================  ========================================  =======================================
                  BioMed-RoBERTa                            RoBERTa-PM
================  ========================================  =======================================
Package name      ``"biomedroberta"``                       ``"roberta-pm"``
Checkpoint        ``allenai/biomed_roberta_base``           ``RoBERTa-base-PM-M3-Voc-distill-hf``
Source            Hugging Face Hub                          Facebook Research bio-lm release
                                                            (downloaded; not on the Hub)
Pretraining data  RoBERTa-base, continued on S2ORC          PubMed abstracts, PMC full text and
                  biomedical full-text papers               MIMIC-III clinical notes
Vocabulary        50,265 tokens (original RoBERTa BPE)      50,008 tokens (domain-specific BPE)
Reference         Gururangan et al., ACL 2020               Lewis et al., ClinicalNLP 2020
================  ========================================  =======================================

Spellings that could refer to either model — for example ``"roberta_pm"``,
``"RoBERTa-PM"`` or ``"biomed_roberta"`` — are rejected with a message naming
both options, so the choice is always explicit.
"""

from __future__ import annotations

import logging
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

#: Package alias -> Hugging Face Hub ID.
#: Architecture groups supported by ``freeze_model_layers()``:
#: BERT (bert, biobert, bioclinicalbert, bluebert, biomedbert, clinicalbert),
#: RoBERTa (biomedroberta), ELECTRA (bioelectra),
#: long-context (longformer, clinicallongformer, bigbird, clinicalbigbird).
TEXT_MODELS: dict[str, str] = {
    "bert":               "bert-base-uncased",
    "biobert":            "dmis-lab/biobert-base-cased-v1.2",
    "bioclinicalbert":    "emilyalsentzer/Bio_ClinicalBERT",
    "bluebert":           "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "biomedbert":         "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "clinicalbert":       "medicalai/ClinicalBERT",
    # BioMed-RoBERTa (Gururangan et al., 2020). Not RoBERTa-PM; see module docstring.
    "biomedroberta":      "allenai/biomed_roberta_base",
    "bioelectra":         "kamalkraj/bioelectra-base-discriminator-pubmed",
    "longformer":         "allenai/longformer-base-4096",
    "clinicallongformer": "yikuan8/Clinical-Longformer",
    "bigbird":            "google/bigbird-roberta-base",
    "clinicalbigbird":    "yikuan8/Clinical-BigBird",
}

#: Remote key -> download URL, for models that are not on the Hugging Face Hub.
REMOTE_MODELS: dict[str, str] = {
    # RoBERTa-PM (Lewis et al., 2020). Not BioMed-RoBERTa; see module docstring.
    "roberta-pm": (
        "https://dl.fbaipublicfiles.com/biolm/"
        "RoBERTa-base-PM-M3-Voc-distill-hf.tar.gz"
    ),
}

#: One-line description per model name, shown by ``multimodalva list-models``.
MODEL_DESCRIPTIONS: dict[str, str] = {
    "bert":               "BERT-base, uncased (general domain)",
    "biobert":            "BioBERT v1.2 (PubMed)",
    "bioclinicalbert":    "Bio+Clinical BERT (MIMIC-III notes)",
    "bluebert":           "BlueBERT (PubMed + MIMIC-III)",
    "biomedbert":         "BiomedBERT (PubMed abstracts + full text)",
    "clinicalbert":       "ClinicalBERT (medicalai)",
    "biomedroberta":      "BioMed-RoBERTa: RoBERTa-base continued on S2ORC biomedical papers; vocab 50,265",
    "bioelectra":         "BioELECTRA discriminator (PubMed)",
    "longformer":         "Longformer-base, 4,096 tokens (general domain)",
    "clinicallongformer": "Clinical-Longformer, 4,096 tokens (MIMIC-III)",
    "bigbird":            "BigBird-RoBERTa-base, 4,096 tokens (general domain)",
    "clinicalbigbird":    "Clinical-BigBird, 4,096 tokens (MIMIC-III)",
    "roberta-pm":         "RoBERTa-PM: PubMed + PMC + MIMIC-III, domain vocab 50,008 (downloaded)",
}

#: Default cache root for downloaded remote models.
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "multimodalva"

# Normalised spellings that could mean either RoBERTa model. Only the exact
# package names "biomedroberta" and "roberta-pm" are accepted.
_AMBIGUOUS_ROBERTA = {
    "robertapm", "robertabasepm", "biomedroberta", "biomedrobertabase",
    "biomedrobertapm", "bioroberta",
}


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------------------
# Remote download helpers
# ---------------------------------------------------------------------------

def _looks_like_model_dir(path: Path) -> bool:
    """Return True when ``path`` looks like a HF-style local model directory."""
    if not path.is_dir():
        return False

    has_config = (path / "config.json").exists()
    has_weights = any(
        (path / filename).exists()
        for filename in (
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        )
    )
    return has_config and has_weights


def _find_extracted_model_dir(root: Path) -> Path | None:
    """Find the actual extracted model directory under ``root``.

    Some archives unpack directly into a single model directory, while others
    add an extra wrapper directory and place the HuggingFace files one level
    deeper. We return the shallowest directory that contains both
    ``config.json`` and model weights.
    """
    if _looks_like_model_dir(root):
        return root

    candidates = sorted(
        (
            path for path in root.rglob("*")
            if _looks_like_model_dir(path)
        ),
        key=lambda p: (len(p.relative_to(root).parts), str(p)),
    )
    return candidates[0] if candidates else None


def download_model(key: str, cache_dir: str | Path | None = None) -> str:
    """Download and extract a remote model checkpoint from REMOTE_MODELS.

    Downloads the archive to a temporary directory (or cache_dir), extracts
    it, removes the archive, and returns the local model directory path for
    use as model_name in train_text(), predict_text(), and prepare_text_dataset().

    Args:
        key: Key in REMOTE_MODELS (e.g. "roberta-pm").
        cache_dir: Directory to extract the model into.
                   Defaults to a new system temp directory (deleted on reboot).

    Returns:
        Absolute path to the extracted model directory.

    Raises:
        ValueError: If key is not in REMOTE_MODELS.

    Example:
        model_path = download_model("roberta-pm")
        train(..., model_name=model_path)
    """
    if key not in REMOTE_MODELS:
        raise ValueError(
            f"Unknown remote model key: '{key}'. "
            f"Available keys: {list(REMOTE_MODELS)}"
        )

    url = REMOTE_MODELS[key]
    archive_name = url.rsplit("/", 1)[-1]  # e.g. RoBERTa-base-PM-M3-Voc-distill-hf.tar.gz

    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="multimodalva_"))
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    existing_model_dir = _find_extracted_model_dir(cache_dir)
    if existing_model_dir is not None:
        logger.info("Reusing cached model at: %s", existing_model_dir)
        return str(existing_model_dir)

    archive_path = cache_dir / archive_name

    logger.info("Downloading %s ...", url)
    urllib.request.urlretrieve(url, archive_path)
    logger.info("Saved archive to %s", archive_path)

    logger.info("Extracting %s ...", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(cache_dir)
    archive_path.unlink()  # remove archive after extraction

    model_dir = _find_extracted_model_dir(cache_dir)
    if model_dir is None:
        raise FileNotFoundError(
            "Downloaded archive extracted successfully, but no HuggingFace-style "
            f"model directory was found under {cache_dir}."
        )

    logger.info("Model ready at: %s", model_dir)
    return str(model_dir)


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def resolve_model_name(model_name: str, cache_dir: str | Path | None = None) -> str:
    """Return a loadable Hugging Face model ID or local path for ``model_name``.

    Args:
        model_name: Package alias, remote key, local model directory, or
            Hugging Face Hub ID. See the module docstring for the full rules.
        cache_dir: Where to extract a remote model. Defaults to
            ``~/.cache/multimodalva/<key>/`` so it is downloaded only once.

    Returns:
        A Hugging Face Hub ID or an absolute local directory path. Calling
        this function again on its own output returns the same value.

    Raises:
        ValueError: If ``model_name`` is empty, or is a spelling that could
            refer to either BioMed-RoBERTa or RoBERTa-PM.
    """
    if not model_name or not str(model_name).strip():
        raise ValueError("model_name must be a non-empty string.")
    name = str(model_name)

    if name in TEXT_MODELS:
        resolved = TEXT_MODELS[name]
        logger.info("Model %r resolved to %s (%s).", name, resolved, MODEL_DESCRIPTIONS.get(name, ""))
        return resolved

    if name in REMOTE_MODELS:
        target = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_ROOT / name
        resolved = download_model(name, cache_dir=target)
        logger.info("Model %r resolved to %s (%s).", name, resolved, MODEL_DESCRIPTIONS.get(name, ""))
        return resolved

    if Path(name).expanduser().is_dir():
        return str(Path(name).expanduser())

    if "/" not in name and _normalise(name) in _AMBIGUOUS_ROBERTA:
        raise ValueError(
            f"Model name {name!r} is ambiguous. Two different biomedical RoBERTa "
            "models are available:\n"
            "  'biomedroberta' -> BioMed-RoBERTa (allenai/biomed_roberta_base; "
            "S2ORC biomedical papers; vocabulary 50,265)\n"
            "  'roberta-pm'    -> RoBERTa-PM (RoBERTa-base-PM-M3-Voc-distill-hf; "
            "PubMed + PMC + MIMIC-III; vocabulary 50,008)\n"
            "Pass one of these names exactly."
        )

    return name
