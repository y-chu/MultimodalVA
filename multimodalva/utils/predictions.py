"""Turn a probability matrix into the three prediction tables every pipeline returns.

Each pipeline ends the same way: it produces an ``(n_samples, n_classes)``
probability matrix and has to render it as ``top1`` / ``full`` / ``topk``. That
rendering is identical everywhere, so it lives here once instead of being
repeated in each pipeline's predict step.

    from multimodalva.utils.predictions import assemble_predictions, save_predictions

    result = assemble_predictions(proba, id2label, true_labels=y_true)
    save_predictions(result, "runs/text/predictions")

``true_labels`` is optional. Pass it when the data is labelled and the tables
gain a ``true_label`` column for scoring; leave it out to score new records that
have no known cause, in which case the column is simply absent.

Column layout (``id`` first when ``ids`` is given, ``true_label`` next when
``true_labels`` is given):

    top1   predicted_label, predicted_prob
    full   prob_0, prob_1, ...        integer class IDs, not label strings
    topk   top1_label, top1_prob, ..., topK_label, topK_prob

Ties are broken toward the lower class ID, deterministically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .types import PredictionResult

logger = logging.getLogger(__name__)

__all__ = [
    "assemble_predictions",
    "save_predictions",
    "load_predictions",
    "resolve_test_ids",
]


def load_predictions(
    path: "str | Path",
    prefix: str = "predictions",
) -> PredictionResult:
    """Read back a :class:`PredictionResult` that a pipeline wrote to disk.

    The inverse of :func:`save_predictions`. Point it at a run directory or
    directly at the ``predictions/`` folder inside one; both work, so a caller
    does not have to know which layout a given pipeline used.

    This is what makes a finished run reusable without retraining — combining
    yesterday's text run with today's tabular run in a soft vote, or resuming an
    ensemble whose base models are already done::

        from multimodalva.utils.predictions import load_predictions
        from multimodalva.ensemble.voting import vote_from_results

        bert = load_predictions("runs/bert")
        lgbm = load_predictions("runs/lgbm")
        voted = vote_from_results([bert, lgbm], id2label=bert.id2label)

    Args:
        path:   A run directory (containing ``predictions/`` and
                ``id2label.json``), or the ``predictions/`` directory itself.
        prefix: Filename stem used when the CSVs were written.

    Returns:
        The reconstructed ``PredictionResult``. ``id2label`` is read from
        ``id2label.json`` when present, and otherwise rebuilt from the
        ``prob_*`` columns with the labels seen in ``top1`` — enough to vote
        with, though the cause names of unpredicted classes cannot be recovered
        that way.

    Raises:
        FileNotFoundError: No ``{prefix}_top1.csv`` under either layout.
    """
    root = Path(path)
    candidates = [root / "predictions", root, root.parent]
    pred_dir = next(
        (c for c in candidates if (c / f"{prefix}_top1.csv").is_file()), None
    )
    if pred_dir is None:
        raise FileNotFoundError(
            f"No {prefix}_top1.csv found in {root}, {root / 'predictions'} or "
            f"{root.parent}. Point load_predictions() at a run directory or at "
            "the predictions directory inside one."
        )

    top1 = pd.read_csv(pred_dir / f"{prefix}_top1.csv")
    full = pd.read_csv(pred_dir / f"{prefix}_full.csv")
    topk = pd.read_csv(pred_dir / f"{prefix}_topk.csv")

    # id2label.json is what every pipeline writes now. training_metadata.json
    # carries the same mapping and is the only copy in runs made before that,
    # so check it too rather than falling back to reconstruction.
    id2label: dict | None = None
    # Ensembles keep the maps in the run root; text and tabular keep them in
    # final/, beside the model weights.
    for base in (pred_dir, pred_dir.parent, pred_dir.parent / "final"):
        direct = base / "id2label.json"
        if direct.is_file():
            with open(direct) as fh:
                id2label = {int(k): v for k, v in json.load(fh).items()}
            break
        meta = base / "training_metadata.json"
        if meta.is_file():
            with open(meta) as fh:
                stored = json.load(fh).get("id2label")
            if stored:
                id2label = {int(k): v for k, v in stored.items()}
                break

    if id2label is None:
        # Fall back to the probability columns. Their suffixes are the class
        # IDs, so the mapping's *keys* are exact; only the names are partial.
        class_ids = sorted(
            int(c[5:]) for c in full.columns if c.startswith("prob_")
        )
        seen = dict.fromkeys(top1["predicted_label"].astype(str))
        id2label = {
            cid: (list(seen)[i] if i < len(seen) else f"class_{cid}")
            for i, cid in enumerate(class_ids)
        }
        logger.warning(
            "No id2label.json beside %s — class names were reconstructed from "
            "the prediction tables and may be incomplete. Probability column "
            "order is still correct.", pred_dir,
        )

    return PredictionResult(top1=top1, full=full, topk=topk, id2label=id2label)


def resolve_test_ids(
    test_df: "pd.DataFrame",
    id_col: str | None,
    keep_mask: "pd.Series | np.ndarray | None" = None,
) -> "list | None":
    """Pull the row identifiers for the rows that will be scored.

    Every pipeline needs the same three steps — check the column exists, drop
    the rows the dataset preparation dropped, hand over a plain list — and
    getting any of them wrong misaligns ids against predictions silently. One
    implementation, so all six pipelines align the same way.

    Args:
        test_df:   The held-out split, before any label-based row dropping.
        id_col:    Column holding the identifier, or ``None`` to skip ids.
        keep_mask: Boolean mask of the rows that survive into scoring. ``None``
                   keeps every row.

    Returns:
        The identifiers as a list, or ``None`` when ``id_col`` is ``None``.

    Raises:
        ValueError: ``id_col`` is not a column of ``test_df``, or the retained
                    identifiers are missing or duplicated.
    """
    if id_col is None:
        return None
    if id_col not in test_df.columns:
        raise ValueError(
            f"id_col {id_col!r} not found in the DataFrame. "
            f"Available columns: {list(test_df.columns)[:20]}"
        )
    kept = test_df if keep_mask is None else test_df[keep_mask]
    if kept[id_col].isna().any():
        raise ValueError(
            f"id_col {id_col!r} contains missing values in rows being scored."
        )
    duplicated = kept.loc[kept[id_col].duplicated(keep=False), id_col]
    if not duplicated.empty:
        shown = sorted(map(str, duplicated.unique()))[:10]
        raise ValueError(
            f"id_col {id_col!r} is not unique in rows being scored; duplicated "
            f"identifier(s): {shown}."
        )
    return kept[id_col].tolist()


def _rank_descending(proba: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the k highest probabilities per row, best first.

    Sorting the negated matrix with a stable sort gives descending order while
    leaving equal probabilities in class-ID order. Sorting ascending and
    reversing would instead put the *highest* class ID first among ties, and
    NumPy's default quicksort is not stable, so tied rows could come back in a
    different order from one call to the next.
    """
    return np.argsort(-proba, axis=1, kind="stable")[:, :k]


def assemble_predictions(
    proba: np.ndarray,
    id2label: dict,
    *,
    true_labels: "list | pd.Series | None" = None,
    ids: "list | pd.Series | None" = None,
    top_k: int = 3,
) -> PredictionResult:
    """Build the three prediction tables from a probability matrix.

    Args:
        proba:       ``(n_samples, n_classes)`` probabilities. Column *j* must
                     hold the class whose ID is the *j*-th smallest key of
                     ``id2label`` — the order every pipeline already uses.
        id2label:    Integer class ID → cause name.
        true_labels: Ground-truth cause names, one per row. Omit for unlabelled
                     data; the ``true_label`` column is then left out entirely.
        ids:         Row identifiers, one per row. When given they become a
                     leading ``id`` column so predictions can be joined back to
                     the source records.
        top_k:       How many classes to list in ``topk``. Capped at the number
                     of classes.

    Returns:
        ``PredictionResult(top1, full, topk, id2label)``.

    Raises:
        ValueError: If ``proba`` is not 2-D, or if its shape, ``true_labels`` or
                    ``ids`` disagree about the number of rows or classes.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError(
            f"proba must be 2-D (n_samples, n_classes); got shape {proba.shape}."
        )

    n_samples, n_cols = proba.shape
    sorted_ids = sorted(id2label)
    if n_cols != len(sorted_ids):
        raise ValueError(
            f"proba has {n_cols} columns but id2label has {len(sorted_ids)} classes. "
            "The probability matrix and the label map must come from the same run."
        )

    def _check_len(name: str, values) -> list:
        seq = list(values)
        if len(seq) != n_samples:
            raise ValueError(
                f"{name} has {len(seq)} entries but proba has {n_samples} rows."
            )
        return seq

    lead: dict[str, list] = {}
    if ids is not None:
        lead["id"] = _check_len("ids", ids)
    if true_labels is not None:
        lead["true_label"] = _check_len("true_labels", true_labels)

    k = min(top_k, len(sorted_ids))
    ranked = _rank_descending(proba, k)
    rows = np.arange(n_samples)

    top1_pos = ranked[:, 0]
    top1 = pd.DataFrame({
        **lead,
        "predicted_label": [id2label[sorted_ids[p]] for p in top1_pos],
        "predicted_prob": proba[rows, top1_pos],
    })

    full = pd.DataFrame({
        **lead,
        **{f"prob_{cid}": proba[:, j] for j, cid in enumerate(sorted_ids)},
    })

    topk_data: dict[str, list] = dict(lead)
    for rank in range(k):
        pos = ranked[:, rank]
        topk_data[f"top{rank + 1}_label"] = [id2label[sorted_ids[p]] for p in pos]
        topk_data[f"top{rank + 1}_prob"] = proba[rows, pos]
    topk = pd.DataFrame(topk_data)

    return PredictionResult(top1=top1, full=full, topk=topk, id2label=id2label)


def save_predictions(
    result: PredictionResult,
    save_dir: "str | Path",
    prefix: str = "predictions",
) -> None:
    """Write the three tables to ``<save_dir>/<prefix>_{top1,full,topk}.csv``.

    Args:
        result:   The result to write.
        save_dir: Target directory. Created if it does not exist.
        prefix:   Filename stem shared by the three CSVs.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    result.top1.to_csv(save_dir / f"{prefix}_top1.csv", index=False)
    result.full.to_csv(save_dir / f"{prefix}_full.csv", index=False)
    result.topk.to_csv(save_dir / f"{prefix}_topk.csv", index=False)
    logger.info(
        "Saved CSVs to %s: %s_top1.csv, %s_full.csv, %s_topk.csv",
        save_dir, prefix, prefix, prefix,
    )
