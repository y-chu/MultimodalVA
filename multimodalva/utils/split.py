"""
Shared train-test split for text, tabular, and ensemble pipelines.

Output:
    train_df, test_df  — DataFrames passed into prepare_text_dataset() / prepare_tabular_dataset()

Also holds the two split helpers every pipeline shares:
    warn_small_classes()    — flag causes too rare for a split to be informative
    stratified_indices()    — a stratified validation split that degrades
                              instead of raising on very small data
"""

import logging
import math
import time
from collections import Counter
from collections.abc import Iterable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split as _sklearn_split

logger = logging.getLogger(__name__)


# Values in a split_col that mark a row as belonging to the test set.
# Everything else (train, TRAIN, 0, False, NaN, ...) is treated as train.
_TEST_MARKERS = frozenset({"test", "testing", "holdout", "val", "valid", "validation"})

#: Below this many rows of a class, a split carries too little of that class for
#: its scores to mean much. Used only to warn — nothing is ever dropped or
#: merged on the package's own initiative.
SMALL_CLASS_MIN = 5

#: Messages warn_small_classes() has already emitted in this process, so a
#: per-fold or per-base-model loop reports the same small classes once instead
#: of once per iteration. Tests call reset_small_class_warnings().
_WARNED: set[tuple] = set()


def reset_small_class_warnings() -> None:
    """Forget which small-class warnings were already emitted in this process."""
    _WARNED.clear()


def warn_small_classes(
    labels: Iterable,
    where: str,
    *,
    classes: Iterable | None = None,
    min_count: int = SMALL_CLASS_MIN,
    log: logging.Logger | None = None,
    once: bool = True,
) -> dict:
    """Warn when a class has fewer than ``min_count`` rows in one part of a split.

    Args:
        labels: The labels of the rows in that part of the split.
        where: Where these rows are, named for the message ("the test set").
        classes: Every class that should be represented, so a class missing
                 from ``labels`` is reported as 0 rather than going unnoticed.
                 Defaults to the classes present in ``labels``.
        min_count: Rows below which a class is called small. Default
                   ``SMALL_CLASS_MIN`` (5).
        log: Logger to warn on. Defaults to this module's.
        once: Suppress an identical message already emitted in this process.

    Returns:
        ``{class: count}`` for the classes below ``min_count``, smallest first.
        Empty when every class has enough rows.
    """
    log = log or logger
    counts = Counter(labels)
    if classes is not None:
        for cls in classes:
            counts.setdefault(cls, 0)
    small = {
        cls: n for cls, n in sorted(counts.items(), key=lambda kv: (kv[1], str(kv[0])))
        if n < min_count
    }
    if not small:
        return {}

    key = (where, min_count, tuple(small.items()))
    if once and key in _WARNED:
        return small
    _WARNED.add(key)

    shown = list(small.items())[:10]
    listed = ", ".join(f"{cls}={n}" for cls, n in shown)
    if len(small) > len(shown):
        listed += f", and {len(small) - len(shown)} more"
    log.warning(
        "Small sample size: %d of %d classes have fewer than %d rows in %s (%s). "
        "Scores for these classes carry large uncertainty, and a stratified "
        "split cannot hold their proportions; consider grouping rare causes, "
        "reporting their scores as indicative only, or a different split size.",
        len(small), len(counts), min_count, where, listed,
    )
    return small


def stratified_indices(
    labels: Sequence,
    test_size: float | int,
    random_state: int,
    *,
    what: str = "validation",
    warn_below: int | None = SMALL_CLASS_MIN,
    label_names: dict | None = None,
    log: logging.Logger | None = None,
) -> tuple[list[int], list[int]]:
    """Split row positions into (train, held-out), stratified where possible.

    The resilience is the point: a stratified split needs one held-out row per
    class and two rows of every class, and on small data — or on a search
    trial's sub-split of it — the requested share falls short and sklearn
    raises. Inside a search that surfaced as "all trials failed", with an error
    blaming GPU memory. So the held-out slice is enlarged to one row per class
    where that is possible, and falls back to an unstratified split only where
    it is not. Splits that worked before take the unchanged path.

    Args:
        labels: Label per row, in row order. Any hashable labels.
        test_size: Share (float) or count (int) held out, as sklearn takes it.
        random_state: Seed for the split — a partitioning decision, so callers
                      pass ``split_seed``.
        what: Name of the held-out part, for the messages ("validation").
        warn_below: Warn when a class has fewer than this many *training* rows
                    after the split. None turns the check off. The held-out
                    slice is not checked: at a 10% share it is expected to hold
                    very few rows of each class.
        label_names: Optional ``{label: display name}`` (an ``id2label``) so the
                     warning names causes instead of integer ids. Display only —
                     the split itself uses ``labels`` as given.
        log: Logger for the warnings. Defaults to this module's.

    Returns:
        (train_idx, held_out_idx) — positions into ``labels``.
    """
    log = log or logger
    labels = list(labels)
    n_rows = len(labels)
    counts = Counter(labels)
    n_classes = len(counts)
    size: float | int = test_size

    if isinstance(test_size, float) and int(math.ceil(test_size * n_rows)) < n_classes < n_rows:
        size = n_classes
        log.warning(
            "%s slice of %.0f%% is %d rows, fewer than the %d classes; "
            "using %d rows so every class can appear once.",
            what.capitalize(), 100 * test_size,
            int(math.ceil(test_size * n_rows)), n_classes, n_classes,
        )

    stratify: list | None = labels
    if n_classes >= n_rows or min(counts.values()) < 2:
        stratify = None
        log.warning(
            "A class has a single training row, so the %s slice cannot be "
            "stratified; splitting at random.", what,
        )

    train_idx, held_idx = _sklearn_split(
        list(range(n_rows)),
        test_size=size,
        random_state=random_state,
        stratify=stratify,
    )
    if warn_below:
        def _name(label):
            if label_names is None:
                return label
            return label_names.get(label, label_names.get(str(label), label))

        warn_small_classes(
            [_name(labels[i]) for i in train_idx],
            f"the training rows left after the {what} split",
            classes=[_name(c) for c in counts],
            min_count=warn_below,
            log=log,
        )
    return train_idx, held_idx


def split(
    df: pd.DataFrame,
    label_col: str,
    text_col: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    split_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into stratified train and test sets.

    Works for text, tabular, and ensemble pipelines — only columns relevant
    to each modality need to be specified.

    Args:
        df: Input DataFrame.
        label_col: Name of the label column (required). Used for stratification.
        text_col: Name of the text column to validate exists. Optional —
                  omit for tabular-only pipelines; provide for text/ensemble.
        test_size: Proportion of data held out for testing. Default 0.2.
        random_state: Random seed for reproducibility. Default 42.
        stratify: Stratified split (preserves class proportions). Default True.
        split_col: Optional column marking a pre-defined train/test assignment.
                   When provided, the DataFrame is partitioned by this column
                   instead of a random split — rows whose value is a recognised
                   test marker ("test"/"val"/... case-insensitive, or a truthy
                   non-string such as True/1) go to the test set, all other rows
                   go to train. Use this to reproduce an external / fixed split
                   (e.g. a committed splits.json) so results are comparable
                   across experiments. ``test_size``/``random_state``/``stratify``
                   are ignored when ``split_col`` is set.

    Returns:
        train_df: Training DataFrame (index reset).
        test_df:  Test DataFrame (index reset).

    Raises:
        ValueError: If label_col, text_col, or split_col (when provided) are not
                    in df, or if split_col yields an empty train or test set.
    """
    started = time.perf_counter()
    cols_to_check = [label_col]
    if text_col is not None:
        cols_to_check.append(text_col)
    if split_col is not None:
        cols_to_check.append(split_col)

    missing = [c for c in cols_to_check if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    if split_col is not None:
        test_mask = df[split_col].map(_is_test_marker)
        train_df = df[~test_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)
        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(
                f"split_col={split_col!r} produced an empty split "
                f"({len(train_df)} train / {len(test_df)} test). Expected test "
                f"rows to be marked with one of {sorted(_TEST_MARKERS)} "
                f"(case-insensitive) or a truthy value."
            )
        logger.info(
            "split(): completed in %.2fs — %d train / %d test rows "
            "(pre-defined via split_col=%r).",
            time.perf_counter() - started,
            len(train_df),
            len(test_df),
            split_col,
        )
        _warn_split_classes(df, train_df, test_df, label_col)
        return train_df, test_df

    if stratify and df[label_col].isna().any():
        n_missing = int(df[label_col].isna().sum())
        raise ValueError(
            f"{n_missing} of {len(df)} rows have a missing {label_col!r} and a "
            "stratified split cannot use them. Drop them first, for example "
            f"df = df[df[{label_col!r}].notna()], or pass stratify=False."
        )

    if stratify:
        counts = df[label_col].value_counts()
        if int(counts.min()) < 2:
            rare = counts[counts < 2].index.tolist()
            raise ValueError(
                "A stratified train/test split needs at least 2 rows per class, "
                f"but these class(es) have fewer: {rare}. Collect/group more "
                "cases, supply a valid fixed split, or deliberately pass "
                "stratify=False (which may leave a class out of training)."
            )
    stratify_col = df[label_col] if stratify else None
    train_df, test_df = _sklearn_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info(
        "split(): completed in %.2fs — %d train / %d test rows (test_size=%.2f, stratify=%s).",
        time.perf_counter() - started,
        len(train_df),
        len(test_df),
        test_size,
        stratify,
    )
    _warn_split_classes(df, train_df, test_df, label_col)
    return train_df, test_df


def _warn_split_classes(df, train_df, test_df, label_col: str) -> None:
    """Report classes left with too few rows on either side of the train/test split."""
    classes = df[label_col].dropna().unique()
    warn_small_classes(train_df[label_col].dropna(), "the training set", classes=classes)
    warn_small_classes(test_df[label_col].dropna(), "the test set", classes=classes)


def _is_test_marker(value) -> bool:
    """Return True when a split_col value marks a row as belonging to the test set.

    Strings are matched case-insensitively against ``_TEST_MARKERS``; other
    values fall back to truthiness (True/1 → test, False/0/NaN → train).
    """
    if isinstance(value, str):
        return value.strip().lower() in _TEST_MARKERS
    if pd.isna(value):
        return False
    return bool(value)
