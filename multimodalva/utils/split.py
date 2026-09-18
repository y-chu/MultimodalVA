"""
Shared train-test split for text, tabular, and ensemble pipelines.

Output:
    train_df, test_df  — DataFrames passed into each pipeline's prepare_dataset()
"""

import logging
import time

import pandas as pd
from sklearn.model_selection import train_test_split as _sklearn_split

logger = logging.getLogger(__name__)


# Values in a split_col that mark a row as belonging to the test set.
# Everything else (train, TRAIN, 0, False, NaN, ...) is treated as train.
_TEST_MARKERS = frozenset({"test", "testing", "holdout", "val", "valid", "validation"})


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
        return train_df, test_df

    if stratify and df[label_col].isna().any():
        n_missing = int(df[label_col].isna().sum())
        raise ValueError(
            f"{n_missing} of {len(df)} rows have a missing {label_col!r} and a "
            "stratified split cannot use them. Drop them first, for example "
            f"df = df[df[{label_col!r}].notna()], or pass stratify=False."
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
    return train_df, test_df


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
