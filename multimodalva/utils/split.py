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


def split(
    df: pd.DataFrame,
    label_col: str,
    text_col: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
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

    Returns:
        train_df: Training DataFrame (index reset).
        test_df:  Test DataFrame (index reset).

    Raises:
        ValueError: If label_col or text_col (when provided) are not in df.
    """
    started = time.perf_counter()
    cols_to_check = [label_col]
    if text_col is not None:
        cols_to_check.append(text_col)

    missing = [c for c in cols_to_check if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}. "
            f"Available columns: {df.columns.tolist()}"
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
