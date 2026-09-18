"""
Step 2 (tabular pipeline): Feature engineering — optional encoding and scaling.

Assumes clean input data (no imputation performed).
All transformations are opt-in via explicit options.

Input:  train_df, test_df from split()
Output: X_train, X_test (numpy arrays), y_train, y_test (integer arrays),
        preprocessor (fitted ColumnTransformer or None), label2id, id2label
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


_VALID_ENCODINGS = {None, "auto", "ordinal", "onehot"}


def _build_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
    encode_categoricals: str | None,
    scale_numeric: bool,
) -> ColumnTransformer | None:
    """Build a ColumnTransformer from the requested transformations.

    Returns None when no transformation is needed; the caller then handles
    raw numpy conversion directly.

    Numeric columns:
        scale_numeric=True  → StandardScaler
        scale_numeric=False → passthrough (no-op; tree models do not need scaling)

    Categorical columns:
        None      → passthrough (caller must ensure columns are already numeric)
        "auto"    → OrdinalEncoder (tree-safe default; same as "ordinal")
        "ordinal" → OrdinalEncoder; NaN re-emitted as NaN (encoded_missing_value=np.nan)
                    so models that handle NaN natively (catboost, lightgbm, xgboost)
                    can learn from missingness directly.
        "onehot"  → OneHotEncoder (sparse_output=False); recommended for MLP/linear models.
    """
    if encode_categoricals not in _VALID_ENCODINGS:
        raise ValueError(
            f"encode_categoricals must be one of {sorted(str(v) for v in _VALID_ENCODINGS)}. "
            f"Got '{encode_categoricals}'."
        )

    needs_transform = (
        (scale_numeric and bool(num_cols))
        or (encode_categoricals is not None and bool(cat_cols))
    )
    if not needs_transform:
        return None

    transformers = []

    if num_cols:
        transformer = StandardScaler() if scale_numeric else "passthrough"
        transformers.append(("numeric", transformer, num_cols))

    if cat_cols:
        if encode_categoricals is not None:
            effective = encode_categoricals if encode_categoricals != "auto" else "ordinal"
            if effective == "onehot":
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:  # "ordinal"
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=np.nan,
                )
            transformers.append(("categorical", enc, cat_cols))
        else:
            transformers.append(("categorical", "passthrough", cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def valid_label_mask(df: pd.DataFrame, label_col: str) -> pd.Series:
    """Rows whose label is usable — not missing and not an empty string.

    Exposed so callers can select the same rows this module keeps, for example
    to line up row identifiers with the prediction tables.
    """
    return ~(df[label_col].isna() | (df[label_col].astype(str).str.strip() == ""))


def prepare_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    drop_missing_label: bool = True,
    encode_categoricals: str | None = "ordinal",
    scale_numeric: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer | None, dict, dict, list[str]]:
    """Preprocess tabular features and encode labels.

    Assumes clean input (no NaN imputation). All transformations are opt-in.
    Preprocessing is fitted on train_df only and applied to both splits.
    Label maps are built from the union of train + test labels.

    Args:
        train_df:            Training DataFrame from split().
        test_df:             Test DataFrame from split().
        feature_cols:        Columns to use as features. label_col is excluded automatically.
        label_col:           Column containing class labels.
        cat_cols:            Categorical columns. Auto-detected from object/category dtype if None.
        num_cols:            Numeric columns. Auto-detected from numeric dtype if None.
        drop_missing_label:  Drop rows where label is NaN or empty string. Default True.
        encode_categoricals: How to encode categorical columns. Default "ordinal".
                             None      — no encoding; columns must already be numeric.
                             "auto"    — OrdinalEncoder (tree-safe; same as "ordinal").
                             "ordinal" — OrdinalEncoder; NaN re-emitted as NaN so native-NaN
                                         models (catboost, lightgbm, xgboost) handle missingness
                                         directly.
                             "onehot"  — OneHotEncoder (sparse_output=False); recommended for
                                         MLP/linear models.
        scale_numeric:       Apply StandardScaler to numeric columns. Default False.
                             Recommended for MLP/linear models; not needed for tree models.

    Returns:
        X_train:      Preprocessed feature matrix (numpy array) for training.
        X_test:       Preprocessed feature matrix (numpy array) for evaluation.
        y_train:      Integer label array for training.
        y_test:       Integer label array for evaluation.
        preprocessor: Fitted ColumnTransformer, or None if no transformation was applied.
                      Bundled into model.joblib by train() for inference on new data.
        label2id:     Dict mapping label strings → integer IDs.
        id2label:     Dict mapping integer IDs → label strings.
        feature_names: Post-transformation feature names for SHAP and visualization.
                      ColumnTransformer prefixes (e.g. "numeric__", "categorical__") are
                      stripped so names match the original column names. OneHot-expanded
                      columns are named "{col}_{category}".

    Raises:
        ValueError: If required columns are missing or no features remain after filtering.
    """
    # Guard: label_col must never be transformed as a feature
    if label_col in feature_cols:
        logger.warning(
            "label_col '%s' found in feature_cols — removing it automatically.", label_col
        )
        feature_cols = [c for c in feature_cols if c != label_col]

    # Validate columns
    for df_name, df in [("train_df", train_df), ("test_df", test_df)]:
        missing = [c for c in feature_cols + [label_col] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Column(s) not found in {df_name}: {missing}. "
                f"Available: {df.columns.tolist()}"
            )

    # Optionally drop rows with missing or empty labels
    if drop_missing_label:
        for df_name, df in [("train_df", train_df), ("test_df", test_df)]:
            bad = ~valid_label_mask(df, label_col)
            if bad.any():
                logger.warning(
                    "Dropping %d rows with missing label in %s.", bad.sum(), df_name
                )
        train_df = train_df[valid_label_mask(train_df, label_col)].reset_index(drop=True)
        test_df = test_df[valid_label_mask(test_df, label_col)].reset_index(drop=True)

    # Auto-detect column types from train_df feature subset
    feat_train = train_df[feature_cols]
    if cat_cols is None:
        cat_cols = feat_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if num_cols is None:
        num_cols = feat_train.select_dtypes(include="number").columns.tolist()

    if not cat_cols and not num_cols:
        raise ValueError(
            "No feature columns found after type detection. "
            "Check feature_cols, cat_cols, and num_cols."
        )

    logger.info(
        "Features: %d numeric, %d categorical. "
        "encode_categoricals=%s, scale_numeric=%s.",
        len(num_cols), len(cat_cols), encode_categoricals, scale_numeric,
    )

    # Build and fit preprocessor (None if no transformation requested)
    preprocessor = _build_preprocessor(
        num_cols, cat_cols, encode_categoricals, scale_numeric
    )
    if preprocessor is not None:
        X_train = preprocessor.fit_transform(train_df[feature_cols])
        X_test = preprocessor.transform(test_df[feature_cols])
        # Strip ColumnTransformer prefixes (e.g. "numeric__age" → "age",
        # "categorical__sex_male" → "sex_male") for clean SHAP axis labels.
        feature_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    else:
        X_train = train_df[feature_cols].to_numpy()
        X_test = test_df[feature_cols].to_numpy()
        feature_names = list(feature_cols)

    # Label maps from union of train + test labels
    all_labels = sorted(
        set(train_df[label_col].astype(str)) | set(test_df[label_col].astype(str))
    )
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    y_train = np.array([label2id[str(lbl)] for lbl in train_df[label_col]], dtype=np.int64)
    y_test = np.array([label2id[str(lbl)] for lbl in test_df[label_col]], dtype=np.int64)

    logger.info(
        "Dataset prepared: %d train, %d test, %d classes.",
        len(y_train), len(y_test), len(label2id),
    )
    return X_train, X_test, y_train, y_test, preprocessor, label2id, id2label, feature_names
