"""
Step 2: Prepare tokenized datasets for transformer pipeline input.

Input:  train_df, test_df from split()
Output: train_dataset, test_dataset, label2id, id2label
        — passed directly into train(), predict(), and optimize()
"""

import logging
import time

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ClassificationDataset(Dataset):
    """PyTorch Dataset for sequence classification.

    Stores tokenized encodings and integer labels. The ``labels`` attribute
    is intentionally public so that split() and optimize() can access it
    for stratified sub-sampling without needing to decode items one by one.

    Sequences are stored without padding. DataCollatorWithPadding in the
    Trainer pads each batch to its own longest sequence at runtime.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int | None = 512,
    ):
        """
        Args:
            texts: List of raw text strings.
            labels: List of integer class IDs.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum number of tokens before truncation.
                - 512 (default) → truncate to 512 tokens (BERT's maximum).
                - None → use the model's built-in maximum
                  (e.g. 512 for BERT, 4096 for Longformer).
                - int → truncate to exactly that many tokens.
        """
        self.labels = labels

        # No padding here — DataCollatorWithPadding in the Trainer pads each
        # batch to its own longest sequence, so train and test are handled
        # independently with no length mismatch.
        tokenizer_kwargs = dict(truncation=True, padding=False)
        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length

        # Tokenize all texts in one vectorized call (fast Rust tokenizer uses
        # internal parallelism when TOKENIZERS_PARALLELISM=true).
        _raw = tokenizer(texts, **tokenizer_kwargs)

        # Pre-convert each sequence's token IDs to a tensor once here so that
        # __getitem__ is a pure list lookup with no tensor-construction overhead.
        # This is especially beneficial for MPS and CPU where __getitem__ is called
        # in the main process (num_workers=0) and the hot path must be cheap.
        self.encodings = {
            k: [torch.as_tensor(seq, dtype=torch.long) for seq in seqs]
            for k, seqs in _raw.items()
        }
        # Pre-build the labels tensor once to avoid torch.tensor() calls per item.
        self._labels_tensor = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # All values are already tensors — pure list lookup, no tensor construction.
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self._labels_tensor[idx]
        return item


def _drop_invalid_rows(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    split: str,
) -> pd.DataFrame:
    """Drop rows where text_col or label_col is NaN or an empty string.

    Logs a warning for each column that has invalid entries, reporting the
    row indices and the count dropped.

    Args:
        df: Input DataFrame.
        text_col: Name of the text column.
        label_col: Name of the label column.
        split: Label for the split ("train" or "test") used in warning messages.

    Returns:
        Cleaned DataFrame with invalid rows removed and index reset.
    """
    mask_invalid = pd.Series(False, index=df.index)

    for col in (text_col, label_col):
        is_null = df[col].isna()
        is_empty = (df[col].astype(str).str.strip() == "") & ~is_null
        col_invalid = is_null | is_empty

        if col_invalid.any():
            bad_indices = df.index[col_invalid].tolist()
            logger.warning(
                "[%s] Column '%s': %d row(s) with missing or empty values "
                "will be excluded. Row indices: %s",
                split, col, len(bad_indices), bad_indices,
            )
        mask_invalid |= col_invalid

    n_dropped = mask_invalid.sum()
    if n_dropped > 0:
        logger.warning(
            "[%s] Dropping %d row(s) in total out of %d.",
            split, n_dropped, len(df),
        )

    return df[~mask_invalid].reset_index(drop=True)


def prepare_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_name: str,
    max_length: int | None = 512,
    use_fast: bool = True,
) -> tuple[ClassificationDataset, ClassificationDataset, dict, dict]:
    """Tokenize text and encode labels, returning ClassificationDataset objects.

    Rows with missing (NaN) or empty string values in text_col or label_col
    are dropped before tokenization, with a warning logged for each.

    Label maps are built from the union of train and test labels so that
    unseen test labels do not cause key errors.

    Args:
        train_df: Training DataFrame from split().
        test_df: Test DataFrame from split().
        text_col: Name of the text column.
        label_col: Name of the label column.
        model_name: HuggingFace model name or local path for the tokenizer.
        max_length: Maximum number of tokens before truncation.
            - 512 (default) → truncate to 512 tokens (BERT's maximum).
            - None → use the model's built-in maximum (e.g. 4096 for Longformer).
            - int → truncate to exactly that many tokens.
        use_fast: Use the HuggingFace fast (Rust) tokenizer. Default True.
                  Set False for models that lack a fast tokenizer
                  (e.g. BlueBERT, BioELECTRA) to avoid a falling-back warning.

    Returns:
        train_dataset: Tokenized ClassificationDataset for training.
        test_dataset:  Tokenized ClassificationDataset for evaluation.
        label2id:      Dict mapping class label strings to integer IDs.
        id2label:      Dict mapping integer IDs to class label strings.
    """
    started = time.perf_counter()
    # Drop rows with missing or empty values before any further processing
    train_df = _drop_invalid_rows(train_df, text_col, label_col, split="train")
    test_df = _drop_invalid_rows(test_df, text_col, label_col, split="test")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    # Build label maps from the union of both splits to avoid missing keys
    label_list = sorted(set(train_df[label_col]).union(test_df[label_col]))
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_labels = train_df[label_col].map(label2id).tolist()
    test_labels = test_df[label_col].map(label2id).tolist()

    train_dataset = ClassificationDataset(
        train_df[text_col].tolist(), train_labels, tokenizer, max_length
    )
    test_dataset = ClassificationDataset(
        test_df[text_col].tolist(), test_labels, tokenizer, max_length
    )

    logger.info(
        "prepare_dataset(): completed in %.2fs — model=%s, use_fast=%s, max_length=%s, "
        "%d train / %d test rows, %d classes.",
        time.perf_counter() - started,
        model_name,
        use_fast,
        max_length,
        len(train_df),
        len(test_df),
        len(label2id),
    )

    return train_dataset, test_dataset, label2id, id2label
