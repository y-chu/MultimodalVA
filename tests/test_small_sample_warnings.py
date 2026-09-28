"""Small-sample guards on the row partitions.

Two things are checked:

1. Every train/test and train/validation split warns when a class is left with
   fewer than ``SMALL_CLASS_MIN`` rows — the user asked for this so a run on
   data too thin for a cause to be learned or scored says so instead of
   reporting a per-class F1 of 0.0 that looks like a modelling result.
2. A validation split does not *fail* on very small data. The stratified split
   needs one held-out row per class; when the requested share is thinner than
   that, sklearn raises, and inside a search that surfaced as "every trial
   failed" with an error blaming dependencies. ``stratified_indices()`` is the
   one implementation all pipelines use, so all of them degrade the same way.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from multimodalva.utils.split import (
    SMALL_CLASS_MIN,
    reset_small_class_warnings,
    split,
    stratified_indices,
    warn_small_classes,
)


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """Each test sees its own warnings: the dedupe is process-wide by design."""
    reset_small_class_warnings()
    yield
    reset_small_class_warnings()


# ---------------------------------------------------------------------------
# warn_small_classes
# ---------------------------------------------------------------------------

def test_warns_and_names_the_small_classes(caplog):
    labels = ["a"] * 40 + ["b"] * 4 + ["c"] * 1
    with caplog.at_level(logging.WARNING):
        small = warn_small_classes(labels, "the training set")
    assert small == {"c": 1, "b": 4}
    assert "Small sample size" in caplog.text
    assert "c=1" in caplog.text and "b=4" in caplog.text
    assert "the training set" in caplog.text
    # The class with enough rows is not named.
    assert "a=" not in caplog.text


def test_silent_when_every_class_is_large_enough(caplog):
    with caplog.at_level(logging.WARNING):
        small = warn_small_classes(["a"] * 10 + ["b"] * 5, "the test set")
    assert small == {}
    assert caplog.text == ""


def test_a_class_absent_from_this_side_is_reported_as_zero(caplog):
    """The dangerous case: the split kept none of a class at all."""
    with caplog.at_level(logging.WARNING):
        small = warn_small_classes(
            ["a"] * 30, "the test set", classes=["a", "b"],
        )
    assert small == {"b": 0}
    assert "b=0" in caplog.text


def test_threshold_is_five_and_is_exclusive():
    assert SMALL_CLASS_MIN == 5
    assert warn_small_classes(["a"] * 5, "x") == {}
    assert warn_small_classes(["a"] * 4, "x") == {"a": 4}


def test_identical_warning_is_emitted_once_per_process(caplog):
    """A per-fold loop reports the same small classes once, not once per fold."""
    labels = ["a"] * 40 + ["b"] * 2
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            assert warn_small_classes(labels, "the training set") == {"b": 2}
    assert caplog.text.count("Small sample size") == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        # A different part of the split is a different message.
        warn_small_classes(labels, "the test set")
    assert caplog.text.count("Small sample size") == 1


def test_long_lists_are_truncated(caplog):
    labels = [f"c{i}" for i in range(25)] + ["big"] * 30
    with caplog.at_level(logging.WARNING):
        small = warn_small_classes(labels, "the training set")
    assert len(small) == 25
    assert "and 15 more" in caplog.text


# ---------------------------------------------------------------------------
# split() — the train/test split every pipeline goes through
# ---------------------------------------------------------------------------

def _frame(counts: dict) -> pd.DataFrame:
    rows = [{"text": f"n {c} {i}", "label": c} for c, n in counts.items() for i in range(n)]
    return pd.DataFrame(rows)

def test_train_test_split_warns_about_both_sides(caplog):
    # 5 rows of "rare": 4 train / 1 test at test_size=0.2 — under 5 on both sides.
    df = _frame({"common": 200, "rare": 5})
    with caplog.at_level(logging.WARNING):
        train_df, test_df = split(df, label_col="label", text_col="text")
    assert "the training set" in caplog.text
    assert "the test set" in caplog.text
    assert len(train_df) + len(test_df) == len(df)


def test_train_test_split_is_silent_on_healthy_data(caplog):
    df = _frame({"a": 100, "b": 100})
    with caplog.at_level(logging.WARNING):
        split(df, label_col="label", text_col="text")
    assert "Small sample size" not in caplog.text


def test_predefined_split_col_is_checked_too(caplog):
    df = _frame({"common": 100, "rare": 3})
    df["fold"] = ["train"] * (len(df) - 20) + ["test"] * 20
    with caplog.at_level(logging.WARNING):
        split(df, label_col="label", text_col="text", split_col="fold")
    assert "Small sample size" in caplog.text


# ---------------------------------------------------------------------------
# stratified_indices() — the validation split
# ---------------------------------------------------------------------------

def test_plain_case_splits_stratified_and_covers_every_row():
    labels = ["a"] * 60 + ["b"] * 40
    train_idx, val_idx = stratified_indices(labels, 0.2, random_state=0)
    assert len(val_idx) == 20
    assert sorted(train_idx + val_idx) == list(range(100))
    held = [labels[i] for i in val_idx]
    assert held.count("a") == 12 and held.count("b") == 8


def test_slice_thinner_than_the_class_count_is_enlarged(caplog):
    """The pre-existing crash: 11 classes, 10% of 90 rows = 9 held-out rows."""
    labels = [f"c{i % 11}" for i in range(90)]
    with caplog.at_level(logging.WARNING):
        train_idx, val_idx = stratified_indices(labels, 0.1, random_state=0)
    assert len(val_idx) == 11
    assert len({labels[i] for i in val_idx}) == 11
    assert "fewer than the 11 classes" in caplog.text


def test_singleton_class_falls_back_to_an_unstratified_split(caplog):
    labels = ["a"] * 30 + ["b"] * 30 + ["lonely"]
    with caplog.at_level(logging.WARNING):
        train_idx, val_idx = stratified_indices(labels, 0.2, random_state=0)
    assert sorted(train_idx + val_idx) == list(range(61))
    assert "cannot be stratified" in caplog.text


def test_same_seed_gives_the_same_partition():
    labels = ["a"] * 50 + ["b"] * 50
    first = stratified_indices(labels, 0.2, random_state=7)
    again = stratified_indices(labels, 0.2, random_state=7)
    other = stratified_indices(labels, 0.2, random_state=8)
    assert first == again
    assert first != other


def test_validation_split_warns_about_the_training_rows_only(caplog):
    """The held-out slice is expected to be thin; the training rows are not."""
    labels = ["common"] * 200 + ["rare"] * 4
    with caplog.at_level(logging.WARNING):
        stratified_indices(labels, 0.1, random_state=0)
    assert "the training rows left after the validation split" in caplog.text
    # Not a second warning about the validation slice itself.
    assert caplog.text.count("Small sample size") == 1


def test_warning_names_causes_not_label_ids(caplog):
    """The text pipeline holds integer labels; the log should still read as causes."""
    labels = [0] * 200 + [1] * 3
    with caplog.at_level(logging.WARNING):
        stratified_indices(
            labels, 0.1, random_state=0, label_names={0: "TB", 1: "Meningitis"},
        )
    assert "Meningitis=" in caplog.text
    assert "1=" not in caplog.text


def test_warning_can_be_turned_off(caplog):
    labels = ["common"] * 200 + ["rare"] * 2
    with caplog.at_level(logging.WARNING):
        stratified_indices(labels, 0.1, random_state=0, warn_below=None)
    assert "Small sample size" not in caplog.text


# ---------------------------------------------------------------------------
# The call sites use the shared helper — not their own copy
# ---------------------------------------------------------------------------

def test_text_val_split_survives_a_slice_thinner_than_the_classes():
    """train_text()'s internal split, the site the original crash came from."""
    from multimodalva.text.train import _val_split

    class _Tiny:
        def __init__(self, labels):
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {"labels": self.labels[i]}

    labels = [i % 11 for i in range(90)]
    train_ds, val_ds = _val_split(_Tiny(labels), 0.1, random_state=0)
    assert len(val_ds) == 11
    assert len(train_ds) + len(val_ds) == 90


def test_every_validation_split_in_the_package_uses_the_shared_helper():
    """Anti-drift: a new copy of the split logic would reintroduce the crash."""
    import ast
    import pathlib

    import multimodalva

    root = pathlib.Path(multimodalva.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "split.py":
            continue  # the one place sklearn's splitter is called
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "train_test_split":
                continue
            kwargs = {kw.arg for kw in node.keywords}
            if "stratify" in kwargs:
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == [], (
        "stratified split(s) outside utils/split.py: "
        + ", ".join(offenders)
        + " — call stratified_indices() so small data degrades instead of raising."
    )


def test_feature_fusion_leaves_its_validation_split_to_automm():
    """AutoMM splits its own holdout; the package only passes the size.

    A package-side carve existed for two days (it gave feature fusion the same
    two-seed separation as every other pipeline) and was removed on 2026-09-24:
    it made the package pre-empt a documented AutoMM behaviour and kept a copy
    of AutoGluon's internal ``default_holdout_frac`` that could drift silently.
    ``_package_reviews/archive_automm_tuning_split_2026-09-24.md`` has the code
    and the reasoning. This test stops it coming back by accident.
    """
    import ast
    import inspect

    from multimodalva.ensemble import feature_fusion as ff

    assert not hasattr(ff, "_carve_tuning_split")
    assert not hasattr(ff, "_automm_default_holdout")

    calls = [
        node for node in ast.walk(ast.parse(inspect.getsource(ff)))
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "fit"
        and any(kw.arg == "train_data" for kw in node.keywords)
    ]
    assert calls, "no predictor.fit(train_data=...) call found"
    for call in calls:
        named = {kw.arg for kw in call.keywords}
        assert "tuning_data" not in named, (
            "feature fusion must let AutoMM hold out its own validation rows"
        )
        assert "seed" in named, "AutoMM's fit() must get the run's train_seed"
