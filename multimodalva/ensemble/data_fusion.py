"""
Ensemble strategy 1 — Data-level fusion — dataset preparation.

Converts each row's tabular features into a natural-language sentence and
concatenates the result with the existing free-text narrative.  The combined
text is then passed to ``DataFusionClassifier`` (``data_fusion_classifier.py``)
for fine-tuning.

Pipeline (this module):
    DataFrame
      → tabular_to_text()  — structured features → natural-language description
      → build_fused_text() — concatenate tabular description + narrative

Question descriptions (qdesc):
    When a ``qdesc`` DataFrame is supplied, each variable's natural-language rendering
    is driven by the ``qdesc.csv`` file in ``utils/``:

        indic  — variable name (matches DataFrame column)
        type   — "demographics" | "symptom" | "environment" | "diagnosis" | "behavior" | "service" | "injury"
        yes    — phrase for positive answer (e.g. "had", "was"), NaN for demographics vars
        no     — phrase for negative answer (e.g. "had no", "was not"), NaN for demographics vars
        desc   — short human-readable description (e.g. "fever")

    Demographics variables (type == "demographics") are joined as a single opening sentence:
        "The deceased was male, 50 to 64 years old."

    All other variables are rendered per answer:
        positive: "[prefix] had fever."
        negative: "[prefix] had no fever."   (only if with_neg=True)

    Variables not in qdesc fall back to ``templates`` / ``binary_map``.

Public API:
    load_qdesc(path)
        — load qdesc.csv from utils/ (or a custom path)
    qdesc_feature_overlap(feature_cols, qdesc, qdesc_path, detail, plot)
        — compare feature columns against qdesc indic values; reports coverage
          before a data-fusion run
    tabular_to_text(row, feature_cols, qdesc, templates, binary_map,
                    with_neg, prefix_cols, yes_no_map, group_symptoms)
        — convert one DataFrame row's features to a natural-language string
    build_fused_text(df, text_col, feature_cols, qdesc, templates, binary_map, ...)
        — apply tabular_to_text() to every row and return the fused text Series
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default separator inserted between the tabular description and the narrative.
# A blank line helps the tokeniser treat them as distinct segments.
DEFAULT_SEPARATOR = "\n\n"

# Default human-readable labels for binary (0/1) indicator columns when qdesc
# is not provided.
DEFAULT_BINARY_MAP: dict = {0: "no", 1: "yes"}

# Default prefix columns for WHO/IVSS VA standard indicator names.
# Maps column name → subject noun/pronoun used when that indicator is positive.
# Evaluated in order; the first matching positive column wins.
DEFAULT_PREFIX_COLS: dict[str, str] = {
    "i022g": "The baby",    # neonate (< 1 month)
    "i022f": "The infant",  # post-neonatal (1–11 months)
    "i022e": "The child",   # child 1–4 years
    "i019a": "He",          # male
    "i019b": "She",         # female
}

# Subject used when no prefix column matches.
DEFAULT_PREFIX_FALLBACK = "The deceased"

# Built-in yes→no verb rules covering all patterns in the standard IVSS VA
# questionnaire.  Used when a qdesc row is missing its yes/no columns, or when
# qdesc is not used at all.  Maps the positive verb phrase → negative phrase.
# The qdesc yes/no columns override these rules on a per-variable basis.
DEFAULT_YES_NO_MAP: dict[str, str] = {
    "did":      "did not",
    "had":      "had no",
    "had ever": "never",
    "was":      "was not",
}

# Fallback positive/negative verbs when a yes value is absent or not found in
# DEFAULT_YES_NO_MAP.
DEFAULT_YES_VERB = "had"
DEFAULT_NO_VERB  = "had no"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_qdesc(path: str | Path | None = None) -> pd.DataFrame:
    """Load the question-description table from *qdesc.csv*.

    Args:
        path: Path to ``qdesc.csv``.  If None, loads from the canonical
              location: ``multimodalva/utils/qdesc.csv``.

    Returns:
        DataFrame with columns: indic, qdesc, sdesc, type, yes, no, desc.

    Raises:
        FileNotFoundError: If the file cannot be found at the given path.
    """
    if path is None:
        path = Path(__file__).parent.parent / "utils" / "qdesc.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"qdesc.csv not found at: {path}")
    return pd.read_csv(path)


# Module-level cache so qdesc.csv is read at most once per process.
_QDESC_CACHE: pd.DataFrame | None = None


def _get_default_qdesc() -> pd.DataFrame | None:
    """Return the cached qdesc DataFrame, loading from utils/qdesc.csv on first call.

    Returns None (and logs a debug message) if the file cannot be found, so
    callers fall back to template / binary_map mode gracefully.
    """
    global _QDESC_CACHE
    if _QDESC_CACHE is None:
        try:
            _QDESC_CACHE = load_qdesc()
            logger.debug("qdesc.csv loaded from utils/ and cached.")
        except FileNotFoundError:
            logger.debug(
                "utils/qdesc.csv not found; tabular_to_text will use "
                "template/binary_map fallback mode."
            )
            return None
    return _QDESC_CACHE


def qdesc_feature_overlap(
    feature_cols: list[str],
    qdesc: "pd.DataFrame | None" = None,
    *,
    qdesc_path: "str | Path | None" = None,
    detail: bool = False,
    plot: bool = True,
) -> dict:
    """Compare feature columns against qdesc indicator names.

    Useful before a data-fusion run: shows which tabular columns will be
    converted to natural language (overlap), which will be silently skipped by
    ``tabular_to_text()`` (only in data — add ``templates=`` to cover these),
    and which qdesc entries have no matching data column (only in qdesc).

    Args:
        feature_cols: Column names present in the tabular DataFrame.
        qdesc:        qdesc DataFrame with at least an ``indic`` column.
                      When ``None``, auto-loaded from ``utils/qdesc.csv``
                      (same cached copy used by ``tabular_to_text()``).
                      Pass the result of ``load_qdesc()`` to use a custom file.
        qdesc_path:   Path to a qdesc CSV or Excel file.  Only used when
                      ``qdesc`` is ``None`` and the default path is wrong.
        detail:       When ``True``, the returned dict also includes
                      ``only_data`` (sorted list) and ``only_qdesc_df``
                      (DataFrame with ``sdesc`` / ``type`` columns when
                      available).  Default ``False``.
        plot:         Show a bar chart of the three set sizes.  Default True.

    Returns:
        dict with keys:

            ``n_features``   — number of feature columns supplied
            ``n_qdesc``      — number of unique indic values in qdesc
            ``n_overlap``    — columns present in both
            ``n_only_data``  — columns present in data but not in qdesc
            ``n_only_qdesc`` — qdesc indicators absent from data
            ``overlap_df``   — DataFrame (``indicator``, ``sdesc``, ``type``)
                               for the overlapping indicators

            When ``detail=True``, also:

            ``only_data``     — sorted list of columns only in the data
            ``only_qdesc_df`` — DataFrame (``indicator``, ``sdesc``, ``type``)
                                for qdesc entries absent from the data

    Raises:
        ValueError: If qdesc cannot be loaded or has no ``indic`` column.

    Example::

        from multimodalva.ensemble.data_fusion import load_qdesc, qdesc_feature_overlap

        # Quick summary + bar chart (auto-loads qdesc.csv)
        result = qdesc_feature_overlap(feature_cols)

        # Inspect unmapped columns before building templates=
        result = qdesc_feature_overlap(feature_cols, detail=True, plot=False)
        print(result["only_data"])       # → add these to templates= in build_fused_text
        print(result["overlap_df"])      # → these will be rendered via qdesc
    """
    # ── Load qdesc ──────────────────────────────────────────────────────────
    if qdesc is None:
        if qdesc_path is not None:
            qdesc = load_qdesc(qdesc_path)
        else:
            qdesc = _get_default_qdesc()
            if qdesc is None:
                raise ValueError(
                    "utils/qdesc.csv not found.  "
                    "Pass the DataFrame directly via qdesc= or supply qdesc_path=."
                )

    if "indic" not in qdesc.columns:
        raise ValueError(
            f"qdesc DataFrame must contain an 'indic' column.  "
            f"Found columns: {list(qdesc.columns)}."
        )

    # ── Build sets ──────────────────────────────────────────────────────────
    feat_set  = set(feature_cols)
    qdesc_set = set(qdesc["indic"].dropna().astype(str))

    overlap       = sorted(feat_set & qdesc_set)
    only_in_data  = sorted(feat_set - qdesc_set)
    only_in_qdesc = sorted(qdesc_set - feat_set)

    logger.info(
        "Feature columns: %d  |  qdesc indic: %d  |  "
        "Overlap: %d  |  Only in data: %d  |  Only in qdesc: %d",
        len(feat_set), len(qdesc_set),
        len(overlap), len(only_in_data), len(only_in_qdesc),
    )

    # ── Build DataFrames ────────────────────────────────────────────────────
    qdesc_indexed = qdesc.set_index("indic")
    extra_cols    = [c for c in ("sdesc", "type") if c in qdesc_indexed.columns]

    def _build_df(keys: list[str]) -> pd.DataFrame:
        valid = [k for k in keys if k in qdesc_indexed.index]
        if not valid:
            return pd.DataFrame(columns=["indicator"] + extra_cols)
        sub = qdesc_indexed.loc[valid, extra_cols].reset_index()
        return sub.rename(columns={"indic": "indicator"})[["indicator"] + extra_cols]

    overlap_df = _build_df(overlap)

    # ── Plot ────────────────────────────────────────────────────────────────
    if plot:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        labels  = ["Overlap\n(both)", "Only in\ndata", "Only in\nqdesc"]
        counts  = [len(overlap), len(only_in_data), len(only_in_qdesc)]
        colours = ["steelblue", "orange", "grey"]

        _, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, counts, color=colours, edgecolor="white", width=0.5)
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(cnt),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax.set_ylabel("Number of indicators")
        ax.set_title(
            f"qdesc / feature-column overlap\n"
            f"({len(feat_set)} feature cols · {len(qdesc_set)} qdesc indic)"
        )
        ax.grid(True, axis="y", alpha=0.35)
        plt.tight_layout()
        plt.show()

    # ── Return ──────────────────────────────────────────────────────────────
    result: dict = {
        "n_features":   len(feat_set),
        "n_qdesc":      len(qdesc_set),
        "n_overlap":    len(overlap),
        "n_only_data":  len(only_in_data),
        "n_only_qdesc": len(only_in_qdesc),
        "overlap_df":   overlap_df,
    }
    if detail:
        result["only_data"]     = only_in_data
        result["only_qdesc_df"] = _build_df(only_in_qdesc)
    return result


def _is_positive(val) -> bool:
    """Return True if *val* represents a positive / 'yes' answer."""
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return str(val).strip().lower() in ("y", "yes", "1", "true")


def _is_negative(val) -> bool:
    """Return True if *val* represents a negative / 'no' answer."""
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return str(val).strip().lower() in ("n", "no", "0", "false")


def _get_prefix(
    row: pd.Series,
    prefix_cols: dict[str, str],
    fallback: str,
) -> str:
    """Determine the subject noun/pronoun for a VA record row.

    Iterates over *prefix_cols* in order and returns the label for the first
    column that is both present in *row* and has a positive value.

    Args:
        row:         A single DataFrame row.
        prefix_cols: Ordered dict mapping column name → subject string.
        fallback:    String returned when no column matches.

    Returns:
        Subject string, e.g. "He", "The child", "The deceased".
    """
    for col, label in prefix_cols.items():
        if col in row.index and _is_positive(row[col]):
            return label
    return fallback


def _join_symptom_list(
    prefix: str,
    verb: str,
    descs: list[str],
    conjunction: str = "and",
) -> str:
    """Format a grouped symptom phrase as a single sentence.

    Examples::

        _join_symptom_list("He", "had", ["fever"])
        → "He had fever."
        _join_symptom_list("He", "had", ["fever", "cough"], "and")
        → "He had fever and cough."
        _join_symptom_list("He", "had no", ["headache", "rash", "vomiting"], "or")
        → "He had no headache, rash, or vomiting."

    Args:
        prefix:      Subject string ("He", "She", "The deceased", …).
        verb:        Verb phrase ("had", "had no", "did not", …).
        descs:       List of symptom/sign descriptions.
        conjunction: Final joining word. Use "and" for positives, "or" for negatives.

    Returns:
        A single formatted sentence string.
    """
    if len(descs) == 1:
        return f"{prefix} {verb} {descs[0]}."
    if len(descs) == 2:
        return f"{prefix} {verb} {descs[0]} {conjunction} {descs[1]}."
    return f"{prefix} {verb} {', '.join(descs[:-1])}, {conjunction} {descs[-1]}."


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------

def tabular_to_text(
    row: pd.Series,
    feature_cols: list[str],
    qdesc: pd.DataFrame | None = None,
    templates: dict[str, str] | None = None,
    binary_map: dict | None = None,
    with_neg: bool = True,
    prefix_cols: dict[str, str] | None = None,
    yes_no_map: dict[str, str] | None = None,
    group_symptoms: bool = True,
) -> str:
    """Convert one DataFrame row's tabular features to a natural-language sentence.

    Two rendering modes are supported and can be combined:

    **qdesc mode** (when ``qdesc`` is provided, or auto-loaded from utils/):
        Uses the question-description table to render each variable as a
        grammatically natural phrase.

        - Demographics variables (``type == "demographics"``): joined as an opening sentence.
          E.g. "The deceased was male, 50 to 64 years old."
        - Other variables: rendered per answer.  Verb phrases come from the
          qdesc ``yes``/``no`` columns when present; otherwise resolved via
          ``yes_no_map`` (the built-in ``DEFAULT_YES_NO_MAP`` covers all standard
          IVSS VA patterns — no need to populate those qdesc columns).

          ``group_symptoms=True`` (default): all symptoms sharing the same verb
          are joined into one sentence — significantly more token-efficient.
          E.g. "He had fever, cough, and difficulty breathing.
                He had no headache, rash, or vomiting."

          ``group_symptoms=False``: one sentence per indicator (original style).
          E.g. "He had fever. He had cough. He had no headache."

    **Template / binary fallback** (for columns not covered by qdesc, or when
    qdesc cannot be found):
        - ``templates[col]``: Python format string receiving ``{value}``.
          E.g. ``{"age": "Patient age: {value} years."}``
        - ``binary_map``: renders 0/1 values as human text.
          Default: ``{0: "no", 1: "yes"}``.
        - Otherwise: ``"col: value."`` pattern.

    Args:
        row:            A single row from a DataFrame (pd.Series).
        feature_cols:   Column names to include in the description.
        qdesc:          Question-description DataFrame (from load_qdesc()).
                        Only the ``indic``, ``type``, and ``desc`` columns are
                        required; ``yes``/``no`` columns are optional and override
                        ``yes_no_map`` when present.
                        If None, auto-loaded from ``utils/qdesc.csv`` on first call
                        and cached.  Pass an empty DataFrame to force template-only mode.
        templates:      Optional dict mapping column name → format string with ``{value}``.
                        Applied to columns not covered by qdesc.
        binary_map:     Mapping for binary (0/1) column values.
                        Default ``{0: "no", 1: "yes"}``.
        with_neg:       Include negative symptom phrases. Default True.
        prefix_cols:    Ordered dict mapping column name → subject noun/pronoun.
                        First positive column determines the prefix.
                        Default: ``DEFAULT_PREFIX_COLS`` (WHO/IVSS standard names).
        yes_no_map:     Dict mapping positive verb → negative verb.
                        Used when a qdesc row's ``yes``/``no`` columns are absent or NaN.
                        Default: ``DEFAULT_YES_NO_MAP`` (covers all standard IVSS patterns).
        group_symptoms: When True (default), group all symptoms sharing the same verb
                        into a single comma-separated sentence instead of one sentence
                        per indicator while preserving all diagnostic information.

    Returns:
        A single natural-language string describing the row's features.
        Returns an empty string if no valid features are found.
    """
    templates   = templates or {}
    binary_map  = binary_map  if binary_map  is not None else DEFAULT_BINARY_MAP
    prefix_cols = prefix_cols if prefix_cols is not None else DEFAULT_PREFIX_COLS
    yes_no_map  = yes_no_map  if yes_no_map  is not None else DEFAULT_YES_NO_MAP

    # Auto-load qdesc from utils/ if not supplied (cached after first load).
    if qdesc is None:
        qdesc = _get_default_qdesc()

    # Restrict to columns actually present in the row.
    present_cols = [c for c in feature_cols if c in row.index]

    parts: list[str] = []

    # ------------------------------------------------------------------
    # qdesc-aware rendering
    # ------------------------------------------------------------------
    if qdesc is not None and not qdesc.empty:
        qdesc_idx = qdesc.set_index("indic")   # index by indicator name

        qdesc_cols = [c for c in present_cols if c in qdesc_idx.index]
        other_cols = [c for c in present_cols if c not in qdesc_idx.index]

        # Subject noun/pronoun derived from age/sex indicator columns.
        prefix = _get_prefix(row, prefix_cols, DEFAULT_PREFIX_FALLBACK)

        # ---- Demographics variables (age, sex) → opening sentence ---------------
        demo_mask  = qdesc_idx["type"] == "demographics"
        demo_indics = set(qdesc_idx[demo_mask].index)
        demo_positive = [
            c for c in qdesc_cols
            if c in demo_indics and _is_positive(row[c])
        ]
        if demo_positive:
            descs = [str(qdesc_idx.loc[c, "desc"]) for c in demo_positive]
            parts.append(f"{prefix} was {', '.join(descs)}.")

        # ---- Symptom / environment / diagnosis / … variables -----------
        non_demo_indics = set(qdesc_idx[~demo_mask].index)
        symp_cols = [c for c in qdesc_cols if c in non_demo_indics]

        if group_symptoms:
            # Grouped mode: collect descs by verb phrase, emit one sentence per
            # unique verb.  Positives joined with "and"; negatives with "or".
            pos_by_verb: dict[str, list[str]] = {}
            neg_by_verb: dict[str, list[str]] = {}

            for col in symp_cols:
                entry = qdesc_idx.loc[col]
                desc  = str(entry["desc"])
                val   = row[col]

                if _is_positive(val):
                    yes_word = (
                        str(entry["yes"])
                        if "yes" in entry.index and pd.notna(entry["yes"])
                        else DEFAULT_YES_VERB
                    )
                    pos_by_verb.setdefault(yes_word, []).append(desc)
                elif _is_negative(val) and with_neg:
                    if "no" in entry.index and pd.notna(entry["no"]):
                        no_word = str(entry["no"])
                    else:
                        yes_word = (
                            str(entry["yes"])
                            if "yes" in entry.index and pd.notna(entry["yes"])
                            else DEFAULT_YES_VERB
                        )
                        no_word = yes_no_map.get(yes_word, DEFAULT_NO_VERB)
                    neg_by_verb.setdefault(no_word, []).append(desc)

            for verb, descs in pos_by_verb.items():
                parts.append(_join_symptom_list(prefix, verb, descs, "and"))
            for verb, descs in neg_by_verb.items():
                parts.append(_join_symptom_list(prefix, verb, descs, "or"))

        else:
            # Per-symptom mode: one sentence per indicator (original behaviour).
            pos_phrases: list[str] = []
            neg_phrases: list[str] = []

            for col in symp_cols:
                entry = qdesc_idx.loc[col]
                desc  = str(entry["desc"])
                val   = row[col]

                if _is_positive(val):
                    yes_word = (
                        str(entry["yes"])
                        if "yes" in entry.index and pd.notna(entry["yes"])
                        else DEFAULT_YES_VERB
                    )
                    pos_phrases.append(f"{prefix} {yes_word} {desc}.")
                elif _is_negative(val) and with_neg:
                    if "no" in entry.index and pd.notna(entry["no"]):
                        no_word = str(entry["no"])
                    else:
                        yes_word = (
                            str(entry["yes"])
                            if "yes" in entry.index and pd.notna(entry["yes"])
                            else DEFAULT_YES_VERB
                        )
                        no_word = yes_no_map.get(yes_word, DEFAULT_NO_VERB)
                    neg_phrases.append(f"{prefix} {no_word} {desc}.")

            parts.extend(pos_phrases)
            parts.extend(neg_phrases)

    else:
        # No qdesc: all columns fall through to template/binary mode.
        other_cols = present_cols

    # ------------------------------------------------------------------
    # Template / binary fallback for columns not covered by qdesc
    # ------------------------------------------------------------------
    for col in other_cols:
        val = row[col]
        try:
            if pd.isna(val):
                continue
        except (TypeError, ValueError):
            pass  # non-scalar (e.g. list) — fall through to rendering

        if col in templates:
            parts.append(templates[col].format(value=val))
        elif val in binary_map:
            parts.append(f"{col}: {binary_map[val]}.")
        elif _is_positive(val) and 1 in binary_map:
            parts.append(f"{col}: {binary_map[1]}.")
        elif _is_negative(val) and 0 in binary_map:
            if with_neg:
                parts.append(f"{col}: {binary_map[0]}.")
        else:
            parts.append(f"{col}: {val}.")

    return " ".join(parts)


def build_fused_text(
    df: pd.DataFrame,
    text_col: str,
    feature_cols: list[str],
    qdesc: pd.DataFrame | None = None,
    templates: dict[str, str] | None = None,
    binary_map: dict | None = None,
    with_neg: bool = True,
    prefix_cols: dict[str, str] | None = None,
    yes_no_map: dict[str, str] | None = None,
    separator: str = DEFAULT_SEPARATOR,
    group_symptoms: bool = True,
) -> pd.Series:
    """Apply tabular_to_text() to every row and concatenate with the narrative.

    Each fused text is:
        <tabular description> <separator> <narrative>

    Placing the structured description before the narrative ensures the LM sees
    it early (before any truncation) and can attend to it during narrative encoding.

    Args:
        df:             Input DataFrame with both tabular features and a text column.
        text_col:       Column containing the free-text narrative.
        feature_cols:   Columns to convert to text (passed to tabular_to_text).
                        The text column itself should NOT be included here.
        qdesc:          Question-description DataFrame (from load_qdesc()).
                        If None, auto-loaded from ``utils/qdesc.csv`` on first call
                        and cached for subsequent calls.  Pass an empty DataFrame
                        to force template-only mode.
        templates:      Optional column-level format strings (see tabular_to_text).
        binary_map:     Optional 0/1 → string mapping (see tabular_to_text).
        with_neg:       Include negative symptom phrases. Default True.
        prefix_cols:    Subject prefix mapping (see tabular_to_text).
        yes_no_map:     Positive → negative verb map (see tabular_to_text).
                        Default: ``DEFAULT_YES_NO_MAP``.
        separator:      String inserted between the tabular description and the narrative.
                        Default is a blank line ("\\n\\n").
        group_symptoms: Group symptoms sharing the same verb into one sentence.
                        Default True (recommended — ~40–50% fewer tokens).
                        See tabular_to_text() for details.

    Returns:
        pd.Series of fused text strings, one per row, same index as df.

    Raises:
        ValueError: If text_col or any feature_col is missing from df.
    """
    # Auto-load qdesc from utils/ if not supplied (cached after first load).
    if qdesc is None:
        qdesc = _get_default_qdesc()

    if text_col not in df.columns:
        raise ValueError(
            f"text_col '{text_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    missing_feat = [c for c in feature_cols if c not in df.columns]
    if missing_feat:
        raise ValueError(
            f"feature_cols not found in DataFrame: {missing_feat}. "
            f"Available columns: {list(df.columns)}"
        )

    def _fuse_row(row: pd.Series) -> str:
        tab_text  = tabular_to_text(
            row,
            feature_cols=feature_cols,
            qdesc=qdesc,
            templates=templates,
            binary_map=binary_map,
            with_neg=with_neg,
            prefix_cols=prefix_cols,
            yes_no_map=yes_no_map,
            group_symptoms=group_symptoms,
        )
        narrative = str(row[text_col]) if pd.notna(row[text_col]) else ""
        if tab_text and narrative:
            return f"{tab_text}{separator}{narrative}"
        return tab_text or narrative

    return df.apply(_fuse_row, axis=1)
