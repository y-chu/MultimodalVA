"""LLMClassifier: cause assignment by prompting a language model, no training.

Chains: split → (dev slice) → cause schema → [selection over prompt
configurations on the dev slice →] prompt every record of the target partition
→ validate each answer → probability tables.

The run contract is the same as every other task's: ``predictions/`` holds the
three tables, ``training_metadata.json`` names the pipeline, ``validation.json``
the selection score, ``hpo/`` the selection when one ran. Two things are
specific to this task. ``predictions/records.jsonl`` (or ``dev/records.jsonl``)
keeps the model's full answer per record — the ranking, confidence, rationale,
evidence and token counts — and is the store a run resumes from. And
``llm_metadata.csv`` beside the tables carries the same fields as a flat table
keyed by id.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.metrics import score_predictions
from ..utils.predictions import assemble_predictions, save_predictions
from ..utils.provenance import enforce_resume_manifest, make_resume_manifest
from ..utils.runtime import RuntimeTracker, track_run
from ..utils.seeds import seed_everything, set_determinism
from ..utils.split import split, stratified_indices
from ..utils.types import PredictionResult
from . import rank_to_probability
from .backends import LLM_BACKENDS, LLM_DEFAULT_MODEL, build_llm_backend, resolve_llm_backend
from .cause_schema import describe_modules, load_cause_schema
from .prompts import PROMPT_VERSION, PromptSpec, format_record
from .response import extract_json, normalise_response, valid_response
from .selection import describe_llm_cell, resolve_llm_hyperparams

logger = logging.getLogger(__name__)

__all__ = [
    "LLMClassifier", "LLM_PROB_SOURCES", "LLM_EVALUATE_ON", "LLM_RECORDS_FILE",
    "resolve_sections", "split_sections", "count_section_markers",
]

#: How the full probability vector is built from the model's answer.
LLM_PROB_SOURCES = ("stated", "alpha", "topk")
#: Which rows the run's predictions are on: every row given (no split), the
#: test partition of the package's split, or a dev slice of its training rows.
LLM_EVALUATE_ON = ("all", "test", "dev")
LLM_RECORDS_FILE = "records.jsonl"
#: Metrics recorded for every dev or selection pass.
LLM_REPORT_METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "csmf_accuracy")
#: Written into the user message when the backend does not enforce the schema.
_SCHEMA_INSTRUCTION = (
    "Respond with exactly one JSON object and nothing else. It must match this "
    "JSON schema:\n"
)


# ---------------------------------------------------------------------------
# Sections: how a row becomes a record
# ---------------------------------------------------------------------------

def resolve_sections(
    df: pd.DataFrame,
    text_col: str | None,
    sections: "dict[str, str] | None",
    section_markers: "dict[str, str | None] | None",
) -> tuple[list[str], "dict[str, str] | None", "dict[str, str | None] | None"]:
    """``(section names, {name: column} or None, {name: marker} or None)``.

    Three ways in: ``text_col`` alone is one section called ``narrative``;
    ``sections`` maps section names to columns; ``text_col`` with
    ``section_markers`` splits one column on marker lines.
    """
    if sections is not None and section_markers is not None:
        raise ValueError("pass sections= or section_markers=, not both")
    if sections is not None:
        if text_col is not None:
            raise ValueError("pass text_col= or sections=, not both")
        if not isinstance(sections, dict) or not sections:
            raise ValueError("sections= must be a non-empty mapping {section name: column}")
        names = [str(k) for k in sections]
        cols = {str(k): str(v) for k, v in sections.items()}
        missing = [c for c in cols.values() if c not in df.columns]
        if missing:
            raise ValueError(f"sections= names column(s) not in the data: {missing}")
        return names, cols, None
    if not text_col:
        raise ValueError("task='llm' requires text_col (or sections=).")
    if text_col not in df.columns:
        raise ValueError(f"text column {text_col!r} is not in the data.")
    if section_markers is None:
        return ["narrative"], {"narrative": text_col}, None
    if not isinstance(section_markers, dict) or not section_markers:
        raise ValueError("section_markers= must be a non-empty mapping {section name: marker line}")
    names = [str(k) for k in section_markers]
    markers = {str(k): (None if v in (None, "") else str(v)) for k, v in section_markers.items()}
    if any(v is None for v in list(markers.values())[1:]):
        raise ValueError(
            "section_markers=: only the first section may have no marker (it "
            "starts at the beginning of the text); every later section needs one."
        )
    if len(names) < 2:
        raise ValueError("section_markers= needs at least two sections; use text_col alone for one.")
    return names, None, markers


def split_sections(text: str, markers: "dict[str, str | None]") -> dict[str, str]:
    """Split one text into named sections on whole-line markers, strictly.

    Every marker with a value must appear exactly once, as a whole line, in the
    order given; a marker that is the same string for several sections is
    matched by position. The number of segments then equals the number of
    sections. Anything else raises, with the reason.
    """
    lines = str(text).split("\n")
    names = list(markers)
    expected = [markers[n] for n in names]
    marker_set = {m for m in expected if m is not None}
    found = [(i, line.strip()) for i, line in enumerate(lines) if line.strip() in marker_set]
    wanted = [m for m in expected if m is not None]
    if [m for _, m in found] != wanted:
        got = [m for _, m in found]
        if len(got) > len(wanted):
            raise ValueError(f"marker(s) found more than once: expected {wanted}, found {got}")
        if len(got) < len(wanted):
            missing = list(wanted)
            for m in got:
                if m in missing:
                    missing.remove(m)
            raise ValueError(f"marker(s) missing: {missing}")
        raise ValueError(f"markers out of order: expected {wanted}, found {got}")
    bounds = [i for i, _ in found]
    out: dict[str, str] = {}
    if expected[0] is None:
        first_end = bounds[0] if bounds else len(lines)
        out[names[0]] = "\n".join(lines[:first_end]).strip()
        starts = bounds
        rest = names[1:]
    else:
        if bounds and "\n".join(lines[: bounds[0]]).strip():
            raise ValueError(
                f"text before the first marker {expected[0]!r} belongs to no section; "
                "give the first section marker None to start it at the beginning"
            )
        starts = bounds
        rest = names
    for k, name in enumerate(rest):
        start = starts[k] + 1
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        out[name] = "\n".join(lines[start:end]).strip()
    return out


def count_section_markers(texts, markers: "dict[str, str | None]") -> dict[str, int]:
    """Per marker, how many texts contain it as a whole line (for preflight)."""
    counts = {m: 0 for m in markers.values() if m is not None}
    for text in texts:
        present = {line.strip() for line in str(text).split("\n")}
        for marker in counts:
            if marker in present:
                counts[marker] += 1
    return counts


def _clean_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def build_records(
    frame: pd.DataFrame,
    *,
    names: list[str],
    section_cols: "dict[str, str] | None",
    markers: "dict[str, str | None] | None",
    text_col: str | None,
    label_col: str | None,
    id_col: str | None,
    sex_col: str | None,
    age_col: str | None,
    id_prefix: str,
) -> list[dict]:
    """One record per row: id, label (None when unlabelled), sections, demographics."""
    records = []
    problems: list[str] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        rid = str(row[id_col]) if id_col else f"{id_prefix}{position}"
        if markers is not None:
            try:
                sections = split_sections(_clean_text(row[text_col]) or "", markers)
            except ValueError as exc:
                problems.append(f"{rid}: {exc}")
                continue
        else:
            sections = {name: _clean_text(row[section_cols[name]]) for name in names}
        if not any(sections.get(name) for name in names):
            problems.append(f"{rid}: every section is empty")
            continue
        records.append({
            "id": rid,
            "label": str(row[label_col]) if label_col else None,
            "sections": {name: sections.get(name) for name in names},
            "sex": _clean_text(row[sex_col]) if sex_col else None,
            "age": (float(row[age_col]) if age_col and _clean_text(row[age_col]) is not None else None),
        })
    if problems:
        shown = "\n  ".join(problems[:10])
        raise ValueError(
            f"{len(problems)} row(s) cannot be rendered as records (the first "
            f"{min(10, len(problems))}):\n  {shown}\nFix the data or drop these "
            "rows; nothing is guessed."
        )
    return records


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------

class LLMClassifier:
    """Cause assignment by prompting a language model.

    Typical usage::

        clf = LLMClassifier(model_name="openai/gpt-oss-20b", output_dir="runs/llm")
        results = clf.run(df, text_col="narrative", label_col="cause",
                          cause_schema="causes.json", context="Rural ...")

    Args:
        model_name: A Hugging Face Hub id or local directory of a chat model, or
            ``"ollama:<tag>"`` for a model served by ollama.
        output_dir: Root directory for all outputs.
        backend: ``"auto"`` (transformers, or ollama for an ``ollama:`` model),
            ``"transformers"`` or ``"ollama"``.
        backend_kwargs: Settings for the backend's constructor —
            ``device_map``, ``dtype``, ``force_load``, ``trust_remote_code`` for
            transformers; ``host``, ``timeout`` for ollama.
        backend_instance: A ready backend object (tests pass a fake). Overrides
            ``model_name`` and ``backend``.
    """

    def __init__(
        self,
        model_name: str = LLM_DEFAULT_MODEL,
        output_dir: "str | Path" = "runs/llm",
        backend: str = "auto",
        backend_kwargs: dict | None = None,
        backend_instance=None,
    ):
        self.model_name = str(model_name)
        self.output_dir = Path(output_dir)
        self.backend_name, self.model_id = resolve_llm_backend(self.model_name, backend)
        if backend not in LLM_BACKENDS:
            raise ValueError(f"backend={backend!r}; use one of {LLM_BACKENDS}")
        self.backend_setting = backend
        self.backend_kwargs = dict(backend_kwargs or {})
        self.backend = backend_instance
        # Populated after run()
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.dev_df: pd.DataFrame | None = None
        self.spec: PromptSpec | None = None
        self.label2id: dict | None = None
        self.id2label: dict | None = None
        self.hyperparams: dict | None = None
        self.best_hyperparams: dict | None = None
        self.predictions: PredictionResult | None = None
        self.train_metadata: dict | None = None

    # -----------------------------------------------------------------------

    @track_run("llm", track=False)
    def run(
        self,
        df: pd.DataFrame,
        label_col: str | None = None,
        text_col: str | None = None,
        *,
        sections: "dict[str, str] | None" = None,
        section_markers: "dict[str, str | None] | None" = None,
        cause_schema=None,
        context: str | None = None,
        sex_col: str | None = None,
        age_col: str | None = None,
        hyperparams: "dict | str | Any | None" = None,
        evaluate_on: str = "all",
        dev_size: float | int = 0.1,
        dev_ids: "list | str | Path | None" = None,
        prob_source: str = "stated",
        rules_arm: str = "literature",
        batch_size: int = 1,
        timeout: int = 900,
        test_size: float = 0.2,
        split_seed: int = 42,
        train_seed: int = 42,
        deterministic: bool = False,
        stratify: bool = True,
        split_col: str | None = None,
        top_k: int = 3,
        id_col: str | None = None,
        resume: bool = True,
    ) -> dict:
        """Prompt the model for every record of the target partition.

        Args:
            df: The records, one per row.
            label_col: The cause-of-death column. Its values are the causes the
                model may return, and the tables gain a ``true_label`` column.
                May be omitted with ``evaluate_on="all"`` (unlabelled data);
                the cause list then comes from ``cause_schema``, which is
                required in that case.
            text_col: The record text, as one section (``narrative``). With
                ``section_markers`` it is the column to split.
            sections: ``{section name: column}`` for records with several
                sources, in record order. Replaces ``text_col``.
            section_markers: ``{section name: marker line}`` to split
                ``text_col`` on whole-line markers. The first section may have
                marker ``None`` (it starts at the beginning). Every marker must
                appear exactly once, in order, or the row is refused.
            cause_schema: Path or dict giving each cause's definition and
                optional WHO mapping (see :mod:`multimodalva.llm.cause_schema`).
                ``None`` uses the cause names alone.
            context: One or two sentences on the population, for M1. Overrides
                the schema's; ``""`` turns M1 off.
            sex_col / age_col: Columns rendered on the record's demographics line.
            hyperparams: The prompt configuration: ``None`` for the defaults
                (M1–M4, low reasoning effort, temperature 0), a dict of
                ``modules`` / ``reasoning_effort`` / ``temperature`` / ``top_p``
                / ``max_new_tokens``, or ``Optimize(...)`` to select a
                configuration on the dev slice (see
                :mod:`multimodalva.llm.selection`).
            evaluate_on: ``"all"`` (default) prompts every row given, with no
                split — plain zero-shot use, on labelled or unlabelled data;
                split the data yourself if you want a holdout. ``"test"`` makes
                the package's train/test split and predicts the test partition,
                so the run is comparable with trained models on the same split;
                ``"dev"`` predicts a dev slice of the training rows only, under
                ``dev/``, and never touches the test rows.
            dev_size: Share (float) or count (int) of the **training** rows
                carved out as the dev slice, stratified, with ``split_seed``.
            dev_ids: Explicit dev rows by ``id_col`` — a list, or the path of a
                ``dev_ids.json`` an earlier run wrote. Must not overlap the test
                partition.
            prob_source: How ``predictions_full.csv`` is built: ``"stated"``
                (the model's own probabilities, zero outside the ranked causes,
                renormalised), ``"alpha"`` (rank and confidence through fixed a
                priori weights) or ``"topk"`` (stated head, uniform tail).
            rules_arm: ``"literature"`` (the WHO rules) or ``"none"`` for M4.
            batch_size: Records generated per call on the transformers backend.
            timeout: Seconds per request on the ollama backend.
            test_size / split_seed / stratify / split_col: As for every task.
            train_seed: The generation seed. Changes nothing at temperature 0.
            deterministic: Accepted for uniformity; recorded, nothing to switch.
            top_k: Classes listed in the topk table.
            id_col: Row identifier, carried into every table and used for resume.
            resume: Continue an interrupted run from ``records.jsonl`` when the
                run's signature matches; ``False`` starts over.

        Returns:
            The usual result dict: ``predictions``, ``train_metadata``,
            ``best_hyperparams``, ``label2id``, ``id2label``, ``output_dir``,
            plus ``records`` (the per-record answers) and ``dev_scores`` for a
            dev run.
        """
        if evaluate_on not in LLM_EVALUATE_ON:
            raise ValueError(f"evaluate_on={evaluate_on!r}; use one of {LLM_EVALUATE_ON}")
        if label_col is not None and label_col not in df.columns:
            raise ValueError(f"label column {label_col!r} is not in the data.")
        if label_col is None and evaluate_on != "all":
            raise ValueError(
                f"evaluate_on={evaluate_on!r} needs label_col=: a split and a dev "
                "slice are stratified on the labels. Unlabelled data runs with "
                "evaluate_on='all'."
            )
        if label_col is None and cause_schema is None:
            raise ValueError(
                "Without label_col= the cause list has to come from cause_schema=; "
                "pass a schema listing the causes the model may return."
            )
        if prob_source not in LLM_PROB_SOURCES:
            raise ValueError(f"prob_source={prob_source!r}; use one of {LLM_PROB_SOURCES}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        hp_kind, hp_fixed, hp_search, grid = resolve_llm_hyperparams(hyperparams)
        names, section_cols, markers = resolve_sections(df, text_col, sections, section_markers)
        for col, what in ((sex_col, "sex_col"), (age_col, "age_col"), (id_col, "id_col")):
            if col is not None and col not in df.columns:
                raise ValueError(f"{what} {col!r} is not in the data.")
        if id_col is not None and (df[id_col].isna().any() or df[id_col].astype(str).duplicated().any()):
            raise ValueError(f"id_col {id_col!r} must contain non-missing, unique values.")
        if deterministic:
            set_determinism(deterministic)
        seed_everything(train_seed)

        tracker = RuntimeTracker(
            self.output_dir, report_name="pipeline_runtime.json",
            metadata={"pipeline": "llm", "model_name": self.model_name,
                      "backend": self.backend_name, "label_col": label_col,
                      "hyperparams_from": hp_kind, "evaluate_on": evaluate_on},
            logger_=logger,
        )

        # --- 1. split (or not) ------------------------------------------------
        if evaluate_on == "all":
            with tracker.stage("split", details={"rows": len(df), "split": "none"}):
                self.train_df = df.iloc[0:0].reset_index(drop=True)
                self.test_df = df.reset_index(drop=True)
                logger.info("No split: every one of the %d rows is prompted.", len(df))
        else:
            with tracker.stage("split", details={"rows": len(df), "test_size": test_size,
                                                 "split_seed": split_seed, "stratify": stratify}):
                self.train_df, self.test_df = split(
                    df, label_col=label_col, text_col=text_col, test_size=test_size,
                    random_state=split_seed, stratify=stratify, split_col=split_col,
                )
                logger.info("Split: %d train, %d test rows.", len(self.train_df), len(self.test_df))

        # --- 2. the causes and the schema -------------------------------------
        causes = sorted({str(c) for c in df[label_col].dropna()}) if label_col else None
        self.spec = load_cause_schema(cause_schema, causes, context=context, sources=names)
        causes = self.spec.causes
        self.label2id = {c: i for i, c in enumerate(causes)}
        self.id2label = {i: c for c, i in self.label2id.items()}
        for warning in describe_modules(self.spec, hp_fixed["modules"], rules_arm):
            logger.warning(warning)

        # --- 3. the dev slice -------------------------------------------------
        needs_dev = evaluate_on == "dev" or hp_kind == "search"
        if needs_dev and evaluate_on == "all":
            raise ValueError(
                "Optimize(...) selects a prompt configuration on a dev slice of the "
                "training rows, which evaluate_on='all' does not make. Use "
                "evaluate_on='test' (select on the dev slice, predict the test "
                "partition) or 'dev' (select and stop), or split the data yourself "
                "and fix the configuration with hyperparams={...}."
            )
        self.dev_df = None
        dev_info: dict | None = None
        if needs_dev:
            self.dev_df, dev_info = self._carve_dev(
                self.train_df, self.test_df, label_col, id_col, dev_size, dev_ids,
                split_seed, test_size, split_col,
            )

        # --- 4. resume signature ---------------------------------------------
        section_columns = sorted(set(section_cols.values())) if section_cols else [text_col]
        signature_cols = ([label_col] if label_col else []) + list(section_columns)
        for col in (sex_col, age_col, id_col):
            if col and col not in signature_cols:
                signature_cols.append(col)
        frames = ({"data": self.test_df[signature_cols]} if evaluate_on == "all"
                  else {"train": self.train_df[signature_cols], "test": self.test_df[signature_cols]})
        if self.dev_df is not None:
            frames["dev"] = self.dev_df[signature_cols]
        schema_doc = self.spec.schema
        manifest = make_resume_manifest(
            dataframes=frames,
            config={
                "pipeline": "llm", "model_name": self.model_name,
                "backend": self.backend_name, "label_col": label_col,
                "sections": names, "section_cols": section_cols,
                "section_markers": markers, "hyperparams": hyperparams,
                "cause_schema": schema_doc, "context": self.spec.population_context,
                "sex_col": sex_col, "age_col": age_col, "rules_arm": rules_arm,
                "prob_source": prob_source, "evaluate_on": evaluate_on,
                "split_seed": split_seed, "train_seed": train_seed,
                "test_size": test_size, "stratify": stratify,
                "prompt_version": PROMPT_VERSION,
            },
        )
        enforce_resume_manifest(
            self.output_dir / "resume_manifest.json", manifest, resume=resume,
            artifacts_exist=any((self.output_dir / n).exists() for n in ("hpo", "predictions", "dev")),
            what="this llm run",
        )

        if dev_info is not None:
            # Written only now: the resume guard above treats dev/ as prior work.
            dev_dir = self.output_dir / "dev"
            dev_dir.mkdir(parents=True, exist_ok=True)
            (dev_dir / "dev_ids.json").write_text(json.dumps(dev_info, indent=2))
            dev_info = {k: v for k, v in dev_info.items() if k != "dev_ids"}
            dev_info["ids_file"] = str(dev_dir / "dev_ids.json")

        # --- 5. the backend (last, so a bad configuration fails before a load) --
        if self.backend is None:
            with tracker.stage("load_model", details={"model": self.model_name}):
                self.backend = build_llm_backend(self.model_name, self.backend_setting,
                                                 **self.backend_kwargs)
        self._check_reasoning_effort(hp_fixed, grid)

        # --- 6. records ---------------------------------------------------------
        make = lambda frame, prefix: build_records(  # noqa: E731
            frame, names=names, section_cols=section_cols, markers=markers,
            text_col=text_col, label_col=label_col, id_col=id_col,
            sex_col=sex_col, age_col=age_col, id_prefix=prefix,
        )
        dev_records = make(self.dev_df, "dev") if self.dev_df is not None else None
        test_records = make(self.test_df, "row") if evaluate_on != "dev" else None

        settings_base = {"seed": train_seed, "timeout": timeout}

        # --- 7. selection ------------------------------------------------------
        if hp_kind == "search":
            with tracker.stage("selection", details={"cells": len(grid), "metric": hp_search.metric,
                                                     "n_dev": len(dev_records)}):
                chosen = self._select(grid, hp_search.metric, dev_records, rules_arm,
                                      settings_base, batch_size, prob_source, top_k)
            self.best_hyperparams = chosen
        else:
            self.best_hyperparams = hp_fixed
            logger.info("Prompt configuration %s — %s.",
                        "FROM CALLER" if hp_kind == "fixed" else "DEFAULT",
                        describe_llm_cell(hp_fixed))
        self.hyperparams = self.best_hyperparams

        # --- 8. the pass that is reported ------------------------------------
        target = "dev" if evaluate_on == "dev" else evaluate_on
        records = dev_records if target == "dev" else test_records
        out_dir = self.output_dir / ("dev" if target == "dev" else "predictions")
        with tracker.stage("predict", details={"partition": target, "n": len(records),
                                               "batch_size": batch_size}):
            answers, system_sha, stats = self._predict_records(
                records, self.best_hyperparams, rules_arm, settings_base, batch_size,
                out_dir / LLM_RECORDS_FILE,
            )
        with tracker.stage("export", details={"prob_source": prob_source}):
            self.predictions = self._export(answers, prob_source, top_k, out_dir, id_col is not None,
                                            labelled=label_col is not None)
        dev_scores = None
        if target == "dev":
            dev_scores = self._scores(self.predictions)
            (out_dir / "dev_scores.json").write_text(json.dumps(
                {"metric": hp_search.metric if hp_search else "f1_macro", "scores": dev_scores,
                 "n": len(answers), "hyperparams": _jsonable_cell(self.best_hyperparams)}, indent=2))
            logger.info("Dev scores (selection data, not performance): %s",
                        ", ".join(f"{k}={v:.4f}" for k, v in dev_scores.items()))

        # --- 9. metadata ------------------------------------------------------
        for name, mapping in (("label2id", self.label2id), ("id2label", self.id2label)):
            (self.output_dir / f"{name}.json").write_text(json.dumps(mapping, indent=2))
        self.train_metadata = {
            "pipeline": "llm",
            "model_name": self.model_name,
            "backend": self.backend.describe(),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": system_sha,
            "hyperparams": _jsonable_cell(self.best_hyperparams),
            "hyperparams_from": hp_kind,
            "rules_arm": rules_arm,
            "sections": names,
            "section_cols": section_cols,
            "section_markers": markers,
            "cause_schema": {"name": self.spec.name,
                             "source": str(cause_schema) if not isinstance(cause_schema, dict) else "dict",
                             "sha256": _sha_json(schema_doc)},
            "context": self.spec.population_context,
            "module_warnings": describe_modules(self.spec, self.best_hyperparams["modules"], rules_arm),
            "prob_source": prob_source,
            "evaluate_on": evaluate_on,
            "dev": dev_info,
            "split_seed": split_seed, "train_seed": train_seed, "deterministic": deterministic,
            "test_size": test_size, "stratify": stratify,
            "label_col": label_col, "id_col": id_col, "sex_col": sex_col, "age_col": age_col,
            "n_train": len(self.train_df), "n_test": len(self.test_df),
            "labelled": label_col is not None,
            "n_dev": len(dev_records) if dev_records is not None else 0,
            "n_scored": len(answers), "failures": 0,
            "label2id": self.label2id, "id2label": self.id2label,
            **stats,
        }
        with open(self.output_dir / "training_metadata.json", "w") as fh:
            json.dump(self.train_metadata, fh, indent=2, default=str)
        tracker.write()
        return {
            "predictions": self.predictions,
            "train_metadata": self.train_metadata,
            "best_hyperparams": _jsonable_cell(self.best_hyperparams),
            "label2id": self.label2id,
            "id2label": self.id2label,
            "output_dir": self.output_dir,
            "records": answers,
            "dev_scores": dev_scores,
            "runtime_report": tracker.report_path,
            "runtime_stage_csv": tracker.stage_csv_path,
        }

    # -----------------------------------------------------------------------
    # pieces
    # -----------------------------------------------------------------------

    def _check_reasoning_effort(self, fixed: dict, grid: "list[dict] | None") -> None:
        cells = grid or [fixed]
        asked = {c["reasoning_effort"] for c in cells if c["reasoning_effort"] is not None}
        if asked and not self.backend.supports_reasoning_effort():
            raise ValueError(
                f"{self.model_name} does not accept a reasoning effort, so "
                f"reasoning_effort={sorted(asked)} cannot be honoured. Pass "
                "hyperparams={'reasoning_effort': None} to run it, and report the "
                "arm as having no reasoning-effort setting rather than the default one."
            )

    def _carve_dev(self, train_df, test_df, label_col, id_col, dev_size, dev_ids,
                   split_seed, test_size, split_col):
        """The dev slice, from the training partition only, and its record."""
        if dev_ids is not None:
            if id_col is None:
                raise ValueError("dev_ids= needs id_col= to say which rows they are.")
            if isinstance(dev_ids, (str, Path)):
                doc = json.loads(Path(dev_ids).read_text())
                dev_ids = doc["dev_ids"] if isinstance(doc, dict) else doc
            wanted = [str(x) for x in dev_ids]
            if len(set(wanted)) != len(wanted):
                raise ValueError("dev_ids contains duplicates.")
            test_ids = set(test_df[id_col].astype(str))
            overlap = sorted(set(wanted) & test_ids)
            if overlap:
                raise ValueError(
                    f"{len(overlap)} dev id(s) are in the test partition: {overlap[:10]}. "
                    "The dev slice must come from the training rows only."
                )
            train_ids = set(train_df[id_col].astype(str))
            unknown = sorted(set(wanted) - train_ids)
            if unknown:
                raise ValueError(
                    f"{len(unknown)} dev id(s) are not training rows: {unknown[:10]}."
                )
            dev_df = train_df[train_df[id_col].astype(str).isin(wanted)].reset_index(drop=True)
            source = "dev_ids"
        else:
            labels = train_df[label_col].tolist()
            n_classes = len(set(labels))
            if isinstance(dev_size, int) and not isinstance(dev_size, bool) and dev_size < n_classes:
                # A slice smaller than the cause list cannot hold every cause, so
                # it is a plain seeded sample: the row count asked for is kept,
                # which is what a few-record trial run wants.
                if dev_size < 1 or dev_size >= len(train_df):
                    raise ValueError(
                        f"dev_size={dev_size} must be between 1 and the number of "
                        f"training rows ({len(train_df)})."
                    )
                logger.warning(
                    "dev_size=%d is smaller than the %d causes, so the dev slice is a "
                    "seeded random sample rather than a stratified one; some causes "
                    "are absent from it.", dev_size, n_classes,
                )
                held = sorted(np.random.default_rng(split_seed).choice(
                    len(train_df), size=dev_size, replace=False).tolist())
                source = "random sample from train (fewer rows than causes)"
            else:
                _, held = stratified_indices(labels, dev_size, split_seed, what="dev", log=logger)
                source = "stratified from train"
            dev_df = train_df.iloc[sorted(held)].reset_index(drop=True)
        ids = dev_df[id_col].astype(str).tolist() if id_col else [f"dev{i}" for i in range(len(dev_df))]
        info = {
            "source": source, "n": len(dev_df), "dev_size": dev_size,
            "split_seed": split_seed, "test_size": test_size,
            "split": "supplied" if split_col else "random",
            "id_col": id_col, "dev_ids": ids,
        }
        logger.info("Dev slice: %d rows (%s); test partition untouched.", len(dev_df), source)
        return dev_df, info

    def _select(self, grid, metric, dev_records, rules_arm, settings_base, batch_size,
                prob_source, top_k) -> dict:
        """Run every cell on the dev slice; write hpo/; return the best cell."""
        hpo_dir = self.output_dir / "hpo"
        hpo_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        logger.info("Prompt configuration FROM SELECTION — %d cell(s) on %d dev records, metric=%s.",
                    len(grid), len(dev_records), metric)
        for index, cell in enumerate(grid):
            cell_dir = hpo_dir / "cells" / f"cell_{index:02d}"
            answers, sha, stats = self._predict_records(
                dev_records, cell, rules_arm, settings_base, batch_size,
                cell_dir / LLM_RECORDS_FILE,
            )
            result = self._export(answers, prob_source, top_k, cell_dir, False, labelled=True)
            scores = self._scores(result)
            rows.append({
                "number": index, "state": "COMPLETE", "value": scores[metric],
                "params_modules": "+".join(cell["modules"]) or "base",
                "params_reasoning_effort": cell["reasoning_effort"],
                "params_temperature": cell["temperature"],
                "params_top_p": cell["top_p"],
                "params_max_new_tokens": cell["max_new_tokens"],
                **{f"user_attrs_{k}": v for k, v in scores.items()},
                "user_attrs_n": len(answers),
                "user_attrs_mean_prompt_tokens": stats["mean_prompt_tokens"],
                "user_attrs_mean_wall_s": stats["mean_wall_s"],
                "prompt_sha256": sha,
            })
            logger.info("  cell %d/%d  %s=%.4f  (%s)", index + 1, len(grid), metric,
                        scores[metric], describe_llm_cell(cell))
        trials = pd.DataFrame(rows)
        trials.to_csv(hpo_dir / "hpo_trials.csv", index=False)
        best = int(trials["value"].idxmax())
        chosen = grid[best]
        # Hyperparameters only: never a score (CLAUDE.md §2).
        (hpo_dir / "best_hyperparams.json").write_text(json.dumps(_jsonable_cell(chosen), indent=2))
        logger.info("Selected cell %d: %s (%s=%.4f on the dev slice).", best,
                    describe_llm_cell(chosen), metric, trials.loc[best, "value"])
        return chosen

    def _predict_records(self, records, cell, rules_arm, settings_base, batch_size, store: Path):
        """Prompt the model for every record not already answered in ``store``."""
        import hashlib  # noqa: PLC0415

        spec = self.spec
        system = spec.system_prompt(modules=cell["modules"], probabilities=True,
                                    evidence=True, rules_arm=rules_arm)
        system_sha = hashlib.sha256(system.encode("utf-8")).hexdigest()
        schema = spec.response_schema(True, True)
        settings = {**settings_base, **{k: cell[k] for k in ("reasoning_effort", "temperature",
                                                                "top_p", "max_new_tokens")}}
        if not self.backend.supports_reasoning_effort():
            settings["reasoning_effort"] = None
        instruction = "" if self.backend.enforces_schema else (
            "\n\n" + _SCHEMA_INSTRUCTION + json.dumps(schema)
        )

        store.parent.mkdir(parents=True, exist_ok=True)
        done: dict[str, dict] = {}
        if store.exists():
            for line in store.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("prompt_sha256") == system_sha and "predicted" in row:
                    done[str(row["id"])] = row
            if done:
                logger.info("Resuming: %d of %d records already answered in %s.",
                            len(done), len(records), store)
        todo = [r for r in records if r["id"] not in done]
        failures: list[dict] = []
        wall_total = 0.0
        with store.open("a") as sink:
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                users = [format_record(sections=r["sections"], sex=r["sex"], age_years=r["age"])
                         + instruction for r in batch]
                raws = self.backend.generate_many(system, users, response_schema=schema,
                                                  settings=settings)
                for record, user, raw in zip(batch, users, raws):
                    attempts = 1
                    parsed = self._parse(raw.content)
                    if parsed is None:
                        # A long record can spend the whole budget on reasoning
                        # and return nothing parsable; one retry at 3x before
                        # the record is a failure.
                        attempts = 2
                        raw = self.backend.generate(
                            system, user, response_schema=schema,
                            settings={**settings, "max_new_tokens": settings["max_new_tokens"] * 3},
                        )
                        parsed = self._parse(raw.content)
                    row = {
                        "id": record["id"], "gold": record["label"],
                        "attempts": attempts, "prompt_tokens": raw.prompt_tokens,
                        "gen_tokens": raw.gen_tokens, "wall_s": round(raw.wall_s, 2),
                        "done_reason": raw.done_reason, "prompt_sha256": system_sha,
                    }
                    if parsed is None:
                        row["error"] = "no parsable response"
                        failures.append(row)
                    else:
                        top1 = parsed["ranked_causes"][0]
                        row.update({
                            "predicted": top1,
                            "ranked_causes": parsed["ranked_causes"],
                            "n_ranked": len(parsed["ranked_causes"]),
                            "confidence": parsed["confidence"],
                            "cause_probabilities": parsed.get("cause_probabilities"),
                            "evidence": parsed.get("evidence"),
                            "rationale": parsed["rationale"],
                            "correct": (bool(top1 == record["label"])
                                        if record["label"] is not None else None),
                        })
                        done[record["id"]] = row
                        wall_total += raw.wall_s
                    sink.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    sink.flush()
                    position = len(done) + len(failures)
                    logger.info("[%d/%d] %5.1fs %s pred=%s gold=%s", position, len(records),
                                raw.wall_s, "FAILED" if parsed is None else "ok",
                                row.get("predicted", "-"), record["label"])
        if failures:
            ids = [f["id"] for f in failures]
            raise RuntimeError(
                f"{len(failures)} of {len(records)} record(s) returned no valid answer "
                f"after a retry: {ids[:10]}{' ...' if len(ids) > 10 else ''}. Nothing "
                "is filled in for them. Run again with resume=True to retry exactly "
                "these records; raise max_new_tokens if done_reason was 'length' "
                f"(see {store})."
            )
        answers = [done[r["id"]] for r in records]
        tokens = [a["prompt_tokens"] for a in answers if a.get("prompt_tokens")]
        walls = [a["wall_s"] for a in answers if a.get("wall_s") is not None]
        stats = {
            "mean_prompt_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
            "mean_wall_s": round(sum(walls) / len(walls), 2) if walls else None,
        }
        return answers, system_sha, stats

    def _parse(self, content: str) -> dict | None:
        parsed = extract_json(content)
        if parsed is None or not valid_response(parsed, self.spec.causes, sources=self.spec.sources):
            return None
        return normalise_response(parsed)

    def _export(self, answers, prob_source, top_k, out_dir: Path, with_ids: bool,
                labelled: bool = True) -> PredictionResult:
        causes = self.spec.causes
        if prob_source == "stated":
            proba = np.vstack([rank_to_probability.stated_vector(a, causes)[0] for a in answers])
        elif prob_source == "alpha":
            proba = rank_to_probability.build_matrix(answers, causes, rank_to_probability.PRIOR_WEIGHTS)
        else:
            proba = np.vstack([rank_to_probability.topk_vector(a, causes) for a in answers])
        result = assemble_predictions(
            proba, self.id2label,
            true_labels=[a["gold"] for a in answers] if labelled else None,
            ids=[a["id"] for a in answers] if with_ids else None, top_k=top_k,
        )
        save_predictions(result, out_dir)
        meta = pd.DataFrame([{
            "id": a["id"], **({"true_label": a["gold"]} if labelled else {}),
            "predicted_label": a["predicted"],
            "confidence": a["confidence"], "n_ranked": a["n_ranked"],
            "ranked_causes": json.dumps(a["ranked_causes"]),
            "evidence": json.dumps(a.get("evidence"), ensure_ascii=False),
            "rationale": a["rationale"], "attempts": a["attempts"],
            "prompt_tokens": a["prompt_tokens"], "gen_tokens": a["gen_tokens"],
            "wall_s": a["wall_s"], "done_reason": a["done_reason"],
        } for a in answers])
        meta.to_csv(out_dir / "llm_metadata.csv", index=False)
        return result

    @staticmethod
    def _scores(result: PredictionResult) -> dict[str, float]:
        return {m: float(score_predictions(result.top1, m)) for m in LLM_REPORT_METRICS}


def _jsonable_cell(cell: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cell.items()}


def _sha_json(doc) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()
