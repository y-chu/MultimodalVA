"""Cause schemas for ``task="llm"``: what each label covers, and the study context.

The cause list itself is read from the data's label column, as for every other
task. A schema is optional and adds what the names alone do not carry:

.. code-block:: json

    {
      "_meta": {"population_context": "...", "age_group": "adult"},
      "causes": {
        "Cardiac disease": {
          "definition": "What this label covers, in your own words.",
          "analysis_va_codes": ["VAs-04.01", "VAs-04.99"],
          "icd10_scope": "I00-I52",
          "master_category": "Cardiac disease"
        }
      }
    }

Per module, what the schema supplies:

=====  ======================================================================
M1     ``_meta.population_context`` (or ``context=`` on the run, which wins)
M2     nothing — always available
M3     per cause, ``definition`` (your words) or ``scope_note`` (generated),
       and ``icd10_scope``; a cause with neither is listed by name only
M4     the WHO cross-cause rules name the categories of ``cause_master.json``;
       a rule applies when every category it names is one of your causes, by
       name or through ``master_category``
=====  ======================================================================

A bundled example, ``who_target_list_adult.json``, holds the Level-2
categories of the WHO target list with their codes and scope.
:func:`cause_schema_template` writes a template to fill in for your own causes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .prompts import (
    COD, DATA_DIR, EVIDENCE_SOURCES, LIVE, PromptSpec, Task, load_knowledge_base,
    parse_modules,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LLM_SCHEMA_EXAMPLE",
    "load_cause_schema",
    "cause_schema_template",
    "describe_modules",
    "build_system_prompt",
]

#: The bundled example schema: the WHO target list, adult, site-neutral.
LLM_SCHEMA_EXAMPLE = DATA_DIR / "who_target_list_adult.json"
LLM_SCHEMA_MASTER = DATA_DIR / "cause_master.json"

#: Keys a cause entry may carry. Anything else is a typo, not an extension.
LLM_SCHEMA_ENTRY_KEYS = frozenset({
    "definition", "scope_note", "icd10_scope", "analysis_va_codes",
    "master_category", "who_category_names",
})
#: A string field carrying this is unfinished; a template is written with it.
LLM_SCHEMA_TODO = "TODO"

_PROMPT_HINT = (
    "Run `multimodalva llm-prompt` with the same cause schema and modules to "
    "print the system prompt the model will read."
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_schema_doc(cause_schema) -> tuple[dict, str]:
    """``(document, name)`` from a path, a dict, or the bundled example's name."""
    if isinstance(cause_schema, dict):
        return cause_schema, str(cause_schema.get("_meta", {}).get("name") or "cause_schema")
    path = Path(str(cause_schema)).expanduser()
    if not path.exists() and path.suffix == "" and (DATA_DIR / f"{path.name}.json").is_file():
        path = DATA_DIR / f"{path.name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"cause_schema {str(cause_schema)!r} is not a file. Pass a path to a "
            f"schema JSON, a dict, or the bundled example {LLM_SCHEMA_EXAMPLE.stem!r}."
        )
    try:
        doc = json.loads(path.read_text())
    except ValueError as exc:
        raise ValueError(f"cause_schema {path} is not valid JSON: {exc}") from exc
    return doc, path.stem


def _check_schema_doc(doc: dict, causes: list[str], master: dict) -> tuple[list[str], list[str]]:
    """``(problems, notes)`` for a schema document against the data's causes."""
    problems: list[str] = []
    notes: list[str] = []
    if not isinstance(doc, dict) or not isinstance(doc.get("causes"), dict):
        return ['cause_schema must be a JSON object with a "causes" mapping'], notes
    extra_top = [k for k in doc if k not in ("_meta", "causes") and not str(k).startswith("_")]
    if extra_top:
        problems.append(
            f"cause_schema has unexpected top-level key(s) {extra_top}; only "
            '"_meta" and "causes" are read (keys starting with "_" are ignored).'
        )
    entries = doc["causes"]
    todo: list[str] = []
    meta = doc.get("_meta") or {}
    if isinstance(meta.get("population_context"), str) and LLM_SCHEMA_TODO in meta["population_context"]:
        todo.append("_meta.population_context")
    for cause, entry in entries.items():
        if not isinstance(entry, dict):
            problems.append(f"cause_schema: the entry for {cause!r} must be an object")
            continue
        unknown = sorted(set(entry) - LLM_SCHEMA_ENTRY_KEYS)
        if unknown:
            problems.append(
                f"cause_schema: {cause!r} has unknown key(s) {unknown}; allowed: "
                f"{sorted(LLM_SCHEMA_ENTRY_KEYS)}"
            )
        for key, value in entry.items():
            if isinstance(value, str) and LLM_SCHEMA_TODO in value:
                todo.append(f"{cause}.{key}")
        codes = entry.get("analysis_va_codes")
        if codes is not None and not (isinstance(codes, list) and all(isinstance(c, str) for c in codes)):
            problems.append(f"cause_schema: {cause!r}.analysis_va_codes must be a list of strings")
        alias = entry.get("master_category")
        if alias is not None and alias not in master:
            problems.append(
                f"cause_schema: {cause!r}.master_category={alias!r} is not a category "
                f"of cause_master.json ({len(master)} categories; see the bundled "
                f"{LLM_SCHEMA_EXAMPLE.stem}.json for the names)"
            )
    if todo:
        problems.append(
            f"cause_schema still has {len(todo)} field(s) marked {LLM_SCHEMA_TODO}: "
            f"{todo[:8]}{' ...' if len(todo) > 8 else ''}. Fill them in, or remove "
            "the field to leave that cause with its name only."
        )
    unmatched = sorted(c for c in causes if c not in entries)
    if unmatched:
        problems.append(
            f"{len(unmatched)} cause(s) in the data are not in the cause schema: "
            f"{unmatched[:10]}{' ...' if len(unmatched) > 10 else ''}. Every label "
            "the model may return must have an entry, even an empty one."
        )
    absent = sorted(c for c in entries if c not in causes)
    if absent:
        notes.append(
            f"{len(absent)} cause(s) in the schema do not occur in the data and are "
            f"left out of the prompt: {absent[:10]}{' ...' if len(absent) > 10 else ''}"
        )
    return problems, notes


def load_cause_schema(
    cause_schema,
    causes: "list[str] | None" = None,
    *,
    context: str | None = None,
    sources: "list[str] | None" = None,
    task: Task = COD,
) -> PromptSpec:
    """Join a cause schema (or none) against the WHO knowledge base.

    Args:
        cause_schema: A path to a schema JSON, a dict of the same shape, the
            name of the bundled example (``"who_target_list_adult"``), or
            ``None`` for a schema built from the cause names alone.
        causes: The labels found in the data. These are the causes the model may
            return; each must have an entry when a schema is given. ``None``
            (unlabelled data) takes every cause the schema lists.
        context: Study context for M1. Overrides the schema's
            ``_meta.population_context``; ``""`` turns M1 off.
        sources: Record section names the model may cite evidence from.
            Defaults to narrative + questionnaire.
        task: The question asked; :data:`multimodalva.llm.prompts.COD` unless
            a caller defines another.

    Returns:
        A :class:`~multimodalva.llm.prompts.PromptSpec` restricted to ``causes``.

    Raises:
        ValueError: A label in the data has no schema entry, a field is still
            marked TODO, an entry has an unknown key, or ``master_category``
            names no known category.
        FileNotFoundError: ``cause_schema`` is a path that does not exist.
    """
    kb = load_knowledge_base()
    master = json.loads(LLM_SCHEMA_MASTER.read_text())["categories"]
    if cause_schema is None:
        if causes is None:
            raise ValueError("load_cause_schema: without a schema the cause list is required")
        causes = sorted({str(c) for c in causes})
        if not causes:
            raise ValueError("load_cause_schema: the cause list is empty")
        doc = {"_meta": {}, "causes": {c: {} for c in causes}}
        name = "causes"
    else:
        doc, name = _read_schema_doc(cause_schema)
        if causes is None:
            if not isinstance(doc, dict) or not isinstance(doc.get("causes"), dict) or not doc["causes"]:
                raise ValueError('cause_schema must be a JSON object with a non-empty "causes" mapping')
            causes = sorted(str(c) for c in doc["causes"])
        causes = sorted({str(c) for c in causes})
        if not causes:
            raise ValueError("load_cause_schema: the cause list is empty")
        problems, notes = _check_schema_doc(doc, causes, master)
        if problems:
            raise ValueError("cause_schema is not usable:\n- " + "\n- ".join(problems))
        for note in notes:
            logger.info(note)
        # The prompt lists the data's causes only; the schema may hold more.
        doc = {
            **{k: v for k, v in doc.items() if k != "causes"},
            "_meta": dict(doc.get("_meta") or {}),
            "causes": {c: dict(doc["causes"][c]) for c in causes},
        }
    return PromptSpec(
        name=name, causes=causes, schema=doc, kb=kb,
        sources=list(sources) if sources else list(EVIDENCE_SOURCES),
        context=context, task=task,
    )


# ---------------------------------------------------------------------------
# What the modules can do with this schema
# ---------------------------------------------------------------------------

def describe_modules(spec: PromptSpec, modules=LIVE, rules_arm: str = "literature") -> list[str]:
    """Warnings about modules that cannot do their full job with this schema.

    Nothing here changes the prompt: :meth:`PromptSpec.system_prompt` already
    leaves out an M1 with no context and an M4 with no applicable rule. This
    says so, before any model is loaded, so the gap is a message rather than a
    silent absence.
    """
    on = set(parse_modules(list(modules)) if not isinstance(modules, tuple) else modules)
    warnings: list[str] = []
    if "M1" in on and not spec.population_context:
        warnings.append(
            "M1 (study context) is off: no context= was given and the cause schema "
            "has no _meta.population_context. Pass context=\"...\" with one or two "
            "sentences on the population these deaths come from."
        )
    if "M3" in on:
        bare = [
            c for c in spec.causes
            if not (spec.schema["causes"][c].get("scope_note")
                    or spec.schema["causes"][c].get("definition")
                    or spec.schema["causes"][c].get("icd10_scope"))
        ]
        if len(bare) == len(spec.causes):
            warnings.append(
                "M3 (cause scope) has nothing to render: no cause has a definition, "
                "so the model sees the cause names and nothing else about what each "
                "one covers. Give each cause a definition in a cause schema "
                "(multimodalva llm-schema writes a template). " + _PROMPT_HINT
            )
        elif bare:
            warnings.append(
                f"M3 (cause scope): {len(bare)} cause(s) have no definition and are "
                f"listed by name only: {bare[:10]}{' ...' if len(bare) > 10 else ''}."
            )
    if "M4" in on:
        block = spec.cross_cause_rules(rules_arm)
        total = len(spec.kb.get("cross_cause_rules", []))
        if not block:
            warnings.append(
                f"M4 (cross-cause rules) is off: none of the {total} WHO rules names "
                "only causes in your list. The rules are written for the WHO-derived "
                f"categories of {LLM_SCHEMA_EXAMPLE.stem}.json; a schema can map your "
                "labels onto them with master_category. " + _PROMPT_HINT
            )
        else:
            used = block.count("\n* ")
            if used < total:
                warnings.append(
                    f"M4 (cross-cause rules): {used} of {total} WHO rules apply to "
                    "this cause list; the rest name a cause that is not in it."
                )
    return warnings


# ---------------------------------------------------------------------------
# Template and rendering helpers
# ---------------------------------------------------------------------------

def cause_schema_template(causes: "list[str]", *, context: str | None = None) -> dict:
    """A schema document to fill in: one entry per cause, definitions marked TODO.

    The WHO target-list categories are listed under ``_who_target_list`` for
    copying codes and scope from; the loader ignores keys starting with ``_``.
    """
    example = json.loads(LLM_SCHEMA_EXAMPLE.read_text())["causes"]
    entries = {}
    for cause in sorted({str(c) for c in causes}):
        known = example.get(cause)
        entries[cause] = (
            {"definition": f"{LLM_SCHEMA_TODO}: what this label covers",
             "analysis_va_codes": [], "icd10_scope": ""}
            if known is None else
            {"definition": known["scope_note"],
             "analysis_va_codes": list(known["analysis_va_codes"]),
             "icd10_scope": known["icd10_scope"],
             "master_category": cause}
        )
    return {
        "_meta": {
            "population_context": context if context is not None else (
                f"{LLM_SCHEMA_TODO}: one or two sentences on the population these "
                "deaths come from, e.g. the setting and its main burdens"
            ),
            "age_group": "adult",
        },
        "causes": entries,
        "_help": {
            "definition": "What this label covers, in your words. Rendered under M3.",
            "analysis_va_codes": "WHO VA cause codes, optional (see _who_target_list).",
            "icd10_scope": "ICD-10 range(s) the label aggregates, optional.",
            "master_category": (
                "The WHO-derived category this label stands for, when its name "
                "differs; lets the M4 cross-cause rules apply."
            ),
            "age_group": "adult | child | neonate; limits which WHO rules apply.",
        },
        "_who_target_list": {
            name: {k: entry[k] for k in ("analysis_va_codes", "icd10_scope", "scope_note")}
            for name, entry in example.items()
        },
    }


def build_system_prompt(
    causes: "list[str]",
    *,
    cause_schema=None,
    modules=LIVE,
    context: str | None = None,
    sources: "list[str] | None" = None,
    rules_arm: str = "literature",
    probabilities: bool = True,
    evidence: bool = True,
) -> str:
    """The system prompt a run would send, for reading before spending GPU time."""
    spec = load_cause_schema(cause_schema, causes, context=context, sources=sources)
    return spec.system_prompt(
        modules=parse_modules(list(modules)) if not isinstance(modules, tuple) else modules,
        probabilities=probabilities, evidence=evidence, rules_arm=rules_arm,
    )
