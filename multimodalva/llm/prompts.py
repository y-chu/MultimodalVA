"""Prompt construction for ``task="llm"``: a fixed base plus opt-in modules.

Three layers, deliberately separate, so one template serves every cause list,
site and age group:

  data/who_kb.json       WHO knowledge base, keyed by WHO VA cause code: verbatim
                         criteria, cross-cause rules. Never edited per site. It is
                         WHO material under CC BY-ND 4.0 - see data/WHO_NOTICE.md.
  a cause schema         the cause list, with each cause's definition and, where
                         known, its WHO code mapping and ICD-10 scope, plus the
                         population context (:mod:`multimodalva.llm.cause_schema`).
  this file              the base text, the modules, and the assembly.

The base text is held by a ``Task`` object (``COD``), and the assembly reads the
task rather than module-level constants. ``PromptSpec`` joins one cause schema
against the knowledge base and renders the system prompt for a chosen set of
modules; :func:`format_record` renders one record as the user message.

The join is deterministic: a schema names WHO codes, the knowledge base supplies
their text. Nothing in any layer is derived from outcome labels or class
frequencies.

Provenance. This file is the source of truth for the prompt. It was lifted on
2026-10-11 from the analysis pipeline it was developed in, where a frozen copy
remains; the rendered text for that pipeline's configuration is byte-identical
(``tests/test_llm_prompt_golden.py``). ``PROMPT_VERSION`` identifies the text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
KB_FILE = DATA_DIR / "who_kb.json"

CONFIDENCE_LEVELS = ["high", "medium", "low"]
#: The record sections a model may cite evidence from, when the caller names
#: none. Each section of the record is one source; the prompt text and the
#: response schema are both rendered from the list in use (``PromptSpec.sources``).
EVIDENCE_SOURCES = ["narrative", "questionnaire"]
#: Section name -> the header it is rendered under in the record. A section not
#: listed here is rendered under its own name, upper-cased.
LLM_SECTION_HEADERS = {
    "narrative": "NARRATIVE",
    "questionnaire": "STRUCTURED QUESTIONNAIRE RESPONSES",
}
EVIDENCE_POLARITY = ["present", "absent"]
EVIDENCE_STANCE = ["supports", "contradicts"]
MAX_EVIDENCE = 3
MAX_RANKED = 6

#: Identifies the text sent to the model. Bump it whenever any model-visible
#: string changes, so a stored run can be tied to the prompt that produced it and
#: an earlier version stays reconstructable.
#:
#:   v1  to 2026-10-05. Selection ran on this: 7 dev cells, drop-M4 selected
#:       (base + M1 + M2 + M3, macro F1 0.588). Its defect: three names for two
#:       WHO concepts ("immediate mode of dying" in the base, "immediate
#:       mechanism of death" in M2), and an M2 chain that skipped the immediate
#:       cause entirely.
#:   v2  2026-10-05. underlying cause / immediate cause / mode of dying defined
#:       once each and used consistently; M2's chain gains the immediate-cause
#:       rung. Selection must be re-run; v1 numbers do not carry over.
PROMPT_VERSION = "v2"

#: Assembly order, and the author's stated order. "live" modules can be switched;
#: held ones are declared here so the vocabulary is stable and a held module
#: cannot be silently confused with an absent one.
MODULES: dict[str, str] = {
    "M1": "study context",
    "M2": "reasoning scaffold",
    "M3": "cause scope (WHO VA categories and ICD-10 range)",
    "M4": "cross-cause rules (condensed WHO)",
}
#: Opt-in modules. Implemented and switchable, but deliberately NOT in `LIVE`, so
#: `--modules all` keeps meaning "every module the opt-out selection ranged over".
#: Few-shot is not part of that selection: it is added forward to a chosen
#: configuration, not eliminated backward out of a full one.
OPT_IN: dict[str, str] = {
    "M5": "few-shot exemplars",
    "M1b": "study context, population prior permitted (mutually exclusive with M1)",
}
HELD_MODULES: dict[str, str] = {
    "M6": "per-cause WHO criteria",
    "M7": "rules learnt from the training split",
    "M3b": "ICD-10 disease names",
}
LIVE = tuple(MODULES)
SELECTABLE = {**MODULES, **OPT_IN}

RULE_ARMS = ["none", "literature", "derived"]


# --------------------------------------------------------------------------
# BASE. Nothing here is removable. MODULES.md gives the reason for each part
# that used to be switchable and now is not.
# --------------------------------------------------------------------------

#: The hard constraint sits in the first sentence, following CODA's menu prompt
#: ("Use only the evidence provided and choose exactly one cause from the
#: numbered list"). Three deliberate departures from CODA are recorded in
#: MODULES.md: no physician persona, "most likely" rather than "exactly one"
#: because this arm must emit a distribution, and naming the data type.
ROLE = (
    "You are assigning the most likely underlying cause of death from "
    "verbal-autopsy data using general WHO physician-review guidance.\n"
    "Use only the evidence contained in the record provided. Do not invent, "
    "embellish or assume any finding that the record does not state."
)

#: This defines the task. A prompt without it asks a different question.
#:
#: Three WHO concepts, named once each and never mixed. Until 2026-10-05 this
#: block said "not the immediate mode of dying", which fused two of them: WHO's
#: *immediate cause* is the terminal disease or injury and is a legitimate thing
#: to identify, while a *mode of death* is a terminal physiological state that
#: WHO says must never be entered as a cause (Section 3, "modes of death such as
#: respiratory failure, heart failure, or brain death etc. should not be
#: considered as immediate causes of death"). M2 then called the same thing "the
#: immediate mechanism of death", a third name. Three names for two concepts is
#: ambiguity this task gains nothing from.
#:
#: WHO writes "mode of death". This says "mode of dying" so that it cannot be
#: misread as a kind of "cause of death"; the examples pin the meaning either way.
UNDERLYING_CAUSE = (
    "UNDERLYING CAUSE OF DEATH\n"
    "Three things are distinct. Only the first is what you report.\n"
    "  underlying cause - the disease or injury that started the chain of events\n"
    "                     leading to death. This is the answer.\n"
    "  immediate cause  - the terminal disease or injury at the end of that\n"
    "                     chain, such as the pneumonia a bedridden stroke patient\n"
    "                     died of. Reason through it to reach the underlying\n"
    "                     cause; it is not itself the answer.\n"
    "  mode of dying    - how the body finally failed: respiratory failure,\n"
    "                     cardiac arrest, heart failure, brain death. It names the\n"
    "                     terminal state without explaining what produced it, so\n"
    "                     it is never a cause. Never report one, and never treat\n"
    "                     one as the immediate cause."
)

OUTPUT_HEAD = "Return, in the required structure:"

OUTPUT_ITEMS = [
    f"""\
  1. ranked_causes: the candidate underlying causes, most likely first, at most
     {MAX_RANKED}, each with the probability you assign it for THIS death.
       cause       - one of the allowed causes. Never repeat a cause.
       probability - a number between 0 and 1.
     Include EVERY cause the record gives a genuine reason to consider, and
     stop there. The two failures cost the same: leaving out a cause you
     actually weighed, and adding one you did not. So do not invent support to
     fill the list, and do not withhold a real alternative in order to look
     decisive. The values are renormalised afterwards, so they need not sum to
     exactly 1, but do not spread them evenly to avoid committing.""",
    """\
  2. confidence: high, medium, or low, for the first-ranked cause.
       high   - clear direct evidence or a reported diagnosis supports it and
                the causal sequence is coherent.
       medium - substantial evidence supports it, but evidence is incomplete,
                indirect, or has a meaningful competing explanation.
       low    - it is the best-supported specific cause despite weak, sparse,
                or conflicting evidence.""",
    """\
  3. rationale: one or two sentences justifying the first-ranked cause.""",
    f"""\
  4. evidence: the findings you actually relied on, at most {MAX_EVIDENCE}. Give one if one
     decided it. Do NOT pad the list, and do not include a span that merely
     describes the timeline or the circumstances of death without bearing on
     which cause it was. Each item:
       quote    - ONE continuous span of the record, copied EXACTLY as it
                  appears, character for character. Do not correct spelling or
                  grammar, do not paraphrase, and do not join two separate
                  passages with "..." - if two passages matter, give them as two
                  items. The span must be findable in the record by exact search.
       source   - {{sources}}, naming which section it came from.
       polarity - present if the record states the finding DID occur; absent if
                  the record states it did NOT occur. "was not sick", "had no
                  fever" and "denied cough" are all absent, not present.
       supports - supports or contradicts the first-ranked cause. A finding
                  recorded as absent can still support a cause.""",
]

#: Asking for a probability per cause - all 19 - was 52% of the generated output
#: and almost all of it zeros: measured across three runs of 30, the median
#: record returned 15-16 exact zeros of 19, only 4 distinct values, and just
#: 5-7% of its stated mass outside the causes it had ranked. The values now ride
#: along with the ranking instead, and the tail is reconstructed at export.
#:
#: MAX_RANKED is 6 rather than 3 so the list can act as its own sentinel: the
#: median record ranks 3, so a list that stops short of 6 - or ends in a zero -
#: is the model stating that nothing further has support, rather than us assuming
#: it.
PROBABILITY_NOTE = (
    "Screening runs drop the probability from item 1 and keep the ranking, "
    "because selection uses macro F1, which needs only the first rank."
)

#: The screening form of item 1: the same ranking, without the numbers.
RANKED_WITHOUT_PROBABILITY = f"""\
  1. ranked_causes: the candidate underlying causes, most likely first, at most
     {MAX_RANKED}. Include every cause the record gives a genuine reason to consider
     and stop there: do not invent support to fill the list, and do not withhold
     a real alternative in order to look decisive. Never repeat a cause."""

#: Task rules. These define the output contract rather than steering diagnosis.
TASK_RULES = [
    "Use only the causes in the list below. Never return a cause outside it.",
    "Every ranked cause must be different.",
    "Rank by how probable each cause is for THIS death. The ranking below the\n"
    "  first position is used, so place genuine alternatives there rather than\n"
    "  padding the list.",
]

#: Diagnostic steers. Measured 2026-10-03 to *help* the residual category rather
#: than suppress it: removing them halves Indeterminate recall. Fixed in the base
#: for that reason; see MODULES.md.
DIAGNOSTIC_STEERS = [
    "Prefer a specific cause first whenever the record provides a reasonable basis\n"
    "  for doing so; use low confidence when that basis is weak.",
    "Assign the residual unknown category first only when the available information\n"
    "  is too sparse or non-specific to favor any specific cause. Do not use it merely\n"
    "  because the leading specific cause is uncertain or because two specific causes\n"
    "  compete.",
    "Judge only what the record states. Do not assume findings that are absent.",
]

#: Bidirectional. The previous version only forbade inferring narrative content
#: from the questionnaire and said nothing about the reverse, which is the
#: direction that contaminates a quoted-evidence output.
EVIDENCE_DISCIPLINE = (
    "Treat the narrative and the questionnaire responses as separate evidence.\n"
    "  Do not infer what one says from what the other says, in either direction,\n"
    "  and do not derive a second finding from a first: if the record reports\n"
    "  pneumonia but does not mention fever, you may not treat fever as present.\n"
    "  A finding that is not recorded is neither present nor absent."
)

CLOSING = "Return only the required structured output."


def _rules_block(task: "Task") -> str:
    rules = [*task.task_rules, *task.diagnostic_steers, task.evidence_discipline]
    return "RULES\n" + "\n".join(f"- {rule}" for rule in rules)


# --------------------------------------------------------------------------
# MODULES
# --------------------------------------------------------------------------

#: M2. Prescribes how to reason. Has a precedent in CODA's prompted agent
#: (Step 2, "Reason through the chain of events leading to the death"), and is a
#: module because the contrast with unprescribed reasoning is the question.
#: Revised 2026-10-05 with the terminology above. Two changes, not one. The names
#: are now the base's names - "the immediate mechanism of death" is gone. And the
#: chain gained its missing link: the old list went straight from the mechanism to
#: the underlying disease, skipping the immediate cause, so the step WHO actually
#: prescribes ("construct a chain of events that place the various conditions in
#: sequence") had no rung in the middle. The mode of dying is what the record
#: usually describes, the immediate cause is what explains it, and the underlying
#: cause is what started it.
REASONING_SCAFFOLD = """\
Read the record, then reason over:
  - the temporal sequence: onset, progression, and death;
  - the mode of dying: how the body finally failed. This is often what the record
    describes most vividly, and it is a clue, never the answer;
  - the immediate cause: the terminal disease or injury that produced that mode
    of dying;
  - the underlying cause: the disease or injury that began the sequence;
  - contributing conditions."""


def _context_block(text: str) -> str:
    """M1."""
    return (
        "STUDY CONTEXT\n"
        f"{text} Treat this only as broad population context: do not infer a "
        "cause for an individual from population prevalence, and do not let this "
        "context override the record."
    )


def _context_block_prior(text: str) -> str:
    """M1b. The same context without the prohibition on using it as a prior.

    M1's second clause forbids the model to reason from population prevalence to an
    individual death at all. That is stronger than intended: a prior is a
    legitimate input to an individual judgement, and what actually needs
    forbidding is letting it override the record - which the final clause already
    does. The prohibition may be part of why the arm under-assigns the residual
    category and the single most common cause: on the held-out split it assigned
    Indeterminate to 8.9% of deaths against a true 15.6%, and reached 0.724 recall
    on HIV/AIDS against 0.875 for feature fusion.
    
    No quantity is supplied. Giving the training cause distribution would make the
    CSMF estimate partly circular and would not transfer - between the nine
    provinces of this study the cause profile differs by a median total variation
    of 0.135, the same magnitude as this arm's entire distributional error. So M1b
    changes only what the model is permitted to do with what it already knows.
    """
    return (
        "STUDY CONTEXT\n"
        f"{text} This is the population these deaths come from, so it is legitimate "
        "background when judging which causes are plausible for an individual death. "
        "It must not override what the record states."
    )


# --------------------------------------------------------------------------
# TASK. The question being asked, as text. The assembly below orders blocks
# and switches modules; everything that makes the prompt a cause-of-death
# prompt is here, so another question can be asked of the same records with
# the same assembly, module switches and record formatting.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """The task-specific text of a prompt, and the two task-specific renderers.

    ``answer_space`` renders the block that states what may be answered - for
    cause of death, the cause list, with M3 scope and M6 criteria inside it.
    ``response_schema`` builds the structured-output schema. Both take the
    ``PromptSpec`` and may read its schema and knowledge base.

    A second task is a second instance. ``dataclasses.replace(COD, ...)`` keeps
    the parts it does not change, which is how a task that appends to this one
    stays a superset of it.
    """

    name: str
    role: str
    definition: str
    output_head: str
    output_items: list[str]
    ranked_without_probability: str
    task_rules: list[str]
    diagnostic_steers: list[str]
    evidence_discipline: str
    reasoning_scaffold: str
    closing: str
    answer_space: Callable[["PromptSpec", set[str]], str]
    response_schema: Callable[["PromptSpec", bool, bool], dict]


def _cod_answer_space(spec: "PromptSpec", on: set[str]) -> str:
    return spec.cause_list_block(scope="M3" in on, criteria="M6" in on)


def _cod_response_schema(
    spec: "PromptSpec", probabilities: bool = True, evidence: bool = True
) -> dict:
    # Probabilities are keyed by cause NAME, not by a letter or index. Named
    # keys matter: a letter-labelled variant of this task changed the model's
    # top choice on 3 of 6 pilot records, because the answer is then indexed
    # by token surface form rather than by cause identity.
    ranked_item: dict = {"type": "string", "enum": spec.causes}
    if probabilities:
        # One ordered list carries the ranking and the values, so a cause
        # name is emitted once rather than twice.
        ranked_item = {
            "type": "object",
            "properties": {
                "cause": {"type": "string", "enum": spec.causes},
                "probability": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["cause", "probability"],
            "additionalProperties": False,
        }
    properties = {
        "ranked_causes": {
            "type": "array",
            "items": ranked_item,
            "minItems": 1,
            "maxItems": MAX_RANKED,
            "uniqueItems": True,
        },
        "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
        "rationale": {"type": "string"},
    }
    required = ["ranked_causes", "confidence", "rationale"]
    if evidence:
        properties["evidence"] = {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_EVIDENCE,
            "items": {
                "type": "object",
                "properties": {
                    "quote": {"type": "string"},
                    "source": {"type": "string", "enum": list(spec.sources)},
                    "polarity": {"type": "string", "enum": EVIDENCE_POLARITY},
                    "supports": {"type": "string", "enum": EVIDENCE_STANCE},
                },
                "required": ["quote", "source", "polarity", "supports"],
                "additionalProperties": False,
            },
        }
        required.append("evidence")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


#: The cause-of-death task: the base text above, unchanged, as one object.
COD = Task(
    name="cod",
    role=ROLE,
    definition=UNDERLYING_CAUSE,
    output_head=OUTPUT_HEAD,
    output_items=OUTPUT_ITEMS,
    ranked_without_probability=RANKED_WITHOUT_PROBABILITY,
    task_rules=TASK_RULES,
    diagnostic_steers=DIAGNOSTIC_STEERS,
    evidence_discipline=EVIDENCE_DISCIPLINE,
    reasoning_scaffold=REASONING_SCAFFOLD,
    closing=CLOSING,
    answer_space=_cod_answer_space,
    response_schema=_cod_response_schema,
)


@dataclass
class PromptSpec:
    """One cause schema joined against the WHO knowledge base.

    Attributes:
        name: A label for the schema (its file stem, or ``"causes"`` when it was
            built from the data alone).
        causes: The cause list, sorted. These are the labels the model may return.
        schema: The cause schema document (``_meta`` + ``causes``).
        kb: The WHO knowledge base.
        sources: The record sections the model may cite evidence from, in record
            order. Defaults to :data:`EVIDENCE_SOURCES`.
        context: Study context for M1. ``None`` falls back to the schema's
            ``_meta.population_context``; an empty string turns M1 off.
        task: The question asked; :data:`COD` unless a caller defines another.
    """

    name: str
    causes: list[str]
    schema: dict
    kb: dict
    definitions: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=lambda: list(EVIDENCE_SOURCES))
    context: str | None = None
    task: Task = COD

    @property
    def population_context(self) -> str:
        if self.context is not None:
            return self.context
        return self.schema.get("_meta", {}).get("population_context", "")

    @property
    def age_group(self) -> str:
        """The age group this schema is for, from ``_meta.age_group``.

        Rule applicability depends on it: WHO writes some arbitration inside a
        childhood or perinatal section, and such a rule must not reach an adult
        prompt merely because the causes it names also exist in an adult schema.
        Empty when the schema does not say, in which case no rule is excluded on
        age.
        """
        return str(self.schema.get("_meta", {}).get("age_group", "") or "")

    def master_name(self, cause: str) -> str:
        """The WHO-derived master category a cause stands for, for M4.

        The cross-cause rules name the master categories of ``cause_master.json``.
        A schema whose labels differ can map each one with ``master_category``;
        without it the label itself is used.
        """
        entry = self.schema.get("causes", {}).get(cause, {}) or {}
        return str(entry.get("master_category") or cause)

    @property
    def unresolved(self) -> list[str]:
        """Schema fields still marked for author review."""
        pending = [
            f"{cause}.{key}"
            for cause, entry in self.schema["causes"].items()
            for key, value in entry.items()
            if isinstance(value, str) and "TODO_AUTHOR_REVIEW" in value
        ]
        if "TODO_AUTHOR_REVIEW" in self.population_context:
            pending.append("_meta.population_context")
        return pending

    # ---- cause list, with or without M3 / M6 -----------------------------

    def cause_list_block(self, scope: bool = False, criteria: bool = False) -> str:
        """The cause list. M3 adds scope, M6 adds WHO criteria.

        Without either it is the names alone, which is what the base carries.
        A schema with nothing to say about any cause renders the names alone
        too, rather than a "what each one covers" heading over nothing.
        """
        entries = self.schema["causes"]
        if scope and not criteria and not any(
            entries[c].get("scope_note") or entries[c].get("definition") or entries[c].get("icd10_scope")
            for c in self.causes
        ):
            scope = False
        if not scope and not criteria:
            return "ALLOWED CAUSES\n" + "; ".join(self.causes)

        header = (
            "ALLOWED CAUSES, WHAT EACH ONE COVERS"
            if not criteria
            else "ALLOWED CAUSES, WHAT EACH ONE COVERS, AND WHO DIAGNOSTIC GUIDANCE"
        )
        lines = [
            header,
            "The WHO VA categories and ICD-10 range state what this study's label",
            "aggregates. They define the label's extent; they are not diagnostic criteria.",
        ]
        if criteria:
            lines.append(
                "The WHO text is diagnostic guidance, not an algorithm or a gate."
            )
        for cause in self.causes:
            entry = self.schema["causes"][cause]
            lines.append(f"\n* {cause}")
            if scope:
                # A caller's own `definition` renders exactly as a generated
                # schema's `scope_note`: both state what the label covers.
                note = entry.get("scope_note") or entry.get("definition")
                if note:
                    lines.append(f"  Covers: {note}")
                if entry.get("icd10_scope"):
                    lines.append(f"  ICD-10: {entry['icd10_scope']}")
            if criteria:
                for code in entry.get("analysis_va_codes", []):
                    text = self.kb["criteria"].get(code)
                    if text and text.strip() != cause:
                        lines.append(f"  WHO {code}: {text}")
                    # The footnote bodies carry the inclusions and exclusions
                    # that bound the category - "Excludes: Neonatal tetanus",
                    # "Includes: Ischaemic heart disease; Pulmonary embolism" -
                    # so emitting the entry without them would state the
                    # criteria while dropping their scope.
                    for note in self.kb.get("criteria_footnotes", {}).get(code, []):
                        lines.append(f"    {note['who_quote']}")
        return "\n".join(lines)

    # ---- M4 --------------------------------------------------------------

    def cross_cause_rules(self, arm: str = "literature", verbatim: bool = False) -> str:
        """M4. Condensed WHO arbitration rules, or the verbatim quotes.

        The prompt ships the summaries; `verbatim` exists so the condensation can
        be audited against the source, and no run has used it.
        """
        if arm not in RULE_ARMS:
            raise ValueError(f"unknown rules arm {arm!r}; expected one of {RULE_ARMS}")
        if arm == "none":
            return ""
        source = (
            self.kb["cross_cause_rules"] if arm == "literature"
            else self.kb.get("derived_rules", [])
        )
        # Two independent gates. A rule is shown only when every cause it
        # arbitrates is in this schema, so a rule written for one site never
        # names a cause another site does not model; and only when its
        # `age_scope` covers this schema's age group, because WHO writes some
        # arbitration inside a childhood or perinatal section. The malnutrition
        # and anaemia rule is the case that forced the second gate: it comes from
        # 4.1.3.1 "Childhood infectious diseases", and the causes it names all
        # exist in the adult schema too, so the label gate alone let a childhood
        # rule into an adult prompt.
        # Rules name master categories; a schema may map its own labels onto
        # them with `master_category`, and the heading then shows the label.
        label_of = {self.master_name(c): c for c in self.causes}
        present = set(label_of)
        age = self.age_group
        rules = [
            r for r in source
            if (r.get("applies_to") == ["ALL"]
                or present.issuperset(set(r.get("applies_to", []))))
            and (not age or not r.get("age_scope") or age in r["age_scope"])
        ]
        if not rules:
            return ""
        header = (
            (
                "CROSS-CAUSE RULES (quoted from WHO physician-review guidance)"
                if verbatim
                else "CROSS-CAUSE RULES (condensed from WHO physician-review guidance)"
            )
            if arm == "literature" else "CROSS-CAUSE RULES"
        )
        lines = [header]
        for rule in rules:
            scope = ", ".join(
                label_of.get(name, name) for name in rule.get("applies_to", [])
            )
            lines.append(f"\n* {scope}")
            text = (
                rule.get("who_quote")
                if verbatim or not rule.get("summary")
                else rule["summary"]
            ) or rule.get("rule_text", "")
            lines.append(f"  {text}")
        return "\n".join(lines)

    # ---- M5 --------------------------------------------------------------

    def few_shot_block(self) -> str:
        """M5. Worked examples. Declared, not shipped.

        The exemplars are verbatim narratives from a study's own records, so no
        exemplar file travels with the package, and a prompt using M5 could not
        be printed in full. Asking for M5 raises rather than silently rendering
        an empty block.
        """
        raise ValueError(
            "module M5 (few-shot exemplars) is not available in this version: it "
            "needs an exemplar file of records from your own data, which the "
            "package does not ship. Use M1-M4."
        )

    # ---- assembly --------------------------------------------------------

    def system_prompt(
        self,
        modules: tuple[str, ...] | list[str] = LIVE,
        probabilities: bool = True,
        evidence: bool = True,
        rules_arm: str = "literature",
        verbatim_rules: bool = False,
    ) -> str:
        """Assemble the base plus the requested modules, in MODULES.md order."""
        unknown = [m for m in modules if m not in SELECTABLE]
        if unknown:
            held = [m for m in unknown if m in HELD_MODULES]
            raise ValueError(
                f"unknown module(s) {unknown}; selectable are {list(SELECTABLE)}"
                + (f". {held} are declared but held, not implemented." if held else "")
            )
        on = set(modules)

        if {"M1", "M1b"} <= on:
            raise ValueError(
                "M1 and M1b are two wordings of the same block; pass one, not both"
            )

        task_text = self.task
        parts = [task_text.role]
        if "M1" in on and self.population_context:
            parts.append(_context_block(self.population_context))
        elif "M1b" in on and self.population_context:
            parts.append(_context_block_prior(self.population_context))

        parts.append(task_text.definition)

        task = ["TASK"]
        if "M2" in on:
            task.append(task_text.reasoning_scaffold)
            task.append("")
        task.append(task_text.output_head)
        items = list(task_text.output_items) if evidence else task_text.output_items[:3]
        if not probabilities:
            # Keep the ranking, drop the number that rides with it.
            items[0] = task_text.ranked_without_probability
        # The evidence item names the sections a quote may come from, which are
        # whatever sections the records have.
        items = [item.replace("{sources}", source_phrase(self.sources)) for item in items]
        task.extend(items)
        parts.append("\n".join(task))

        parts.append(_rules_block(task_text))

        if "M4" in on:
            block = self.cross_cause_rules(rules_arm, verbatim_rules)
            if block:
                parts.append(block)

        if "M5" in on:
            parts.append(self.few_shot_block())

        parts.append(task_text.answer_space(self, on))
        parts.append(task_text.closing)
        return "\n\n".join(parts)

    # ---- response schema -------------------------------------------------

    def response_schema(
        self, probabilities: bool = True, evidence: bool = True
    ) -> dict:
        """The structured-output schema for this task."""
        return self.task.response_schema(self, probabilities, evidence)


def source_phrase(sources: list[str] | tuple[str, ...]) -> str:
    """``"narrative, or questionnaire"`` for the evidence item, from the sources."""
    names = [str(x) for x in sources]
    if not names:
        raise ValueError("a record needs at least one section")
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, or {names[1]}"
    return ", ".join(names[:-1]) + f", or {names[-1]}"


def section_header(name: str) -> str:
    """The header a section is rendered under: the known ones by name, the rest upper-cased."""
    return LLM_SECTION_HEADERS.get(name, str(name).replace("_", " ").upper())


def load_knowledge_base() -> dict:
    """The bundled WHO knowledge base, parsed."""
    return json.loads(KB_FILE.read_text())


def format_record(
    narrative: str | None = None,
    structured_responses: str | None = None,
    sex: str | None = None,
    age_years: float | None = None,
    demographics: bool = True,
    *,
    sections: "dict[str, str | None] | None" = None,
) -> str:
    """Render one record as the user message.

    Two forms. The positional one takes a narrative and a questionnaire
    rendering, which are the ``narrative`` and ``questionnaire`` sections. The
    keyword form takes ``sections``, an ordered mapping of section name to text,
    for records with any number of sources; a section whose text is empty or
    missing is left out, with no empty header.

    Args:
        narrative: The free text.
        structured_responses: The questionnaire rendered as text.
        sex: Rendered on the demographics line when given.
        age_years: Age in years, rendered on the demographics line when given.
        demographics: Whether to render the demographics line at all.
        sections: ``{section name: text}`` in record order. Mutually exclusive
            with the positional sections.

    Raises:
        ValueError: No section carries any text.
    """
    if sections is not None and (narrative is not None or structured_responses is not None):
        raise ValueError("pass either sections= or narrative/structured_responses, not both")
    parts = ["VERBAL-AUTOPSY RECORD"]
    if demographics and (sex or age_years is not None):
        bits = []
        if age_years is not None and age_years == age_years:
            bits.append(f"{int(round(float(age_years)))}-year-old")
        if sex:
            bits.append(str(sex))
        parts.append("DEMOGRAPHICS: " + " ".join(bits))
    if sections is None:
        sections = {"narrative": narrative, "questionnaire": structured_responses}
        # Both halves are optional, because a tabular-only record has no
        # narrative. The two headers are also what lets a model name which half
        # a quoted span came from, so they are never merged into one section.
        for name, text in sections.items():
            if text is not None:
                parts.extend([section_header(name), str(text).strip()])
    else:
        for name, text in sections.items():
            if text is None or (isinstance(text, float) and text != text):
                continue
            body = str(text).strip()
            if not body:
                continue
            parts.extend([section_header(name), body])
    if len(parts) == 1 or (len(parts) == 2 and parts[1].startswith("DEMOGRAPHICS: ")):
        raise ValueError("format_record was given no section with any text")
    return "\n".join(parts)


def check_knowledge_base(spec: PromptSpec) -> list[str]:
    """Integrity violations in who_kb.json. Empty list means clean.

    These are the invariants that were each violated at some point and fixed on
    2026-10-05, so the check exists to stop them regressing silently. It is a
    pure function returning problems, not a printer, so it can be asserted on.
    """
    kb = spec.kb
    problems: list[str] = []

    # Every provenance claim must resolve to a field that exists. The metadata
    # named who_guidance[*].guidance and _shared_guidance[*].guidance, neither of
    # which was ever a key in this file.
    boundary = kb["_meta"]["who_pcva_guidance"].get("quotation_boundary", "")
    for claimed in re.findall(r"\b([a-z_]+)\[", boundary):
        if claimed not in kb:
            problems.append(f"quotation_boundary names absent top-level key {claimed!r}")

    # Every guidance entry needs page provenance, even if that provenance says
    # the page could not be verified.
    for code in kb["criteria"]:
        if code not in kb.get("criteria_source", {}):
            problems.append(f"criteria[{code}] has no criteria_source entry")

    # Rule hygiene.
    ids = [r.get("rule_id") for r in kb["cross_cause_rules"]]
    for rid in {i for i in ids if ids.count(i) > 1}:
        problems.append(f"duplicate rule_id {rid!r}")
    for rule in kb["cross_cause_rules"]:
        if not rule.get("who_quote"):
            problems.append(f"rule {rule.get('rule_id')!r} has no who_quote")
        if not rule.get("age_scope"):
            problems.append(f"rule {rule.get('rule_id')!r} has no age_scope")

    # A footnote callout must not survive in prompt-facing text: stripped of its
    # body it reads as part of the sentence ("positive test for COVID-19.33").
    for code, text in kb["criteria"].items():
        for match in re.finditer(r"[a-z0-9)’]\.(\d{2})\b", text):
            problems.append(
                f"criteria[{code}] still carries footnote callout {match.group(1)!r}")

    # Every recovered footnote must belong to an entry that exists.
    for code in kb.get("criteria_footnotes", {}):
        if code not in kb["criteria"]:
            problems.append(f"criteria_footnotes[{code}] has no criteria entry")

    # Every verbatim field must be findable in the source, which is the whole
    # point of keeping who_source/*_raw.txt. End-of-line hyphenation is rejoined
    # with its hyphen rather than flattened to a space: the PDF writes
    # "non-communicable" and breaks it after the hyphen, so collapsing that break
    # to a space invents one. criteria entries are exempt where a footnote callout
    # or an extracted page footer was removed - those removals are declared in
    # _meta.normalisation_classes and listed here so the exemption is explicit.
    FOOTNOTE_EXEMPT = {"VAs-01.08", "VAs-01.11", "VAs-01.13", "VAs-04.01",
                       "VAs-04.02", "VAs-06.02", "VAs-10.02", "VAs-10.03"}
    source_dir = Path(__file__).resolve().parent / "who_source"
    raw_files = sorted(source_dir.glob("*_raw.txt")) if source_dir.exists() else []
    if raw_files:
        def _norm(text: str) -> str:
            text = re.sub(r"-\s*\n\s*", "-", text)
            return re.sub(r"\s+", " ", text).strip()

        corpus = _norm(" ".join(f.read_text() for f in raw_files))
        verbatim: list[tuple[str, str]] = []
        for rule in kb["cross_cause_rules"]:
            for quote_field in ("who_quote", "who_quote_2", "who_quote_3", "who_quote_4"):
                if rule.get(quote_field):
                    verbatim.append((f"{rule['rule_id']}.{quote_field}", rule[quote_field]))
        for code, notes in kb.get("criteria_footnotes", {}).items():
            for note in notes:
                verbatim.append((f"criteria_footnotes[{code}]#{note['callout']}",
                                 note["who_quote"]))
        for group, note in kb.get("group_notes", {}).items():
            verbatim.append((f"group_notes[{group}]", note["who_quote"]))
        for code, text in kb["criteria"].items():
            if code not in FOOTNOTE_EXEMPT:
                verbatim.append((f"criteria[{code}]", text))
        for where, text in verbatim:
            if _norm(text) not in corpus:
                problems.append(f"{where} is declared verbatim but is not in who_source/")

    # The extraction artifacts, in any verbatim-or-prompt-facing field.
    texts = {f"criteria[{c}]": t for c, t in kb["criteria"].items()}
    texts.update({f"{r['rule_id']}.who_quote": r.get("who_quote", "")
                  for r in kb["cross_cause_rules"]})
    for where, text in texts.items():
        for artifact in ("ch ildren", "s upporting", "non- communicable",
                         "( information", "symptom- based"):
            if artifact in text:
                problems.append(f"{where} still carries extraction artifact {artifact!r}")

    # age_scope is study-authored, so a wrong scope is self-consistent and the
    # gate below cannot catch it. The source citation can: WHO puts the rule in a
    # named section, and a rule cited to a childhood or perinatal section must not
    # claim adult scope. This is exactly how the malnutrition rule was wrong.
    for rule in kb["cross_cause_rules"]:
        source = (rule.get("source") or "").lower()
        if ("childhood" in source or "perinatal" in source) and \
                "adult" in (rule.get("age_scope") or []):
            problems.append(
                f"rule {rule.get('rule_id')!r} is cited to a childhood/perinatal "
                f"section but claims adult age_scope")

    # A rule drawn from a childhood or perinatal section must not be emitted into
    # an adult prompt. This is the one that was actually wrong in a prompt.
    if spec.age_group:
        for rule in kb["cross_cause_rules"]:
            scope = rule.get("age_scope") or []
            if scope and spec.age_group not in scope:
                block = spec.cross_cause_rules("literature")
                if rule.get("summary", "\0") in block:
                    problems.append(
                        f"rule {rule['rule_id']!r} is {scope} but reaches the "
                        f"{spec.age_group} prompt")
    return problems


def check_terminology(spec: PromptSpec, modules=LIVE) -> list[str]:
    """Terminology violations in the assembled prompt. Empty list means clean.

    WHO's three concepts must each have exactly one name in the text the model
    sees. The failure this guards against is the one that was there until
    2026-10-05: "immediate mode of dying" in the base and "immediate mechanism of
    death" in M2, i.e. three names covering two concepts, one of which WHO
    explicitly says is not a cause.
    """
    # Only study-authored text is bound by this convention. A verbatim WHO quote
    # is not: WHO's own Section 3 says "terminal event that occurred", and
    # rewriting a quotation to fit our vocabulary would make it not a quotation.
    # So this checks the assembled base and the study-authored module text, plus
    # every rule `summary` (authored, prompt-facing), and never a `who_quote`.
    text = spec.system_prompt(modules=[m for m in modules if m not in ("M4", "M6")])
    authored = [text] + [
        rule.get("summary") or ""
        for rule in spec.kb.get("cross_cause_rules", [])
    ]
    text = "\n".join(authored)
    problems: list[str] = []

    # Banned spellings: each one is a second name for something already named.
    for banned, why in [
        ("mechanism", "a third name for the mode of dying"),
        ("mode of death", "WHO's spelling; this prompt says 'mode of dying' throughout"),
        ("immediate mode", "fuses 'immediate cause' with 'mode of dying'"),
        ("terminal event", "a fourth name; say 'immediate cause'"),
    ]:
        if banned in text.lower():
            problems.append(f"prompt says {banned!r} - {why}")

    # The three concepts must be defined, and the mode of dying must be excluded
    # as an answer wherever it is mentioned at all.
    if "underlying cause" not in text.lower():
        problems.append("prompt never names the underlying cause")
    if "mode of dying" in text.lower() and "never a cause" not in text.lower():
        problems.append("prompt names the mode of dying without excluding it as a cause")
    if "immediate cause" in text.lower() and "not itself the answer" not in text.lower():
        problems.append("prompt names the immediate cause without saying it is not the answer")
    return problems


def parse_modules(values: list[str] | None) -> tuple[str, ...]:
    """Turn a --modules argument into a module tuple.

    'all' is every live module, 'none' is the base alone, otherwise the names
    given. Accepts lower case.
    """
    if not values:
        return LIVE
    cleaned = [v.strip() for v in values if v.strip()]
    lowered = [v.lower() for v in cleaned]
    if lowered == ["all"]:
        return LIVE
    if lowered == ["none"]:
        return ()
    canonical = {m.lower(): m for m in {**SELECTABLE, **HELD_MODULES}}
    return tuple(canonical.get(v.lower(), v) for v in cleaned)
