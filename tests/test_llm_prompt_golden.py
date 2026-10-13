"""The package prompt is the frozen analysis prompt, byte for byte.

``multimodalva/llm/prompts.py`` was lifted from an analysis pipeline whose own
copy is frozen (other work pins its hash). These tests compare the two: every
module combination, both output forms, both rule arms, the response schema and
the record format. They need the analysis tree, so they skip elsewhere; a
stored digest of the default prompt runs everywhere.

The WHO excerpts in ``who_kb.json`` are checked against the raw extraction of
the source sections the same way: every quoted field must be a substring of the
source after whitespace normalisation, which is what "unmodified except for
format" means in ``WHO_NOTICE.md``.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import os
import re
import sys
from pathlib import Path

import pytest

from multimodalva.llm import prompts

PACKAGE_ROOT = Path(prompts.__file__).resolve().parents[2]
ANALYSIS_LLM = Path(
    os.environ.get("MULTIMODALVA_ANALYSIS_LLM_DIR")
    or PACKAGE_ROOT.parent / "Analysis" / "utils" / "llm"
)
SCHEMA = ANALYSIS_LLM.parent.parent / "RSA" / "llm" / "cause_schemas" / "sancodv_group2_adult.json"

#: sha256 of the default prompt (M1-M4, probabilities, evidence, literature
#: rules) rendered for the analysis schema on 2026-10-11. Regenerate only with
#: a deliberate PROMPT_VERSION bump.
DEFAULT_PROMPT_SHA256 = "2b1977cedeef8e07"

needs_analysis = pytest.mark.skipif(
    not (ANALYSIS_LLM / "prompts.py").is_file() or not SCHEMA.is_file(),
    reason="the analysis tree with the frozen prompt is not on this machine",
)


@pytest.fixture(scope="module")
def frozen():
    sys.path.insert(0, str(ANALYSIS_LLM))
    try:
        module = importlib.import_module("prompts")
    finally:
        sys.path.remove(str(ANALYSIS_LLM))
    if module is prompts:
        pytest.skip("the analysis prompts module resolved to the package copy")
    return module


@pytest.fixture(scope="module")
def specs(frozen):
    doc = json.loads(SCHEMA.read_text())
    kb = json.loads((ANALYSIS_LLM / "who_kb.json").read_text())
    old = frozen.load(str(SCHEMA))
    new = prompts.PromptSpec(name=SCHEMA.stem, causes=sorted(doc["causes"]), schema=doc, kb=kb)
    return old, new


@needs_analysis
def test_knowledge_base_is_the_frozen_one():
    ours = (PACKAGE_ROOT / "multimodalva" / "llm" / "data" / "who_kb.json").read_bytes()
    theirs = (ANALYSIS_LLM / "who_kb.json").read_bytes()
    assert hashlib.sha256(ours).hexdigest() == hashlib.sha256(theirs).hexdigest()


@needs_analysis
def test_every_module_combination_renders_identically(specs):
    old, new = specs
    modules = list(prompts.LIVE)
    compared = 0
    for k in range(len(modules) + 1):
        for combo in itertools.combinations(modules, k):
            for probabilities, evidence, arm, verbatim in itertools.product(
                (True, False), (True, False), ("literature", "none"), (False, True)
            ):
                kwargs = dict(modules=combo, probabilities=probabilities, evidence=evidence,
                              rules_arm=arm, verbatim_rules=verbatim)
                assert old.system_prompt(**kwargs) == new.system_prompt(**kwargs), kwargs
                compared += 1
    assert compared == 256


@needs_analysis
def test_response_schema_and_record_format_are_identical(specs, frozen):
    old, new = specs
    for probabilities, evidence in itertools.product((True, False), (True, False)):
        assert old.response_schema(probabilities, evidence) == new.response_schema(probabilities, evidence)
    assert frozen.format_record("N", "Q", sex="male", age_years=40.0) == prompts.format_record(
        sections={"narrative": "N", "questionnaire": "Q"}, sex="male", age_years=40.0)
    assert frozen.format_record("N", None, sex="female") == prompts.format_record(
        sections={"narrative": "N"}, sex="female")
    assert frozen.format_record(None, "Q") == prompts.format_record(sections={"questionnaire": "Q"})


@needs_analysis
def test_default_prompt_digest(specs):
    _, new = specs
    digest = hashlib.sha256(new.system_prompt().encode("utf-8")).hexdigest()
    assert digest.startswith(DEFAULT_PROMPT_SHA256)
    assert prompts.PROMPT_VERSION == "v2"


@needs_analysis
def test_knowledge_base_integrity_and_terminology(specs):
    _, new = specs
    assert prompts.check_knowledge_base(new) == []
    assert prompts.check_terminology(new) == []


def _norm(text: str) -> str:
    """Whitespace, line-wrap hyphenation and glued footnote callouts: the format
    changes WHO_NOTICE.md declares. Applied to both sides."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s+", "-", text)                       # "non- communicable"
    text = re.sub(r"(?<=\d)\.\d{1,2}\b", "", text)          # "COVID-19.33" (callout after a digit)
    text = re.sub(r"(?<=[A-Za-z\).])\d{1,2}\b", "", text)   # "Tetanus31", "disease34"
    return re.sub(r"\s+", " ", text).strip()


def _present(quote: str, corpus: str) -> bool:
    """The quote, or — where the PDF extraction interleaved a footnote body into
    an entry (recorded in the knowledge base's excerpt conventions) — each of
    its bullet fragments."""
    if _norm(quote) in corpus:
        return True
    fragments = [f.strip(" .") for f in _norm(quote).split("•") if f.strip(" .")]
    return len(fragments) > 1 and all(f in corpus for f in fragments)


@pytest.mark.skipif(not (ANALYSIS_LLM / "who_source").is_dir(),
                    reason="the raw WHO section extractions are not on this machine")
def test_who_quotes_are_verbatim_in_the_source():
    """What WHO_NOTICE.md promises: format changes only, no altered words."""
    kb = json.loads((PACKAGE_ROOT / "multimodalva" / "llm" / "data" / "who_kb.json").read_text())
    corpus = _norm(" ".join(p.read_text() for p in sorted((ANALYSIS_LLM / "who_source").glob("*_raw.txt"))))
    quoted: list[tuple[str, str]] = []
    quoted += [(f"criteria[{code}]", text) for code, text in kb["criteria"].items()]
    for code, notes in kb.get("criteria_footnotes", {}).items():
        quoted += [(f"criteria_footnotes[{code}]", note["who_quote"]) for note in notes]
    for group, note in kb.get("group_notes", {}).items():
        quoted.append((f"group_notes[{group}]", note["who_quote"]))
    for name, block in kb.get("shared_guidance", {}).items():
        quoted.append((f"shared_guidance[{name}]", block["guidance"]))
    for rule in kb["cross_cause_rules"]:
        for key in ("who_quote", "who_quote_2", "who_quote_3", "who_quote_4"):
            if rule.get(key):
                quoted.append((f"{rule['rule_id']}.{key}", rule[key]))
    assert len(quoted) > 60
    missing = [label for label, text in quoted if not _present(text, corpus)]
    assert not missing, f"{len(missing)} quoted field(s) are not in the source text: {missing[:5]}"
