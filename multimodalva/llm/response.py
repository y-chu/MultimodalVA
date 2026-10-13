"""Reading a model's answer: JSON extraction, validation, and the harmony format.

A backend hands back text. Three things happen to it here, in order: the JSON
object is located (a model may wrap it in a code fence or precede it with
prose); it is checked against the same constraints the response schema states,
because a server that was given the schema does not always enforce all of it;
and the model-facing shape (one ranked list carrying cause and probability) is
translated into the stored shape (a list of names plus a mapping).
"""

from __future__ import annotations

import json
import re

from .prompts import CONFIDENCE_LEVELS, EVIDENCE_POLARITY, EVIDENCE_STANCE, MAX_EVIDENCE, MAX_RANKED

__all__ = ["extract_json", "valid_response", "normalise_response", "split_harmony"]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

#: Markers of the gpt-oss "harmony" response format. The final answer sits in
#: the ``final`` channel; everything before it is reasoning.
_HARMONY_FINAL = "<|channel|>final<|message|>"
_HARMONY_ENDS = ("<|return|>", "<|end|>", "<|call|>")


def split_harmony(text: str) -> tuple[str, str | None]:
    """``(final content, reasoning or None)`` from a harmony-format completion.

    Text without harmony markers is returned unchanged with no reasoning, so
    this is safe to apply to any model's output.
    """
    if _HARMONY_FINAL not in text:
        return text, None
    head, _, tail = text.rpartition(_HARMONY_FINAL)
    for end in _HARMONY_ENDS:
        cut = tail.find(end)
        if cut >= 0:
            tail = tail[:cut]
    # Reasoning is whatever sat in the earlier channels' message bodies.
    bodies = re.findall(r"<\|message\|>(.*?)(?:<\|end\|>|<\|call\|>|$)", head, re.DOTALL)
    reasoning = re.sub(r"\s+", " ", " ".join(bodies)).strip() or None
    return tail.strip(), reasoning


def extract_json(text: str) -> dict | None:
    """The first JSON object in ``text``, or ``None`` when there is none."""
    if not isinstance(text, str) or not text.strip():
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
        start, end = candidate.find("{"), candidate.rfind("}")
        if 0 <= start < end:
            try:
                parsed = json.loads(candidate[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except ValueError:
                continue
    return None


def valid_response(
    parsed: object,
    causes: list[str],
    *,
    sources: list[str] | tuple[str, ...],
    probabilities: bool = True,
    evidence: bool = True,
) -> bool:
    """Mirror the response schema's constraints before accepting a response.

    A grammar-constrained server does not reliably enforce every constraint
    (``uniqueItems`` is known not to be), and a plain completion enforces none,
    so every constraint is re-checked here.
    """
    if not isinstance(parsed, dict):
        return False
    ranked = parsed.get("ranked_causes")
    if not isinstance(ranked, list) or not 1 <= len(ranked) <= MAX_RANKED:
        return False
    if probabilities:
        if not all(
            isinstance(item, dict)
            and item.get("cause") in causes
            and isinstance(item.get("probability"), (int, float))
            and not isinstance(item.get("probability"), bool)
            and 0 <= float(item["probability"]) <= 1
            for item in ranked
        ):
            return False
        names = [item["cause"] for item in ranked]
    else:
        if not all(isinstance(item, str) and item in causes for item in ranked):
            return False
        names = ranked
    if len(set(names)) != len(names):
        return False
    if parsed.get("confidence") not in CONFIDENCE_LEVELS:
        return False
    if not isinstance(parsed.get("rationale"), str):
        return False
    if evidence:
        items = parsed.get("evidence")
        if not (isinstance(items, list) and 1 <= len(items) <= MAX_EVIDENCE):
            return False
        for item in items:
            if not (
                isinstance(item, dict)
                and isinstance(item.get("quote"), str)
                and item.get("quote").strip()
                and item.get("source") in sources
                and item.get("polarity") in EVIDENCE_POLARITY
                and item.get("supports") in EVIDENCE_STANCE
            ):
                return False
    return True


def normalise_response(parsed: dict) -> dict:
    """Translate the model-facing shape into the stored shape.

    The model returns one ordered list carrying the ranking and the
    probabilities together. Stored rows keep ``ranked_causes`` as a list of
    names and ``cause_probabilities`` as a sparse mapping over the causes the
    model listed; the full vector is reconstructed at export.
    """
    ranked = parsed.get("ranked_causes") or []
    if ranked and isinstance(ranked[0], dict):
        parsed["cause_probabilities"] = {
            item["cause"]: float(item["probability"]) for item in ranked
        }
        parsed["ranked_causes"] = [item["cause"] for item in ranked]
    return parsed
