"""Cause assignment by prompting a language model — ``task="llm"``.

No training: a fixed prompt (a base plus the modules M1–M4), a cause schema
describing what each label covers, and a chat model that returns a ranked list
of causes with probabilities, a confidence level, a rationale and quoted
evidence. The model runs in this process through transformers or on a local
ollama server.

    from multimodalva import run

    run(task="llm", data="clean.csv", label_col="cause", text_col="narrative",
        model="openai/gpt-oss-20b", cause_schema="causes.json",
        context="Deaths from a rural district with a high burden of HIV.",
        output_dir="runs/llm")

Importing this package does not load torch or transformers; the backend does
that when a run starts.
"""

from .backends import LLM_BACKENDS, LLM_DEFAULT_MODEL, LLM_REASONING_EFFORTS
from .cause_schema import (
    LLM_SCHEMA_EXAMPLE, build_system_prompt, cause_schema_template, describe_modules,
    load_cause_schema,
)
from .llm_classifier import LLM_EVALUATE_ON, LLM_PROB_SOURCES, LLMClassifier
from .prompts import LIVE as LLM_MODULES_LIVE
from .prompts import MODULES as LLM_MODULES
from .prompts import PROMPT_VERSION, format_record
from .selection import LLM_HYPERPARAMS_DEFAULT, LLM_SELECTION_GRIDS

__all__ = [
    "LLMClassifier",
    "load_cause_schema",
    "cause_schema_template",
    "build_system_prompt",
    "describe_modules",
    "format_record",
    "LLM_BACKENDS",
    "LLM_DEFAULT_MODEL",
    "LLM_REASONING_EFFORTS",
    "LLM_SCHEMA_EXAMPLE",
    "LLM_MODULES",
    "LLM_MODULES_LIVE",
    "LLM_HYPERPARAMS_DEFAULT",
    "LLM_SELECTION_GRIDS",
    "LLM_PROB_SOURCES",
    "LLM_EVALUATE_ON",
    "PROMPT_VERSION",
]
