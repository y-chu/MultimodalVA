#!/usr/bin/env python3
"""
Build API documentation for MultimodalVA.

Usage:
    python docs/build_docs.py            # update docs/index.html in-place
    python docs/build_docs.py --check    # dry-run: exit 1 if changes needed

How it works:
    1. Reads Python source files and extracts function/class signatures and
       docstrings using the standard ``ast`` module (no imports required).
    2. Parses Google-style docstrings (Args, Returns, Raises, Attributes sections).
    3. Generates HTML matching the existing CSS conventions in docs/index.html
       (.sig, .fn-name, .cls-name, .param, .returns span classes).
    4. Replaces content between <!-- AUTODOC:start:X --> / <!-- AUTODOC:end:X -->
       markers in docs/index.html.

Static sections (Installation, Quick Start, Demo Scripts, Design Notes, and the
Supported models tables) are untouched — only the AUTODOC regions are regenerated.
"""

from __future__ import annotations

import ast
import html
import re
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_FILE = REPO_ROOT / "docs" / "index.html"

# ---------------------------------------------------------------------------
# Documentation plan
#
# Each section in DOC_PLAN maps to one <!-- AUTODOC:start/end:X --> region.
# Two entry kinds:
#   "standalone" — one anchor + h3 heading for a single function or class.
#   "group"      — one anchor + h3 heading for a named group; h4 per member.
#
# Class entries support an optional "methods" list (method names to document)
# and "extra_funcs" (module-level functions to append after the class).
# ---------------------------------------------------------------------------

DOC_PLAN: dict[str, list[dict]] = {
    "api": [
        {
            "kind": "standalone",
            "anchor": "api-run",
            "heading": "run()",
            "file": "multimodalva/runner.py",
            "name": "run",
        },
    ],
    "utils": [
        {
            "kind": "standalone",
            "anchor": "utils-split",
            "heading": "split()",
            "file": "multimodalva/utils/split.py",
            "name": "split",
        },
        {
            "kind": "standalone",
            "anchor": "utils-types",
            "heading": "PredictionResult",
            "file": "multimodalva/utils/types.py",
            "name": "PredictionResult",
        },
        {
            "kind": "group",
            "anchor": "utils-metrics",
            "heading": "Metrics",
            "items": [
                {"file": "multimodalva/utils/metrics.py", "name": "csmf_accuracy"},
                {"file": "multimodalva/utils/metrics.py", "name": "score_predictions"},
                {"file": "multimodalva/utils/metrics.py", "name": "sample_hyperparams"},
            ],
        },
        {
            "kind": "group",
            "anchor": "utils-runtime",
            "heading": "Device and run helpers",
            "items": [
                {"file": "multimodalva/utils/runtime.py", "name": "get_device"},
                {"file": "multimodalva/utils/runtime.py", "name": "is_cuda"},
                {"file": "multimodalva/utils/runtime.py", "name": "is_mps"},
                {"file": "multimodalva/utils/runtime.py", "name": "empty_accelerator_cache"},
                {"file": "multimodalva/utils/runtime.py", "name": "distributed_state"},
                {"file": "multimodalva/utils/runtime.py", "name": "resolve_seed"},
            ],
        },
    ],
    "text": [
        {
            "kind": "standalone",
            "anchor": "text-classifier",
            "heading": "TextClassifier",
            "file": "multimodalva/text/text_classifier.py",
            "name": "TextClassifier",
            "methods": ["run"],
        },
        {
            "kind": "standalone",
            "anchor": "text-dataset",
            "heading": "prepare_dataset()",
            "file": "multimodalva/text/dataset.py",
            "name": "prepare_dataset",
        },
        {
            "kind": "standalone",
            "anchor": "text-train",
            "heading": "train()",
            "file": "multimodalva/text/train.py",
            "name": "train",
        },
        {
            "kind": "standalone",
            "anchor": "text-predict",
            "heading": "predict()",
            "file": "multimodalva/text/predict.py",
            "name": "predict",
        },
        {
            "kind": "group",
            "anchor": "text-hpo",
            "heading": "HPO",
            "items": [
                {"file": "multimodalva/text/hpo.py", "name": "optimize"},
                {"file": "multimodalva/text/hpo.py", "name": "optimize_ray"},
            ],
        },
    ],
    "tabular": [
        {
            "kind": "standalone",
            "anchor": "tabular-classifier",
            "heading": "TabularClassifier",
            "file": "multimodalva/tabular/tabular_classifier.py",
            "name": "TabularClassifier",
            "methods": ["run"],
        },
        {
            "kind": "standalone",
            "anchor": "tabular-dataset",
            "heading": "prepare_dataset()",
            "file": "multimodalva/tabular/dataset.py",
            "name": "prepare_dataset",
        },
        {
            "kind": "standalone",
            "anchor": "tabular-train",
            "heading": "train()",
            "file": "multimodalva/tabular/train.py",
            "name": "train",
        },
        {
            "kind": "standalone",
            "anchor": "tabular-predict",
            "heading": "predict()",
            "file": "multimodalva/tabular/predict.py",
            "name": "predict",
        },
        {
            "kind": "group",
            "anchor": "tabular-hpo",
            "heading": "HPO",
            "items": [
                {"file": "multimodalva/tabular/hpo.py", "name": "optimize"},
                {"file": "multimodalva/tabular/hpo.py", "name": "optimize_ray"},
            ],
        },
    ],
    "ensemble": [
        {
            "kind": "standalone",
            "anchor": "ensemble-dispatcher",
            "heading": "EnsembleClassifier",
            "file": "multimodalva/ensemble/ensemble_classifier.py",
            "name": "EnsembleClassifier",
            "methods": [
                "run",
                "train_base_models",
                "train_meta_learner_stage",
                "train_class_voter_stage",
                "predict_test",
                "predict_test_class_voter",
            ],
        },
        {
            "kind": "standalone",
            "anchor": "ensemble-data-fusion",
            "heading": "DataFusionClassifier",
            "file": "multimodalva/ensemble/data_fusion_classifier.py",
            "name": "DataFusionClassifier",
            "methods": ["run"],
        },
        {
            "kind": "standalone",
            "anchor": "ensemble-feature-fusion",
            "heading": "FeatureFusionClassifier",
            "file": "multimodalva/ensemble/feature_fusion.py",
            "name": "FeatureFusionClassifier",
            "methods": ["run"],
        },
        {
            "kind": "standalone",
            "anchor": "ensemble-voting",
            "heading": "SoftVotingClassifier / vote_from_results()",
            "file": "multimodalva/ensemble/voting.py",
            "name": "SoftVotingClassifier",
            "methods": ["run"],
            "extra_funcs": ["vote_from_results"],
        },
        {
            "kind": "standalone",
            "anchor": "ensemble-stacking",
            "heading": "StackingClassifier",
            "file": "multimodalva/ensemble/stacking.py",
            "name": "StackingClassifier",
            "methods": [
                "train_base_models",
                "train_meta_learner_stage",
                "train_class_voter_stage",
                "predict_test",
                "predict_test_class_voter",
                "run",
            ],
            "extra_funcs": ["generate_oof_predictions"],
        },
    ],
    "results": [
        {
            "kind": "group",
            "anchor": "results-train",
            "heading": "Training diagnostics",
            "items": [
                {"file": "multimodalva/results/vis_train.py", "name": "plot_loss_curves"},
                {"file": "multimodalva/results/vis_train.py", "name": "hpo_leaderboard"},
            ],
        },
        {
            "kind": "group",
            "anchor": "results-predict",
            "heading": "Prediction visualization",
            "items": [
                {"file": "multimodalva/results/vis_predict.py", "name": "performance_leaderboard"},
                {"file": "multimodalva/results/vis_predict.py", "name": "topk_from_full"},
                {"file": "multimodalva/results/vis_predict.py", "name": "topk_accuracy"},
                {"file": "multimodalva/results/vis_predict.py", "name": "cause_accuracy_heatmap"},
                {"file": "multimodalva/results/vis_predict.py", "name": "cause_accuracy_diff_heatmap"},
                {"file": "multimodalva/results/vis_predict.py", "name": "confusion_heatmap"},
            ],
        },
        {
            "kind": "group",
            "anchor": "results-calibration",
            "heading": "Calibration",
            "items": [
                {"file": "multimodalva/results/calibration.py", "name": "calibration_summary"},
                {"file": "multimodalva/results/calibration.py", "name": "classwise_bin_data"},
                {"file": "multimodalva/results/calibration.py", "name": "ece_score"},
                {"file": "multimodalva/results/calibration.py", "name": "mce_score"},
                {"file": "multimodalva/results/calibration.py", "name": "brier_multiclass"},
            ],
        },
        {
            "kind": "group",
            "anchor": "results-bootstrap",
            "heading": "Confidence intervals",
            "items": [
                {"file": "multimodalva/results/bootstrap.py", "name": "bootstrap_ci"},
                {"file": "multimodalva/results/bootstrap.py", "name": "paired_bootstrap_ci"},
                {"file": "multimodalva/results/bootstrap.py", "name": "predictions_frame"},
            ],
        },
    ],
}

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _parse_file(filepath: str | Path) -> ast.Module:
    src = (REPO_ROOT / filepath).read_text(encoding="utf-8")
    return ast.parse(src)


def _find_node(tree: ast.Module, name: str):
    """Return the top-level AST node for *name* (function or class)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return node
    return None


def _find_method(class_node: ast.ClassDef, method_name: str):
    """Return the AST node for *method_name* inside a class."""
    for node in ast.walk(class_node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == method_name:
                return node
    return None


def _sig_params(func_node, skip_self: bool = True) -> list[str]:
    """Return a list of 'param' or 'param=default' strings."""
    args = func_node.args
    all_args = args.posonlyargs + args.args
    n_no_default = len(all_args) - len(args.defaults)
    params: list[str] = []

    for i, arg in enumerate(all_args):
        if skip_self and arg.arg == "self":
            continue
        default_idx = i - n_no_default
        if default_idx >= 0:
            params.append(f"{arg.arg}={ast.unparse(args.defaults[default_idx])}")
        else:
            params.append(arg.arg)

    if args.vararg:
        params.append(f"*{args.vararg.arg}")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            params.append(f"{arg.arg}={ast.unparse(default)}")
        else:
            params.append(f"{arg.arg}")

    if args.kwarg:
        params.append(f"**{args.kwarg.arg}")

    return params


def _simplify_annotation(ann_str: str) -> str | None:
    """Simplify a type annotation string for display in the signature."""
    ann_str = ann_str.strip()
    if ann_str in ("None", ""):
        return None
    # Strip module prefixes: pd.DataFrame → DataFrame, np.ndarray → ndarray
    ann_str = re.sub(r'\b\w+\.(\w+)', r'\1', ann_str)
    # tuple[A, B] → (A, B)
    m = re.match(r'^tuple\[(.+)\]$', ann_str)
    if m:
        ann_str = f"({m.group(1)})"
    return ann_str


def _return_display(func_node, parsed_doc: dict) -> str | None:
    """Determine what to show after → in the signature."""
    # Named returns from docstring (e.g. train_df, test_df) take priority.
    # Only accept simple identifiers — skip names that contain spaces or backticks
    # (those are descriptions that happen to have a colon, e.g. "Tuple ``(a, b)`` where:").
    returns = parsed_doc.get("returns", [])
    named = [name for name, _ in returns if name and re.match(r'^\w+$', name)]
    if len(named) > 1:
        return f"({', '.join(named)})"
    if len(named) == 1:
        return named[0]

    # Fall back to type annotation
    if func_node.returns:
        return _simplify_annotation(ast.unparse(func_node.returns))
    return None


# ---------------------------------------------------------------------------
# Docstring parser (Google-style)
# ---------------------------------------------------------------------------

_STRUCTURED_SECTIONS = {
    "Args:": "args",
    "Parameters:": "args",
    "Attributes:": "attributes",
    "Returns:": "returns",
    "Raises:": "raises",
}


def parse_docstring(raw: str | None) -> dict:
    """Parse a Google-style docstring into structured components.

    Returns a dict with keys:
        desc        — list of paragraph strings (the description)
        args        — list of (name, description) tuples
        attributes  — list of (name, description) tuples (for NamedTuples)
        returns     — list of (name, description) tuples; name may be ""
        raises      — list of (exc_type, description) tuples
        examples    — list of raw code block strings
    """
    result: dict = {
        "desc": [],
        "args": [],
        "attributes": [],
        "returns": [],
        "raises": [],
        "examples": [],
    }
    if not raw:
        return result

    lines = textwrap.dedent(raw).strip().splitlines()

    section = "desc"       # current parsing section
    current_key: str | None = None
    current_parts: list[str] = []
    desc_lines: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_parts
        if current_key is not None and section in result:
            result[section].append((current_key, " ".join(current_parts).strip()))
        current_key = None
        current_parts = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ---- section header detection ----
        if stripped in _STRUCTURED_SECTIONS:
            flush()
            section = _STRUCTURED_SECTIONS[stripped]
            i += 1
            continue

        # Unrecognised "Header:" lines (Formula:, Note:, Notes:, etc.) — collect
        # as description text. Only trigger for top-level lines (indent ≤ 4) in
        # non-desc sections to avoid false positives on continuation lines.
        _indent = len(line) - len(line.lstrip(" "))
        if section != "desc" and _indent <= 4 and re.match(r'^[A-Z][A-Za-z ]+:$', stripped):
            flush()
            section = "desc"
            desc_lines.append("")  # paragraph break
            desc_lines.append(stripped)
            i += 1
            continue

        # ---- code example detection (:: suffix in desc) ----
        if section == "desc" and stripped.endswith("::") and len(stripped) > 2:
            label = stripped[:-2].strip()
            if label:
                desc_lines.append(label + ":")
            # Collect indented code block that follows
            i += 1
            code_lines: list[str] = []
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                if next_stripped == "" or next_line.startswith("    "):
                    code_lines.append(next_line)
                    i += 1
                else:
                    break
            code_text = textwrap.dedent("\n".join(code_lines)).strip()
            if code_text:
                result["examples"].append(code_text)
            continue

        # ---- structured section parsing (Args / Returns / Raises / Attributes) ----
        if section in ("args", "attributes", "returns", "raises"):
            indent = len(line) - len(line.lstrip(" "))

            if stripped == "":
                flush()
                i += 1
                continue

            # New entry: indent ≤ 4 and has a colon (e.g. "    name: description")
            if indent <= 4 and ":" in stripped and not stripped.startswith("`"):
                flush()
                colon_pos = stripped.index(":")
                key = stripped[:colon_pos].strip()
                val = stripped[colon_pos + 1:].strip()
                current_key = key
                current_parts = [val] if val else []
            elif current_key is not None:
                # Continuation line for current entry
                current_parts.append(stripped)
            else:
                # Section text with no name prefix (e.g. a plain Returns description)
                current_key = ""
                current_parts = [stripped]

            i += 1
            continue

        # ---- description line ----
        desc_lines.append(stripped)
        i += 1

    flush()

    # Build desc paragraphs (split on blank lines)
    paragraphs: list[str] = []
    current_para: list[str] = []
    for dl in desc_lines:
        if dl:
            current_para.append(dl)
        else:
            if current_para:
                paragraphs.append(" ".join(current_para))
            current_para = []
    if current_para:
        paragraphs.append(" ".join(current_para))
    result["desc"] = paragraphs

    return result


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _h(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text), quote=False)


def _inline_code(text: str) -> str:
    """Replace ``backtick`` spans with <code> tags."""
    return re.sub(r'``([^`]+)``', lambda m: f'<code>{_h(m.group(1))}</code>', _h(text))


def _render_sig(name: str, params: list[str], return_type: str | None,
                is_class: bool = False, is_method: bool = False) -> str:
    """Render an HTML .sig div with colored spans."""
    name_class = "cls-name" if is_class else "fn-name"
    prefix = "  " if is_method else ""

    # Build param spans
    param_html = ", ".join(
        f'<span class="param">{_h(p)}</span>' for p in params
    )

    ret_html = ""
    if return_type:
        ret_html = f' → <span class="returns">{_h(return_type)}</span>'

    return (
        f'    <div class="sig">\n'
        f'      {prefix}<span class="{name_class}">{_h(name)}</span>'
        f'({param_html}){ret_html}\n'
        f'    </div>'
    )


def _render_param_table(entries: list[tuple[str, str]], header: str = "Parameter") -> str:
    """Render a parameter/attribute table."""
    if not entries:
        return ""
    rows = "\n".join(
        f'      <tr><td><code>{_h(name)}</code></td>'
        f'<td>{_inline_code(desc)}</td></tr>'
        for name, desc in entries
        if name  # skip unnamed entries
    )
    if not rows:
        return ""
    return (
        f'    <table>\n'
        f'      <tr><th>{header}</th><th>Description</th></tr>\n'
        f'{rows}\n'
        f'    </table>'
    )


def _render_func_block(
    func_node,
    heading_level: int = 4,
    anchor: str | None = None,
    heading_text: str | None = None,
    is_class: bool = False,
    is_method: bool = False,
) -> str:
    """Render the full HTML block for one function or method."""
    raw_doc = ast.get_docstring(func_node)
    parsed = parse_docstring(raw_doc)

    name = func_node.name
    display_name = heading_text or (f"{name}()" if not is_class else name)

    params = _sig_params(func_node, skip_self=is_method or is_class)
    ret = _return_display(func_node, parsed)

    parts: list[str] = []

    # Anchor + heading
    if anchor:
        parts.append(f'    <a class="anchor" id="{anchor}"></a>')
    tag = f"h{heading_level}"
    parts.append(f'    <{tag}>{_h(display_name)}</{tag}>')

    # Signature
    parts.append(_render_sig(name, params, ret, is_class=is_class, is_method=is_method))

    # Description
    for para in parsed["desc"]:
        parts.append(f'    <p>{_inline_code(para)}</p>')

    # Examples from class docstring (before methods)
    for code in parsed["examples"]:
        parts.append(f'    <pre><code>{_h(code)}</code></pre>')

    # Args / Attributes table
    if parsed["args"]:
        parts.append(_render_param_table(parsed["args"], "Parameter"))
    if parsed["attributes"]:
        parts.append(_render_param_table(parsed["attributes"], "Attribute"))

    # Returns
    returns = parsed["returns"]
    named_returns = [(n, d) for n, d in returns if n]
    unnamed_returns = [(n, d) for n, d in returns if not n]
    if len(named_returns) > 1:
        parts.append(_render_param_table(named_returns, "Returns"))
    elif named_returns:
        n, d = named_returns[0]
        if d:
            parts.append(f'    <p><strong>Returns</strong> <code>{_h(n)}</code>: {_inline_code(d)}</p>')
    elif unnamed_returns:
        combined = " ".join(d for _, d in unnamed_returns if d)
        if combined:
            parts.append(f'    <p><strong>Returns:</strong> {_inline_code(combined)}</p>')

    # Raises
    if parsed["raises"]:
        parts.append(_render_param_table(parsed["raises"], "Raises"))

    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Section HTML generators
# ---------------------------------------------------------------------------

def _render_standalone(spec: dict) -> str:
    """Generate HTML for a standalone function or class entry."""
    filepath = spec["file"]
    name = spec["name"]
    anchor = spec["anchor"]
    heading = spec.get("heading", f"{name}()")
    methods = spec.get("methods", [])
    extra_funcs = spec.get("extra_funcs", [])

    tree = _parse_file(filepath)
    node = _find_node(tree, name)
    if node is None:
        return f'    <!-- WARNING: {name} not found in {filepath} -->'

    parts: list[str] = []

    is_class = isinstance(node, ast.ClassDef)

    if is_class:
        # Class heading + __init__ signature + class docstring
        init_node = _find_method(node, "__init__")
        raw_class_doc = ast.get_docstring(node)
        parsed_class = parse_docstring(raw_class_doc)

        # Use __init__ signature if available, otherwise empty params
        if init_node:
            params = _sig_params(init_node, skip_self=True)
        else:
            params = []

        parts.append(f'    <a class="anchor" id="{anchor}"></a>')
        parts.append(f'    <h3>{_h(heading)}</h3>')
        parts.append(_render_sig(name, params, None, is_class=True))

        for para in parsed_class["desc"]:
            parts.append(f'    <p>{_inline_code(para)}</p>')
        for code in parsed_class["examples"]:
            parts.append(f'    <pre><code>{_h(code)}</code></pre>')

        # __init__ args table (from __init__ docstring or class docstring)
        if init_node:
            raw_init_doc = ast.get_docstring(init_node)
            parsed_init = parse_docstring(raw_init_doc)
            if parsed_init["args"]:
                parts.append(_render_param_table(parsed_init["args"], "Parameter"))
            elif parsed_class["args"]:
                parts.append(_render_param_table(parsed_class["args"], "Parameter"))
        elif parsed_class["args"]:
            parts.append(_render_param_table(parsed_class["args"], "Parameter"))

        if parsed_class["attributes"]:
            parts.append(_render_param_table(parsed_class["attributes"], "Attribute"))

        # Methods
        for method_name in methods:
            method_node = _find_method(node, method_name)
            if method_node is None:
                parts.append(f'    <!-- WARNING: method {method_name} not found -->')
                continue
            parts.append("")
            parts.append(_render_func_block(
                method_node,
                heading_level=4,
                heading_text=f".{method_name}()",
                is_method=True,
            ))

    else:
        # Plain function
        block = _render_func_block(
            node,
            heading_level=3,
            anchor=anchor,
            heading_text=heading,
        )
        parts.append(block)

    # Extra module-level functions appended after class
    for func_name in extra_funcs:
        func_node = _find_node(tree, func_name)
        if func_node is None:
            parts.append(f'    <!-- WARNING: {func_name} not found in {filepath} -->')
            continue
        parts.append("")
        parts.append(_render_func_block(
            func_node,
            heading_level=4,
            heading_text=f"{func_name}()",
        ))

    return "\n".join(p for p in parts if p is not None)


def _render_group(spec: dict) -> str:
    """Generate HTML for a group entry (h3 heading + h4 per member function)."""
    anchor = spec["anchor"]
    heading = spec["heading"]
    items = spec["items"]

    parts: list[str] = [
        f'    <a class="anchor" id="{anchor}"></a>',
        f'    <h3>{_h(heading)}</h3>',
    ]

    for item in items:
        filepath = item["file"]
        name = item["name"]
        tree = _parse_file(filepath)
        node = _find_node(tree, name)
        if node is None:
            parts.append(f'    <!-- WARNING: {name} not found in {filepath} -->')
            continue
        parts.append("")
        parts.append(_render_func_block(node, heading_level=4, heading_text=f"{name}()"))

    return "\n".join(p for p in parts if p is not None)


def _render_section(section_entries: list[dict]) -> str:
    """Render all entries for one AUTODOC section."""
    blocks: list[str] = []
    for entry in section_entries:
        kind = entry["kind"]
        if kind == "standalone":
            blocks.append(_render_standalone(entry))
        elif kind == "group":
            blocks.append(_render_group(entry))
        else:
            blocks.append(f'    <!-- WARNING: unknown kind {kind!r} -->')
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Marker injection
# ---------------------------------------------------------------------------

_START_RE = re.compile(r'<!-- AUTODOC:start:(\w+) -->')
_END_RE = re.compile(r'<!-- AUTODOC:end:(\w+) -->')


def inject(html_src: str) -> str:
    """Replace AUTODOC-marked regions with freshly generated HTML."""
    lines = html_src.splitlines(keepends=True)
    out: list[str] = []
    skip = False

    for line in lines:
        # Start marker — emit line, then emit generated content, then skip until end
        start_m = _START_RE.search(line)
        if start_m:
            section_id = start_m.group(1)
            out.append(line)
            if section_id in DOC_PLAN:
                generated = _render_section(DOC_PLAN[section_id])
                out.append(generated + "\n")
            else:
                out.append(f"    <!-- WARNING: no DOC_PLAN entry for {section_id!r} -->\n")
            skip = True
            continue

        # End marker — stop skipping and emit the end-marker line
        end_m = _END_RE.search(line)
        if end_m:
            skip = False
            out.append(line)
            continue

        if not skip:
            out.append(line)

    return "".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    check_only = "--check" in sys.argv

    original = DOCS_FILE.read_text(encoding="utf-8")
    updated = inject(original)

    if check_only:
        if updated != original:
            print("docs/index.html is out of date. Run: python docs/build_docs.py", file=sys.stderr)
            sys.exit(1)
        print("docs/index.html is up to date.")
    else:
        if updated == original:
            print("docs/index.html already up to date — no changes written.")
        else:
            DOCS_FILE.write_text(updated, encoding="utf-8")
            print(f"docs/index.html updated.")


if __name__ == "__main__":
    main()
