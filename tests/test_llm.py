"""``task="llm"``: the run contract, the guards, and the selection — with a fake backend.

No model is loaded anywhere here. ``FakeBackend`` answers from the record
text, which is enough to exercise everything around the model: sections and
markers, the cause schema, the dev slice, resume, the failure path, the module
grid, and the preflight pair (a config preflight accepts really runs; one it
rejects really fails).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import multimodalva as mv
from multimodalva import Optimize
from multimodalva.llm import LLMClassifier, cause_schema_template, load_cause_schema
from multimodalva.llm.backends import FakeBackend, resolve_llm_backend
from multimodalva.llm.llm_classifier import split_sections
from multimodalva.llm.prompts import LIVE, format_record, source_phrase
from multimodalva.llm.response import extract_json, split_harmony, valid_response
from multimodalva.llm.selection import resolve_llm_hyperparams, validate_llm_fixed
from multimodalva.utils.runtime import RUN_RESULT_KEYS

LABEL = "cause_of_death"


@pytest.fixture(scope="module")
def df():
    return mv.data("va_sample_text_only", n_per_class=6)


@pytest.fixture(scope="module")
def causes(df):
    return sorted(df[LABEL].unique())


def _record_text(user: str) -> str:
    body = user.split("\nRespond with exactly one JSON object", 1)[0]
    return body.split("NARRATIVE\n", 1)[1] if "NARRATIVE\n" in body else body


def make_responder(causes, *, fail_on: str | None = None, sources=("narrative",)):
    """Deterministic answers from the record text; one record can be made to fail."""

    def respond(system, user):
        text = _record_text(user)
        if fail_on and fail_on in text:
            return "I cannot answer this."
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        first, second = causes[h % len(causes)], causes[(h + 1) % len(causes)]
        quote = text.strip().split(". ")[0][:40]
        # Prefer a cause that is literally named in the text, so scores vary by prompt.
        for cause in causes:
            if cause.split(" ")[0].lower() in text.lower() and "M2" in system:
                first = cause
                break
        return json.dumps({
            "ranked_causes": [{"cause": first, "probability": 0.7},
                              {"cause": second, "probability": 0.2}],
            "confidence": "medium", "rationale": "the record says so",
            "evidence": [{"quote": quote, "source": sources[0],
                          "polarity": "present", "supports": "supports"}],
        })

    return respond


def _run(df, causes, tmp, **kw):
    backend = kw.pop("backend_instance", None) or FakeBackend(make_responder(causes))
    kw.setdefault("cause_schema", "who_target_list_adult")
    kw.setdefault("context", "A test population.")
    return mv.run(task="llm", data=df, label_col=LABEL, text_col="narrative", id_col="id",
                  output_dir=tmp, model="fake/model", backend_instance=backend, **kw)


# --- the run contract ----------------------------------------------------------

def test_run_contract_and_files(df, causes, tmp_path):
    out = _run(df, causes, tmp_path, evaluate_on="test")
    assert not [k for k in RUN_RESULT_KEYS if k not in out]
    root = Path(out["output_dir"])
    for name in ("predictions/predictions_top1.csv", "predictions/predictions_full.csv",
                 "predictions/predictions_topk.csv", "predictions/records.jsonl",
                 "predictions/llm_metadata.csv", "training_metadata.json", "validation.json",
                 "runtime/pipeline_runtime.json", "label2id.json", "resume_manifest.json"):
        assert (root / name).is_file(), name
    top1 = out["predictions"].top1
    assert len(top1) == len(pd.read_csv(root / "predictions/llm_metadata.csv"))
    assert list(top1.columns[:2]) == ["id", "true_label"]
    full = out["predictions"].full
    probs = full[[c for c in full.columns if c.startswith("prob_")]].to_numpy()
    assert probs.shape[1] == len(causes) and (abs(probs.sum(axis=1) - 1) < 1e-9).all()
    meta = out["train_metadata"]
    assert meta["pipeline"] == "llm" and meta["prompt_version"] and len(meta["prompt_sha256"]) == 64
    assert meta["hyperparams"]["modules"] == list(LIVE)
    assert out["validation"]["source"] == "none"
    assert out["best_hyperparams"] and "value" not in out["best_hyperparams"]
    assert len(top1) == meta["n_test"] < len(df)
    # rows are in the partition's order, keyed by id
    records = [json.loads(l) for l in (root / "predictions/records.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == top1["id"].tolist()


def test_the_default_prompts_every_row_with_no_split(df, causes, tmp_path):
    out = _run(df, causes, tmp_path)
    assert len(out["predictions"].top1) == len(df)
    meta = out["train_metadata"]
    assert meta["evaluate_on"] == "all" and meta["n_train"] == 0 and meta["labelled"]
    assert "true_label" in out["predictions"].top1.columns
    assert not (Path(out["output_dir"]) / "dev").exists()


def test_unlabelled_records_need_a_schema_and_get_no_true_label(df, causes, tmp_path):
    unlabelled = df.drop(columns=[LABEL])
    backend = FakeBackend(make_responder(causes))
    with pytest.raises(TypeError, match="label_col"):
        mv.run(task="llm", data=unlabelled, text_col="narrative", id_col="id",
               output_dir=tmp_path, model="fake", backend_instance=backend,
               evaluate_on="test", cause_schema="who_target_list_adult")
    with pytest.raises(ValueError, match="cause_schema"):
        mv.run(task="llm", data=unlabelled, text_col="narrative", id_col="id",
               output_dir=tmp_path, model="fake", backend_instance=backend)
    schema = {"causes": {c: {} for c in causes}}
    out = mv.run(task="llm", data=unlabelled, text_col="narrative", id_col="id",
                 output_dir=tmp_path, model="fake", backend_instance=backend,
                 cause_schema=schema)
    top1 = out["predictions"].top1
    assert len(top1) == len(df) and "true_label" not in top1.columns
    assert list(out["id2label"].values()) == causes
    meta = pd.read_csv(Path(out["output_dir"]) / "predictions/llm_metadata.csv")
    assert "true_label" not in meta.columns and len(meta) == len(df)
    assert not out["train_metadata"]["labelled"]
    report = mv.preflight(task="llm", data=unlabelled, text_col="narrative", id_col="id",
                          output_dir=tmp_path / "pf", cause_schema=schema)
    assert report["ok"], report["problems"]
    assert "unlabelled" in report["facts"]["rows"]


def test_selection_needs_a_split(df, causes, tmp_path):
    with pytest.raises(ValueError, match="evaluate_on='test'"):
        _run(df, causes, tmp_path, hyperparams=Optimize())
    report = mv.preflight(task="llm", data=df, label_col=LABEL, text_col="narrative",
                          output_dir=tmp_path, hyperparams=Optimize())
    assert not report["ok"] and "dev slice" in "\n".join(report["problems"])


def test_a_dev_run_never_writes_predictions(df, causes, tmp_path):
    out = _run(df, causes, tmp_path, evaluate_on="dev", dev_size=0.2)
    root = Path(out["output_dir"])
    assert not (root / "predictions").exists()
    assert (root / "dev/predictions_top1.csv").is_file()
    assert (root / "dev/dev_scores.json").is_file()
    assert out["validation"]["source"] == "dev_slice"
    ids = json.loads((root / "dev/dev_ids.json").read_text())
    assert ids["split_seed"] == 42 and ids["dev_size"] == 0.2 and len(ids["dev_ids"]) == out["train_metadata"]["n_dev"]
    assert out["dev_scores"]["f1_macro"] >= 0
    # the same slice again, by its ids file
    again = _run(df, causes, tmp_path / "again", evaluate_on="dev",
                 dev_ids=str(root / "dev/dev_ids.json"))
    assert json.loads((Path(again["output_dir"]) / "dev/dev_ids.json").read_text())["dev_ids"] == ids["dev_ids"]


def test_dev_ids_in_the_test_partition_are_refused(df, causes, tmp_path):
    clf = LLMClassifier("fake", tmp_path, backend_instance=FakeBackend(make_responder(causes)))
    from multimodalva.utils.split import split as split_frame

    train, test = split_frame(df, label_col=LABEL, test_size=0.2, random_state=42)
    with pytest.raises(ValueError, match="test partition"):
        clf.run(df, label_col=LABEL, text_col="narrative", id_col="id", evaluate_on="dev",
                dev_ids=[test["id"].iloc[0], train["id"].iloc[0]])


# --- failures and resume ---------------------------------------------------------

def test_a_failed_record_blocks_export_and_resume_retries_only_it(df, causes, tmp_path):
    from multimodalva.utils.split import split as split_frame

    _, test = split_frame(df, label_col=LABEL, test_size=0.2, random_state=42)
    target = str(test["narrative"].iloc[0]).strip()[:60]
    bad = FakeBackend(make_responder(causes, fail_on=target))
    with pytest.raises(RuntimeError, match="no valid answer"):
        _run(df, causes, tmp_path, backend_instance=bad, evaluate_on="test")
    assert not (tmp_path / "predictions/predictions_top1.csv").exists()
    assert sum(target in _record_text(c["user"]) for c in bad.calls) == 2, "one retry at 3x"
    good = FakeBackend(make_responder(causes))
    out = _run(df, causes, tmp_path, backend_instance=good, evaluate_on="test")
    # the resumed run called the model for the one record that had no answer
    assert len(good.calls) == 1 and target in _record_text(good.calls[0]["user"])
    assert len(out["predictions"].top1) == out["train_metadata"]["n_test"] == len(test)


def test_a_changed_configuration_refuses_to_resume(df, causes, tmp_path):
    _run(df, causes, tmp_path)
    with pytest.raises(RuntimeError, match="Refusing to resume"):
        _run(df, causes, tmp_path, hyperparams={"modules": ["M2", "M3"]})
    _run(df, causes, tmp_path, hyperparams={"modules": ["M2", "M3"]}, resume=False)


# --- sections --------------------------------------------------------------------

def test_sections_as_columns_and_the_source_enum(df, causes, tmp_path):
    two = df.assign(q="The respondent reported fever: yes.")
    backend = FakeBackend(make_responder(causes, sources=("questionnaire",)))
    out = mv.run(task="llm", data=two, label_col=LABEL, id_col="id", output_dir=tmp_path,
                 model="fake", backend_instance=backend,
                 sections={"narrative": "narrative", "questionnaire": "q"})
    assert out["train_metadata"]["sections"] == ["narrative", "questionnaire"]
    user = backend.calls[0]["user"]
    assert "NARRATIVE\n" in user and "STRUCTURED QUESTIONNAIRE RESPONSES\n" in user
    assert "narrative, or questionnaire, naming which section" in backend.calls[0]["system"]
    assert '"enum": ["narrative", "questionnaire"]' in json.dumps(
        json.loads(user.split("JSON schema:\n", 1)[1]))


def test_single_section_prompt_names_only_that_section(df, causes, tmp_path):
    backend = FakeBackend(make_responder(causes))
    _run(df, causes, tmp_path, backend_instance=backend)
    assert "source   - narrative, naming which section" in backend.calls[0]["system"]
    assert source_phrase(["a", "b", "c"]) == "a, b, or c"


def test_section_markers_split_strictly():
    markers = {"narrative": None, "questionnaire": "======="}
    assert split_sections("story\n=======\nq1\nq2", markers) == {"narrative": "story", "questionnaire": "q1\nq2"}
    with pytest.raises(ValueError, match="more than once"):
        split_sections("a\n=======\nb\n=======\nc", markers)
    with pytest.raises(ValueError, match="missing"):
        split_sections("a\nb", markers)
    three = {"narrative": "NARRATIVE", "questionnaire": "QUESTIONNAIRE", "lab": "LAB"}
    with pytest.raises(ValueError, match="out of order"):
        split_sections("NARRATIVE\na\nLAB\nc\nQUESTIONNAIRE\nb", three)
    assert split_sections("NARRATIVE\na\nQUESTIONNAIRE\nb\nLAB\nc", three) == {
        "narrative": "a", "questionnaire": "b", "lab": "c"}


def test_section_markers_through_run_and_preflight(df, causes, tmp_path):
    fused = df.assign(fused=df["narrative"] + "\n=======\nFever: yes. Cough: no.")
    markers = {"narrative": None, "questionnaire": "======="}
    report = mv.preflight(task="llm", data=fused, label_col=LABEL, text_col="fused",
                          section_markers=markers, output_dir=tmp_path)
    assert report["ok"], report["problems"]
    assert "=======" in report["facts"]["section markers"]
    backend = FakeBackend(make_responder(causes))
    out = mv.run(task="llm", data=fused, label_col=LABEL, text_col="fused", id_col="id",
                 output_dir=tmp_path, model="fake", backend_instance=backend,
                 section_markers=markers)
    assert out["train_metadata"]["section_markers"] == markers
    assert "STRUCTURED QUESTIONNAIRE RESPONSES\nFever: yes." in backend.calls[0]["user"]
    # a row whose marker is missing is refused, and preflight says so first
    broken = fused.copy()
    broken.loc[0, "fused"] = "no marker here"
    report = mv.preflight(task="llm", data=broken, label_col=LABEL, text_col="fused",
                          section_markers=markers, output_dir=tmp_path / "b")
    assert not report["ok"] and "missing from some rows" in "\n".join(report["problems"])
    with pytest.raises(ValueError, match="cannot be rendered"):
        mv.run(task="llm", data=broken, label_col=LABEL, text_col="fused", id_col="id",
               output_dir=tmp_path / "b", model="fake", backend_instance=backend,
               section_markers=markers)


def test_format_record_drops_empty_sections():
    text = format_record(sections={"narrative": "N", "questionnaire": ""}, sex="male")
    assert "STRUCTURED QUESTIONNAIRE" not in text and text.endswith("NARRATIVE\nN")
    with pytest.raises(ValueError):
        format_record(sections={"narrative": "", "questionnaire": None})


# --- the cause schema ------------------------------------------------------------

def test_schema_checks(causes, tmp_path):
    with pytest.raises(ValueError, match="not in the cause schema"):
        load_cause_schema({"causes": {"Pneumonia": {}}}, causes)
    template = cause_schema_template(causes)
    path = tmp_path / "t.json"
    path.write_text(json.dumps(template))
    with pytest.raises(ValueError, match="TODO"):
        load_cause_schema(path, causes)
    with pytest.raises(ValueError, match="unknown key"):
        load_cause_schema({"causes": {c: {"defn": "x"} for c in causes}}, causes)
    with pytest.raises(ValueError, match="master_category"):
        load_cause_schema({"causes": {**{c: {} for c in causes}, causes[0]: {"master_category": "Nope"}}}, causes)
    spec = load_cause_schema(None, causes)
    assert spec.causes == causes and spec.population_context == ""


def test_no_definitions_warns_and_names_only(df, causes, tmp_path, caplog):
    # Labels the WHO rules do not know, and no schema: every module but M2 is bare.
    own = df.assign(**{LABEL: "Cause " + df[LABEL].str[:3].str.upper()})
    own_causes = sorted(own[LABEL].unique())
    backend = FakeBackend(make_responder(own_causes))
    with caplog.at_level("WARNING"):
        out = mv.run(task="llm", data=own, label_col=LABEL, text_col="narrative", id_col="id",
                     output_dir=tmp_path, model="fake", backend_instance=backend)
    warnings = "\n".join(out["train_metadata"]["module_warnings"])
    assert "M1 (study context) is off" in warnings
    assert "M3 (cause scope) has nothing to render" in warnings and "llm-prompt" in warnings
    # two WHO rules apply to every cause list; the other fifteen name causes
    assert "M4 (cross-cause rules): 2 of 17" in warnings
    system = backend.calls[0]["system"]
    assert "ALLOWED CAUSES\n" + "; ".join(own_causes) in system and "STUDY CONTEXT" not in system
    assert system.count("\n* ") == 2, "the two universal WHO rules, and no per-cause scope"


def test_master_category_alias_lets_rules_apply(causes):
    renamed = {f"Label {i}": c for i, c in enumerate(causes)}
    schema = {"causes": {k: {"master_category": v, "definition": v} for k, v in renamed.items()}}
    spec = load_cause_schema(schema, list(renamed))
    block = spec.cross_cause_rules("literature")
    assert block and "* Label" in block


# --- hyperparameters and selection ---------------------------------------------------

def test_fixed_hyperparams_are_validated():
    full = validate_llm_fixed({"modules": ["M3", "M1"], "temperature": 0.5})
    assert full["modules"] == ("M1", "M3") and full["reasoning_effort"] == "low"
    for bad in ({"modules": ["M9"]}, {"nope": 1}, {"temperature": -1}, {"top_p": 0},
                {"max_new_tokens": 0}, {"reasoning_effort": "max"}):
        with pytest.raises(ValueError):
            validate_llm_fixed(bad)
    with pytest.raises(ValueError, match="M5"):
        load_cause_schema(None, ["a", "b"]).system_prompt(modules=("M5",))


def test_optimize_fields_that_do_not_apply_are_refused():
    for opt in (Optimize(n_trials=5), Optimize(cv=False), Optimize(backend="ray"),
                Optimize(pruning=True), Optimize(extra={"ray_address": "auto"}),
                Optimize(space={"temperature": ("float", 0.0, 1.0)}),
                Optimize(space={"learning_rate": ("categorical", [1])})):
        with pytest.raises(ValueError):
            resolve_llm_hyperparams(opt)
    kind, first, search, grid = resolve_llm_hyperparams(Optimize())
    assert kind == "search" and len(grid) == 5 and grid[0]["modules"] == tuple(LIVE)
    _, _, _, grid = resolve_llm_hyperparams(Optimize(extra={"modules": "all_subsets"}))
    assert len(grid) == 16
    _, _, _, grid = resolve_llm_hyperparams(Optimize(
        extra={"modules": [["M2"], ["M2", "M3"]]},
        space={"reasoning_effort": ("categorical", ["low", "high"])}))
    assert len(grid) == 4 and {g["reasoning_effort"] for g in grid} == {"low", "high"}


def test_selection_runs_the_grid_on_the_dev_slice_and_picks_one(df, causes, tmp_path):
    backend = FakeBackend(make_responder(causes))
    out = _run(df, causes, tmp_path, hyperparams=Optimize(metric="f1_macro"), dev_size=0.25,
               backend_instance=backend, evaluate_on="test")
    root = Path(out["output_dir"])
    trials = pd.read_csv(root / "hpo/hpo_trials.csv")
    assert len(trials) == 5 and trials["value"].notna().all()
    best = json.loads((root / "hpo/best_hyperparams.json").read_text())
    assert set(best) == {"modules", "reasoning_effort", "temperature", "top_p", "max_new_tokens"}
    assert out["validation"]["source"] == "dev_slice"
    assert out["validation"]["scores"]["f1_macro"] == pytest.approx(trials["value"].max())
    assert (root / "predictions/predictions_top1.csv").is_file()
    systems = {c["system"] for c in backend.calls}
    assert len(systems) == 5
    # the test pass used the chosen configuration
    assert out["train_metadata"]["hyperparams"]["modules"] == best["modules"]


def test_reasoning_effort_is_refused_where_unsupported(df, causes, tmp_path):
    backend = FakeBackend(make_responder(causes), reasoning_effort=False)
    with pytest.raises(ValueError, match="reasoning effort"):
        _run(df, causes, tmp_path, backend_instance=backend)
    out = _run(df, causes, tmp_path, backend_instance=backend,
               hyperparams={"reasoning_effort": None})
    assert out["train_metadata"]["hyperparams"]["reasoning_effort"] is None


# --- preflight pair ----------------------------------------------------------------

def test_preflight_accepts_what_runs_and_rejects_what_fails(df, causes, tmp_path):
    good = dict(task="llm", data=df, label_col=LABEL, text_col="narrative", id_col="id",
                cause_schema="who_target_list_adult", context="x", output_dir=tmp_path,
                hyperparams={"modules": ["M1", "M2"]})
    report = mv.preflight(**good)
    assert report["ok"], report["problems"]
    assert report["facts"]["sections"] == "narrative"
    assert "cause schema" in report["facts"] and "hyperparameters" in report["facts"]
    mv.run(**good, model="fake", backend_instance=FakeBackend(make_responder(causes)))

    bad = dict(good, hyperparams=Optimize(n_trials=3), output_dir=tmp_path / "bad",
               evaluate_on="test")
    report = mv.preflight(**bad)
    assert not report["ok"] and "does not apply to task='llm'" in "\n".join(report["problems"])
    with pytest.raises(ValueError, match="does not apply"):
        mv.run(**bad, model="fake", backend_instance=FakeBackend(make_responder(causes)))

    report = mv.preflight(**dict(good, cause_schema={"causes": {"Pneumonia": {}}}))
    assert not report["ok"] and "not in the cause schema" in "\n".join(report["problems"])
    report = mv.preflight(**dict(good, evaluate_on="dev"))
    assert "dev slice" in report["facts"] and "evaluated on" in report["facts"]


# --- backends and responses ------------------------------------------------------------

def test_backend_resolution():
    assert resolve_llm_backend("ollama:gpt-oss:20b") == ("ollama", "gpt-oss:20b")
    assert resolve_llm_backend("openai/gpt-oss-20b") == ("transformers", "openai/gpt-oss-20b")
    assert resolve_llm_backend("gpt-oss:20b", "ollama") == ("ollama", "gpt-oss:20b")
    with pytest.raises(ValueError):
        resolve_llm_backend("ollama:x", "transformers")


def test_harmony_and_json_extraction():
    text = ('<|channel|>analysis<|message|>thinking...<|end|><|start|>assistant'
            '<|channel|>final<|message|>{"a": 1}<|return|>')
    content, reasoning = split_harmony(text)
    assert content == '{"a": 1}' and reasoning == "thinking..."
    assert split_harmony("plain") == ("plain", None)
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Sure: {"a": 3} done') == {"a": 3}
    assert extract_json("nothing") is None
    ok = {"ranked_causes": [{"cause": "A", "probability": 0.5}], "confidence": "high",
          "rationale": "r", "evidence": [{"quote": "q", "source": "narrative",
                                          "polarity": "present", "supports": "supports"}]}
    assert valid_response(ok, ["A", "B"], sources=["narrative"])
    assert not valid_response(ok, ["A", "B"], sources=["questionnaire"])
    dup = dict(ok, ranked_causes=[{"cause": "A", "probability": 0.5}] * 2)
    assert not valid_response(dup, ["A", "B"], sources=["narrative"])


def test_predict_from_pretrained_refuses_an_llm_run(df, causes, tmp_path):
    out = _run(df, causes, tmp_path)
    with pytest.raises(ValueError, match="no trained artifact"):
        mv.predict_from_pretrained(out["output_dir"], df, text_col="narrative")


# --- CLI ----------------------------------------------------------------------------

def test_cli_flags_map_to_run_arguments(monkeypatch, tmp_path, df):
    from multimodalva import cli

    seen = {}
    monkeypatch.setattr("multimodalva.runner.run", lambda **cfg: seen.update(cfg))
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    cli.main(["run", "--task", "llm", "--data", str(csv), "--label-col", LABEL,
              "--text-col", "narrative", "--model", "ollama:gpt-oss:20b",
              "--context", "ctx", "--modules", "M1,M3", "--reasoning-effort", "none",
              "--section-markers", "narrative=none,questionnaire=====",
              "--evaluate-on", "dev", "--dev-size", "0.3", "--output-dir", str(tmp_path)])
    assert seen["hyperparams"] == {"modules": ["M1", "M3"], "reasoning_effort": None}
    assert seen["section_markers"] == {"narrative": None, "questionnaire": "===="}
    assert seen["evaluate_on"] == "dev" and seen["dev_size"] == 0.3 and seen["context"] == "ctx"
    with pytest.raises(SystemExit):
        cli.main(["run", "--task", "llm", "--data", str(csv), "--label-col", LABEL,
                  "--text-col", "narrative", "--optimize", "--modules", "M1"])


def test_cli_llm_prompt_and_schema(tmp_path, df, capsys):
    from multimodalva import cli

    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    assert cli.main(["llm-prompt", "--data", str(csv), "--label-col", LABEL,
                     "--cause-schema", "who_target_list_adult", "--modules", "M3"]) == 0
    printed = capsys.readouterr().out
    assert "ALLOWED CAUSES, WHAT EACH ONE COVERS" in printed and "STUDY CONTEXT" not in printed
    out = tmp_path / "schema.json"
    assert cli.main(["llm-schema", "--data", str(csv), "--label-col", LABEL, "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert set(doc["causes"]) == set(df[LABEL].unique()) and "_who_target_list" in doc
