"""Prompt a language model instead of training one — ``task="llm"``.

On the built-in synthetic data:

  1. define your own cause list, with a definition per cause — the schema;
  2. print the system prompt that schema produces, to read before spending
     GPU time;
  3. write the schema to a file, which is how a real project keeps it;
  4. run the pipeline on a few records (only with --run: the default model is
     a 13 GB download that needs a GPU or a large-memory machine).

    python examples/09_prompt_a_language_model.py              # steps 1-3, no model
    python examples/09_prompt_a_language_model.py --run        # + openai/gpt-oss-20b
    python examples/09_prompt_a_language_model.py --run --model ollama:gpt-oss:20b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import multimodalva as mv
from multimodalva.llm import build_system_prompt, load_cause_schema

# ---------------------------------------------------------------------------
# 1. Your cause list. The labels are whatever your data uses; the definitions
#    are what the model is told each label covers (module M3). analysis_va_codes
#    and icd10_scope are optional, and master_category maps a label onto the
#    WHO-derived category the cross-cause rules (M4) are written for.
#    `multimodalva llm-schema --data my.csv --label-col cause` writes this as
#    a template with one entry per label found in the data.
# ---------------------------------------------------------------------------
CAUSE_SCHEMA = {
    "_meta": {
        "population_context": "Deaths from a rural district where infectious "
                              "disease and injuries are the main burdens.",
        "age_group": "adult",
    },
    "causes": {
        "Infection": {
            "definition": "Any infectious disease as the underlying cause, "
                          "including HIV/AIDS, tuberculosis, pneumonia and diarrhoea.",
        },
        "Cardiovascular": {
            "definition": "Heart disease and stroke.",
            "icd10_scope": "I00-I99",
        },
        "Cancer": {
            "definition": "Any malignant neoplasm.",
            "icd10_scope": "C00-C97",
            "master_category": "Neoplasms",
        },
        "Injury": {
            "definition": "Death from an external cause: transport accident, "
                          "assault, suicide, other injury.",
        },
        "Other": {
            "definition": "Any underlying cause not covered above.",
        },
    },
}

# The synthetic data uses finer labels; fold them into the five above so the
# example has a cause list of its own.
GROUPING = {
    "HIV/AIDS": "Infection", "Pulmonary tuberculosis": "Infection",
    "Pneumonia": "Infection", "Diarrhea": "Infection",
    "Cardiac disease": "Cardiovascular", "Cerebrovascular disease": "Cardiovascular",
    "Neoplasms": "Cancer",
    "Traffic / transport accident": "Injury", "Assault": "Injury",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Run the model (step 4).")
    parser.add_argument("--model", default="openai/gpt-oss-20b",
                        help="A Hugging Face Hub id (default), or ollama:<tag> for a "
                             "model served by a local ollama server.")
    parser.add_argument("--n", type=int, default=5, help="Records to run (dev slice).")
    parser.add_argument("--output-dir", default="runs/llm_demo")
    args = parser.parse_args()

    df = mv.data("va_sample_text_only", n_per_class=8)
    df["cause"] = df["cause_of_death"].map(GROUPING).fillna("Other")
    causes = sorted(df["cause"].unique())

    # 2. The prompt this schema produces. Reading it is the cheapest check that
    #    the definitions say what you mean.
    spec = load_cause_schema(CAUSE_SCHEMA, causes)
    print(f"{len(causes)} causes: {causes}\n")
    print(spec.cause_list_block(scope=True) + "\n")          # the part your schema wrote
    prompt = build_system_prompt(causes, cause_schema=CAUSE_SCHEMA)
    print(f"(the whole system prompt is {len(prompt):,} characters; "
          "`multimodalva llm-prompt --cause-schema <file>` prints it)\n")

    # 3. Keep the schema as a file beside the data; pass its path as cause_schema=.
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema_path = out / "cause_schema.json"
    schema_path.write_text(json.dumps(CAUSE_SCHEMA, indent=2))
    print(f"Wrote {schema_path}\n")

    if not args.run:
        print("Pass --run to prompt the model on a few records.")
        return 0

    # 4. Run it on a dev slice of the training rows; the test rows are not read.
    result = mv.run(
        task="llm", data=df, label_col="cause", text_col="narrative", id_col="id",
        model=args.model, cause_schema=str(schema_path),
        evaluate_on="dev", dev_size=args.n,
        output_dir=out,
    )
    print(result["predictions"].top1.head(args.n).to_string())
    print("\nDev scores (selection data, not performance):", result["dev_scores"])
    print(f"\nEverything is under {out}: dev/predictions_*.csv, dev/llm_metadata.csv "
          "(ranking, confidence, rationale, evidence per record), training_metadata.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
