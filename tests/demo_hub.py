"""Demos for publishing and re-using trained models via the Hugging Face Hub.

Four self-contained modes:

  1. push       Train a tiny model on synthetic VA data, then publish it to the
                Hub. Also shows the standalone util for pushing an existing run.
  2. classify   Load a published model from the Hub and classify a narrative
                directly (it already has a cause-of-death classification head).
  3. finetune   Load a published model from the Hub as the starting checkpoint
                and continue fine-tuning on your own VA data.
  4. sample-data  Write tiny, class-balanced synthetic CSVs that show the exact
                  input schema the package expects — usable for smoke tests.

Usage:
    # 4. Make sample data (no model download, fully offline) — start here:
    python tests/demo_hub.py sample-data --output-dir tests/sample_data
    #    WHO 2016 ODK schema (Id10xxx codes, broad cause grouping labels, instrument-modeled):
    python tests/demo_hub.py sample-data --schema who2016 --output-dir tests/sample_data

    # 1. Train + push (needs `huggingface-cli login` or --hub-token):
    python tests/demo_hub.py push --repo-id your-org/va-bert-demo

    #    Push an EXISTING run directory (any past run), no retraining:
    python tests/demo_hub.py push --repo-id your-org/va-bert-demo \\
        --from-dir runs/demo_text/final

    # 2. Classify a narrative with a published model:
    python tests/demo_hub.py classify --repo-id your-org/va-bert-demo \\
        --text "male adult with cough, fast breathing and fever for 9 days"

    # 3. Continue fine-tuning a published model on your own data:
    python tests/demo_hub.py finetune --repo-id your-org/va-bert-demo \\
        --data tests/sample_data/va_sample.csv --output-dir runs/va_finetuned

Notes:
    - All training data here is SYNTHETIC (see demo_utils.make_sample_va_df).
    - `sample-data` and `classify` need no training; `classify` downloads the repo.
    - Publishing requires authentication: run `huggingface-cli login` once, or
      pass --hub-token. Use --public to make the repo public (default: private).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from demo_utils import ensure_dir, load_data, make_sample_va_df, make_who2016_va_df


# --- Mode 4: sample data --------------------------------------------------------
def make_sample_data(args: argparse.Namespace) -> None:
    """Write synthetic CSVs showing the expected input schema."""
    out = ensure_dir(args.output_dir)

    if args.schema == "who2016":
        # Larger, instrument-modeled dataset using WHO 2016 ODK Id10xxx codes +
        # broad cause grouping cause labels — pairs with utils/qdesc_who2016.csv.
        df = make_who2016_va_df(n_samples=args.n_samples)
        path = out / "va_who2016_sample.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {len(df)} rows × {df.shape[1]} columns (WHO 2016 ODK schema).\n")
        print("Schema (pairs with multimodalva/utils/qdesc_who2016.csv):")
        print("  id             : fake record id")
        print("  cause_of_death : label (broad cause grouping level)")
        print("  narrative      : synthetic free-text VA narrative")
        print("  sex, age_group : demographics (male/female; adult/child/neonate)")
        print("  Id10xxx        : WHO 2016 ODK indicators, yes/no (tabular features)\n")
        print(df["cause_of_death"].value_counts().to_string())
        print(f"\nCSV: {path}")
        print("\nSmoke-test it (pass the WHO2016 qdesc to data fusion):")
        print(f"  python tests/demo_text_classification.py run --data {path} "
              "--label cause_of_death --text-col narrative --exclude-cols id")
        print(f"  python tests/demo_tabular_classification.py run --data {path} "
              "--label cause_of_death --exclude-cols id,narrative")
        return

    # Class-balanced InterVA i-code-style sample using InterVA i-code variables —
    # pairs with the default utils/qdesc.csv (auto-loaded by data fusion).
    df = make_sample_va_df(n_per_class=args.n_per_class)
    multimodal_path = out / "va_sample.csv"
    text_only_path = out / "va_sample_text_only.csv"
    df.to_csv(multimodal_path, index=False)
    df[["id", "cause_of_death", "narrative"]].to_csv(text_only_path, index=False)

    print(f"Wrote {len(df)} rows × {df.shape[1]} columns (InterVA i-code i-code schema).\n")
    print("Schema (pairs with the default multimodalva/utils/qdesc.csv):")
    print("  id             : fake record id")
    print("  cause_of_death : label (broad cause grouping level)")
    print("  narrative      : synthetic free-text VA narrative")
    print("  i019a/i019b    : sex; i022x : age band (binary demographics)")
    print("  i147o, i153o...: InterVA indicators, y/n (tabular features)\n")
    print(df["cause_of_death"].value_counts().to_string())
    print(f"\nMultimodal CSV : {multimodal_path}")
    print(f"Text-only CSV  : {text_only_path}")
    print("\nSmoke-test it:")
    print(f"  python tests/demo_text_classification.py run --data {multimodal_path} "
          "--label cause_of_death --text-col narrative --exclude-cols id")
    print(f"  python tests/demo_tabular_classification.py run --data {multimodal_path} "
          "--label cause_of_death --exclude-cols id,narrative")


# --- Mode 1: train + push -------------------------------------------------------
def push(args: argparse.Namespace) -> None:
    """Train a tiny model on synthetic data and push it, or push an existing dir."""
    if args.from_dir:
        # Standalone util — publish any saved run directory (no retraining).
        from multimodalva.utils import push_to_hub

        url = push_to_hub(
            args.from_dir,
            args.repo_id,
            private=not args.public,
            token=args.hub_token,
            model_kind=args.model_kind,
        )
        print(f"Published existing model dir {args.from_dir} -> {url}")
        return

    # Train a small model on synthetic data, then push via the run() flag.
    from multimodalva.text import TextClassifier

    df, _ = load_data("demo", "cause", text_col="narrative", n_samples=60)
    clf = TextClassifier(model_name=args.model, output_dir=ensure_dir(args.output_dir))
    results = clf.run(
        df=df,
        text_col="narrative",
        label_col="cause",
        max_length=128,
        hyperparams={"epochs": 1, "batch_size": 4, "learning_rate": 2e-5},
        batch_size=8,
        top_k=3,
        # --- publishing ---
        push_to_hub=True,
        hub_repo_id=args.repo_id,
        hub_private=not args.public,
        hub_token=args.hub_token,
    )
    print(f"Trained and published -> {results['hub_url']}")


# --- Mode 2: load + classify ----------------------------------------------------
def classify(args: argparse.Namespace) -> None:
    """Load a published model from the Hub and classify a single narrative."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    text = args.text or (
        "male adult with cough, fast breathing, and fever for nine days"
    )
    tok = AutoTokenizer.from_pretrained(args.repo_id, token=args.hub_token)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.repo_id, token=args.hub_token
    ).eval()

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(-1)[0]

    order = probs.argsort(descending=True)[: args.top_k]
    print(f"Narrative: {text}\n")
    print(f"Predicted cause of death (top-{args.top_k}):")
    for rank, idx in enumerate(order, 1):
        label = model.config.id2label[int(idx)]
        print(f"  {rank}. {label:<32} {float(probs[idx]):.3f}")


# --- Mode 3: load + continue fine-tuning ---------------------------------------
def finetune(args: argparse.Namespace) -> None:
    """Use a published model as the base checkpoint and fine-tune on your data.

    Passing the Hub repo id as ``model_name`` makes ``from_pretrained`` pull the
    fine-tuned encoder. ``ignore_mismatched_sizes=True`` lets you fine-tune even
    when your cause set differs from the published model's (a fresh head is
    initialised); it is a no-op when the cause sets match.
    """
    from multimodalva.text import TextClassifier

    if args.data == "demo":
        df, _ = load_data("demo", "cause", text_col="narrative", n_samples=60)
    else:
        df, _ = load_data(args.data, "cause", text_col="narrative")

    print(f"Fine-tuning from published model '{args.repo_id}' on {len(df)} rows.")
    clf = TextClassifier(model_name=args.repo_id, output_dir=ensure_dir(args.output_dir))
    results = clf.run(
        df=df,
        text_col="narrative",
        label_col="cause",
        max_length=128,
        hyperparams={
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 2e-5,
            # Allow a different cause set than the published model; reinitialise head.
            "ignore_mismatched_sizes": True,
        },
        batch_size=8,
        top_k=3,
        hub_token=args.hub_token,  # forwarded only if you also set push_to_hub=True
    )
    top1 = results["predictions"].top1
    acc = (top1["true_label"] == top1["predicted_label"]).mean()
    print(f"Fine-tuned model saved to {results['output_dir']} (top-1 acc {acc:.3f}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="sample-data",
        choices=["push", "classify", "finetune", "sample-data"],
        help="Demo to run (default: sample-data).",
    )
    parser.add_argument("--repo-id", default=None, dest="repo_id",
                        help="Hugging Face repo id, e.g. 'your-org/va-bert-demo'.")
    parser.add_argument("--from-dir", default=None, dest="from_dir",
                        help="(push) Publish this existing saved model dir instead "
                             "of training (e.g. runs/demo_text/final).")
    parser.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT",
                        help="(push) Base checkpoint to fine-tune (default: Bio_ClinicalBERT).")
    parser.add_argument("--model-kind", default="text", dest="model_kind",
                        choices=["text", "data_fusion"],
                        help="(push --from-dir) Card type; data_fusion adds the "
                             "fused-input note (default: text).")
    parser.add_argument("--text", default=None,
                        help="(classify) Narrative to classify.")
    parser.add_argument("--data", default="demo",
                        help="(finetune) CSV path or 'demo' for synthetic data.")
    parser.add_argument("--hub-token", default=None, dest="hub_token",
                        help="HF auth token (else uses cached `huggingface-cli login`).")
    parser.add_argument("--public", action="store_true",
                        help="(push) Make the repo public (default: private).")
    parser.add_argument("--schema", default="simple", choices=["simple", "who2016"],
                        help="(sample-data) 'simple' = tiny human-readable multimodal "
                             "CSV; 'who2016' = instrument-modeled dataset with WHO 2016 ODK "
                             "Id10xxx codes + broad cause grouping labels (default: simple).")
    parser.add_argument("--n-per-class", type=int, default=6, dest="n_per_class",
                        help="(sample-data, simple) Rows per cause (default: 6; with 11 "
                             "causes this keeps a 0.2 test split stratifiable).")
    parser.add_argument("--n-samples", type=int, default=400, dest="n_samples",
                        help="(sample-data, who2016) Total rows (default: 400).")
    parser.add_argument("--max-length", type=int, default=128, dest="max_length",
                        help="Tokeniser max sequence length (default: 128).")
    parser.add_argument("--top-k", type=int, default=3, dest="top_k",
                        help="Top-k predictions to show (default: 3).")
    parser.add_argument("--output-dir", default="runs/demo_hub", dest="output_dir",
                        help="Output directory (default: runs/demo_hub).")

    args = parser.parse_args()

    if args.mode == "sample-data":
        make_sample_data(args)
        return

    if args.mode in ("push", "classify", "finetune") and not args.repo_id:
        parser.error(f"mode '{args.mode}' requires --repo-id.")

    if args.mode == "push":
        push(args)
    elif args.mode == "classify":
        classify(args)
    else:
        finetune(args)


if __name__ == "__main__":
    main()
