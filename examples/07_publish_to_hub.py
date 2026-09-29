#!/usr/bin/env python3
"""
Publish a trained text or data-fusion model to the Hugging Face Hub.

Text and data-fusion runs are saved in standard Hugging Face format — the cause
labels live in ``config.json`` and LoRA adapters are merged into the base
weights — so anything published this way reloads with plain ``transformers``.

    python examples/07_publish_to_hub.py                      # check only, uploads nothing
    python examples/07_publish_to_hub.py your-org/va-bert-cod # actually publishes

The second form needs `huggingface-cli login` first. Publishing is private by
default; pass ``private=False`` below to make the repository public.

Tabular, voting and stacking runs are not in a Hub-native format. Share those as
a run directory — see `08_predict_with_a_trained_model.py` for loading one back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUN_DIR = Path("runs/text")          # a finished text or data-fusion run


def describe(final_dir: Path) -> None:
    """What would be uploaded, and whether the directory looks publishable."""
    required = ["config.json"]
    weights = ["model.safetensors", "pytorch_model.bin"]
    missing = [f for f in required if not (final_dir / f).is_file()]
    if not any((final_dir / w).is_file() for w in weights):
        missing.append("model.safetensors (or pytorch_model.bin)")
    print(f"Source: {final_dir}")
    if missing:
        print(f"  NOT publishable — missing: {', '.join(missing)}")
        print("  Only text and data-fusion runs are in Hugging Face format.")
        return
    cfg = json.loads((final_dir / "config.json").read_text())
    labels = cfg.get("id2label") or {}
    print(f"  architecture : {(cfg.get('architectures') or ['?'])[0]}")
    print(f"  causes       : {len(labels)}")
    if labels:
        shown = list(labels.values())[:3]
        print(f"                 {', '.join(map(str, shown))}"
              + (" ..." if len(labels) > 3 else ""))
    total = sum(f.stat().st_size for f in final_dir.rglob("*") if f.is_file())
    print(f"  upload size  : {total / 1e6:.0f} MB")
    if all(str(v).startswith("LABEL_") for v in labels.values()):
        print("  WARNING: id2label is LABEL_0, LABEL_1, ... — the causes were not "
              "recorded, and anyone reloading this gets meaningless labels.")


def main() -> None:
    final_dir = RUN_DIR / "final" if (RUN_DIR / "final").is_dir() else RUN_DIR
    if not final_dir.is_dir():
        print(f"No run at {RUN_DIR}. Train one first, for example:\n"
              "    multimodalva run --task text --data clean.csv "
              "--label-col cause \\\n"
              "        --text-col narrative --output-dir runs/text")
        return

    describe(final_dir)

    if len(sys.argv) < 2:
        print("\nNothing was uploaded. To publish:\n"
              "    huggingface-cli login\n"
              f"    python {sys.argv[0]} your-org/va-bert-cod")
        return

    from multimodalva.utils import push_to_hub

    repo_id = sys.argv[1]
    print(f"\nPublishing to {repo_id} (private) ...")
    url = push_to_hub(
        final_dir, repo_id,
        private=True,              # False to publish publicly
        model_kind="text",         # "data_fusion" records the fused-input format
        # metrics={"f1_macro": 0.62},   # shown on the generated model card
    )
    print(f"Done: {url}")
    print("\nReload it anywhere:\n"
          "    from transformers import AutoModelForSequenceClassification\n"
          f"    m = AutoModelForSequenceClassification.from_pretrained('{repo_id}')\n"
          "or keep fine-tuning it:\n"
          "    TextClassifier(model_name='" + repo_id + "').run(\n"
          "        df=my_df, text_col='narrative', label_col='cause',\n"
          "        hyperparams={'ignore_mismatched_sizes': True})")


if __name__ == "__main__":
    main()
