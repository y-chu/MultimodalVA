#!/usr/bin/env python3
"""
Score new records with runs that have already finished — one call per pipeline.

A finished run assigns causes to new records without retraining. The call is the
same whatever trained the run: the artifact records its own task, so you give it
the source directory and the columns that pipeline reads.

    single model  ->  predict_from_pretrained(source, df, ...)
    voting/stacking -> predict_ensemble_from_pretrained(source, df, ...)

Which columns each pipeline needs:

    text, data_fusion   text_col=
    tabular             feature_cols= (read from the artifact if it recorded them)
    feature_fusion      text_col= and feature_cols=
    voting, stacking    whatever the base models need

Run it as-is (no GPU, no network, no data of your own, ~1 minute):

    python examples/08_predict_with_a_trained_model.py

On your own data, change two things: point ``source`` at your finished run, and
read your records with ``pd.read_csv`` instead of ``data()``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import multimodalva as mv
from multimodalva import (
    data, predict_ensemble_from_pretrained, predict_from_pretrained,
)
from multimodalva.utils.metrics import score_predictions

SPECS = [{"model_name": "random_forest"}, {"model_name": "naive_bayes"}]


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="mmva_predict_demo_"))
    try:
        train_df = data("va_sample", n_per_class=20, seed=0)
        features = [c for c in train_df.columns
                    if c not in ("cause_of_death", "narrative", "id")]
        common = dict(data=train_df, label_col="cause_of_death",
                      features=features, id_col="id")

        print("Training three runs to load back (a minute) ...")
        mv.run(task="tabular", model="random_forest",
               output_dir=workdir / "tabular", **common)
        mv.run(task="voting", tabular_models=SPECS,
               output_dir=workdir / "voting", **common)
        mv.run(task="stacking", tabular_models=SPECS,
               init_kwargs={"n_folds": 3},
               output_dir=workdir / "stacking", **common)

        # New records. Drop the label to imitate records with no cause recorded.
        new_df = data("va_sample", n_per_class=5, seed=99)
        unlabelled = new_df.drop(columns=["cause_of_death"])

        # --- 1. a single model --------------------------------------------
        print("\n1. Single model — predictions only")
        res = predict_from_pretrained(workdir / "tabular", unlabelled, id_col="id")
        print(res.top1.head(3).to_string(index=False))
        print(f"   {len(res.top1)} records, {len(res.id2label)} causes, "
              f"{res.full.shape[1]} probability columns")

        # --- 2. the same call, with the true cause available ---------------
        print("\n2. Single model — labelled records, so performance too")
        scored = predict_from_pretrained(workdir / "tabular", new_df,
                                         label_col="cause_of_death", id_col="id")
        for metric in ("accuracy", "f1_macro", "csmf_accuracy"):
            print(f"   {metric:16} {score_predictions(scored.top1, metric):.3f}")

        # --- 3. ensembles: one entry point, combined the way they trained ---
        for task in ("voting", "stacking"):
            print(f"\n3. {task} — combined the way that run combined it")
            out = predict_ensemble_from_pretrained(
                workdir / task, new_df, label_col="cause_of_death", id_col="id")
            print(f"   combiner: {out.checks.combiner or 'soft vote'}, "
                  f"{len(out.checks.base_models)} base models")
            print(f"   accuracy {score_predictions(out.top1, 'accuracy'):.3f}")

        # --- 4. what it detected -------------------------------------------
        print("\n4. What it detected about the model and your input")
        for field, value in vars(scored.checks).items():
            if value not in (None, [], {}) and "dir" not in field and field != "source":
                print(f"   {field:22} {value}")

        print("""
Text pipelines take the same call, plus the narrative column:

    predict_from_pretrained("runs/text", new_df, text_col="narrative")

The command line does all of the above, reading the task from the artifact:

    multimodalva predict --source runs/tabular --data new.csv --id-col id \\
        --output-dir preds/
    multimodalva predict --source runs/stacking --data new.csv --dry-run
""")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
