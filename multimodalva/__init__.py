"""MultimodalVA: cause-of-death classification from verbal autopsy data.

Everything runs through one function. Pick a task, point it at a table, say
which column holds the cause of death:

    import multimodalva as mv

    df = mv.data("va_sample")            # or your own DataFrame / CSV path
    out = mv.run(
        task="text",
        data=df,
        label_col="cause_of_death",
        text_col="narrative",
        output_dir="runs/first",
    )
    out["predictions"].top1.head()

``mv.SUPPORTED_TASKS`` lists the six tasks: ``text``, ``tabular``,
``data_fusion``, ``feature_fusion``, ``voting``, ``stacking``. Swapping the
``task`` is usually the only change needed — tabular tasks want ``features=``
instead of ``text_col=``, and the ensembles want both.

``mv.data(name)`` returns a small synthetic DataFrame shaped like the input the
package expects; ``mv.list_datasets()`` names them. Nothing here is real data.

The same thing from a shell, with a YAML config instead of keyword arguments:

    multimodalva run --task text --data clean.csv --label-col cause \
        --text-col narrative --output-dir runs/first
    multimodalva list-datasets
    multimodalva list-models

Every task returns the same ``PredictionResult`` (``top1``, ``full``, ``topk``),
so anything in ``multimodalva.results`` accepts the output of any pipeline.

Subpackages, if you want the individual steps rather than the whole pipeline:
``text``, ``tabular``, ``ensemble``, ``results``, ``utils``. Importing this
module does not pull torch or transformers — those load when a pipeline runs.
"""

__version__ = "0.1.0"

# Synthetic example datasets — preview the input shape the package expects:
#   from multimodalva import data, list_datasets
#   list_datasets()            # {name: description}
#   df = data("va_sample")     # a ready-to-use DataFrame
# Lightweight (pandas/numpy only) — importing this does not pull the transformer stack.
from .datasets import data, list_datasets  # noqa: E402

# One-call, config-driven entry point for every pipeline. Heavy dependencies
# (torch/transformers/autogluon) are imported lazily inside run(), so importing
# this name does not pull the transformer stack.
from .runner import run, SUPPORTED_TASKS  # noqa: E402
# Same config as run(), checked without training anything.
from .runner import preflight  # noqa: E402
from .utils.optimize_config import Optimize  # noqa: E402
# Score new data with an already-trained model (local run, Hub id or URL).
# torch/transformers load only when a text model is used.
from .inference import (  # noqa: E402
    predict_from_pretrained,
    predict_ensemble_from_pretrained,
)

__all__ = ["data", "list_datasets", "run", "preflight", "Optimize",
           "SUPPORTED_TASKS",
           "predict_from_pretrained", "predict_ensemble_from_pretrained"]
