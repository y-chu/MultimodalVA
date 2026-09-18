# Sample data

These files are **synthetic**. Nothing here comes from a real verbal autopsy.
Narratives are built from fixed templates, and symptom answers are drawn at random
from cause-conditioned probabilities. No real record is reproduced, in whole or in part.

They exist so the tests and the examples in the README can run without any data of
your own.

## Regenerating them

Each file is reproducible from `multimodalva.datasets`:

```python
from multimodalva import data

data("va_sample", n_per_class=6).to_csv("va_sample.csv", index=False)
data("va_sample_text_only", n_per_class=6).to_csv("va_sample_text_only.csv", index=False)
data("va_who2016").to_csv("va_who2016_sample.csv", index=False)
```

The generators are seeded, so these commands reproduce the files byte for byte.
Note the `n_per_class=6` — the default is 4, which gives a smaller file.

## What is in each file

| file | rows | tabular coding | pairs with |
|---|---|---|---|
| `va_sample.csv` | 66 | InterVA i-codes (`y`/`n`) | `multimodalva/utils/qdesc.csv` |
| `va_sample_text_only.csv` | 66 | none — id, label, narrative | (text pipeline) |
| `va_who2016_sample.csv` | 400 | WHO 2016 ODK `Id10xxx` (`yes`/`no`) | `multimodalva/utils/qdesc_who2016.csv` |

Cause labels are at the `broad cause grouping` level.

To see every dataset the package ships, run `multimodalva list-datasets`.
