"""
Synthetic example datasets — so users can preview the input the package expects.

Quick look (R-style ``data()``):

    from multimodalva import data, list_datasets

    list_datasets()              # {name: description}
    df = data("va_sample")       # InterVA i-code-style InterVA i-code sample
    df = data("va_who2016")      # WHO 2016 ODK Id10xxx sample
    df = data("va_sample", n_per_class=10, seed=0)   # kwargs forwarded to the generator

All data is fully SYNTHETIC (narratives generated from original templates; symptom
responses drawn from cause-conditioned probabilities). No real records are reproduced.
Cause labels are broad cause grouping-level.

| name                  | tabular coding              | pairs with                          |
|-----------------------|-----------------------------|-------------------------------------|
| ``va_sample``         | InterVA i-codes (y/n)       | ``multimodalva/utils/qdesc.csv``    |
| ``va_sample_text_only`` | id + label + narrative    | (text pipeline)                     |
| ``va_who2016``        | WHO 2016 ODK Id10xxx (yes/no)| ``multimodalva/utils/qdesc_who2016.csv`` |
| ``va_demo``           | human-readable (fever/cough)| (quick toy)                         |
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "data",
    "list_datasets",
    "make_demo_va_df",
    "make_sample_va_df",
    "make_who2016_va_df",
    "CAUSES",
    "ISAMPLE_COLS",
    "WHO2016_COLS",
]


CAUSES = [
    "HIV/AIDS",
    "Pneumonia",
    "Traffic / transport accident",
    "Diabetes",
]


def _build_va_row(cause: str, rng: np.random.Generator) -> dict:
    """Build a single synthetic VA record for ``cause`` (shared schema)."""
    age = int(rng.integers(18, 85))
    sex = rng.choice(["female", "male"])
    fever = int(cause in {"HIV/AIDS", "Pneumonia"} or rng.random() < 0.08)
    cough = int(cause == "Pneumonia" or rng.random() < 0.10)
    weight_loss = int(cause == "HIV/AIDS" or rng.random() < 0.06)
    injury = int(cause == "Traffic / transport accident" or rng.random() < 0.03)
    polyuria = int(cause == "Diabetes" or rng.random() < 0.05)
    chest_pain = int(cause == "Traffic / transport accident" or rng.random() < 0.07)
    symptom_days = int(rng.integers(1, 40))

    if cause == "HIV/AIDS":
        narrative = (
            f"{sex} adult with prolonged fever, weight loss, weakness, and recurrent illness "
            f"for {symptom_days} days."
        )
    elif cause == "Pneumonia":
        narrative = (
            f"{sex} adult with cough, fast breathing, fever, and chest symptoms "
            f"for {symptom_days} days."
        )
    elif cause == "Traffic / transport accident":
        narrative = (
            f"{sex} adult involved in a road traffic injury with sudden collapse and chest pain."
        )
    else:
        narrative = (
            f"{sex} adult with excessive urination, thirst, weakness, and gradual decline "
            f"over {symptom_days} days."
        )

    return {
        "narrative": narrative,
        "age": age,
        "sex": sex,
        "fever": fever,
        "cough": cough,
        "weight_loss": weight_loss,
        "injury": injury,
        "polyuria": polyuria,
        "chest_pain": chest_pain,
        "symptom_days": symptom_days,
        "cause": cause,
    }


def make_demo_va_df(n_samples: int = 120, seed: int = 7) -> pd.DataFrame:
    """Create a small multimodal verbal-autopsy-like DataFrame."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(CAUSES, size=n_samples, p=[0.35, 0.25, 0.20, 0.20])
    return pd.DataFrame([_build_va_row(cause, rng) for cause in labels])


# ---------------------------------------------------------------------------
# InterVA i-code-style synthetic sample (InterVA "i"-code variables)
# ---------------------------------------------------------------------------
# Tabular columns use the InterVA "i"-code indicators with "y"/"n" values, as in
# the the source datasets, and pair with the default multimodalva/utils/
# qdesc.csv (auto-loaded by data fusion). Demographics are binary i-codes
# (i019a/i019b sex; i022x age band), so the package's prefix + demographics
# rendering works fully. Cause labels are broad cause grouping-level. Fully synthetic.

# Binary demographics i-codes (one sex + one age band set to "y" per row).
_ISAMPLE_DEMO = ["i019a", "i019b", "i022a", "i022b", "i022c", "i022d", "i022e", "i022g"]

# Clinical i-codes (symptom / diagnosis / injury / behavior) emitted as features.
_ISAMPLE_CLIN = [
    "i147o", "i152o", "i153o", "i155o", "i157o", "i159o", "i166o", "i168o",
    "i174o", "i181o", "i186o", "i188o", "i194o", "i204o", "i207o", "i208o",
    "i214o", "i217o", "i219o", "i225o", "i230o", "i233o", "i243o", "i245o",
    "i249o", "i253o", "i258o", "i259o", "i265o", "i268o", "i269o", "i270o",
    "i125o", "i127o", "i132o", "i133o", "i134o", "i135o", "i137o", "i141o",
    "i077o", "i079o", "i084o", "i090o", "i091o", "i092o", "i099o",
    "i411o", "i412o",
]
ISAMPLE_COLS = _ISAMPLE_DEMO + _ISAMPLE_CLIN  # full tabular feature set

# broad cause grouping cause -> (age_group, P(female), characteristic positive i-codes,
# narrative clause pool). Same style as make_who2016_va_df but i-coded.
_ISAMPLE_PROFILES: dict[str, dict] = {
    "HIV/AIDS": dict(age="adult", pf=0.5,
        pos=["i243o", "i153o", "i152o", "i181o", "i245o", "i127o", "i268o"],
        clauses=["had been losing weight for months", "had a persistent cough and night sweats",
                 "suffered repeated bouts of diarrhoea", "had tested positive for HIV earlier",
                 "grew steadily weaker and thinner"]),
    "Pulmonary tuberculosis": dict(age="adult", pf=0.45,
        pos=["i153o", "i155o", "i157o", "i152o", "i243o", "i147o", "i125o"],
        clauses=["coughed for weeks and sometimes brought up blood", "had drenching night sweats and fever",
                 "was told by a clinic it was tuberculosis", "lost much weight while coughing"]),
    "Cardiac disease": dict(age="adult", pf=0.5,
        pos=["i174o", "i168o", "i159o", "i249o", "i133o", "i132o"],
        clauses=["complained of chest pain and shortness of breath", "could not breathe lying flat",
                 "had swelling of both legs", "had a known heart condition"]),
    "Cerebrovascular disease": dict(age="adult", pf=0.5,
        pos=["i217o", "i214o", "i258o", "i259o", "i207o", "i141o", "i132o"],
        clauses=["suddenly lost consciousness one morning", "was paralysed on one side afterwards",
                 "had a severe headache before collapsing", "had high blood pressure for years"]),
    "Diabetes": dict(age="adult", pf=0.55,
        pos=["i270o", "i225o", "i243o", "i230o", "i134o"],
        clauses=["was always thirsty and passed urine often", "had a foot sore that would not heal",
                 "had been diagnosed with diabetes", "lost weight despite eating"]),
    "Neoplasms": dict(age="adult", pf=0.5,
        pos=["i253o", "i204o", "i243o", "i268o", "i137o", "i194o"],
        clauses=["had a growing lump doctors called a tumour", "had a swelling in the abdomen",
                 "wasted away over many months", "had been diagnosed with cancer"]),
    "Chronic pulmonary disease and asthma": dict(age="adult", pf=0.5,
        pos=["i159o", "i168o", "i153o", "i166o", "i135o"],
        clauses=["had longstanding breathing difficulty", "wheezed and struggled for air",
                 "had a chronic cough for years", "had asthma treated at the clinic"]),
    "Pneumonia": dict(age="adult", pf=0.5,
        pos=["i147o", "i153o", "i166o", "i159o", "i174o"],
        clauses=["developed a fever with a productive cough", "breathed fast and with difficulty",
                 "had chest pain and fever for a few days"]),
    "Diarrhea": dict(age="child", pf=0.5,
        pos=["i181o", "i188o", "i269o", "i147o", "i186o"],
        clauses=["had frequent watery stools for days", "vomited and could not keep fluids down",
                 "had sunken eyes from dehydration"]),
    "Traffic / transport accident": dict(age="adult", pf=0.4,
        pos=["i077o", "i079o", "i217o"],
        clauses=["was struck by a vehicle on the road", "died shortly after a road crash",
                 "was injured in a collision while travelling"]),
    "Assault": dict(age="adult", pf=0.25,
        pos=["i077o", "i090o", "i091o", "i092o", "i411o"],
        clauses=["was attacked and beaten by others", "was shot during a robbery",
                 "was stabbed in a fight", "died of injuries inflicted by someone else"]),
}


def _isample_age_band(age_group: str, rng: np.random.Generator) -> tuple[str, int]:
    """Return (age-band i-code, age years) for an age group."""
    if age_group == "neonate":
        return "i022g", 0
    if age_group == "child":
        age = int(rng.integers(1, 15))
        return ("i022e" if age <= 4 else "i022d"), age
    age = int(rng.integers(15, 85))
    band = "i022c" if age <= 49 else ("i022b" if age <= 64 else "i022a")
    return band, age


def make_sample_va_df(n_per_class: int = 4, seed: int = 11) -> pd.DataFrame:
    """Class-balanced synthetic VA sample in InterVA i-code style for smoke tests.

    Tabular columns are InterVA **"i"-code** indicators (``i019a``, ``i147o`` …)
    with ``"y"``/``"n"`` values, matching the InterVA i-code datasets and the default
    ``multimodalva/utils/qdesc.csv`` (auto-loaded by data fusion). Guarantees
    exactly ``n_per_class`` rows per cause so even a tiny sample stratifies.

    Columns: ``id``, ``cause_of_death`` (broad cause grouping), ``narrative``, then i-code
    demographics (``i019a/b`` sex, ``i022x`` age band) and clinical indicators.
    Fully synthetic — narratives are generated from original templates.
    """
    rng = np.random.default_rng(seed)
    causes = list(_ISAMPLE_PROFILES)
    base_p = 0.05
    plan = [(c, k) for c in causes for k in range(n_per_class)]
    # interleave so the file isn't blocked by cause
    order = rng.permutation(len(plan))

    rows: list[dict] = []
    for new_i, idx in enumerate(order):
        cause, _ = plan[idx]
        prof = _ISAMPLE_PROFILES[cause]
        age_group = prof["age"]
        sex = "female" if rng.random() < prof["pf"] else "male"
        band, age = _isample_age_band(age_group, rng)

        rec: dict = {"id": f"VA{new_i + 1:04d}", "cause_of_death": cause}
        # demographics: exactly one sex + one age band positive
        for c in _ISAMPLE_DEMO:
            rec[c] = "n"
        rec["i019a" if sex == "male" else "i019b"] = "y"
        rec[band] = "y"
        # clinical indicators
        pos = set(prof["pos"])
        for c in _ISAMPLE_CLIN:
            p = rng.uniform(0.7, 0.92) if c in pos else base_p
            rec[c] = "y" if rng.random() < p else "n"

        # narrative
        subj = ("He" if sex == "male" else "She") if age_group == "adult" else (
            "The child" if age_group == "child" else "The newborn")
        who = (f"a {age}-year-old {sex}" if age_group == "adult"
               else f"a {age}-year-old child" if age_group == "child"
               else "a newborn baby")
        clauses = list(prof["clauses"])
        rng.shuffle(clauses)
        k = int(rng.integers(2, min(3, len(clauses)) + 1))
        rec["narrative"] = (
            f"The deceased was {who}. {subj} {'; '.join(clauses[:k])}. "
            f"The family sought care before death."
        )
        rows.append(rec)

    cols = ["id", "cause_of_death", "narrative"] + ISAMPLE_COLS
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# WHO 2016 ODK synthetic generator (modeled on the a study site dataset shape)
# ---------------------------------------------------------------------------
# Tabular columns use WHO 2016 ODK indicator codes (Id10xxx) with "yes"/"no"
# values, pairing with multimodalva/utils/qdesc_who2016.csv.  Cause labels are
# broad cause grouping-level.  Everything below is SYNTHETIC — narratives are generated
# from original templates, not copied from any real dataset.

# Id10xxx symptom/diagnosis/injury/behavior columns emitted by the generator
# (all present in qdesc_who2016.csv).
WHO2016_COLS = [
    "Id10147", "Id10149", "Id10152", "Id10153", "Id10155", "Id10157", "Id10159",
    "Id10166", "Id10168", "Id10174", "Id10181", "Id10186", "Id10188", "Id10194",
    "Id10204", "Id10207", "Id10208", "Id10214", "Id10217", "Id10219", "Id10225",
    "Id10230", "Id10233", "Id10243", "Id10245", "Id10249", "Id10253", "Id10258",
    "Id10259", "Id10265", "Id10268", "Id10269", "Id10270",
    "Id10125", "Id10126", "Id10132", "Id10133", "Id10134", "Id10135", "Id10137",
    "Id10138", "Id10141",
    "Id10077", "Id10079", "Id10084", "Id10090", "Id10091", "Id10092", "Id10099",
    "Id10411", "Id10412",
    "Id10305", "Id10301", "Id10323",
    "Id10112", "Id10113", "Id10275", "Id10281", "Id10284", "Id10290", "Id10347",
    "Id10363",
]

# Per-cause profile: age group, P(female), characteristic positive indicators
# (high probability), and narrative clause fragments. broad cause grouping labels.
_WHO2016_PROFILES: dict[str, dict] = {
    "HIV/AIDS": dict(age="adult", pf=0.5,
        pos=["Id10243", "Id10153", "Id10152", "Id10181", "Id10245", "Id10126", "Id10268", "Id10149"],
        clauses=["had been losing weight for several months", "suffered from a persistent cough",
                 "had repeated bouts of diarrhoea and night sweats", "tested positive for HIV some years earlier",
                 "grew progressively weaker and thinner"]),
    "Pulmonary tuberculosis": dict(age="adult", pf=0.45,
        pos=["Id10153", "Id10155", "Id10157", "Id10152", "Id10243", "Id10147", "Id10125"],
        clauses=["coughed for many weeks, sometimes bringing up blood", "had drenching night sweats and a long-standing fever",
                 "had been told by a clinic she had tuberculosis", "lost a great deal of weight while coughing"]),
    "Cardiac disease": dict(age="adult", pf=0.5,
        pos=["Id10174", "Id10168", "Id10159", "Id10249", "Id10133", "Id10132"],
        clauses=["complained of chest pain and shortness of breath", "could not breathe well when lying flat",
                 "had swelling of both legs", "had a known heart condition", "collapsed after gripping his chest"]),
    "Cerebrovascular disease": dict(age="adult", pf=0.5,
        pos=["Id10217", "Id10214", "Id10258", "Id10259", "Id10207", "Id10141", "Id10132"],
        clauses=["suddenly lost consciousness one morning", "was left paralysed on one side of the body",
                 "had a severe headache before collapsing", "had high blood pressure for years",
                 "could no longer speak or move after the attack"]),
    "Diabetes": dict(age="adult", pf=0.55,
        pos=["Id10270", "Id10225", "Id10243", "Id10230", "Id10134"],
        clauses=["was always thirsty and passed urine very often", "had a sore on the foot that would not heal",
                 "had been diagnosed with diabetes", "lost weight despite eating normally"]),
    "Neoplasms": dict(age="adult", pf=0.5,
        pos=["Id10253", "Id10204", "Id10243", "Id10268", "Id10137", "Id10194"],
        clauses=["had a growing lump that doctors said was a tumour", "had a swelling in the abdomen",
                 "wasted away over many months", "had been diagnosed with cancer"]),
    "Chronic pulmonary disease and asthma": dict(age="adult", pf=0.5,
        pos=["Id10159", "Id10168", "Id10153", "Id10166", "Id10135", "Id10138"],
        clauses=["had longstanding breathing difficulty that worsened over time", "wheezed and struggled for air",
                 "had a chronic cough for years", "had asthma attacks treated at the clinic"]),
    "Pneumonia": dict(age="adult", pf=0.5,
        pos=["Id10147", "Id10153", "Id10166", "Id10159", "Id10174"],
        clauses=["developed a fever with a productive cough", "breathed fast and with difficulty",
                 "had chest pain and fever for a few days before death"]),
    "Diarrhea": dict(age="child", pf=0.5,
        pos=["Id10181", "Id10188", "Id10269", "Id10147", "Id10186"],
        clauses=["had frequent watery stools for several days", "vomited and could not keep fluids down",
                 "had sunken eyes from dehydration", "passed loose stools until death"]),
    "Traffic / transport accident": dict(age="adult", pf=0.4,
        pos=["Id10077", "Id10079", "Id10217"],
        clauses=["was struck by a vehicle on the road", "died shortly after a road crash",
                 "was injured in a collision while travelling"]),
    "Assault": dict(age="adult", pf=0.25,
        pos=["Id10077", "Id10090", "Id10091", "Id10092", "Id10411"],
        clauses=["was attacked and beaten by others", "was shot during a robbery",
                 "was stabbed in a fight", "died of injuries inflicted by another person"]),
    "Suicide": dict(age="adult", pf=0.4,
        pos=["Id10077", "Id10099", "Id10084", "Id10412"],
        clauses=["took his own life", "was found after a self-inflicted injury",
                 "died after swallowing poison intentionally"]),
    "Maternal conditions": dict(age="adult", pf=1.0,
        pos=["Id10305", "Id10301", "Id10194", "Id10323"],
        clauses=["was pregnant and bled heavily before death", "died soon after a difficult delivery",
                 "had convulsions late in pregnancy", "had severe abdominal pain while pregnant"]),
    "Birth asphyxia": dict(age="neonate", pf=0.5,
        pos=["Id10112", "Id10113", "Id10281", "Id10275"],
        clauses=["did not breathe or cry at birth", "needed help to breathe after delivery",
                 "became unresponsive soon after being born", "had fits shortly after birth"]),
    "Prematurity": dict(age="neonate", pf=0.5,
        pos=["Id10347", "Id10363", "Id10112", "Id10284"],
        clauses=["was born many weeks early and very small", "was tiny and could not keep warm",
                 "had trouble breathing from the first hours of life"]),
}

_AGE_RANGES = {"adult": (18, 84), "child": (1, 14), "neonate": (0, 0)}
_SUBJECT = {"male": "He", "female": "She"}


def make_who2016_va_df(n_samples: int = 400, seed: int = 2016) -> pd.DataFrame:
    """Synthetic WHO 2016 ODK verbal-autopsy dataset (broad cause grouping labels).

    Modeled on the *shape* of the a study site dataset (cause mix, demographics, narrative
    style) but fully synthetic: narratives are generated from original templates
    and symptom responses are randomly drawn from cause-conditioned probabilities.
    No real records are reproduced.

    Columns: ``id``, ``cause_of_death``, ``narrative``, ``sex``, ``age_group``,
    then WHO 2016 ODK indicator columns (``Id10xxx``) with ``"yes"``/``"no"``
    values — pairs with ``multimodalva/utils/qdesc_who2016.csv``.
    """
    rng = np.random.default_rng(seed)
    causes = list(_WHO2016_PROFILES)
    # a study site-like skew toward HIV/AIDS, TB, cardiac, stroke; lighter tail otherwise.
    weights = np.array([
        0.20, 0.10, 0.10, 0.07, 0.07, 0.07, 0.05, 0.05, 0.04,
        0.05, 0.05, 0.03, 0.03, 0.02, 0.02,
    ])
    weights = weights / weights.sum()
    labels = rng.choice(causes, size=n_samples, p=weights)

    base_p = 0.05  # background positive rate for non-characteristic indicators
    rows: list[dict] = []
    for i, cause in enumerate(labels):
        prof = _WHO2016_PROFILES[cause]
        age_group = prof["age"]
        sex = "female" if rng.random() < prof["pf"] else "male"
        lo, hi = _AGE_RANGES[age_group]
        age = int(rng.integers(lo, hi + 1)) if hi > lo else 0

        rec: dict = {"id": f"VA{i + 1:04d}", "cause_of_death": cause,
                     "sex": sex, "age_group": age_group}
        pos = set(prof["pos"])
        for col in WHO2016_COLS:
            p = rng.uniform(0.7, 0.92) if col in pos else base_p
            rec[col] = "yes" if rng.random() < p else "no"

        # --- narrative: subject + age/sex + 2-3 cause clauses (shuffled) ---
        subj = _SUBJECT[sex] if age_group == "adult" else (
            "The child" if age_group == "child" else "The newborn")
        if age_group == "adult":
            who = f"a {age}-year-old {sex}"
        elif age_group == "child":
            who = f"a {age}-year-old child"
        else:
            who = "a newborn baby"
        clauses = list(prof["clauses"])
        rng.shuffle(clauses)
        k = int(rng.integers(2, min(3, len(clauses)) + 1))
        body = "; ".join(clauses[:k])
        rec["narrative"] = (
            f"The deceased was {who}. {subj} {body}. "
            f"The family sought care before death."
        )
        rows.append(rec)

    cols = ["id", "cause_of_death", "narrative", "sex", "age_group"] + WHO2016_COLS
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# Public registry + loaders
# ---------------------------------------------------------------------------
# name -> (loader(**kwargs) -> DataFrame, one-line description)
_DATASETS: dict[str, tuple] = {
    "va_sample": (
        make_sample_va_df,
        "InterVA i-code-style: InterVA i-code indicators (y/n) + broad cause grouping labels; "
        "pairs with utils/qdesc.csv. kwargs: n_per_class, seed.",
    ),
    "va_sample_text_only": (
        lambda **kw: make_sample_va_df(**kw)[["id", "cause_of_death", "narrative"]],
        "Text-only view of va_sample: id, cause_of_death, narrative. kwargs: n_per_class, seed.",
    ),
    "va_who2016": (
        make_who2016_va_df,
        "WHO 2016 ODK Id10xxx indicators (yes/no) + sex/age_group + broad cause grouping labels; "
        "pairs with utils/qdesc_who2016.csv. kwargs: n_samples, seed.",
    ),
    "va_demo": (
        make_demo_va_df,
        "Tiny 4-cause human-readable toy (fever/cough/... columns). kwargs: n_samples, seed.",
    ),
}


def list_datasets() -> dict[str, str]:
    """Return ``{name: description}`` for every bundled example dataset."""
    return {name: desc for name, (_, desc) in _DATASETS.items()}


def data(name: str | None = None, **kwargs) -> "pd.DataFrame | dict[str, str]":
    """Load a synthetic example dataset by name (R-style ``data()``).

    Args:
        name: Dataset name. Call with no argument to get ``list_datasets()``
              (the ``{name: description}`` mapping). Available names:
              ``"va_sample"``, ``"va_sample_text_only"``, ``"va_who2016"``, ``"va_demo"``.
        **kwargs: Forwarded to the underlying generator (e.g. ``seed``,
                  ``n_per_class`` for va_sample*, ``n_samples`` for va_who2016/va_demo).

    Returns:
        A pandas DataFrame for the named dataset, or the ``{name: description}``
        mapping when ``name`` is None.

    Raises:
        ValueError: If ``name`` is not a known dataset.
    """
    if name is None:
        return list_datasets()
    if name not in _DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(_DATASETS)}. "
            "Call data() with no argument to see descriptions."
        )
    loader, _ = _DATASETS[name]
    return loader(**kwargs)
