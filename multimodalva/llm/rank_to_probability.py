"""Reconstruct a full probability vector over the cause list from ranked LLM output.

Adapted from the rank-reweighting framework of Chen et al. (LAVA,
arXiv:2509.09602, Sec. 1.4). The citation applies only to the post-hoc conversion
from ranked labels to a full probability vector. The VA prompt and experiment
were developed independently for this study; they do not reproduce LAVA's
prompt, examples, or experimental design. No LAVA code is used.

For record i with ranked causes r_i1..r_iK (K = MAX_RANKED) and confidence g_i:

    q_ic = a_j^(g_i)                              if c = r_ij, j = 1..K
         = (1 - sum_j a_j^(g_i)) * pi~_ic         otherwise

    pi~_ic = pi_c / sum_{c' not in R_i} pi_c'

subject to a_1 >= a_2 >= ... >= a_K >= 0 and sum_j a_j <= 1, with a separate
weight vector per confidence level.

The weights are fixed a priori (`PRIOR_WEIGHTS`), with pi uniform over the causes.
No labels enter anywhere, which is what lets this support a claim about what the
model achieves WITHOUT labelled data. It is an explicitly assumed mapping, not a
calibrated one, and must be reported as such.

A label-fitted variant existed until 2026-10-06 and was abandoned; the decision
and its rationale are recorded where the function used to be, below.

Nothing in this study reports a reconstructed vector as a calibrated
probability. `export_predictions.py` defaults to `stated`, the model's own
numbers, and the selected configuration is exported that way.
"""

from __future__ import annotations

import numpy as np

#: The cap on how many causes the model may rank. Imported rather than restated:
#: this module decodes what that prompt produced, so a mismatch between the two
#: silently drops a ranked cause. It was 5 here while the prompt allowed 6 from
#: the October restructure until 2026-10-05.
from .prompts import MAX_RANKED

# A priori weights, one per rank. Ordered and summing below 1, leaving mass for
# the causes outside the ranked list; flatter as confidence falls. Not tuned on
# any data, and deliberately not fitted here - `fit_weights` is the fitted
# variant and a different estimand.
#
# The sixth entry was added on 2026-10-05, when MAX_RANKED had been 6 for weeks
# while this table had five rows. Two ways to add it, and the choice matters:
# continue the decay and let the head sum rise, or hold the head sum and take the
# mass from the first five. This continues the decay. a_j means "the mass a cause
# in position j deserves", which is a prior about the position and has nothing to
# do with how long the list happens to be; rescaling the first five would assert
# that a model listing one more candidate makes its own first choice less
# credible, and that is an extra assumption with nothing behind it.
#
# Only records that actually rank six causes are affected - `build_matrix` slices
# the vector to the number returned - which was about 2% of development records.
PRIOR_WEIGHTS: dict[str, list[float]] = {
    "high": [0.60, 0.15, 0.08, 0.04, 0.03, 0.02],
    "medium": [0.45, 0.18, 0.10, 0.06, 0.04, 0.03],
    "low": [0.30, 0.18, 0.12, 0.08, 0.06, 0.04],
}

#: Fail at import rather than silently truncating if the two ever diverge again.
for _level, _w in PRIOR_WEIGHTS.items():
    if len(_w) != MAX_RANKED:
        raise ValueError(
            f"PRIOR_WEIGHTS[{_level!r}] has {len(_w)} weights but the prompt allows "
            f"{MAX_RANKED} ranked causes; a rank would get no weight"
        )
    if any(b > a for a, b in zip(_w, _w[1:])):
        raise ValueError(f"PRIOR_WEIGHTS[{_level!r}] is not non-increasing: {_w}")
    if sum(_w) >= 1.0:
        raise ValueError(
            f"PRIOR_WEIGHTS[{_level!r}] sums to {sum(_w):.3f}; it must leave mass "
            "for the causes outside the ranked list"
        )


def _tail(causes: list[str], ranked: list[str], prevalence: np.ndarray) -> np.ndarray:
    """Normalized prevalence over the causes outside the ranked list."""
    index = {c: k for k, c in enumerate(causes)}
    mask = np.ones(len(causes), dtype=bool)
    for cause in ranked:
        if cause in index:
            mask[index[cause]] = False
    tail = np.where(mask, prevalence, 0.0)
    total = tail.sum()
    if total <= 0:  # every cause ranked, or zero prevalence outside
        return np.zeros(len(causes))
    return tail / total


def build_matrix(
    records: list[dict],
    causes: list[str],
    weights: dict[str, list[float]],
    prevalence: np.ndarray | None = None,
) -> np.ndarray:
    """records: [{"ranked_causes": [...up to MAX_RANKED...], "confidence": ...}, ...]"""
    causes = list(causes)
    index = {c: k for k, c in enumerate(causes)}
    if prevalence is None:
        prevalence = np.ones(len(causes)) / len(causes)
    prevalence = np.asarray(prevalence, dtype=float)

    out = np.zeros((len(records), len(causes)))
    for row, record in enumerate(records):
        # Duplicate ranks would let a later, smaller weight overwrite an earlier,
        # larger one, silently losing mass and mis-ranking. The runner rejects
        # duplicates, but this function is also run over stored output.
        ranked = list(dict.fromkeys(record["ranked_causes"]))[:MAX_RANKED]
        if len(ranked) < len(list(record["ranked_causes"])[:MAX_RANKED]):
            raise ValueError(
                f"record {row} has duplicate ranked causes: {record['ranked_causes']}"
            )
        # A model allowed MAX_RANKED causes may distinguish fewer. Only the weights
        # for the ranks actually returned are spent; the rest of the mass goes to
        # the tail, so a short list widens the distribution rather than inflating
        # the ranks that are present.
        alpha = np.asarray(weights[record.get("confidence", "medium")], dtype=float)
        alpha = alpha[: len(ranked)]
        out[row] = (1.0 - alpha.sum()) * _tail(causes, ranked, prevalence)
        for position, cause in enumerate(ranked):
            if cause in index:
                out[row, index[cause]] = alpha[position]
        total = out[row].sum()
        if total > 0:
            out[row] /= total
    return out


# ---------------------------------------------------------------------------
# REMOVED 2026-10-06: fit_weights(), the label-fitted "alpha" variant
#
# Decision: abandoned, and taken out of the code rather than left available, so
# it cannot quietly influence a later result. The previous version is in
# _rollback_who_kb_2026-10-05/rank_to_probability.py.orig.
#
# It fitted one weight vector per confidence level by linear program, minimising
#     sum_c | mean_i q_ic - target_c |
# against the training cause distribution. Measured on the 300-record development
# split:
#
#                      top-1 acc    CSMF    log loss   Brier   exact zeros
#     prior alpha         0.6400   0.8900     1.4680   0.5777            0
#     fitted alpha        0.1433   0.9504     4.1451   0.9653          433
#
# Rationale for removal, in order of weight:
#
# 1. The objective is saturated by ignoring the input. CSMF accuracy compares
#    MARGINAL distributions, so a constant predictor emitting the training
#    prevalence for every record scores CSMF = 1.0000 while getting top-1 right
#    23% of the time - the base rate. The LP minimises exactly that marginal
#    distance, so its optimum points at "discard the model's output, emit the
#    prior", and it goes there: fitted medium weights were [0.006 x5, 0.0], a tail
#    of 0.9698, and on the 155 medium records the argmax agreed with the model's
#    own first-ranked cause on 0 of 155, landing on the most prevalent cause 115
#    times. The ranking - the only thing the LLM actually produced - was gone.
# 2. A one-dimensional marginal target cannot identify an 18-parameter per-case
#    mapping. The design matrix had rank 16 of 18; high_r6 and low_r6 had all-zero
#    columns because no high- or low-confidence record ranked six causes. A 1e-4
#    pull toward the prior moved high_r6 from 0.0 to 0.02 with the residual
#    unchanged to six decimals.
# 3. Weak monotonicity (a_j >= a_{j+1}) permits all-equal, and an L1 objective over
#    a polytope attains its optimum at a vertex, so the fitted high vector was
#    [0.1294 x5, 0.0] - five ranks declared equally likely.
# 4. It was never called. No export, no run, no reported number used it; the
#    selection metrics come from top-1 labels and never touch this module's
#    reconstructions.
# 5. Reporting it would need an argument for preferring an imposed mapping over the
#    model's own stated probabilities, which for an unsupervised comparator is the
#    weaker position.
#
# Reviving it means changing the OBJECTIVE, not restoring this function: add a
# proper scoring rule (log loss or Brier) so per-case probabilities are constrained
# at all, require strict decrease a_j >= (1+d) a_{j+1} so the ranking survives, pin
# ranks with no sample support to the prior, and regularise toward the prior. That
# is a separate piece of methodological work, not a bug fix.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Three probability vectors from one model response
#
# They are different estimands and must be reported as such:
#
#   alpha      rank + confidence, numbers supplied externally by the weight
#              table. Comparable to the LAVA-style reconstruction. Carries
#              log2(3) bits of per-case confidence information and no more: two
#              records with the same ranking and level get identical vectors.
#   topk       the model's own stated probabilities for the ranked causes,
#              with the remaining mass spread over the rest by prevalence.
#              Comparable to CODA, which instead assigns the tail zero.
#   stated     the model's stated probabilities over the whole cause list,
#              renormalized.
#              The only one with per-case numeric variation everywhere, but the
#              tail values are confabulated rather than estimated.
#
# None of the three is a model-internal distribution. For a reasoning model the
# answer-token distribution is degenerate: measured mean top-1 token probability
# was 1.0000 for gpt-oss:20b with reasoning on and 0.9987 with it off, because
# the answer is emitted after the reasoning has already resolved it. A
# non-reasoning model answering in one token does yield a spread distribution
# (medgemma:27b: mean 0.486), so the collapse is caused by the generation that
# precedes the answer, not by the model family.
# ---------------------------------------------------------------------------


def stated_vector(
    record: dict, causes: list[str], floor: float = 0.0
) -> tuple[np.ndarray, dict]:
    """Renormalize the model's stated probabilities over the full cause list.

    Returns the vector and diagnostics: the raw sum before renormalization, the
    share of stated mass outside the ranked causes, and whether the stated
    argmax agrees with the stated first rank.
    """
    stated = record.get("cause_probabilities") or {}
    raw = np.array([float(stated.get(c, 0.0)) for c in causes])
    total = raw.sum()
    ranked = list(record.get("ranked_causes") or [])
    index = {c: k for k, c in enumerate(causes)}
    tail_idx = [index[c] for c in causes if c not in ranked[:MAX_RANKED]]

    diagnostics = {
        "stated_sum": float(total),
        "stated_zeros": int((raw == 0).sum()),
        "stated_distinct": int(len(set(raw.tolist()))),
        "stated_tail_share": float(raw[tail_idx].sum() / total) if total > 0 else None,
        "argmax_matches_rank1": (
            bool(causes[int(raw.argmax())] == ranked[0]) if ranked and total > 0 else None
        ),
    }
    if total <= 0:
        return np.ones(len(causes)) / len(causes), diagnostics
    # Models asked for a value per cause routinely return exact zeros for every
    # cause outside the ones they ranked (measured on 19 causes: 14-15 zero, 4-5
    # distinct values).
    # Those zeros are structural, so log loss is undefined and classwise
    # calibration degenerates unless a floor is applied. `floor` is reported, not
    # hidden: it is a choice about the estimand, not a numerical detail.
    if floor > 0:
        raw = np.maximum(raw, floor * total)
    return raw / raw.sum(), diagnostics


def topk_vector(
    record: dict,
    causes: list[str],
    prevalence: np.ndarray | None = None,
    floor: float = 0.0,
) -> np.ndarray:
    """Stated probabilities for the ranked causes; prevalence spread over the rest.

    `floor` is what keeps the vector strictly positive, and it is NOT optional in
    practice. The tail here is `1 - sum(stated over the ranked causes)`, and models
    hand back a ranked list whose stated values already sum to 1 or more on **88.9%
    of development records** (measured 2026-10-05 across 2,149 scored responses). On
    those records the tail is exactly zero, a median of 16 of 19 causes get
    probability 0, and log loss is undefined - the opposite of what this function
    was documented to guarantee until that was measured.

    So pass a floor whenever the output feeds log loss or classwise calibration.
    The floor is a choice about the estimand, not a numerical detail, and
    `export_predictions.py` records it.
    """
    causes = list(causes)
    index = {c: k for k, c in enumerate(causes)}
    stated = record.get("cause_probabilities") or {}
    ranked = [c for c in list(record.get("ranked_causes") or [])[:MAX_RANKED] if c in index]
    if prevalence is None:
        prevalence = np.ones(len(causes)) / len(causes)

    out = np.zeros(len(causes))
    head = 0.0
    for cause in ranked:
        value = max(float(stated.get(cause, 0.0)), 0.0)
        out[index[cause]] = value
        head += value
    head = min(head, 1.0)
    out *= (head / out.sum()) if out.sum() > 0 else 0.0
    out += (1.0 - head) * _tail(causes, ranked, np.asarray(prevalence, dtype=float))
    if floor > 0:
        # Applied after the tail so it reaches the causes the tail left at zero,
        # which is the whole point: when head >= 1 the tail term contributes
        # nothing at all.
        out = np.maximum(out, floor * max(out.sum(), 1e-12))
    total = out.sum()
    return out / total if total > 0 else np.ones(len(causes)) / len(causes)


def build_all(
    records: list[dict],
    causes: list[str],
    weights: dict[str, list[float]] | None = None,
    prevalence: np.ndarray | None = None,
    floor: float = 0.0,
) -> dict[str, np.ndarray]:
    """All three vectors plus per-record diagnostics, from the same responses.

    `floor` reaches both `stated` and `topk`, the two that can return exact zeros.
    `alpha` cannot: its tail is `1 - sum(alpha over the ranks returned)`, which the
    import-time check on PRIOR_WEIGHTS keeps strictly positive, and the cause list
    (19) is longer than MAX_RANKED (6) so some cause is always outside the ranking.
    Was `stated_floor` until 2026-10-05, when it turned out topk needed it more.
    """
    weights = weights or PRIOR_WEIGHTS
    stated, diags = [], []
    for record in records:
        vector, diagnostic = stated_vector(record, causes, floor=floor)
        stated.append(vector)
        diags.append(diagnostic)
    return {
        "alpha": build_matrix(records, causes, weights, prevalence),
        "topk": np.vstack([topk_vector(r, causes, prevalence, floor=floor) for r in records]),
        "stated": np.vstack(stated),
        "diagnostics": diags,
    }
