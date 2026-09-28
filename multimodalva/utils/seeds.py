"""The two seeds a run needs, and what each one is responsible for.

A run has two independent sources of randomness, and conflating them makes
neither measurable:

``split_seed``
    Every decision about **which rows go where** — the train/test split, the
    cross-validation folds inside a hyperparameter search, the early-stopping
    validation slice, and stacking's out-of-fold partition. Varying it alone
    measures how much the answer depends on which deaths landed in the test
    set: sampling uncertainty.

``train_seed``
    Every decision about **what the model does with the rows it is given** —
    weight initialisation, dropout, batch order, the Optuna sampler's
    proposals, each estimator's own ``random_state``, and AutoMM's trainer.
    Varying it alone, with ``hyperparams`` fixed, measures model stochasticity
    with the data partition and the hyperparameters held constant.

Folds follow ``split_seed`` deliberately. If they followed ``train_seed``, then
changing ``train_seed`` would change which hyperparameters won the search, so a
set of "model stochasticity" replicates would silently also vary in their
hyperparameters. It would also break the validation leaderboard: candidates can
only be compared on their cross-validated scores when they were scored on the
same folds.

There is no third seed for folds. Whether hyperparameter selection is stable
against fold composition is answered more directly and far more cheaply by the
``cv_std_<metric>`` and ``fold_<i>_<metric>`` columns that every search already
records per trial.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

__all__ = [
    "seed_everything",
    "set_determinism",
    "reject_removed_seed_args",
    "REMOVED_SEED_ARGS",
]


#: Seed arguments that used to exist, and what replaced them. A caller still
#: passing one gets an error naming the replacement, because a silently ignored
#: seed is the worst outcome: the run looks reproducible and is not.
REMOVED_SEED_ARGS: dict[str, str] = {
    "random_state": (
        "split_seed= (which rows go where) and/or train_seed= (model "
        "stochasticity). It used to mean both at once"
    ),
    "set_seed": "train_seed= (it was only an alias that overrode random_state)",
    "automm_seed": "train_seed= (AutoMM's trainer seed is now the training seed)",
}


def reject_removed_seed_args(caller: str, **kwargs) -> None:
    """Raise if a caller passed a seed argument that no longer exists.

    Args:
        caller: Name to quote in the message, e.g. ``"run(task='text')"``.
        kwargs: The suspect names mapped to whatever was passed; ``None``
                counts as not passed.

    Raises:
        TypeError: One or more removed seed arguments were supplied.
    """
    offenders = {k: v for k, v in kwargs.items() if v is not None}
    if not offenders:
        return
    lines = [
        f"  {name}= is gone; use {REMOVED_SEED_ARGS.get(name, 'split_seed/train_seed')}."
        for name in sorted(offenders)
    ]
    raise TypeError(
        f"{caller} no longer accepts "
        f"{', '.join(sorted(offenders))}.\n" + "\n".join(lines) + "\n"
        "One seed used to drive both the data split and the model, so neither "
        "could be varied on its own. Pass split_seed= to resample the split and "
        "train_seed= to reseed the model."
    )


def seed_everything(train_seed: int) -> None:
    """Seed every global RNG that model training draws from.

    Covers Python's ``random``, NumPy's legacy global RNG, and Torch on CPU,
    CUDA and MPS. Per-estimator seeds (an sklearn ``random_state``, the Optuna
    sampler, the Hugging Face Trainer) are passed explicitly at their call
    sites; this handles the libraries that only expose a process-wide seed.

    Torch is imported lazily so that seeding a tabular-only run does not pull
    in the transformer stack.

    Args:
        train_seed: The run's model-stochasticity seed.
    """
    import random

    import numpy as np

    random.seed(train_seed)
    np.random.seed(train_seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    # MPS keeps its own generator. torch.manual_seed covers it on current
    # versions, but seeding it directly is harmless and explicit.
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        try:
            mps.manual_seed(train_seed)
        except Exception:  # noqa: BLE001 - older torch without mps.manual_seed
            pass


def set_determinism(deterministic: bool) -> None:
    """Trade speed, and some robustness, for bit-for-bit repeatability.

    With ``deterministic=False`` (the default) a fixed pair of seeds gives the
    same predicted labels every time, while the probabilities can differ in the
    last few digits — around 1e-16 on CPU from float summation order, and up to
    about 1e-7 on an Ampere-or-newer GPU where TF32 matrix multiply is on. That
    is enough for every metric this package reports.

    With ``deterministic=True`` the probabilities are reproducible to the bit on
    the same machine. The costs are real and worth stating plainly:

    * TF32 is switched off, so matrix multiply on Ampere+ GPUs runs roughly
      two to three times slower.
    * Torch is told to use deterministic kernels. Any operation without one
      **raises** ``RuntimeError`` rather than falling back, so a model that
      trains fine by default can fail outright here.
    * cuBLAS needs a fixed workspace, set through an environment variable that
      only takes effect before the first CUDA context is created.

    Use it for the run that backs a published number, not for day-to-day work.
    Nothing here makes results comparable across different GPU models or
    library versions.

    Args:
        deterministic: Whether to demand deterministic kernels.
    """
    try:
        import torch
    except ImportError:
        if deterministic:
            logger.warning(
                "deterministic=True but torch is not installed; only the "
                "Python and NumPy RNGs are affected."
            )
        return

    if not deterministic:
        # Undo a previous deterministic=True in the same process. These are
        # process-wide settings, so without this a notebook that ran one
        # deterministic run would keep every later run slow and liable to raise,
        # even with deterministic=False passed explicitly. TF32 is left for the
        # training code to re-enable on CUDA, which it does once it sees
        # determinism is off.
        if torch.are_deterministic_algorithms_enabled():
            logger.info(
                "deterministic=False: switching off the deterministic mode a "
                "previous run in this process turned on."
            )
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        return

    # Must be set before the first CUDA context; warn rather than fail if late.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        logger.warning(
            "deterministic=True was requested after CUDA was already "
            "initialised. CUBLAS_WORKSPACE_CONFIG cannot take effect now, so "
            "some cuBLAS kernels may stay nondeterministic. Set "
            "deterministic=True before the first CUDA call to avoid this."
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Default in this package is TF32 on, for speed. It is not bit-comparable
    # with fp32, so it has to go when determinism is the point.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Also a TF32 route, and one that AutoMM and other non-text paths hit too,
    # since they never pass through train_text()'s own precision setting.
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:  # noqa: BLE001
        pass

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not enable deterministic algorithms (%s). cuDNN determinism "
            "and TF32-off are still in effect.", exc,
        )

    logger.info(
        "deterministic=True: TF32 off, cuDNN deterministic, deterministic "
        "kernels required. Training will be slower, and an operation with no "
        "deterministic implementation will raise instead of falling back."
    )
