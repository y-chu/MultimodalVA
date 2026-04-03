"""NumPy/joblib compatibility helpers shared across tabular and ensemble code."""

from __future__ import annotations

import contextlib
import copyreg
import inspect
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def ensure_numpy_rng_compat() -> None:
    """Patch NumPy/joblib unpickling to tolerate cross-version NumPy artifacts."""
    import sys
    import numpy.random as _npr

    # NumPy 1.x and 2.x disagree on some internal module paths that can leak into
    # pickle/joblib payloads. Register the common aliases before any load().
    try:
        import numpy.core as _npcore
        sys.modules.setdefault("numpy.core", _npcore)
    except Exception:
        _npcore = None

    try:
        import numpy.core.numeric as _npnumeric
        sys.modules.setdefault("numpy.core.numeric", _npnumeric)
        sys.modules.setdefault("numpy._core.numeric", _npnumeric)
    except Exception:
        pass

    if _npcore is not None:
        sys.modules.setdefault("numpy._core", _npcore)

    sys.modules.setdefault("numpy.random._mt19937", _npr)

    try:
        import numpy.random._pickle as _nrp
    except (ImportError, AttributeError):
        return

    ctor_name = None
    for candidate in ("_bit_generator_ctor", "__bit_generator_ctor"):
        if hasattr(_nrp, candidate):
            ctor_name = candidate
            break
    if ctor_name is None:
        return

    original_ctor = getattr(_nrp, ctor_name)
    if getattr(original_ctor, "_compat_patched", False):
        return

    def _compat_ctor(bit_generator: Any):
        if isinstance(bit_generator, _CompatBitGenerator):
            return bit_generator
        if isinstance(bit_generator, type):
            return bit_generator()
        if isinstance(bit_generator, str) and hasattr(np.random, bit_generator):
            return getattr(np.random, bit_generator)()
        raise ValueError(f"{bit_generator!r} is not a known BitGenerator module.")

    _compat_ctor._compat_patched = True
    setattr(_nrp, ctor_name, _compat_ctor)


class _CompatBitGenerator:
    """Prediction-only stand-in for legacy pickled NumPy BitGenerators."""

    def __init__(self, *_args, **_kwargs):
        self.state = None

    def __setstate__(self, state):
        self.state = state


class _CompatGenerator:
    """Prediction-only stand-in for legacy pickled NumPy Generators."""

    def __init__(self, bit_generator=None, *_args, **_kwargs):
        self.bit_generator = bit_generator
        self.state = None

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


def load_joblib_compat(filename, mmap_mode=None):
    """Load a joblib artifact with NumPy-version compatibility fallbacks.

    Normal ``joblib.load`` is tried first. If NumPy internal module or RNG-state
    incompatibilities break unpickling, retry with a custom unpickler that
    remaps legacy NumPy RNG classes to inert placeholders. This is intended for
    prediction-time loading where persisted RNG state is irrelevant.
    """
    ensure_numpy_rng_compat()

    try:
        return joblib.load(filename, mmap_mode=mmap_mode)
    except Exception as exc:
        if not _should_retry_compat_load(exc):
            raise
        return _load_joblib_with_numpy_fallbacks(filename, mmap_mode=mmap_mode)


def sanitize_random_state(estimator, random_state: int = 42, _seen: set[int] | None = None) -> None:
    """Reset fitted estimator ``random_state*`` attributes to an integer seed."""
    if _seen is None:
        _seen = set()
    try:
        obj_id = id(estimator)
    except Exception:
        return
    if obj_id in _seen:
        return
    _seen.add(obj_id)

    try:
        attrs = list(vars(estimator))
    except TypeError:
        return

    for attr_name in attrs:
        if "random_state" not in attr_name:
            continue
        try:
            value = getattr(estimator, attr_name, None)
        except Exception:
            continue
        if value is None or isinstance(value, int):
            continue
        try:
            setattr(estimator, attr_name, random_state)
        except (AttributeError, TypeError):
            pass

    for attr_name in attrs:
        try:
            value = getattr(estimator, attr_name, None)
        except Exception:
            continue
        _walk_nested(value, lambda item: sanitize_random_state(item, random_state, _seen))


def strip_fit_rng(estimator, _seen: set[int] | None = None) -> None:
    """Remove NumPy RNG objects from fitted estimators before pickling."""
    if _seen is None:
        _seen = set()
    try:
        obj_id = id(estimator)
    except Exception:
        return
    if obj_id in _seen:
        return
    _seen.add(obj_id)

    rng_types = (np.random.RandomState, np.random.Generator, np.random.BitGenerator)

    try:
        attrs = list(vars(estimator))
    except TypeError:
        return

    for attr_name in attrs:
        try:
            value = getattr(estimator, attr_name, None)
        except Exception:
            continue

        if isinstance(value, rng_types):
            try:
                setattr(estimator, attr_name, None)
            except (AttributeError, TypeError):
                pass
            continue

        _walk_nested(value, lambda item: strip_fit_rng(item, _seen))


def prepare_estimator_for_joblib(estimator, random_state: int = 42) -> None:
    """Make an estimator safer to save and reload across NumPy versions."""
    sanitize_random_state(estimator, random_state=random_state)
    strip_fit_rng(estimator)


def _all_numpy_rng_types() -> set[type]:
    """Return all concrete NumPy RNG classes used by pickle dispatch."""
    types: set[type] = {np.random.RandomState, np.random.Generator, np.random.BitGenerator}

    def _collect(cls: type) -> None:
        for sub in cls.__subclasses__():
            types.add(sub)
            _collect(sub)

    _collect(np.random.BitGenerator)
    return types


def _null_rng(_obj: object) -> tuple[type, tuple]:
    return (type(None), ())


@contextlib.contextmanager
def null_rng_pickler():
    """Serialize any remaining NumPy RNG object as ``None`` within the context."""
    rng_types = _all_numpy_rng_types()
    previous = {rng_type: copyreg.dispatch_table.get(rng_type) for rng_type in rng_types}
    for rng_type in rng_types:
        copyreg.dispatch_table[rng_type] = _null_rng
    try:
        yield
    finally:
        for rng_type, reducer in previous.items():
            if reducer is None:
                copyreg.dispatch_table.pop(rng_type, None)
            else:
                copyreg.dispatch_table[rng_type] = reducer


def _walk_nested(value, visit) -> None:
    if hasattr(value, "__dict__"):
        visit(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if hasattr(item, "__dict__"):
                visit(item)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for item in value.flat:
            if hasattr(item, "__dict__"):
                visit(item)
    elif isinstance(value, dict):
        for item in value.values():
            if hasattr(item, "__dict__"):
                visit(item)


def _should_retry_compat_load(exc: Exception) -> bool:
    text = str(exc)
    markers = (
        "numpy._core",
        "numpy.core.numeric",
        "BitGenerator module",
        "legacy MT19937 state",
        "numpy.random",
    )
    return any(marker in text for marker in markers)


def _load_joblib_with_numpy_fallbacks(filename, mmap_mode=None):
    from joblib.numpy_pickle import NumpyUnpickler

    class _CompatNumpyUnpickler(NumpyUnpickler):
        _CLASS_ALIASES = {
            ("numpy.random._mt19937", "MT19937"): _CompatBitGenerator,
            ("numpy.random", "MT19937"): _CompatBitGenerator,
            ("numpy.random._pcg64", "PCG64"): _CompatBitGenerator,
            ("numpy.random._philox", "Philox"): _CompatBitGenerator,
            ("numpy.random._sfc64", "SFC64"): _CompatBitGenerator,
            ("numpy.random._generator", "Generator"): _CompatGenerator,
            ("numpy.random", "Generator"): _CompatGenerator,
        }
        _MODULE_ALIASES = {
            "numpy._core": "numpy.core",
            "numpy._core.numeric": "numpy.core.numeric",
        }

        def find_class(self, module, name):
            module = self._MODULE_ALIASES.get(module, module)
            alias = self._CLASS_ALIASES.get((module, name))
            if alias is not None:
                return alias
            return super().find_class(module, name)

    if Path is not None and isinstance(filename, Path):
        filename = str(filename)

    with _prediction_pickle_rng_compat():
        if hasattr(filename, "read"):
            fobj = filename
            filename = getattr(fobj, "name", "")
            return _compat_unpickle(fobj, filename, mmap_mode, _CompatNumpyUnpickler)

        with open(filename, "rb") as f:
            return _compat_unpickle(f, filename, mmap_mode, _CompatNumpyUnpickler)


def _compat_unpickle(fobj, filename, mmap_mode, unpickler_cls):
    unpickler = _build_numpy_unpickler(unpickler_cls, filename, fobj, mmap_mode)
    try:
        obj = unpickler.load()
        if unpickler.compat_mode:
            warnings.warn(
                (
                    "The file '%s' has been generated with a joblib version less "
                    "than 0.10. Please regenerate this pickle file."
                ) % filename,
                DeprecationWarning,
                stacklevel=3,
            )
        return obj
    except UnicodeDecodeError as exc:
        new_exc = ValueError(
            "You may be trying to read with python 3 a joblib pickle generated "
            "with python 2. This feature is not supported by joblib."
        )
        new_exc.__cause__ = exc
        raise new_exc


def _build_numpy_unpickler(unpickler_cls, filename, fobj, mmap_mode):
    params = inspect.signature(unpickler_cls).parameters
    kwargs = {}
    if "mmap_mode" in params:
        kwargs["mmap_mode"] = mmap_mode
    if "ensure_native_byte_order" in params:
        kwargs["ensure_native_byte_order"] = True

    attempts = []
    if kwargs:
        attempts.append(kwargs)
    attempts.extend(
        [
            {"mmap_mode": mmap_mode, "ensure_native_byte_order": True},
            {"mmap_mode": mmap_mode},
            {"ensure_native_byte_order": True},
            {},
        ]
    )

    last_error = None
    seen = set()
    for attempt in attempts:
        key = tuple(sorted(attempt.items()))
        if key in seen:
            continue
        seen.add(key)
        try:
            return unpickler_cls(filename, fobj, **attempt)
        except TypeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    return unpickler_cls(filename, fobj)


@contextlib.contextmanager
def _prediction_pickle_rng_compat():
    """Force legacy NumPy RNG pickle constructors to return inert placeholders."""
    patched = []

    try:
        import numpy.random._pickle as _nrp
    except Exception:
        _nrp = None

    if _nrp is not None:
        for ctor_name in ("_bit_generator_ctor", "__bit_generator_ctor"):
            if hasattr(_nrp, ctor_name):
                original = getattr(_nrp, ctor_name)

                def _compat_ctor(_bit_generator, _orig=original):
                    if isinstance(_bit_generator, _CompatBitGenerator):
                        return _bit_generator
                    return _CompatBitGenerator()

                setattr(_nrp, ctor_name, _compat_ctor)
                patched.append((_nrp, ctor_name, original))

    module_aliases = {
        "numpy.random._mt19937": sys.modules.get("numpy.random"),
    }
    previous_modules = {}
    for name, module in module_aliases.items():
        previous_modules[name] = sys.modules.get(name)
        if module is not None:
            sys.modules[name] = module

    try:
        yield
    finally:
        for obj, name, original in reversed(patched):
            setattr(obj, name, original)
        for name, original in previous_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
