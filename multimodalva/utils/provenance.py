"""
Detect when a saved result was built from inputs that have since changed.

Pipelines reuse finished work: if a meta-learner is already on disk, it is loaded
rather than refitted. That is only correct while the files it was built from are
unchanged. Re-run a base model and the out-of-fold matrix underneath changes, but
the meta-learner sitting next to it still looks finished, so a stale result gets
reported as if it were current.

These helpers record a fingerprint of the files a stage actually reads, and check
them before reusing its output:

    from multimodalva.utils.provenance import record_inputs, check_inputs

    # when saving
    metadata = record_inputs(metadata, {"oof_meta_X": oof_path, "oof_y": y_path})

    # before reusing
    report = check_inputs(metadata, {"oof_meta_X": oof_path, "oof_y": y_path})
    if report.stale:
        ...

**What is fingerprinted.** The content of the files the stage consumes — the
out-of-fold matrix, the base models' prediction tables — not model weights and
not timestamps. Timestamps are unreliable: cloud sync rewrites them, copying a
directory rewrites them, and a stage that skips its work can leave a fresh
timestamp on an untouched file. Content hashing answers the question that
actually matters: are these the same numbers the result was built from?

Because the fingerprint covers what is consumed, re-running an upstream stage and
getting identical predictions does **not** invalidate anything. Only a real
change in the numbers does.

**Results saved before this existed** carry no fingerprint. They are reported as
``unknown`` and reused as before — an older run is never invalidated just for
being older.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20          # 1 MiB
METADATA_KEY = "inputs"   # where fingerprints live inside a metadata dict
RESUME_MANIFEST_SCHEMA = 1


def file_fingerprint(path: str | Path) -> dict:
    """Fingerprint one file: its SHA-256, size and name.

    Args:
        path: File to read.

    Returns:
        Dict with ``sha256``, ``bytes`` and ``name``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    digest = hashlib.sha256()
    size = 0
    with open(p, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size, "name": p.name}


def fingerprint_inputs(inputs: dict) -> dict:
    """Fingerprint several files, keyed by a name of your choosing.

    Args:
        inputs: Mapping of label → file path, for the files a stage reads.
                Missing files are recorded as missing rather than raising, so a
                fingerprint can still be written when an optional input is absent.

    Returns:
        Mapping of label → fingerprint dict.
    """
    out: dict[str, dict] = {}
    for label, path in inputs.items():
        try:
            out[label] = file_fingerprint(path)
        except FileNotFoundError:
            out[label] = {"sha256": None, "bytes": None, "name": Path(path).name,
                          "missing": True}
    return out


def record_inputs(metadata: dict, inputs: dict) -> dict:
    """Add input fingerprints to a metadata dict before it is saved.

    Args:
        metadata: The stage's metadata dict (modified in place and returned).
        inputs:   Mapping of label → file path, for the files this stage read.

    Returns:
        The same metadata dict, with an ``inputs`` entry holding the
        fingerprints and the time they were taken.
    """
    metadata[METADATA_KEY] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": fingerprint_inputs(inputs),
    }
    return metadata


@dataclass
class StaleReport:
    """Result of comparing a saved result's inputs with the current files.

    Attributes:
        status:  ``"match"`` (safe to reuse), ``"stale"`` (inputs changed),
                 ``"missing"`` (an input is gone), or ``"unknown"`` (the result
                 predates input tracking, or recorded no fingerprints).
        changed: Labels of the inputs that differ or are missing.
        details: One human-readable line per changed input.
    """

    status: str
    changed: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def stale(self) -> bool:
        """True when the recorded inputs no longer match what is on disk."""
        return self.status in ("stale", "missing")

    def message(self, what: str) -> str:
        """A sentence explaining what changed, for logs and error messages."""
        if self.status == "unknown":
            return (f"{what} was saved before input tracking existed, so it "
                    "cannot be checked against its inputs.")
        if not self.stale:
            return f"{what} is up to date with its inputs."
        return (f"{what} was built from inputs that have since changed: "
                + "; ".join(self.details))


def check_inputs(metadata: dict | None, inputs: dict) -> StaleReport:
    """Compare a saved result's recorded inputs with the files on disk now.

    Args:
        metadata: The metadata dict saved alongside the result, or ``None``.
        inputs:   Mapping of label → current file path, using the same labels
                  that were passed to :func:`record_inputs`.

    Returns:
        A :class:`StaleReport`. Results saved without fingerprints report
        ``"unknown"`` and are treated as reusable — existing work is never
        invalidated merely for predating this check.
    """
    recorded = (metadata or {}).get(METADATA_KEY, {}).get("files")
    if not recorded:
        return StaleReport(status="unknown")

    changed: list[str] = []
    details: list[str] = []
    missing = False

    for label, was in recorded.items():
        path = inputs.get(label)
        if path is None:
            continue                      # caller is not checking this input now
        try:
            now = file_fingerprint(path)
        except FileNotFoundError:
            missing = True
            changed.append(label)
            details.append(f"{label} is missing ({path})")
            continue
        if was.get("sha256") != now["sha256"]:
            changed.append(label)
            details.append(
                f"{label} changed (recorded {str(was.get('sha256'))[:8]}…, "
                f"now {now['sha256'][:8]}…)"
            )

    if missing:
        return StaleReport(status="missing", changed=changed, details=details)
    if changed:
        return StaleReport(status="stale", changed=changed, details=details)
    return StaleReport(status="match")


def resolve_stale(
    report: StaleReport,
    what: str,
    *,
    on_stale: str = "auto",
    cheap: bool = True,
    force: bool = False,
) -> bool:
    """Decide whether to rebuild a stage whose inputs may have changed.

    The default, ``"auto"``, scales the response to what recomputing costs:
    a cheap stage is rebuilt silently, while an expensive one stops with an
    explanation rather than quietly spending hours or days.

    Args:
        report:   Result of :func:`check_inputs`.
        what:     Name of the stage or artifact, used in messages.
        on_stale: ``"auto"`` (default), ``"recompute"``, ``"error"``, ``"warn"``
                  (reuse, but log a warning), or ``"ignore"`` (reuse silently).
        cheap:    Whether rebuilding is quick (seconds to minutes). Only affects
                  ``"auto"``.
        force:    When True, rebuild regardless — the caller's own ``force`` flag.

    Returns:
        True to rebuild the stage, False to reuse the saved result.

    Raises:
        ValueError: If ``on_stale`` is not one of the accepted values, or when
                    the inputs changed and the policy is to stop.
    """
    valid = {"auto", "recompute", "error", "warn", "ignore"}
    if on_stale not in valid:
        raise ValueError(f"on_stale must be one of {sorted(valid)}, got {on_stale!r}.")

    if force:
        return True
    if not report.stale:
        if report.status == "unknown":
            logger.debug("%s: %s", what, report.message(what))
        return False
    if on_stale == "ignore":
        return False
    if on_stale == "warn":
        logger.warning("%s Reusing it anyway.", report.message(what))
        return False
    if on_stale == "recompute" or (on_stale == "auto" and cheap):
        logger.info("%s Rebuilding it.", report.message(what))
        return True

    raise ValueError(
        f"{report.message(what)} Rebuilding it is expensive, so it is not done "
        "automatically. Pass force=True to rebuild, or on_stale='ignore' to "
        "reuse the existing result as-is."
    )


def guard_inputs(
    report: StaleReport,
    what: str,
    fix: str,
    *,
    on_stale: str = "auto",
) -> None:
    """Stop a stage that would consume a result built from changed inputs.

    Used where the stale result cannot simply be rebuilt on the spot — reusing it
    would silently report numbers that no longer follow from the current inputs,
    so the caller is told what to re-run instead.

    Args:
        report:   Result of :func:`check_inputs`.
        what:     Name of the stale artifact, used in the message.
        fix:      What the user should do, e.g. ``"re-run
                  train_meta_learner_stage()"``.
        on_stale: ``"auto"`` / ``"error"`` stop; ``"warn"`` continues with a
                  warning; ``"ignore"`` continues silently.

    Raises:
        ValueError: If the inputs changed and the policy is to stop.
    """
    if not report.stale:
        return
    if on_stale == "ignore":
        return
    if on_stale == "warn":
        logger.warning("%s Continuing anyway.", report.message(what))
        return
    raise ValueError(
        f"{report.message(what)} Using it now would report results that no "
        f"longer follow from the current inputs. To fix: {fix}. To use it "
        "as-is, pass on_stale='ignore'."
    )


# ---------------------------------------------------------------------------
# Whole-run resume signatures
# ---------------------------------------------------------------------------

def _jsonable(value):
    """Convert configuration values to deterministic JSON-compatible values."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(
            value.items(), key=lambda item: str(item[0])
        )}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v) for v in value), key=repr)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__type__": type(value).__name__, "repr": repr(value)}


def dataframe_fingerprint(df, columns: list[str] | None = None) -> dict:
    """Fingerprint DataFrame values, order, index and schema.

    ``columns`` should name only what a pipeline consumes, so an unrelated
    annotation column does not invalidate an expensive checkpoint.
    """
    import pandas as pd  # local: provenance file hashing stays lightweight

    frame = df if columns is None else df.loc[:, columns]
    schema = {
        "rows": len(frame),
        "columns": [str(c) for c in frame.columns],
        "dtypes": [str(t) for t in frame.dtypes],
        "index_type": type(frame.index).__name__,
    }
    digest = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    )
    row_hashes = pd.util.hash_pandas_object(
        frame, index=True, categorize=True
    ).to_numpy(dtype="uint64", copy=False)
    digest.update(row_hashes.tobytes())
    return {**schema, "sha256": digest.hexdigest()}


def make_resume_manifest(*, dataframes: dict, config: dict) -> dict:
    """Build the content signature written before resumable work starts."""
    core = {
        "schema": RESUME_MANIFEST_SCHEMA,
        "dataframes": {
            name: dataframe_fingerprint(value)
            for name, value in sorted(dataframes.items())
        },
        "config": _jsonable(config),
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return {
        **core,
        "signature": hashlib.sha256(encoded.encode()).hexdigest(),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


#: Accepted values of ``resume=``. ``"adopt"`` is for artifacts made before
#: resume manifests existed: it resumes them and records the current signature,
#: instead of refusing them for having no signature to check.
RESUME_MODES = (True, False, "adopt")


def _normalise_resume(resume, what: str) -> tuple[bool, bool]:
    """``(resume, adopt)`` from a ``resume=`` value, rejecting anything else."""
    if resume is True or resume is False:
        return bool(resume), False
    if isinstance(resume, str) and resume.lower() == "adopt":
        return True, True
    raise ValueError(
        f"resume={resume!r} is not valid for {what}. Use True (reuse artifacts "
        "whose recorded signature matches), False (rebuild from this call), or "
        '"adopt" (reuse artifacts that carry no signature yet — for runs made '
        "before resume manifests existed — and record the current one)."
    )


def enforce_resume_manifest(
    path: str | Path,
    expected: dict,
    *,
    resume: "bool | str",
    artifacts_exist: bool,
    what: str,
    log: logging.Logger | None = None,
) -> None:
    """Refuse to reuse artifacts not proven to match current data/config.

    A missing manifest is safe only when there is no prior work. Runs made
    before this guard are not guessed compatible: use a new output directory,
    start fresh after moving/removing the old artifacts, or say ``resume="adopt"``
    to accept them deliberately.

    ``resume="adopt"`` covers only the case where there is **no signature to
    check** — a missing or unreadable manifest. A manifest that is present and
    *disagrees* is positive evidence that the artifacts came from different data
    or settings, so it still raises: adopting a known contradiction is exactly
    the silent mixing this guard exists to prevent.
    """
    log = log or logger
    resume, adopt = _normalise_resume(resume, what)
    path = Path(path)
    saved = None
    unreadable: str | None = None
    if path.is_file():
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            unreadable = str(exc)
            if resume and artifacts_exist and not adopt:
                raise RuntimeError(
                    f"Cannot verify whether {what} is safe to resume because "
                    f"{path} is unreadable: {exc}. Use a new output_dir, "
                    'resume="adopt" to accept the existing artifacts anyway, or '
                    "move/remove them before starting fresh."
                ) from exc

    if resume and artifacts_exist:
        if saved is None:
            if adopt:
                log.warning(
                    'resume="adopt": %s carries no usable resume signature (%s), '
                    "so its existing artifacts are being reused on your word "
                    "rather than on a check, and the signature of THIS call is "
                    "being recorded for next time. Nothing verified that they "
                    "came from the same data, split and settings. Use "
                    "resume=False to rebuild them instead.",
                    what,
                    f"{path.name} is unreadable: {unreadable}" if unreadable
                    else f"{path.name} does not exist",
                )
                saved = None   # fall through to writing the current manifest
            else:
                raise RuntimeError(
                    f"Refusing to resume {what}: artifacts already exist but "
                    f"{path.name} does not, so their data/configuration cannot be "
                    "verified. Use a new output_dir, resume=\"adopt\" to accept "
                    "them deliberately (for runs made before resume manifests "
                    "existed), or move/remove the old artifacts before starting "
                    "fresh."
                )
        if saved is not None and saved.get("signature") != expected.get("signature"):
            changed = []
            if saved.get("dataframes") != expected.get("dataframes"):
                changed.append("data/split")
            if saved.get("config") != expected.get("config"):
                changed.append("configuration")
            reason = " and ".join(changed) or "signature/schema"
            raise RuntimeError(
                f"Refusing to resume {what}: its recorded {reason} differs "
                "from this call. Reusing it could silently mix runs. Use a "
                "new output_dir or move/remove the old artifacts before "
                'starting fresh. resume="adopt" does not override this: a '
                "manifest that disagrees is evidence of a real difference, not "
                "a missing check."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(expected, indent=2, default=str))
    temporary.replace(path)
