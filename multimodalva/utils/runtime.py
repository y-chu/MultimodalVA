"""Runtime timing and accelerator monitoring helpers."""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return the current local timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def distributed_state() -> tuple[int, int]:
    """Return ``(rank, world_size)`` from the env vars torchrun sets.

    Both default sensibly on a single-process run, so callers can use this to
    guard rank-0-only work without checking whether they are distributed.
    """
    try:
        rank = int(os.environ.get("RANK", "0"))
    except ValueError:
        rank = 0
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    return rank, max(1, world_size)


def resolve_seed(random_state: int, set_seed: int | None) -> int:
    """Pick the effective seed, letting ``set_seed`` win over ``random_state``.

    ``set_seed`` is the single-value convenience knob on the pipeline wrappers:
    passing it makes one number drive the split, HPO and training seeds.
    """
    if set_seed is None:
        return int(random_state)
    resolved = int(set_seed)
    if resolved != int(random_state):
        logger.info(
            "set_seed=%d provided; overriding random_state=%d.",
            resolved,
            random_state,
        )
    return resolved


def is_cuda() -> bool:
    """True when a CUDA GPU is available."""
    import torch

    return torch.cuda.is_available()


def is_mps() -> bool:
    """True when Apple Silicon MPS is the accelerator in use.

    CUDA wins when both are somehow present, so this returns False on a CUDA
    machine even if MPS also reports itself as available.
    """
    import torch

    return torch.backends.mps.is_available() and not torch.cuda.is_available()


def get_device() -> Any:
    """Return the best available torch device: CUDA, then MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_accelerator_cache() -> None:
    """Release cached GPU or MPS memory.

    Safe to call on any device — the CUDA call is a no-op without a GPU, and the
    MPS branch only runs on Apple Silicon. Call it between HPO trials, where
    freed memory is what keeps the next trial from running out.
    """
    import torch

    torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _jsonable(value: Any) -> Any:
    """Convert nested values into JSON-safe Python objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _monitoring_enabled() -> bool:
    """Return True when GPU monitoring is enabled by environment."""
    raw = os.environ.get("MULTIMODALVA_ENABLE_GPU_MONITOR", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _monitor_interval_seconds() -> float:
    """Return the GPU monitoring sample interval in seconds."""
    raw = os.environ.get("MULTIMODALVA_GPU_MONITOR_INTERVAL_SEC", "15")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 15.0


def _resolve_visible_cuda_id(device_index: int) -> str:
    """Map a torch-visible CUDA index to the identifier accepted by nvidia-smi."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return str(device_index)

    entries = [part.strip() for part in visible.split(",") if part.strip()]
    if 0 <= device_index < len(entries):
        return entries[device_index]
    return str(device_index)


def _capture_torch_snapshot(device: Any = None) -> dict[str, Any]:
    """Capture accelerator memory stats from torch when available."""
    snapshot: dict[str, Any] = {
        "accelerator": "cpu",
        "device_index": None,
        "device_name": None,
    }
    try:
        import torch
    except ImportError:
        return snapshot

    if device is not None:
        try:
            dev = torch.device(device)
        except Exception:
            dev = device
        dev_type = getattr(dev, "type", str(dev))
        dev_index = getattr(dev, "index", None)
    elif torch.cuda.is_available():
        dev_type = "cuda"
        dev_index = torch.cuda.current_device()
    elif torch.backends.mps.is_available():
        dev_type = "mps"
        dev_index = None
    else:
        dev_type = "cpu"
        dev_index = None

    snapshot["accelerator"] = dev_type
    snapshot["device_index"] = dev_index

    if dev_type == "cuda" and torch.cuda.is_available():
        idx = 0 if dev_index is None else int(dev_index)
        props = torch.cuda.get_device_properties(idx)
        snapshot.update(
            {
                "device_name": props.name,
                "gpu_total_memory_mb": round(props.total_memory / (1024 ** 2), 2),
                "torch_memory_allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024 ** 2), 2),
                "torch_memory_reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024 ** 2), 2),
                "torch_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(idx) / (1024 ** 2), 2),
                "torch_max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(idx) / (1024 ** 2), 2),
            }
        )
    elif dev_type == "mps" and torch.backends.mps.is_available():
        current_alloc = None
        driver_alloc = None
        if hasattr(torch.mps, "current_allocated_memory"):
            try:
                current_alloc = round(torch.mps.current_allocated_memory() / (1024 ** 2), 2)
            except Exception:
                current_alloc = None
        if hasattr(torch.mps, "driver_allocated_memory"):
            try:
                driver_alloc = round(torch.mps.driver_allocated_memory() / (1024 ** 2), 2)
            except Exception:
                driver_alloc = None
        snapshot.update(
            {
                "device_name": "Apple MPS",
                "mps_current_allocated_mb": current_alloc,
                "mps_driver_allocated_mb": driver_alloc,
            }
        )

    return snapshot


def _capture_nvidia_smi_snapshot(device_index: int | None) -> dict[str, Any]:
    """Capture device-wide NVIDIA utilization metrics via nvidia-smi."""
    if device_index is None:
        return {}

    query = ",".join(
        [
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
            "temperature.gpu",
            "power.draw",
        ]
    )
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                _resolve_visible_cuda_id(device_index),
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        parts = [part.strip() for part in proc.stdout.strip().split(",")]
        if len(parts) != 6:
            return {}
        gpu_util, mem_util, mem_used, mem_total, temp_c, power_w = parts
        return {
            "gpu_utilization_pct": float(gpu_util),
            "gpu_memory_utilization_pct": float(mem_util),
            "gpu_memory_used_mb": float(mem_used),
            "gpu_memory_total_mb_nvidia_smi": float(mem_total),
            "gpu_temperature_c": float(temp_c),
            "gpu_power_w": float(power_w) if power_w not in {"N/A", "[Not Supported]"} else None,
        }
    except Exception:
        return {}


def capture_accelerator_snapshot(device: Any = None) -> dict[str, Any]:
    """Capture the current accelerator snapshot for CUDA/MPS/CPU runs."""
    rank, world_size = distributed_state()
    snapshot = {
        "timestamp": _now_iso(),
        "rank": rank,
        "world_size": world_size,
    }
    torch_snapshot = _capture_torch_snapshot(device)
    snapshot.update(torch_snapshot)
    if snapshot.get("accelerator") == "cuda":
        snapshot.update(
            _capture_nvidia_smi_snapshot(snapshot.get("device_index"))
        )
    return snapshot


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return round(max(values), 2)


def _ranked_output_path(path: str | Path) -> Path:
    """Apply a rank suffix for multi-process runs to avoid write collisions."""
    out = Path(path)
    rank, world_size = distributed_state()
    if world_size > 1:
        return out.with_name(f"{out.stem}_rank{rank}{out.suffix}")
    return out


class GpuUsageMonitor:
    """Sample accelerator usage periodically and append rows to a CSV file."""

    FIELDNAMES = [
        "timestamp",
        "stage",
        "elapsed_seconds",
        "rank",
        "world_size",
        "accelerator",
        "device_index",
        "device_name",
        "gpu_utilization_pct",
        "gpu_memory_utilization_pct",
        "gpu_memory_used_mb",
        "gpu_memory_total_mb",
        "gpu_temperature_c",
        "gpu_power_w",
        "torch_memory_allocated_mb",
        "torch_memory_reserved_mb",
        "torch_max_memory_allocated_mb",
        "torch_max_memory_reserved_mb",
        "mps_current_allocated_mb",
        "mps_driver_allocated_mb",
    ]

    def __init__(
        self,
        output_path: str | Path,
        *,
        stage: str,
        device: Any = None,
        sample_interval_seconds: float | None = None,
        logger_: logging.Logger | None = None,
    ):
        self.output_path = _ranked_output_path(output_path)
        self.stage = stage
        self.device = device
        self.sample_interval_seconds = (
            sample_interval_seconds
            if sample_interval_seconds is not None
            else _monitor_interval_seconds()
        )
        self.logger = logger_ or logger
        self.records: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at_perf: float | None = None
        self._enabled = _monitoring_enabled()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the monitor thread and record the first sample."""
        self._started_at_perf = time.perf_counter()
        if not self._enabled:
            return

        try:
            import torch

            torch_snapshot = _capture_torch_snapshot(self.device)
            if torch_snapshot.get("accelerator") == "cuda" and torch.cuda.is_available():
                idx = torch_snapshot.get("device_index")
                idx = 0 if idx is None else int(idx)
                torch.cuda.reset_peak_memory_stats(idx)
        except Exception:
            pass

        self._append_sample()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name=f"multimodalva-gpu-monitor-{self.stage}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return a summary of the collected records."""
        if self._enabled:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=self.sample_interval_seconds + 2.0)
            self._append_sample()
        summary = self.summary()
        if summary.get("gpu_log_path"):
            self.logger.info(
                "GPU usage log for stage '%s' saved to %s",
                self.stage,
                summary["gpu_log_path"],
            )
        return summary

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._append_sample()

    def _append_sample(self) -> None:
        snapshot = capture_accelerator_snapshot(self.device)
        started = self._started_at_perf or time.perf_counter()
        snapshot["stage"] = self.stage
        snapshot["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        snapshot["gpu_memory_total_mb"] = (
            snapshot.get("gpu_memory_total_mb_nvidia_smi")
            or snapshot.get("gpu_total_memory_mb")
        )

        with self._lock:
            self.records.append(snapshot)

        if snapshot.get("accelerator") not in {"cuda", "mps"}:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_path.exists()
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if write_header:
                writer.writeheader()
            row = {field: snapshot.get(field) for field in self.FIELDNAMES}
            writer.writerow(row)

    def summary(self) -> dict[str, Any]:
        """Summarize the collected samples for this stage."""
        with self._lock:
            records = list(self.records)

        if records:
            first = records[0]
        else:
            first = capture_accelerator_snapshot(self.device)

        summary = {
            "accelerator": first.get("accelerator", "cpu"),
            "device_index": first.get("device_index"),
            "device_name": first.get("device_name"),
            "sample_count": len(records),
            "sample_interval_seconds": self.sample_interval_seconds,
            "gpu_log_path": str(self.output_path) if self.output_path.exists() else None,
        }

        util_vals = [float(v) for v in (r.get("gpu_utilization_pct") for r in records) if v is not None]
        mem_used_vals = [float(v) for v in (r.get("gpu_memory_used_mb") for r in records) if v is not None]
        mem_util_vals = [float(v) for v in (r.get("gpu_memory_utilization_pct") for r in records) if v is not None]
        torch_max_alloc_vals = [
            float(v) for v in (r.get("torch_max_memory_allocated_mb") for r in records) if v is not None
        ]
        torch_max_reserved_vals = [
            float(v) for v in (r.get("torch_max_memory_reserved_mb") for r in records) if v is not None
        ]
        mps_alloc_vals = [float(v) for v in (r.get("mps_current_allocated_mb") for r in records) if v is not None]
        mps_driver_vals = [float(v) for v in (r.get("mps_driver_allocated_mb") for r in records) if v is not None]

        summary.update(
            {
                "gpu_utilization_pct_mean": _mean(util_vals),
                "gpu_utilization_pct_max": _max(util_vals),
                "gpu_memory_used_mb_max": _max(mem_used_vals),
                "gpu_memory_utilization_pct_max": _max(mem_util_vals),
                "torch_max_memory_allocated_mb_max": _max(torch_max_alloc_vals),
                "torch_max_memory_reserved_mb_max": _max(torch_max_reserved_vals),
                "mps_current_allocated_mb_max": _max(mps_alloc_vals),
                "mps_driver_allocated_mb_max": _max(mps_driver_vals),
            }
        )
        return summary


class RuntimeTracker:
    """Collect stage timings and optionally GPU usage summaries for a run."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        report_name: str = "runtime_report.json",
        metadata: dict[str, Any] | None = None,
        logger_: logging.Logger | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.runtime_dir = self.output_dir / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = _ranked_output_path(self.runtime_dir / report_name)
        self.stage_csv_path = _ranked_output_path(self.runtime_dir / "stage_timings.csv")
        self.logger = logger_ or logger
        self.metadata = dict(metadata or {})
        self.stages: list[dict[str, Any]] = []
        self._run_started_at = _now_iso()
        self._run_started_perf = time.perf_counter()
        self.write()

    def update_metadata(self, **kwargs: Any) -> None:
        """Merge new metadata into the runtime report and write it to disk."""
        self.metadata.update(kwargs)
        self.write()

    def add_stage(self, event: dict[str, Any]) -> None:
        """Append a fully-formed stage event and persist the runtime report."""
        self.stages.append(_jsonable(event))
        self.write()

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        details: dict[str, Any] | None = None,
        monitor_gpu: bool = False,
        device: Any = None,
        gpu_log_name: str = "gpu_usage.csv",
    ):
        """Context manager that records stage timing and optional GPU usage."""
        started_at = _now_iso()
        started_perf = time.perf_counter()
        error_type = None
        error_message = None
        monitor = None
        if monitor_gpu:
            monitor = GpuUsageMonitor(
                self.runtime_dir / gpu_log_name,
                stage=name,
                device=device,
                logger_=self.logger,
            )
            monitor.start()
        self.logger.info("Stage '%s' started.", name)
        try:
            yield monitor
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
            raise
        finally:
            gpu_summary = monitor.stop() if monitor is not None else None
            elapsed = round(time.perf_counter() - started_perf, 3)
            event = {
                "stage": name,
                "status": "error" if error_type else "ok",
                "started_at": started_at,
                "ended_at": _now_iso(),
                "elapsed_seconds": elapsed,
                "details": _jsonable(details or {}),
                "gpu_summary": _jsonable(gpu_summary or {}),
                "error_type": error_type,
                "error_message": error_message,
            }
            self.add_stage(event)
            self.logger.info(
                "Stage '%s' completed in %.2fs.",
                name,
                elapsed,
            )

    def write(self) -> None:
        """Write the JSON runtime report and a CSV stage summary."""
        payload = {
            "run_started_at": self._run_started_at,
            "last_updated_at": _now_iso(),
            "elapsed_seconds_total": round(time.perf_counter() - self._run_started_perf, 3),
            "metadata": _jsonable(self.metadata),
            "stages": _jsonable(self.stages),
        }
        with open(self.report_path, "w") as f:
            json.dump(payload, f, indent=2)
        self._write_stage_csv()

    def _write_stage_csv(self) -> None:
        fieldnames = [
            "stage",
            "status",
            "started_at",
            "ended_at",
            "elapsed_seconds",
            "accelerator",
            "device_name",
            "gpu_utilization_pct_mean",
            "gpu_utilization_pct_max",
            "gpu_memory_used_mb_max",
            "torch_max_memory_allocated_mb_max",
            "mps_current_allocated_mb_max",
            "gpu_log_path",
            "details_json",
            "error_type",
            "error_message",
        ]
        rows: list[dict[str, Any]] = []
        for stage in self.stages:
            gpu_summary = stage.get("gpu_summary") or {}
            rows.append(
                {
                    "stage": stage.get("stage"),
                    "status": stage.get("status"),
                    "started_at": stage.get("started_at"),
                    "ended_at": stage.get("ended_at"),
                    "elapsed_seconds": stage.get("elapsed_seconds"),
                    "accelerator": gpu_summary.get("accelerator"),
                    "device_name": gpu_summary.get("device_name"),
                    "gpu_utilization_pct_mean": gpu_summary.get("gpu_utilization_pct_mean"),
                    "gpu_utilization_pct_max": gpu_summary.get("gpu_utilization_pct_max"),
                    "gpu_memory_used_mb_max": gpu_summary.get("gpu_memory_used_mb_max"),
                    "torch_max_memory_allocated_mb_max": gpu_summary.get("torch_max_memory_allocated_mb_max"),
                    "mps_current_allocated_mb_max": gpu_summary.get("mps_current_allocated_mb_max"),
                    "gpu_log_path": gpu_summary.get("gpu_log_path"),
                    "details_json": json.dumps(_jsonable(stage.get("details") or {}), sort_keys=True),
                    "error_type": stage.get("error_type"),
                    "error_message": stage.get("error_message"),
                }
            )
        with open(self.stage_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
