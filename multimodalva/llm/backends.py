"""Where the model runs: a Hugging Face checkpoint in this process, or an ollama server.

Every backend implements the same small protocol, so the classifier never
branches on where the model lives::

    backend = build_llm_backend("openai/gpt-oss-20b")        # transformers
    backend = build_llm_backend("ollama:gpt-oss:20b")        # ollama, by prefix
    raw = backend.generate(system, user, response_schema=schema, settings=settings)

``settings`` is the resolved generation configuration
(:data:`multimodalva.llm.selection.LLM_HYPERPARAMS_DEFAULT` plus ``seed`` and
``timeout``). Heavy imports happen inside the backends, so importing this module
costs nothing.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .response import split_harmony

logger = logging.getLogger(__name__)

__all__ = [
    "LLM_BACKENDS", "LLM_DEFAULT_MODEL", "LLM_REASONING_EFFORTS", "LLMRaw",
    "LLMBackend", "TransformersBackend", "OllamaBackend", "FakeBackend",
    "resolve_llm_backend", "build_llm_backend", "supports_reasoning_effort",
]

LLM_BACKENDS = ("auto", "transformers", "ollama")
LLM_DEFAULT_MODEL = "openai/gpt-oss-20b"
LLM_REASONING_EFFORTS = ("low", "medium", "high")
#: ``model="ollama:<tag>"`` names a model served by ollama.
LLM_OLLAMA_PREFIX = "ollama:"
#: Model families whose server accepts a reasoning-effort setting. Everything
#: else rejects it, so it is omitted rather than sent and caught.
LLM_THINKING_MODEL_PREFIXES = ("gpt-oss", "deepseek-r1", "qwen3", "magistral")
LLM_OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass
class LLMRaw:
    """One completion, with what the backend knows about its cost."""

    content: str
    prompt_tokens: int | None = None
    gen_tokens: int | None = None
    wall_s: float = 0.0
    done_reason: str | None = None
    reasoning: str | None = None


class LLMBackend(Protocol):
    """What every backend provides."""

    name: str
    model: str
    #: Whether the server constrains the output to the response schema. When it
    #: does not, the classifier states the schema in the user message instead.
    enforces_schema: bool

    def generate(self, system: str, user: str, *, response_schema: dict,
                 settings: dict) -> LLMRaw: ...

    def generate_many(self, system: str, users: list[str], *, response_schema: dict,
                      settings: dict) -> list[LLMRaw]: ...

    def supports_reasoning_effort(self) -> bool: ...

    def describe(self) -> dict: ...


def supports_reasoning_effort(model: str) -> bool:
    """Whether ``model`` belongs to a family that takes a reasoning-effort setting."""
    tag = str(model).split("/")[-1].lower()
    return tag.startswith(LLM_THINKING_MODEL_PREFIXES)


def resolve_llm_backend(model: str, backend: str = "auto") -> tuple[str, str]:
    """``(backend name, model id)`` from the model string and the backend setting.

    ``"ollama:<tag>"`` selects ollama by itself; otherwise ``"auto"`` means
    transformers.
    """
    if backend not in LLM_BACKENDS:
        raise ValueError(
            f"backend={backend!r} is not recognised. Use "
            f"{', '.join(repr(b) for b in LLM_BACKENDS)}."
        )
    model = str(model)
    if model.lower().startswith(LLM_OLLAMA_PREFIX):
        if backend == "transformers":
            raise ValueError(
                f"model={model!r} names an ollama model but backend='transformers' "
                "was asked for. Drop the 'ollama:' prefix for a Hub checkpoint, or "
                "pass backend='ollama'."
            )
        return "ollama", model[len(LLM_OLLAMA_PREFIX):]
    if backend == "ollama":
        return "ollama", model
    return "transformers", model


def build_llm_backend(model: str, backend: str = "auto", **backend_kwargs: Any) -> LLMBackend:
    """Construct the backend for ``model`` (loads the model or contacts the server)."""
    kind, model_id = resolve_llm_backend(model, backend)
    if kind == "ollama":
        return OllamaBackend(model_id, **backend_kwargs)
    return TransformersBackend(model_id, **backend_kwargs)


# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------

def _host_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _accelerator_memory_bytes() -> tuple[str, int | None]:
    """``(device kind, bytes the model can use)`` without importing torch at module load."""
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        total = 0
        for index in range(torch.cuda.device_count()):
            total += torch.cuda.mem_get_info(index)[1]
        return "cuda", total
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", _host_memory_bytes()   # unified memory
    return "cpu", _host_memory_bytes()


def _mxfp4_kernels_available() -> bool:
    """Whether the native MXFP4 path can run here: CUDA Hopper+, triton >= 3.4, kernels."""
    import importlib.util  # noqa: PLC0415

    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        if major < 9:
            return False
    except Exception:
        return False
    if importlib.util.find_spec("kernels") is None:
        return False
    try:
        import triton  # noqa: PLC0415

        return tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 4)
    except Exception:
        return False


def check_transformers_memory(model_id: str, *, force_load: bool = False) -> dict:
    """Estimate what loading ``model_id`` needs here, and refuse an impossible load.

    Reads the same ``config.json`` the loader reads. A checkpoint stored in
    MXFP4 keeps that size only on the native kernel path (CUDA Hopper or later,
    with ``kernels`` and ``triton>=3.4``); everywhere else transformers
    dequantises the experts to bf16 at load, about four times the stored size.
    When that estimate exceeds the memory the device can give, loading is
    refused with the ollama backend named as the way to run the same model on
    this machine. ``force_load=True`` overrides the refusal.

    Returns the estimate, so it can be recorded. Never raises on a network or
    metadata failure: it logs and returns what it has.
    """
    report: dict[str, Any] = {"model": model_id, "checked": False}
    try:
        local = Path(model_id).expanduser()
        if local.is_dir():
            config = json.loads((local / "config.json").read_text())
            stored = sum(p.stat().st_size for p in local.glob("*.safetensors"))
        else:
            from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

            config = json.loads(Path(hf_hub_download(model_id, "config.json")).read_text())
            info = HfApi().model_info(model_id, files_metadata=True)
            # Top-level shards only: a repo may carry a second copy of the
            # weights in a subfolder (gpt-oss ships `original/`), which is not
            # what from_pretrained loads.
            stored = sum((s.size or 0) for s in info.siblings
                         if s.rfilename.endswith(".safetensors") and "/" not in s.rfilename)
        quant = str((config.get("quantization_config") or {}).get("quant_method") or "").lower()
        device, available = _accelerator_memory_bytes()
        native = quant == "mxfp4" and _mxfp4_kernels_available()
        if quant == "mxfp4" and not native:
            estimate = int(stored * 4)
            path = "mxfp4 dequantised to bf16"
        else:
            estimate = int(stored * 1.1)
            path = "native mxfp4" if native else (quant or "as stored")
        report.update(checked=True, stored_bytes=stored, estimate_bytes=estimate,
                      load_path=path, device=device, available_bytes=available)
        gib = 1024 ** 3
        if available is not None and estimate > available:
            message = (
                f"{model_id} needs about {estimate / gib:.0f} GB to load here "
                f"({path}; checkpoint {stored / gib:.0f} GB) and this machine's "
                f"{device} can give about {available / gib:.0f} GB. "
            )
            if quant == "mxfp4":
                message += (
                    "The native MXFP4 path needs a CUDA Hopper-class GPU with the "
                    "`kernels` package and triton>=3.4 (pip install 'multimodalva[llm]'); "
                    "elsewhere transformers dequantises to bf16. "
                )
            message += (
                "To run this model on this machine, serve it with ollama and pass "
                'model="ollama:<tag>". Pass force_load=True in backend_kwargs to try anyway.'
            )
            if force_load:
                logger.warning(message)
            else:
                raise MemoryError(message)
        elif available is not None and estimate > 0.85 * available:
            logger.warning(
                "%s is estimated at %.0f GB (%s) against about %.0f GB on %s: "
                "little headroom for the KV cache and long prompts.",
                model_id, estimate / gib, path, available / gib, device,
            )
    except MemoryError:
        raise
    except Exception as exc:  # metadata or network failure: say so, do not block
        logger.info("Could not estimate memory for %s before loading (%s).", model_id, exc)
    return report


class TransformersBackend:
    """A causal LM loaded with transformers, generating in this process.

    Args:
        model_id: Hub id or local directory of a chat model.
        device_map: Passed to ``from_pretrained``; ``"auto"`` spreads a large
            model over the visible accelerators.
        dtype: ``"auto"`` keeps the checkpoint's dtype (or dequantises MXFP4
            where the native path is unavailable).
        force_load: Skip the memory refusal in :func:`check_transformers_memory`.
        trust_remote_code: Forwarded to transformers.
    """

    name = "transformers"
    enforces_schema = False

    def __init__(self, model_id: str, *, device_map: str | None = "auto",
                 dtype: str = "auto", force_load: bool = False,
                 trust_remote_code: bool = False):
        self.model = str(model_id)
        self.memory_report = check_transformers_memory(self.model, force_load=force_load)
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype == "auto":
            load_kwargs["torch_dtype"] = "auto"
        else:
            load_kwargs["torch_dtype"] = getattr(torch, dtype)
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        started = time.perf_counter()
        try:
            self.model_obj = AutoModelForCausalLM.from_pretrained(self.model, **load_kwargs)
        except ValueError as exc:
            if device_map is None or "device_map" not in str(exc):
                raise
            # A model class that does not support device_map: load on one device.
            load_kwargs.pop("device_map")
            self.model_obj = AutoModelForCausalLM.from_pretrained(self.model, **load_kwargs)
            from ..utils.runtime import get_device  # noqa: PLC0415

            self.model_obj.to(get_device())
        self.model_obj.eval()
        self.load_s = round(time.perf_counter() - started, 1)
        template = getattr(self.tokenizer, "chat_template", None) or ""
        self._harmony = "reasoning_effort" in template
        self.revision = getattr(getattr(self.model_obj, "config", None), "_commit_hash", None)
        logger.info("Loaded %s with transformers in %.0fs (%s).", self.model, self.load_s,
                    "harmony chat template" if self._harmony else "plain chat template")

    # -- protocol -------------------------------------------------------------

    def supports_reasoning_effort(self) -> bool:
        return self._harmony

    def describe(self) -> dict:
        model = self.model_obj
        dtype = str(getattr(model, "dtype", None))
        try:
            device = str(next(model.parameters()).device)
        except StopIteration:
            device = "unknown"
        return {
            "backend": self.name, "model": self.model, "revision": self.revision,
            "dtype": dtype, "device": device, "load_s": self.load_s,
            "chat_template": "harmony" if self._harmony else "plain",
            "response_format": "schema stated in the user message; validated after",
            "memory": self.memory_report,
        }

    def generate(self, system: str, user: str, *, response_schema: dict,
                 settings: dict) -> LLMRaw:
        return self.generate_many(system, [user], response_schema=response_schema,
                                  settings=settings)[0]

    def generate_many(self, system: str, users: list[str], *, response_schema: dict,
                      settings: dict) -> list[LLMRaw]:
        torch = self._torch
        tok = self.tokenizer
        conversations = [
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
            for user in users
        ]
        template_kwargs: dict[str, Any] = {}
        effort = settings.get("reasoning_effort")
        if self._harmony and effort:
            template_kwargs["reasoning_effort"] = effort
        encoded = tok.apply_chat_template(
            conversations, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, padding=True, **template_kwargs,
        )
        device = next(self.model_obj.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items() if hasattr(v, "to")}
        temperature = float(settings.get("temperature", 0.0))
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(settings.get("max_new_tokens", 2048)),
            "pad_token_id": tok.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature,
                              top_p=float(settings.get("top_p", 1.0)))
        else:
            gen_kwargs.update(do_sample=False)
        seed = settings.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model_obj.generate(**encoded, **gen_kwargs)
        wall = time.perf_counter() - started
        input_len = encoded["input_ids"].shape[1]
        results = []
        attention = encoded.get("attention_mask")
        for row in range(output.shape[0]):
            new_tokens = output[row, input_len:]
            # Trailing pad after the end of sequence is not generated text.
            if tok.pad_token_id is not None:
                keep = (new_tokens != tok.pad_token_id).nonzero()
                if len(keep):
                    new_tokens = new_tokens[: int(keep[-1]) + 1]
                else:
                    new_tokens = new_tokens[:0]
            raw_text = tok.decode(new_tokens, skip_special_tokens=False)
            content, reasoning = split_harmony(raw_text)
            if reasoning is None and "<|channel|>" not in raw_text:
                content = tok.decode(new_tokens, skip_special_tokens=True)
            prompt_tokens = int(attention[row].sum()) if attention is not None else input_len
            results.append(LLMRaw(
                content=content.strip(), prompt_tokens=prompt_tokens,
                gen_tokens=int(new_tokens.shape[0]), wall_s=wall / len(users),
                done_reason="length" if new_tokens.shape[0] >= gen_kwargs["max_new_tokens"] else "stop",
                reasoning=reasoning,
            ))
        return results


# ---------------------------------------------------------------------------
# ollama
# ---------------------------------------------------------------------------

def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "").strip() or LLM_OLLAMA_DEFAULT_HOST
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/")


class OllamaBackend:
    """A model served by a local ollama server, with grammar-constrained JSON output.

    Args:
        model_tag: The ollama tag, e.g. ``"gpt-oss:20b"``.
        host: Server URL; defaults to ``$OLLAMA_HOST`` or ``http://127.0.0.1:11434``.
        timeout: Seconds to wait for one completion (also settable per call
            through ``settings["timeout"]``).
    """

    name = "ollama"
    enforces_schema = True

    def __init__(self, model_tag: str, *, host: str | None = None, timeout: int = 900):
        self.model = str(model_tag)
        self.host = (host or _ollama_host()).rstrip("/")
        self.timeout = int(timeout)
        self.digest: str | None = None
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as response:
                tags = json.load(response).get("models", [])
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ConnectionError(
                f"No ollama server at {self.host} ({exc}). Start it (`ollama serve`) "
                "or set OLLAMA_HOST, and pull the model first (`ollama pull "
                f"{self.model}`)."
            ) from exc
        names = {m.get("name"): m for m in tags}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise ValueError(
                f"ollama at {self.host} does not have {self.model!r}. Available: "
                f"{sorted(n for n in names if n)}. Pull it with `ollama pull {self.model}`."
            )
        self.digest = (names.get(self.model) or names.get(f"{self.model}:latest") or {}).get("digest")
        logger.info("ollama %s at %s (digest %s).", self.model, self.host, (self.digest or "?")[:12])

    def supports_reasoning_effort(self) -> bool:
        return supports_reasoning_effort(self.model)

    def describe(self) -> dict:
        return {
            "backend": self.name, "model": self.model, "host": self.host,
            "digest": self.digest,
            "response_format": "JSON schema enforced by the server",
        }

    def generate(self, system: str, user: str, *, response_schema: dict,
                 settings: dict) -> LLMRaw:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "format": response_schema,
            "stream": False,
            "options": {
                "temperature": float(settings.get("temperature", 0.0)),
                "top_p": float(settings.get("top_p", 1.0)),
                "seed": int(settings.get("seed", 42)),
                "num_predict": int(settings.get("max_new_tokens", 2048)),
            },
        }
        effort = settings.get("reasoning_effort")
        if effort and self.supports_reasoning_effort():
            payload["think"] = effort
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=int(settings.get("timeout") or self.timeout)) as response:
            raw = json.load(response)
        wall = time.perf_counter() - started
        message = raw.get("message") or {}
        return LLMRaw(
            content=str(message.get("content") or ""),
            prompt_tokens=raw.get("prompt_eval_count"),
            gen_tokens=raw.get("eval_count"),
            wall_s=wall,
            done_reason=raw.get("done_reason"),
            reasoning=message.get("thinking") or None,
        )

    def generate_many(self, system: str, users: list[str], *, response_schema: dict,
                      settings: dict) -> list[LLMRaw]:
        # The server handles one request at a time per model; batching is sequential.
        return [self.generate(system, user, response_schema=response_schema, settings=settings)
                for user in users]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class FakeBackend:
    """A backend for tests: ``responder(system, user) -> str`` supplies the text."""

    name = "fake"
    enforces_schema = False

    def __init__(self, responder: Callable[[str, str], str], *, model: str = "fake",
                 reasoning_effort: bool = True):
        self.model = model
        self._responder = responder
        self._effort = reasoning_effort
        self.calls: list[dict] = []

    def supports_reasoning_effort(self) -> bool:
        return self._effort

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model}

    def generate(self, system: str, user: str, *, response_schema: dict,
                 settings: dict) -> LLMRaw:
        self.calls.append({"system": system, "user": user, "settings": dict(settings)})
        text = self._responder(system, user)
        return LLMRaw(content=text, prompt_tokens=len(system.split()) + len(user.split()),
                      gen_tokens=len(text.split()), wall_s=0.0, done_reason="stop")

    def generate_many(self, system: str, users: list[str], *, response_schema: dict,
                      settings: dict) -> list[LLMRaw]:
        return [self.generate(system, u, response_schema=response_schema, settings=settings)
                for u in users]
