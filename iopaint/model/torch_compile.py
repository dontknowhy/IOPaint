"""Optional runtime ``torch.compile`` for diffusion UNets.

Only plain SD/SDXL models opt in (``InpaintModel.supports_torch_compile``).  The
Inductor CUDA backend needs Triton, which requires compute capability >= 7.0;
older GPUs (and ROCm/CPU/MPS) stay on eager instead of failing at request time.

The compiled call is lazy: the first request pays the compilation cost.  Any
failure disables the wrapper for that model and replays the call on the eager
module, so a Dynamo/Inductor problem cannot turn into a 500.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

# Triton's CUDA backend refuses sm < 7.0 (Pascal).  Keep the constant here so
# tests can pin the gate without touching torch internals.
MIN_CUDA_CAPABILITY = 7

_COMPILE = getattr(torch, "compile", None)


def _cuda_capability(device: torch.device) -> Optional[Tuple[int, int]]:
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        return None
    return int(major), int(minor)


def can_compile_unet(device: torch.device, model: Any, kwargs: Dict[str, Any]) -> bool:
    """Whether this model/device/request combination may use torch.compile.

    Kept as a pure predicate so the gate is testable without a real GPU.
    """
    if not kwargs.get("torch_compile", False):
        return False
    if not getattr(model, "supports_torch_compile", False):
        return False
    if kwargs.get("cpu_offload", False):
        return False
    if getattr(device, "type", None) != "cuda":
        return False
    # Triton on ROCm has different constraints; keep the first version CUDA-only.
    if getattr(torch.version, "hip", None) is not None:
        return False
    capability = _cuda_capability(device)
    return capability is not None and capability[0] >= MIN_CUDA_CAPABILITY


class CompiledUNet(torch.nn.Module):
    """``pipe.unet`` replacement that compiles lazily and never breaks a request.

    The original module is kept as the source of truth.  Attribute access is
    forwarded so diffusers can still read ``config``/``dtype``/``device``.
    """

    _original: torch.nn.Module
    _compiled: Optional[torch.nn.Module]
    _disabled: bool
    _compile_lock: "threading.Lock"
    _pipe: Any

    def __init__(self, original: torch.nn.Module, pipe: Any = None):
        super().__init__()
        # Keep both modules out of ``_modules``: the compiled wrapper already
        # references the original weights, and registering both would duplicate
        # parameters in state_dict()/parameters() for no inference benefit.
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_compiled", None)
        object.__setattr__(self, "_disabled", False)
        object.__setattr__(self, "_compile_lock", threading.Lock())
        object.__setattr__(self, "_pipe", pipe)

    def __getattr__(self, name: str) -> Any:
        if not name.startswith("_"):
            original = self.__dict__.get("_original")
            if original is not None:
                return getattr(original, name)
        return super().__getattr__(name)

    def _ensure_compiled(self) -> Optional[torch.nn.Module]:
        if _COMPILE is None:
            self._disable("torch.compile is not available in this PyTorch build")
            return None
        if self._compiled is None:
            with self._compile_lock:
                if self._compiled is None:
                    # dynamic=True avoids one recompile per crop size; fullgraph
                    # stays off so an unsupported op falls back to eager instead
                    # of aborting the whole request.  Keep the result out of
                    # ``_modules`` (see __init__).
                    object.__setattr__(
                        self,
                        "_compiled",
                        _COMPILE(self._original, dynamic=True, fullgraph=False),
                    )
        return self._compiled

    def _disable(self, reason: str) -> None:
        if self._disabled:
            return
        self._disabled = True
        logger.warning(f"{reason}; using the eager UNet for this model")
        self._restore_pipe()

    def _restore_pipe(self) -> None:
        """Put the original UNet back so diffusers sees real modules again.

        LoRA loading walks ``unet`` and mutates its attention processors;
        leaving the thin wrapper installed would hide those submodules.
        """
        pipe = self._pipe
        if pipe is None:
            return
        try:
            if getattr(pipe, "unet", None) is self:
                pipe.unet = self._original
        except Exception as e:  # pragma: no cover - attribute assignment is defensive
            logger.warning(f"could not restore the eager UNet ({e})")

    def set_lora_enabled(self, enabled: bool) -> None:
        """LCM LoRA mutates the UNet; stay eager once it is used.

        Re-enabling LoRA after disable_lora() can leave guards/graphs stale, so
        this is a one-way switch for the lifetime of the loaded model.
        """
        if enabled:
            self._disable("LCM LoRA is enabled")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._disabled:
            return self._original(*args, **kwargs)
        compiled = self._ensure_compiled()
        if compiled is None:
            return self._original(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - fallback is the whole point
            first_line = str(e).splitlines()[0] if str(e) else ""
            self._disable(f"torch.compile failed ({type(e).__name__}: {first_line})")
            return self._original(*args, **kwargs)

    def train(self, mode: bool = True) -> "CompiledUNet":
        self._original.train(mode)
        if self._compiled is not None:
            self._compiled.train(mode)
        return self

    def to(self, *args: Any, **kwargs: Any) -> "CompiledUNet":
        self._original.to(*args, **kwargs)
        if self._compiled is not None:
            self._compiled.to(*args, **kwargs)
        return self

    def _apply(self, fn: Any, recurse: bool = True) -> "CompiledUNet":
        # ``nn.Module.to()/half()/cuda()`` on the pipeline walk children through
        # ``_apply``; without this the wrapper's empty parameter list would silently
        # skip the real UNet.
        self._original._apply(fn, recurse=recurse)
        if self._compiled is not None:
            self._compiled._apply(fn, recurse=recurse)
        return self


def maybe_compile_unet(
    model: Any, device: torch.device, kwargs: Dict[str, Any]
) -> Optional[CompiledUNet]:
    """Replace ``model.model.unet`` with :class:`CompiledUNet` when allowed."""
    if not kwargs.get("torch_compile", False):
        return None
    if not getattr(model, "supports_torch_compile", False):
        logger.info(
            "torch.compile is only enabled for plain SD/SDXL models; using eager"
        )
        return None
    if kwargs.get("cpu_offload", False):
        logger.warning("torch.compile is ignored because --cpu-offload is enabled")
        return None
    if not can_compile_unet(device, model, kwargs):
        logger.warning(
            "torch.compile requires CUDA compute capability >= %s; using eager",
            MIN_CUDA_CAPABILITY,
        )
        return None

    pipe = getattr(model, "model", None)
    if pipe is None:
        logger.warning("torch.compile skipped: model pipeline has no UNet")
        return None
    original = getattr(pipe, "unet", None)
    if original is None:
        logger.warning("torch.compile skipped: model pipeline has no UNet")
        return None

    wrapper = CompiledUNet(original, pipe)
    try:
        pipe.unet = wrapper
    except Exception as e:  # pragma: no cover - attribute assignment is defensive
        logger.warning(f"torch.compile skipped: cannot replace UNet ({e})")
        return None
    logger.info(
        "torch.compile enabled for the UNet; the first inference will compile the graph"
    )
    return wrapper
