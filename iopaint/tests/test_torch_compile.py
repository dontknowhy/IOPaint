"""torch.compile gate / fallback tests (no GPU, no weights, no real Inductor)."""
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch

from iopaint.model import torch_compile as tc


class _Scale(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class _ConfiguredScale(_Scale):
    config: Dict[str, int] = {"in_channels": 9}


class _Boom(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("synthetic compile failure")


class _Model:
    supports_torch_compile = True

    def __init__(self, unet: torch.nn.Module | None = None) -> None:
        self.model = SimpleNamespace(unet=unet)


class _NoCompileModel:
    supports_torch_compile = False
    model = SimpleNamespace(unet=None)


def _kwargs(**overrides: Any) -> Dict[str, Any]:
    return {"torch_compile": True, **overrides}


def _compile_scale(module: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
    return _Scale()


def _compile_boom(module: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
    return _Boom()


class TestCompileGate:
    def test_requires_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (8, 0))
        assert not tc.can_compile_unet(torch.device("cuda"), _Model(), {})

    def test_accepts_ampere(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (8, 0))
        assert tc.can_compile_unet(torch.device("cuda"), _Model(), _kwargs())

    def test_rejects_pascal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (6, 1))
        assert not tc.can_compile_unet(torch.device("cuda"), _Model(), _kwargs())

    def test_rejects_non_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (8, 0))
        assert not tc.can_compile_unet(torch.device("cpu"), _Model(), _kwargs())

    def test_rejects_cpu_offload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (8, 0))
        assert not tc.can_compile_unet(
            torch.device("cuda"), _Model(), _kwargs(cpu_offload=True)
        )

    def test_rejects_unsupported_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (8, 0))
        assert not tc.can_compile_unet(
            torch.device("cuda"), _NoCompileModel(), _kwargs()
        )

    def test_rejects_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (9, 0))
        monkeypatch.setattr(torch.version, "hip", "6.0")
        assert not tc.can_compile_unet(torch.device("cuda"), _Model(), _kwargs())


class TestCompiledUNet:
    def test_uses_compiled_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[Dict[str, Any]] = []

        def fake_compile(module: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
            calls.append(kwargs)
            return _Scale()

        monkeypatch.setattr(tc, "_COMPILE", fake_compile)
        wrapper = tc.CompiledUNet(_Scale())
        out = wrapper(torch.tensor([2.0]))
        assert out.tolist() == [4.0]
        assert not wrapper._disabled
        assert calls == [{"dynamic": True, "fullgraph": False}]

    def test_failure_falls_back_to_eager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        original = _Scale()
        pipe = SimpleNamespace(unet=original)

        def fake_compile(module: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
            calls.append(1)
            return _Boom()

        monkeypatch.setattr(tc, "_COMPILE", fake_compile)
        wrapper = tc.CompiledUNet(original, pipe)
        pipe.unet = wrapper
        out = wrapper(torch.tensor([2.0]))
        assert out.tolist() == [4.0]
        assert wrapper._disabled
        assert pipe.unet is original
        # A disabled wrapper must not call the compiled module again.
        assert wrapper(torch.tensor([3.0])).tolist() == [6.0]
        assert calls == [1]

    def test_lora_disables_compiled_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tc, "_COMPILE", _compile_scale)
        original = _Scale()
        pipe = SimpleNamespace(unet=original)
        wrapper = tc.CompiledUNet(original, pipe)
        pipe.unet = wrapper
        assert wrapper(torch.tensor([2.0])).tolist() == [4.0]
        wrapper.set_lora_enabled(True)
        assert wrapper._disabled
        # diffusers' LoRA loader must see the real module again.
        assert pipe.unet is original
        assert wrapper(torch.tensor([2.0])).tolist() == [4.0]

    def test_forwards_original_attributes(self) -> None:
        wrapper = tc.CompiledUNet(_ConfiguredScale())
        assert wrapper.config == {"in_channels": 9}

    def test_apply_reaches_original_parameters(self) -> None:
        original = torch.nn.Linear(2, 2)
        wrapper = tc.CompiledUNet(original)
        wrapper.to(torch.float64)
        assert original.weight.dtype == torch.float64
        # A pipeline-level .to() walks children through _apply.
        torch.nn.Sequential(wrapper).to(torch.float32)
        assert original.weight.dtype == torch.float32


class TestMaybeCompileUnet:
    def test_wraps_supported_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (8, 0))
        monkeypatch.setattr(tc, "_COMPILE", _compile_scale)
        original = _Scale()
        model = _Model(original)

        wrapper = tc.maybe_compile_unet(model, torch.device("cuda"), _kwargs())

        assert isinstance(wrapper, tc.CompiledUNet)
        assert isinstance(model.model.unet, tc.CompiledUNet)
        assert wrapper._pipe is model.model

    def test_keeps_eager_on_old_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (6, 1))
        original = _Scale()
        model = _Model(original)

        wrapper = tc.maybe_compile_unet(model, torch.device("cuda"), _kwargs())

        assert wrapper is None
        assert model.model.unet is original

    def test_keeps_eager_without_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (8, 0))
        original = _Scale()
        model = _Model(original)

        assert tc.maybe_compile_unet(model, torch.device("cuda"), {}) is None
        assert model.model.unet is original

    def test_keeps_eager_with_cpu_offload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tc, "_cuda_capability", lambda device: (8, 0))
        original = _Scale()
        model = _Model(original)

        assert (
            tc.maybe_compile_unet(
                model, torch.device("cuda"), _kwargs(cpu_offload=True)
            )
            is None
        )
        assert model.model.unet is original
