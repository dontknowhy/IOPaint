"""`LDM.forward` 的 amp autocast 跟随模型所在设备，不写死 cuda。

ROCm 的 `device.type` 也是 "cuda"，但 Intel XPU 报的是 "xpu"；写死 "cuda" 在 xpu 上
等于没开 autocast。这里不加载权重，只验证包装器把 `self.device.type` 交给
`torch.autocast`，并让 cpu/mps 保持旧行为（enabled=False）。
"""
import pytest
import torch

import iopaint.model.ldm as ldm


def test_autocast_uses_model_device_type(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    class FakeAutocast:
        def __init__(self, device_type: str, enabled: bool = True) -> None:
            seen.append((device_type, enabled))

        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(ldm.torch, "autocast", FakeAutocast)

    @ldm._autocast_on_device
    def forward(self: object, image: object) -> str:
        return "result"

    class Xpu:
        device = torch.device("xpu")

    class Cpu:
        device = torch.device("cpu")

    assert forward(Xpu(), None) == "result"
    assert forward(Cpu(), None) == "result"
    assert seen == [("xpu", True), ("cpu", False)]


def test_autocast_follows_cuda_device_index(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    class FakeAutocast:
        def __init__(self, device_type: str, enabled: bool = True) -> None:
            seen.append((device_type, enabled))

        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(ldm.torch, "autocast", FakeAutocast)

    @ldm._autocast_on_device
    def forward(self: object, image: object) -> str:
        return "result"

    class Cuda1:
        device = torch.device("cuda:1")

    assert forward(Cuda1(), None) == "result"
    assert seen == [("cuda", True)]
