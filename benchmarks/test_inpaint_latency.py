"""推理延迟跑分：LaMa 单次 + SD1.5 UNet 单步。

本地跑，**不进 CI、不进 fast suite**：

    python benchmarks/run.py -k lama
    IOPAINT_BENCH_GPU=1 python benchmarks/run.py -k "lama or sd15"

`pytest.ini` 的 `testpaths = iopaint/tests`，所以 `pytest -m "not slow"` 与
`scripts/check.sh` 不会收集本文件。权重不存在时自动 skip；没有 CUDA 时自动 skip。

**必须写明跑分用的显卡**：多卡机器上 `cuda:0` 不一定是最快或最空闲的那块。
每条结果都带显卡名/显存/算力（`extra.gpu`），换机器后可以直接对比；显存不够跑
fp32 UNet 的卡会 skip 而不是 OOM。计时前后调用 `torch.cuda.synchronize`，
否则拿到的是 kernel 入队时间。
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from harness import add_extra, measure
from iopaint.model.lama import LaMa
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, InpaintRequest

pytestmark = [pytest.mark.bench, pytest.mark.slow]

WARMUP = 2
RUNS = 7

# 用哪块卡由环境变量选，默认 0；多卡时显式指定，避免把慢卡/小显存卡的数字当成整机结果
GPU_ENV = "IOPAINT_BENCH_GPU"
# SD1.5 inpainting UNet fp32 batch=2 实测峰值 ~3.9 GB，加上驱动/上下文留到 5 GB；
# 显存更小的卡会 OOM，直接 skip 并说明原因
MIN_UNET_MEMORY_MIB = 5120


def _bench_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the inference benchmark")
    index = int(os.environ.get(GPU_ENV, "0"))
    if index >= torch.cuda.device_count():
        pytest.fail(
            f"{GPU_ENV}={index} but only {torch.cuda.device_count()} CUDA device(s) visible"
        )
    return torch.device(f"cuda:{index}")


def _device_label(device: torch.device) -> str:
    props = torch.cuda.get_device_properties(device)
    return (
        f"{props.name} cuda:{device.index} "
        f"{props.total_memory / 2**20:.0f}MiB sm_{props.major}{props.minor}"
    )


@pytest.fixture(scope="module")
def bench_device() -> torch.device:
    return _bench_device()


@pytest.fixture(scope="module")
def lama_model(bench_device: torch.device):
    if not LaMa.is_downloaded():
        pytest.skip("big-lama.pt is not downloaded")
    return ModelManager(name="lama", device=bench_device)


@pytest.mark.parametrize("size", [256, 512, 1024])
def test_lama_latency(lama_model, bench_device: torch.device, size: int) -> None:
    image = np.full((size, size, 3), 128, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[size // 4 : size // 2, size // 4 : size // 2] = 255
    config = InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)

    def run() -> None:
        lama_model(image, mask, config)

    torch.cuda.reset_peak_memory_stats(bench_device)
    measure(
        f"lama.{size}",
        run,
        warmup=WARMUP,
        rounds=RUNS,
        sync=lambda: torch.cuda.synchronize(bench_device),
        extra={"gpu": _device_label(bench_device)},
    )
    add_extra(
        f"lama.{size}",
        peak_alloc_mb=round(torch.cuda.max_memory_allocated(bench_device) / 2**20),
        peak_reserved_mb=round(torch.cuda.max_memory_reserved(bench_device) / 2**20),
    )


@pytest.fixture(scope="module")
def sd15_unet(bench_device: torch.device):
    try:
        from diffusers import UNet2DConditionModel
    except ImportError as e:  # pragma: no cover - requirements pin diffusers
        pytest.skip(f"diffusers unavailable: {e}")

    total_mib = torch.cuda.get_device_properties(bench_device).total_memory / 2**20
    if total_mib < MIN_UNET_MEMORY_MIB:
        pytest.skip(
            f"SD1.5 UNet fp32 needs ~{MIN_UNET_MEMORY_MIB} MiB; "
            f"{_device_label(bench_device)} is too small (pick another {GPU_ENV})"
        )

    capability = torch.cuda.get_device_capability(bench_device)
    dtype = torch.float16 if capability[0] >= 7 else torch.float32
    try:
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            subfolder="unet",
            local_files_only=True,
            variant="fp16",
            torch_dtype=torch.float16,
        )
    except (OSError, EnvironmentError) as e:
        pytest.skip(f"SD1.5 inpainting UNet is not cached: {e}")

    if dtype == torch.float32:
        unet = unet.to(dtype=dtype)
    return unet.to(bench_device).eval()


def test_sd15_unet_step(sd15_unet, bench_device: torch.device) -> None:
    cross_dim = sd15_unet.config.cross_attention_dim
    in_channels = sd15_unet.config.in_channels
    sample = torch.randn(2, in_channels, 64, 64, device=bench_device)
    timestep = torch.tensor([500], device=bench_device)
    encoder = torch.randn(2, 77, cross_dim, device=bench_device)

    def run() -> None:
        with torch.inference_mode():
            sd15_unet(sample, timestep, encoder)

    torch.cuda.reset_peak_memory_stats(bench_device)
    measure(
        "sd15.unet.step",
        run,
        warmup=2,
        rounds=5,
        sync=lambda: torch.cuda.synchronize(bench_device),
        extra={
            "gpu": _device_label(bench_device),
            "dtype": str(sd15_unet.dtype).replace("torch.", ""),
        },
    )
    add_extra(
        "sd15.unet.step",
        peak_alloc_mb=round(torch.cuda.max_memory_allocated(bench_device) / 2**20),
        peak_reserved_mb=round(torch.cuda.max_memory_reserved(bench_device) / 2**20),
    )
