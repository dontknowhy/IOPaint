"""big-lama 的 TorchScript 图重放：输出与原图一致，结构不认识时退回原模块。

`.pt` 是 tracing 产物、图改不了，所以重放在构造时先用一张随机 64² 输入跟原模块比一次，
误差超过 1e-3 就返回 None，调用方继续用原模块。这里守两件事：重放与原图的浮点差在
1e-5 以内（uint8 上只会有 ±1 LSB，模型本身每次运行就差这么多），以及未知结构确实退回。
"""
from typing import Tuple

import numpy as np
import pytest
import torch

from iopaint.helper import load_jit_model
from iopaint.model.helper.lama_graph import LaMaGeneratorReplay, build_lama_replay
from iopaint.model.helper.tensors import binary_mask, upload_uint8
from iopaint.model.lama import LAMA_MODEL_MD5, LAMA_MODEL_URL
from iopaint.model_manager import ModelManager
from iopaint.tests.utils import check_device

SIZE = 128


def _inputs(device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(0)
    image = upload_uint8(
        rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8), device
    )
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[: SIZE // 2, : SIZE // 3] = 255
    return image, binary_mask(mask, device)


def test_replay_falls_back_on_unknown_layout() -> None:
    class Stub(torch.nn.Module):
        def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return image

    assert build_lama_replay(Stub()) is None


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_replay_matches_traced_model(device: str) -> None:
    check_device(device)
    model = load_jit_model(LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()
    replay = build_lama_replay(model)
    assert isinstance(replay, LaMaGeneratorReplay)

    image, mask = _inputs(device)
    with torch.no_grad():
        expected = model(image, mask)
        actual = replay(image, mask)

    assert expected.shape == actual.shape
    assert torch.max(torch.abs(expected - actual)).item() < 1e-5


@pytest.mark.slow
def test_model_manager_uses_replay() -> None:
    check_device("cuda")
    manager = ModelManager(name="lama", device=torch.device("cuda"))
    # ModelManager.model 是 LaMa 实例，它的 .model 才是加载进来的推理模块
    assert isinstance(manager.model.model, LaMaGeneratorReplay)
