"""WebUI 性能路径：矢量掩膜、显示预览响应、ControlNet/BrushNet 模块缓存。

这些用例不加载权重，全部走 cv2 / 本地假模块，属于 fast suite。
"""
import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

import iopaint.model.controlnet as controlnet_module
from iopaint.api import (
    PREVIEW_MAX_PIXELS,
    Api,
    _display_preview_jpeg,
)
from iopaint.helper import rasterize_mask_strokes
from iopaint.model.brushnet import brushnet_wrapper
from iopaint.model.controlnet import ControlNet
from iopaint.schema import (
    ApiConfig,
    Device,
    InteractiveSegModel,
    MaskStroke,
    RealESRGANModel,
)


# ---------------------------------------------------------------------------
# 矢量掩膜
# ---------------------------------------------------------------------------


def test_rasterize_mask_strokes_round_caps_and_line_8() -> None:
    strokes = [MaskStroke(pts=[[10, 50], [110, 50]], size=5)]
    mask = rasterize_mask_strokes((100, 120), strokes)

    assert mask.shape == (100, 120)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}

    ys, xs = np.nonzero(mask)
    # LINE_8 不抗锯齿，圆头端点按 thickness//2 外扩到端点之外（方头会停在 10/110）
    assert xs.min() <= 8
    assert xs.max() >= 112
    assert ys.min() < 50 < ys.max()


def test_rasterize_mask_strokes_single_point_and_clipping() -> None:
    mask = rasterize_mask_strokes(
        (20, 20),
        [
            MaskStroke(pts=[[5, 5]], size=5),
            MaskStroke(pts=[[-100, 500], [500, -100]], size=3),
        ],
    )
    # 单点画成实心圆
    assert mask[5, 5] == 255
    # 越界坐标被夹到图像内（(-100,500)->(0,19)），不能抛异常
    assert mask[19, 0] == 255
    assert mask[0, 19] == 255


def test_rasterize_mask_strokes_empty() -> None:
    mask = rasterize_mask_strokes((16, 8), [])
    assert mask.shape == (16, 8)
    assert not mask.any()


def test_mask_stroke_schema_rejects_bad_geometry() -> None:
    with pytest.raises(ValidationError):
        MaskStroke(pts=[], size=4)
    with pytest.raises(ValidationError):
        MaskStroke(pts=[[1, 2]], size=0)
    with pytest.raises(ValidationError):
        MaskStroke(pts=[[1, 2, 3]], size=4)


# ---------------------------------------------------------------------------
# 显示预览响应
# ---------------------------------------------------------------------------


def _config(tmp_path: Path, **kwargs: Any) -> ApiConfig:
    return ApiConfig(
        host="127.0.0.1",
        port=8080,
        inbrowser=False,
        model="cv2",
        no_half=True,
        low_mem=False,
        cpu_offload=False,
        disable_nsfw_checker=True,
        local_files_only=True,
        cpu_textencoder=False,
        device=Device.cpu,
        input=None,
        mask_dir=None,
        output_dir=tmp_path,
        quality=95,
        empty_cache_after_inpaint=False,
        enable_interactive_seg=False,
        interactive_seg_model=InteractiveSegModel.vit_b,
        interactive_seg_device=Device.cpu,
        enable_remove_bg=False,
        remove_bg_device=Device.cpu,
        remove_bg_model="u2net",
        enable_anime_seg=False,
        enable_realesrgan=False,
        realesrgan_device=Device.cpu,
        realesrgan_model=RealESRGANModel.realesr_general_x4v3,
        enable_gfpgan=False,
        gfpgan_device=Device.cpu,
        enable_restoreformer=False,
        restoreformer_device=Device.cpu,
        **kwargs,
    )


def _client(tmp_path: Path, **kwargs: Any) -> Tuple[Api, TestClient]:
    api = Api(FastAPI(), _config(tmp_path, **kwargs))
    return api, TestClient(api.app)


def _png_b64(width: int, height: int) -> str:
    arr = np.full((height, width, 3), 120, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _inpaint(client: TestClient, body: dict) -> Any:
    return client.post("/api/v1/inpaint", json=body)


def _decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def test_display_preview_is_bounded_jpeg() -> None:
    arr = np.full((4200, 4200, 3), (40, 90, 160), dtype=np.uint8)
    preview = _display_preview_jpeg(arr, None)

    assert preview is not None
    assert preview.startswith(b"\xff\xd8")
    image = _decode(preview)
    assert image.size[0] * image.size[1] <= PREVIEW_MAX_PIXELS
    assert image.size == (4096, 4096)


def test_display_preview_small_image_is_untouched() -> None:
    assert (
        _display_preview_jpeg(np.zeros((10, 10, 3), dtype=np.uint8), None)
        is None
    )


def test_inpaint_accepts_strokes_without_mask(tmp_path: Path) -> None:
    _, client = _client(tmp_path)
    resp = _inpaint(
        client,
        {
            "image": _png_b64(64, 64),
            "mask_strokes": [
                {"pts": [[10, 10], [50, 50]], "size": 5},
            ],
            "hd_strategy": "Original",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["X-Image-Width"] == "64"
    assert resp.headers["X-Image-Height"] == "64"
    assert resp.headers["X-Revision"]
    _decode(resp.content)


def test_inpaint_large_image_returns_display_preview(tmp_path: Path) -> None:
    _, client = _client(tmp_path)
    resp = _inpaint(
        client,
        {
            # 17.64MP：超过 16Mi 的预览上限。空笔迹 = 不跑推理，只测响应路径。
            "image": _png_b64(4200, 4200),
            "mask_strokes": [],
            "hd_strategy": "Original",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("image/jpeg")
    assert resp.headers["X-Image-Width"] == "4200"
    assert resp.headers["X-Image-Height"] == "4200"
    image = _decode(resp.content)
    assert image.size[0] * image.size[1] <= PREVIEW_MAX_PIXELS


def test_inpaint_without_mask_or_strokes_is_rejected(tmp_path: Path) -> None:
    _, client = _client(tmp_path)
    resp = _inpaint(client, {"image": _png_b64(16, 16)})
    assert resp.status_code == 400
    assert "mask or mask_strokes" in resp.text


def test_large_image_without_cache_stays_full_resolution(tmp_path: Path) -> None:
    _, client = _client(tmp_path, revision_cache_mb=0)
    resp = _inpaint(
        client,
        {
            "image": _png_b64(4200, 4200),
            "mask_strokes": [],
            "hd_strategy": "Original",
        },
    )
    assert resp.status_code == 200, resp.text
    assert "X-Revision" not in resp.headers
    assert resp.headers["content-type"].startswith("image/png")
    assert _decode(resp.content).size == (4200, 4200)


# ---------------------------------------------------------------------------
# ControlNet / BrushNet 模块缓存
# ---------------------------------------------------------------------------


class _FakeModule:
    def __init__(self, dtype: Any) -> None:
        self.dtype = dtype
        self.device: Any = None

    def to(self, device: Any) -> "_FakeModule":
        self.device = device
        return self


def _controlnet(
    cache: Optional[Dict[str, Any]], dtype: torch.dtype = torch.float16
) -> ControlNet:
    model = object.__new__(ControlNet)  # type: ignore[type-abstract]
    model.torch_dtype = dtype
    model.local_files_only = True
    model._controlnet_cache = cache
    return model


def test_controlnet_module_cache_reuses_latest_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    def fake_from_pretrained(
        pretrained_model_name_or_path: str, **kwargs: Any
    ) -> _FakeModule:
        loaded.append(pretrained_model_name_or_path)
        return _FakeModule(kwargs["torch_dtype"])

    monkeypatch.setattr(
        controlnet_module.ControlNetModel,
        "from_pretrained",
        fake_from_pretrained,
    )
    cache: dict = {}
    model = _controlnet(cache)

    first = model._load_controlnet("method-a")
    assert model._load_controlnet("method-a") is first
    assert loaded == ["method-a"]
    assert list(cache) == ["method-a"]

    model._load_controlnet("method-b")
    assert loaded == ["method-a", "method-b"]
    # 只留最近一个 method
    assert list(cache) == ["method-b"]

    # dtype 变了（fp32 OOM 回退 fp16）必须重载，不能把 fp32 挂到 fp16 pipeline
    model.torch_dtype = torch.float32
    model._load_controlnet("method-b")
    assert loaded == ["method-a", "method-b", "method-b"]


def test_controlnet_without_cache_does_not_retain_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    def fake_from_pretrained(
        pretrained_model_name_or_path: str, **kwargs: Any
    ) -> _FakeModule:
        loaded.append(pretrained_model_name_or_path)
        return _FakeModule(kwargs["torch_dtype"])

    monkeypatch.setattr(
        controlnet_module.ControlNetModel,
        "from_pretrained",
        fake_from_pretrained,
    )
    model = _controlnet(None)
    model._load_controlnet("method-a")
    model._load_controlnet("method-a")
    assert loaded == ["method-a", "method-a"]


def test_brushnet_module_cache_reuses_latest_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[str] = []

    def fake_from_pretrained(method: str, **kwargs: Any) -> _FakeModule:
        loaded.append(method)
        return _FakeModule(kwargs["torch_dtype"])

    monkeypatch.setattr(
        brushnet_wrapper.BrushNetModel,
        "from_pretrained",
        fake_from_pretrained,
    )
    cache: dict = {}
    first = brushnet_wrapper.load_brushnet_module(
        "brush-a", cache, torch.float16, True
    )
    assert (
        brushnet_wrapper.load_brushnet_module("brush-a", cache, torch.float16, True)
        is first
    )
    brushnet_wrapper.load_brushnet_module("brush-b", cache, torch.float16, True)
    assert loaded == ["brush-a", "brush-b"]
    assert list(cache) == ["brush-b"]


def test_model_manager_injects_module_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iopaint.model_manager import ModelManager

    monkeypatch.setattr(ModelManager, "scan_models", lambda self: [])
    monkeypatch.setattr(
        ModelManager, "_load", lambda self, name, device, **kwargs: (None, device)
    )
    manager = ModelManager("lama", torch.device("cpu"))

    assert manager.kwargs["controlnet_module_cache"] is manager.controlnet_module_cache
    assert manager.kwargs["brushnet_module_cache"] is manager.brushnet_module_cache
