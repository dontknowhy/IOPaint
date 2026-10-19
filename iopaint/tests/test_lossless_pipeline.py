"""无损盒子端到端：掩膜外像素、多轮导出、revision 协议与 DCT 局部导出。"""
import base64
import io
import math
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, JpegImagePlugin

from iopaint.api import Api
from iopaint.schema import (
    ApiConfig,
    Device,
    InteractiveSegModel,
    RealESRGANModel,
)


def _config(tmp_path: Path, **kwargs: Any) -> ApiConfig:
    cfg = ApiConfig(
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
    return cfg


def _client(tmp_path: Path, **kwargs: Any) -> Tuple[Api, TestClient]:
    api = Api(FastAPI(), _config(tmp_path, **kwargs))
    return api, TestClient(api.app)


def _encode_jpeg(
    arr: np.ndarray,
    quality: int = 95,
    subsampling: Optional[int] = None,
    progressive: bool = False,
) -> bytes:
    buf = io.BytesIO()
    kwargs: dict[str, Any] = {
        "format": "JPEG",
        "quality": quality,
        "progressive": progressive,
    }
    if subsampling is not None:
        kwargs["subsampling"] = subsampling
    Image.fromarray(arr).save(buf, **kwargs)
    return buf.getvalue()


def _b64(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _mask_b64(shape: Tuple[int, int, int], box: Tuple[int, int, int, int]) -> str:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 255
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return _b64(buf.getvalue(), "image/png")


def _decode(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def _inpaint(
    client: TestClient,
    image_b64: Optional[str] = None,
    revision: Optional[str] = None,
    mask: Optional[str] = None,
) -> Any:
    body: dict[str, Any] = {
        "mask": mask,
        "hd_strategy": "Original",
        "cv2_radius": 3,
    }
    if image_b64 is not None:
        body["image"] = image_b64
    if revision is not None:
        body["revision"] = revision
    return client.post("/api/v1/inpaint", json=body)


def _export(client: TestClient, **body: Any) -> Any:
    return client.post("/api/v1/export", json=body)


def test_jpeg_ten_rounds_keep_unmasked_pixels_exact(tmp_path: Path) -> None:
    arr = np.zeros((96, 128, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.arange(128, dtype=np.uint8), (96, 1))
    arr[:, :, 1] = np.tile(np.arange(96, dtype=np.uint8)[:, None], (1, 128))
    arr[:, :, 2] = 90
    source = _encode_jpeg(arr)
    first_decoded = _decode(source)

    api, client = _client(tmp_path)
    boxes = [(8 + i * 9, 12, 16 + i * 9, 20) for i in range(10)]
    revision = None
    for i, box in enumerate(boxes):
        resp = _inpaint(
            client,
            image_b64=None if revision else _b64(source),
            revision=revision,
            mask=_mask_b64(arr.shape, box),
        )
        assert resp.status_code == 200, resp.text
        revision = resp.headers.get("X-Revision")
        assert revision, "无损路径必须返回 X-Revision"

    out = _export(client, revision=revision, format="png")
    assert out.status_code == 200
    canonical = _decode(out.content)

    outside = np.ones(arr.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in boxes:
        outside[y1:y2, x1:x2] = False
    assert np.abs(canonical.astype(int) - first_decoded.astype(int))[outside].max() == 0
    assert api.revision_cache.stats["hits_revision"] >= 9


def test_export_quality_does_not_drift_across_rounds(tmp_path: Path) -> None:
    arr = np.zeros((64, 96, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.arange(96, dtype=np.uint8), (64, 1))
    arr[:, :, 1] = np.tile(np.arange(64, dtype=np.uint8)[:, None], (1, 96))
    arr[:, :, 2] = 120
    source = _encode_jpeg(arr)
    mask = _mask_b64(arr.shape, (24, 16, 56, 40))

    _, client = _client(tmp_path)
    revision = None
    psnrs = []
    for _ in range(10):
        resp = _inpaint(
            client,
            image_b64=None if revision else _b64(source),
            revision=revision,
            mask=mask,
        )
        assert resp.status_code == 200
        revision = resp.headers["X-Revision"]

        canonical = _decode(
            _export(client, revision=revision, format="png").content
        )
        jpeg = _export(
            client,
            revision=revision,
            format="jpeg",
            quality=100,
            subsampling="444",
            export_mode="full",
        )
        psnrs.append(_psnr(_decode(jpeg.content), canonical))

    assert abs(psnrs[0] - psnrs[-1]) < 0.1


def test_export_default_is_q100_444(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    # 加一点结构，避免纯噪声把 PSNR 拉到 40dB 以下
    arr[:32, :32] = (arr[:32, :32] // 2 + 64)
    source = _encode_jpeg(arr)
    _, client = _client(tmp_path)
    resp = _export(
        client,
        image=_b64(source),
        format="jpeg",
        export_mode="full",
    )
    assert resp.status_code == 200
    img = Image.open(io.BytesIO(resp.content))
    assert JpegImagePlugin.get_sampling(img) == 0
    assert _psnr(_decode(resp.content), _decode(source)) > 45


def test_export_keep_uses_source_sampling(tmp_path: Path) -> None:
    arr = np.full((32, 32, 3), 120, dtype=np.uint8)
    source = _encode_jpeg(arr, quality=80, subsampling=2)
    _, client = _client(tmp_path)
    resp = _export(
        client,
        image=_b64(source),
        format="jpeg",
        subsampling="keep",
        export_mode="full",
    )
    assert resp.status_code == 200
    assert JpegImagePlugin.get_sampling(Image.open(io.BytesIO(resp.content))) == 2


def test_revision_only_request_and_410(tmp_path: Path) -> None:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :, 0] = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    arr[:, :, 1] = 100
    arr[:, :, 2] = 60
    source = _encode_jpeg(arr)
    _, client = _client(tmp_path)

    missing = _inpaint(client, revision="deadbeef", mask=_mask_b64(arr.shape, (4, 4, 12, 12)))
    assert missing.status_code == 410
    missing_export = _export(client, revision="deadbeef")
    assert missing_export.status_code == 410

    first = _inpaint(
        client, image_b64=_b64(source), mask=_mask_b64(arr.shape, (4, 4, 12, 12))
    )
    rev = first.headers["X-Revision"]
    second = _inpaint(client, revision=rev, mask=_mask_b64(arr.shape, (16, 16, 24, 24)))
    assert second.status_code == 200
    assert second.headers["X-Revision"] != rev


def test_p0_response_hash_hit_without_revision(tmp_path: Path) -> None:
    arr = np.full((48, 48, 3), (30, 100, 200), dtype=np.uint8)
    source = _encode_jpeg(arr)
    api, client = _client(tmp_path)

    first = _inpaint(
        client, image_b64=_b64(source), mask=_mask_b64(arr.shape, (8, 8, 24, 24))
    )
    assert first.status_code == 200
    # 老客户端：下一轮把上一轮响应字节原样传回来
    second = _inpaint(
        client,
        image_b64=_b64(first.content),
        mask=_mask_b64(arr.shape, (24, 24, 40, 40)),
    )
    assert second.status_code == 200
    assert api.revision_cache.stats["hits_hash"] >= 1


def test_stateless_lossless_wire_format(tmp_path: Path) -> None:
    arr = np.full((32, 32, 3), 80, dtype=np.uint8)
    source = _encode_jpeg(arr)
    _, client = _client(tmp_path, stateless_lossless=True)
    resp = _inpaint(
        client, image_b64=_b64(source), mask=_mask_b64(arr.shape, (4, 4, 12, 12))
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert resp.headers["X-Internal-Format"] == "png"


def test_stateless_lossless_webp_wire_format(tmp_path: Path) -> None:
    arr = np.full((32, 32, 3), 80, dtype=np.uint8)
    arr[:, :16, 0] = 140
    source = _encode_jpeg(arr)
    mask = _mask_b64(arr.shape, (4, 4, 12, 12))

    _, png_client = _client(tmp_path / "png", stateless_lossless=True)
    png = _inpaint(png_client, image_b64=_b64(source), mask=mask)
    assert png.status_code == 200

    _, webp_client = _client(
        tmp_path / "webp", stateless_lossless=True, wire_format="webp"
    )
    webp = _inpaint(webp_client, image_b64=_b64(source), mask=mask)
    assert webp.status_code == 200
    assert webp.headers["content-type"].startswith("image/webp")
    assert webp.headers["X-Internal-Format"] == "webp"
    # lossless WebP 与 PNG 承载的是同一份像素
    assert np.array_equal(_decode(webp.content), _decode(png.content))


def test_cache_disabled_still_serves_legacy_path(tmp_path: Path) -> None:
    arr = np.full((32, 32, 3), 80, dtype=np.uint8)
    source = _encode_jpeg(arr)
    _, client = _client(tmp_path, revision_cache_mb=0)
    resp = _inpaint(
        client, image_b64=_b64(source), mask=_mask_b64(arr.shape, (4, 4, 12, 12))
    )
    assert resp.status_code == 200
    assert "X-Revision" not in resp.headers


def test_empty_mask_does_not_add_revision(tmp_path: Path) -> None:
    arr = np.full((32, 32, 3), 80, dtype=np.uint8)
    source = _encode_jpeg(arr)
    _, client = _client(tmp_path)
    first = _inpaint(
        client, image_b64=_b64(source), mask=_mask_b64(arr.shape, (4, 4, 12, 12))
    )
    rev = first.headers["X-Revision"]
    empty = np.zeros(arr.shape[:2], dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(empty).save(buf, format="PNG")
    second = _inpaint(client, revision=rev, mask=_b64(buf.getvalue(), "image/png"))
    assert second.status_code == 200
    assert second.headers["X-Revision"] == rev


def test_dct_patch_export_keeps_outside_pixels(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    arr[:32, :32] = (arr[:32, :32] // 3 + 40)
    source = _encode_jpeg(arr, subsampling=0)
    first_decoded = _decode(source)
    box = (16, 16, 48, 48)

    _, client = _client(tmp_path)
    inpainted = _inpaint(
        client, image_b64=_b64(source), mask=_mask_b64(arr.shape, box)
    )
    rev = inpainted.headers["X-Revision"]
    out = _export(client, revision=rev, format="jpeg")
    assert out.status_code == 200
    assert out.headers["X-Export-Mode"] == "dct_patch"

    decoded = _decode(out.content)
    mask = np.zeros(arr.shape[:2], dtype=bool)
    mask[box[1] : box[3], box[0] : box[2]] = True
    assert np.abs(decoded.astype(int) - first_decoded.astype(int))[~mask].max() == 0


def test_dct_patch_falls_back_for_progressive(tmp_path: Path) -> None:
    arr = np.full((32, 32, 3), 120, dtype=np.uint8)
    source = _encode_jpeg(arr, progressive=True)
    _, client = _client(tmp_path)
    first = _inpaint(
        client, image_b64=_b64(source), mask=_mask_b64(arr.shape, (8, 8, 24, 24))
    )
    rev = first.headers["X-Revision"]
    out = _export(client, revision=rev, format="jpeg")
    assert out.status_code == 200
    assert out.headers["X-Export-Mode"] == "full"
