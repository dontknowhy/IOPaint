"""动态上下文（hd_strategy=Dynamic）的决策与回贴行为。

不加载权重：假模型只记录收到的窗口尺寸并返回纯色/原图，所有断言都落在
`dynamic_context` 的几何决策与 `InpaintModel._dynamic_context_call` 的贴回范围上。
"""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest
from fastapi import FastAPI

from iopaint.api import Api
from iopaint.model.base import InpaintModel
from iopaint.model.helper.dynamic_context import (
    ContextDecision,
    complexity_score,
    decide_context,
    geometry_pad,
    mask_area_ratio,
    ring_metrics,
    seam_rerun_pad,
    seam_score,
)
from iopaint.schema import ApiConfig, DynamicContextConfig, HDStrategy, InpaintRequest
from iopaint.tests.test_api import _minimal_config, _png_image, _png_mask


class RecordingErase(InpaintModel):
    name = "recording_erase"
    pad_mod = 8

    def __init__(self, mode: str = "constant", fill: int = 200) -> None:
        self.forward_shapes: list[Tuple[int, int]] = []
        self.mode = mode
        self.fill = fill
        super().__init__("cpu")

    @staticmethod
    def is_downloaded() -> bool:
        return True

    def init_model(self, device: str, **kwargs: object) -> None:
        pass

    def forward(
        self, image: np.ndarray, mask: np.ndarray, config: InpaintRequest
    ) -> np.ndarray:
        self.forward_shapes.append(tuple(image.shape[:2]))
        if self.mode == "copy":
            return np.ascontiguousarray(image)
        return np.full_like(image, self.fill)


def dynamic_cfg(**kwargs: Any) -> DynamicContextConfig:
    return DynamicContextConfig(**kwargs)


def flat_case(
    height: int = 400,
    width: int = 400,
    box: Tuple[int, int, int, int] = (180, 180, 220, 220),
    value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    image = np.full((height, width, 3), value, np.uint8)
    mask = np.zeros((height, width), np.uint8)
    top, left, bottom, right = box
    mask[top:bottom, left:right] = 255
    return image, mask


def textured_case(
    box: Tuple[int, int, int, int] = (180, 180, 220, 220), period: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    height = width = 400
    yy, xx = np.mgrid[0:height, 0:width]
    checker = ((yy // period + xx // period) % 2) * 255
    image = np.stack([checker, checker, checker], axis=-1).astype(np.uint8)
    mask = np.zeros((height, width), np.uint8)
    top, left, bottom, right = box
    mask[top:bottom, left:right] = 255
    return image, mask


def noise_patch(seed: int = 7, size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), np.uint8)
    mask[80:120, 80:120] = 255
    return image, mask


# ---------------------------------------------------------------------------
# 几何决策
# ---------------------------------------------------------------------------
class TestGeometry:
    def test_area_ratio(self) -> None:
        _, mask = flat_case()
        assert mask_area_ratio(mask) == pytest.approx(40 * 40 / (400 * 400))
        assert mask_area_ratio(np.zeros((10, 10), np.uint8)) == 0.0
        assert mask_area_ratio(np.full((10, 10), 255, np.uint8)) == 1.0

    def test_geometry_pad_tiers(self) -> None:
        cfg = dynamic_cfg()
        assert geometry_pad(0.001, cfg) == cfg.base_pad
        assert geometry_pad(0.06, cfg) == int(cfg.base_pad * 1.5 + 0.5)
        assert geometry_pad(0.2, cfg) == cfg.area_high_pad

    def test_large_mask_skips_complexity(self) -> None:
        height = width = 400
        image = np.zeros((height, width, 3), np.uint8)
        mask = np.zeros((height, width), np.uint8)
        mask[100:300, 100:300] = 255  # 25% -> 几何 pad 256 < skip 384
        cfg = dynamic_cfg(area_ratio_high=0.15, area_high_pad=384)
        decision = decide_context(image, mask, (100, 100, 300, 300), cfg)
        assert decision.geometry_pad == 384
        assert decision.pad == 384
        assert decision.metrics is None
        assert "complexity_skipped" in decision.flags

    def test_force_pad_overrides_metrics(self) -> None:
        image, mask = noise_patch()
        decision = decide_context(image, mask, (80, 80, 120, 120), dynamic_cfg(force_pad=77))
        assert decision.pad == 77
        assert "forced_pad" in decision.flags
        assert decision.metrics is None

    def test_edge_clamped_window_stays_in_bounds(self) -> None:
        image, mask = flat_case(box=(0, 0, 40, 40))
        decision = decide_context(image, mask, (0, 0, 40, 40), dynamic_cfg())
        left, top, right, bottom = decision.rect
        assert (left, top) == (0, 0)
        assert right <= 400 and bottom <= 400
        assert "edge_clamped" in decision.flags

    def test_full_mask_has_empty_ring(self) -> None:
        image = np.zeros((100, 100, 3), np.uint8)
        mask = np.full((100, 100), 255, np.uint8)
        decision = decide_context(image, mask, (0, 0, 100, 100), dynamic_cfg())
        assert "ring_empty" in decision.flags
        assert decision.complexity == 0.0

    def test_deterministic(self) -> None:
        image, mask = noise_patch()
        cfg = dynamic_cfg()
        first = decide_context(image, mask, (80, 80, 120, 120), cfg)
        second = decide_context(image, mask.copy(), (80, 80, 120, 120), cfg)
        assert first.to_dict() == second.to_dict()


# ---------------------------------------------------------------------------
# ring 复杂度
# ---------------------------------------------------------------------------
class TestComplexity:
    def test_flat_background_is_low(self) -> None:
        image, mask = flat_case()
        decision = decide_context(image, mask, (180, 180, 220, 220), dynamic_cfg())
        assert decision.complexity < 0.05
        assert decision.pad == dynamic_cfg().base_pad

    def test_noise_is_more_complex_than_stripes(self) -> None:
        image, mask = noise_patch()
        noise = ring_metrics(image, mask, dynamic_cfg())
        striped_image, striped_mask = textured_case()
        stripes = ring_metrics(striped_image, striped_mask, dynamic_cfg())
        cfg = dynamic_cfg()
        assert complexity_score(noise, cfg) > complexity_score(stripes, cfg)
        assert complexity_score(noise, cfg) > 0.5

    def test_complexity_raises_pad(self) -> None:
        image, mask = noise_patch()
        decision = decide_context(image, mask, (80, 80, 120, 120), dynamic_cfg())
        assert decision.dynamic_pad > dynamic_cfg().base_pad
        assert decision.pad == decision.dynamic_pad
        assert "forced_pad" not in decision.flags

    def test_pad_never_exceeds_max(self) -> None:
        image, mask = noise_patch()
        cfg = dynamic_cfg(max_pad=96)
        decision = decide_context(image, mask, (80, 80, 120, 120), cfg)
        assert decision.pad == 96
        assert "pad_capped" in decision.flags


# ---------------------------------------------------------------------------
# 裁块与回贴
# ---------------------------------------------------------------------------
class TestDynamicCall:
    def request(self, **cfg_kwargs: Any) -> InpaintRequest:
        return InpaintRequest(
            hd_strategy=HDStrategy.DYNAMIC,
            dynamic_context=dynamic_cfg(**cfg_kwargs),
        )

    def test_window_matches_decision(self) -> None:
        model = RecordingErase()
        image, mask = flat_case()
        decision = decide_context(
            image, mask, (180, 180, 220, 220), dynamic_cfg()
        )
        out = model(image.copy(), mask.copy(), self.request())
        left, top, right, bottom = decision.rect
        assert model.forward_shapes == [((right - left + 7) // 8 * 8, (bottom - top + 7) // 8 * 8)]
        assert out.shape == image.shape

    def test_only_mask_area_is_pasted(self) -> None:
        model = RecordingErase()
        image, mask = flat_case()
        out = model(image.copy(), mask.copy(), self.request())
        # 羽化边允许过渡，mask 中心的像素必须完全是推理结果
        assert out[198:202, 198:202].max() == 200
        assert np.mean(out[mask >= 127] == 200) > 0.6
        # 离 mask 足够远的地方逐像素保持原图
        assert np.all(out[:140, :, :] == 0)

    def test_two_contours_two_windows(self) -> None:
        model = RecordingErase()
        image = np.zeros((400, 400, 3), np.uint8)
        mask = np.zeros((400, 400), np.uint8)
        mask[60:90, 60:90] = 255
        mask[300:330, 300:330] = 255
        out = model(image.copy(), mask.copy(), self.request())
        assert len(model.forward_shapes) == 2
        assert out[70:80, 70:80].max() == 200
        assert out[310:320, 310:320].max() == 200

    def test_max_input_side_downscales_forward(self) -> None:
        model = RecordingErase()
        image, mask = flat_case(height=600, width=600, box=(280, 280, 320, 320))
        out = model(
            image.copy(), mask.copy(), self.request(force_pad=64, max_input_side=128)
        )
        assert max(model.forward_shapes[0]) <= 128
        assert out.shape == image.shape
        assert out[298:302, 298:302].max() == 200

    def test_seam_triggers_one_rerun(self) -> None:
        model = RecordingErase(mode="constant", fill=200)
        image, mask = gradient_case()
        out = model(image.copy(), mask.copy(), self.request(force_pad=64))
        assert len(model.forward_shapes) == 2
        assert out.shape == image.shape

    def test_smooth_result_does_not_rerun(self) -> None:
        model = RecordingErase(mode="copy")
        image, mask = gradient_case()
        model(image.copy(), mask.copy(), self.request(force_pad=64))
        assert len(model.forward_shapes) == 1

    def test_low_gradient_seam_is_not_retried(self) -> None:
        # 平坦背景上比值超过 1.5，但窄带梯度低于 seam_min_grad 时不算可见接缝
        rng = np.random.default_rng(0)
        image = (100 + rng.integers(0, 3, (400, 400, 3))).astype(np.uint8)
        mask = np.zeros((400, 400), np.uint8)
        mask[180:220, 180:220] = 255
        model = RecordingErase(mode="constant", fill=104)
        model(image.copy(), mask.copy(), self.request(force_pad=64))
        assert len(model.forward_shapes) == 1


# ---------------------------------------------------------------------------
# 接缝分数工具
# ---------------------------------------------------------------------------
class TestSeamScore:
    def test_step_is_larger_than_smooth_fill(self) -> None:
        image, mask = textured_case()
        bgr = cv2_bgr(image)
        smooth = bgr.copy()
        smooth[mask >= 127] = 128
        smooth_score, _, _ = seam_score(smooth, mask, dynamic_cfg())
        step = bgr.copy()
        step[mask >= 127] = 255
        step_score, band, _ = seam_score(step, mask, dynamic_cfg())
        assert band > 10
        assert step_score > smooth_score

    def test_empty_side_returns_zero(self) -> None:
        image = np.zeros((50, 50, 3), np.uint8)
        mask = np.full((50, 50), 255, np.uint8)
        assert seam_score(image, mask, dynamic_cfg()) == (0.0, 0.0, 0.0)

    def test_rerun_pad_is_1_5x_clamped(self) -> None:
        cfg = dynamic_cfg(min_pad=64, max_pad=512)
        assert seam_rerun_pad(128, cfg) == 192
        assert seam_rerun_pad(400, cfg) == 512


def gradient_case(
    box: Tuple[int, int, int, int] = (180, 180, 220, 220)
) -> Tuple[np.ndarray, np.ndarray]:
    height = width = 400
    yy, xx = np.mgrid[0:height, 0:width]
    ramp = (60 + xx * 0.2).astype(np.uint8)
    image = np.stack([ramp, ramp, ramp], axis=-1)
    mask = np.zeros((height, width), np.uint8)
    top, left, bottom, right = box
    mask[top:bottom, left:right] = 255
    return image, mask


def cv2_bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, :, ::-1])


# ---------------------------------------------------------------------------
# 服务端配置覆盖请求
# ---------------------------------------------------------------------------
class TestApiConfigOverride:
    def test_config_overrides_request_dynamic_context(self, tmp_path: Path) -> None:
        cfg = _minimal_config(tmp_path)
        cfg.dynamic_context = DynamicContextConfig(base_pad=111)
        api = Api(FastAPI(), cfg)

        req = InpaintRequest(
            hd_strategy=HDStrategy.DYNAMIC,
            dynamic_context=DynamicContextConfig(base_pad=64),
        )
        api._inpaint_blocking(
            req, image_bytes=_png_image(), mask_bytes=_png_mask()
        )

        assert req.dynamic_context.base_pad == 111


def test_decision_is_json_friendly() -> None:
    image, mask = noise_patch()
    decision: ContextDecision = decide_context(
        image, mask, (80, 80, 120, 120), dynamic_cfg()
    )
    data = decision.to_dict()
    assert set(data) >= {"bbox", "rect", "pad", "complexity", "metrics", "flags"}
    assert isinstance(data["bbox"], list)


def test_load_dynamic_context_from_json(tmp_path: Path) -> None:
    from iopaint.cli import _load_dynamic_context

    path = tmp_path / "dynamic.json"
    path.write_text('{"base_pad": 96, "enable_complexity": false}')
    cfg = _load_dynamic_context(path)
    assert cfg.base_pad == 96
    assert cfg.enable_complexity is False
    assert cfg.max_pad == DynamicContextConfig().max_pad


def test_load_dynamic_context_rejects_bad_json(tmp_path: Path) -> None:
    from iopaint.cli import _load_dynamic_context

    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(SystemExit):
        _load_dynamic_context(path)
