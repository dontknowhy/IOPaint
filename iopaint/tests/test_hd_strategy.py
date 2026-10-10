"""hd_strategy 的 crop/resize 应该同时作用于 erase 与 diffusion 模型。

历史上 `DiffusionInpaintModel.__call__` 覆盖了基类，不读 `hd_strategy`，只认
`sd_scale`（而前端根本没下发），于是 SD 系永远整图全分辨率入网：大图在 8G 卡上
要么 OOM、要么一步一两分钟。现在 diffusion 走和 erase 相同的 crop/resize，本文件
守这条线，并锁死 `ApiConfig.hd_strategy_crop_margin` 覆盖请求值的约定。

不加载任何权重：假模型的 forward 只记录拿到的张量尺寸，返回纯色。
"""
import numpy as np
from fastapi import FastAPI

from iopaint.api import Api
from iopaint.model.base import DiffusionInpaintModel, InpaintModel
from iopaint.schema import ApiConfig, HDStrategy, InpaintRequest, ModelInfo, ModelType
from iopaint.tests.test_api import _b64_image, _b64_mask, _minimal_config


class _RecordingMixin:
    def _record(self, image):
        self.forward_shapes.append(tuple(image.shape[:2]))
        return np.full_like(image, 200)


class RecordingSD(_RecordingMixin, DiffusionInpaintModel):
    name = "recording_sd"
    pad_mod = 8
    min_size = 512

    def __init__(self, device=None, **kwargs):
        self.forward_shapes = []
        super().__init__(device or "cpu", **kwargs)

    @staticmethod
    def is_downloaded() -> bool:
        return True

    def init_model(self, device, **kwargs):
        pass

    def forward(self, image, mask, config):
        return self._record(image)


class RecordingErase(_RecordingMixin, InpaintModel):
    name = "recording_erase"
    pad_mod = 8

    def __init__(self, device=None, **kwargs):
        self.forward_shapes = []
        super().__init__(device or "cpu", **kwargs)

    @staticmethod
    def is_downloaded() -> bool:
        return True

    def init_model(self, device, **kwargs):
        pass

    def forward(self, image, mask, config):
        return self._record(image)


def make_sd() -> RecordingSD:
    return RecordingSD(
        model_info=ModelInfo(
            name="recording_sd", path="/fake", model_type=ModelType.DIFFUSERS_SD_INPAINT
        )
    )


def make_erase() -> RecordingErase:
    return RecordingErase()


def make_case(height=1200, width=1600, box=(300, 400, 800, 900)):
    """默认：1600x1200 图 + 500x500 的方形掩膜（bbox 可预测）。"""
    image = np.zeros((height, width, 3), np.uint8)
    mask = np.zeros((height, width), np.uint8)
    top, left, bottom, right = box
    mask[top:bottom, left:right] = 255
    return image, mask


def crop_cfg(**kwargs) -> InpaintRequest:
    return InpaintRequest(
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_trigger_size=640,
        hd_strategy_crop_margin=128,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 分派：erase 用 _pad_forward，diffusion 用 _scaled_pad_forward
# ---------------------------------------------------------------------------
class TestForwardRegionDispatch:
    def test_erase_uses_pad_forward(self):
        model = make_erase()
        assert type(model)._forward_region is InpaintModel._forward_region

    def test_diffusion_uses_scaled_pad_forward(self):
        model = make_sd()
        assert type(model)._forward_region is DiffusionInpaintModel._forward_region


# ---------------------------------------------------------------------------
# diffusion 现在也走 hd_strategy
# ---------------------------------------------------------------------------
class TestDiffusionHdStrategy:
    def test_crop_runs_on_mask_region(self):
        # 500x500 掩膜 + 128px 边距 = 756 -> 取模 8 = 760，而不是整图 1600x1200
        model = make_sd()
        image, mask = make_case()
        out = model(image.copy(), mask.copy(), crop_cfg())

        assert model.forward_shapes == [(760, 760)]
        assert out.shape == image.shape

    def test_crop_small_mask_padded_to_min_size(self):
        # 40x40 掩膜 + 两侧 128px 边距 = 296，被 SD 的 min_size=512 顶上去
        model = make_sd()
        image, mask = make_case(box=(600, 800, 640, 840))
        out = model(image.copy(), mask.copy(), crop_cfg())

        assert model.forward_shapes == [(512, 512)]
        assert out.shape == image.shape

    def test_resize_downscales_longer_side(self):
        model = make_sd()
        image, mask = make_case(height=2000, width=3000, box=(900, 1400, 1100, 1600))
        out = model(
            image.copy(),
            mask.copy(),
            InpaintRequest(
                hd_strategy=HDStrategy.RESIZE, hd_strategy_resize_limit=1024
            ),
        )

        assert max(model.forward_shapes[0]) == 1024
        assert model.forward_shapes[0] != image.shape[:2]
        assert out.shape == image.shape

    def test_small_image_stays_full_size(self):
        # 长边不超过 trigger 时不裁剪，行为与改动前一致
        model = make_sd()
        image, mask = make_case(height=600, width=600, box=(200, 200, 400, 400))
        out = model(image.copy(), mask.copy(), crop_cfg())

        assert model.forward_shapes == [(600, 600)]
        assert out.shape == image.shape


class TestEraseStillWorks:
    def test_crop_unchanged(self):
        model = make_erase()
        image, mask = make_case()
        out = model(image.copy(), mask.copy(), crop_cfg())

        assert model.forward_shapes == [(760, 760)]
        assert out.shape == image.shape


# ---------------------------------------------------------------------------
# config.json 的 hd_strategy_crop_margin 覆盖请求值
# ---------------------------------------------------------------------------
class TestCropMarginConfig:
    def test_default_is_128(self):
        assert ApiConfig.model_fields["hd_strategy_crop_margin"].default == 128

    def test_config_overrides_request_value(self, tmp_path):
        cfg = _minimal_config(tmp_path)  # model=cv2：免权重
        cfg.hd_strategy_crop_margin = 321
        api = Api(FastAPI(), cfg)

        req = InpaintRequest(
            image=_b64_image(), mask=_b64_mask(), hd_strategy_crop_margin=128
        )
        api._inpaint_blocking(req)

        assert req.hd_strategy_crop_margin == 321
