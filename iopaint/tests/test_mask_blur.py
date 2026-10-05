"""D-17：sd_mask_blur 参数已删除，但合成羽化必须固定保留。

decision 里原本写的是"删除不改变模型输出"，实测是错的：被模糊的 mask 会一路传到
base.py 的 sd_keep_unmasked_area 合成（该字段默认 True），直接删掉会让 SD inpaint
的边缘从羽化变成硬边（默认配置实测 4660px 差异 / 单通道最大 140）。所以最终做法是
**删参数、留固定羽化**（DiffusionInpaintModel.COMPOSITE_MASK_BLUR），本文件守这条线。

不加载任何权重：FakeSD 的 forward 返回纯色，只考察前/后处理与合成。
"""
import cv2
import numpy as np
import torch

from iopaint.model.base import DiffusionInpaintModel
from iopaint.schema import HDStrategy, InpaintRequest, ModelInfo, ModelType


class FakeSD(DiffusionInpaintModel):
    name = "fake_sd"
    pad_mod = 8

    @staticmethod
    def is_downloaded():
        return True

    def init_model(self, device, **kwargs):
        pass

    def forward(self, image, mask, config):
        # 模型"填"出来的内容：整块纯色，颜色本身不重要
        return np.full_like(image, 200)


def make_model() -> FakeSD:
    return FakeSD(
        device=torch.device("cpu"),
        model_info=ModelInfo(
            name="fake_sd", path="/fake", model_type=ModelType.DIFFUSERS_SD
        ),
    )


def make_cfg(**kwargs) -> InpaintRequest:
    defaults = dict(hd_strategy=HDStrategy.ORIGINAL, sd_keep_unmasked_area=True)
    defaults.update(kwargs)
    return InpaintRequest(**defaults)


def make_case():
    image = np.zeros((128, 128, 3), np.uint8)
    image[..., 0] = np.linspace(0, 255, 128, dtype=np.uint8)
    mask = np.zeros((128, 128), np.uint8)
    mask[32:96, 32:96] = 255
    return image, mask


# 2 * 11 + 1：删除前 schema 里 sd_mask_blur 的默认值 11。写死字面量而不是读
# COMPOSITE_MASK_BLUR，这样"羽化被去掉"这种回归会以断言失败而不是属性错误暴露
KERNEL = 23


class TestSchemaFieldRemoved:
    def test_field_gone_from_schema(self):
        assert "sd_mask_blur" not in InpaintRequest.model_fields

    def test_old_client_payload_still_accepted(self):
        # D-17 承诺的兼容性：改之前前端提交的就是 sd_mask_blur，
        # pydantic v2 默认忽略未知字段，不能因此 422
        cfg = InpaintRequest(sd_mask_blur=11)
        assert not hasattr(cfg, "sd_mask_blur")

    def test_constant_keeps_old_default(self):
        # 固定羽化用的就是删除前的 schema 默认值 11
        assert getattr(DiffusionInpaintModel, "COMPOSITE_MASK_BLUR", None) == 11


class TestFeatherKept:
    def test_post_process_feathers_mask(self):
        model, cfg = make_model(), make_cfg()
        image, mask = make_case()
        result = np.full_like(image, 200)

        _, _, out_mask = model.forward_post_process(result, image, mask, cfg)

        assert np.array_equal(out_mask, cv2.GaussianBlur(mask, (KERNEL, KERNEL), 0))
        # 羽化确实生效：mask 边界上有 0/255 之外的中间值
        assert set(np.unique(out_mask).tolist()) - {0, 255}

    def test_extender_does_not_blur_twice(self):
        # D-17 删掉了 use_extender 时的第二次模糊，outpainting 与普通路径口径一致
        model = make_model()
        image, mask = make_case()
        result = np.full_like(image, 200)

        # 返回值是 (result, image, mask)，取 mask 要用 [2]
        plain = model.forward_post_process(
            result, image, mask, make_cfg(use_extender=False)
        )[2]
        extender = model.forward_post_process(
            result, image, mask, make_cfg(use_extender=True)
        )[2]

        assert np.array_equal(plain, extender)

    def test_blend_uses_feathered_mask(self):
        model, cfg = make_model(), make_cfg()
        image, mask = make_case()

        out = model(image.copy(), mask.copy(), cfg)

        # 参考实现：羽化 mask + 纯色结果合成（float32，见 D-18）
        feathered = cv2.GaussianBlur(mask, (KERNEL, KERNEL), 0)
        alpha = feathered[:, :, np.newaxis].astype(np.float32) / 255.0
        expected = 200 * alpha + image[:, :, ::-1] * (1.0 - alpha)
        got = np.clip(out, 0, 255)
        assert np.allclose(got, np.clip(expected, 0, 255), atol=1)

    def test_hard_edge_would_differ(self):
        # 反证：如果不羽化（decision 字面的"直接删"），输出会明显不同
        model, cfg = make_model(), make_cfg()
        image, mask = make_case()

        out = np.clip(model(image.copy(), mask.copy(), cfg), 0, 255).astype(np.uint8)

        hard_alpha = mask[:, :, np.newaxis] / 255.0
        hard = (
            200 * hard_alpha + image[:, :, ::-1] * (1 - hard_alpha)
        ).astype(np.uint8)
        assert not np.array_equal(out, hard)
        # 差异只出现在羽化过的边界带上，且幅度不小（当初实测最大 140/255）
        diff = np.abs(out.astype(np.int16) - hard.astype(np.int16))
        assert diff.max() >= 50
