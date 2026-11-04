"""erase 档的定尺寸瓦片路径与上卡/量化助手。

migan/fcf 原先各自复制了一份逐行相同的 `__call__`，现在统一走
`InpaintModel._fixed_size_call`；这里把原实现留作参考实现，逐场景比对像素。

助手测试守的是「设备端量化与 CPU 端 norm_img/clip 逐位相同」这条前提——每请求
传输优化（uint8 上卡、设备端量化）的全部收益都建立在它上面。

不加载任何权重：假模型的 forward 只按输入返回可预测的像素。
"""
import cv2
import numpy as np
import pytest
import torch

from iopaint.helper import boxes_from_mask, norm_img, resize_max_size
from iopaint.model.base import InpaintModel
from iopaint.model.helper.tensors import (
    binary_mask,
    to_bgr_uint8,
    upload_uint8,
)
from iopaint.schema import HDStrategy, InpaintRequest


class StubTile(InpaintModel):
    """固定 512 瓦片的假模型：输出依赖输入的空间结构，crop 框一变结果就变。"""

    name = "stub_tile"
    min_size = 512
    pad_mod = 512
    pad_to_square = True
    fixed_size = 512

    def __init__(self, device: str = "cpu", **kwargs: object) -> None:
        self.forward_shapes: list[tuple[int, ...]] = []
        super().__init__(device, **kwargs)

    @staticmethod
    def is_downloaded() -> bool:
        return True

    def init_model(self, device: str, **kwargs: object) -> None:
        pass

    def forward(
        self, image: np.ndarray, mask: np.ndarray, config: InpaintRequest
    ) -> np.ndarray:
        self.forward_shapes.append(tuple(image.shape[:2]))
        tinted = (image.astype(np.uint16) * 127 // 255 + 96).astype(np.uint8)
        striped = tinted.copy()
        striped[::3] = 255 - striped[::3]
        return striped[:, :, ::-1].copy()


def _reference_fixed_size_call(
    model: InpaintModel, image: np.ndarray, mask: np.ndarray, config: InpaintRequest
) -> np.ndarray:
    """migan 原 `__call__` 的实现，作为逐像素比对的参考。"""
    if image.shape[0] == 512 and image.shape[1] == 512:
        return model._pad_forward(image, mask, config)

    boxes = boxes_from_mask(mask)
    crop_result = []
    config.hd_strategy_crop_margin = 128
    for box in boxes:
        crop_image, crop_mask, crop_box = model._crop_box(image, mask, box, config)
        origin_size = crop_image.shape[:2]
        resize_image = resize_max_size(crop_image, size_limit=512)
        resize_mask = resize_max_size(crop_mask, size_limit=512)
        inpaint_result = model._pad_forward(resize_image, resize_mask, config)

        inpaint_result = cv2.resize(
            inpaint_result,
            (origin_size[1], origin_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        original_pixel_indices = crop_mask < 127
        inpaint_result[original_pixel_indices] = crop_image[:, :, ::-1][
            original_pixel_indices
        ]

        crop_result.append((inpaint_result, crop_box))

    inpaint_result = image[:, :, ::-1].copy()
    for crop_image, crop_box in crop_result:
        x1, y1, x2, y2 = crop_box
        inpaint_result[y1:y2, x1:x2, :] = crop_image

    return inpaint_result


def make_case(
    height: int, width: int, boxes: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(height * width)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), np.uint8)
    for top, left, bottom, right in boxes:
        mask[top:bottom, left:right] = 255
    return image, mask


def cfg(
    strategy: HDStrategy = HDStrategy.ORIGINAL, margin: int = 128
) -> InpaintRequest:
    return InpaintRequest(
        hd_strategy=strategy,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_crop_margin=margin,
        hd_strategy_resize_limit=1280,
    )


CASES = {
    "square_512": (512, 512, [(200, 200, 300, 300)]),
    "single_box": (600, 800, [(100, 120, 260, 400)]),
    "two_boxes": (1200, 1600, [(100, 100, 300, 300), (700, 900, 1000, 1300)]),
    "edge_box": (4000, 3000, [(10, 10, 700, 900)]),
    "tall_box": (3000, 4000, [(200, 3000, 2400, 3600)]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_fixed_size_call_matches_reference(name: str) -> None:
    height, width, boxes = CASES[name]
    image, mask = make_case(height, width, boxes)
    reference_model = StubTile()

    expected = _reference_fixed_size_call(
        reference_model, image.copy(), mask.copy(), cfg()
    )
    actual = StubTile()(image.copy(), mask.copy(), cfg())

    assert np.array_equal(actual, expected), name


@pytest.mark.parametrize("strategy", [HDStrategy.ORIGINAL, HDStrategy.RESIZE, HDStrategy.CROP])
@pytest.mark.parametrize("margin", [32, 128, 321])
def test_fixed_size_call_ignores_hd_strategy(
    strategy: HDStrategy, margin: int
) -> None:
    image, mask = make_case(1200, 1600, [(100, 100, 300, 300)])
    expected = _reference_fixed_size_call(
        StubTile(), image.copy(), mask.copy(), cfg(HDStrategy.ORIGINAL, 128)
    )
    actual = StubTile()(image.copy(), mask.copy(), cfg(strategy, margin))
    assert np.array_equal(actual, expected)


def test_fixed_size_forwards_are_all_square() -> None:
    image, mask = make_case(1200, 1600, [(100, 100, 300, 300), (700, 900, 1000, 1300)])
    model = StubTile()
    model(image, mask, cfg())
    assert model.forward_shapes
    assert all(shape == (512, 512) for shape in model.forward_shapes)


def test_fixed_size_keeps_unmasked_pixels() -> None:
    image, mask = make_case(1200, 1600, [(100, 100, 300, 300), (700, 900, 1000, 1300)])
    out = StubTile()(image.copy(), mask.copy(), cfg())
    keep = mask < 127
    assert np.array_equal(out[keep], image[:, :, ::-1][keep])


def test_migan_and_fcf_use_base_call() -> None:
    from iopaint.model.fcf import FcF
    from iopaint.model.mi_gan import MIGAN

    for cls in (MIGAN, FcF):
        assert cls.fixed_size == 512
        assert cls.__call__ is InpaintModel.__call__
        assert cls._fixed_size_call is InpaintModel._fixed_size_call


def test_upload_uint8_matches_norm_img() -> None:
    image = np.random.default_rng(0).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    expected = torch.from_numpy(norm_img(image)).unsqueeze(0)
    assert torch.equal(upload_uint8(image, "cpu"), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 的标量除法是近似的")
def test_upload_uint8_matches_norm_img_on_cuda() -> None:
    # CUDA 上 `tensor / 标量` 走近似除法（126/256 个取值差 1 ULP），除数必须是张量；
    # CPU 上两种写法结果相同，因此这一条只能在有 CUDA 的机器上拦住回归。
    image = np.arange(256, dtype=np.uint8)[None, :, None].repeat(3, axis=2)
    expected = torch.from_numpy(norm_img(image)).unsqueeze(0).cuda()
    assert torch.equal(upload_uint8(image, "cuda"), expected)


@pytest.mark.parametrize("threshold", [0, 120, 127])
def test_binary_mask_matches_norm_img_threshold(threshold: int) -> None:
    mask = np.random.default_rng(1).integers(0, 256, (37, 53), dtype=np.uint8)
    expected = (
        torch.from_numpy((norm_img(mask) > threshold / 255) * 1).unsqueeze(0).float()
    )
    actual = binary_mask(mask, "cpu", threshold=threshold)
    assert torch.equal(actual, expected)
    # [H, W, 1] 形态（api 实际下发的形状）必须给出同样的结果
    assert torch.equal(binary_mask(mask[:, :, None], "cpu", threshold=threshold), actual)


def test_to_bgr_uint8_truncates_like_numpy() -> None:
    output = torch.rand(1, 3, 17, 19)
    expected = np.clip(output[0].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
    assert np.array_equal(to_bgr_uint8(output, scale=255), expected[:, :, ::-1])


def test_to_bgr_uint8_rounds_like_gan_path() -> None:
    output = torch.rand(1, 3, 17, 19) * 2 - 1
    expected = ((output[0].permute(1, 2, 0).numpy() * 127.5 + 127.5).round()).astype(
        np.uint8
    )
    assert np.array_equal(
        to_bgr_uint8(output, scale=127.5, offset=127.5, rounding=True),
        expected[:, :, ::-1],
    )


def test_to_bgr_uint8_gray_output() -> None:
    output = torch.rand(1, 1, 17, 19)
    expected = np.clip(output[0].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
    assert np.array_equal(
        to_bgr_uint8(output, scale=255), cv2.cvtColor(expected, cv2.COLOR_GRAY2BGR)
    )
