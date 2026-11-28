import abc
import json
from typing import Dict, List, Optional, Tuple

import cv2
import torch
import numpy as np
from loguru import logger

from iopaint.helper import (
    boxes_from_mask,
    crop_rect,
    merge_overlapping_rects,
    resize_max_size,
    pad_img_to_modulo,
    switch_mps_device,
)
from iopaint.schema import DynamicContextConfig, InpaintRequest, HDStrategy, SDSampler
from .helper.dynamic_context import (
    ContextDecision,
    decide_context,
    seam_rerun_pad,
    seam_score,
)
from .helper.g_diffuser_bot import expand_image
from .utils import get_scheduler


class InpaintModel:
    name = "base"
    min_size: Optional[int] = None
    pad_mod = 8
    pad_to_square = False
    # 只能吃正方形定尺寸瓦片的模型（migan/fcf）：非空时 __call__ 走 _fixed_size_call
    fixed_size: Optional[int] = None
    is_erase_model = False
    # 只有纯 SD/SDXL 的 UNet 支持运行时 torch.compile（见 model/torch_compile.py）
    supports_torch_compile = False

    def __init__(self, device, **kwargs):
        """

        Args:
            device:
        """
        device = switch_mps_device(self.name, device)
        self.device = device
        self.init_model(device, **kwargs)

    @abc.abstractmethod
    def init_model(self, device, **kwargs): ...

    @staticmethod
    @abc.abstractmethod
    def is_downloaded() -> bool:
        return False

    @abc.abstractmethod
    def forward(self, image, mask, config: InpaintRequest):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W] 255 为 masks 区域
        return: BGR IMAGE
        """
        ...

    @staticmethod
    def download(): ...

    def _pad_forward(self, image, mask, config: InpaintRequest):
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(
            image, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )
        pad_mask = pad_img_to_modulo(
            mask, mod=self.pad_mod, square=self.pad_to_square, min_size=self.min_size
        )

        # logger.info(f"final forward pad size: {pad_image.shape}")

        result = self.forward(pad_image, pad_mask, config)
        result = result[0:origin_height, 0:origin_width, :]

        result, image, mask = self.forward_post_process(result, image, mask, config)

        if config.sd_keep_unmasked_area:
            # D-18/BUG-5：原写法 `mask / 255` 由 uint8 升成 float64，12MP 图
            # 这一步的分配峰值实测 641MB；显式 float32 后降到 366MB。
            # 二值 0/255 mask 下与 float64 逐像素相同；羽化后的部分 mask 会
            # 有约 0.5% 的元素在 uint8 截断边界上差 1/255（肉眼不可见）。
            alpha = mask[:, :, np.newaxis].astype(np.float32) / 255.0
            result = result * alpha + image[:, :, ::-1] * (1.0 - alpha)
        return result

    def forward_post_process(self, result, image, mask, config):
        return result, image, mask

    def _forward_region(self, image, mask, config: InpaintRequest):
        """对一块区域做推理。

        erase 模型直接 padding；diffusion 模型覆盖本方法，先按 sd_scale 缩放，
        这样 hd_strategy 的 crop/resize 对两类模型都成立。
        """
        return self._pad_forward(image, mask, config)

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        if self.fixed_size is not None:
            return self._fixed_size_call(image, mask, config)

        inpaint_result = None
        # logger.info(f"hd_strategy: {config.hd_strategy}")
        if config.hd_strategy == HDStrategy.DYNAMIC:
            return self._dynamic_context_call(image, mask, config)
        if config.hd_strategy == HDStrategy.CROP:
            if max(image.shape) > config.hd_strategy_crop_trigger_size:
                logger.info("Run crop strategy")
                crop_result = []
                for rect in self._crop_rects(mask, config):
                    crop_image, crop_mask = self._crop_region(image, mask, rect)
                    crop_result.append(
                        (self._forward_region(crop_image, crop_mask, config), rect)
                    )

                # image[:, :, ::-1] 是负步长视图，直接赋值会写穿调用方的 image
                inpaint_result = image[:, :, ::-1].copy()
                for crop_image, crop_box in crop_result:
                    x1, y1, x2, y2 = crop_box
                    inpaint_result[y1:y2, x1:x2, :] = crop_image

        elif config.hd_strategy == HDStrategy.RESIZE:
            if max(image.shape) > config.hd_strategy_resize_limit:
                origin_size = image.shape[:2]
                downsize_image = resize_max_size(
                    image, size_limit=config.hd_strategy_resize_limit
                )
                downsize_mask = resize_max_size(
                    mask, size_limit=config.hd_strategy_resize_limit
                )

                logger.info(
                    f"Run resize strategy, origin size: {image.shape} forward size: {downsize_image.shape}"
                )
                inpaint_result = self._forward_region(
                    downsize_image, downsize_mask, config
                )

                # only paste masked area result
                inpaint_result = cv2.resize(
                    inpaint_result,
                    (origin_size[1], origin_size[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
                np.copyto(
                    inpaint_result,
                    image[:, :, ::-1],
                    where=(mask < 127)[..., np.newaxis],
                )

        if inpaint_result is None:
            inpaint_result = self._forward_region(image, mask, config)

        return inpaint_result

    def _dynamic_context_call(
        self, image: np.ndarray, mask: np.ndarray, config: InpaintRequest
    ) -> np.ndarray:
        """`hd_strategy=Dynamic`：每个掩膜轮廓按动态 padding 单独裁剪推理。

        与 CROP 的区别：边距来自 `dynamic_context` 的几何规则 + ring 复杂度（不再
        固定 128），窗口超过 `max_input_side` 时先缩小推理再放大回贴，回贴只覆盖
        mask 及羽化边（区域外逐像素等于原图），可选接缝兜底重跑一次。
        """
        cfg = config.dynamic_context
        inpaint_result = image[:, :, ::-1].copy()
        for box in boxes_from_mask(mask):
            decision = decide_context(image, mask, box, cfg)
            result, rect, seam_info = self._forward_dynamic_crop(
                image, mask, decision, cfg, config
            )
            self._paste_mask_region(inpaint_result, result, mask, rect, cfg)
            record = decision.to_dict()
            record.update(seam_info)
            logger.info(
                "dynamic_context " + json.dumps(record, ensure_ascii=False)
            )
        return inpaint_result

    def _forward_dynamic_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        decision: ContextDecision,
        cfg: DynamicContextConfig,
        config: InpaintRequest,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], Dict[str, object]]:
        """跑一个动态裁块；接缝超阈值时按 1.5 倍 padding 重跑一次。"""
        crop_image, crop_mask = self._crop_region(image, mask, decision.rect)
        result = self._dynamic_region_forward(crop_image, crop_mask, cfg, config)
        rect = decision.rect
        seam_info: Dict[str, object] = {}
        if not cfg.enable_seam_fallback:
            return result, rect, seam_info

        score, band_grad, ring_grad = seam_score(result, crop_mask, cfg)
        seam_info.update(
            {
                "seam_score": round(score, 4),
                "seam_band_grad": round(band_grad, 2),
                "seam_ring_grad": round(ring_grad, 2),
                "seam_rerun": False,
                "seam_failed": False,
            }
        )
        if score <= cfg.seam_ratio or band_grad < cfg.seam_min_grad:
            # 平坦区域里比值会被极小的 ring 梯度放大，窄带梯度本身很低时不算接缝
            return result, rect, seam_info

        new_pad = seam_rerun_pad(decision.pad, cfg)
        if new_pad <= decision.pad:
            # 已经顶到 max_pad，重跑也换不来更多上下文
            seam_info["seam_failed"] = True
            return result, rect, seam_info

        rect = crop_rect(np.asarray(decision.bbox), new_pad, mask.shape)
        crop_image, crop_mask = self._crop_region(image, mask, rect)
        result = self._dynamic_region_forward(crop_image, crop_mask, cfg, config)
        score2, band_grad2, ring_grad2 = seam_score(result, crop_mask, cfg)
        seam_info.update(
            {
                "seam_rerun": True,
                "pad_after_rerun": new_pad,
                "seam_score_after_rerun": round(score2, 4),
                "seam_band_grad_after_rerun": round(band_grad2, 2),
                "seam_ring_grad_after_rerun": round(ring_grad2, 2),
                "seam_failed": score2 > cfg.seam_ratio
                and band_grad2 >= cfg.seam_min_grad,
            }
        )
        return result, rect, seam_info

    def _dynamic_region_forward(
        self,
        crop_image: np.ndarray,
        crop_mask: np.ndarray,
        cfg: DynamicContextConfig,
        config: InpaintRequest,
    ) -> np.ndarray:
        """窗口超过 max_input_side 时缩到上限推理，再把结果放大回窗口尺寸。"""
        if max(crop_image.shape[:2]) <= cfg.max_input_side:
            return self._forward_region(crop_image, crop_mask, config)
        origin_size = crop_image.shape[:2]
        resize_image = resize_max_size(crop_image, size_limit=cfg.max_input_side)
        resize_mask = resize_max_size(crop_mask, size_limit=cfg.max_input_side)
        result = self._forward_region(resize_image, resize_mask, config)
        return cv2.resize(
            result,
            (origin_size[1], origin_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def _paste_mask_region(
        inpaint_result: np.ndarray,
        result: np.ndarray,
        mask: np.ndarray,
        rect: Tuple[int, int, int, int],
        cfg: DynamicContextConfig,
    ) -> None:
        """只把 mask（含 feather 羽化边）贴回去，窗口其余像素保持原图。

        result 是裁块坐标下的 BGR；羽化只改 mask 外 paste_feather 像素内的一圈，
        再远一点逐像素沿用原图。
        """
        l, t, r, b = rect
        target = inpaint_result[t:b, l:r]
        source = inpaint_result[t:b, l:r].copy()
        crop_mask = mask[t:b, l:r]
        binary = (crop_mask >= 127).astype(np.float32)

        feather = int(cfg.paste_feather)
        if feather > 0:
            k = 2 * feather + 1
            alpha = cv2.GaussianBlur(binary, (k, k), 0)
            # 羽化只向 mask 外淡出：mask 内必须保持 1，否则边界一圈会把原图
            # （待删除的水印/文字）按 50% 混回来，看起来像半透明残影。
            alpha = np.maximum(alpha, binary)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            region = cv2.dilate(binary.astype(np.uint8), kernel) > 0
        else:
            alpha = binary
            region = binary > 0

        blended = (
            result.astype(np.float32) * alpha[:, :, np.newaxis]
            + source.astype(np.float32) * (1.0 - alpha[:, :, np.newaxis])
        )
        target[region] = blended[region].astype(np.uint8)

    def _fixed_size_call(
        self, image: np.ndarray, mask: np.ndarray, config: InpaintRequest
    ) -> np.ndarray:
        """定尺寸模型（fixed_size 非空）的瓦片推理。

        按 mask 找出瓦片，缩到 fixed_size 以内做前向，再把结果放回原尺寸，掩膜外的
        像素一律回贴原图。这类模型不读 hd_strategy：输入必须是正方形且边长固定。
        """
        size = self.fixed_size
        assert size is not None  # __call__ 只在 fixed_size 非空时分派到这里
        if image.shape[0] == size and image.shape[1] == size:
            return self._pad_forward(image, mask, config)

        crop_result = []
        # 上下文边距固定 128，不随 config.hd_strategy_crop_margin 变化：瓦片要缩放到
        # 定尺寸，边距一旦变化连像素结果都会变。合并间距仍按 config 走。
        config.hd_strategy_crop_margin = 128
        for rect in self._crop_rects(mask, config):
            crop_image, crop_mask = self._crop_region(image, mask, rect)
            crop_box = list(rect)
            origin_size = crop_image.shape[:2]
            resize_image = resize_max_size(crop_image, size_limit=size)
            resize_mask = resize_max_size(crop_mask, size_limit=size)
            inpaint_result = self._pad_forward(resize_image, resize_mask, config)

            # only paste masked area result
            inpaint_result = cv2.resize(
                inpaint_result,
                (origin_size[1], origin_size[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            np.copyto(
                inpaint_result,
                crop_image[:, :, ::-1],
                where=(crop_mask < 127)[..., np.newaxis],
            )

            crop_result.append((inpaint_result, crop_box))

        inpaint_result = image[:, :, ::-1].copy()
        for crop_image, crop_box in crop_result:
            x1, y1, x2, y2 = crop_box
            inpaint_result[y1:y2, x1:x2, :] = crop_image

        return inpaint_result

    def _crop_rects(
        self, mask: np.ndarray, config: InpaintRequest
    ) -> List[Tuple[int, int, int, int]]:
        """CROP 策略要跑的裁剪矩形：每个掩膜轮廓一个，可按距离把相邻的并成一个。"""
        rects = [
            crop_rect(box, config.hd_strategy_crop_margin, mask.shape)
            for box in boxes_from_mask(mask)
        ]
        if config.hd_strategy_crop_merge and len(rects) > 1:
            rects = merge_overlapping_rects(
                rects, config.hd_strategy_crop_merge_max_pixels
            )
        return rects

    @staticmethod
    def _crop_region(
        image: np.ndarray, mask: np.ndarray, rect: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        l, t, r, b = rect
        return image[t:b, l:r, :], mask[t:b, l:r]

    def _crop_box(self, image, mask, box, config: InpaintRequest):
        """

        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1]
            box: [left,top,right,bottom]

        Returns:
            BGR IMAGE, (l, r, r, b)
        """
        rect = crop_rect(box, config.hd_strategy_crop_margin, image.shape)
        crop_img, crop_mask = self._crop_region(image, mask, rect)
        return crop_img, crop_mask, list(rect)

    def _calculate_cdf(self, histogram):
        cdf = histogram.cumsum()
        total = cdf[-1]
        if total == 0:
            # 选区为空（mask==0 区域没有像素）时 cdf[-1]==0，除零会得到 NaN，
            # 进而 LUT 出全黑图。返回全 0 CDF，等价于“无变换”。
            return np.zeros_like(cdf, dtype=np.float64)
        normalized_cdf = cdf / float(total)
        return normalized_cdf

    def _calculate_lookup(self, source_cdf, reference_cdf):
        # For each source index, find the first reference index whose CDF is
        # greater than or equal to the source CDF value.
        return np.clip(np.searchsorted(reference_cdf, source_cdf), 0, 255)

    def _match_histograms(self, source, reference, mask):
        transformed_channels = []
        if len(mask.shape) == 3:
            mask = mask[:, :, -1]

        # mask==0（非遮罩区）为空时没有可统计的像素：直方图匹配无从谈起，
        # 直接返回原图，否则会得到全黑结果。
        if not np.any(mask == 0):
            return source

        for channel in range(source.shape[-1]):
            source_channel = source[:, :, channel]
            reference_channel = reference[:, :, channel]

            # only calculate histograms for non-masked parts
            source_histogram, _ = np.histogram(source_channel[mask == 0], 256, [0, 256])
            reference_histogram, _ = np.histogram(
                reference_channel[mask == 0], 256, [0, 256]
            )

            source_cdf = self._calculate_cdf(source_histogram)
            reference_cdf = self._calculate_cdf(reference_histogram)

            lookup = self._calculate_lookup(source_cdf, reference_cdf)

            transformed_channels.append(cv2.LUT(source_channel, lookup))

        result = cv2.merge(transformed_channels)
        result = cv2.convertScaleAbs(result)

        return result

    def _apply_cropper(self, image, mask, config: InpaintRequest):
        img_h, img_w = image.shape[:2]
        l, t, w, h = (
            config.croper_x,
            config.croper_y,
            config.croper_width,
            config.croper_height,
        )
        r = l + w
        b = t + h

        l = max(l, 0)
        r = min(r, img_w)
        t = max(t, 0)
        b = min(b, img_h)

        crop_img = image[t:b, l:r, :]
        crop_mask = mask[t:b, l:r]
        return crop_img, crop_mask, (l, t, r, b)

class DiffusionInpaintModel(InpaintModel):
    def __init__(self, device, **kwargs):
        self.model_info = kwargs["model_info"]
        self.model_id_or_path = self.model_info.path
        self._scheduler_cache = {}
        self._compiled_unet = None
        super().__init__(device, **kwargs)
        if kwargs.get("torch_compile", False):
            from .torch_compile import maybe_compile_unet

            self._compiled_unet = maybe_compile_unet(self, self.device, kwargs)

    def set_lora_enabled(self, enabled: bool) -> None:
        """Keep the UNet eager once LCM LoRA mutates it."""
        if self._compiled_unet is not None:
            self._compiled_unet.set_lora_enabled(enabled)

    @torch.no_grad()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: BGR IMAGE
        """
        # boxes = boxes_from_mask(mask)
        if config.use_croper:
            crop_img, crop_mask, (l, t, r, b) = self._apply_cropper(image, mask, config)
            crop_image = self._scaled_pad_forward(crop_img, crop_mask, config)
            # 同 __call__：负步长视图，赋值会写穿 image
            inpaint_result = image[:, :, ::-1].copy()
            inpaint_result[t:b, l:r, :] = crop_image
        elif config.use_extender:
            inpaint_result = self._do_outpainting(image, config)
        else:
            # diffusion 模型此前直接整图 _scaled_pad_forward；现在和 erase 模型一样走
            # hd_strategy 的 crop/resize，避免大图全分辨率入网（8G 卡上会 OOM）
            return super().__call__(image, mask, config)

        return inpaint_result

    def _forward_region(self, image, mask, config: InpaintRequest):
        # hd_strategy 的 crop/resize 只负责圈定区域；这里再按 sd_scale 缩放
        return self._scaled_pad_forward(image, mask, config)

    def _do_outpainting(self, image, config: InpaintRequest):
        # cropper 和 image 在同一个坐标系下，croper_x/y 可能为负数
        # 从 image 中 crop 出 outpainting 区域
        image_h, image_w = image.shape[:2]
        cropper_l = config.extender_x
        cropper_t = config.extender_y
        cropper_r = config.extender_x + config.extender_width
        cropper_b = config.extender_y + config.extender_height
        image_l = 0
        image_t = 0
        image_r = image_w
        image_b = image_h

        # 类似求 IOU
        l = max(cropper_l, image_l)
        t = max(cropper_t, image_t)
        r = min(cropper_r, image_r)
        b = min(cropper_b, image_b)

        assert (
            0 <= l < r and 0 <= t < b
        ), f"cropper and image not overlap, {l},{t},{r},{b}"

        cropped_image = image[t:b, l:r, :]
        padding_l = max(0, image_l - cropper_l)
        padding_t = max(0, image_t - cropper_t)
        padding_r = max(0, cropper_r - image_r)
        padding_b = max(0, cropper_b - image_b)

        expanded_image, mask_image = expand_image(
            cropped_image,
            left=padding_l,
            top=padding_t,
            right=padding_r,
            bottom=padding_b,
        )

        # 最终扩大了的 image, BGR
        expanded_cropped_result_image = self._scaled_pad_forward(
            expanded_image, mask_image, config
        )

        # RGB -> BGR
        outpainting_image = cv2.copyMakeBorder(
            image,
            left=padding_l,
            top=padding_t,
            right=padding_r,
            bottom=padding_b,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )[:, :, ::-1]

        # 把 cropped_result_image 贴到 outpainting_image 上，这一步不需要 blend
        paste_t = 0 if config.extender_y < 0 else config.extender_y
        paste_l = 0 if config.extender_x < 0 else config.extender_x

        outpainting_image[
            paste_t : paste_t + expanded_cropped_result_image.shape[0],
            paste_l : paste_l + expanded_cropped_result_image.shape[1],
            :,
        ] = expanded_cropped_result_image
        return outpainting_image

    def _scaled_pad_forward(self, image, mask, config: InpaintRequest):
        longer_side_length = int(config.sd_scale * max(image.shape[:2]))
        origin_size = image.shape[:2]
        downsize_image = resize_max_size(image, size_limit=longer_side_length)
        downsize_mask = resize_max_size(mask, size_limit=longer_side_length)
        if config.sd_scale != 1:
            logger.info(
                f"Resize image to do sd inpainting: {image.shape} -> {downsize_image.shape}"
            )
        inpaint_result = self._pad_forward(downsize_image, downsize_mask, config)
        # only paste masked area result
        inpaint_result = cv2.resize(
            inpaint_result,
            (origin_size[1], origin_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        return inpaint_result

    def set_scheduler(self, config: InpaintRequest):
        scheduler_config = self.model.scheduler.config
        sd_sampler = config.sd_sampler
        if config.sd_lcm_lora and self.model_info.support_lcm_lora:
            sd_sampler = SDSampler.lcm
            logger.info(f"LCM Lora enabled, use {sd_sampler} sampler")
        cache_key = (sd_sampler, config.sd_lcm_lora)
        scheduler = self._scheduler_cache.get(cache_key)
        if scheduler is None:
            scheduler = get_scheduler(sd_sampler, scheduler_config)
            self._scheduler_cache[cache_key] = scheduler
        self.model.scheduler = scheduler

    # 合成羽化半径。原 sd_mask_blur 参数已按 D-17 删除，但 sd_keep_unmasked_area
    # 的合成需要羽化边缘：实测去掉羽化后接缝会从平滑过渡变成硬边
    # （默认路径 4660px 差异 / 单通道最大 140，outpainting 14407px / 141），
    # 所以固定沿用原默认值 11，保证默认输出与删除前逐像素一致。
    COMPOSITE_MASK_BLUR = 11

    def forward_post_process(self, result, image, mask, config):
        # 羽化放在直方图匹配之前：_match_histograms 拿到的一向是羽化后的 mask，
        # 这个口径不能变
        k = 2 * self.COMPOSITE_MASK_BLUR + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

        if config.sd_match_histograms:
            result = self._match_histograms(result, image[:, :, ::-1], mask)
        return result, image, mask
