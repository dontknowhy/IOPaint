import os
import time

import cv2
import torch
from loguru import logger

from iopaint.helper import get_cache_path_by_url, load_jit_model, download_model
from .base import InpaintModel
from .helper.tensors import binary_mask, to_bgr_uint8
from iopaint.schema import InpaintRequest


MANGA_INPAINTOR_MODEL_URL = os.environ.get(
    "MANGA_INPAINTOR_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/manga/manga_inpaintor.jit",
)
MANGA_INPAINTOR_MODEL_MD5 = os.environ.get(
    "MANGA_INPAINTOR_MODEL_MD5", "7d8b269c4613b6b3768af714610da86c"
)

MANGA_LINE_MODEL_URL = os.environ.get(
    "MANGA_LINE_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/manga/erika.jit",
)
MANGA_LINE_MODEL_MD5 = os.environ.get(
    "MANGA_LINE_MODEL_MD5", "0c926d5a4af8450b0d00bc5b9a095644"
)


class Manga(InpaintModel):
    name = "manga"
    pad_mod = 16
    is_erase_model = True

    def init_model(self, device, **kwargs):
        self.inpaintor_model = load_jit_model(
            MANGA_INPAINTOR_MODEL_URL, device, MANGA_INPAINTOR_MODEL_MD5
        )
        self.line_model = load_jit_model(
            MANGA_LINE_MODEL_URL, device, MANGA_LINE_MODEL_MD5
        )
        self.seed = 42

    @staticmethod
    def download():
        download_model(MANGA_INPAINTOR_MODEL_URL, MANGA_INPAINTOR_MODEL_MD5)
        download_model(MANGA_LINE_MODEL_URL, MANGA_LINE_MODEL_MD5)

    @staticmethod
    def is_downloaded() -> bool:
        model_paths = [
            get_cache_path_by_url(MANGA_INPAINTOR_MODEL_URL),
            get_cache_path_by_url(MANGA_LINE_MODEL_URL),
        ]
        return all([os.path.exists(it) for it in model_paths])

    def forward(self, image, mask, config: InpaintRequest):
        """
        image: [H, W, C] RGB
        mask: [H, W, 1]
        return: BGR IMAGE
        """
        # 局部 generator：噪声序列与原先的全局 manual_seed 逐位相同，但不再每请求
        # 重置 random/numpy/torch 的全局种子。
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_img = (
            torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(self.device).float()
        )
        start = time.time()
        lines = self.line_model(gray_img)
        lines = torch.clamp(lines, 0, 255)
        logger.debug(f"erika_model time: {time.time() - start}")

        mask = binary_mask(mask, self.device)
        noise = torch.randn(
            mask.shape, generator=generator, dtype=mask.dtype, device=self.device
        )
        ones = torch.ones_like(mask)

        gray_img = gray_img / 255 * 2 - 1.0
        lines = lines / 255 * 2 - 1.0

        start = time.time()
        inpainted_image = self.inpaintor_model(gray_img, lines, mask, noise, ones)
        logger.debug(f"image_inpaintor_model time: {time.time() - start}")

        return to_bgr_uint8(inpainted_image, scale=127.5, offset=127.5)
