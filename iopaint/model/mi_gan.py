import os

import torch

from iopaint.helper import (
    load_jit_model,
    download_model,
    get_cache_path_by_url,
)
from .base import InpaintModel
from .helper.tensors import binary_mask, to_bgr_uint8, upload_uint8
from iopaint.schema import InpaintRequest

MIGAN_MODEL_URL = os.environ.get(
    "MIGAN_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt",
)
MIGAN_MODEL_MD5 = os.environ.get("MIGAN_MODEL_MD5", "76eb3b1a71c400ee3290524f7a11b89c")


class MIGAN(InpaintModel):
    name = "migan"
    min_size = 512
    pad_mod = 512
    pad_to_square = True
    fixed_size = 512
    is_erase_model = True

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(MIGAN_MODEL_URL, device, MIGAN_MODEL_MD5).eval()

    @staticmethod
    def download():
        download_model(MIGAN_MODEL_URL, MIGAN_MODEL_MD5)

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(MIGAN_MODEL_URL))

    def forward(self, image, mask, config: InpaintRequest):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W] mask area == 255
        return: BGR IMAGE
        """
        image = upload_uint8(image, self.device)
        image.mul_(2).sub_(1)  # [0, 1] -> [-1, 1]
        mask = binary_mask(mask, self.device, threshold=120)

        erased_img = image * (1 - mask)
        input_image = torch.cat([0.5 - mask, erased_img], dim=1)

        output = self.model(input_image)
        return to_bgr_uint8(output, scale=127.5, offset=127.5, rounding=True)
