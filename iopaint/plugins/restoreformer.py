from contextlib import nullcontext

import cv2
import numpy as np
from loguru import logger

from iopaint.helper import download_model
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.schema import RunPluginRequest


class RestoreFormerPlugin(BasePlugin):
    name = "RestoreFormer"
    support_gen_image = True

    def __init__(self, device, upscaler=None):
        super().__init__()
        from .gfpganer import MyGFPGANer

        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
        model_md5 = "eaeeff6c4a1caa1673977cb374e6f699"
        model_path = download_model(url, model_md5)
        logger.info(f"RestoreFormer model path: {model_path}")

        # upscaler 是 RealESRGAN 插件实例，只取它的 .model 当 bg_upsampler；
        # 但它的 RealESRGANer 把 img/output 存在实例上，所以推理要持它的锁，
        # 见 _enhance。
        self.upscaler = upscaler
        self.face_enhancer = MyGFPGANer(
            model_path=model_path,
            upscale=1,
            arch="RestoreFormer",
            channel_multiplier=2,
            device=device,
            bg_upsampler=upscaler.model if upscaler is not None else None,
        )

    def _enhance(self, bgr_np_img: np.ndarray) -> np.ndarray:
        """持 upscaler 插件的锁跑整次 enhance。

        bg_upsampler 是共享的 RealESRGANer，它的 img/output 是实例状态，两个
        插件各持各的 plugin.lock 时仍会互相踩。锁序恒为
        plugin.lock → upscaler.lock，RealESRGAN 插件只取自己的锁，不会成环。
        """
        lock = self.upscaler.lock if self.upscaler is not None else nullcontext()
        with lock:
            if self.upscaler is not None:
                # RealESRGAN 运行期 switch_model 只替换 upscaler.model；构造时抓的
                # 旧 RealESRGANer 会让背景继续用旧模型，必须在共享锁内重新绑定。
                self.face_enhancer.bg_upsampler = self.upscaler.model
            return self.face_enhancer.enhance(
                bgr_np_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5,
            )[2]

    def gen_image(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)
        logger.info(f"RestoreFormer input shape: {bgr_np_img.shape}")
        bgr_output = self._enhance(bgr_np_img)
        logger.info(f"RestoreFormer output shape: {bgr_output.shape}")
        return bgr_output
