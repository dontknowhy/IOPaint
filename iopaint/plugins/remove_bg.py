import os
import cv2
import numpy as np
from loguru import logger
import torch
from torch.hub import get_dir

from iopaint.i18n import cli_text
from iopaint.plugins.base_plugin import BasePlugin
from iopaint.schema import Device, RunPluginRequest, RemoveBGModel


def _rmbg_remove(device, *args, **kwargs):
    from rembg import remove

    return remove(*args, **kwargs)


def _version_tuple(v: str) -> tuple[int, ...]:
    """取前三段数字元组：字符串比较会把 "2.0.100" 误判为旧于 "2.0.59"。"""
    return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())


class RemoveBG(BasePlugin):
    name = "RemoveBG"
    support_gen_mask = True
    support_gen_image = True

    def __init__(self, model_name, device):
        # check_dep 需要 self.model_name 判断是否走 briaai 路径，赋值必须早于
        # BasePlugin.__init__。
        self.model_name = model_name
        self.device = device
        super().__init__()

        if model_name.startswith("birefnet"):
            import rembg

            if _version_tuple(rembg.__version__) < (2, 0, 59):
                raise ValueError(cli_text("cli.error.remove_bg_upgrade"))

        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")
        os.environ["U2NET_HOME"] = model_dir

        self._init_session(model_name)

    def _init_session(self, model_name: str):
        self.device_warning()

        if model_name == RemoveBGModel.briaai_rmbg_1_4:
            from iopaint.plugins.briarmbg import (
                create_briarmbg_session,
                briarmbg_process,
            )

            self.session = create_briarmbg_session().to(self.device)
            self.remove = briarmbg_process
        elif model_name == RemoveBGModel.briaai_rmbg_2_0:
            from iopaint.plugins.briarmbg2 import (
                create_briarmbg2_session,
                briarmbg2_process,
            )

            self.session = create_briarmbg2_session().to(self.device)
            self.remove = briarmbg2_process
        else:
            from rembg import new_session

            self.session = new_session(model_name=model_name)
            self.remove = _rmbg_remove

    def switch_model(self, new_model_name):
        if self.model_name == new_model_name:
            return

        logger.info(
            f"Switching removebg model from {self.model_name} to {new_model_name}"
        )
        self._init_session(new_model_name)
        self.model_name = new_model_name

    @torch.inference_mode()
    def gen_image(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)

        # return BGRA image
        output = self.remove(self.device, bgr_np_img, session=self.session)
        return cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)

    @torch.inference_mode()
    def gen_mask(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        bgr_np_img = cv2.cvtColor(rgb_np_img, cv2.COLOR_RGB2BGR)

        # return BGR image, 255 means foreground, 0 means background
        output = self.remove(
            self.device, bgr_np_img, session=self.session, only_mask=True
        )
        return output

    def check_dep(self):
        # briaai 模型走 torch + huggingface_hub 自己的 session，不需要 rembg；
        # 其余模型（含 birefnet*）仍用 rembg 的 new_session。
        if self.model_name in (
            RemoveBGModel.briaai_rmbg_1_4,
            RemoveBGModel.briaai_rmbg_2_0,
        ):
            return None
        try:
            import rembg
        except ImportError as e:
            import traceback

            error_msg = traceback.format_exc()
            return f"Install rembg failed, Error details:\n{error_msg}"

    def device_warning(self):
        if self.device == Device.cuda and self.model_name not in [
            RemoveBGModel.briaai_rmbg_1_4,
            RemoveBGModel.briaai_rmbg_2_0,
        ]:
            logger.warning(
                f"remove_bg_device=cuda only supports briaai models({RemoveBGModel.briaai_rmbg_1_4.value}/{RemoveBGModel.briaai_rmbg_2_0.value})"
            )
