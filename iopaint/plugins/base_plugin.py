import threading

from loguru import logger
import numpy as np

from iopaint.exceptions import ModelLoadError
from iopaint.schema import RunPluginRequest


class BasePlugin:
    name: str
    support_gen_image: bool = False
    support_gen_mask: bool = False

    def __init__(self):
        # CON-1: 插件实例带着可变状态（InteractiveSeg 的 predictor/prev_img_md5、
        # 各模型句柄），并发请求会互相踩。与 ModelManager.lock 同一模式：
        # 锁放在组件上、在 executor 内持锁，不在 async 层持锁。
        self.lock = threading.Lock()
        err_msg = self.check_dep()
        if err_msg:
            logger.error(err_msg)
            raise ModelLoadError(err_msg)

    def gen_image(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        # return RGBA np image or BGR np image
        ...

    def gen_mask(self, rgb_np_img, req: RunPluginRequest) -> np.ndarray:
        # return GRAY or BGR np image, 255 means foreground, 0 means background
        ...

    def check_dep(self):
        ...

    def switch_model(self, new_model_name: str):
        ...
