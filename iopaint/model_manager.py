from typing import List, Dict, Optional
import threading

import torch
from fastapi import HTTPException
from loguru import logger
import numpy as np

from iopaint.download import scan_models
from iopaint.helper import switch_mps_device
from iopaint.model import models, ControlNet, SD, SDXL
from iopaint.model.brushnet.brushnet_wrapper import BrushNetWrapper
from iopaint.model.brushnet.brushnet_xl_wrapper import BrushNetXLWrapper
from iopaint.model.power_paint.power_paint_v2 import PowerPaintV2
from iopaint.model.utils import torch_gc, is_local_files_only
from iopaint.runtime import (
    load_with_fallback,
    device_str,
    is_out_of_memory,
    next_bigger_gpu,
    _release_memory,
)
from iopaint.schema import InpaintRequest, ModelInfo, ModelType


class ModelManager:
    def __init__(self, name: str, device: torch.device, **kwargs):
        self.name = name
        self.device = device
        # 显存不足触发的自动回退提示，去重后通过 /api/v1/server-config 下发给 WebUI
        self.device_notices: List[str] = []
        self.kwargs = kwargs
        self.lock = threading.Lock()
        self.available_models: Dict[str, ModelInfo] = {}
        self.scan_models()
        self.enable_controlnet = kwargs.get("enable_controlnet", False)
        controlnet_method = kwargs.get("controlnet_method", None)
        if (
            controlnet_method is None
            and name in self.available_models
            and self.available_models[name].support_controlnet
        ):
            controlnet_method = self.available_models[name].controlnets[0]
        self.controlnet_method = controlnet_method

        self.enable_brushnet = kwargs.get("enable_brushnet", False)
        self.brushnet_method = kwargs.get("brushnet_method", None)

        self.enable_powerpaint_v2 = kwargs.get("enable_powerpaint_v2", False)

        self.model, self.device = self._load(name, self.device)

    @property
    def device_notice(self) -> Optional[str]:
        """去重后的回退提示，空列表返回 None。"""
        return "; ".join(dict.fromkeys(self.device_notices)) or None

    def _load(self, name: str, device, **extra_kwargs):
        """加载模型；显存不足时自动换到更有富余的 GPU，最后回退 CPU。

        返回 (model, 实际生效的 device)。非 OOM 异常原样抛出。
        """
        base_kwargs = {**self.kwargs, **extra_kwargs}
        notices: List[str] = []

        def loader(dev):
            try:
                return self.init_model(name, dev, **base_kwargs)
            except Exception as e:
                if not self._should_retry_fp16(dev, e, base_kwargs):
                    raise
                err = str(e).splitlines()[0]
                # 必须在 e 还活着时切断 traceback 链：它挂着 loader → init_model →
                # 只装了一半的 pipeline（模块间互相引用成环），不清就回收不掉
                e.__traceback__ = None
                e.__context__ = None
                e.__cause__ = None
                failed = device_str(dev)
                _release_memory(dev)
                # 同一张卡退 fp16：大模型（PowerPaint V2/SDXL）fp32 装不下但 fp16 能装；
                # 这排在 load_with_fallback 换卡/回退 CPU 之前。fp16 也失败就交给外层换卡，
                # 不额外记提示，免得和换卡提示重复。
                model = self.init_model(
                    name, dev, **{**base_kwargs, "force_fp16": True}
                )
                msg = f"{failed} out of VRAM with fp32, retried with fp16"
                logger.warning(f"{msg} (cause: {err})")
                notices.append(msg)
                return model

        model, effective_device, runtime_notices = load_with_fallback(loader, device)
        for it in [*notices, *runtime_notices]:
            if it not in self.device_notices:
                self.device_notices.append(it)
        # 终端里明确写出「模型最终跑在哪」，和启动时打印的 GPU 清单对得上
        logger.info(f"Model {name} loaded on {device_str(effective_device)}")
        return model, effective_device

    @staticmethod
    def _should_retry_fp16(dev, e: BaseException, kwargs: dict) -> bool:
        """fp32 装不下时，是否在同一张卡上退到 fp16。

        只在 cuda + OOM + 用户没显式要 fp32(`no_half`) + 还没退过(`force_fp16`) 时重试。
        """
        return (
            device_str(dev).split(":")[0] == "cuda"
            and is_out_of_memory(e)
            and not kwargs.get("no_half", False)
            and not kwargs.get("force_fp16", False)
        )

    @property
    def current_model(self) -> ModelInfo:
        return self.available_models[self.name]

    def init_model(self, name: str, device, **kwargs):
        logger.info(f"Loading model: {name}")
        if name not in self.available_models:
            raise NotImplementedError(
                f"Unsupported model: {name}. Available models: {list(self.available_models.keys())}"
            )

        model_info = self.available_models[name]
        kwargs = {
            **kwargs,
            "model_info": model_info,
            "enable_controlnet": self.enable_controlnet,
            "controlnet_method": self.controlnet_method,
            "enable_brushnet": self.enable_brushnet,
            "brushnet_method": self.brushnet_method,
        }

        if model_info.support_controlnet and self.enable_controlnet:
            return ControlNet(device, **kwargs)

        if model_info.support_brushnet and self.enable_brushnet:
            if model_info.model_type == ModelType.DIFFUSERS_SD:
                return BrushNetWrapper(device, **kwargs)
            elif model_info.model_type == ModelType.DIFFUSERS_SDXL:
                return BrushNetXLWrapper(device, **kwargs)

        if model_info.support_powerpaint_v2 and self.enable_powerpaint_v2:
            return PowerPaintV2(device, **kwargs)

        if model_info.name in models:
            return models[name](device, **kwargs)

        if model_info.model_type in [
            ModelType.DIFFUSERS_SD_INPAINT,
            ModelType.DIFFUSERS_SD,
        ]:
            return SD(device, **kwargs)

        if model_info.model_type in [
            ModelType.DIFFUSERS_SDXL_INPAINT,
            ModelType.DIFFUSERS_SDXL,
        ]:
            return SDXL(device, **kwargs)

        raise NotImplementedError(f"Unsupported model: {name}")

    @torch.inference_mode()
    def __call__(self, image, mask, config: InpaintRequest):
        """
        Args:
            image: [H, W, C] RGB
            mask: [H, W, 1] 255 means area to repaint
            config:

        Returns:
            BGR image
        """
        # Serialize model calls: the underlying pipeline shares mutable state
        # (e.g. self.model.scheduler is swapped per request), so concurrent
        # calls would race on it.
        with self.lock:
            return self._run(image, mask, config)

    def _run(self, image, mask, config: InpaintRequest):
        # 切换失败且回滚也失败时 self.model 为 None（REL-2/D-9）；
        # 给出明确 503 而不是 AttributeError 级联
        if self.model is None:
            raise HTTPException(
                status_code=503,
                detail="Model load failed, please switch model",
            )
        self._prepare_model(config)
        return self._infer_with_oom_fallback(image, mask, config)

    def _prepare_model(self, config: InpaintRequest) -> None:
        """按请求切换模型侧的状态（controlnet / brushnet / powerpaint / lcm lora）。"""
        if config.enable_controlnet:
            self.switch_controlnet_method(config)
        if config.enable_brushnet:
            self.switch_brushnet_method(config)

        self.enable_disable_powerpaint_v2(config)
        self.enable_disable_lcm_lora(config)

    def _infer_with_oom_fallback(
        self, image: np.ndarray, mask: np.ndarray, config: InpaintRequest
    ) -> np.ndarray:
        """跑一次推理；显存不足时换到更有富余的 GPU，最后回退 CPU 重试。

        模型能装下不代表推理装得下：`load_with_fallback` 只护住加载，erase/diffusion
        模型在大图上的一次前向仍可能 OOM（本机 P620 2G 上，90MP 图 + 较长笔画把 crop
        裁到 2591x900，LaMa 前向要 574MB）。此前的行为是直接把 torch 的 OOM 抛成 500，
        WebUI 只弹一条塞满 TorchScript traceback 的 toast —— 用户看到的就是「日志说
        要处理，实际啥也没干」。这里复用加载回退的同一套选卡逻辑，在推理失败时把
        模型重新加载到下一块卡（或 CPU）再重试，并把提示记进 device_notice。
        """
        tried: List = []
        while True:
            try:
                return self.model(image, mask, config).astype(np.uint8)
            except Exception as e:
                if (
                    device_str(self.device).split(":")[0] != "cuda"
                    or not is_out_of_memory(e)
                ):
                    raise
                tried.append(self.device)
                failed = device_str(self.device)
                err = str(e).splitlines()[0]
                # 同 _load：必须在 e 还活着时切断 traceback 链，否则它挂着
                # self.model → 失败卡上的权重，下面的回收只是空转。
                e.__traceback__ = None
                e.__context__ = None
                e.__cause__ = None
                logger.warning(f"Inference on {failed} ran out of VRAM: {err}")
                # 丢掉模型引用才能把失败那张卡的显存还回去
                model = self.model
                self.model = None
                del model
                _release_memory(self.device)

                bigger = next_bigger_gpu(tried)
                if bigger is not None:
                    target = torch.device(f"cuda:{bigger}")
                    msg = f"{failed} out of VRAM during inference, switched to cuda:{bigger}"
                else:
                    target = torch.device("cpu")
                    msg = (
                        f"{failed} out of VRAM during inference and no other GPU has "
                        f"more headroom, fell back to CPU (inference will be much slower)"
                    )
                logger.warning(msg)
                if msg not in self.device_notices:
                    self.device_notices.append(msg)

                # 权重 / dtype / offload 状态都绑定在设备上，换设备只能重新加载
                model, effective_device = self._load(self.name, target)
                self.model = model
                self.device = effective_device
                self._prepare_model(config)

    def scan_models(self) -> List[ModelInfo]:
        available_models = scan_models()
        self.available_models = {it.name: it for it in available_models}
        return available_models

    def get_available_models(self) -> List[ModelInfo]:
        return list(self.available_models.values())

    def switch(self, new_name: str):
        # 未知模型名先拒（此前 available_models[new_name] 抛 KeyError → 500，
        # 且此时 self.name 已经被改成新值，状态被污染）
        if new_name not in self.available_models:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown model: {new_name}",
            )
        if new_name == self.name:
            return

        with self.lock:
            old_name = self.name
            old_controlnet_method = self.controlnet_method

            new_controlnet_method = old_controlnet_method
            if (
                self.available_models[new_name].support_controlnet
                and new_controlnet_method
                not in self.available_models[new_name].controlnets
            ):
                new_controlnet_method = self.available_models[new_name].controlnets[0]

            # init_model 内部读 self.controlnet_method，所以新值先提交；
            # self.name 等 init_model 成功之后再提交（REL-2/D-9）。
            # 中途失败一律回滚到旧状态，避免"名字换了模型没换"。
            self.controlnet_method = new_controlnet_method
            # 原实现是 del self.model，失败后 AttributeError 会级联；
            # 置 None 让 _run 能给出 503
            self.model = None
            torch_gc()

            try:
                # TODO: enable/disable controlnet without reload model
                model, effective_device = self._load(
                    new_name, switch_mps_device(new_name, self.device)
                )
            except Exception as e:
                self.controlnet_method = old_controlnet_method
                logger.info(
                    f"Switch model from {old_name} to {new_name} failed: {e}, rollback"
                )
                try:
                    self.model = self.init_model(
                        old_name, switch_mps_device(old_name, self.device), **self.kwargs
                    )
                except Exception as rollback_err:
                    logger.error(
                        f"Rollback to model {old_name} failed too: {rollback_err}"
                    )
                    self.model = None
                raise
            self.name = new_name
            self.model = model
            self.device = effective_device

    def switch_brushnet_method(self, config):
        if not self.available_models[self.name].support_brushnet:
            return

        if (
            self.enable_brushnet
            and config.brushnet_method
            and self.brushnet_method != config.brushnet_method
        ):
            old_brushnet_method = self.brushnet_method
            # 先让模型完成切换，成功后再提交状态（失败则保留旧值）
            self.model.switch_brushnet_method(config.brushnet_method)
            self.brushnet_method = config.brushnet_method
            logger.info(
                f"Switch Brushnet method from {old_brushnet_method} to {config.brushnet_method}"
            )

        elif self.enable_brushnet != config.enable_brushnet:
            old_enable_brushnet = self.enable_brushnet
            old_brushnet_method = self.brushnet_method
            self.enable_brushnet = config.enable_brushnet
            self.brushnet_method = config.brushnet_method

            pipe_components = {
                "vae": self.model.model.vae,
                "text_encoder": self.model.model.text_encoder,
                "unet": self.model.model.unet,
            }
            if hasattr(self.model.model, "text_encoder_2"):
                pipe_components["text_encoder_2"] = self.model.model.text_encoder_2
            if hasattr(self.model.model, "tokenizer"):
                pipe_components["tokenizer"] = self.model.model.tokenizer
            if hasattr(self.model.model, "tokenizer_2"):
                pipe_components["tokenizer_2"] = self.model.model.tokenizer_2

            try:
                model, effective_device = self._load(
                    self.name,
                    switch_mps_device(self.name, self.device),
                    pipe_components=pipe_components,
                )
            except Exception as e:
                self.enable_brushnet = old_enable_brushnet
                self.brushnet_method = old_brushnet_method
                logger.error(f"Switch Brushnet enable/disable failed: {e}")
                raise
            self.model = model
            self.device = effective_device

            if not config.enable_brushnet:
                logger.info("BrushNet Disabled")
            else:
                logger.info("BrushNet Enabled")

    def switch_controlnet_method(self, config):
        if not self.available_models[self.name].support_controlnet:
            return

        if (
            self.enable_controlnet
            and config.controlnet_method
            and self.controlnet_method != config.controlnet_method
        ):
            old_controlnet_method = self.controlnet_method
            # 先让模型完成切换，成功后再提交状态（失败则保留旧值）
            self.model.switch_controlnet_method(config.controlnet_method)
            self.controlnet_method = config.controlnet_method
            logger.info(
                f"Switch Controlnet method from {old_controlnet_method} to {config.controlnet_method}"
            )
        elif self.enable_controlnet != config.enable_controlnet:
            old_enable_controlnet = self.enable_controlnet
            old_controlnet_method = self.controlnet_method
            self.enable_controlnet = config.enable_controlnet
            self.controlnet_method = config.controlnet_method

            pipe_components = {
                "vae": self.model.model.vae,
                "text_encoder": self.model.model.text_encoder,
                "unet": self.model.model.unet,
            }
            if hasattr(self.model.model, "text_encoder_2"):
                pipe_components["text_encoder_2"] = self.model.model.text_encoder_2

            try:
                model, effective_device = self._load(
                    self.name,
                    switch_mps_device(self.name, self.device),
                    pipe_components=pipe_components,
                )
            except Exception as e:
                # init_model 失败时旧模型还在，只需把状态改回去（REL-2/D-9）
                self.enable_controlnet = old_enable_controlnet
                self.controlnet_method = old_controlnet_method
                logger.error(f"Switch controlnet enable/disable failed: {e}")
                raise
            self.model = model
            self.device = effective_device
            if not config.enable_controlnet:
                logger.info("Disable controlnet")
            else:
                logger.info(f"Enable controlnet: {config.controlnet_method}")

    def enable_disable_powerpaint_v2(self, config: InpaintRequest):
        if not self.available_models[self.name].support_powerpaint_v2:
            return

        if self.enable_powerpaint_v2 != config.enable_powerpaint_v2:
            old_enable_powerpaint_v2 = self.enable_powerpaint_v2
            self.enable_powerpaint_v2 = config.enable_powerpaint_v2

            # PowerPaintV2 会重新加载全部权重，不像 brushnet/controlnet 复用 pipe_components；
            # 旧模型留在显存里会让新模型装不下（切 V2 时 OOM→回退 CPU）。先释放再加载，
            # 失败再回滚重载旧模型。
            self.model = None
            torch_gc()

            try:
                model, effective_device = self._load(
                    self.name, switch_mps_device(self.name, self.device)
                )
            except Exception as e:
                self.enable_powerpaint_v2 = old_enable_powerpaint_v2
                logger.error(f"Switch PowerPaintV2 enable/disable failed: {e}")
                try:
                    self.model = self.init_model(
                        self.name,
                        switch_mps_device(self.name, self.device),
                        **self.kwargs,
                    )
                except Exception as rollback_err:
                    logger.error(f"Rollback PowerPaintV2 failed too: {rollback_err}")
                    self.model = None
                raise
            self.model = model
            self.device = effective_device
            if config.enable_powerpaint_v2:
                logger.info("Enable PowerPaintV2")
            else:
                logger.info("Disable PowerPaintV2")

    def enable_disable_lcm_lora(self, config: InpaintRequest):
        if not self.available_models[self.name].support_lcm_lora:
            return

        if config.sd_lcm_lora:
            # TODO: change this if load other lora is supported
            if not bool(self.model.model.get_list_adapters()):
                logger.info("Load LCM LORA")
                self.model.model.load_lora_weights(
                    self.model.lcm_lora_id,
                    weight_name="pytorch_lora_weights.safetensors",
                    local_files_only=is_local_files_only(),
                )
            else:
                logger.info("Enable LCM LORA")
                self.model.model.enable_lora()
        elif bool(self.model.model.get_list_adapters()):
            logger.info("Disable LCM LORA")
            self.model.model.disable_lora()
