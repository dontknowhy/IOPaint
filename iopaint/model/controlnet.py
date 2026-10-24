from typing import Any, cast

import PIL.Image
import cv2
import torch
from diffusers import ControlNetModel
from loguru import logger
from iopaint.schema import InpaintRequest, ModelType

from .base import DiffusionInpaintModel
from .helper.controlnet_preprocess import (
    make_canny_control_image,
    make_openpose_control_image,
    make_depth_control_image,
    make_inpaint_control_image,
)
from .helper.cpu_text_encoder import CPUTextEncoderWrapper
from .original_sd_configs import get_config_files
from .utils import (
    cuda_device_index,
    get_scheduler,
    handle_from_pretrained_exceptions,
    get_torch_dtype,
    enable_low_mem,
    is_local_files_only,
)


class ControlNet(DiffusionInpaintModel):
    name = "controlnet"
    pad_mod = 8
    min_size = 512

    @property
    def lcm_lora_id(self):
        if self.model_info.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SD_INPAINT,
        ]:
            return "latent-consistency/lcm-lora-sdv1-5"
        if self.model_info.model_type in [
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]:
            return "latent-consistency/lcm-lora-sdxl"
        raise NotImplementedError(f"Unsupported controlnet lcm model {self.model_info}")

    def init_model(self, device: torch.device, **kwargs):
        model_info = kwargs["model_info"]
        controlnet_method = kwargs["controlnet_method"]

        self.model_info = model_info
        self.controlnet_method = controlnet_method

        model_kwargs = {
            **kwargs.get("pipe_components", {}),
            "local_files_only": is_local_files_only(**kwargs),
        }
        self.local_files_only = model_kwargs["local_files_only"]

        disable_nsfw_checker = kwargs["disable_nsfw"] or kwargs.get(
            "cpu_offload", False
        )
        if disable_nsfw_checker:
            logger.info("Disable Stable Diffusion Model NSFW checker")
            model_kwargs.update(
                dict(
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False,
                )
            )

        use_gpu, torch_dtype = get_torch_dtype(device, kwargs.get("no_half", False), kwargs.get("force_fp16", False))
        self.torch_dtype = torch_dtype

        original_config_file_name = "v1"
        if model_info.model_type in [
            ModelType.DIFFUSERS_SD,
            ModelType.DIFFUSERS_SD_INPAINT,
        ]:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline as PipeClass,
            )

            original_config_file_name = "v1"

        elif model_info.model_type in [
            ModelType.DIFFUSERS_SDXL,
            ModelType.DIFFUSERS_SDXL_INPAINT,
        ]:
            from diffusers import (
                StableDiffusionXLControlNetInpaintPipeline as PipeClass,
            )

            original_config_file_name = "xl"

        self._controlnet_cache = kwargs.get("controlnet_module_cache")
        controlnet = self._load_controlnet(controlnet_method)
        if model_info.is_single_file_diffusers:
            if self.model_info.model_type == ModelType.DIFFUSERS_SD:
                model_kwargs["num_in_channels"] = 4
            else:
                model_kwargs["num_in_channels"] = 9

            self.model = PipeClass.from_single_file(
                model_info.path,
                controlnet=controlnet,
                load_safety_checker=not disable_nsfw_checker,
                torch_dtype=torch_dtype,
                original_config_file=get_config_files()[original_config_file_name],
                **model_kwargs,
            )
        else:
            self.model = handle_from_pretrained_exceptions(
                PipeClass.from_pretrained,
                pretrained_model_name_or_path=model_info.path,
                controlnet=controlnet,
                variant="fp16",
                torch_dtype=torch_dtype,
                **model_kwargs,
            )

        enable_low_mem(self.model, kwargs.get("low_mem", False))

        if kwargs.get("cpu_offload", False) and use_gpu:
            logger.info("Enable sequential cpu offload")
            self.model.enable_sequential_cpu_offload(gpu_id=cuda_device_index(device))
        else:
            self.model = self.model.to(device)
            if kwargs["sd_cpu_textencoder"]:
                logger.info("Run Stable Diffusion TextEncoder on CPU")
                self.model.text_encoder = CPUTextEncoderWrapper(
                    self.model.text_encoder, torch_dtype
                )

        self.callback = kwargs.pop("callback", None)

    def _load_controlnet(self, method: str) -> ControlNetModel:
        """按 method 取 ControlNet 模块，命中 manager 级缓存就不读盘。

        缓存只留最近一个 method：开关切换会重建 pipeline（类不同，无法只换引用），
        但模块本身可跨 pipeline 复用。dtype 不匹配（fp32 加载失败后回退 fp16）
        直接丢弃重载，避免把 fp32 权重挂到 fp16 pipeline 上。
        """
        cache = self._controlnet_cache
        if cache is not None:
            cached = cache.get(method)
            if cached is not None and cached.dtype == self.torch_dtype:
                logger.info(f"Reuse cached ControlNet module: {method}")
                return cached
            cache.pop(method, None)
        controlnet = ControlNetModel.from_pretrained(
            pretrained_model_name_or_path=method,
            local_files_only=self.local_files_only,
            torch_dtype=self.torch_dtype,
        )
        if cache is not None:
            cache.clear()
            cache[method] = controlnet
        return controlnet

    def switch_controlnet_method(self, new_method: str) -> None:
        self.controlnet_method = new_method
        controlnet = cast(Any, self._load_controlnet(new_method)).to(
            self.model.device
        )
        self.model.controlnet = controlnet

    def _get_control_image(self, image, mask):
        if "canny" in self.controlnet_method:
            control_image = make_canny_control_image(image)
        elif "openpose" in self.controlnet_method:
            control_image = make_openpose_control_image(image)
        elif "depth" in self.controlnet_method:
            control_image = make_depth_control_image(image)
        elif "inpaint" in self.controlnet_method:
            control_image = make_inpaint_control_image(image, mask)
        else:
            raise NotImplementedError(f"{self.controlnet_method} not implemented")
        return control_image

    def forward(self, image, mask, config: InpaintRequest):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """
        scheduler_config = self.model.scheduler.config
        cache_key = (config.sd_sampler, False)
        scheduler = self._scheduler_cache.get(cache_key)
        if scheduler is None:
            scheduler = get_scheduler(config.sd_sampler, scheduler_config)
            self._scheduler_cache[cache_key] = scheduler
        self.model.scheduler = scheduler

        img_h, img_w = image.shape[:2]
        control_image = self._get_control_image(image, mask)
        mask_image = PIL.Image.fromarray(mask[:, :, -1], mode="L")
        image = PIL.Image.fromarray(image)

        output = self.model(
            image=image,
            mask_image=mask_image,
            control_image=control_image,
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            num_inference_steps=config.sd_steps,
            guidance_scale=config.sd_guidance_scale,
            output_type="np",
            callback_on_step_end=self.callback,
            height=img_h,
            width=img_w,
            generator=torch.manual_seed(config.sd_seed),
            controlnet_conditioning_scale=config.controlnet_conditioning_scale,
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output
