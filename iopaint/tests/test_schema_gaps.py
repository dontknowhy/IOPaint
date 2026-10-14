"""schema.py 残余缺口补测：ModelInfo.controlnets 对 PowerPaint 的名字特判、
brushnets 的 SDXL 分支。
"""
from iopaint.const import (
    POWERPAINT_NAME,
    SD_CONTROLNET_CHOICES,
    SDXL_BRUSHNET_CHOICES,
)
from iopaint.schema import ModelInfo, ModelType


def test_powerpaint_name_falls_back_to_sd_controlnets():
    # PowerPaint 在 download.py 里登记为 DIFFUSERS_OTHER，controlnets 靠
    # name == POWERPAINT_NAME 的特判返回 SD 列表
    info = ModelInfo(
        name=POWERPAINT_NAME, path="test", model_type=ModelType.DIFFUSERS_OTHER
    )
    assert info.controlnets == SD_CONTROLNET_CHOICES


def test_sdxl_brushnets():
    info = ModelInfo(name="sdxl", path="test", model_type=ModelType.DIFFUSERS_SDXL)
    assert info.brushnets == SDXL_BRUSHNET_CHOICES
