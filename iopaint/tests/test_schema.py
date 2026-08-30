"""Tests for iopaint.schema – Pydantic models and validation."""
import random

import pytest

from iopaint.schema import (
    AdjustMaskRequest,
    ApiConfig,
    Choices,
    CV2Flag,
    Device,
    HDStrategy,
    InpaintRequest,
    LDMSampler,
    ModelInfo,
    ModelType,
    PowerPaintTask,
    RealESRGANModel,
    RemoveBGModel,
    RunPluginRequest,
    SDSampler,
    InteractiveSegModel,
)


# ---------------------------------------------------------------------------
# Choices base class
# ---------------------------------------------------------------------------

class TestChoices:
    def test_values_returns_list(self):
        vals = Device.values()
        assert isinstance(vals, list)
        assert "cpu" in vals

    def test_all_choices_have_values(self):
        for cls in [RealESRGANModel, RemoveBGModel, Device, InteractiveSegModel]:
            vals = cls.values()
            assert len(vals) > 0


# ---------------------------------------------------------------------------
# ModelInfo computed fields
# ---------------------------------------------------------------------------

class TestModelInfo:
    def _make(self, model_type=ModelType.INPAINT, name="lama"):
        return ModelInfo(name=name, path="test", model_type=model_type)

    def test_inpaint_need_prompt_false(self):
        info = self._make(ModelType.INPAINT)
        assert info.need_prompt is False

    def test_diffusers_sd_need_prompt_true(self):
        info = self._make(ModelType.DIFFUSERS_SD)
        assert info.need_prompt is True

    def test_diffusers_sdxl_inpaint_need_prompt(self):
        info = self._make(ModelType.DIFFUSERS_SDXL_INPAINT)
        assert info.need_prompt is True

    def test_powerpaint_need_prompt(self):
        from iopaint.const import POWERPAINT_NAME
        info = self._make(ModelType.DIFFUSERS_OTHER, name=POWERPAINT_NAME)
        assert info.need_prompt is True

    def test_controlnets_empty_for_inpaint(self):
        info = self._make(ModelType.INPAINT)
        assert info.controlnets == []

    def test_controlnets_sd(self):
        info = self._make(ModelType.DIFFUSERS_SD)
        assert len(info.controlnets) > 0

    def test_controlnets_sdxl(self):
        info = self._make(ModelType.DIFFUSERS_SDXL)
        assert len(info.controlnets) > 0

    def test_controlnets_sd2(self):
        info = self._make(ModelType.DIFFUSERS_SD, name="sd21-base")
        assert "thibaud/controlnet-sd21" in str(info.controlnets)

    def test_brushnets_sd(self):
        info = self._make(ModelType.DIFFUSERS_SD)
        assert len(info.brushnets) > 0

    def test_brushnets_empty_for_inpaint(self):
        info = self._make(ModelType.INPAINT)
        assert info.brushnets == []

    def test_support_strength_diffusers(self):
        info = self._make(ModelType.DIFFUSERS_SD)
        assert info.support_strength is True

    def test_support_strength_inpaint(self):
        info = self._make(ModelType.INPAINT)
        assert info.support_strength is False

    def test_support_outpainting(self):
        info = self._make(ModelType.DIFFUSERS_SD)
        assert info.support_outpainting is True

    def test_support_lcm_lora(self):
        info = self._make(ModelType.DIFFUSERS_SDXL)
        assert info.support_lcm_lora is True

    def test_support_controlnet(self):
        info = self._make(ModelType.DIFFUSERS_SD_INPAINT)
        assert info.support_controlnet is True

    def test_support_brushnet_sdxl(self):
        info = self._make(ModelType.DIFFUSERS_SDXL)
        assert info.support_brushnet is True

    def test_support_powerpaint_v2_sd(self):
        from iopaint.const import POWERPAINT_NAME
        info = self._make(ModelType.DIFFUSERS_SD, name="custom-model")
        assert info.support_powerpaint_v2 is True

    def test_support_powerpaint_v2_excluded(self):
        from iopaint.const import POWERPAINT_NAME
        info = self._make(ModelType.DIFFUSERS_SD, name=POWERPAINT_NAME)
        assert info.support_powerpaint_v2 is False


# ---------------------------------------------------------------------------
# InpaintRequest
# ---------------------------------------------------------------------------

class TestInpaintRequest:
    def test_defaults(self):
        req = InpaintRequest(image="", mask="")
        assert req.ldm_steps == 20
        assert req.sd_seed == 42
        assert req.sd_strength == 1.0
        assert req.hd_strategy == HDStrategy.CROP

    def test_random_seed(self):
        req = InpaintRequest(image="", mask="", sd_seed=-1)
        assert 1 <= req.sd_seed <= 99999999

    def test_extender_disables_controlnet_scale(self):
        req = InpaintRequest(
            image="", mask="",
            use_extender=True,
            enable_controlnet=True,
            controlnet_conditioning_scale=0.5,
        )
        assert req.controlnet_conditioning_scale == 0

    def test_extender_sets_strength(self):
        req = InpaintRequest(image="", mask="", use_extender=True, sd_strength=0.5)
        assert req.sd_strength == 1.0

    def test_brushnet_disables_controlnet(self):
        req = InpaintRequest(
            image="", mask="",
            enable_brushnet=True,
            enable_controlnet=True,
        )
        assert req.enable_controlnet is False

    def test_brushnet_disables_lcm_lora(self):
        req = InpaintRequest(
            image="", mask="",
            enable_brushnet=True,
            sd_lcm_lora=True,
        )
        assert req.sd_lcm_lora is False

    def test_controlnet_does_not_disable_brushnet(self):
        # The validator only disables controlnet when brushnet is enabled, not vice versa
        req = InpaintRequest(
            image="", mask="",
            enable_controlnet=True,
            enable_brushnet=True,
        )
        assert req.enable_controlnet is False
        assert req.enable_brushnet is True


# ---------------------------------------------------------------------------
# AdjustMaskRequest
# ---------------------------------------------------------------------------

class TestAdjustMaskRequest:
    def test_expand(self):
        req = AdjustMaskRequest(mask="abc", operate="expand", kernel_size=5)
        assert req.operate == "expand"

    def test_shrink(self):
        req = AdjustMaskRequest(mask="abc", operate="shrink", kernel_size=10)
        assert req.kernel_size == 10

    def test_reverse(self):
        req = AdjustMaskRequest(mask="abc", operate="reverse")
        assert req.operate == "reverse"


# ---------------------------------------------------------------------------
# RunPluginRequest
# ---------------------------------------------------------------------------

class TestRunPluginRequest:
    def test_defaults(self):
        req = RunPluginRequest(name="test", image="abc")
        assert req.scale == 2.0
        assert req.clicks == []


# ---------------------------------------------------------------------------
# SDSampler / LDMSampler
# ---------------------------------------------------------------------------

class TestSamplers:
    def test_sdsampler_all_values(self):
        vals = [m.value for m in SDSampler.__members__.values()]
        assert "Euler" in vals
        assert "DDIM" in vals
        assert "LCM" in vals

    def test_ldmsampler_values(self):
        vals = [m.value for m in LDMSampler.__members__.values()]
        assert "ddim" in vals
        assert "plms" in vals


# ---------------------------------------------------------------------------
# Device / RemoveBGModel / RealESRGANModel
# ---------------------------------------------------------------------------

class TestDeviceAndModels:
    def test_device_values(self):
        assert set(Device.values()) == {"cpu", "cuda", "mps"}

    def test_removebg_has_models(self):
        vals = RemoveBGModel.values()
        assert len(vals) > 5

    def test_realesrgan_has_models(self):
        vals = RealESRGANModel.values()
        assert len(vals) >= 3
