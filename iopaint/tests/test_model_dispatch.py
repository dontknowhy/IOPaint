"""model_manager 的装载分派与运行时开关。

不加载任何权重：scan_models 打桩、六类模型构造器（ControlNet/SD/SDXL/
BrushNetWrapper/BrushNetXLWrapper/PowerPaintV2）与 erase 模型注册表 models
全部替换为记录器，只验证 init_model 的分派优先级、kwargs 透传，以及
switch_controlnet/brushnet、enable_disable_powerpaint_v2/lcm_lora 的状态提交与回滚。

与 test_model_state.py 的分工：那里验证「切换失败如何回滚」，这里验证
「什么情况下装载哪个类」和「开关切换成功/失败后的状态」。
"""
import numpy as np
import pytest
import torch

from iopaint.const import SD_BRUSHNET_CHOICES, SD_CONTROLNET_CHOICES
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest, ModelInfo, ModelType

LAMA = "lama"
FAKE_ERASE = "fake-erase"
OTHER = "unsupported-model"
SD_INPAINT_NAME = "runwayml/stable-diffusion-inpainting"
SDXL_INPAINT_NAME = "stabilityai/stable-diffusion-xl-inpainting"
SD_NAME = "sd-normal"
SDXL_NAME = "sdxl-normal"
ALL_MODELS = [LAMA, FAKE_ERASE, OTHER, SD_INPAINT_NAME, SDXL_INPAINT_NAME, SD_NAME, SDXL_NAME]

_TYPE_BY_NAME = {
    LAMA: ModelType.INPAINT,
    FAKE_ERASE: ModelType.INPAINT,
    OTHER: ModelType.DIFFUSERS_OTHER,
    SD_INPAINT_NAME: ModelType.DIFFUSERS_SD_INPAINT,
    SDXL_INPAINT_NAME: ModelType.DIFFUSERS_SDXL_INPAINT,
    SD_NAME: ModelType.DIFFUSERS_SD,
    SDXL_NAME: ModelType.DIFFUSERS_SDXL,
}


def _info(name: str) -> ModelInfo:
    return ModelInfo(name=name, path=f"/fake/{name}", model_type=_TYPE_BY_NAME[name])


class FakePipe:
    """reload 分支会从旧模型上拆组件再传给 init_model，用开关控制缺哪些。

    lora 接口挂在 pipe 上（见 enable_disable_lcm_lora 里的 self.model.model.*），
    与 BuiltModel.calls 共用一个列表便于统一断言。
    """

    def __init__(self, calls, with_text_encoder_2=False, with_tokenizers=False):
        self.calls = calls
        self.vae = object()
        self.text_encoder = object()
        self.unet = object()
        self.adapters = {}  # 非空 = lora 已加载
        if with_text_encoder_2:
            self.text_encoder_2 = object()
        if with_tokenizers:
            self.tokenizer = object()
            self.tokenizer_2 = object()

    def get_list_adapters(self):
        return self.adapters

    def load_lora_weights(self, *args, **kwargs):
        self.calls.append(("load_lora_weights", (args, kwargs)))
        self.adapters["lcm"] = object()

    def enable_lora(self):
        self.calls.append(("enable_lora", ()))

    def disable_lora(self):
        self.calls.append(("disable_lora", ()))


class BuiltModel:
    """init_model 构造器的替身：记录调用，暴露开关切换接口。

    pipe 的形状模仿真实模型：SD 系没有 text_encoder_2，SDXL 系有。
    """

    def __init__(self, name: str, kind: str, kwargs: dict):
        self.name = name
        self.kind = kind  # 由哪个构造器造出来（ControlNet/SD/...）
        self.kwargs = kwargs
        self.calls = []
        self.lcm_lora_id = "fake/lcm-lora"
        self.fail_switch = False
        is_sdxl = "sdxl" in name.lower()
        self.model = FakePipe(
            self.calls,
            with_text_encoder_2=is_sdxl,
            with_tokenizers=is_sdxl,
        )

    def _record(self, name, *args):
        if self.fail_switch:
            raise RuntimeError(f"{name} boom")
        self.calls.append((name, args))

    def switch_controlnet_method(self, method):
        self._record("switch_controlnet_method", method)

    def switch_brushnet_method(self, method):
        self._record("switch_brushnet_method", method)

    def __call__(self, image, mask, config):
        # 返回 float，验证 _run 里的 .astype(np.uint8)
        return np.full((8, 8, 3), 200.7, np.float32)


@pytest.fixture
def build(monkeypatch):
    state = {"fail_names": set(), "inits": []}  # inits: (kind, kwargs)

    def make_recorder(kind):
        def ctor(device, **kwargs):
            name = kwargs["model_info"].name
            if name in state["fail_names"]:
                raise RuntimeError(f"load {name} failed")
            state["inits"].append((kind, kwargs))
            return BuiltModel(name, kind, kwargs)

        return ctor

    for cls_name, kind in [
        ("ControlNet", "ControlNet"),
        ("SD", "SD"),
        ("SDXL", "SDXL"),
        ("BrushNetWrapper", "BrushNetWrapper"),
        ("BrushNetXLWrapper", "BrushNetXLWrapper"),
        ("PowerPaintV2", "PowerPaintV2"),
    ]:
        monkeypatch.setattr(
            f"iopaint.model_manager.{cls_name}", make_recorder(kind)
        )
    # erase 模型注册表：只有登记过的名字才走 models[name] 分支
    erase_ctor = make_recorder("Erase")
    monkeypatch.setattr(
        "iopaint.model_manager.models",
        {LAMA: erase_ctor, FAKE_ERASE: erase_ctor},
    )

    def fake_scan(self):
        self.available_models = {n: _info(n) for n in ALL_MODELS}
        return list(self.available_models.values())

    monkeypatch.setattr(ModelManager, "scan_models", fake_scan)

    def _build(name=FAKE_ERASE, **kwargs):
        return ModelManager(name=name, device=torch.device("cpu"), **kwargs), state

    return _build


class TestInitModelDispatch:
    def test_unknown_name_raises(self, build):
        mm, state = build()
        before = len(state["inits"])
        with pytest.raises(NotImplementedError) as ei:
            mm.init_model("not-a-model", torch.device("cpu"))
        # 错误信息要列出可用模型，方便排障
        assert FAKE_ERASE in str(ei.value)
        assert len(state["inits"]) == before  # 连装载尝试都不该有

    def test_unsupported_type_raises(self, build):
        """在 available_models 里但类型不被支持（DIFFUSERS_OTHER 且不在 models 注册表）。"""
        mm, state = build()
        before = len(state["inits"])
        with pytest.raises(NotImplementedError):
            mm.init_model(OTHER, torch.device("cpu"))
        assert len(state["inits"]) == before

    def test_erase_model_uses_registry(self, build):
        mm, state = build(name=FAKE_ERASE)
        assert mm.model.kind == "Erase"

    def test_lama_uses_registry(self, build):
        mm, state = build(name=LAMA)
        assert mm.model.kind == "Erase"

    def test_sd_inpaint_goes_to_sd(self, build):
        mm, state = build(name=SD_INPAINT_NAME)
        assert mm.model.kind == "SD"

    def test_sdxl_inpaint_goes_to_sdxl(self, build):
        mm, state = build(name=SDXL_INPAINT_NAME)
        assert mm.model.kind == "SDXL"

    def test_sd_normal_goes_to_sd(self, build):
        mm, state = build(name=SD_NAME)
        assert mm.model.kind == "SD"

    def test_sdxl_normal_goes_to_sdxl(self, build):
        mm, state = build(name=SDXL_NAME)
        assert mm.model.kind == "SDXL"

    def test_controlnet_beats_everything(self, build):
        """controlnet 分支排在 brushnet/powerpaint/models 之前。"""
        mm, state = build(
            name=SD_NAME,
            enable_controlnet=True,
            enable_brushnet=True,
            enable_powerpaint_v2=True,
        )
        assert mm.model.kind == "ControlNet"

    def test_brushnet_sd_uses_wrapper(self, build):
        mm, state = build(name=SD_NAME, enable_brushnet=True)
        assert mm.model.kind == "BrushNetWrapper"

    def test_brushnet_sdxl_uses_xl_wrapper(self, build):
        mm, state = build(name=SDXL_NAME, enable_brushnet=True)
        assert mm.model.kind == "BrushNetXLWrapper"

    def test_powerpaint_v2_beats_registry_and_sd(self, build):
        mm, state = build(name=SD_NAME, enable_powerpaint_v2=True)
        assert mm.model.kind == "PowerPaintV2"

    def test_kwargs_carry_model_info_and_switches(self, build):
        mm, state = build(
            name=SD_NAME,
            enable_controlnet=True,
            controlnet_method=SD_CONTROLNET_CHOICES[0],
        )
        kind, kwargs = state["inits"][-1]
        assert kind == "ControlNet"
        assert kwargs["model_info"].name == SD_NAME
        assert kwargs["enable_controlnet"] is True
        assert kwargs["controlnet_method"] == SD_CONTROLNET_CHOICES[0]
        assert kwargs["enable_brushnet"] is False
        # ModelManager 侧 brushnet_method 构造默认是 None（不是 schema 的默认值）
        assert kwargs["brushnet_method"] is None

    def test_constructor_picks_first_controlnet_when_none(self, build):
        """controlnet_method=None 时，构造期应挑 available_models 里该模型的第一个 controlnet。"""
        mm, state = build(name=SD_NAME, enable_controlnet=True, controlnet_method=None)
        assert mm.controlnet_method == SD_CONTROLNET_CHOICES[0]

    def test_constructor_keeps_explicit_controlnet_method(self, build):
        explicit = SD_CONTROLNET_CHOICES[1]
        mm, state = build(
            name=SD_NAME, enable_controlnet=True, controlnet_method=explicit
        )
        assert mm.controlnet_method == explicit


class TestLcmLora:
    def test_non_lcm_model_is_noop(self, build):
        mm, state = build(name=LAMA)
        mm.enable_disable_lcm_lora(InpaintRequest(sd_lcm_lora=True))
        assert mm.model.calls == []

    def test_first_enable_loads_lora(self, build):
        from iopaint.model.utils import is_local_files_only

        mm, state = build(name=SD_INPAINT_NAME)
        mm.enable_disable_lcm_lora(InpaintRequest(sd_lcm_lora=True))
        assert mm.model.calls == [
            (
                "load_lora_weights",
                (
                    ("fake/lcm-lora",),
                    {
                        "weight_name": "pytorch_lora_weights.safetensors",
                        "local_files_only": is_local_files_only(),
                    },
                ),
            )
        ]

    def test_second_enable_uses_enable_lora(self, build):
        mm, state = build(name=SD_INPAINT_NAME)
        mm.model.model.adapters["lcm"] = object()
        mm.enable_disable_lcm_lora(InpaintRequest(sd_lcm_lora=True))
        assert mm.model.calls == [("enable_lora", ())]

    def test_disable_with_adapters(self, build):
        mm, state = build(name=SD_INPAINT_NAME)
        mm.model.model.adapters["lcm"] = object()
        mm.enable_disable_lcm_lora(InpaintRequest(sd_lcm_lora=False))
        assert mm.model.calls == [("disable_lora", ())]

    def test_disable_without_adapters_is_noop(self, build):
        mm, state = build(name=SD_INPAINT_NAME)
        mm.enable_disable_lcm_lora(InpaintRequest(sd_lcm_lora=False))
        assert mm.model.calls == []


class TestRunSwitches:
    def test_run_triggers_controlnet_method_switch(self, build):
        mm, state = build(
            name=SD_NAME,
            enable_controlnet=True,
            controlnet_method=SD_CONTROLNET_CHOICES[0],
        )
        new_method = SD_CONTROLNET_CHOICES[1]
        out = mm(
            np.zeros((8, 8, 3), np.uint8),
            np.zeros((8, 8), np.uint8),
            InpaintRequest(enable_controlnet=True, controlnet_method=new_method),
        )
        assert mm.controlnet_method == new_method
        assert mm.model.calls == [("switch_controlnet_method", (new_method,))]
        # _run 必须把 float 结果转成 uint8
        assert out.dtype == np.uint8

    def test_run_triggers_brushnet_method_switch(self, build):
        mm, state = build(name=SD_NAME, enable_brushnet=True)
        new_method = SD_BRUSHNET_CHOICES[1]
        mm(
            np.zeros((8, 8, 3), np.uint8),
            np.zeros((8, 8), np.uint8),
            InpaintRequest(enable_brushnet=True, brushnet_method=new_method),
        )
        assert mm.brushnet_method == new_method
        assert mm.model.calls == [("switch_brushnet_method", (new_method,))]

    def test_run_without_switches_does_not_touch_methods(self, build):
        mm, state = build(name=SD_NAME)
        mm(
            np.zeros((8, 8, 3), np.uint8),
            np.zeros((8, 8), np.uint8),
            InpaintRequest(),
        )
        assert mm.model.calls == []


class TestBrushnetToggle:
    def test_method_switch_success_commits(self, build):
        mm, state = build(name=SD_NAME, enable_brushnet=True)
        new_method = SD_BRUSHNET_CHOICES[1]
        mm.switch_brushnet_method(
            InpaintRequest(enable_brushnet=True, brushnet_method=new_method)
        )
        assert mm.brushnet_method == new_method
        assert mm.model.calls == [("switch_brushnet_method", (new_method,))]

    def test_method_switch_failure_keeps_old_method(self, build):
        mm, state = build(
            name=SD_NAME,
            enable_brushnet=True,
            brushnet_method=SD_BRUSHNET_CHOICES[0],
        )
        mm.model.fail_switch = True
        with pytest.raises(RuntimeError):
            mm.switch_brushnet_method(
                InpaintRequest(
                    enable_brushnet=True, brushnet_method=SD_BRUSHNET_CHOICES[1]
                )
            )
        assert mm.brushnet_method == SD_BRUSHNET_CHOICES[0]

    def test_non_brushnet_model_is_noop(self, build):
        mm, state = build(name=SD_INPAINT_NAME)  # SD inpaint 不支持 brushnet
        before = list(state["inits"])
        mm.switch_brushnet_method(InpaintRequest(enable_brushnet=True))
        assert state["inits"] == before
        assert mm.enable_brushnet is False

    def test_enable_toggle_reloads_with_pipe_components(self, build):
        """enable→disable 走 reload 分支：旧模型的组件要传给 init_model 复用。

        注意 method 必须与 config 相同，否则会先走 method-switch 分支而非 toggle。
        """
        mm, state = build(
            name=SDXL_NAME,
            enable_brushnet=True,
            brushnet_method=SD_BRUSHNET_CHOICES[0],
        )
        before = len(state["inits"])
        old_pipe = mm.model.model

        mm.switch_brushnet_method(InpaintRequest(enable_brushnet=False))

        assert mm.enable_brushnet is False
        assert len(state["inits"]) == before + 1
        kind, kwargs = state["inits"][-1]
        # 状态在 reload 前已提交为 disable，所以重建出来的是普通 SDXL（不带 BrushNet）
        assert kind == "SDXL"
        components = kwargs["pipe_components"]
        # SDXL 有 text_encoder_2/tokenizer/tokenizer_2，都必须被复用
        assert set(components) == {
            "vae",
            "text_encoder",
            "unet",
            "text_encoder_2",
            "tokenizer",
            "tokenizer_2",
        }
        # 组件必须就是切换前那个模型身上的（复用而不是重建）
        assert components["unet"] is old_pipe.unet
        assert components["text_encoder_2"] is old_pipe.text_encoder_2

    def test_enable_toggle_builds_brushnet_model(self, build):
        """disable→enable 方向：重建出来的必须是 BrushNet 变体。"""
        mm, state = build(
            name=SD_NAME,
            enable_brushnet=False,
            brushnet_method=SD_BRUSHNET_CHOICES[0],
        )
        assert mm.model.kind == "SD"
        before = len(state["inits"])

        mm.switch_brushnet_method(
            InpaintRequest(
                enable_brushnet=True, brushnet_method=SD_BRUSHNET_CHOICES[0]
            )
        )

        assert mm.enable_brushnet is True
        assert state["inits"][-1][0] == "BrushNetWrapper"

    def test_enable_toggle_failure_restores_state(self, build):
        mm, state = build(
            name=SD_NAME,
            enable_brushnet=True,
            brushnet_method=SD_BRUSHNET_CHOICES[0],
        )
        state["fail_names"].add(SD_NAME)

        with pytest.raises(RuntimeError):
            mm.switch_brushnet_method(InpaintRequest(enable_brushnet=False))

        assert mm.enable_brushnet is True
        assert mm.brushnet_method == SD_BRUSHNET_CHOICES[0]
        assert mm.model.name == SD_NAME  # 旧模型还在


class TestControlnetToggle:
    def test_non_controlnet_model_is_noop(self, build):
        """erase 模型不支持 controlnet：直接早退，不该触发任何 reload。"""
        mm, state = build(name=FAKE_ERASE)
        before = list(state["inits"])
        mm.switch_controlnet_method(InpaintRequest(enable_controlnet=True))
        assert state["inits"] == before
        assert mm.enable_controlnet is False

    def test_enable_toggle_success_commits(self, build):
        mm, state = build(
            name=SD_NAME,
            enable_controlnet=True,
            controlnet_method=SD_CONTROLNET_CHOICES[0],
        )
        before = len(state["inits"])

        mm.switch_controlnet_method(InpaintRequest(enable_controlnet=False))

        assert mm.enable_controlnet is False
        assert len(state["inits"]) == before + 1
        kind, kwargs = state["inits"][-1]
        # 状态在 reload 前已提交为 disable，所以重建出来的是普通 SD（不带 ControlNet）
        assert kind == "SD"
        # SD 没有 text_encoder_2，组件集合就是三个基础件
        assert set(kwargs["pipe_components"]) == {"vae", "text_encoder", "unet"}

    def test_enable_toggle_builds_controlnet_model(self, build):
        """disable→enable 方向：重建出来的必须是 ControlNet 变体。"""
        mm, state = build(
            name=SD_NAME,
            enable_controlnet=False,
            controlnet_method=SD_CONTROLNET_CHOICES[0],
        )
        assert mm.model.kind == "SD"
        before = len(state["inits"])

        mm.switch_controlnet_method(
            InpaintRequest(
                enable_controlnet=True, controlnet_method=SD_CONTROLNET_CHOICES[0]
            )
        )

        assert mm.enable_controlnet is True
        assert state["inits"][-1][0] == "ControlNet"

    def test_enable_toggle_failure_restores_state(self, build):
        mm, state = build(
            name=SD_NAME,
            enable_controlnet=True,
            controlnet_method=SD_CONTROLNET_CHOICES[0],
        )
        state["fail_names"].add(SD_NAME)

        with pytest.raises(RuntimeError):
            mm.switch_controlnet_method(InpaintRequest(enable_controlnet=False))

        assert mm.enable_controlnet is True
        assert mm.controlnet_method == SD_CONTROLNET_CHOICES[0]
        assert mm.model.name == SD_NAME


class TestPowerPaintV2Toggle:
    def test_disable_success_commits(self, build):
        mm, state = build(name=SD_NAME, enable_powerpaint_v2=True)
        assert mm.model.kind == "PowerPaintV2"
        before = len(state["inits"])

        mm.enable_disable_powerpaint_v2(InpaintRequest(enable_powerpaint_v2=False))

        assert mm.enable_powerpaint_v2 is False
        assert len(state["inits"]) == before + 1
        assert state["inits"][-1][0] == "SD"  # 关掉后回到普通 SD

    def test_unsupported_model_is_noop(self, build):
        mm, state = build(name=LAMA)
        before = list(state["inits"])
        mm.enable_disable_powerpaint_v2(InpaintRequest(enable_powerpaint_v2=True))
        assert state["inits"] == before
        assert mm.enable_powerpaint_v2 is False

    def test_failure_leaves_model_none_then_503(self, build):
        """reload 失败且回滚也失败时 self.model 必须为 None，调用得到 503。"""
        from fastapi import HTTPException

        mm, state = build(name=SD_NAME, enable_powerpaint_v2=True)
        state["fail_names"].add(SD_NAME)

        with pytest.raises(RuntimeError):
            mm.enable_disable_powerpaint_v2(InpaintRequest(enable_powerpaint_v2=False))

        assert mm.enable_powerpaint_v2 is True  # 状态回滚
        assert mm.model is None

        with pytest.raises(HTTPException) as ei:
            mm(
                np.zeros((8, 8, 3), np.uint8),
                np.zeros((8, 8), np.uint8),
                InpaintRequest(),
            )
        assert ei.value.status_code == 503
