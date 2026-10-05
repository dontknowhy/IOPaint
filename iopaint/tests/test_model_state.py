"""D-9 模型切换状态机（REL-2）。

不加载任何权重：scan_models 与 init_model 全部打桩，只验证「状态提交 / 回滚」的顺序，
以及失败后不会退化成 AttributeError 级联。
"""
import numpy as np
import pytest
import torch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from iopaint.const import SD_CONTROLNET_CHOICES, SDXL_CONTROLNET_CHOICES
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest, ModelInfo, ModelType

LAMA = "lama"
CV2 = "cv2"
SD_NAME = "runwayml/stable-diffusion-inpainting"
SDXL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
ALL_MODELS = [LAMA, CV2, SD_NAME, SDXL_NAME]


def _info(name: str) -> ModelInfo:
    if name == SDXL_NAME:
        model_type = ModelType.DIFFUSERS_SDXL
    elif name == SD_NAME:
        model_type = ModelType.DIFFUSERS_SD_INPAINT
    else:
        model_type = ModelType.INPAINT
    return ModelInfo(name=name, path=f"/fake/{name}", model_type=model_type)


class FakePipe:
    """reload 分支会从旧模型上拆 vae/text_encoder/unet 再传给 init_model。"""

    def __init__(self):
        self.vae = object()
        self.text_encoder = object()
        self.unet = object()
        self.text_encoder_2 = object()


class FakeModel:
    def __init__(self, name: str, built_with=None):
        self.name = name
        # init_model 构造时读到的 self.controlnet_method，用来证明新方法确实参与了装载
        self.built_with = built_with
        self.fail_switch = False
        self.switched_methods = []
        self.model = FakePipe()

    def switch_controlnet_method(self, new_method: str):
        if self.fail_switch:
            raise RuntimeError("model.switch_controlnet_method boom")
        self.switched_methods.append(new_method)

    def __call__(self, image, mask, config):
        raise AssertionError("状态机测试不应该走到真正的推理")


@pytest.fixture
def build(monkeypatch):
    state = {"fail_names": set(), "init_calls": [], "built_with": []}

    def fake_scan(self):
        self.available_models = {n: _info(n) for n in ALL_MODELS}
        return list(self.available_models.values())

    def fake_init(self, name, device, **kwargs):
        state["init_calls"].append(name)
        # 镜像真实 init_model 的取值优先级：kwargs 里的值会被 self.* 覆盖
        # （见 model_manager.init_model 中 `{**kwargs, ..., "controlnet_method": ...}`），
        # 所以这里记录的就是真实装载时生效的方法。
        state["built_with"].append(self.controlnet_method)
        if name in state["fail_names"]:
            raise RuntimeError(f"load {name} failed")
        return FakeModel(name, built_with=self.controlnet_method)

    monkeypatch.setattr(ModelManager, "scan_models", fake_scan)
    monkeypatch.setattr(ModelManager, "init_model", fake_init)

    def _build(name=LAMA, **kwargs):
        return ModelManager(name=name, device=torch.device("cpu"), **kwargs), state

    return _build


class TestSwitchValidation:
    def test_unknown_model_is_422_and_state_untouched(self, build):
        """此前 available_models[new_name] 抛 KeyError → 500，且 self.name 已被改掉。"""
        mm, state = build(name=LAMA)
        before = list(state["init_calls"])

        with pytest.raises(HTTPException) as ei:
            mm.switch("not-a-model")

        assert ei.value.status_code == 422
        assert mm.name == LAMA
        assert mm.model.name == LAMA
        assert state["init_calls"] == before  # 连装载尝试都不该有

    def test_switch_success_commits_name(self, build):
        mm, state = build(name=LAMA)
        mm.switch(CV2)
        assert mm.name == CV2
        assert mm.model.name == CV2
        assert state["init_calls"] == [LAMA, CV2]

    def test_switch_same_name_is_noop(self, build):
        mm, state = build(name=LAMA)
        mm.switch(LAMA)
        assert state["init_calls"] == [LAMA]  # 只有构造时那一次


class TestSwitchFailureRollback:
    def test_failure_keeps_old_name_and_model(self, build):
        mm, state = build(name=LAMA)
        state["fail_names"].add(CV2)

        with pytest.raises(RuntimeError):
            mm.switch(CV2)

        assert mm.name == LAMA
        assert isinstance(mm.model, FakeModel)
        assert mm.model.name == LAMA  # 回滚时重建了旧模型

    def test_rollback_failure_gives_503_not_attribute_error(self, build):
        """原实现是 `del self.model`：回滚再失败就只剩 AttributeError 级联。"""
        mm, state = build(name=LAMA)
        state["fail_names"].update({CV2, LAMA})

        with pytest.raises(RuntimeError):
            mm.switch(CV2)

        assert mm.name == LAMA
        assert mm.model is None

        with pytest.raises(HTTPException) as ei:
            mm(
                np.zeros((8, 8, 3), np.uint8),
                np.zeros((8, 8), np.uint8),
                InpaintRequest(),
            )
        assert ei.value.status_code == 503

    def test_lock_released_after_failed_switch(self, build):
        """异常路径不能把 lock 永久占住（否则后续请求全部挂死）。"""
        mm, state = build(name=LAMA)
        state["fail_names"].add(CV2)
        with pytest.raises(RuntimeError):
            mm.switch(CV2)

        state["fail_names"].clear()
        mm.switch(CV2)  # 还拿得到锁 → 能正常切换，而不是死锁
        assert mm.name == CV2
        assert mm.model.name == CV2


class TestControlnetState:
    def test_switch_picks_target_model_controlnet_method(self, build):
        """init_model 内部读 self.controlnet_method，
        新方法必须在装载新模型之前就位，否则 controlnet 权重会按旧方法装。"""
        mm, state = build(
            name=SD_NAME,
            enable_controlnet=True,
            controlnet_method=SD_CONTROLNET_CHOICES[0],
        )
        mm.switch(SDXL_NAME)

        assert mm.controlnet_method == SDXL_CONTROLNET_CHOICES[0]
        assert state["built_with"][-1] == SDXL_CONTROLNET_CHOICES[0]
        assert mm.model.built_with == SDXL_CONTROLNET_CHOICES[0]
        # kwargs 里仍是启动时的旧值 —— 生效的是 self.controlnet_method
        assert mm.kwargs["controlnet_method"] == SD_CONTROLNET_CHOICES[0]

    def test_switch_failure_rolls_back_controlnet_method(self, build):
        old_method = SD_CONTROLNET_CHOICES[0]
        mm, state = build(
            name=SD_NAME, enable_controlnet=True, controlnet_method=old_method
        )
        state["fail_names"].add(SDXL_NAME)

        with pytest.raises(RuntimeError):
            mm.switch(SDXL_NAME)

        assert mm.name == SD_NAME
        assert mm.controlnet_method == old_method
        assert mm.model.name == SD_NAME

    def test_controlnet_method_commits_after_model_success(self, build):
        old_method = SD_CONTROLNET_CHOICES[0]
        new_method = SD_CONTROLNET_CHOICES[1]
        mm, state = build(
            name=SD_NAME, enable_controlnet=True, controlnet_method=old_method
        )

        mm.model.fail_switch = True
        with pytest.raises(RuntimeError):
            mm.switch_controlnet_method(
                InpaintRequest(enable_controlnet=True, controlnet_method=new_method)
            )
        assert mm.controlnet_method == old_method  # 模型没切成功，状态不能先改

        mm.model.fail_switch = False
        mm.switch_controlnet_method(
            InpaintRequest(enable_controlnet=True, controlnet_method=new_method)
        )
        assert mm.controlnet_method == new_method
        assert mm.model.switched_methods == [new_method]

    def test_controlnet_enable_toggle_failure_restores_state(self, build):
        """enable/disable 走 reload 分支：init_model 失败时旧模型还在，
        只需把 enable/controlnet_method 改回去。"""
        mm, state = build(
            name=SD_NAME,
            enable_controlnet=True,
            controlnet_method=SD_CONTROLNET_CHOICES[0],
        )
        # 让下一次 init_model（reload）失败
        state["fail_names"].add(SD_NAME)

        with pytest.raises(RuntimeError):
            mm.switch_controlnet_method(InpaintRequest(enable_controlnet=False))

        assert mm.enable_controlnet is True
        assert mm.controlnet_method == SD_CONTROLNET_CHOICES[0]
        assert mm.model.name == SD_NAME


class TestSwitchRoute:
    """HTTP 层：未知模型名必须是 422，而不是 KeyError → 500。"""

    def test_post_unknown_model_returns_422(self, tmp_path):
        from iopaint.api import Api
        from iopaint.tests.test_api_phase1 import _minimal_config

        app = FastAPI()
        Api(app, _minimal_config(tmp_path))
        client = TestClient(app)

        before = client.get("/api/v1/model")
        assert before.status_code == 200
        current_name = before.json()["name"]

        resp = client.post(
            "/api/v1/model",
            content='{"name": "not-a-model"}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        assert "not-a-model" in resp.json()["detail"]

        after = client.get("/api/v1/model")
        assert after.status_code == 200
        assert after.json()["name"] == current_name  # 状态没被污染
