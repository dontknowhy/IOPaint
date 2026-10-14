"""plugins/__init__.py:build_plugins 的分支补测。

插件类本身（下载权重、建网络）不在 fast 测试范围：这里把六个插件类打桩成
记录构造参数的假类，只验证「开关 → 建哪个插件、传什么参数、upscaler 怎么挂」。
"""
import pytest

from iopaint import plugins as plugins_mod
from iopaint.plugins import build_plugins
from iopaint.schema import Device, InteractiveSegModel, RealESRGANModel, RemoveBGModel


def _install_fake(monkeypatch, attr: str, name: str, store: dict):
    """把 iopaint.plugins.<attr> 换成假类：记录实例（含构造参数），类名作 .name。"""

    class Fake:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            store.append(self)

    Fake.name = name
    monkeypatch.setattr(plugins_mod, attr, Fake)
    return Fake


def _kwargs(**overrides):
    params = dict(
        enable_interactive_seg=False,
        interactive_seg_model=InteractiveSegModel.sam2_1_tiny,
        interactive_seg_device=Device.cpu,
        enable_remove_bg=False,
        remove_bg_device=Device.cpu,
        remove_bg_model=RemoveBGModel.briaai_rmbg_1_4,
        enable_anime_seg=False,
        enable_realesrgan=False,
        realesrgan_device=Device.cpu,
        realesrgan_model=RealESRGANModel.realesr_general_x4v3,
        enable_gfpgan=False,
        gfpgan_device=Device.cpu,
        enable_restoreformer=False,
        restoreformer_device=Device.cpu,
        no_half=False,
    )
    params.update(overrides)
    return params


class TestBuildPluginsDisabled:
    def test_all_disabled_returns_empty(self):
        assert build_plugins(**_kwargs()) == {}


class TestBuildPluginsIndividual:
    def test_interactive_seg(self, monkeypatch):
        created = []
        _install_fake(monkeypatch, "InteractiveSeg", "InteractiveSeg", created)
        res = build_plugins(
            **_kwargs(
                enable_interactive_seg=True,
                interactive_seg_model=InteractiveSegModel.sam2_1_small,
                interactive_seg_device=Device.cuda,
            )
        )
        assert list(res.keys()) == ["InteractiveSeg"]
        assert created[0].args == (InteractiveSegModel.sam2_1_small, Device.cuda)

    def test_remove_bg(self, monkeypatch):
        created = []
        _install_fake(monkeypatch, "RemoveBG", "RemoveBG", created)
        res = build_plugins(
            **_kwargs(
                enable_remove_bg=True,
                remove_bg_model=RemoveBGModel.briaai_rmbg_1_4,
                remove_bg_device=Device.cuda,
            )
        )
        assert list(res.keys()) == ["RemoveBG"]
        assert created[0].args == (RemoveBGModel.briaai_rmbg_1_4, Device.cuda)

    def test_anime_seg(self, monkeypatch):
        created = []
        _install_fake(monkeypatch, "AnimeSeg", "AnimeSeg", created)
        res = build_plugins(**_kwargs(enable_anime_seg=True))
        assert list(res.keys()) == ["AnimeSeg"]
        assert created[0].args == ()

    def test_realesrgan_forwards_no_half(self, monkeypatch):
        created = []
        _install_fake(monkeypatch, "RealESRGANUpscaler", "RealESRGAN", created)
        res = build_plugins(
            **_kwargs(
                enable_realesrgan=True,
                realesrgan_model=RealESRGANModel.realesr_general_x4v3,
                realesrgan_device=Device.cuda,
                no_half=True,
            )
        )
        assert list(res.keys()) == ["RealESRGAN"]
        assert created[0].args == (
            RealESRGANModel.realesr_general_x4v3,
            Device.cuda,
        )
        assert created[0].kwargs == {"no_half": True}


class TestBuildPluginsUpscalerWiring:
    """GFPGAN / RestoreFormer 的 upscaler 跟随 --enable-realesrgan。"""

    def test_gfpgan_without_realesrgan_has_no_upscaler(self, monkeypatch):
        created = {}
        _install_fake(monkeypatch, "GFPGANPlugin", "GFPGAN", created.setdefault("gfpgan", []))
        res = build_plugins(**_kwargs(enable_gfpgan=True))
        assert res["GFPGAN"].kwargs["upscaler"] is None

    def test_gfpgan_with_realesrgan_reuses_instance(self, monkeypatch):
        created = {}
        _install_fake(monkeypatch, "RealESRGANUpscaler", "RealESRGAN", created.setdefault("re", []))
        _install_fake(monkeypatch, "GFPGANPlugin", "GFPGAN", created.setdefault("gfpgan", []))
        res = build_plugins(**_kwargs(enable_gfpgan=True, enable_realesrgan=True))
        assert res["GFPGAN"].kwargs["upscaler"] is res["RealESRGAN"]

    def test_restoreformer_without_realesrgan_has_no_upscaler(self, monkeypatch):
        created = {}
        _install_fake(monkeypatch, "RestoreFormerPlugin", "RestoreFormer", created.setdefault("rf", []))
        res = build_plugins(**_kwargs(enable_restoreformer=True))
        assert res["RestoreFormer"].kwargs["upscaler"] is None

    def test_restoreformer_with_realesrgan_reuses_instance(self, monkeypatch):
        created = {}
        _install_fake(monkeypatch, "RealESRGANUpscaler", "RealESRGAN", created.setdefault("re", []))
        _install_fake(monkeypatch, "RestoreFormerPlugin", "RestoreFormer", created.setdefault("rf", []))
        res = build_plugins(**_kwargs(enable_restoreformer=True, enable_realesrgan=True))
        assert res["RestoreFormer"].kwargs["upscaler"] is res["RealESRGAN"]


class TestBuildPluginsAll:
    def test_all_enabled(self, monkeypatch):
        created = {}
        for attr, name, key in [
            ("InteractiveSeg", "InteractiveSeg", "seg"),
            ("RemoveBG", "RemoveBG", "rmbg"),
            ("AnimeSeg", "AnimeSeg", "anime"),
            ("RealESRGANUpscaler", "RealESRGAN", "re"),
            ("GFPGANPlugin", "GFPGAN", "gfpgan"),
            ("RestoreFormerPlugin", "RestoreFormer", "rf"),
        ]:
            _install_fake(monkeypatch, attr, name, created.setdefault(key, []))
        res = build_plugins(
            **_kwargs(
                enable_interactive_seg=True,
                enable_remove_bg=True,
                enable_anime_seg=True,
                enable_realesrgan=True,
                enable_gfpgan=True,
                enable_restoreformer=True,
            )
        )
        assert sorted(res.keys()) == sorted(
            ["InteractiveSeg", "RemoveBG", "AnimeSeg", "RealESRGAN", "GFPGAN", "RestoreFormer"]
        )
        # GFPGAN 与 RestoreFormer 共享同一个 RealESRGAN 实例
        assert res["GFPGAN"].kwargs["upscaler"] is res["RealESRGAN"]
        assert res["RestoreFormer"].kwargs["upscaler"] is res["RealESRGAN"]
