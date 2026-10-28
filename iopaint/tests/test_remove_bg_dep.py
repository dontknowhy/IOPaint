"""RemoveBG 的依赖检查与 rembg 版本比较（不下载权重）。

briaai/RMBG-1.4 与 briaai/RMBG-2.0 走 torch + huggingface_hub 自己的 session，
不需要 rembg；其余模型仍走 rembg 的 session。版本必须按数值比较，
否则 "2.0.100" 会被字符串比较误判为旧于 "2.0.59"。
"""
import sys
import types
from pathlib import Path

import pytest

import iopaint.plugins.remove_bg as remove_bg
from iopaint.plugins.remove_bg import RemoveBG, _version_tuple
from iopaint.schema import Device, RemoveBGModel


def _without_rembg(monkeypatch: pytest.MonkeyPatch) -> None:
    # sys.modules 里放 None：`import rembg` 抛 ImportError，模拟只装了 requirements.txt
    monkeypatch.setitem(sys.modules, "rembg", None)


@pytest.mark.parametrize(
    "model_name",
    [RemoveBGModel.briaai_rmbg_1_4, RemoveBGModel.briaai_rmbg_2_0],
)
def test_briaai_models_do_not_require_rembg(
    monkeypatch: pytest.MonkeyPatch, model_name: RemoveBGModel
) -> None:
    _without_rembg(monkeypatch)
    plugin = RemoveBG.__new__(RemoveBG)
    plugin.model_name = model_name
    assert plugin.check_dep() is None


def test_rembg_backed_models_still_require_rembg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_rembg(monkeypatch)
    plugin = RemoveBG.__new__(RemoveBG)
    plugin.model_name = RemoveBGModel.u2net
    err = plugin.check_dep()
    assert err is not None
    assert "Install rembg failed" in err


def test_init_sets_model_name_before_check_dep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """check_dep 需要 self.model_name：赋值必须早于 BasePlugin.__init__。"""
    _without_rembg(monkeypatch)
    monkeypatch.setenv("U2NET_HOME", "")
    monkeypatch.setattr(remove_bg, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(RemoveBG, "_init_session", lambda self, name: None)

    plugin = RemoveBG(RemoveBGModel.briaai_rmbg_1_4, Device.cpu)

    assert plugin.model_name == RemoveBGModel.briaai_rmbg_1_4
    assert plugin.device == Device.cpu


def test_newer_rembg_version_is_not_mistaken_for_older(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """"2.0.100" 必须被当作 >= 2.0.59（字符串比较会误判成旧版并拒绝启动）。"""
    fake_rembg = types.ModuleType("rembg")
    setattr(fake_rembg, "__version__", "2.0.100")
    monkeypatch.setitem(sys.modules, "rembg", fake_rembg)
    monkeypatch.setenv("U2NET_HOME", "")
    monkeypatch.setattr(remove_bg, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(RemoveBG, "_init_session", lambda self, name: None)

    plugin = RemoveBG(RemoveBGModel.birefnet_general, Device.cpu)

    assert plugin.model_name == RemoveBGModel.birefnet_general


def test_version_tuple_is_numeric() -> None:
    assert _version_tuple("2.0.100") > _version_tuple("2.0.59")
    assert _version_tuple("2.0.59.post1") == (2, 0, 59)
    assert _version_tuple("1.4") == (1, 4)
