"""cli.py 缺口补测：`_resolve_cli_device`、download/list/run 分发、start 参数校验与生命周期分支。

沿用 test_cli.py 的 CliRunner 进程内调用（不绑端口）。打桩边界：只桩外部依赖
（download.scan_models / cli_download_model、batch_inpaint、installer、Api.launch、
hub_is_reachable、webbrowser），被测对象（`_resolve_cli_device` 与各命令本体）保持真实执行。
"""
import asyncio
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from loguru import logger
from typer.testing import CliRunner

from iopaint import api as api_mod
from iopaint.cli import _resolve_cli_device, typer_app
from iopaint.exceptions import ModelLoadError
from iopaint.schema import Device

runner = CliRunner()

# start/run 的回调会把这些写进环境变量（local_files_only、setup_model_dir），
# 快照-恢复，避免污染同进程内的其它测试。
_ENV_KEYS = ["TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "U2NET_HOME", "XDG_CACHE_HOME"]


@pytest.fixture(autouse=True)
def _restore_env():
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@contextmanager
def captured_logs(level: str = "WARNING"):
    """loguru sink 收集日志消息文本，测试内断言告警/错误内容。"""
    messages: list = []
    sink = logger.add(messages.append, level=level, format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink)


def _patch_hub_ok(monkeypatch):
    """start 会探测 HuggingFace 可达性（真网络、最长 3s）；测试统一跳过探测。"""
    monkeypatch.setattr("iopaint.cli.hub_is_reachable", lambda: True)


class TestResolveCliDevice:
    def test_invalid_device_exits_1(self, monkeypatch):
        with captured_logs("ERROR") as messages:
            with pytest.raises(SystemExit) as exc:
                _resolve_cli_device("bogus-device", None)
        assert exc.value.code == 1
        assert any("Invalid --device" in m for m in messages)

    def test_gpu_id_logs_and_returns_gpu_id(self, monkeypatch):
        # resolve_gpu_id 只在真 cuda 且序号合法时返回 id；这里打桩让它确定性返回，
        # 覆盖 `Using GPU` 日志分支（本机无卡/CI 无卡时也稳定）。
        monkeypatch.setattr("iopaint.cli.resolve_gpu_id", lambda device, gpu_id: 3)
        with captured_logs("INFO") as messages:
            device, gpu_id = _resolve_cli_device("cuda:3", None)
        assert gpu_id == 3
        assert device in (Device.cuda, Device.cpu)  # check_device 视机器而定
        assert any("Using GPU 3" in m for m in messages)


class TestInstallPluginsPackages:
    def test_delegates_to_installer(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "iopaint.installer.install_plugins_package", lambda: calls.append(True)
        )
        result = runner.invoke(typer_app, ["install-plugins-packages"])
        assert result.exit_code == 0
        assert calls == [True]


class TestDownloadCommand:
    def test_delegates_to_cli_download_model(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "iopaint.download.cli_download_model", lambda model: calls.append(model)
        )
        result = runner.invoke(typer_app, ["download", "--model", "org/model"])
        assert result.exit_code == 0
        assert calls == ["org/model"]

    def test_model_load_error_exits_1(self, monkeypatch):
        def boom(model):
            raise ModelLoadError("no space left")

        monkeypatch.setattr("iopaint.download.cli_download_model", boom)
        with captured_logs("ERROR") as messages:
            result = runner.invoke(typer_app, ["download", "--model", "org/model"])
        assert result.exit_code == 1
        assert any("Failed to download model org/model" in m for m in messages)


class TestListCommand:
    def test_prints_scanned_model_names(self, monkeypatch):
        monkeypatch.setattr(
            "iopaint.download.scan_models",
            lambda: [SimpleNamespace(name="m1"), SimpleNamespace(name="m2")],
        )
        result = runner.invoke(typer_app, ["list"])
        assert result.exit_code == 0
        assert "m1" in result.output
        assert "m2" in result.output


class TestRunCommand:
    def test_calls_batch_inpaint_with_resolved_device(self, tmp_path, monkeypatch):
        image = tmp_path / "image.png"
        mask = tmp_path / "mask.png"
        output = tmp_path / "out"
        for p in (image, mask):
            p.write_bytes(b"")
        calls = []
        monkeypatch.setattr(
            "iopaint.download.scan_models",
            lambda: [SimpleNamespace(name="lama")],
        )
        monkeypatch.setattr(
            "iopaint.batch_processing.batch_inpaint",
            lambda *args: calls.append(args),
        )
        result = runner.invoke(
            typer_app,
            [
                "run",
                "--model", "lama",
                "--device", "cpu",
                "--image", str(image),
                "--mask", str(mask),
                "--output", str(output),
                "--concat",
            ],
        )
        assert result.exit_code == 0
        assert len(calls) == 1
        model, device, img, msk, out, config, concat = calls[0]
        assert model == "lama"
        assert device == torch.device("cpu")
        assert img == image
        assert msk == mask
        assert out == output
        assert config is None
        assert concat is True

    def test_download_failure_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("iopaint.download.scan_models", lambda: [])

        def boom(model):
            raise ModelLoadError("gone")

        monkeypatch.setattr("iopaint.download.cli_download_model", boom)
        result = runner.invoke(
            typer_app,
            [
                "run",
                "--device", "cpu",
                "--image", str(tmp_path / "i.png"),
                "--mask", str(tmp_path / "m.png"),
                "--output", str(tmp_path / "o"),
            ],
        )
        assert result.exit_code == 1

    def test_missing_model_downloaded_then_batch_inpaint(self, tmp_path, monkeypatch):
        downloaded = []
        batched = []
        monkeypatch.setattr("iopaint.download.scan_models", lambda: [])
        monkeypatch.setattr(
            "iopaint.download.cli_download_model", lambda m: downloaded.append(m)
        )
        monkeypatch.setattr(
            "iopaint.batch_processing.batch_inpaint",
            lambda *args: batched.append(args),
        )
        result = runner.invoke(
            typer_app,
            [
                "run",
                "--model", "org/new-model",
                "--device", "cpu",
                "--image", str(tmp_path / "i.png"),
                "--mask", str(tmp_path / "m.png"),
                "--output", str(tmp_path / "o"),
            ],
        )
        assert result.exit_code == 0
        assert downloaded == ["org/new-model"]
        assert len(batched) == 1
        assert batched[0][0] == "org/new-model"


class TestStartValidation:
    def test_mask_dir_not_exists_exits_1(self, tmp_path):
        with captured_logs("ERROR") as messages:
            result = runner.invoke(
                typer_app,
                ["start", "--model", "cv2", "--device", "cpu",
                 "--mask-dir", str(tmp_path / "nonexistent")],
            )
        assert result.exit_code == 1
        assert any("invalid --mask-dir" in m for m in messages)

    def test_input_dir_without_output_dir_exits_1(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        with captured_logs("ERROR") as messages:
            result = runner.invoke(
                typer_app,
                ["start", "--model", "cv2", "--device", "cpu",
                 "--input", str(img_dir)],
            )
        assert result.exit_code == 1
        assert any("--output-dir must be set" in m for m in messages)


class TestStartFullLaunch:
    """走到 Api 构造 + launch 的完整 start 路径（launch 打桩，不绑端口、不起服务）。"""

    def test_force_fp16_with_no_half_warns(self, tmp_path, monkeypatch):
        _patch_hub_ok(monkeypatch)
        monkeypatch.setattr(
            "iopaint.download.scan_models",
            lambda: [SimpleNamespace(name="cv2")],
        )
        with captured_logs("WARNING") as messages:
            with patch("iopaint.api.Api.launch", autospec=True) as mock_launch:
                result = runner.invoke(
                    typer_app,
                    ["start", "--model", "cv2", "--device", "cpu",
                     "--force-fp16", "--no-half"],
                )
        assert result.exit_code == 0
        mock_launch.assert_called_once()
        assert any("--force-fp16 is ignored" in m for m in messages)

    def test_mask_dir_is_absolute_in_config(self, tmp_path, monkeypatch):
        _patch_hub_ok(monkeypatch)
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        monkeypatch.setattr(
            "iopaint.download.scan_models",
            lambda: [SimpleNamespace(name="cv2")],
        )
        with patch("iopaint.api.Api.launch", autospec=True) as mock_launch:
            result = runner.invoke(
                typer_app,
                ["start", "--model", "cv2", "--device", "cpu",
                 "--mask-dir", str(mask_dir)],
            )
        assert result.exit_code == 0
        mock_launch.assert_called_once()
        api = mock_launch.call_args[0][0]  # autospec：首参是 Api 实例
        assert api.config.mask_dir == mask_dir.absolute()

    def test_missing_model_download_failure_exits_1(self, tmp_path, monkeypatch):
        _patch_hub_ok(monkeypatch)
        monkeypatch.setattr("iopaint.download.scan_models", lambda: [])

        def boom(model):
            raise ModelLoadError("gone")

        monkeypatch.setattr("iopaint.download.cli_download_model", boom)
        with captured_logs("ERROR") as messages:
            result = runner.invoke(
                typer_app, ["start", "--model", "org/missing", "--device", "cpu"]
            )
        assert result.exit_code == 1
        assert any("Failed to download model org/missing" in m for m in messages)

    def test_api_init_model_load_error_exits_1(self, monkeypatch):
        _patch_hub_ok(monkeypatch)
        monkeypatch.setattr(
            "iopaint.download.scan_models",
            lambda: [SimpleNamespace(name="cv2")],
        )

        class BoomApi:
            def __init__(self, app, config):
                raise ModelLoadError("plugin deps missing")

        monkeypatch.setattr("iopaint.api.Api", BoomApi)
        with captured_logs("ERROR") as messages:
            result = runner.invoke(
                typer_app, ["start", "--model", "cv2", "--device", "cpu"]
            )
        assert result.exit_code == 1
        assert any("Failed to start IOPaint" in m for m in messages)

    def test_inbrowser_lifespan_opens_browser(self, monkeypatch):
        _patch_hub_ok(monkeypatch)
        monkeypatch.setattr(
            "iopaint.download.scan_models",
            lambda: [SimpleNamespace(name="cv2")],
        )
        # Api.launch 被换成捕获 self：随后手动驱动 app 的 lifespan，
        # 等价于 uvicorn 启动时执行 cli.start 里定义的 lifespan 闭包。
        monkeypatch.setattr(api_mod, "main_loop", None)  # 快照-恢复，防泄漏
        captured = []
        opened = []
        monkeypatch.setattr(
            "iopaint.api.Api.launch", lambda self: captured.append(self)
        )
        monkeypatch.setattr(
            "webbrowser.open", lambda url, **kw: opened.append(url)
        )
        result = runner.invoke(
            typer_app,
            ["start", "--model", "cv2", "--device", "cpu",
             "--inbrowser", "--port", "19999"],
        )
        assert result.exit_code == 0
        assert len(captured) == 1

        async def drive_lifespan(app):
            async with app.router.lifespan_context(app):
                pass

        asyncio.run(drive_lifespan(captured[0].app))
        assert opened == ["http://localhost:19999"]


class TestStartWebConfig:
    def test_calls_main_with_config_file(self, tmp_path, monkeypatch):
        calls = []
        fake_module = types.ModuleType("iopaint.web_config")
        fake_module.main = lambda config_file: calls.append(config_file)
        monkeypatch.setitem(sys.modules, "iopaint.web_config", fake_module)
        config_file = tmp_path / "my-config.json"
        result = runner.invoke(
            typer_app, ["start-web-config", "--config-file", str(config_file)]
        )
        assert result.exit_code == 0
        assert calls == [config_file]

    def test_default_config_file(self, monkeypatch):
        calls = []
        fake_module = types.ModuleType("iopaint.web_config")
        fake_module.main = lambda config_file: calls.append(config_file)
        monkeypatch.setitem(sys.modules, "iopaint.web_config", fake_module)
        result = runner.invoke(typer_app, ["start-web-config"])
        assert result.exit_code == 0
        assert calls == [Path("config.json")]
