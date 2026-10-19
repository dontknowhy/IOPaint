"""web_config.py 缺口补测：load_config / save_config / change_current_model，
以及 main() 的 UI 组装冒烟（launch 打桩，不起服务）。

web_config 顶层 import gradio（~2s），只在此文件付一次。save_config 的签名与
gr 组件列表是位置绑定的（AGENTS.md gotcha），这里用 **kwargs 调用以稳住解耦。
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import iopaint.web_config as web_config
from iopaint.web_config import WebConfig, change_current_model, load_config, save_config


@pytest.fixture(autouse=True)
def _config_file(tmp_path, monkeypatch):
    """save_config 写模块级 _config_file；每个用例指向 tmp 里的独立文件。"""
    cfg = tmp_path / "config.json"
    monkeypatch.setattr(web_config, "_config_file", cfg)
    return cfg


def _save_kwargs(**overrides):
    params = dict(web_config.default_configs)
    params.update(overrides)
    return params


class TestLoadConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.json")
        assert isinstance(cfg, WebConfig)
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080
        assert cfg.model == web_config.DEFAULT_MODEL

    def test_valid_file_merges_over_defaults(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"host": "0.0.0.0", "port": 9999}), encoding="utf-8")
        cfg = load_config(p)
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9999
        # 未覆盖项保持默认
        assert cfg.model == web_config.DEFAULT_MODEL

    def test_invalid_json_falls_back_to_defaults(self, tmp_path, capsys):
        p = tmp_path / "c.json"
        p.write_text("{ broken json", encoding="utf-8")
        cfg = load_config(p)
        assert cfg.host == "127.0.0.1"
        assert "Load config file failed" in capsys.readouterr().out


class TestSaveConfig:
    def test_writes_json_and_returns_msg(self, _config_file):
        msg = save_config(**_save_kwargs(host="0.0.0.0", port=9000, model=" lama "))
        assert "Successful save config" in msg
        saved = json.loads(_config_file.read_text(encoding="utf-8"))
        assert saved["host"] == "0.0.0.0"
        assert saved["port"] == 9000
        assert saved["model"] == "lama"  # strip 后写入

    def test_dot_paths_become_none(self, _config_file):
        save_config(
            **_save_kwargs(input=".", output_dir=".", mask_dir=".")
        )
        saved = json.loads(_config_file.read_text(encoding="utf-8"))
        assert saved["input"] is None
        assert saved["output_dir"] is None
        assert saved["mask_dir"] is None

    def test_missing_input_returns_error(self, _config_file, tmp_path):
        msg = save_config(**_save_kwargs(input=str(tmp_path / "ghost")))
        assert msg == "[Error] Input file or directory does not exist"
        assert not _config_file.exists()  # 校验失败不落盘

    def test_existing_input_accepted(self, _config_file, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        msg = save_config(**_save_kwargs(input=str(img_dir)))
        assert "Successful save config" in msg
        saved = json.loads(_config_file.read_text(encoding="utf-8"))
        assert saved["input"] == str(img_dir)

    def test_write_failure_returns_message(self, _config_file, monkeypatch):
        """目标路径不可写（这里用目录顶替文件）→ 返回错误串，不抛。"""

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("builtins.open", boom)
        msg = save_config(**_save_kwargs())
        assert msg.startswith("Save configure file failed: ")
        assert "disk full" in msg


class TestSaveConfigLocalized:
    """save_config 的动态消息按请求的 `Accept-Language` 本地化，无请求时英文。"""

    @staticmethod
    def _request(accept_language: str) -> Any:
        # save_config 只读 request.headers，用 SimpleNamespace 顶替 gr.Request
        return SimpleNamespace(headers={"accept-language": accept_language})

    def test_zh_cn_success_message(self, _config_file: Path) -> None:
        msg = save_config(
            **_save_kwargs(), request=self._request("zh-CN,zh;q=0.9,en;q=0.8")
        )
        assert "已成功保存配置到" in msg
        assert str(_config_file.absolute()) in msg

    def test_zh_cn_missing_input(self, _config_file: Path, tmp_path: Path) -> None:
        msg = save_config(
            **_save_kwargs(input=str(tmp_path / "ghost")),
            request=self._request("zh-CN"),
        )
        assert msg == "[错误] 输入文件或目录不存在"
        assert not _config_file.exists()

    def test_unavailable_locale_falls_back_to_en(self, _config_file: Path) -> None:
        msg = save_config(**_save_kwargs(), request=self._request("ja-JP,ja;q=0.9"))
        assert "Successful save config to" in msg

    def test_without_request_is_english(self, _config_file: Path) -> None:
        msg = save_config(**_save_kwargs())
        assert "Successful save config to" in msg


class TestChangeCurrentModel:
    def test_passthrough(self):
        assert change_current_model("runwayml/stable-diffusion-inpainting") == (
            "runwayml/stable-diffusion-inpainting"
        )


class TestMainSmoke:
    def test_builds_ui_and_launches(self, tmp_path, monkeypatch):
        """main() 组装 gr UI 并 launch：launch 打桩，scan_models 桩掉避免扫盘。"""
        import gradio as gr

        monkeypatch.setattr(web_config, "scan_models", lambda: [])
        launches = []

        def fake_launch(self, *args, **kwargs):
            launches.append(kwargs)

        monkeypatch.setattr(gr.Blocks, "launch", fake_launch)

        cfg_file = tmp_path / "config.json"
        web_config.main(cfg_file)

        assert web_config._config_file == cfg_file
        assert len(launches) == 1
        assert launches[0]["inbrowser"] is True
        assert launches[0]["show_api"] is False
        # 自定义翻译随 launch 交给 Gradio 前端（gr.I18n 实例，不便逐字段断言）
        assert "i18n" in launches[0]
