"""Tests for iopaint.download – model scanning helpers (no network)."""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iopaint.download import folder_name_to_show_name


class TestFolderNameToShowName:
    def test_basic(self):
        assert folder_name_to_show_name("models--foo--bar") == "foo/bar"

    def test_no_prefix(self):
        assert folder_name_to_show_name("some--name") == "some/name"

    def test_single_segment(self):
        assert folder_name_to_show_name("models--single") == "single"

    def test_empty(self):
        result = folder_name_to_show_name("models----")
        assert result == "/"


class TestScanSingleFileDiffusionModels:
    def test_empty_cache_dir(self, tmp_path):
        from iopaint.download import scan_single_file_diffusion_models
        result = scan_single_file_diffusion_models(tmp_path)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_nonexistent_stable_diffusion_dir(self, tmp_path):
        from iopaint.download import scan_single_file_diffusion_models
        # Should not raise even if stable_diffusion dir doesn't exist
        result = scan_single_file_diffusion_models(tmp_path)
        assert isinstance(result, list)


class TestScanDirtyModelIndex:
    """BUG-7/D-18：单个坏 model_index.json 只 warning + 跳过，不打断整个扫描。

    旧实现里 `data["_class_name"]` 缺键、顶层不是对象、文件打不开都会直接抛，
    而 ModelManager.__init__ 里就会调 scan_models() → 一个脏文件让服务起不来。
    """

    @staticmethod
    def _write_model_index(path: Path, content: str):
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_index.json").write_text(content, encoding="utf-8")

    def test_scan_diffusers_skips_dirty_files(self, tmp_path, monkeypatch):
        import huggingface_hub.constants as hf_const
        from iopaint import download
        from iopaint.download import scan_diffusers_models

        root = tmp_path / "hub"
        # 正常 diffusers SD 模型（层级 models--x--y/snapshots/rev/model_index.json）
        self._write_model_index(
            root / "models--good--sd" / "snapshots" / "abc",
            json.dumps({"_class_name": "StableDiffusionPipeline"}),
        )
        self._write_model_index(root / "models--bad--json" / "snapshots" / "abc", "{ not json")
        self._write_model_index(root / "models--miss--key" / "snapshots" / "abc", "{}")
        self._write_model_index(
            root / "models--arr--list" / "snapshots" / "abc", json.dumps([1, 2])
        )
        # 是个目录 → open() 抛 IsADirectoryError(OSError)
        (root / "models--dir--open" / "snapshots" / "abc" / "model_index.json").mkdir(
            parents=True
        )

        monkeypatch.setattr(hf_const, "HF_HUB_CACHE", str(root))
        fake_logger = MagicMock()
        monkeypatch.setattr(download, "logger", fake_logger)

        result = scan_diffusers_models()

        assert [m.name for m in result] == ["good/sd"]
        # 4 个坏目录全部只 warning、不抛
        assert fake_logger.warning.call_count == 4
        assert all(
            "Skip invalid model_index.json" in str(c)
            for c in fake_logger.warning.call_args_list
        )

    def test_scan_converted_skips_dirty_files(self, tmp_path, monkeypatch):
        from iopaint import download
        from iopaint.download import _scan_converted_diffusers_models

        root = tmp_path / "stable_diffusion"
        self._write_model_index(
            root / "my-inpaint",
            json.dumps({"_class_name": "StableDiffusionInpaintPipeline"}),
        )
        self._write_model_index(root / "bad-json", "{oops")
        self._write_model_index(root / "no-key", "{}")

        fake_logger = MagicMock()
        monkeypatch.setattr(download, "logger", fake_logger)

        result = _scan_converted_diffusers_models(root)

        assert [m.name for m in result] == ["my-inpaint"]
        assert fake_logger.warning.call_count == 2
        assert all("skip it" in str(c) for c in fake_logger.warning.call_args_list)


class TestGetSdModelType:
    def test_inpaint_in_name(self, tmp_path):
        from iopaint.download import get_sd_model_type
        # Create a fake file with "inpaint" in name
        fake = tmp_path / "model-inpaint.safetensors"
        fake.write_bytes(b"fake")
        result = get_sd_model_type(str(fake))
        assert result is not None
        from iopaint.schema import ModelType
        assert result == ModelType.DIFFUSERS_SD_INPAINT
