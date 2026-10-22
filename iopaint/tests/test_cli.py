"""Tests for iopaint.cli – CLI argument parsing and validation."""
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from iopaint.cli import _resolve_export_format, typer_app
from iopaint.schema import Device


runner = CliRunner()


class TestCliHelp:
    def test_root_help(self):
        result = runner.invoke(typer_app, ["--help"])
        assert result.exit_code == 0
        assert "IOPaint" in result.output or "iopaint" in result.output.lower()

    def test_start_help(self):
        result = runner.invoke(typer_app, ["start", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()

    def test_list_help(self):
        result = runner.invoke(typer_app, ["list", "--help"])
        assert result.exit_code == 0


class TestExportFormat:
    def test_empty_means_follow_source(self) -> None:
        assert _resolve_export_format("") is None
        assert _resolve_export_format("  ") is None

    def test_normalizes_case_and_dot(self) -> None:
        assert _resolve_export_format(" PNG ") == "png"
        assert _resolve_export_format(".jpg") == "jpg"
        assert _resolve_export_format(".TIF") == "tiff"

    def test_rejects_unknown_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid --export-format"):
            _resolve_export_format("avif")


class TestCliStartValidation:
    def test_missing_input_with_output_dir(self, tmp_path):
        result = runner.invoke(typer_app, [
            "start",
            "--model", "cv2",
            "--device", "cpu",
            "--input", str(tmp_path / "nonexistent"),
        ])
        # Should exit with error because input doesn't exist
        assert result.exit_code != 0

    def test_valid_input_dir(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        output_dir = tmp_path / "output"

        # 必须 mock 掉 Api.launch：它会真的起 uvicorn 阻塞住并占住端口，
        # 这里只验证参数解析和 ApiConfig 是否被正确组装出来。
        with patch("iopaint.api.Api.launch") as mock_launch:
            result = runner.invoke(typer_app, [
                "start",
                "--model", "cv2",
                "--device", "cpu",
                "--input", str(img_dir),
                "--output-dir", str(output_dir),
                "--port", "19999",
            ])

        mock_launch.assert_called_once()
        assert result.exit_code == 0
        assert "cv2" in result.output

    def test_config_json_export_force_reaches_api_config(
        self, tmp_path: Path
    ) -> None:
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {
                    "export_force": True,
                    "export_format": "png",
                    "export_quality": 80,
                    "export_subsampling": "420",
                    "export_mode": "full",
                    "export_png_compress_level": 9,
                    "export_webp_lossless": True,
                }
            ),
            encoding="utf-8",
        )
        captured: list[Any] = []
        with (
            patch("iopaint.cli.hub_is_reachable", lambda: True),
            patch("iopaint.api.Api.launch", lambda self: captured.append(self.config)),
        ):
            result = runner.invoke(
                typer_app,
                [
                    "start",
                    "--config",
                    str(config),
                    "--model",
                    "cv2",
                    "--device",
                    "cpu",
                    "--port",
                    "19999",
                ],
            )

        assert result.exit_code == 0
        assert len(captured) == 1
        cfg = captured[0]
        assert cfg.export_force is True
        assert cfg.export_format == "png"
        assert cfg.export_quality == 80
        assert cfg.export_subsampling == "420"
        assert cfg.export_mode == "full"
        assert cfg.export_png_compress_level == 9
        assert cfg.export_webp_lossless is True
