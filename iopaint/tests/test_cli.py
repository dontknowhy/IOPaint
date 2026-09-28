"""Tests for iopaint.cli – CLI argument parsing and validation."""
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from iopaint.cli import typer_app
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
