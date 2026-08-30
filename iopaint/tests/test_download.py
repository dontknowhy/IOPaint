"""Tests for iopaint.download – model scanning helpers (no network)."""
import json
import os
from pathlib import Path
from unittest.mock import patch

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
