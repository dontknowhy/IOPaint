"""Tests for iopaint.batch_processing – utility functions only (no model)."""
from pathlib import Path

import pytest

from iopaint.batch_processing import glob_images


class TestGlobImages:
    def test_single_file(self, tmp_path):
        p = tmp_path / "image.png"
        p.write_bytes(b"fake")
        result = glob_images(p)
        assert len(result) == 1
        assert "image" in result

    def test_directory(self, tmp_path):
        for name in ["a.png", "b.jpg", "c.jpeg", "d.txt", "e.bmp"]:
            (tmp_path / name).write_bytes(b"fake")
        result = glob_images(tmp_path)
        assert len(result) == 3
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_empty_directory(self, tmp_path):
        result = glob_images(tmp_path)
        assert len(result) == 0

    def test_nonexistent_path(self, tmp_path):
        p = tmp_path / "does_not_exist"
        result = glob_images(p)
        # Should return None (function has no explicit return for non-file non-dir)
        assert result is None or len(result) == 0
