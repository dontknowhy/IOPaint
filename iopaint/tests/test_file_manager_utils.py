"""Tests for iopaint.file_manager utilities."""
import hashlib
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from iopaint.file_manager.utils import (
    aspect_to_string,
    generate_filename,
    glob_img,
    parse_size,
)


# ---------------------------------------------------------------------------
# parse_size
# ---------------------------------------------------------------------------

class TestParseSize:
    def test_int(self):
        assert parse_size(100) == [100, 100]

    def test_tuple(self):
        assert parse_size((200, 100)) == (200, 100)

    def test_list(self):
        assert parse_size([300, 150]) == [300, 150]

    def test_single_element_tuple(self):
        assert parse_size((50,)) == (50, 50)

    def test_string_format(self):
        assert parse_size("320x240") == [320, 240]

    def test_single_string(self):
        assert parse_size("128x128") == [128, 128]

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            parse_size("abc")

    def test_string_single_int(self):
        result = parse_size("64x64")
        assert result == [64, 64]


# ---------------------------------------------------------------------------
# aspect_to_string
# ---------------------------------------------------------------------------

class TestAspectToString:
    def test_tuple(self):
        assert aspect_to_string((100, 200)) == "100x200"

    def test_list(self):
        assert aspect_to_string([300, 400]) == "300x400"

    def test_string_passthrough(self):
        assert aspect_to_string("100x200") == "100x200"


# ---------------------------------------------------------------------------
# generate_filename
# ---------------------------------------------------------------------------

class TestGenerateFilename:
    def test_deterministic(self):
        d = Path("/tmp/test")
        name1 = generate_filename(d, "img.png", "100x100", "fit", 90)
        name2 = generate_filename(d, "img.png", "100x100", "fit", 90)
        assert name1 == name2

    def test_different_inputs_different_names(self):
        d = Path("/tmp/test")
        name1 = generate_filename(d, "img1.png")
        name2 = generate_filename(d, "img2.png")
        assert name1 != name2

    def test_ends_with_jpg(self):
        name = generate_filename(Path("/tmp"), "test.png")
        assert name.endswith(".jpg")


# ---------------------------------------------------------------------------
# glob_img
# ---------------------------------------------------------------------------

class TestGlobImg:
    def test_glob_directory(self, tmp_path):
        for name in ["a.png", "b.jpg", "c.txt", "d.jpeg"]:
            (tmp_path / name).write_bytes(b"fake")
        result = list(glob_img(tmp_path))
        names = {p.name for p in result}
        assert "a.png" in names
        assert "b.jpg" in names
        assert "d.jpeg" in names
        assert "c.txt" not in names

    def test_glob_single_file(self, tmp_path):
        p = tmp_path / "test.png"
        p.write_bytes(b"fake")
        result = list(glob_img(p))
        assert len(result) == 1
        assert result[0] == p

    def test_glob_non_image_file(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_bytes(b"fake")
        result = list(glob_img(p))
        assert len(result) == 0

    def test_glob_empty_directory(self, tmp_path):
        result = list(glob_img(tmp_path))
        assert len(result) == 0

    def test_glob_case_insensitive(self, tmp_path):
        for name in ["A.PNG", "b.JPG"]:
            (tmp_path / name).write_bytes(b"fake")
        result = list(glob_img(tmp_path))
        assert len(result) == 2
