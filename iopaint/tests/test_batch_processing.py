"""Tests for iopaint.batch_processing – utility functions only (no model)."""
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from iopaint.batch_processing import batch_inpaint, glob_images


def _write_image(path: Path, size=(32, 32)):
    Image.new("RGB", size, color=(20, 40, 60)).save(path)


def _write_mask(path: Path, size=(32, 32)):
    arr = np.zeros(size, dtype=np.uint8)
    arr[8:24, 8:24] = 255
    Image.fromarray(arr).save(path)


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


class TestDuplicateStem:
    """D-15：a.jpg + a.png 此前共用 key 'a'，后扫描到的静默覆盖前者 → 少跑一张。"""

    def test_duplicate_stems_keep_both_files(self, tmp_path):
        for name in ["a.png", "a.jpg", "b.png"]:
            (tmp_path / name).write_bytes(b"fake")

        result = glob_images(tmp_path)

        assert len(result) == 3
        assert result["a"] != result["a_1"]
        assert {result["a"].name, result["a_1"].name} == {"a.jpg", "a.png"}
        assert result["b"].name == "b.png"

    def test_triple_duplicate(self, tmp_path):
        for ext in ["png", "jpg", "jpeg"]:
            (tmp_path / f"a.{ext}").write_bytes(b"fake")

        result = glob_images(tmp_path)

        assert set(result) == {"a", "a_1", "a_2"}
        assert len({p.name for p in result.values()}) == 3

    def test_duplicate_emits_warning(self, tmp_path, caplog):
        for name in ["a.png", "a.jpg"]:
            (tmp_path / name).write_bytes(b"fake")

        messages = []
        sink = logger.add(messages.append, level="WARNING")
        try:
            glob_images(tmp_path)
        finally:
            logger.remove(sink)

        text = "".join(str(m) for m in messages)
        assert "Duplicate stem 'a'" in text


class TestBatchSameName:
    """D-15：同名输入的 mask 配对，以及输出目录 == 输入目录时的覆盖保护。"""

    def test_duplicate_images_both_use_base_mask(self, tmp_path):
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        out_dir = tmp_path / "out"
        img_dir.mkdir()
        mask_dir.mkdir()

        _write_image(img_dir / "a.jpg")
        _write_image(img_dir / "a.png")
        _write_mask(mask_dir / "a.png")

        batch_inpaint(
            model="cv2",
            device=torch.device("cpu"),
            image=img_dir,
            mask=mask_dir,
            output=out_dir,
        )

        # 两张图都要被处理；a_1（即 a.jpg）通过基础 stem 回退共用 a.png 的 mask
        assert {p.name for p in out_dir.glob("*.png")} == {"a.png", "a_1.png"}

    def test_missing_mask_still_skips(self, tmp_path):
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        out_dir = tmp_path / "out"
        img_dir.mkdir()
        mask_dir.mkdir()

        _write_image(img_dir / "a.png")
        _write_image(img_dir / "b.png")
        _write_mask(mask_dir / "a.png")

        batch_inpaint(
            model="cv2",
            device=torch.device("cpu"),
            image=img_dir,
            mask=mask_dir,
            output=out_dir,
        )

        # 没有 mask 的 b.png 仍然跳过，不能因为回退逻辑被误配成 a 的 mask
        assert {p.name for p in out_dir.glob("*.png")} == {"a.png"}

    def test_output_dir_equal_input_does_not_overwrite(self, tmp_path):
        data_dir = tmp_path / "data"
        mask_dir = tmp_path / "masks"
        data_dir.mkdir()
        mask_dir.mkdir()

        _write_image(data_dir / "a.png")
        original_bytes = (data_dir / "a.png").read_bytes()
        _write_mask(mask_dir / "a.png")

        batch_inpaint(
            model="cv2",
            device=torch.device("cpu"),
            image=data_dir,
            mask=mask_dir,
            output=data_dir,  # 故意指向输入目录
        )

        assert (data_dir / "a.png").read_bytes() == original_bytes  # 原图未被覆盖
        assert (data_dir / "a_1.png").exists()  # 结果改名落盘
