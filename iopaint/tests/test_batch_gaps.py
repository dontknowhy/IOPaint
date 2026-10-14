"""batch_processing.py 缺口补测：输入/输出参数校验的退出码、config JSON 加载、
单 mask 文件复用、mask 尺寸不符时的 resize、concat 拼接、输出撞名时的 _N 后缀递增。

全部用真实 cv2 模型（无权重、CPU 即跑），不绑端口。
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

from iopaint.batch_processing import batch_inpaint


def _write_image(path: Path, size=(32, 32)):
    Image.new("RGB", size, color=(20, 40, 60)).save(path)


def _write_mask(path: Path, size=(32, 32)):
    arr = np.zeros(size, dtype=np.uint8)
    arr[8:24, 8:24] = 255
    Image.fromarray(arr).save(path)


def _make_inputs(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    _write_image(img_dir / "a.png")
    _write_mask(mask_dir / "a.png")
    return img_dir, mask_dir


class TestBatchValidation:
    def test_image_dir_with_file_output_exits(self, tmp_path):
        img_dir, mask_dir = _make_inputs(tmp_path)
        output_file = tmp_path / "out.png"
        output_file.write_bytes(b"i am a file")

        with pytest.raises(SystemExit) as exc:
            batch_inpaint(
                model="cv2",
                device=torch.device("cpu"),
                image=img_dir,
                mask=mask_dir,
                output=output_file,
            )

        assert exc.value.code == 1

    def test_empty_image_folder_exits(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        _write_mask(mask_dir / "a.png")

        with pytest.raises(SystemExit) as exc:
            batch_inpaint(
                model="cv2",
                device=torch.device("cpu"),
                image=img_dir,
                mask=mask_dir,
                output=tmp_path / "out",
            )

        assert exc.value.code == -1

    def test_empty_mask_folder_exits(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _write_image(img_dir / "a.png")
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()

        with pytest.raises(SystemExit) as exc:
            batch_inpaint(
                model="cv2",
                device=torch.device("cpu"),
                image=img_dir,
                mask=mask_dir,
                output=tmp_path / "out",
            )

        assert exc.value.code == -1


class TestBatchConfig:
    def test_config_json_is_loaded(self, tmp_path):
        img_dir, mask_dir = _make_inputs(tmp_path)
        config = tmp_path / "config.json"
        config.write_text('{"sd_seed": 4242}', encoding="utf-8")

        messages = []
        sink = logger.add(messages.append, level="INFO")
        try:
            batch_inpaint(
                model="cv2",
                device=torch.device("cpu"),
                image=img_dir,
                mask=mask_dir,
                output=tmp_path / "out",
                config=config,
            )
        finally:
            logger.remove(sink)

        text = "".join(str(m) for m in messages)
        assert "Using config" in text
        assert "4242" in text  # config 里的值真的进了 InpaintRequest
        assert (tmp_path / "out" / "a.png").exists()


class TestBatchMaskHandling:
    def test_single_mask_file_applied_to_all_images(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _write_image(img_dir / "a.png")
        _write_image(img_dir / "b.png")
        mask_file = tmp_path / "mask.png"
        _write_mask(mask_file)
        out_dir = tmp_path / "out"

        batch_inpaint(
            model="cv2",
            device=torch.device("cpu"),
            image=img_dir,
            mask=mask_file,
            output=out_dir,
        )

        assert (out_dir / "a.png").exists()
        assert (out_dir / "b.png").exists()

    def test_mask_size_mismatch_is_resized(self, tmp_path, capsys):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _write_image(img_dir / "a.png", size=(32, 32))
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        _write_mask(mask_dir / "a.png", size=(16, 16))
        out_dir = tmp_path / "out"

        batch_inpaint(
            model="cv2",
            device=torch.device("cpu"),
            image=img_dir,
            mask=mask_dir,
            output=out_dir,
        )

        assert "resize mask" in capsys.readouterr().out
        out = Image.open(out_dir / "a.png")
        assert out.size == (32, 32)


class TestBatchOutput:
    def test_concat_joins_img_mask_result(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _write_image(img_dir / "a.png", size=(32, 24))
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()
        _write_mask(mask_dir / "a.png", size=(32, 24))
        out_dir = tmp_path / "out"

        batch_inpaint(
            model="cv2",
            device=torch.device("cpu"),
            image=img_dir,
            mask=mask_dir,
            output=out_dir,
            concat=True,
        )

        out = Image.open(out_dir / "a.png")
        assert out.size == (96, 24)  # 原图 + mask + 结果，宽 3 倍

    def test_output_collision_walks_past_existing_suffixes(self, tmp_path):
        """输入目录里 a.png、a_1.png 都存在且 output==input 时，结果要写到 a_2.png。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir()

        _write_image(data_dir / "a.png")
        _write_image(data_dir / "a_1.png")
        original_a = (data_dir / "a.png").read_bytes()
        original_a1 = (data_dir / "a_1.png").read_bytes()
        _write_mask(mask_dir / "a.png")
        _write_mask(mask_dir / "a_1.png")

        batch_inpaint(
            model="cv2",
            device=torch.device("cpu"),
            image=data_dir,
            mask=mask_dir,
            output=data_dir,  # 故意指向输入目录
        )

        # stem "a"：a.png 被占、a_1.png 也被占 → 写 a_2.png
        assert (data_dir / "a_2.png").exists()
        # stem "a_1"：a_1_1.png 未被占用 → 直接写
        assert (data_dir / "a_1_1.png").exists()
        assert (data_dir / "a.png").read_bytes() == original_a
        assert (data_dir / "a_1.png").read_bytes() == original_a1
