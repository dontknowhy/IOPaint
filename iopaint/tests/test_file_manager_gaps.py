"""file_manager 包缺口补测：_media_names 缓存、缩略图分支、colormode/background、
storage_backends 的建目录/竞态、utils 的嗅探回退。

全部在 tmp_path 里用 PIL 造真图，不绑端口（方法级直调 + TestClient 都不用起服务）。
"""
import errno
import os
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from PIL import Image

from iopaint.file_manager import FileManager
from iopaint.file_manager.storage_backends import (
    BaseStorageBackend,
    FilesystemStorageBackend,
)
from iopaint.file_manager.utils import glob_img, parse_size, sniff_media_type


def _save(directory, name, size=(32, 24), color=(10, 20, 30), fmt="PNG"):
    directory.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    path = directory / name
    img.save(path, format=fmt)
    return path


def _build(tmp_path, mask_dir=None):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    app = FastAPI()
    fm = FileManager(
        app=app, input_dir=input_dir, mask_dir=mask_dir, output_dir=output_dir
    )
    return fm, input_dir, output_dir


class TestGetDir:
    def test_output_tab(self, tmp_path):
        fm, _, output_dir = _build(tmp_path)
        assert fm._get_dir("output") == output_dir

    def test_unknown_tab_is_422(self, tmp_path):
        fm, _, _ = _build(tmp_path)
        with pytest.raises(HTTPException) as exc:
            fm._get_dir("bogus")
        assert exc.value.status_code == 422


class TestMediaNamesCache:
    """_media_cache 按 (mtime, ctime) 复用条目，变化/增删时重建。"""

    def test_populates_and_reuses_entries(self, tmp_path):
        fm, input_dir, _ = _build(tmp_path)
        _save(input_dir, "a.png", size=(32, 24))
        _save(input_dir, "b.png", size=(16, 8))

        first = fm._media_names(input_dir)
        assert [m.name for m in first] == ["a.png", "b.png"]
        assert (first[0].width, first[0].height) == (32, 24)
        assert (first[1].width, first[1].height) == (16, 8)

        second = fm._media_names(input_dir)
        # 未变化：返回同一批对象（缓存命中，不再 open 图片）
        assert second[0] is first[0]
        assert second[1] is first[1]

    def test_mtime_change_rereads(self, tmp_path):
        fm, input_dir, _ = _build(tmp_path)
        path = _save(input_dir, "a.png", size=(32, 24))
        first = fm._media_names(input_dir)

        new_time = first[0].mtime + 10
        os.utime(path, (new_time, new_time))

        second = fm._media_names(input_dir)
        assert second[0] is not first[0]
        assert second[0].mtime == new_time

    def test_delete_updates_cache(self, tmp_path):
        fm, input_dir, _ = _build(tmp_path)
        _save(input_dir, "a.png")
        _save(input_dir, "b.png")
        fm._media_names(input_dir)
        assert len(fm._media_cache[str(input_dir.absolute())]) == 2

        (input_dir / "b.png").unlink()
        res = fm._media_names(input_dir)
        assert [m.name for m in res] == ["a.png"]
        # len(cached) != len(new_cache) → 缓存重建，不再留 b.png
        assert list(fm._media_cache[str(input_dir.absolute())].keys()) == ["a.png"]


class TestThumbnailBranches:
    def test_none_directory_is_422(self, tmp_path):
        fm, _, _ = _build(tmp_path)
        with pytest.raises(HTTPException) as exc:
            fm.get_thumbnail(None, "a.png", 16, 16)
        assert exc.value.status_code == 422

    def test_missing_file_is_422(self, tmp_path):
        fm, input_dir, _ = _build(tmp_path)
        _save(input_dir, "a.png")
        with pytest.raises(HTTPException) as exc:
            fm.get_thumbnail(input_dir, "missing.png", 16, 16)
        assert exc.value.status_code == 422

    def test_zero_zero_defaults_to_256_width(self, tmp_path):
        fm, input_dir, _ = _build(tmp_path)
        _save(input_dir, "a.png", size=(40, 20))
        _, (width, height) = fm.get_thumbnail(input_dir, "a.png", 0, 0)
        assert width == 256
        assert height == 128  # 20 * 256 / 40

    def test_zero_width_computed_from_height(self, tmp_path):
        fm, input_dir, _ = _build(tmp_path)
        _save(input_dir, "a.png", size=(40, 20))
        _, (width, height) = fm.get_thumbnail(input_dir, "a.png", 0, 10)
        assert height == 10
        assert width == 20  # 40 * 10 / 20

    def test_existing_thumbnail_skips_regeneration(self, tmp_path):
        fm, input_dir, _ = _build(tmp_path)
        _save(input_dir, "a.png", size=(40, 20))
        path1, size1 = fm.get_thumbnail(input_dir, "a.png", 16, 10)
        assert os.path.exists(path1)
        path2, size2 = fm.get_thumbnail(input_dir, "a.png", 16, 10)
        assert path2 == path1
        assert size2 == size1

    def test_truncated_image_returns_path_without_file(self, tmp_path):
        """图片损坏（load 抛 OSError）→ warning 后返回路径，不写缩略图、不抛。"""
        fm, input_dir, _ = _build(tmp_path)
        path = _save(input_dir, "bad.png", size=(64, 64))
        data = path.read_bytes()
        path.write_bytes(data[: int(len(data) * 0.6)])

        thumb_path, _ = fm.get_thumbnail(input_dir, "bad.png", 16, 16)
        assert not os.path.exists(thumb_path)

    def test_crop_non_fit_uses_thumbnail_resize(self, tmp_path):
        fm, input_dir, _ = _build(tmp_path)
        _save(input_dir, "a.png", size=(40, 20))
        path, (width, height) = fm.get_thumbnail(
            input_dir, "a.png", 16, 16, crop="scale"
        )
        assert os.path.exists(path)
        with Image.open(path) as img:
            # crop!=fit 走 Image.thumbnail：保持宽高比，40x20 → 16x8
            assert (img.width, img.height) == (16, 8)

    def test_background_option_pads_to_square(self, tmp_path):
        """background 选项可用：非方形图被居中贴到灰底方画布上。

        旧实现用 `(a-b)/2` 浮点坐标，Pillow>=10 的 paste 直接 TypeError；
        修复为整数除法后此分支应真正产出方图。
        """
        fm, input_dir, _ = _build(tmp_path)
        _save(input_dir, "a.png", size=(40, 20))
        path, _size = fm.get_thumbnail(input_dir, "a.png", 16, 8, background=0xFF)
        with Image.open(path) as img:
            assert img.size[0] == img.size[1]  # 居中后是正方形


class TestColormode:
    def test_rgba_passthrough_for_rgb_request(self):
        img = Image.new("RGBA", (4, 4))
        assert FileManager.colormode(img, "RGB") is img

    def test_la_converted_to_rgba(self):
        img = Image.new("LA", (4, 4))
        out = FileManager.colormode(img, "RGB")
        assert out.mode == "RGBA"

    def test_gray_converts_to_l(self):
        img = Image.new("RGB", (4, 4))
        assert FileManager.colormode(img, "GRAY").mode == "L"

    def test_other_mode_converted(self):
        img = Image.new("RGB", (4, 4))
        assert FileManager.colormode(img, "CMYK").mode == "CMYK"


class TestBackground:
    def test_pads_non_square_image_to_square(self):
        """非方形图 → 居中贴到 max 边长的灰底画布，结果为正方形。"""
        img = Image.new("RGB", (40, 20), color=(5, 6, 7))
        out = FileManager.background(img)
        assert out.mode == "L"
        assert out.size == (40, 40)


class TestGetFormat:
    def test_option_wins(self, tmp_path):
        fm, input_dir, _ = _build(tmp_path)
        path = _save(input_dir, "a.png")
        with Image.open(path) as img:
            assert fm._get_format(img, format="WEBP") == "WEBP"

    def test_image_format_used(self, tmp_path):
        fm, input_dir, _ = _build(tmp_path)
        path = _save(input_dir, "a.png")
        with Image.open(path) as img:
            assert fm._get_format(img) == "PNG"

    def test_default_jpeg_when_no_format(self, tmp_path):
        fm, _, _ = _build(tmp_path)
        img = Image.new("RGB", (4, 4))  # 内存图无 format
        assert fm._get_format(img) == "JPEG"


class TestFilesystemStorageBackend:
    def test_abstract_methods_raise(self, tmp_path):
        fs = FilesystemStorageBackend()
        with pytest.raises(NotImplementedError):
            BaseStorageBackend.read(fs, "x")
        with pytest.raises(NotImplementedError):
            BaseStorageBackend.exists(fs, "x")
        with pytest.raises(NotImplementedError):
            BaseStorageBackend.save(fs, "x", b"y")

    def test_save_creates_nested_directory(self, tmp_path):
        fs = FilesystemStorageBackend()
        target = tmp_path / "a" / "b" / "c.png"
        fs.save(str(target), b"data")
        assert target.read_bytes() == b"data"

    def test_save_swallows_eexist_race(self, tmp_path, monkeypatch):
        """makedirs 与并发写撞车抛 EEXIST → 吞掉继续写。"""
        fs = FilesystemStorageBackend()
        real_makedirs = os.makedirs

        def racing_makedirs(path, *args, **kwargs):
            real_makedirs(path, *args, **kwargs)  # 目录实际建好
            raise OSError(errno.EEXIST, "File exists")

        monkeypatch.setattr(os, "makedirs", racing_makedirs)
        target = tmp_path / "race" / "x.png"
        fs.save(str(target), b"data")
        assert target.read_bytes() == b"data"

    def test_save_reraises_other_oserror(self, tmp_path, monkeypatch):
        fs = FilesystemStorageBackend()

        def failing_makedirs(path, *args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(os, "makedirs", failing_makedirs)
        with pytest.raises(OSError) as exc:
            fs.save(str(tmp_path / "full" / "x.png"), b"data")
        assert exc.value.errno == errno.ENOSPC

    def test_save_into_file_path_raises_ioerror(self, tmp_path):
        """目标目录位置被一个同名文件占着 → IOError，不静默覆盖。"""
        fs = FilesystemStorageBackend()
        blocker = tmp_path / "blocker"
        blocker.write_bytes(b"")
        with pytest.raises(IOError):
            fs.save(str(blocker / "x.png"), b"data")


class TestUtilsGaps:
    def test_sniff_unreadable_falls_back_to_extension(self, tmp_path):
        missing = tmp_path / "ghost.png"
        assert sniff_media_type(missing) == "image/png"
        weird = tmp_path / "ghost.unknownext"
        assert sniff_media_type(weird) == "application/octet-stream"

    def test_sniff_webp_by_magic(self, tmp_path):
        p = tmp_path / "a.bin"
        p.write_bytes(b"RIFF\x24\x00\x00\x00WEBPVP8 ")
        assert sniff_media_type(p) == "image/webp"

    def test_parse_size_single_int_string(self):
        assert parse_size("64") == [64, 64]

    def test_glob_img_recursive(self, tmp_path):
        (tmp_path / "sub" / "deep").mkdir(parents=True)
        (tmp_path / "top.png").write_bytes(b"x")
        (tmp_path / "sub" / "mid.png").write_bytes(b"x")
        (tmp_path / "sub" / "deep" / "bottom.png").write_bytes(b"x")
        names = sorted(p.name for p in glob_img(tmp_path, recursive=True))
        assert names == ["bottom.png", "mid.png", "top.png"]
