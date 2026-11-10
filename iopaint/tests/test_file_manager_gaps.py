"""file_manager 包缺口补测：_media_names 缓存、缩略图分支、colormode/background、
storage_backends 的建目录/竞态、utils 的嗅探回退。

全部在 tmp_path 里用 PIL 造真图，不绑端口（方法级直调 + TestClient 都不用起服务）。
"""
import errno
import os
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
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

    def test_non_image_entries_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        """软链断裂 / 名为 x.png 的目录 / 非图片字节：跳过该项，列表仍 200。

        旧实现直接在 os.stat/Image.open 上抛 OSError，整个 /medias 变 500。
        """
        fm, input_dir, _ = _build(tmp_path)
        _save(input_dir, "ok.png")
        os.symlink(tmp_path / "ghost.png", input_dir / "broken.png")
        (input_dir / "dir.png").mkdir()
        (input_dir / "junk.png").write_bytes(b"not an image")

        for _ in range(2):  # 第二次走缓存，结果一致
            res = fm._media_names(input_dir)
            assert [m.name for m in res] == ["ok.png"]

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

    def test_replaced_source_regenerates_thumbnail(self, tmp_path: Path) -> None:
        """同名替换源图后必须重新生成缩略图（缓存键包含源 mtime_ns）。"""
        fm, input_dir, _ = _build(tmp_path)
        path = _save(input_dir, "a.png", size=(40, 20))
        old_thumb, old_size = fm.get_thumbnail(input_dir, "a.png", 16, 0)
        assert old_size == (16, 8)

        _save(input_dir, "a.png", size=(32, 32), color=(200, 0, 0))
        new_time_ns = int(os.stat(path).st_mtime_ns) + 10**9
        os.utime(path, ns=(new_time_ns, new_time_ns))

        new_thumb, new_size = fm.get_thumbnail(input_dir, "a.png", 16, 0)
        assert new_thumb != old_thumb
        assert new_size == (16, 16)
        with Image.open(new_thumb) as img:
            assert (img.width, img.height) == (16, 16)

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


class TestEndpoints:
    """端点层不因坏输入变 500：目录路径 / 损坏源图的缩略图。"""

    def test_media_file_rejects_directory(self, tmp_path: Path) -> None:
        fm, input_dir, _ = _build(tmp_path)
        (input_dir / "sub").mkdir(parents=True)
        client = TestClient(fm.app, raise_server_exceptions=False)
        resp = client.get(
            "/api/v1/media_file", params={"tab": "input", "filename": "sub"}
        )
        assert resp.status_code == 422

    def test_corrupt_source_thumbnail_is_422(self, tmp_path: Path) -> None:
        fm, input_dir, _ = _build(tmp_path)
        path = _save(input_dir, "bad.png", size=(64, 64))
        data = path.read_bytes()
        path.write_bytes(data[: int(len(data) * 0.6)])
        client = TestClient(fm.app, raise_server_exceptions=False)
        resp = client.get(
            "/api/v1/media_thumbnail_file",
            params={"tab": "input", "filename": "bad.png", "width": 16, "height": 16},
        )
        assert resp.status_code == 422


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


class TestMediaPagination:
    """/medias 的 limit/offset/sort/order：翻页结果拼起来必须是整个目录。"""

    def _client_with(
        self, tmp_path: Path, count: int = 5
    ) -> tuple[TestClient, Path]:
        fm, input_dir, _ = _build(tmp_path)
        base = 1_700_000_000
        for i in range(count):
            path = _save(input_dir, f"img{i}.png", size=(16 + i, 12 + i))
            os.utime(path, ns=((base + i) * 10**9,) * 2)
        return TestClient(fm.app), input_dir

    def test_pages_cover_the_whole_sorted_list(self, tmp_path: Path) -> None:
        client, _ = self._client_with(tmp_path)
        resp = client.get("/api/v1/medias", params={"tab": "input", "limit": 2})
        assert resp.status_code == 200
        assert resp.headers["X-Total-Count"] == "5"

        names: list[str] = []
        for offset in (0, 2, 4):
            page = client.get(
                "/api/v1/medias",
                params={"tab": "input", "limit": 2, "offset": offset},
            )
            assert page.status_code == 200
            names.extend(item["name"] for item in page.json())
        assert names == [f"img{i}.png" for i in range(5)]

    def test_sort_by_mtime_desc(self, tmp_path: Path) -> None:
        client, _ = self._client_with(tmp_path)
        page = client.get(
            "/api/v1/medias",
            params={"tab": "input", "sort": "mtime", "order": "desc", "limit": 3},
        )
        assert [item["name"] for item in page.json()] == [
            "img4.png",
            "img3.png",
            "img2.png",
        ]

    def test_limit_zero_returns_everything(self, tmp_path: Path) -> None:
        client, _ = self._client_with(tmp_path)
        page = client.get("/api/v1/medias", params={"tab": "input", "limit": 0})
        assert len(page.json()) == 5
        assert page.headers["X-Total-Count"] == "5"

    def test_offset_past_the_end_is_empty_not_error(self, tmp_path: Path) -> None:
        client, _ = self._client_with(tmp_path)
        page = client.get(
            "/api/v1/medias", params={"tab": "input", "limit": 2, "offset": 99}
        )
        assert page.status_code == 200
        assert page.json() == []
        assert page.headers["X-Total-Count"] == "5"

    def test_invalid_sort_is_422(self, tmp_path: Path) -> None:
        client, _ = self._client_with(tmp_path)
        resp = client.get("/api/v1/medias", params={"tab": "input", "sort": "size"})
        assert resp.status_code == 422

    def test_paging_keeps_cached_dimensions_for_later_pages(self, tmp_path: Path) -> None:
        """翻页只 open 当前页；两页都访问过后再翻页不再打开图片。"""
        fm, input_dir, _ = _build(tmp_path)
        for i in range(4):
            _save(input_dir, f"img{i}.png")
        fm._media_page(input_dir, 2, 0, "name", "asc")
        fm._media_page(input_dir, 2, 2, "name", "asc")

        opened = []
        real_open = Image.open

        def counting_open(*args, **kwargs):  # type: ignore[no-untyped-def]
            opened.append(args[0] if args else kwargs.get("fp"))
            return real_open(*args, **kwargs)

        Image.open = counting_open
        try:
            items, total = fm._media_page(input_dir, 2, 0, "name", "asc")
            second, _ = fm._media_page(input_dir, 2, 2, "name", "asc")
        finally:
            Image.open = real_open
        assert [item.name for item in items] == ["img0.png", "img1.png"]
        assert [item.name for item in second] == ["img2.png", "img3.png"]
        assert total == 4
        assert opened == []
