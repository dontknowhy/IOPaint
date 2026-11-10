import os
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from PIL import Image, ImageOps, PngImagePlugin
from fastapi import FastAPI, HTTPException
from loguru import logger
from starlette.responses import FileResponse, Response

from ..schema import MediasResponse, MediaTab

LARGE_ENOUGH_NUMBER = 10
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
from .storage_backends import FilesystemStorageBackend
from .utils import IMG_SUFFIX, aspect_to_string, generate_filename, sniff_media_type


class FileManager:
    def __init__(self, app: FastAPI, input_dir: Path, mask_dir: Path, output_dir: Path):
        self.app = app
        self.input_dir: Path = input_dir
        self.mask_dir: Path = mask_dir
        self.output_dir: Path = output_dir

        self.image_dir_filenames = []
        self.output_dir_filenames = []
        # Cache of MediasResponse keyed by directory; re-reads only changed files.
        self._media_cache: Dict[str, Dict[str, MediasResponse]] = {}
        # 已算好的缩略图路径：key 是 (目录, 相对文件名, 请求参数)，命中时只校验
        # 源文件 (mtime_ns, size)，不必再 open 原图读宽高。
        self._thumbnail_cache: Dict[tuple, tuple] = {}
        if not self.thumbnail_directory.exists():
            self.thumbnail_directory.mkdir(parents=True)

        # fmt: off
        self.app.add_api_route("/api/v1/medias", self.api_medias, methods=["GET"], response_model=List[MediasResponse])
        self.app.add_api_route("/api/v1/media_file", self.api_media_file, methods=["GET"])
        self.app.add_api_route("/api/v1/media_thumbnail_file", self.api_media_thumbnail_file, methods=["GET"])
        # fmt: on

    def api_medias(
        self,
        tab: MediaTab,
        response: Response,
        limit: Optional[int] = None,
        offset: int = 0,
        sort: str = "name",
        order: str = "asc",
    ) -> List[MediasResponse]:
        # 分页只影响“为这一页 open 图片取宽高”；扫描+stat 全目录仍然很便宜
        # （10k 文件 13.6ms），所以排序可以覆盖整个目录。
        if sort not in ("name", "ctime", "mtime") or order not in ("asc", "desc"):
            raise HTTPException(
                status_code=422, detail=f"invalid sort: {sort} {order}"
            )
        if offset < 0 or (limit is not None and limit < 0):
            raise HTTPException(status_code=422, detail="invalid limit/offset")
        items, total = self._media_page(self._get_dir(tab), limit, offset, sort, order)
        response.headers["X-Total-Count"] = str(total)
        return items

    def api_media_file(self, tab: MediaTab, filename: str) -> FileResponse:
        file_path = self._get_file(tab, filename)
        return FileResponse(file_path, media_type=sniff_media_type(file_path))

    # tab=${tab}?filename=${filename.name}?width=${width}&height=${height}
    def api_media_thumbnail_file(
        self, tab: MediaTab, filename: str, width: int, height: int
    ) -> FileResponse:
        img_dir = self._get_dir(tab)
        thumb_filepath, (width, height) = self.get_thumbnail(
            img_dir, filename, width=width, height=height
        )
        # 源图损坏时 get_thumbnail 只告警、不写缩略图；直接交给 FileResponse
        # 会因文件不存在变 500
        if not os.path.exists(thumb_filepath):
            raise HTTPException(
                status_code=422, detail=f"thumbnail unavailable: {filename}"
            )
        return FileResponse(
            thumb_filepath,
            headers={
                "X-Width": str(width),
                "X-Height": str(height),
            },
            media_type=sniff_media_type(thumb_filepath),
        )

    def _get_dir(self, tab: MediaTab) -> Optional[Path]:
        if tab == "input":
            return self.input_dir
        elif tab == "output":
            return self.output_dir
        elif tab == "mask":
            return self.mask_dir
        else:
            raise HTTPException(status_code=422, detail=f"tab not found: {tab}")

    def _get_file(self, tab: MediaTab, filename: str) -> Path:
        directory = self._get_dir(tab)
        file_path = self._safe_join(directory, filename, tab)
        # is_file()：`filename=sub` 这类指向目录的路径原来能通过 exists()，
        # 交给 FileResponse 时才炸成 500
        if not file_path.is_file():
            raise HTTPException(status_code=422, detail=f"file not found: {filename}")
        return file_path

    @staticmethod
    def _safe_join(directory: Optional[Path], filename: str, tab: str) -> Path:
        """Resolve `filename` under `directory`, rejecting path traversal.

        `filename` 是用户可控的查询参数，直接 `dir / filename` 会允许
        `filename=../../../etc/passwd` 逃出托管目录（SEC-1/SEC-2）。这里先
        resolve 两侧再做相对性判断；resolve 同时展开符号链接，避免软链逃逸。
        `directory is None`（如未配置 --mask-dir）原来会抛 TypeError 变 500，改 422。
        """
        if directory is None:
            raise HTTPException(
                status_code=422, detail=f"{tab} directory is not configured"
            )
        base = directory.resolve()
        try:
            file_path = (base / filename).resolve()
            file_path.relative_to(base)
        except (ValueError, OSError):  # 不在 base 之下 / 非法路径
            raise HTTPException(
                status_code=422, detail=f"invalid filename: {filename}"
            )
        return file_path

    @property
    def thumbnail_directory(self) -> Path:
        return self.output_dir / "thumbnails"

    @staticmethod
    def _scan_image_entries(directory: Path) -> List[Tuple[str, os.stat_result]]:
        """`(name, stat)` of regular image files, sorted by name.

        用 `os.scandir` 的 `DirEntry.stat()` 而不是 `Path.glob` + `os.stat(path)`：
        后者每张图都要重新解析一次路径（1500 张实测 14.5ms → 3.5ms）。软链断裂、
        名为 `x.png` 的目录、扫描期间被删掉的条目都跳过，原来这些会让
        `/api/v1/medias` 整个 500。
        """
        entries = []
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    if os.path.splitext(entry.name)[1] not in IMG_SUFFIX:
                        continue
                    try:
                        if not entry.is_file():
                            continue
                        entries.append((entry.name, entry.stat()))
                    except OSError:
                        continue
        except OSError:
            return []
        entries.sort(key=lambda item: item[0])
        return entries

    def _media_names(self, directory: Path) -> List[MediasResponse]:
        return self._media_page(directory, None, 0, "name", "asc")[0]

    def _media_page(
        self,
        directory: Optional[Path],
        limit: Optional[int],
        offset: int,
        sort: str,
        order: str,
    ) -> Tuple[List[MediasResponse], int]:
        if directory is None:
            return [], 0
        entries = self._scan_image_entries(directory)
        if sort == "name":
            entries.sort(key=lambda item: item[0])
        else:
            field = "st_ctime" if sort == "ctime" else "st_mtime"
            entries.sort(key=lambda item: (getattr(item[1], field), item[0]))
        if order == "desc":
            entries.reverse()
        total = len(entries)
        # limit 省略/为 0 表示整份列表（前端搜索需要全量名单）
        page = entries[offset:] if limit in (None, 0) else entries[offset : offset + limit]

        cache_key = str(directory.absolute())
        cached = self._media_cache.get(cache_key, {})
        present = {name for name, _ in entries}
        # 只更新本页的条目；其他页已缓存的宽高必须保留，否则翻页会重复 open
        new_cache = {name: entry for name, entry in cached.items() if name in present}
        changed = False
        res = []
        for name, st in page:
            entry = cached.get(name)
            if (
                entry is not None
                and entry.mtime == st.st_mtime
                and entry.ctime == st.st_ctime
            ):
                new_cache[name] = entry
            else:
                try:
                    with Image.open(directory / name) as img:
                        entry = MediasResponse(
                            name=name,
                            height=img.height,
                            width=img.width,
                            ctime=st.st_ctime,
                            mtime=st.st_mtime,
                        )
                except OSError:
                    # 非图片 / 损坏 / 扫描后被删：跳过这一项，别让整个列表 500
                    continue
                changed = True
                new_cache[name] = entry
            res.append(entry)
        if changed or len(cached) != len(new_cache):
            self._media_cache[cache_key] = new_cache
        return res, total

    def get_thumbnail(
        self, directory: Path, original_filename: str, width, height, **options
    ):
        if directory is None:
            raise HTTPException(status_code=422, detail="directory is not configured")
        directory = Path(directory)
        storage = FilesystemStorageBackend(self.app)
        crop = options.get("crop", "fit")
        background = options.get("background")
        quality = options.get("quality", 90)

        original_path, original_filename = os.path.split(original_filename)
        # SEC-2: original_filename 拼进读取路径，必须先做路径遍历校验
        original_filepath = self._safe_join(
            directory, os.path.join(original_path, original_filename), "tab"
        )
        try:
            source_stat = original_filepath.stat()
        except OSError:
            raise HTTPException(
                status_code=422, detail=f"file not found: {original_filename}"
            )

        # 缩略图命中只需源文件指纹。原实现每次都 `storage.read()` 整个原图再
        # `Image.open(BytesIO(...))`，12MP JPEG 命中一次也要读 11.8MB / ~2ms。
        cache_key = (
            str(directory),
            original_path,
            original_filename,
            width,
            height,
            crop,
            background,
            quality,
        )
        cached = self._thumbnail_cache.get(cache_key)
        if cached is not None:
            thumbnail_filepath, cached_size, mtime_ns, size = cached
            if (
                mtime_ns == source_stat.st_mtime_ns
                and size == source_stat.st_size
                and os.path.exists(thumbnail_filepath)
            ):
                return thumbnail_filepath, cached_size

        try:
            image = Image.open(original_filepath)
        except OSError:
            raise HTTPException(
                status_code=422, detail=f"file not found: {original_filename}"
            )
        try:
            # get original image format（必须在 load()/convert 之前读，见 helper 同类注释）
            options["format"] = options.get("format", image.format) or "JPEG"
            # 缩略图文件名后缀必须与实际写出的字节一致（CONS-3）
            format_ext = {
                "JPEG": "jpg",
                "JPG": "jpg",
                "PNG": "png",
                "GIF": "gif",
                "WEBP": "webp",
                "BMP": "bmp",
                "TIFF": "tiff",
            }.get(str(options["format"]).upper(), "jpg")

            # keep ratio resize
            if not width and not height:
                width = 256

            if width != 0:
                height = int(image.height * width / image.width)
            else:
                width = int(image.width * height / image.height)

            thumbnail_size = (width, height)

            thumbnail_filename = generate_filename(
                directory,
                original_filename,
                aspect_to_string(thumbnail_size),
                crop,
                background,
                quality,
                # 同名替换源文件后旧缩略图必须失效
                source_stat.st_mtime_ns,
                ext=format_ext,
            )

            thumbnail_filepath = os.path.join(
                self.thumbnail_directory, original_path, thumbnail_filename
            )

            if storage.exists(thumbnail_filepath):
                self._thumbnail_cache[cache_key] = (
                    thumbnail_filepath,
                    thumbnail_size,
                    source_stat.st_mtime_ns,
                    source_stat.st_size,
                )
                return thumbnail_filepath, (width, height)

            try:
                image.load()
            except (IOError, OSError):
                logger.warning(f"Thumbnail cannot load image: {original_filepath}")
                return thumbnail_filepath, (width, height)

            thumbnail = self._create_thumbnail(
                image, thumbnail_size, crop, background=background
            )
            try:
                raw_data = self.get_raw_data(thumbnail, **options)
            finally:
                thumbnail.close()
            storage.save(thumbnail_filepath, raw_data)
            self._thumbnail_cache[cache_key] = (
                thumbnail_filepath,
                thumbnail_size,
                source_stat.st_mtime_ns,
                source_stat.st_size,
            )

            return thumbnail_filepath, (width, height)
        finally:
            # 原实现用 BytesIO 托管原图，磁盘版必须显式关闭，否则每张缓存
            # 未命中的缩略图都会漏一个 fd（_create_thumbnail 返回的是新图）
            image.close()

    def get_raw_data(self, image, **options):
        data = {
            "format": self._get_format(image, **options),
            "quality": options.get("quality", 90),
        }

        _file = BytesIO()
        image.save(_file, **data)
        return _file.getvalue()

    @staticmethod
    def colormode(image, colormode="RGB"):
        if colormode == "RGB" or colormode == "RGBA":
            if image.mode == "RGBA":
                return image
            if image.mode == "LA":
                return image.convert("RGBA")
            return image.convert(colormode)

        if colormode == "GRAY":
            return image.convert("L")

        return image.convert(colormode)

    @staticmethod
    def background(original_image, color=0xFF):
        size = (max(original_image.size),) * 2
        image = Image.new("L", size, color)
        image.paste(
            original_image,
            # 整数除法：Pillow>=10 的 paste 拒绝浮点坐标（旧版 /2 会算出 .5 直接 TypeError）
            tuple((a - b) // 2 for a, b in zip(size, original_image.size)),
        )

        return image

    def _get_format(self, image, **options):
        if options.get("format"):
            return options.get("format")
        if image.format:
            return image.format

        return "JPEG"

    def _create_thumbnail(self, image, size, crop="fit", background=None):
        resample = Image.Resampling.LANCZOS

        if crop == "fit":
            image = ImageOps.fit(image, size, resample)
        else:
            image = image.copy()
            image.thumbnail(size, resample=resample)

        if background is not None:
            image = self.background(image)

        image = self.colormode(image)

        return image
