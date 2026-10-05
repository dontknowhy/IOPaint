import os
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image, ImageOps, PngImagePlugin
from fastapi import FastAPI, HTTPException
from loguru import logger
from starlette.responses import FileResponse

from ..schema import MediasResponse, MediaTab

LARGE_ENOUGH_NUMBER = 10
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
from .storage_backends import FilesystemStorageBackend
from .utils import aspect_to_string, generate_filename, glob_img, sniff_media_type


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
        if not self.thumbnail_directory.exists():
            self.thumbnail_directory.mkdir(parents=True)

        # fmt: off
        self.app.add_api_route("/api/v1/medias", self.api_medias, methods=["GET"], response_model=List[MediasResponse])
        self.app.add_api_route("/api/v1/media_file", self.api_media_file, methods=["GET"])
        self.app.add_api_route("/api/v1/media_thumbnail_file", self.api_media_thumbnail_file, methods=["GET"])
        # fmt: on

    def api_medias(self, tab: MediaTab) -> List[MediasResponse]:
        img_dir = self._get_dir(tab)
        return self._media_names(img_dir)

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
        if not file_path.exists():
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

    def _media_names(self, directory: Path) -> List[MediasResponse]:
        if directory is None:
            return []
        names = sorted([it.name for it in glob_img(directory)])
        cache_key = str(directory.absolute())
        cached = self._media_cache.get(cache_key, {})
        new_cache = {}
        res = []
        changed = False
        for name in names:
            path = directory / name
            st = os.stat(path)
            entry = cached.get(name)
            if (
                entry is not None
                and entry.mtime == st.st_mtime
                and entry.ctime == st.st_ctime
            ):
                new_cache[name] = entry
            else:
                with Image.open(path) as img:
                    entry = MediasResponse(
                        name=name,
                        height=img.height,
                        width=img.width,
                        ctime=st.st_ctime,
                        mtime=st.st_mtime,
                    )
                changed = True
            new_cache[name] = entry
            res.append(entry)
        if changed or len(cached) != len(new_cache):
            self._media_cache[cache_key] = new_cache
        return res

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
        if not original_filepath.exists():
            raise HTTPException(
                status_code=422, detail=f"file not found: {original_filename}"
            )
        image = Image.open(BytesIO(storage.read(original_filepath)))
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
                ext=format_ext,
            )

            thumbnail_filepath = os.path.join(
                self.thumbnail_directory, original_path, thumbnail_filename
            )

            if storage.exists(thumbnail_filepath):
                return thumbnail_filepath, (width, height)

            try:
                image.load()
            except (IOError, OSError):
                logger.warning(f"Thumbnail cannot load image: {original_filepath}")
                return thumbnail_filepath, (width, height)

            image = self._create_thumbnail(
                image, thumbnail_size, crop, background=background
            )

            raw_data = self.get_raw_data(image, **options)
            storage.save(thumbnail_filepath, raw_data)

            return thumbnail_filepath, (width, height)
        finally:
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
            tuple(map(lambda x: (x[0] - x[1]) / 2, zip(size, original_image.size))),
        )

        return image

    def _get_format(self, image, **options):
        if options.get("format"):
            return options.get("format")
        if image.format:
            return image.format

        return "JPEG"

    def _create_thumbnail(self, image, size, crop="fit", background=None):
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:  # pylint: disable=raise-missing-from
            resample = Image.ANTIALIAS

        if crop == "fit":
            image = ImageOps.fit(image, size, resample)
        else:
            image = image.copy()
            image.thumbnail(size, resample=resample)

        if background is not None:
            image = self.background(image)

        image = self.colormode(image)

        return image
