# Copy from: https://github.com/silentsokolov/flask-thumbnails/blob/master/flask_thumbnails/utils.py
import hashlib
import mimetypes
from pathlib import Path

from typing import Union


def generate_filename(directory: Path, original_filename, *options, ext: str = "jpg") -> str:
    text = str(directory.absolute()) + original_filename
    for v in options:
        text += "%s" % v
    md5_hash = hashlib.md5()
    md5_hash.update(text.encode("utf-8"))
    return md5_hash.hexdigest() + "." + ext


# Magic bytes of the image formats we serve; used so Content-Type follows the
# actual bytes on disk, not a (possibly renamed) filename extension.
_MAGIC_MEDIA_TYPES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


def sniff_media_type(file_path: Union[Path, str]) -> str:
    """Best-effort media type: magic bytes first, filename extension last."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(16)
    except OSError:
        head = b""

    for magic, media_type in _MAGIC_MEDIA_TYPES:
        if head.startswith(magic):
            return media_type
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"

    guessed, _ = mimetypes.guess_type(str(file_path))
    return guessed or "application/octet-stream"


def parse_size(size):
    if isinstance(size, int):
        # If the size parameter is a single number, assume square aspect.
        return [size, size]

    if isinstance(size, (tuple, list)):
        if len(size) == 1:
            # If single value tuple/list is provided, exand it to two elements
            return size + type(size)(size)
        return size

    try:
        thumbnail_size = [int(x) for x in size.lower().split("x", 1)]
    except ValueError:
        raise ValueError(  # pylint: disable=raise-missing-from
            "Bad thumbnail size format. Valid format is INTxINT."
        )

    if len(thumbnail_size) == 1:
        # If the size parameter only contains a single integer, assume square aspect.
        thumbnail_size.append(thumbnail_size[0])

    return thumbnail_size


def aspect_to_string(size):
    if isinstance(size, str):
        return size

    return "x".join(map(str, size))


IMG_SUFFIX = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def glob_img(p: Union[Path, str], recursive: bool = False):
    p = Path(p)
    if p.is_file() and p.suffix in IMG_SUFFIX:
        yield p
    else:
        if recursive:
            files = Path(p).glob("**/*.*")
        else:
            files = Path(p).glob("*.*")

        for it in files:
            if it.suffix not in IMG_SUFFIX:
                continue
            yield it
