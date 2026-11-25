# Copy from https://github.com/silentsokolov/flask-thumbnails/blob/master/flask_thumbnails/storage_backends.py
import errno
import os
import tempfile
from abc import ABC, abstractmethod


class BaseStorageBackend(ABC):
    def __init__(self, app=None):
        self.app = app

    @abstractmethod
    def read(self, filepath, mode="rb", **kwargs):
        raise NotImplementedError

    @abstractmethod
    def exists(self, filepath):
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath, data):
        raise NotImplementedError


class FilesystemStorageBackend(BaseStorageBackend):
    def read(self, filepath, mode="rb", **kwargs):
        with open(filepath, mode) as f:  # pylint: disable=unspecified-encoding
            return f.read()

    def exists(self, filepath):
        # 目录不该冒充缩略图：os.path.exists 对目录也返回 True，会让命中路径
        # 把一个同名目录当合法缩略图交给 FileResponse（读取方向与 save() 的
        # isdir 检查对称）。
        return os.path.isfile(filepath)

    def save(self, filepath, data):
        directory = os.path.dirname(filepath)

        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        if not os.path.isdir(directory):
            raise IOError("{} is not a directory".format(directory))

        # 原子写：先写同目录临时文件再 os.replace 落位。最终路径永远要么是
        # 旧的好图、要么不存在，绝不半截 —— 读侧 get_thumbnail 的命中只做
        # exists 校验，一旦写坏会一直被反复返回（源文件 mtime/size 不变时
        # generate_filename 也不会换名自愈）。
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, filepath)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
