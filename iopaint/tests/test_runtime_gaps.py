"""runtime.py 缺口补测：设备探测降级、GPU 枚举失败分支、显存查询与释放、
next_bigger_gpu 边界、setup_model_dir 建目录，以及 import 期包版本探测的
PackageNotFoundError 分支（用 reload + 临时打桩 importlib.metadata 覆盖）。

所有测试都不碰真实 CUDA：torch.cuda.* 一律 monkeypatch 打桩。
"""
import importlib
import importlib.metadata as md
import os
from types import SimpleNamespace

import pytest
import torch

import iopaint.runtime as runtime
from iopaint.runtime import (
    GpuInfo,
    check_device,
    cuda_device_index,
    describe_gpus,
    dump_gpu_info,
    list_gpus,
    next_bigger_gpu,
    query_free_mb,
    setup_model_dir,
)
from iopaint.schema import Device


class TestDumpGpuInfo:
    def test_without_cuda_device(self, capsys, monkeypatch):
        monkeypatch.setattr("iopaint.runtime.list_gpus", lambda query_free=False: [])

        dump_gpu_info()

        assert "no CUDA device detected" in capsys.readouterr().out


class TestCheckDevice:
    def test_cuda_on_macos_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        assert check_device(Device.cuda) == Device.cpu

    def test_cuda_unavailable_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert check_device(Device.cuda) == Device.cpu

    def test_mps_unavailable_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert check_device(Device.mps) == Device.cpu

    def test_mps_available_stays_mps(self, monkeypatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert check_device(Device.mps) == Device.mps


class TestQueryFreeMb:
    def test_success_converts_bytes_to_mb(self, monkeypatch):
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda idx: (2048 << 20, 8192 << 20)
        )
        assert query_free_mb(0) == 2048

    def test_failure_returns_none(self, monkeypatch):
        def boom(idx):
            raise RuntimeError("no cuda context")

        monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
        assert query_free_mb(0) is None


class TestListGpus:
    def test_without_cuda_returns_empty(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert list_gpus() == []

    def test_properties_failure_yields_zero_total(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

        def boom(idx):
            raise RuntimeError("props unavailable")

        monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "FakeGPU")

        assert list_gpus() == [GpuInfo(0, "FakeGPU", 0)]

    def test_name_failure_falls_back_to_index(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda idx: SimpleNamespace(total_memory=8 << 30),
        )

        def boom(idx):
            raise RuntimeError("name unavailable")

        monkeypatch.setattr(torch.cuda, "get_device_name", boom)

        gpus = list_gpus()
        assert gpus[0].name == "cuda:0"
        assert gpus[0].total_mb == 8192

    def test_enumeration_failure_returns_empty(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def boom():
            raise RuntimeError("cuda driver gone")

        monkeypatch.setattr(torch.cuda, "device_count", boom)
        assert list_gpus() == []


class TestDescribeGpus:
    def test_without_cuda(self, monkeypatch):
        monkeypatch.setattr("iopaint.runtime.list_gpus", lambda query_free=False: [])
        assert describe_gpus() == "no CUDA device detected"


class TestEnvGpuId:
    def test_non_strict_returns_none_for_garbage(self, monkeypatch):
        monkeypatch.setenv("IOPaint_GPU", "not-a-number")
        assert runtime._env_gpu_id(strict=False) is None


class TestCudaDeviceIndex:
    def test_invalid_suffix_returns_zero(self):
        assert cuda_device_index("cuda:abc") == 0


class TestReleaseMemory:
    def test_without_cuda_device_skips_empty_cache(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

        runtime._release_memory(torch.device("cuda:0"))

        assert calls == []

    def test_none_device_calls_empty_cache_directly(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))

        runtime._release_memory()

        assert calls == ["empty"]

    def test_swallows_empty_cache_error(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def boom():
            raise RuntimeError("driver error")

        monkeypatch.setattr(torch.cuda, "empty_cache", boom)

        runtime._release_memory()  # 不抛异常即可


class TestNextBiggerGpu:
    def test_no_gpus_returns_none(self, monkeypatch):
        monkeypatch.setattr("iopaint.runtime.list_gpus", lambda query_free=False: [])
        assert next_bigger_gpu([torch.device("cuda:0")]) is None

    def test_last_tried_gpu_missing_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "iopaint.runtime.list_gpus",
            lambda query_free=False: [GpuInfo(0, "Fake", 8192, 4096)],
        )
        assert next_bigger_gpu([torch.device("cuda:7")]) is None


class TestSetupModelDir:
    def test_creates_missing_dir(self, tmp_path, monkeypatch):
        # 先登记旧值，测试结束由 monkeypatch 恢复环境变量
        for key in ("U2NET_HOME", "XDG_CACHE_HOME"):
            monkeypatch.setenv(key, os.environ.get(key, ""))

        new_dir = tmp_path / "models"
        assert not new_dir.exists()

        result = setup_model_dir(new_dir)

        assert new_dir.is_dir()
        assert result == new_dir.absolute()
        assert os.environ["U2NET_HOME"] == str(new_dir.absolute())
        assert os.environ["XDG_CACHE_HOME"] == str(new_dir.absolute())


def test_missing_package_version_recorded_as_na(monkeypatch):
    """import 期某个候选包没装时版本记 "N/A"（runtime.py:45-46）。

    本机候选包装齐了，直接 import 走不到 except 分支，所以用 reload +
    临时打桩 importlib.metadata.version 覆盖。reload 共享 module dict，
    打的桩对 re-executed 的版本探测循环同样生效。
    """
    real_version = md.version

    def fake_version(name):
        if name == "rembg":
            raise md.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(md, "version", fake_version)
    try:
        importlib.reload(runtime)
        assert runtime._package_versions["rembg"] == "N/A"
        # 其余包照常记录
        assert runtime._package_versions["torch"] == real_version("torch")
    finally:
        monkeypatch.undo()  # 撤销打桩
        importlib.reload(runtime)  # 恢复真实版本

    try:
        expected = real_version("rembg")
    except md.PackageNotFoundError:
        expected = "N/A"
    assert runtime._package_versions["rembg"] == expected
