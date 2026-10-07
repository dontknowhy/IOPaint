"""GPU 选择 + 显存不足自动回退。

不下载权重、不碰网络，纯逻辑，CPU-only 的机器上也能跑。
"""
import pytest
import torch

from iopaint.model.utils import get_torch_dtype
from iopaint.model_manager import ModelManager
from iopaint.runtime import (
    GpuInfo,
    cuda_device_index,
    device_str,
    is_out_of_memory,
    load_with_fallback,
    next_bigger_gpu,
    parse_device_spec,
    resolve_gpu_id,
    to_torch_device,
)
from iopaint.schema import Device


def _gpus(*specs):
    return [
        GpuInfo(i, f"FakeGPU-{i}", total, free)
        for i, (total, free) in enumerate(specs)
    ]


def _fake_gpus(monkeypatch, *specs):
    # 签名要跟真身对齐：next_bigger_gpu 会带 query_free=True 调用
    monkeypatch.setattr(
        "iopaint.runtime.list_gpus", lambda query_free=False: _gpus(*specs)
    )


# ---------------------------------------------------------------------------
# --device / --gpu-id 解析
# ---------------------------------------------------------------------------
class TestParseDeviceSpec:
    @pytest.mark.parametrize(
        "spec,expected_device,expected_gpu_id",
        [
            ("cpu", Device.cpu, None),
            ("cuda", Device.cuda, None),
            ("mps", Device.mps, None),
            ("CUDA", Device.cuda, None),
            ("  cuda:1  ", Device.cuda, 1),
            ("CUDA:0", Device.cuda, 0),
        ],
    )
    def test_valid(self, spec, expected_device, expected_gpu_id):
        assert parse_device_spec(spec) == (expected_device, expected_gpu_id)

    @pytest.mark.parametrize(
        "spec", ["gpu", "cuda:x", "cuda:", "cuda:-1", "cpu:1", "", "cuda:1.5"]
    )
    def test_invalid(self, spec):
        with pytest.raises(ValueError):
            parse_device_spec(spec)

    def test_gpu_id_conflicts_with_device_index(self):
        with pytest.raises(ValueError, match="conflicts"):
            parse_device_spec("cuda:1", gpu_id=2)

    def test_gpu_id_matching_device_index_is_ok(self):
        assert parse_device_spec("cuda:1", gpu_id=1) == (Device.cuda, 1)

    @pytest.mark.parametrize("spec", ["cpu", "mps"])
    def test_gpu_index_rejected_for_non_cuda(self, spec):
        with pytest.raises(ValueError, match="only valid with device=cuda"):
            parse_device_spec(spec, gpu_id=0)

    def test_env_var_selects_gpu(self, monkeypatch):
        monkeypatch.setenv("IOPaint_GPU", "1")
        assert parse_device_spec("cuda") == (Device.cuda, 1)

    def test_explicit_index_wins_over_env_var(self, monkeypatch):
        monkeypatch.setenv("IOPaint_GPU", "1")
        assert parse_device_spec("cuda:0") == (Device.cuda, 0)

    def test_env_var_ignored_when_not_cuda(self, monkeypatch):
        monkeypatch.setenv("IOPaint_GPU", "1")
        assert parse_device_spec("cpu") == (Device.cpu, None)

    def test_env_var_invalid(self, monkeypatch):
        monkeypatch.setenv("IOPaint_GPU", "abc")
        with pytest.raises(ValueError, match="IOPaint_GPU"):
            parse_device_spec("cuda")


class TestResolveGpuId:
    def test_in_range(self, monkeypatch):
        _fake_gpus(monkeypatch, (1989, 1383), (8109, 8018))
        assert resolve_gpu_id(Device.cuda, 1) == 1

    def test_out_of_range(self, monkeypatch):
        _fake_gpus(monkeypatch, (1989, 1383), (8109, 8018))
        with pytest.raises(ValueError, match="out of range"):
            resolve_gpu_id(Device.cuda, 5)

    def test_no_cuda_device_is_silent(self, monkeypatch):
        monkeypatch.setattr(
            "iopaint.runtime.list_gpus", lambda query_free=False: []
        )
        assert resolve_gpu_id(Device.cuda, 3) is None

    @pytest.mark.parametrize("device", [Device.cpu, Device.mps])
    def test_ignored_for_non_cuda(self, device):
        assert resolve_gpu_id(device, 3) is None

    def test_ignored_when_index_is_none(self, monkeypatch):
        _fake_gpus(monkeypatch, (1989, 1383))
        assert resolve_gpu_id(Device.cuda, None) is None


class TestToTorchDevice:
    def test_cpu_and_mps_ignore_gpu_id(self):
        assert to_torch_device(Device.cpu) == torch.device("cpu")
        assert to_torch_device(Device.mps) == torch.device("mps")

    def test_cuda_without_index(self):
        assert str(to_torch_device(Device.cuda)) == "cuda"

    def test_cuda_with_index(self):
        assert str(to_torch_device(Device.cuda, 2)) == "cuda:2"


# ---------------------------------------------------------------------------
# device 表示的归一化
# ---------------------------------------------------------------------------
class TestDeviceStr:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (Device.cuda, "cuda"),
            (Device.mps, "mps"),
            (torch.device("cuda"), "cuda"),
            (torch.device("cuda:1"), "cuda:1"),
            (torch.device("cpu"), "cpu"),
            ("cuda", "cuda"),
        ],
    )
    def test_device_str(self, value, expected):
        # Python 3.11+ 里 str(Device.cuda) == "Device.cuda"，必须走这条归一化
        assert device_str(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (torch.device("cuda:2"), 2),
            (torch.device("cuda"), 0),
            (Device.cuda, 0),
            (Device.cpu, 0),
            (torch.device("cpu"), 0),
        ],
    )
    def test_cuda_device_index(self, value, expected):
        assert cuda_device_index(value) == expected


class TestGetTorchDtype:
    @pytest.mark.parametrize(
        "device,use_gpu",
        [
            (torch.device("cuda"), True),
            (torch.device("cuda:1"), True),
            (Device.cuda, True),
            (torch.device("cpu"), False),
            (Device.cpu, False),
            (torch.device("mps"), False),
        ],
    )
    def test_default_is_fp16_on_cuda(self, device, use_gpu):
        use_fp16, dtype = get_torch_dtype(device, no_half=False)
        assert use_fp16 is use_gpu
        assert dtype == (torch.float16 if use_gpu else torch.float32)

    @pytest.mark.parametrize("device", [torch.device("cuda:1"), Device.cuda])
    def test_no_half_forces_fp32(self, device):
        use_fp16, dtype = get_torch_dtype(device, no_half=True)
        assert use_fp16 is True
        assert dtype == torch.float32


# ---------------------------------------------------------------------------
# 显存不足的判定与选卡
# ---------------------------------------------------------------------------
class TestIsOutOfMemory:
    def test_message_matched(self):
        assert is_out_of_memory(
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        )
        assert is_out_of_memory(RuntimeError("HIP out of memory"))

    def test_native_error_class(self):
        cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if cls is not None:
            assert is_out_of_memory(cls("boom"))

    @pytest.mark.parametrize(
        "exc", [ValueError("bad weights"), RuntimeError("cudnn error")]
    )
    def test_other_errors_are_not_oom(self, exc):
        assert not is_out_of_memory(exc)


class TestNextBiggerGpu:
    def test_empty_tried(self):
        assert next_bigger_gpu([]) is None

    def test_picks_bigger_gpu(self, monkeypatch):
        _fake_gpus(monkeypatch, (1989, 1383), (8109, 8018))
        assert next_bigger_gpu([torch.device("cuda")]) == 1

    def test_same_total_more_free_counts_as_bigger(self, monkeypatch):
        _fake_gpus(monkeypatch, (8109, 100), (8109, 7000))
        assert next_bigger_gpu([torch.device("cuda")]) == 1

    def test_smaller_gpu_is_skipped(self, monkeypatch):
        _fake_gpus(monkeypatch, (8109, 7000), (1989, 1383))
        assert next_bigger_gpu([torch.device("cuda")]) is None

    def test_already_tried_gpus_are_excluded(self, monkeypatch):
        _fake_gpus(monkeypatch, (1989, 1383), (8109, 8018))
        tried = [torch.device("cuda"), torch.device("cuda:1")]
        assert next_bigger_gpu(tried) is None

    def test_picks_largest_first(self, monkeypatch):
        _fake_gpus(monkeypatch, (1989, 1383), (8109, 8018), (24576, 20000))
        assert next_bigger_gpu([torch.device("cuda")]) == 2


# ---------------------------------------------------------------------------
# 加载回退链路
# ---------------------------------------------------------------------------
class TestLoadWithFallback:
    def test_no_oom_uses_preferred_device(self):
        model, device, notices = load_with_fallback(
            lambda d: ("model", d), torch.device("cuda")
        )
        assert model == ("model", torch.device("cuda"))
        assert device.type == "cuda"
        assert notices == []

    def test_oom_switches_to_bigger_gpu(self, monkeypatch):
        _fake_gpus(monkeypatch, (1989, 1383), (8109, 8018))
        tried = []

        def loader(device):
            tried.append(str(device))
            if device.type == "cuda" and (device.index or 0) == 0:
                raise RuntimeError("CUDA out of memory")
            return "ok"

        model, device, notices = load_with_fallback(loader, torch.device("cuda"))
        assert model == "ok"
        assert str(device) == "cuda:1"
        assert tried == ["cuda", "cuda:1"]
        assert len(notices) == 1
        assert "cuda:1" in notices[0]

    def test_all_gpus_oom_falls_back_to_cpu(self, monkeypatch):
        _fake_gpus(monkeypatch, (1989, 1383), (8109, 8018))

        def loader(device):
            if device.type == "cuda":
                raise RuntimeError("CUDA out of memory")
            return "cpu-model"

        model, device, notices = load_with_fallback(loader, torch.device("cuda"))
        assert model == "cpu-model"
        assert device.type == "cpu"
        assert len(notices) == 2
        assert "CPU" in notices[-1]

    def test_single_gpu_oom_goes_straight_to_cpu(self, monkeypatch):
        _fake_gpus(monkeypatch, (1989, 1383))

        def loader(device):
            if device.type == "cuda":
                raise RuntimeError("CUDA out of memory")
            return "cpu-model"

        model, device, notices = load_with_fallback(loader, torch.device("cuda"))
        assert model == "cpu-model"
        assert device.type == "cpu"
        assert len(notices) == 1

    def test_non_oom_error_is_not_retried(self):
        def loader(device):
            raise ValueError("bad weights")

        with pytest.raises(ValueError, match="bad weights"):
            load_with_fallback(loader, torch.device("cuda"))

    def test_cpu_failure_raises(self):
        def loader(device):
            raise RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError):
            load_with_fallback(loader, torch.device("cpu"))


# ---------------------------------------------------------------------------
# 回退时必须真正释放半路失败的模型（否则失败的卡上会一直挂着一两个 G）
# ---------------------------------------------------------------------------
class TestFallbackReleasesMemory:
    def test_release_memory_runs_after_exception_is_gone(self, monkeypatch):
        """_release_memory 必须在 except 块外面调用。

        在 except 里面跑时 `e.__traceback__` 还挂着 loader → init_model 的帧 →
        self → 只挪了一半的 pipeline，`gc.collect()` 只是空转，那一个多 G 永远还不回去
        （nvidia-smi 里看着就是「失败的卡莫名其妙被占着」）。
        """
        import sys

        import iopaint.runtime as runtime

        _fake_gpus(monkeypatch, (1989, 1383), (8109, 8018))
        seen = {}
        original = runtime._release_memory

        def spy(device=None):
            seen["exc_during_release"] = sys.exc_info()[0]
            seen["device"] = device
            return original(device)

        monkeypatch.setattr(runtime, "_release_memory", spy)

        def loader(device):
            if device.type == "cuda":
                raise RuntimeError("CUDA out of memory")
            return "cpu-model"

        load_with_fallback(loader, torch.device("cuda"))
        assert seen["exc_during_release"] is None, "清理时异常还活着，traceback 会拽住半成品模型"
        # empty_cache 要切回失败的那张卡，否则 current device 已经是下一块卡
        assert str(seen["device"]).split(":")[0] == "cuda"

    def test_half_built_model_is_reachable_after_fallback(self, monkeypatch):
        """loader 里建到一半、自引用成环的对象，回退返回后必须已经没人引用。"""
        import gc
        import weakref

        _fake_gpus(monkeypatch, (1989, 1383), (8109, 8018))
        refs = []

        class Node:
            pass

        def loader(device):
            if device.type == "cuda":
                # 模拟 SD.init_model 里 self.model 建到一半、模块互相引用的 pipeline
                root = Node()
                root.self_ref = root
                root.child = Node()
                root.child.parent = root
                refs.append(weakref.ref(root))
                gc.collect()  # 先把它推进 gen1，避免被后续偶发的 gen0 回收误伤
                raise RuntimeError("CUDA out of memory")
            return "cpu-model"

        load_with_fallback(loader, torch.device("cuda"))
        assert refs[0]() is None, "半成品模型还被引用着（traceback 没切断，或清理没跑到）"


class TestGpuEnumeration:
    """启动时枚举显卡不能顺手把每张卡都建上 CUDA 上下文（实测每张 ~40MB）。"""

    @staticmethod
    def _patch_fake_cuda(monkeypatch):
        import torch

        class Props:
            def __init__(self, total):
                self.total_memory = total

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: f"FakeGPU-{i}")
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda i: Props(8 << 30)
        )
        calls = []

        def spy(index):
            calls.append(index)
            return 1234

        monkeypatch.setattr("iopaint.runtime.query_free_mb", spy)
        return calls

    def test_default_does_not_query_free_memory(self, monkeypatch):
        calls = self._patch_fake_cuda(monkeypatch)
        from iopaint.runtime import list_gpus

        gpus = list_gpus()
        assert calls == [], "默认枚举不该查空闲显存 —— 那会给每张卡建 CUDA 上下文"
        assert [it.free_mb for it in gpus] == [None, None]
        assert [it.total_mb for it in gpus] == [8192, 8192]

    def test_query_free_opt_in(self, monkeypatch):
        calls = self._patch_fake_cuda(monkeypatch)
        from iopaint.runtime import list_gpus

        gpus = list_gpus(query_free=True)
        assert calls == [0, 1]
        assert [it.free_mb for it in gpus] == [1234, 1234]

    def test_describe_hides_unknown_free(self):
        from iopaint.runtime import GpuInfo

        assert GpuInfo(1, "FakeGPU-1", 8109).describe() == "GPU1 FakeGPU-1 (total 8109 MB)"
        assert (
            GpuInfo(1, "FakeGPU-1", 8109, 8018).describe()
            == "GPU1 FakeGPU-1 (total 8109 MB, free 8018 MB)"
        )
        assert GpuInfo(1, "FakeGPU-1", 8109).room == (8109, -1)

    def test_dump_gpu_info_only_queries_target_card(self, monkeypatch, capsys):
        from iopaint.runtime import dump_gpu_info

        monkeypatch.setattr(
            "iopaint.runtime.list_gpus",
            lambda query_free=False: _gpus((1989, None), (8109, None)),
        )
        calls = []
        monkeypatch.setattr(
            "iopaint.runtime.query_free_mb", lambda i: (calls.append(i), 1313)[1]
        )

        # 指定 GPU1 → 只查 GPU1
        dump_gpu_info(Device.cuda, 1)
        assert calls == [1]
        assert "free 1313 MB" in capsys.readouterr().out

        # 非 cuda / 未指定 → 一张都不查（零上下文）
        dump_gpu_info(Device.cpu, None)
        assert calls == [1]
        assert "free" not in capsys.readouterr().out


class TestGetTorchDtype:
    """get_torch_dtype 的 device 判定。

    这里改过一个真 bug：原来拿 `str(device)` 去比 `"cuda"`，Python 3.11+ 里
    `str(Device.cuda) == "Device.cuda"`、`str(torch.device("cuda:1")) == "cuda:1"`，
    两个都永远不等于 `"cuda"` —— 枚举和 `cuda:N` 会静默退回 fp32（更慢、更吃显存）。
    """

    @pytest.mark.parametrize(
        "device", [torch.device("cuda"), torch.device("cuda:1"), Device.cuda]
    )
    def test_cuda_defaults_to_fp16(self, device):
        use_gpu, dtype = get_torch_dtype(device, no_half=False)
        assert use_gpu is True
        assert dtype == torch.float16

    @pytest.mark.parametrize(
        "device", [torch.device("cpu"), Device.cpu, torch.device("mps"), Device.mps]
    )
    def test_non_cuda_stays_fp32(self, device):
        use_gpu, dtype = get_torch_dtype(device, no_half=False)
        assert dtype == torch.float32
        # use_gpu 只喂给 cpu_offload 分支，非 cuda 必须是 False
        assert use_gpu is False

    @pytest.mark.parametrize("device", [torch.device("cuda:1"), Device.cuda])
    def test_no_half_forces_fp32_but_still_gpu(self, device):
        use_gpu, dtype = get_torch_dtype(device, no_half=True)
        assert use_gpu is True
        assert dtype == torch.float32


class _RecordingPipe:
    """假 pipeline，只记录 enable_sequential_cpu_offload 收到的 gpu_id。"""

    def __init__(self):
        self.offload_gpu_ids = []
        self.moved_to = None

    def enable_sequential_cpu_offload(self, gpu_id=None):
        self.offload_gpu_ids.append(gpu_id)

    def to(self, device):
        self.moved_to = device
        return self


class TestCpuOffloadUsesDeviceIndex:
    """`--cpu-offload` + `cuda:N`：gpu_id 必须是 N。

    这些地方原来全是硬编码 `gpu_id=0`，`--device cuda:1 --cpu-offload` 会把模型
    offload 到另一张卡上（甚至直接和默认卡抢显存）。
    """

    @staticmethod
    def _load_sd(monkeypatch, device, **kwargs):
        from iopaint.model import sd as sd_mod

        pipe = _RecordingPipe()

        class FakePipeline:
            @staticmethod
            def from_pretrained(*args, **kw):
                return pipe

        monkeypatch.setattr(
            "diffusers.pipelines.stable_diffusion.StableDiffusionInpaintPipeline",
            FakePipeline,
        )

        model = object.__new__(sd_mod.SD)
        model.model_info = type("Info", (), {"is_single_file_diffusers": False})()
        model.model_id_or_path = "fake/model"
        model.init_model(device, **kwargs)
        return model, pipe

    @pytest.mark.parametrize(
        "device,expected",
        [
            (torch.device("cuda"), 0),
            (torch.device("cuda:1"), 1),
            (Device.cuda, 0),
        ],
    )
    def test_offload_gpu_id_matches_device(self, monkeypatch, device, expected):
        _, pipe = self._load_sd(monkeypatch, device, cpu_offload=True)
        assert pipe.offload_gpu_ids == [expected]
        assert pipe.moved_to is None, "cpu_offload 分支不该再 .to(device)"

    def test_without_offload_it_moves_to_the_device(self, monkeypatch):
        _, pipe = self._load_sd(monkeypatch, torch.device("cuda:1"))
        assert pipe.offload_gpu_ids == []
        assert str(pipe.moved_to) == "cuda:1"

    @pytest.mark.parametrize("device", [torch.device("cpu"), Device.cpu])
    def test_offload_never_turns_on_for_cpu(self, monkeypatch, device):
        # use_gpu=False 时不能调 enable_sequential_cpu_offload，否则连 CPU 都跑不起来
        _, pipe = self._load_sd(monkeypatch, device, cpu_offload=True)
        assert pipe.offload_gpu_ids == []


# ---------------------------------------------------------------------------
# ModelManager 层：记录回退提示
# ---------------------------------------------------------------------------
def _bare_manager(device):
    """绕开 __init__（scan_models 要扫模型目录），只装 _load 需要的字段。"""
    mm = ModelManager.__new__(ModelManager)
    mm.name = "cv2"
    mm.device = device
    mm.device_notices = []
    mm.kwargs = {}
    return mm


class TestModelManagerFallback:
    @pytest.mark.parametrize(
        "notices,expected",
        [
            ([], None),
            (["a"], "a"),
            (["a", "b", "a"], "a; b"),
        ],
    )
    def test_device_notice_dedup(self, notices, expected):
        mm = _bare_manager(torch.device("cpu"))
        mm.device_notices = list(notices)
        assert mm.device_notice == expected

    def test_load_records_notice_once(self, monkeypatch):
        _fake_gpus(monkeypatch, (1989, 1383), (8109, 8018))
        mm = _bare_manager(torch.device("cuda"))

        def fake_init(name, device, **kwargs):
            if device.type == "cuda" and (device.index or 0) == 0:
                raise RuntimeError("CUDA out of memory")
            return f"model@{device}"

        mm.init_model = fake_init
        model, device = mm._load("cv2", torch.device("cuda"))
        assert model == "model@cuda:1"
        assert str(device) == "cuda:1"
        assert mm.device_notice is not None
        assert "cuda:1" in mm.device_notice

        # 再触发一次同样的回退，提示不能重复堆叠
        mm._load("cv2", torch.device("cuda"))
        assert len(mm.device_notices) == 1

    def test_load_failure_leaves_notices_untouched(self):
        mm = _bare_manager(torch.device("cuda"))
        mm.init_model = lambda name, device, **kwargs: (_ for _ in ()).throw(
            ValueError("bad weights")
        )
        with pytest.raises(ValueError):
            mm._load("cv2", torch.device("cuda"))
        assert mm.device_notices == []
