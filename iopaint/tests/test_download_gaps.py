"""download.py 缺口补测：cli_download_model 分派、单文件头检查、模型类型推断、
扫描缓存与 diffusers/converted 扫描的分类分支。

全部离线：safetensors/ckpt 用真实小文件（只写头，无权重语义），HF_HUB_CACHE 用
tmp 目录 + monkeypatch，下载分支只打桩 handle_from_pretrained_exceptions。
"""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from iopaint.const import ANYTEXT_NAME
from iopaint.download import (
    _inspect_single_file,
    cli_download_model,
    get_sd_model_type,
    get_sdxl_model_type,
    scan_converted_diffusers_models,
    scan_diffusers_models,
    scan_single_file_diffusion_models,
)
from iopaint.schema import ModelType


def _write_safetensors(path: Path, keys_shapes: dict):
    """写一个只有头与零张量的真 safetensors 文件。"""
    tensors = {k: torch.zeros(*shape) for k, shape in keys_shapes.items()}
    save_file(tensors, str(path))


def _write_ckpt(path: Path, keys_shapes: dict):
    tensors = {k: torch.zeros(*shape) for k, shape in keys_shapes.items()}
    torch.save(tensors, str(path))


class TestCliDownloadModel:
    def test_erase_model_calls_model_download(self, monkeypatch):
        import iopaint.model as model_pkg

        calls = []

        class FakeErase:
            is_erase_model = True

            @classmethod
            def download(cls):
                calls.append("erase")

        monkeypatch.setitem(model_pkg.models, "fake-erase-model", FakeErase)
        cli_download_model("fake-erase-model")
        assert calls == ["erase"]

    def test_anytext_calls_model_download(self, monkeypatch):
        import iopaint.model as model_pkg

        calls = []

        class FakeAnyText:
            is_erase_model = False

            @classmethod
            def download(cls):
                calls.append(ANYTEXT_NAME)

        monkeypatch.setitem(model_pkg.models, ANYTEXT_NAME, FakeAnyText)
        cli_download_model(ANYTEXT_NAME)
        assert calls == [ANYTEXT_NAME]

    def test_huggingface_model_delegates_to_diffusion_download(self, monkeypatch):
        calls = []

        def fake_handle(fn, **kwargs):
            calls.append((fn, kwargs))
            return "/fake/downloaded"

        monkeypatch.setattr(
            "iopaint.model.utils.handle_from_pretrained_exceptions", fake_handle
        )
        cli_download_model("org/some-model")
        assert len(calls) == 1
        fn, kwargs = calls[0]
        # 传入的是 DiffusionPipeline.download（未绑定函数），参数为 fp16 variant
        assert fn.__qualname__.endswith("DiffusionPipeline.download")
        assert kwargs == {"pretrained_model_name": "org/some-model", "variant": "fp16"}


class TestInspectSingleFile:
    def test_safetensors_unet_conv_in(self, tmp_path):
        f = tmp_path / "model.safetensors"
        _write_safetensors(f, {"unet.conv_in.weight": (4, 4, 3, 3)})
        assert _inspect_single_file(str(f)) == (4, False)

    def test_safetensors_comfyui_key_and_sdxl_marker(self, tmp_path):
        f = tmp_path / "model.safetensors"
        _write_safetensors(
            f,
            {
                "model.diffusion_model.input_blocks.0.0.weight": (4, 9, 3, 3),
                "conditioner.embeddings.0": (2,),
            },
        )
        assert _inspect_single_file(str(f)) == (9, True)

    def test_safetensors_corrupt_returns_none(self, tmp_path):
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"not a safetensors file")
        assert _inspect_single_file(str(f)) is None

    def test_ckpt_unet_conv_in(self, tmp_path):
        f = tmp_path / "model.ckpt"
        _write_ckpt(f, {"unet.conv_in.weight": (4, 4, 3, 3)})
        assert _inspect_single_file(str(f)) == (4, False)

    def test_ckpt_sdxl_marker(self, tmp_path):
        f = tmp_path / "model.ckpt"
        _write_ckpt(
            f,
            {
                "model.diffusion_model.input_blocks.0.0.weight": (4, 9, 3, 3),
                "conditioner.embeddings.0": (2,),
            },
        )
        assert _inspect_single_file(str(f)) == (9, True)

    def test_ckpt_corrupt_returns_none(self, tmp_path):
        f = tmp_path / "model.ckpt"
        f.write_bytes(b"garbage not a pickle")
        assert _inspect_single_file(str(f)) is None

    def test_ckpt_unknown_keys_gives_no_in_channels(self, tmp_path):
        f = tmp_path / "model.ckpt"
        _write_ckpt(f, {"some.other.weight": (2, 2)})
        assert _inspect_single_file(str(f)) == (None, False)


class TestGetSdModelType:
    def test_inpaint_name_short_circuits(self, tmp_path):
        f = tmp_path / "sd-inpaint.safetensors"
        f.write_bytes(b"")  # 名字命中即返回，不读文件
        assert get_sd_model_type(str(f)) == ModelType.DIFFUSERS_SD_INPAINT

    def test_corrupt_file_returns_none(self, tmp_path):
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"junk")
        assert get_sd_model_type(str(f)) is None

    def test_sdxl_layout_returns_none(self, tmp_path):
        f = tmp_path / "model.safetensors"
        _write_safetensors(f, {"conditioner.embeddings.0": (2,)})
        assert get_sd_model_type(str(f)) is None

    def test_nine_channels_is_inpaint(self, tmp_path):
        f = tmp_path / "model.safetensors"
        _write_safetensors(f, {"unet.conv_in.weight": (4, 9, 3, 3)})
        assert get_sd_model_type(str(f)) == ModelType.DIFFUSERS_SD_INPAINT

    def test_four_channels_is_sd(self, tmp_path):
        f = tmp_path / "model.safetensors"
        _write_safetensors(f, {"unet.conv_in.weight": (4, 4, 3, 3)})
        assert get_sd_model_type(str(f)) == ModelType.DIFFUSERS_SD

    def test_other_channels_returns_none(self, tmp_path):
        f = tmp_path / "model.safetensors"
        _write_safetensors(f, {"unet.conv_in.weight": (4, 8, 3, 3)})
        assert get_sd_model_type(str(f)) is None


class TestGetSdxlModelType:
    def test_inpaint_in_path_short_circuits(self, tmp_path):
        f = tmp_path / "sdxl-inpaint.safetensors"
        f.write_bytes(b"")
        assert get_sdxl_model_type(str(f)) == ModelType.DIFFUSERS_SDXL_INPAINT

    def test_corrupt_file_returns_none(self, tmp_path):
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"junk")
        assert get_sdxl_model_type(str(f)) is None

    def test_non_sdxl_layout_returns_none(self, tmp_path):
        f = tmp_path / "model.safetensors"
        _write_safetensors(f, {"unet.conv_in.weight": (4, 4, 3, 3)})
        assert get_sdxl_model_type(str(f)) is None

    def test_nine_channels_is_sdxl_inpaint(self, tmp_path):
        f = tmp_path / "model.safetensors"
        _write_safetensors(
            f,
            {"unet.conv_in.weight": (4, 9, 3, 3), "conditioner.e": (2,)},
        )
        assert get_sdxl_model_type(str(f)) == ModelType.DIFFUSERS_SDXL_INPAINT

    def test_four_channels_is_sdxl(self, tmp_path):
        f = tmp_path / "model.safetensors"
        _write_safetensors(
            f,
            {"unet.conv_in.weight": (4, 4, 3, 3), "conditioner.e": (2,)},
        )
        assert get_sdxl_model_type(str(f)) == ModelType.DIFFUSERS_SDXL

    def test_other_channels_returns_none(self, tmp_path):
        f = tmp_path / "model.safetensors"
        _write_safetensors(
            f,
            {"unet.conv_in.weight": (4, 8, 3, 3), "conditioner.e": (2,)},
        )
        assert get_sdxl_model_type(str(f)) is None


class TestScanSingleFileCache:
    def _make_dirs(self, cache_dir: Path):
        (cache_dir / "stable_diffusion").mkdir(parents=True)
        (cache_dir / "stable_diffusion_xl").mkdir(parents=True)

    def test_scan_classifies_and_writes_cache(self, tmp_path):
        sd = tmp_path / "stable_diffusion"
        xl = tmp_path / "stable_diffusion_xl"
        self._make_dirs(tmp_path)
        _write_safetensors(sd / "sd.safetensors", {"unet.conv_in.weight": (4, 4, 3, 3)})
        _write_safetensors(
            sd / "inpaint.safetensors", {"unet.conv_in.weight": (4, 9, 3, 3)}
        )
        (sd / "broken.safetensors").write_bytes(b"junk")  # 坏文件 → 跳过
        (sd / "notes.txt").write_text("x")  # 非 ckpt 后缀 → 跳过
        _write_safetensors(
            xl / "xl.safetensors",
            {"unet.conv_in.weight": (4, 4, 3, 3), "conditioner.e": (2,)},
        )
        (xl / "broken2.ckpt").write_bytes(b"junk")  # sdxl 目录坏文件 → continue

        res = scan_single_file_diffusion_models(tmp_path)

        by_name = {m.name: m for m in res}
        assert by_name["sd.safetensors"].model_type == ModelType.DIFFUSERS_SD
        assert by_name["inpaint.safetensors"].model_type == ModelType.DIFFUSERS_SD_INPAINT
        assert by_name["xl.safetensors"].model_type == ModelType.DIFFUSERS_SDXL
        assert all(m.is_single_file_diffusers for m in res)
        assert all(Path(m.path).is_absolute() for m in res)
        assert "broken.safetensors" not in by_name
        assert "notes.txt" not in by_name
        # 缓存文件已写盘，且记录了分类结果
        sd_cache = json.loads((sd / "iopaint_cache.json").read_text(encoding="utf-8"))
        assert sd_cache["sd.safetensors"] == "diffusers_sd"
        xl_cache = json.loads((xl / "iopaint_cache.json").read_text(encoding="utf-8"))
        assert xl_cache["xl.safetensors"] == "diffusers_sdxl"

    def test_invalid_cache_file_falls_back_to_inspect(self, tmp_path):
        sd = tmp_path / "stable_diffusion"
        xl = tmp_path / "stable_diffusion_xl"
        self._make_dirs(tmp_path)
        (sd / "iopaint_cache.json").write_text("{ not json", encoding="utf-8")
        (xl / "iopaint_cache.json").write_text("[1, 2]", encoding="utf-8")  # 非 dict
        _write_safetensors(sd / "sd.safetensors", {"unet.conv_in.weight": (4, 4, 3, 3)})

        res = scan_single_file_diffusion_models(tmp_path)

        assert [m.name for m in res] == ["sd.safetensors"]

    def test_cache_hit_skips_header_inspect(self, tmp_path):
        """缓存命中时不再读文件头：坏文件凭缓存条目也能被列出。"""
        sd = tmp_path / "stable_diffusion"
        xl = tmp_path / "stable_diffusion_xl"
        self._make_dirs(tmp_path)
        (sd / "junk.safetensors").write_bytes(b"junk")
        (sd / "iopaint_cache.json").write_text(
            json.dumps({"junk.safetensors": "diffusers_sd"}), encoding="utf-8"
        )
        (xl / "junk2.ckpt").write_bytes(b"junk")
        (xl / "iopaint_cache.json").write_text(
            json.dumps({"junk2.ckpt": "diffusers_sdxl_inpaint"}), encoding="utf-8"
        )

        res = scan_single_file_diffusion_models(tmp_path)

        by_name = {m.name: m.model_type for m in res}
        assert by_name["junk.safetensors"] == ModelType.DIFFUSERS_SD
        assert by_name["junk2.ckpt"] == ModelType.DIFFUSERS_SDXL_INPAINT


class TestScanDiffusersClassification:
    """scan_diffusers_models 的 _class_name → ModelType 分类与去重。

    布局：HF_HUB_CACHE/models--x--y/snapshots/<rev>/model_index.json，
    名字取自 parent.parent.parent.name。
    """

    @staticmethod
    def _write_index(root: Path, repo: str, rev: str, content):
        p = root / repo / "snapshots" / rev
        p.mkdir(parents=True, exist_ok=True)
        body = content if isinstance(content, str) else json.dumps(content)
        (p / "model_index.json").write_text(body, encoding="utf-8")

    def test_powerpaint_is_other(self, tmp_path, monkeypatch):
        import huggingface_hub.constants as hf_const

        root = tmp_path / "hub"
        root.mkdir(exist_ok=True)
        self._write_index(
            root, "models--PowerPaint--v2", "abc",
            {"_class_name": "StableDiffusionPipeline"},
        )
        monkeypatch.setattr(hf_const, "HF_HUB_CACHE", str(root))
        res = scan_diffusers_models()
        assert [m.name for m in res] == ["PowerPaint/v2"]
        assert res[0].model_type == ModelType.DIFFUSERS_OTHER

    def test_sdxl_class_names(self, tmp_path, monkeypatch):
        import huggingface_hub.constants as hf_const

        root = tmp_path / "hub"
        root.mkdir(exist_ok=True)
        self._write_index(
            root, "models--a--xl", "r", {"_class_name": "StableDiffusionXLPipeline"}
        )
        self._write_index(
            root, "models--b--xl-inpaint", "r",
            {"_class_name": "StableDiffusionXLInpaintPipeline"},
        )
        monkeypatch.setattr(hf_const, "HF_HUB_CACHE", str(root))
        res = {m.name: m.model_type for m in scan_diffusers_models()}
        assert res["a/xl"] == ModelType.DIFFUSERS_SDXL
        assert res["b/xl-inpaint"] == ModelType.DIFFUSERS_SDXL_INPAINT

    def test_other_known_class_names(self, tmp_path, monkeypatch):
        import huggingface_hub.constants as hf_const

        root = tmp_path / "hub"
        root.mkdir(exist_ok=True)
        for repo, cls in [
            ("models--i--i2p", "StableDiffusionInstructPix2PixPipeline"),
            ("models--p--be", "PaintByExamplePipeline"),
            ("models--k--v22", "KandinskyV22InpaintPipeline"),
            ("models--t--anytext", "AnyText"),
        ]:
            self._write_index(root, repo, "r", {"_class_name": cls})
        monkeypatch.setattr(hf_const, "HF_HUB_CACHE", str(root))
        res = scan_diffusers_models()
        assert len(res) == 4
        assert all(m.model_type == ModelType.DIFFUSERS_OTHER for m in res)

    def test_unknown_class_skipped_and_duplicate_deduped(self, tmp_path, monkeypatch):
        import huggingface_hub.constants as hf_const

        root = tmp_path / "hub"
        root.mkdir(exist_ok=True)
        self._write_index(
            root, "models--x--y", "r1", {"_class_name": "StableDiffusionPipeline"}
        )
        # 同 repo 的第二个 snapshot（同名）→ 去重跳过
        self._write_index(
            root, "models--x--y", "r2", {"_class_name": "StableDiffusionPipeline"}
        )
        # 未知 class → 跳过
        self._write_index(
            root, "models--u--k", "r", {"_class_name": "SomethingNewPipeline"}
        )
        monkeypatch.setattr(hf_const, "HF_HUB_CACHE", str(root))
        res = scan_diffusers_models()
        assert [m.name for m in res] == ["x/y"]
        assert res[0].model_type == ModelType.DIFFUSERS_SD


class TestScanConvertedClassification:
    """_scan_converted_diffusers_models：名字取自 parent.name，四类 + 去重 + 未知跳过。"""

    @staticmethod
    def _write_index(dir_path: Path, content):
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "model_index.json").write_text(
            json.dumps(content), encoding="utf-8"
        )

    def test_all_class_names_and_unknown(self, tmp_path):
        sd = tmp_path / "stable_diffusion"
        xl = tmp_path / "stable_diffusion_xl"
        self._write_index(sd / "a-sd", {"_class_name": "StableDiffusionPipeline"})
        self._write_index(sd / "b-inpaint", {"_class_name": "StableDiffusionInpaintPipeline"})
        self._write_index(xl / "c-xl", {"_class_name": "StableDiffusionXLPipeline"})
        self._write_index(
            xl / "d-xl-inpaint", {"_class_name": "StableDiffusionXLInpaintPipeline"}
        )
        self._write_index(xl / "e-unknown", {"_class_name": "NopePipeline"})

        res = scan_converted_diffusers_models(tmp_path)

        by_name = {m.name: m.model_type for m in res}
        assert by_name == {
            "a-sd": ModelType.DIFFUSERS_SD,
            "b-inpaint": ModelType.DIFFUSERS_SD_INPAINT,
            "c-xl": ModelType.DIFFUSERS_SDXL,
            "d-xl-inpaint": ModelType.DIFFUSERS_SDXL_INPAINT,
        }
        # path 指向模型目录（absolute）
        assert all(Path(m.path).is_absolute() for m in res)

    def test_duplicate_name_deduped(self, tmp_path):
        sd = tmp_path / "stable_diffusion"
        self._write_index(sd / "dup", {"_class_name": "StableDiffusionPipeline"})
        # 嵌套目录里出现同名 parent → 第二个被去重
        self._write_index(sd / "outer" / "dup", {"_class_name": "StableDiffusionPipeline"})

        res = scan_converted_diffusers_models(tmp_path)

        assert [m.name for m in res] == ["dup"]
