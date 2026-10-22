"""iopaint/i18n.py 的语言包加载与 key 对齐（不导入 gradio，保持快测）。

和前端 `web_app/src/lib/i18n.test.ts` 对称：两边都遍历 `locales/*.json`，因此新增
一门语言会自动纳入校验，不需要手工维护清单。
"""
import os
import string
import subprocess
import sys
from typing import Any, Dict, Iterator, Set, Tuple

import pytest

from iopaint import const
from iopaint.i18n import (
    build_en_catalog,
    cli_text,
    load_catalogs,
    pick_locale,
    resolve_cli_locale,
    translate,
)


def _flatten(
    value: Dict[str, Any], prefix: str = ""
) -> Iterator[Tuple[str, str]]:
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            yield from _flatten(child, path)
        else:
            yield path, child


def test_load_catalogs_includes_en_and_zh_cn() -> None:
    catalogs = load_catalogs()
    assert "en" in catalogs
    assert "zh-CN" in catalogs


def test_every_locale_matches_en_keys_and_types() -> None:
    catalogs = load_catalogs()
    en_keys = set(dict(_flatten(catalogs["en"])))
    assert en_keys, "en catalog should not be empty"
    for locale, catalog in catalogs.items():
        assert isinstance(catalog, dict), locale
        mapping = dict(_flatten(catalog))
        assert set(mapping) == en_keys, f"{locale} keys differ from en"
        assert all(isinstance(v, str) for v in mapping.values()), locale


@pytest.mark.parametrize(
    "key, expected",
    [
        ("web_config.no_half", const.NO_HALF_HELP),
        ("web_config.model_dir", const.MODEL_DIR_HELP),
        ("web_config.gpu_id", f"GPU index {const.GPU_ID_HELP}"),
        ("web_config.quality", f"Image Quality ({const.QUALITY_HELP})"),
        ("web_config.input", f"Input file or directory. {const.INPUT_HELP}"),
        ("web_config.output_dir", f"Output directory. {const.OUTPUT_DIR_HELP}"),
    ],
)
def test_en_catalog_tracks_const_help(key: str, expected: str) -> None:
    """en 目录由 const.py 的 *_HELP 组装：这里锁死映射，防止两边漂移。"""
    mapping = dict(_flatten(build_en_catalog()))
    assert mapping[key] == expected


def _placeholders(text: str) -> Set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_pick_locale_matches_exact_or_falls_back_to_en() -> None:
    """只精确匹配（大小写不敏感），无对应目录时回退 en —— 与前端自定义 key 的回退一致。"""
    assert pick_locale("zh-CN,zh;q=0.9,en;q=0.8") == "zh-CN"
    assert pick_locale("ZH-cn") == "zh-CN"
    assert pick_locale("ja-JP,ja;q=0.9") == "en"
    assert pick_locale(None) == "en"
    assert pick_locale("") == "en"


def test_pick_locale_respects_q_values() -> None:
    assert pick_locale("en;q=0.5,zh-CN;q=0.9") == "zh-CN"
    assert pick_locale("zh-CN;q=0.1,en;q=0.9") == "en"


def test_translate_uses_locale_and_falls_back() -> None:
    assert translate("web_config.host", "zh-CN") == "主机"
    # zh-CN 目录里缺的 key 回退 en 目录；en 也缺则原样返回 key
    assert translate("web_config.msg_input_missing", "zh-CN") == "[错误] 输入文件或目录不存在"
    assert translate("does.not.exist", "zh-CN") == "does.not.exist"
    assert (
        translate(
            "web_config.msg_save_success",
            "zh-CN",
            time="12:00:00",
            path="/tmp/x.json",
        )
        == "[12:00:00] 已成功保存配置到：/tmp/x.json"
    )


def test_dynamic_message_placeholders_match_en() -> None:
    """`web_config.msg_*` 的占位符集合必须与 en 一致，否则参数替换会静默留白。"""
    catalogs = load_catalogs()
    en = dict(_flatten(catalogs["en"]))
    for locale, catalog in catalogs.items():
        for key, value in _flatten(catalog):
            if not key.startswith("web_config.msg_"):
                continue
            assert _placeholders(value) == _placeholders(en[key]), f"{locale}:{key}"


def test_cli_catalog_placeholders_match_en() -> None:
    """`cli.*` 的占位符集合必须与 en 一致，否则参数替换会静默留白。"""
    catalogs = load_catalogs()
    en = dict(_flatten(catalogs["en"]))
    cli_keys = [key for key in en if key.startswith("cli.")]
    assert cli_keys, "en cli catalog should not be empty"
    for locale, catalog in catalogs.items():
        mapping = dict(_flatten(catalog))
        for key in cli_keys:
            assert _placeholders(mapping[key]) == _placeholders(en[key]), f"{locale}:{key}"


def test_cli_text_follows_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IOPAINT_LOCALE", "zh-CN")
    assert cli_text("cli.command.start") == "启动 IOPaint 服务"
    # zh-CN 目录里缺的 key 回退 en 目录；en 也缺则原样返回 key
    assert cli_text("cli.nope") == "cli.nope"

    monkeypatch.setenv("IOPAINT_LOCALE", "en")
    assert cli_text("cli.command.start") == "Start IOPaint server"


def test_resolve_cli_locale_prefers_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`IOPAINT_LOCALE` 是显式指定：认不出就用 en，不再看系统变量。"""
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    monkeypatch.setenv("IOPAINT_LOCALE", "en")
    assert resolve_cli_locale() == "en"

    monkeypatch.setenv("IOPAINT_LOCALE", "ja-JP")
    assert resolve_cli_locale() == "en"

    monkeypatch.setenv("IOPAINT_LOCALE", "zh_CN")
    assert resolve_cli_locale() == "zh-CN"

    # 显式 C/POSIX 是“要求英文”，不落到 LANG
    monkeypatch.setenv("IOPAINT_LOCALE", "C")
    assert resolve_cli_locale() == "en"


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"LANG": "zh_CN.UTF-8"}, "zh-CN"),
        ({"LC_ALL": "zh_CN.UTF-8"}, "zh-CN"),
        ({"LC_MESSAGES": "zh_CN"}, "zh-CN"),
        # zh 只写语言：唯一一门 zh-* 目录命中
        ({"LANG": "zh"}, "zh-CN"),
        # LC_ALL 优先于 LC_MESSAGES / LANG
        ({"LC_ALL": "en_US.UTF-8", "LANG": "zh_CN.UTF-8"}, "en"),
        ({"LC_MESSAGES": "en_US.UTF-8", "LANG": "zh_CN.UTF-8"}, "en"),
        ({"LANG": "ja_JP.UTF-8"}, "en"),
        ({"LANG": "C"}, "en"),
        ({"LANG": "POSIX"}, "en"),
        # C/POSIX 一旦出现就定成英文，不再看后面的 LANG（POSIX 优先级）
        ({"LC_ALL": "C", "LANG": "zh_CN.UTF-8"}, "en"),
        ({"LC_MESSAGES": "POSIX", "LANG": "zh_CN.UTF-8"}, "en"),
        ({"LANG": ""}, "en"),
        ({}, "en"),
    ],
)
def test_resolve_cli_locale_reads_system_env(
    monkeypatch: pytest.MonkeyPatch, env: dict, expected: str
) -> None:
    for name in ("IOPAINT_LOCALE", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert resolve_cli_locale() == expected


def test_cli_help_uses_locale_in_subprocess() -> None:
    """帮助串在 `iopaint.cli` 导入时取好，用真进程验证 IOPAINT_LOCALE 生效。"""
    env = {**os.environ, "IOPAINT_LOCALE": "zh-CN"}
    proc = subprocess.run(
        [sys.executable, "-m", "iopaint", "start", "--help"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "启动 IOPaint 服务" in proc.stdout
    # Rich markup：目录里写的是 `\[...]`，渲染后必须留下字面量方括号
    assert "[lama" in proc.stdout
    assert "\\[lama" not in proc.stdout
