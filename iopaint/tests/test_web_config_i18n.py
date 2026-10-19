"""Gradio 配置页 i18n 的浏览器端验证（Playwright）。

证明两件事：

1. `gr.I18n` 收到的**嵌套**自定义目录真的能在浏览器里解析 —— 静态 label 随浏览器
   locale 切成中文 / 英文；
2. 后端 `save_config` 的**动态**消息按请求的 `Accept-Language` 本地化
   （zh-CN → 中文，en-US → 英文）。

没有 playwright 时整模块跳过（`pip install playwright && playwright install chromium`）。
和 `test_webui.py` 不同，这里驱动的是 Gradio 自带前端，不需要 `iopaint/web_app`
构建产物，因此在没装 Node 的环境里也能跑。
"""
import socket
import time
import urllib.request
from pathlib import Path
from typing import Callable, Iterator, List

import pytest

pw = pytest.importorskip("playwright", reason="playwright not installed")
from playwright.sync_api import (  # noqa: E402
    Browser,
    BrowserContext,
    Page,
    expect,
    sync_playwright,
)

import iopaint.web_config as web_config  # noqa: E402
from iopaint.i18n import build_i18n  # noqa: E402

PageFactory = Callable[[str], Page]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _wait_until_ready(url: str) -> None:
    for _ in range(200):
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.1)
    pytest.fail(f"Gradio server at {url} did not become ready")


@pytest.fixture(scope="module")
def gradio_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """在空闲端口起真 Gradio 配置页；`scan_models` 打桩避免扫盘。"""
    config_file = tmp_path_factory.mktemp("web_config") / "config.json"
    original_scan = web_config.scan_models
    web_config.scan_models = lambda: []
    i18n = build_i18n()
    demo = web_config.build_demo(config_file, i18n)
    port = _free_port()
    demo.launch(
        server_port=port,
        prevent_thread_lock=True,
        inbrowser=False,
        show_api=False,
        i18n=i18n,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_until_ready(url)
        yield url
    finally:
        demo.close()
        web_config.scan_models = original_scan


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as p:
        instance = p.chromium.launch(headless=True)
        yield instance
        instance.close()


@pytest.fixture()
def open_page(browser: Browser) -> Iterator[PageFactory]:
    """返回 `(locale) -> Page`；用例结束后统一关掉各自的 context。"""
    contexts: List[BrowserContext] = []

    def _open(locale: str) -> Page:
        context = browser.new_context(locale=locale)
        contexts.append(context)
        return context.new_page()

    yield _open
    for context in contexts:
        context.close()


def test_static_labels_are_chinese_for_zh_cn(
    gradio_server: str, open_page: PageFactory
) -> None:
    page = open_page("zh-CN")
    page.goto(gradio_server)
    # 这两条都是自定义 key（不在 Gradio 内置目录里），能显示中文即证明嵌套目录生效
    expect(page.get_by_role("button", name="保存配置")).to_be_visible()
    expect(page.get_by_role("tab", name="常规")).to_be_visible()
    expect(page.get_by_role("tab", name="插件")).to_be_visible()


def test_static_labels_are_english_for_en_us(
    gradio_server: str, open_page: PageFactory
) -> None:
    page = open_page("en-US")
    page.goto(gradio_server)
    expect(page.get_by_role("button", name="Save configurations")).to_be_visible()
    expect(page.get_by_role("tab", name="Common")).to_be_visible()
    expect(page.get_by_role("tab", name="Plugins")).to_be_visible()


def test_dynamic_save_message_is_chinese_for_zh_cn(
    gradio_server: str, open_page: PageFactory
) -> None:
    page = open_page("zh-CN")
    page.goto(gradio_server)
    page.get_by_role("button", name="保存配置").click()
    expect(page.get_by_text("已成功保存配置到")).to_be_visible(timeout=20000)


def test_dynamic_save_message_is_english_for_en_us(
    gradio_server: str, open_page: PageFactory
) -> None:
    page = open_page("en-US")
    page.goto(gradio_server)
    page.get_by_role("button", name="Save configurations").click()
    expect(page.get_by_text("Successful save config to")).to_be_visible(timeout=20000)
