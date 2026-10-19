"""WebUI integration tests using Playwright.

Requires:  ``pip install playwright && playwright install chromium``
Run with:  ``pytest iopaint/tests/test_webui.py -v``

These drive the real built frontend in a real browser, so they need both playwright
*and* the build output in ``iopaint/web_app``. Without the build output the whole module
skips (see ``WEB_APP_DIR``) — the ``webui`` CI job builds it first.
"""
import base64
import io
import json
import time
from pathlib import Path
from typing import Any

import pytest

pw = pytest.importorskip("playwright", reason="playwright not installed")
from playwright.sync_api import Browser, BrowserContext, expect, sync_playwright  # noqa: E402

from iopaint.api import WEB_APP_DIR

pytestmark = pytest.mark.skipif(
    not WEB_APP_DIR.is_dir(),
    reason=f"前端没构建（{WEB_APP_DIR} 不存在），先 npm run build 并拷进 iopaint/web_app",
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def browser_instance():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


# webui 的断言全部用英文文案，而界面默认按浏览器语言自动选择（中文开发机上
# navigator.language=zh-CN 会渲染成中文）。显式固定 locale=en-US，让结果与
# 开发机/CI 的系统语言无关。
def _new_context(browser_instance: Browser, **kwargs: Any) -> BrowserContext:
    return browser_instance.new_context(locale="en-US", **kwargs)


@pytest.fixture(scope="module")
def page(browser_instance, iopaint_server):
    """One shared page for the whole module – fast, no per-test context creation."""
    ctx = _new_context(browser_instance, viewport={"width": 1280, "height": 720})
    pg = ctx.new_page()
    pg.goto(iopaint_server, wait_until="load", timeout=30000)
    pg.wait_for_timeout(2000)
    yield pg
    ctx.close()


def _upload_image(page_obj, tmp_path):
    """Upload a test PNG and wait for the canvas to appear."""
    test_img = tmp_path / "test.png"
    from PIL import Image
    Image.new("RGB", (256, 256), color=(100, 200, 50)).save(str(test_img))
    page_obj.locator('input[type="file"]').first.set_input_files(str(test_img))
    page_obj.locator("canvas").first.wait_for(state="visible", timeout=10000)


def _fresh_page(browser_instance, iopaint_server):
    """独立 context 的干净页面：settings 存在 localStorage 里，
    新 context 才能保证「默认设置」的断言不被共享 page 的历史污染。"""
    ctx = _new_context(browser_instance, viewport={"width": 1280, "height": 720})
    pg = ctx.new_page()
    pg.goto(iopaint_server, wait_until="load", timeout=30000)
    pg.wait_for_timeout(2000)
    return ctx, pg


def _draw_stroke(page_obj, dx=40):
    """在画布中心横着划一笔并松手（auto 模式下松手即触发 /inpaint）。"""
    bb = page_obj.locator("canvas").first.bounding_box()
    cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
    page_obj.mouse.move(cx - dx, cy)
    page_obj.mouse.down()
    page_obj.mouse.move(cx + dx, cy, steps=10)
    page_obj.mouse.up()


def _save_button(page_obj):
    """Save Image 按钮：IconButton 没有可访问名字，靠 lucide-download 图标定位，
    全仓库只有 Editor 的保存按钮用这个图标。renders.length 决定它的 disabled。"""
    return page_obj.locator("button:has(svg.lucide-download)")


def _wait_interactive(page_obj):
    """Radix Select/Dialog 关闭动画期间会给 body 挂 pointer-events:none，
    此时画布上的鼠标事件全部落空（实测约 300ms）。等它复位再画。"""
    page_obj.wait_for_function(
        "() => document.body.style.pointerEvents !== 'none'", timeout=5000
    )


# ── Tests ─────────────────────────────────────────────────────────────

class TestPageLoad:
    def test_title(self, page):
        title = page.title()
        assert title != "", f"Page title is empty"

    def test_no_console_errors(self, page):
        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.reload(wait_until="load")
        page.wait_for_timeout(2000)
        critical = [e for e in errors if "ResizeObserver" not in e]
        assert len(critical) == 0, f"Console errors: {critical}"

    def test_initial_state_no_canvas(self, page):
        prompt = page.get_by_text("Click here or drag an image file")
        expect(prompt).to_be_visible()

    def test_canvas_appears_after_upload(self, page, tmp_path):
        _upload_image(page, tmp_path)
        expect(page.locator("canvas").first).to_be_visible()


class TestFileUpload:
    def test_upload_replaces_image(self, page, tmp_path):
        _upload_image(page, tmp_path)
        from PIL import Image
        img2 = tmp_path / "test2.png"
        Image.new("RGB", (128, 128), color=(200, 50, 100)).save(str(img2))
        page.locator('input[type="file"]').first.set_input_files(str(img2))
        page.wait_for_timeout(1000)
        expect(page.locator("canvas").first).to_be_visible()


class TestBrushTools:
    def test_brush_cursor(self, page, tmp_path):
        bb = page.locator("canvas").first.bounding_box()
        page.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        page.wait_for_timeout(300)
        assert page.locator("canvas").first.is_visible()

    def test_draw_stroke(self, page, tmp_path):
        bb = page.locator("canvas").first.bounding_box()
        cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
        page.mouse.move(cx - 50, cy)
        page.mouse.down()
        page.mouse.move(cx + 50, cy, steps=10)
        page.mouse.up()
        page.wait_for_timeout(300)
        assert page.locator("canvas").first.is_visible()

    def test_right_click_no_crash(self, page, tmp_path):
        bb = page.locator("canvas").first.bounding_box()
        page.mouse.click(
            bb["x"] + bb["width"] / 2,
            bb["y"] + bb["height"] / 2,
            button="right",
        )
        page.wait_for_timeout(300)
        assert page.locator("canvas").first.is_visible()

    def test_bracket_size_change(self, page, tmp_path):
        page.keyboard.press("[")
        page.keyboard.press("]")
        page.wait_for_timeout(200)
        assert page.locator("canvas").first.is_visible()


class TestKeyboardShortcuts:
    def test_ctrl_z_undo(self, page, tmp_path):
        bb = page.locator("canvas").first.bounding_box()
        page.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        page.mouse.down()
        page.mouse.up()
        page.keyboard.press("Control+z")
        page.wait_for_timeout(300)
        assert page.locator("canvas").first.is_visible()

    def test_ctrl_y_redo(self, page, tmp_path):
        bb = page.locator("canvas").first.bounding_box()
        page.mouse.move(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        page.mouse.down()
        page.mouse.up()
        page.keyboard.press("Control+y")
        page.wait_for_timeout(300)
        assert page.locator("canvas").first.is_visible()

    def test_space_hand_toggle(self, page, tmp_path):
        page.keyboard.down("Space")
        page.wait_for_timeout(200)
        page.keyboard.up("Space")
        page.wait_for_timeout(200)
        assert page.locator("canvas").first.is_visible()


class TestModelUI:
    def test_model_name_visible(self, page, tmp_path):
        expect(page.get_by_text("cv2").first).to_be_visible()

    def test_cv2_flag_selector(self, page, tmp_path):
        expect(page.get_by_text("CV2 Flag").first).to_be_visible()


class TestInpaintFlow:
    def test_draw_and_see_stroke(self, page, tmp_path):
        bb = page.locator("canvas").first.bounding_box()
        cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
        page.mouse.move(cx - 40, cy - 40)
        page.mouse.down()
        page.mouse.move(cx + 40, cy + 40, steps=10)
        page.mouse.up()
        page.wait_for_timeout(500)
        assert page.locator("canvas").first.is_visible()


class TestResponsive:
    def test_mobile(self, browser_instance, iopaint_server):
        ctx = _new_context(browser_instance, viewport={"width": 375, "height": 667})
        pg = ctx.new_page()
        pg.goto(iopaint_server, wait_until="load", timeout=30000)
        pg.wait_for_timeout(2000)
        expect(pg.get_by_text("Tap here to load your picture")).to_be_visible()
        ctx.close()

    def test_tablet(self, browser_instance, iopaint_server):
        ctx = _new_context(browser_instance, viewport={"width": 768, "height": 1024})
        pg = ctx.new_page()
        pg.goto(iopaint_server, wait_until="load", timeout=30000)
        pg.wait_for_timeout(2000)
        prompt = pg.get_by_text("Click here or drag an image file")
        prompt2 = pg.get_by_text("Tap here to load your picture")
        assert prompt.count() > 0 or prompt2.count() > 0
        ctx.close()


# ── 之前没被覆盖的几处契约 ─────────────────────────────────────────────

class TestSetFileTiming:
    """setFile 的时序契约：抽 prompt 不能挡在「打开图片」前面。

    gen-info 要把整张图上传一次（实测 230~290ms），它必须在后台跑。
    这里把 gen-info 整个挂死（永远不返回），画布仍然必须出现 ——
    串行版本会一直等它，直接超时。
    """

    def test_gen_info_delay_does_not_block_opening_image(
        self, browser_instance, iopaint_server, tmp_path
    ):
        from PIL import Image

        ctx = _new_context(browser_instance, viewport={"width": 1280, "height": 720})
        pg = ctx.new_page()
        pg.goto(iopaint_server, wait_until="load", timeout=30000)
        pg.wait_for_timeout(2000)

        hits = []

        def hang_gen_info(route):
            # 故意既不 continue 也不 fulfill，让 gen-info 永远挂着。
            # 注意不能在 handler 里 time.sleep：sync API 会连带阻塞事件循环，
            # wait_for_timeout 也一起停摆，测出来的是 sleep 的时长而不是页面的。
            hits.append(1)

        pg.route("**/gen-info", hang_gen_info)

        test_img = tmp_path / "timing.png"
        Image.new("RGB", (256, 256), (10, 120, 200)).save(str(test_img))

        t0 = time.time()
        pg.locator('input[type="file"]').first.set_input_files(str(test_img))
        # gen-info 永远不返回时画布仍然要出现；串行版本会一直等在这里直到超时
        pg.locator("canvas").first.wait_for(state="visible", timeout=6000)
        canvas_after = time.time() - t0

        assert hits, "gen-info 根本没被调用，这条测试是空转的"
        assert canvas_after < 3.0, f"画布等了 {canvas_after:.2f}s 才出来"

        pg.unroute("**/gen-info")
        ctx.close()


class TestMaskContract:
    """掩膜从 canvas 一路到后端的契约：PNG / 全尺寸 / 笔画真的在上面。

    generateMask → canvasToBlob 是异步的（原来用同步的 toDataURL），
    这条测试卡住「异步化之后拿到的东西还是不是那张掩膜」。
    """

    def test_mask_sent_to_backend_is_full_size_png(self, page, tmp_path):
        from PIL import Image

        _upload_image(page, tmp_path)
        captured = {}

        def capture(route):
            # post_data_buffer 是属性（bytes），不是方法
            captured["body"] = route.request.post_data_buffer
            route.abort()

        page.route("**/inpaint", capture)

        bb = page.locator("canvas").first.bounding_box()
        cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
        page.mouse.move(cx - 30, cy - 30)
        page.mouse.down()
        page.mouse.move(cx + 30, cy + 30, steps=10)
        page.mouse.up()

        deadline = time.time() + 8
        while time.time() < deadline and "body" not in captured:
            page.wait_for_timeout(200)
        page.unroute("**/inpaint")

        assert "body" in captured, "画完一笔却没有发出 /inpaint 请求"

        payload = json.loads(captured["body"].decode("utf-8"))
        mask = payload["mask"]
        assert mask.startswith("data:image/png;base64,"), mask[:80]

        raw = base64.b64decode(mask.split(",", 1)[1])
        im = Image.open(io.BytesIO(raw))
        assert im.size == (256, 256), f"掩膜是 {im.size}，应该是原图的 (256, 256)"

        # 笔画必须真的落到掩膜上（背景是透明的）
        _lo, alpha_max = im.convert("RGBA").split()[-1].getextrema()
        assert alpha_max > 0, "掩膜全透明 —— 笔画没画进去"


class TestDeviceNoticeToast:
    """显存回退提示：server-config.deviceNotice → WebUI toast，且同一条只弹一次。"""

    NOTICE = "cuda:0 out of VRAM, switched to cuda:1"

    def test_notice_toast_shown_exactly_once(self, browser_instance, iopaint_server):
        def inject(route):
            data = route.fetch().json()
            data["deviceNotice"] = self.NOTICE
            route.fulfill(json=data)

        ctx = _new_context(browser_instance, viewport={"width": 1280, "height": 720})
        pg = ctx.new_page()
        pg.route("**/server-config", inject)
        pg.goto(iopaint_server, wait_until="load", timeout=30000)

        expect(pg.get_by_text(self.NOTICE)).to_be_visible(timeout=8000)

        # 页面加载会拉两次 server-config（App 的 getServerConfig + Settings 的
        # react-query），第二趟必须被 setServerConfig 里的 shownDeviceNotice 挡掉
        pg.wait_for_timeout(1500)
        count = pg.get_by_text(self.NOTICE).count()
        assert count == 1, f"同一条 deviceNotice 弹了 {count} 次"

        ctx.close()


# ── 第二轮扩充：对话框热键、笔刷热键、cv2 整链路、错误 toast、开关契约 ──

class TestDialogHotkeys:
    """h / s 两个对话框热键在真实浏览器里的开关契约。"""

    def test_h_toggles_hotkeys_dialog(self, page):
        page.keyboard.press("h")
        expect(page.get_by_text("Toggle Hotkeys Dialog")).to_be_visible(timeout=3000)
        # 第二次按 h 必须关掉：useToggle + useHotKey 的 enabled 状态不能把自己卡死
        page.keyboard.press("h")
        expect(page.get_by_text("Toggle Hotkeys Dialog")).to_be_hidden(timeout=3000)

    def test_s_opens_settings_dialog_and_escape_closes(self, page):
        page.keyboard.press("s")
        heading = page.get_by_role("heading", name="Settings")
        expect(heading).to_be_visible(timeout=3000)
        # General 页里的开关文案在场，说明 tab 内容真的渲染了（不是空壳对话框）
        page.get_by_role("button", name="General").click()
        expect(page.get_by_text("Enable manual inpainting")).to_be_visible(timeout=3000)
        page.keyboard.press("Escape")
        expect(heading).to_be_hidden(timeout=3000)


class TestBrushSizeHotkeys:
    """[ / ] 必须真的改到底部笔刷滑块的值（role=slider, max=200 是笔刷那根；
    cv2 面板的 radius 滑块是 max=100，用 aria-valuemax 区分）。"""

    def test_bracket_keys_adjust_brush_slider(self, page, tmp_path):
        # 底部笔刷滑块随 Editor 一起挂载，没有图片时不渲染，先上传
        _upload_image(page, tmp_path)
        slider = page.locator('[role="slider"][aria-valuemax="200"]')
        expect(slider).to_be_visible(timeout=3000)

        before = int(slider.get_attribute("aria-valuenow"))
        page.keyboard.press("]")
        # not_to_have_attribute 会轮询到超时：等 React 把新值刷进 aria-valuenow
        expect(slider).not_to_have_attribute("aria-valuenow", str(before), timeout=3000)
        after_inc = int(slider.get_attribute("aria-valuenow"))
        assert after_inc > before, f"']' 没有增大笔刷: {before} -> {after_inc}"

        page.keyboard.press("[")
        expect(slider).not_to_have_attribute("aria-valuenow", str(after_inc), timeout=3000)
        after_dec = int(slider.get_attribute("aria-valuenow"))
        assert after_dec < after_inc, f"'[' 没有减小笔刷: {after_inc} -> {after_dec}"


class TestInpaintRoundTrip:
    """cv2 真后端整链路：画一笔 → /inpaint 200 → 渲染进 renders →
    Save 解禁；Ctrl+Z 撤掉渲染（renders 弹栈）、Ctrl+Shift+Z 恢复。"""

    def test_stroke_to_render_and_undo_redo(
        self, browser_instance, iopaint_server, tmp_path
    ):
        ctx, pg = _fresh_page(browser_instance, iopaint_server)
        _upload_image(pg, tmp_path)

        save_btn = _save_button(pg)
        expect(save_btn).to_be_disabled(timeout=3000)

        with pg.expect_response("**/inpaint", timeout=20000) as resp_info:
            _draw_stroke(pg)
        resp = resp_info.value
        assert resp.status == 200, f"/inpaint 返回 {resp.status}"
        assert len(resp.body()) > 0, "/inpaint 返回了空 body"

        expect(save_btn).to_be_enabled(timeout=10000)
        # 成功路径不应出现拦截器给错误加的截图提示
        assert pg.get_by_text("Please take a screenshot", exact=False).count() == 0

        pg.keyboard.press("Control+z")
        expect(save_btn).to_be_disabled(timeout=3000)
        pg.keyboard.press("Control+Shift+z")
        expect(save_btn).to_be_enabled(timeout=3000)

        ctx.close()


class TestLosslessRevision:
    """第二轮开始只传 revision + mask；undo 后回到原始文件字节。

    无损依赖两层契约：
    1. 有 revision 的 render 不再把上一轮（有损）响应字节当输入；
    2. undo 撤掉 render 后，原始上传文件仍是唯一输入源，而不是最后一帧的 JPEG。
    """

    def test_second_round_omits_image_and_sends_revision(
        self,
        browser_instance: Browser,
        iopaint_server: str,
        tmp_path: Path,
    ) -> None:
        ctx, pg = _fresh_page(browser_instance, iopaint_server)
        _upload_image(pg, tmp_path)

        with pg.expect_response("**/inpaint", timeout=20000) as resp_info:
            _draw_stroke(pg)
        first = resp_info.value
        assert first.status == 200, f"第一轮 /inpaint 返回 {first.status}"
        revision = first.headers.get("x-revision")
        assert revision, "第一轮响应缺少 X-Revision"

        expect(_save_button(pg)).to_be_enabled(timeout=10000)
        _wait_interactive(pg)

        with pg.expect_request("**/inpaint", timeout=20000) as req_info:
            with pg.expect_response("**/inpaint", timeout=20000) as second_info:
                _draw_stroke(pg, dx=30)

        payload = json.loads(req_info.value.post_data)
        assert "image" not in payload, "第二轮仍在上传整张图片"
        assert payload.get("revision") == revision, payload.get("revision")

        second = second_info.value
        assert second.status == 200, f"第二轮 /inpaint 返回 {second.status}"
        # cv2 在纯色图上可能没有像素变化，此时 revision 保持不变（不新增空 revision）
        assert second.headers.get("x-revision"), "第二轮响应缺少 X-Revision"

        ctx.close()

    def test_undo_then_edit_uses_original_file(
        self,
        browser_instance: Browser,
        iopaint_server: str,
        tmp_path: Path,
    ) -> None:
        ctx, pg = _fresh_page(browser_instance, iopaint_server)
        _upload_image(pg, tmp_path)

        with pg.expect_response("**/inpaint", timeout=20000) as resp_info:
            _draw_stroke(pg)
        assert resp_info.value.status == 200
        expect(_save_button(pg)).to_be_enabled(timeout=10000)

        pg.keyboard.press("Control+z")
        expect(_save_button(pg)).to_be_disabled(timeout=3000)
        _wait_interactive(pg)

        raw = (tmp_path / "test.png").read_bytes()
        original = "data:image/png;base64," + base64.b64encode(raw).decode()

        with pg.expect_request("**/inpaint", timeout=20000) as req_info:
            with pg.expect_response("**/inpaint", timeout=20000) as second_info:
                _draw_stroke(pg, dx=30)

        payload = json.loads(req_info.value.post_data)
        assert "revision" not in payload, "undo 后仍复用了已撤销的 revision"
        assert payload.get("image") == original, "undo 后没有回到原始文件字节"
        assert second_info.value.status == 200

        ctx.close()


class TestInpaintErrorToast:
    """后端 500 → axios 拦截器把 detail 拍平成一行 + 追加截图提示 → destructive
    toast；且失败不会产生 render（Save 保持 disabled）。"""

    DETAIL = "boom from webui test"

    def test_error_detail_reaches_toast(self, browser_instance, iopaint_server, tmp_path):
        ctx, pg = _fresh_page(browser_instance, iopaint_server)

        def fail(route):
            route.fulfill(
                status=500,
                content_type="application/json",
                body=json.dumps({"detail": self.DETAIL}),
            )

        pg.route("**/inpaint", fail)
        _upload_image(pg, tmp_path)
        _draw_stroke(pg)

        expect(pg.get_by_text(self.DETAIL)).to_be_visible(timeout=8000)
        expect(
            pg.get_by_text(
                "Please take a screenshot of the detailed error message in your terminal"
            )
        ).to_be_visible(timeout=3000)
        expect(_save_button(pg)).to_be_disabled(timeout=3000)

        ctx.close()


class TestCv2FlagRequest:
    """设置面板里换 CV2 Flag → 后续 /inpaint 请求体里的 cv2_flag 必须跟着变
    （设置 → axios payload 的贯通契约）。"""

    def test_selecting_telea_changes_payload(
        self, browser_instance, iopaint_server, tmp_path
    ):
        ctx, pg = _fresh_page(browser_instance, iopaint_server)

        trigger = pg.locator("button").filter(has_text="INPAINT_NS")
        expect(trigger).to_be_visible(timeout=3000)
        trigger.click()
        pg.get_by_role("option", name="INPAINT_TELEA").click()
        expect(
            pg.locator("button").filter(has_text="INPAINT_TELEA")
        ).to_be_visible(timeout=3000)
        _wait_interactive(pg)

        captured = {}

        def capture(route):
            captured["body"] = route.request.post_data_buffer
            route.abort()

        pg.route("**/inpaint", capture)
        _upload_image(pg, tmp_path)
        _draw_stroke(pg)

        deadline = time.time() + 8
        while time.time() < deadline and "body" not in captured:
            pg.wait_for_timeout(200)
        pg.unroute("**/inpaint")

        assert "body" in captured, "画完一笔却没有发出 /inpaint 请求"
        payload = json.loads(captured["body"].decode("utf-8"))
        assert payload["cv2_flag"] == "INPAINT_TELEA", payload["cv2_flag"]

        ctx.close()


class TestManualInpainting:
    """General 页的 Enable manual inpainting 开关：打开后画笔不再自动打
    /inpaint，Shift+R 才触发（编辑器里的手动模式热键契约）。"""

    def test_manual_mode_defers_run_until_shift_r(
        self, browser_instance, iopaint_server, tmp_path
    ):
        ctx, pg = _fresh_page(browser_instance, iopaint_server)

        # s 开对话框 → General → 拨开关 → 再按 s（Settings 的热键 handler 里
        # toggleOpen 后用闭包里的 open==true 显式调 onSubmit，落库不依赖 DOM 提交）
        pg.keyboard.press("s")
        expect(pg.get_by_role("heading", name="Settings")).to_be_visible(timeout=3000)
        pg.get_by_role("button", name="General").click()
        manual_switch = pg.get_by_role("switch").first
        manual_switch.click()
        expect(manual_switch).to_have_attribute("aria-checked", "true", timeout=3000)
        pg.keyboard.press("s")
        expect(pg.get_by_role("heading", name="Settings")).to_be_hidden(timeout=3000)
        # 模态 Dialog 关闭动画同样会短暂锁住 body 的 pointer-events
        _wait_interactive(pg)

        _upload_image(pg, tmp_path)

        captured = []

        def capture(route):
            captured.append(route.request.post_data_buffer)
            route.abort()

        pg.route("**/inpaint", capture)

        _draw_stroke(pg)
        pg.wait_for_timeout(1500)
        assert not captured, "手动模式下画完一笔不该自动打 /inpaint"

        pg.keyboard.press("Shift+R")
        deadline = time.time() + 8
        while time.time() < deadline and not captured:
            pg.wait_for_timeout(200)
        pg.unroute("**/inpaint")
        assert captured, "Shift+R 没有触发 /inpaint"

        ctx.close()
