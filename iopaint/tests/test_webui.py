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

import pytest

pw = pytest.importorskip("playwright", reason="playwright not installed")
from playwright.sync_api import sync_playwright, expect  # noqa: E402

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


@pytest.fixture(scope="module")
def page(browser_instance, iopaint_server):
    """One shared page for the whole module – fast, no per-test context creation."""
    ctx = browser_instance.new_context(viewport={"width": 1280, "height": 720})
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
        ctx = browser_instance.new_context(viewport={"width": 375, "height": 667})
        pg = ctx.new_page()
        pg.goto(iopaint_server, wait_until="load", timeout=30000)
        pg.wait_for_timeout(2000)
        expect(pg.get_by_text("Tap here to load your picture")).to_be_visible()
        ctx.close()

    def test_tablet(self, browser_instance, iopaint_server):
        ctx = browser_instance.new_context(viewport={"width": 768, "height": 1024})
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

        ctx = browser_instance.new_context(viewport={"width": 1280, "height": 720})
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

        ctx = browser_instance.new_context(viewport={"width": 1280, "height": 720})
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
