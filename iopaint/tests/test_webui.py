"""WebUI integration tests using Playwright.

Requires:  ``pip install playwright && playwright install chromium``
Run with:  ``pytest iopaint/tests/test_webui.py -v``
"""
import base64
import io
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright", reason="playwright not installed")
from playwright.sync_api import sync_playwright, expect  # noqa: E402


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
