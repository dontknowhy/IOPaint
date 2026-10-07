#!/usr/bin/env python3
"""WebUI 性能基准测试（Playwright，跨内核）。

默认在 **chromium / firefox / webkit** 三个内核上各跑一遍同一组场景，
这样能看出某处优化是不是只在 Blink 上生效（比如 rAF 合帧、脏矩形重绘、
字体加载策略在 Gecko/WebKit 上的收益可能完全不同）。

用法::

    python scripts/perf_webui.py --json /tmp/perf-before.json
    # ...改代码、npm run build、拷进 iopaint/web_app...
    python scripts/perf_webui.py --json /tmp/perf-after.json --baseline /tmp/perf-before.json

    python scripts/perf_webui.py --engine chromium          # 只跑一个内核
    python scripts/perf_webui.py --url http://127.0.0.1:8080  # 复用已启动的服务

所有指标都是**越小越好**。

场景与口径：

``load``
    导航时序（TTFB / 脚本耗时 / DCL / load）、FCP、LCP、资源字节数、
    加载期 longtask。每个场景都用全新 context + 冷缓存，跨内核可比。
``open_image``
    选中一张 4096x3072 大图到画布可见的耗时。
``stroke``
    在大图上匀速拖一段长笔画期间的**帧间隔**（avg / p95 / max）、
    longtask、React commit 数。画笔手感好坏看这一段。
``cv2_radius`` / ``brush_slider``
    拖侧栏 CV2 Radius 滑块、底部画笔大小滑块期间的同一组指标。
    专门衡量「无关状态变化把整个 Editor 拖着重绘」的浪费。
``resize`` / ``idle``
    改窗口尺寸、以及完全空闲时的 commit 数（空闲应为 0）。

React commit 数通过在页面加载前注入 ``__REACT_DEVTOOLS_GLOBAL_HOOK__`` 采集：
React mount 时会主动探测这个全局钩子，每次 commit 回调一次 ——
这是**精确**的渲染提交计数，不需要往生产代码里塞任何探针。

注意：`longtask` / `largest-contentful-paint` 并非三个内核都支持，
脚本会读 `PerformanceObserver.supportedEntryTypes`，不支持的指标显示为 `n/a`，
不会拿 0 冒充「没有长任务」。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_APP_DIR = REPO_ROOT / "iopaint" / "web_app"

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sys.exit("需要 playwright：pip install playwright && playwright install chromium")


# ── 注入到页面里的探针（在应用脚本之前执行）────────────────────────────

INIT_SCRIPT = r"""
(() => {
  // 1) React 提交计数：React mount 时探测该全局钩子，每次 commit 回调一次
  const hook = {
    renderers: new Map(),
    supportsFiber: true,
    isDisabled: false,
    _nextId: 1,
    onCommitFiberRoot() { window.__reactCommits = (window.__reactCommits || 0) + 1 },
    onCommitFiberUnmount() {},
    onPostCommitFiberRoot() {},
    inject(renderer) {
      const id = hook._nextId++;
      hook.renderers.set(id, renderer);
      return id;
    },
    on() {}, off() {}, onerror() {}, onunhandledrejection() {},
  };
  Object.defineProperty(window, "__REACT_DEVTOOLS_GLOBAL_HOOK__", {
    value: hook, configurable: true,
  });
  window.__reactCommits = 0;

  // 2) 帧间隔记录器（三个内核都支持）
  window.__frameDeltas = [];
  let last = performance.now();
  const tick = (t) => {
    window.__frameDeltas.push(t - last);
    last = t;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  // 3) longtask 记录器（Blink 独有；Gecko/WebKit 不支持时静默降级）
  window.__longTasks = [];
  const supported = PerformanceObserver.supportedEntryTypes || [];
  window.__caps = {
    longtask: supported.includes("longtask"),
    lcp: supported.includes("largest-contentful-paint"),
    fcp: supported.includes("paint"),
  };
  if (window.__caps.longtask) {
    try {
      new PerformanceObserver((list) => {
        for (const e of list.getEntries()) window.__longTasks.push(e.duration);
      }).observe({ type: "longtask", buffered: true });
    } catch (e) { window.__caps.longtask = false; }
  }

  // 4) 打开图片的时间线：file input change → 图片 src 被赋值 → 图片 onload。
  //    用来把「后端抽 prompt 卡在关键路径上」和「浏览器解码」两段拆开看。
  window.__openTimeline = { fileChangeAt: 0, imgSrcAt: 0, imgLoadAt: 0 };
  window.__blobImg = null;
  document.addEventListener(
    "change",
    (e) => {
      const t = e.target;
      if (t && t.type === "file" && !window.__openTimeline.fileChangeAt) {
        window.__openTimeline.fileChangeAt = performance.now();
      }
    },
    true
  );
  const srcDesc = Object.getOwnPropertyDescriptor(
    HTMLImageElement.prototype, "src"
  );
  if (srcDesc && srcDesc.get && srcDesc.set) {
    Object.defineProperty(HTMLImageElement.prototype, "src", {
      configurable: true,
      get() { return srcDesc.get.call(this); },
      set(v) {
        if (String(v).startsWith("blob:") && !window.__openTimeline.imgSrcAt) {
          window.__openTimeline.imgSrcAt = performance.now();
          // 必须挂到元素自己身上：useImage 那个 <img> 从不插入文档，
          // load 事件的传播路径里根本没有 document。
          const mark = (kind) => () => {
            if (!window.__openTimeline.imgLoadAt) {
              window.__openTimeline.imgLoadAt = performance.now();
              window.__openTimeline.imgLoadKind = kind;
            }
          };
          this.addEventListener("load", mark("load"));
          this.addEventListener("error", mark("error"));
        }
        srcDesc.set.call(this, v);
      },
    });
  }

  // 编辑器什么时候挂上、画布什么时候拿到真实尺寸 —— 中间那段最容易藏大坑
  window.__openTimeline.canvasMountedAt = 0;
  window.__openTimeline.canvasSizedAt = 0;
  const mo = new MutationObserver(() => {
    if (!window.__openTimeline.canvasMountedAt && document.querySelector("canvas")) {
      window.__openTimeline.canvasMountedAt = performance.now();
      mo.disconnect();  // 只盯这一件事，别给后面的交互场景留常驻开销
    }
  });
  if (document) {
    // 观察 document 本身：init script 在 <html> 解析出来之前就跑了，
    // 盯 document.documentElement 在 Chromium/WebKit 上会拿到 null。
    mo.observe(document, { childList: true, subtree: true });
  }
  const pollCanvasSize = () => {
    const c = document.querySelector("canvas");
    if (c) {
      if (c.width > 500 && !window.__openTimeline.canvasSizedAt) {
        window.__openTimeline.canvasSizedAt = performance.now();
      }
      // Editor 里 TransformComponent 用 visibility 控制「还没居中就不显示」，
      // 记下它从 hidden 翻回 visible 的时刻，用来分辨「应用真的画好了」和
      // 「Playwright 认为可见」之间差的是什么
      const vis = getComputedStyle(c).visibility;
      if (vis === "hidden") {
        window.__openTimeline.sawHidden = true;
      } else if (window.__openTimeline.sawHidden && !window.__openTimeline.shownAt) {
        window.__openTimeline.shownAt = performance.now();
      }
      if (window.__openTimeline.canvasSizedAt && window.__openTimeline.shownAt) {
        return;
      }
    }
    requestAnimationFrame(pollCanvasSize);
  };
  requestAnimationFrame(pollCanvasSize);

  window.__resetPerf = () => {
    window.__frameDeltas = [];
    window.__longTasks = [];
    window.__reactCommits = 0;
  };
  window.__snapshotPerf = () => {
    const frames = window.__frameDeltas.slice().sort((a, b) => a - b);
    const tasks = window.__longTasks;
    const n = frames.length;
    const pick = (q) => (n ? frames[Math.min(n - 1, Math.floor(n * q))] : 0);
    return {
      frames: n,
      frame_avg: n ? frames.reduce((a, b) => a + b, 0) / n : 0,
      frame_p95: pick(0.95),
      frame_max: n ? frames[n - 1] : 0,
      longtask_total: tasks.reduce((a, b) => a + b, 0),
      longtask_count: tasks.length,
      commits: window.__reactCommits,
      caps: window.__caps,
    };
  };
})();
"""

LOAD_SCRIPT = r"""
() => {
  const nav = performance.getEntriesByType("navigation")[0] || {};
  const res = performance.getEntriesByType("resource");
  const paints = performance.getEntriesByType("paint");
  const supported = PerformanceObserver.supportedEntryTypes || [];
  const lcpEntries = supported.includes("largest-contentful-paint")
    ? performance.getEntriesByType("largest-contentful-paint") : [];
  const fcpEntry = paints.find((p) => p.name === "first-contentful-paint");
  const totalBytes = res.reduce((a, r) => a + (r.encodedBodySize || 0), 0);
  const fontBytes = res
    .filter((r) => r.name.includes(".woff"))
    .reduce((a, r) => a + (r.encodedBodySize || 0), 0);
  return {
    ttfb: (nav.responseStart || 0) - (nav.requestStart || 0),
    // responseEnd -> DCL：下载 + 解析 + 执行主脚本的代理指标
    script_time: (nav.domContentLoadedEventEnd || 0) - (nav.responseEnd || 0),
    dcl: (nav.domContentLoadedEventEnd || 0) - (nav.startTime || 0),
    load: (nav.loadEventEnd || 0) - (nav.startTime || 0),
    fcp: fcpEntry ? fcpEntry.startTime : null,
    lcp: lcpEntries.length ? lcpEntries[lcpEntries.length - 1].startTime : null,
    resources: res.length,
    total_bytes: totalBytes,
    font_bytes: fontBytes,
    longtask_total: window.__longTasks.reduce((a, b) => a + b, 0),
    longtask_count: window.__longTasks.length,
    caps: window.__caps,
  };
}
"""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, port: int):
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.url = f"http://127.0.0.1:{port}"

    def __enter__(self):
        output_dir = Path(tempfile.mkdtemp(prefix="iopaint-perf-out-"))
        self.proc = subprocess.Popen(
            [
                sys.executable, "-m", "iopaint", "start",
                "--model", "cv2", "--device", "cpu",
                "--port", str(self.port), "--output-dir", str(output_dir),
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                urllib.request.urlopen(self.url, timeout=2)
            except urllib.error.HTTPError:
                pass  # 4xx/5xx 也说明服务起来了
            except Exception:
                time.sleep(0.3)
                continue
            return self
        raise RuntimeError("server did not start in 30s")

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def make_big_image(path: Path, w: int = 4096, h: int = 3072) -> Path:
    """默认 4096x3072：超过 4MP 阈值，会走 maskScale 缩放路径。

    用渐变而非噪点 —— 解码代价一样，但 PNG 不会膨胀成十几 MB。
    """
    import math

    from PIL import Image

    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        fy = y / h
        for x in range(0, w, 4):
            fx = x / w
            r = int(127 + 120 * math.sin(fx * 12.5 + fy * 3.1))
            g = int(127 + 120 * math.sin(fy * 9.7 - fx * 4.3))
            b = int(127 + 120 * math.sin((fx + fy) * 7.9))
            for dx in range(4):
                if x + dx < w:
                    px[x + dx, y] = (r, g, b)
    img.save(str(path))
    return path


def make_api_recorder(out: list[dict]):
    """记录 /api/v1/* 请求的墙钟耗时（open_image 的时间都花在哪儿）。

    不用 Playwright 的 ``request.timing``：它的 ``startTime`` 是 epoch 毫秒而
    ``responseEnd`` 是相对时间，两者相减得到的是天文数字。
    """
    starts: dict[str, float] = {}

    def on_request(request) -> None:
        if "/api/v1/" in request.url:
            starts[request.url] = time.perf_counter()

    def on_finished(request) -> None:
        t0 = starts.pop(request.url, None)
        if t0 is None:
            return
        out.append({
            "api": request.url.split("/api/v1/", 1)[1].split("?")[0],
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    def on_fail(request) -> None:
        starts.pop(request.url, None)

    return on_request, on_finished, on_fail


def canvas_micro_bench(page) -> dict:
    """在页面里直接复刻 flushStroke 的画布操作，分离「2D 指令成本」和
    「图层重绘成本」。

    每个操作后用 1x1 的 getImageData 强制栅格化 —— 否则 Canvas2D 只是把指令
    记进命令流，performance.now() 量到的是微秒级的假数据。
    """
    js = """(size) => {
      const W = size.W, H = size.H;
      const mk = (w, h) => { const c = document.createElement('canvas');
                             c.width = w; c.height = h; return c };
      const base = mk(W, H), disp = mk(W, H);
      const bctx = base.getContext('2d');
      bctx.fillStyle = '#fff'; bctx.fillRect(0, 0, W, H);
      const ctx = disp.getContext('2d');
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
      const scale = 1;
      const pad = 32;
      const N = 8;
      const flush = (x, y) => ctx.getImageData(x, y, 1, 1);

      const bench = (name, fn) => {
        const t0 = performance.now();
        for (let i = 0; i < N; i++) fn(200 + i * 3, 200 + i * 3);
        return [name, +((performance.now() - t0) / N).toFixed(2)];
      };

      // 模拟一次 flushStroke：脏矩形 256x256（约等于大笔刷）
      const op = (x, y) => {
        const x0 = Math.round(x), y0 = Math.round(y), bw = 256, bh = 256;
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(x0, y0, bw, bh);
        ctx.drawImage(base, x0, y0, bw, bh, x0, y0, bw, bh);
        ctx.beginPath(); ctx.rect(x0, y0, bw, bh); ctx.clip();
        ctx.setTransform(scale, 0, 0, scale, 0, 0);
        ctx.strokeStyle = '#ffcc00bb';
        ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = 60;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        for (let k = 1; k < 90; k++) ctx.lineTo(x0 + Math.sin(k) * 120, y0 + k * 2);
        ctx.stroke();
        ctx.restore();
        flush(x0, y0);
      };

      const out = { canvas: `${W}x${H}` };
      out.clearRect = bench('clearRect', (x, y) => {
        ctx.clearRect(x, y, 256, 256); flush(x, y);
      })[1];
      out.drawImage = bench('drawImage', (x, y) => {
        ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.drawImage(base, x, y, 256, 256, x, y, 256, 256);
        ctx.restore(); flush(x, y);
      })[1];
      out.full_flush = bench('full_flush', op)[1];
      return out;
    }"""
    out = {}
    for size in ({"W": 2298, "H": 1724}, {"W": 768, "H": 512}):
        r = page.evaluate(js, size)
        out[r["canvas"]] = {k: v for k, v in r.items() if k != "canvas"}
    return out


def mask_encode_bench(page) -> dict:
    """全分辨率掩膜 PNG 编码：``canvasToImage`` 现在用的是同步的 ``toDataURL``，
    ``showPrevMask``（悬停「重跑上次掩膜」按钮）和 ``runInpainting`` 都会走到。

    同步编码会把主线程整段占住，这里量一下到底有多贵。"""
    return page.evaluate(
        """async () => {
      const W = 4096, H = 3072;
      const c = document.createElement('canvas');
      c.width = W; c.height = H;
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#fff';
      for (let i = 0; i < 40; i++) {
        ctx.beginPath(); ctx.arc(i * 97, i * 71, 60, 0, Math.PI * 2); ctx.fill();
      }
      const out = {};
      let t = performance.now();
      const dataUrl = c.toDataURL('image/png');
      out.toDataURL = +(performance.now() - t).toFixed(1);
      out.dataUrlKB = Math.round(dataUrl.length / 1024);
      t = performance.now();
      const blob = await new Promise((res) => c.toBlob(res, 'image/png'));
      out.toBlob = +(performance.now() - t).toFixed(1);
      out.blobKB = Math.round(blob.size / 1024);
      return out;
    }"""
    )


def run_on_engine(browser_type, url: str, image: Path) -> dict:
    """在一个内核上跑完所有场景。"""
    import math

    # WebKit 的宿主依赖校验要求 libicu*76，本机是 78（Debian forky）。
    # 我们把 76 的 .deb 解到 webkit bundle 的 lib/ 里补齐了，校验器不认识这个
    # 办法 —— 只在这里（且只对 webkit）跳过校验。
    if browser_type.name == "webkit":
        os.environ["PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS"] = "1"

    results: dict = {}

    browser = browser_type.launch(headless=True)

    # ── load（全新 context，冷缓存）──────────────────────────────
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script(INIT_SCRIPT)
    page = ctx.new_page()
    page.goto(url, wait_until="load", timeout=60000)
    page.wait_for_timeout(1500)
    results["load"] = page.evaluate(LOAD_SCRIPT)
    ctx.close()

    # ── 交互场景共用一个 context ─────────────────────────────────
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    ctx.add_init_script(INIT_SCRIPT)
    page = ctx.new_page()
    page.goto(url, wait_until="load", timeout=60000)
    page.wait_for_timeout(1500)

    t0 = time.perf_counter()
    # 顺带记下 open 期间发生的 API 请求，用来判断打开图片的延迟花在哪
    api_calls: list[dict] = []
    on_req, on_fin, on_fail = make_api_recorder(api_calls)
    page.on("request", on_req)
    page.on("requestfinished", on_fin)
    page.on("requestfailed", on_fail)
    t_set = time.perf_counter()
    page.locator('input[type="file"]').first.set_input_files(str(image))
    # Playwright 把文件塞进 <input> 本身也要走一遍协议，先把它从应用耗时里摘出去
    results["set_input_files_ms"] = (time.perf_counter() - t_set) * 1000
    t_app = time.perf_counter()
    page.locator("canvas").first.wait_for(state="visible", timeout=30000)
    results["open_image_ms"] = (time.perf_counter() - t_app) * 1000
    t_vis_page = page.evaluate("performance.now()")
    # 真正「看到图」的时刻：图片画布上已经有不透明像素
    try:
        page.wait_for_function(
            """() => {
              const c = document.querySelectorAll('canvas')[0];
              if (!c || c.width < 500) return false;
              try {
                return c.getContext('2d').getImageData(1, 1, 1, 1).data[3] > 0;
              } catch (e) { return false; }
            }""",
            timeout=30000,
        )
        results["open_image_paint_ms"] = (time.perf_counter() - t0) * 1000
        t_paint_page = page.evaluate("performance.now()")
    except Exception:
        results["open_image_paint_ms"] = None
        t_paint_page = None
    page.wait_for_timeout(1500)  # 等图片解码完，不计入 open 延迟
    results["open_image_api"] = sorted(api_calls, key=lambda c: -c["ms"])[:6]
    # 用页面自己的 performance.now() 拆开「change → 赋 src → onload → 画出来」
    # 四段，否则光看一个总数字分不清是后端慢还是解码慢。
    tl = page.evaluate("() => Object.assign({}, window.__openTimeline)")
    results["open_timeline_raw"] = tl
    base = tl.get("fileChangeAt") or 0
    if base:
        results["open_timeline"] = {
            "to_img_src": round((tl.get("imgSrcAt") or 0) - base, 1),
            "img_decode": round((tl.get("imgLoadAt") or 0) - (tl.get("imgSrcAt") or 0), 1),
            "to_mounted": round((tl.get("canvasMountedAt") or 0) - base, 1),
            "to_sized": round((tl.get("canvasSizedAt") or 0) - base, 1),
            "to_shown": round((tl.get("shownAt") or 0) - base, 1),
            "to_visible": round(t_vis_page - base, 1),
            "to_paint": round((t_paint_page or 0) - base, 1),
        }

    box = page.locator("canvas").first.bounding_box()
    assert box, "canvas not found"

    # 轨迹按画布尺寸取比例，换 --image-size 时仍然画在图内
    half_w = box["width"] * 0.4
    half_h = box["height"] * 0.35

    def drag_stroke(steps: int = 90, pause_ms: int = 14):
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        page.mouse.move(cx - half_w, cy - half_h)
        page.mouse.down()
        for i in range(steps):
            t = i / (steps - 1)
            x = cx - half_w + t * half_w * 2
            y = cy - half_h + half_h * 2 * (0.5 - 0.5 * math.cos(t * 12.566))
            page.mouse.move(x, y)
            page.wait_for_timeout(pause_ms)
        page.mouse.up()

    # ── hover（对照组）：同样的轨迹但不按下鼠标 ────────────────
    # 用来把「React 事件处理 + 笔刷光标 DOM 更新」和「画布栅格化」的成本分开
    def hover_stroke(steps: int = 90, pause_ms: int = 14):
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        page.mouse.move(cx - half_w, cy - half_h)
        for i in range(steps):
            t = i / (steps - 1)
            x = cx - half_w + t * half_w * 2
            y = cy - half_h + half_h * 2 * (0.5 - 0.5 * math.cos(t * 12.566))
            page.mouse.move(x, y)
            page.wait_for_timeout(pause_ms)

    page.evaluate("window.__resetPerf()")
    hover_stroke()
    page.wait_for_timeout(300)
    results["hover"] = page.evaluate("window.__snapshotPerf()")

    # ── 画布微基准：定位 WebKit 笔画卡顿到底在 2D 指令还是图层重绘 ──
    results["canvas_micro"] = canvas_micro_bench(page)
    results["mask_encode"] = mask_encode_bench(page)

    # ── stroke ─────────────────────────────────────────────────
    page.evaluate("window.__resetPerf()")
    drag_stroke()
    page.wait_for_timeout(300)
    results["stroke"] = page.evaluate("window.__snapshotPerf()")

    # ── 侧栏 CV2 Radius 滑块（无关设置变化 → Editor 是否被拖下水）──
    def pick_thumb(matcher):
        """按 bounding box 找 Radix Slider 的 thumb（role=slider）。"""
        thumbs = page.locator('[role="slider"]')
        for i in range(thumbs.count()):
            b = thumbs.nth(i).bounding_box()
            if b and matcher(b):
                return b
        return None

    def drag_thumb(target, moves=40, dx=3, pause=16):
        page.mouse.move(target["x"] + target["width"] / 2,
                        target["y"] + target["height"] / 2)
        page.mouse.down()
        for i in range(moves):
            page.mouse.move(target["x"] + target["width"] / 2 + (i - moves // 2) * dx,
                            target["y"] + target["height"] / 2)
            page.wait_for_timeout(pause)
        page.mouse.up()

    radius_box = None
    radius_input = page.locator("#cv2-radius")
    if radius_input.count() > 0:
        ibox = radius_input.bounding_box()
        if ibox:
            radius_box = pick_thumb(
                lambda b: abs(b["y"] - ibox["y"]) < 40 and b["x"] > ibox["x"] - 300
            )
    if radius_box:
        page.evaluate("window.__resetPerf()")
        page.evaluate("window.__editorRenders = 0")
        drag_thumb(radius_box)
        page.wait_for_timeout(200)
        results["cv2_radius"] = page.evaluate("window.__snapshotPerf()")
        # Editor 自己的渲染次数（生产代码里临时插的计数器）。commit 总数
        # 分不出「SidePanel 自己重绘」和「Editor 被陪绑重绘」，这个可以。
        results["editor_renders_cv2"] = page.evaluate(
            "(window.__editorRenders !== undefined ? window.__editorRenders : null)"
        )

    # ── 底部画笔大小滑块 ───────────────────────────────────────
    viewport_h = page.viewport_size["height"]
    brush_box = pick_thumb(lambda b: b["y"] > viewport_h - 140)
    if brush_box:
        page.evaluate("window.__resetPerf()")
        drag_thumb(brush_box, dx=4)
        page.wait_for_timeout(200)
        results["brush_slider"] = page.evaluate("window.__snapshotPerf()")
    results["_targets"] = {
        "cv2_radius_thumb": bool(radius_box),
        "brush_thumb": bool(brush_box),
    }

    # ── resize ─────────────────────────────────────────────────
    page.evaluate("window.__resetPerf()")
    for w in (1200, 1300, 1400, 1500, 1440):
        page.set_viewport_size({"width": w, "height": 900})
        page.wait_for_timeout(120)
    page.wait_for_timeout(300)
    results["resize"] = page.evaluate("window.__snapshotPerf()")

    # ── idle：没人操作时不应该有 React 提交 ─────────────────────
    page.evaluate("window.__resetPerf()")
    page.wait_for_timeout(1000)
    results["idle"] = page.evaluate("window.__snapshotPerf()")

    ctx.close()
    browser.close()
    return results


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.1f}"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, dict):
        return ""
    return str(v)


def delta(now, base) -> str:
    if base is None or not isinstance(now, (int, float)) or not isinstance(base, (int, float)):
        return ""
    if base == 0:
        return "n/a" if now == 0 else "new"
    pct_change = (now - base) / base * 100
    if abs(pct_change) < 0.5:
        return "="
    return f"{'↓' if pct_change < 0 else '↑'}{abs(pct_change):.1f}%"


# 每个场景要展示的字段
SECTIONS: list[tuple[str, list[str] | None]] = [
    ("load", ["ttfb", "script_time", "dcl", "load", "fcp", "lcp",
              "total_bytes", "font_bytes", "resources", "longtask_total"]),
    ("open_image_ms", None),
    ("set_input_files_ms", None),
    ("open_image_paint_ms", None),
    ("hover", ["frame_avg", "frame_p95", "frame_max", "longtask_total",
               "longtask_count", "commits"]),
    ("stroke", ["frame_avg", "frame_p95", "frame_max", "longtask_total",
                "longtask_count", "commits"]),
    ("cv2_radius", ["frame_avg", "frame_p95", "frame_max", "longtask_total",
                    "longtask_count", "commits"]),
    ("brush_slider", ["frame_avg", "frame_p95", "frame_max", "longtask_total",
                      "longtask_count", "commits"]),
    ("resize", ["frame_max", "longtask_total", "commits"]),
    ("idle", ["frame_max", "longtask_total", "commits"]),
]


def report(all_results: dict, baseline: dict | None) -> None:
    engines = list(all_results.keys())
    col = max(14, max(len(e) for e in engines) + 2)
    label_w = 34

    header = f"{'metric':<{label_w}}" + "".join(
        f"{e:>{col}}" + f"{'Δ':>10}" for e in engines
    )
    print(header)
    print("-" * len(header))

    for section, fields in SECTIONS:
        rows: list[tuple[str, dict]] = []
        if fields is None:
            rows.append((section, None))
        else:
            rows = [(f, f) for f in fields]

        printed_header = False
        for label, field in rows:
            cells = []
            any_value = False
            for e in engines:
                now = all_results[e].get(section)
                base = (baseline or {}).get(e, {}).get(section)
                if field is None:
                    n, b = now, base
                else:
                    if not isinstance(now, dict):
                        n, b = None, None
                    else:
                        n = now.get(field)
                        b = base.get(field) if isinstance(base, dict) else None
                        if field.endswith(("longtask_total", "longtask_count")):
                            caps = now.get("caps") or {}
                            if not caps.get("longtask", True):
                                n = None
                if n is not None:
                    any_value = True
                cells.append((n, delta(n, b) if b is not None else ""))
            if not any_value:
                continue
            if not printed_header:
                print(f"[{section}]")
                printed_header = True
            line = f"  {label:<{label_w - 2}}"
            for n, d in cells:
                line += f"{fmt(n):>{col}}" + (f"{d:>10}" if d else f"{'':>10}")
            print(line)

    # 诊断信息：open 期间的 API 耗时 + 滑块定位是否成功
    for e in engines:
        r = all_results.get(e) or {}
        api = r.get("open_image_api")
        if api:
            print(f"\n[{e}] open_image 期间的 API（花得最多的几个）:")
            for c in api:
                print(f"    {c['api']:<32}{c['ms']:>9.1f} ms")
        if r.get("_targets"):
            print(f"[{e}] 滑块定位: {r['_targets']}")
        if r.get("editor_renders_cv2") is not None:
            print(f"[{e}] 拖 cv2 radius 期间 Editor 重绘次数: "
                  f"{r['editor_renders_cv2']}")
        micro = r.get("canvas_micro")
        if micro:
            print(f"[{e}] 画布微基准（单次操作耗时 ms，越小越好）:")
            for canvas_name, ops in micro.items():
                cells = "  ".join(f"{k}={v:>8.2f}" for k, v in ops.items())
                print(f"    {canvas_name:<10} {cells}")
        enc = r.get("mask_encode")
        if enc:
            print(f"[{e}] 4096x3072 掩膜 PNG 编码: "
                  f"toDataURL={enc.get('toDataURL')}ms（同步/阻塞）, "
                  f"toBlob={enc.get('toBlob')}ms（异步）")
        tl = r.get("open_timeline")
        if tl:
            raw = r.get("open_timeline_raw") or {}
            print(f"[{e}] 打开图片时间线（相对 file input change，ms）: "
                  f"赋 src {tl['to_img_src']} → 解码 {tl['img_decode']} "
                  f"({raw.get('imgLoadKind')}) → 挂载 {tl['to_mounted']} → "
                  f"定尺寸 {tl['to_sized']} → 显示 {tl['to_shown']} → "
                  f"Playwright可见 {tl['to_visible']} → 上色 {tl['to_paint']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="WebUI 跨内核性能基准",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--json", help="把本次结果写到这个文件")
    ap.add_argument("--baseline", help="对比用的上次 JSON")
    ap.add_argument("--engine", default="chromium,firefox,webkit",
                    help="逗号分隔的内核列表（默认 chromium,firefox,webkit）")
    ap.add_argument("--image-size", default="4096x3072",
                    metavar="WxH", help="测试图片尺寸（默认 4096x3072）")
    ap.add_argument("--url", help="直接给一个已运行的服务地址，跳过拉起服务")
    args = ap.parse_args()

    if not WEB_APP_DIR.is_dir():
        sys.exit(f"{WEB_APP_DIR} 不存在，先 npm run build 并拷贝 dist")

    tmp = Path(tempfile.mkdtemp(prefix="iopaint-perf-"))
    try:
        iw, ih = (int(v) for v in args.image_size.lower().split("x"))
    except ValueError:
        sys.exit(f"--image-size 格式不对：{args.image_size}（应为 4096x3072）")
    image = make_big_image(tmp / "big.png", iw, ih)
    print(f"test image: {image} ({image.stat().st_size / 1e6:.1f} MB, {iw}x{ih})")

    wanted = [e.strip() for e in args.engine.split(",") if e.strip()]
    if args.url:
        _run_and_report(args, wanted, image, args.url)
        return

    with Server(free_port()) as srv:
        _run_and_report(args, wanted, image, srv.url)


def _run_and_report(args, engines: list[str], image: Path, url: str | None = None) -> None:
    with sync_playwright() as p:
        available = {}
        for name in ("chromium", "firefox", "webkit"):
            try:
                getattr(p, name).executable_path
                available[name] = getattr(p, name)
            except Exception:
                pass

        chosen = [e for e in engines if e in available]
        skipped = [e for e in engines if e not in available]
        if skipped:
            print(f"跳过未安装的内核: {', '.join(skipped)}")
        if not chosen:
            sys.exit("没有可用的内核")

        all_results: dict = {}
        for name in chosen:
            print(f"→ {name} ...", flush=True)
            try:
                all_results[name] = run_on_engine(available[name], url, image)
            except Exception as exc:  # 一个内核挂了不要拖垮整个基准
                print(f"  ! {name} 失败: {exc}")
                all_results[name] = {}

    baseline = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
    if args.json:
        Path(args.json).write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"written: {args.json}")
    print()
    report(all_results, baseline)


if __name__ == "__main__":
    main()
