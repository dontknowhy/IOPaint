// utils.ts 纯逻辑单测（vitest + jsdom）。只测不依赖真实 canvas/网络的部分：
// 错误消息格式化、保存文件名、包围盒、笔迹绘制（fake ctx）、点击类型判定、坐标换算、
// 显示画布位图缩放。
import { describe, expect, it, vi } from "vitest"
import {
  buildDownloadName,
  drawLineWithinRect,
  drawLines,
  extFromMime,
  fitCanvasBitmap,
  generateMask,
  generateMaskPreview,
  getErrorMessage,
  isMidClick,
  isRightClick,
  lineGroupToMaskStrokes,
  mouseXY,
  parseImageSize,
  strokeBounds,
  touchPointXY,
} from "./utils"
import { BRUSH_COLOR, MASK_CANVAS_MAX_PIXELS } from "./const"
import type { Line, Point } from "./types"

describe("getErrorMessage", () => {
  it("returns message of an Error", () => {
    expect(getErrorMessage(new Error("boom"))).toBe("boom")
  })

  it("returns the string itself", () => {
    expect(getErrorMessage("plain failure")).toBe("plain failure")
  })

  it("serializes plain objects", () => {
    expect(getErrorMessage({ detail: "bad" })).toBe('{"detail":"bad"}')
  })

  it("falls back to String() for circular objects", () => {
    const circular: Record<string, unknown> = {}
    circular.self = circular
    expect(getErrorMessage(circular)).toBe("[object Object]")
  })
})

describe("extFromMime", () => {
  it("maps known mime types", () => {
    expect(extFromMime("image/png")).toBe(".png")
    expect(extFromMime("image/jpeg")).toBe(".jpg")
    expect(extFromMime("image/tiff")).toBe(".tiff")
  })

  it("ignores mime parameters and case", () => {
    expect(extFromMime("image/PNG; charset=utf-8")).toBe(".png")
  })

  it("returns undefined for unknown/empty mime", () => {
    expect(extFromMime("application/octet-stream")).toBeUndefined()
    expect(extFromMime(undefined)).toBeUndefined()
    expect(extFromMime(null)).toBeUndefined()
  })
})

describe("buildDownloadName", () => {
  it("keeps original extension when mime is unknown", () => {
    expect(buildDownloadName("a.png", "_edited", "application/x-foo")).toBe(
      "a_edited.png"
    )
  })

  it("uses mime extension when known (CONS-3)", () => {
    // 输入名是 .png 但真实字节是 JPEG → 以响应 Content-Type 为准
    expect(buildDownloadName("a.png", "", "image/jpeg")).toBe("a.jpg")
  })

  it("keeps original extension when mime missing", () => {
    expect(buildDownloadName("photo.jpeg", "_mask")).toBe("photo_mask.jpeg")
  })
})

describe("strokeBounds", () => {
  it("computes bbox without pad", () => {
    const pts: Point[] = [
      { x: 10, y: 20 },
      { x: 30, y: 8 },
    ]
    expect(strokeBounds({ pts })).toEqual({ x: 10, y: 8, w: 20, h: 12 })
  })

  it("expands by pad on all sides", () => {
    const pts: Point[] = [
      { x: 0, y: 0 },
      { x: 10, y: 10 },
    ]
    expect(strokeBounds({ pts }, 5)).toEqual({ x: -5, y: -5, w: 20, h: 20 })
  })

  it("returns null for an empty stroke", () => {
    expect(strokeBounds({ pts: [] })).toBeNull()
  })
})

describe("drawLines", () => {
  function fakeCtx() {
    const calls: string[][] = []
    return {
      calls,
      ctx: {
        strokeStyle: "",
        lineCap: "",
        lineJoin: "",
        lineWidth: 0,
        beginPath: () => calls.push(["beginPath"]),
        moveTo: (x: number, y: number) => calls.push(["moveTo", `${x}`, `${y}`]),
        lineTo: (x: number, y: number) => calls.push(["lineTo", `${x}`, `${y}`]),
        stroke: () => calls.push(["stroke"]),
      } as unknown as CanvasRenderingContext2D,
    }
  }

  it("strokes a multi-point line with round caps", () => {
    const { calls, ctx } = fakeCtx()
    const lines: Line[] = [{ size: 6, pts: [{ x: 1, y: 2 }, { x: 3, y: 4 }, { x: 5, y: 6 }] }]

    drawLines(ctx, lines)

    expect(ctx.strokeStyle).toBe(BRUSH_COLOR)
    expect(ctx.lineCap).toBe("round")
    expect(ctx.lineJoin).toBe("round")
    expect(ctx.lineWidth).toBe(6)
    expect(calls).toEqual([
      ["beginPath"],
      ["moveTo", "1", "2"],
      ["lineTo", "3", "4"],
      ["lineTo", "5", "6"],
      ["stroke"],
    ])
  })

  it("skips empty strokes and zero-size lines", () => {
    const { calls, ctx } = fakeCtx()
    const lines: Line[] = [
      { size: 4, pts: [] },
      { size: 0, pts: [{ x: 1, y: 1 }] },
      { pts: [{ x: 1, y: 1 }] }, // size undefined → 同样跳过
    ]

    drawLines(ctx, lines)

    expect(calls).toEqual([])
  })
})

describe("drawLineWithinRect", () => {
  function fakeCtx() {
    const calls: string[][] = []
    return {
      calls,
      ctx: {
        strokeStyle: "",
        lineCap: "",
        lineJoin: "",
        lineWidth: 0,
        beginPath: () => calls.push(["beginPath"]),
        moveTo: (x: number, y: number) => calls.push(["moveTo", `${x}`, `${y}`]),
        lineTo: (x: number, y: number) => calls.push(["lineTo", `${x}`, `${y}`]),
        stroke: () => calls.push(["stroke"]),
      } as unknown as CanvasRenderingContext2D,
    }
  }

  it("只描与 rect 相交的线段，剩余线段合成一条子路径", () => {
    const { calls, ctx } = fakeCtx()
    const line: Line = {
      size: 4,
      pts: [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 20, y: 0 },
        { x: 30, y: 0 },
        { x: 40, y: 0 },
        { x: 50, y: 0 },
      ],
    }

    // rect 只盖住 x∈[18,22]：命中的是 (10,20) 与 (20,30) 两段
    drawLineWithinRect(ctx, line, { x: 18, y: -2, w: 4, h: 4 })

    expect(ctx.strokeStyle).toBe(BRUSH_COLOR)
    expect(ctx.lineCap).toBe("round")
    expect(ctx.lineJoin).toBe("round")
    expect(ctx.lineWidth).toBe(4)
    expect(calls).toEqual([
      ["beginPath"],
      ["moveTo", "10", "0"],
      ["lineTo", "20", "0"],
      ["lineTo", "30", "0"],
      ["stroke"],
    ])
  })

  it("不相邻的两段各起一条子路径，但仍只 stroke() 一次", () => {
    const { calls, ctx } = fakeCtx()
    const line: Line = {
      size: 4,
      pts: [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 20, y: 100 },
        { x: 30, y: 100 },
        { x: 40, y: 0 },
        { x: 50, y: 0 },
      ],
    }

    // y∈[-5,5] 的 rect 只与第 1 段(y=0)、以及回到 y=0 的第 4/5 段相交；
    // 中间两段在 y=100 外，被跳过
    drawLineWithinRect(ctx, line, { x: -5, y: -5, w: 60, h: 10 })

    expect(calls.filter((c) => c[0] === "stroke")).toHaveLength(1)
    expect(calls).toEqual([
      ["beginPath"],
      ["moveTo", "0", "0"],
      ["lineTo", "10", "0"],
      ["lineTo", "20", "100"],
      ["moveTo", "30", "100"],
      ["lineTo", "40", "0"],
      ["lineTo", "50", "0"],
      ["stroke"],
    ])
  })

  it("单点退回 drawLines（圆头笔帽画圆点，没有 lineTo）", () => {
    const { calls, ctx } = fakeCtx()

    drawLineWithinRect(ctx, { size: 4, pts: [{ x: 3, y: 5 }] }, {
      x: 0,
      y: 0,
      w: 10,
      h: 10,
    })

    expect(calls).toEqual([["beginPath"], ["moveTo", "3", "5"], ["stroke"]])
  })

  it("没有 size 的笔迹什么也不画", () => {
    const { calls, ctx } = fakeCtx()

    drawLineWithinRect(ctx, { pts: [{ x: 1, y: 1 }, { x: 2, y: 2 }] }, {
      x: 0,
      y: 0,
      w: 10,
      h: 10,
    })

    expect(calls).toEqual([])
  })
})

describe("click type", () => {
  it("detects right/mid click from nativeEvent", () => {
    const mk = (button: number) =>
      ({ nativeEvent: { button } }) as never
    expect(isRightClick(mk(2))).toBe(true)
    expect(isRightClick(mk(0))).toBe(false)
    expect(isMidClick(mk(1))).toBe(true)
    expect(isMidClick(mk(2))).toBe(false)
  })
})

describe("mouseXY", () => {
  it("uses offsetX/offsetY for mouse events", () => {
    const ev = { nativeEvent: { offsetX: 12, offsetY: 34 } } as never
    expect(mouseXY(ev)).toEqual({ x: 12, y: 34 })
  })

  it("scales touch coordinates by canvas rect", () => {
    const target = {
      getBoundingClientRect: () => ({ x: 100, y: 50, width: 200, height: 100 }),
      offsetWidth: 400,
      offsetHeight: 200,
    }
    const touch = { clientX: 200, clientY: 100 }
    expect(touchPointXY(touch as Touch, target as never)).toEqual({
      x: 200, // (200-100)/200*400
      y: 100, // (100-50)/100*200
    })
  })
})

describe("fitCanvasBitmap", () => {
  // 4032x6048 的真实照片超过 16Mi px 上限，位图按 sqrt(maxPixels/px) 缩小到 3344x5017。
  const MAX = 16 * 1024 * 1024

  it("covers every bitmap pixel at the exact drawn size", () => {
    const fit = fitCanvasBitmap(4032, 6048, MAX)
    expect(fit.bitmapWidth).toBe(3344)
    expect(fit.bitmapHeight).toBe(5017)
    // 反算缩放让 drawImage 的目标矩形正好铺满位图：用统一 scale 时会差
    // 5017 - 6048*0.829456844 = 0.445px，Firefox/WebKit 把最后一行画成半透明。
    expect(4032 * fit.scaleX).toBeCloseTo(fit.bitmapWidth, 10)
    expect(6048 * fit.scaleY).toBeCloseTo(fit.bitmapHeight, 10)
  })

  it("keeps the bitmap within maxPixels", () => {
    const fit = fitCanvasBitmap(4032, 6048, MAX)
    expect(fit.bitmapWidth * fit.bitmapHeight).toBeLessThanOrEqual(MAX)
  })

  it("leaves images under the limit untouched", () => {
    expect(fitCanvasBitmap(640, 480, MAX)).toEqual({
      bitmapWidth: 640,
      bitmapHeight: 480,
      scaleX: 1,
      scaleY: 1,
    })
  })

  it("falls back to 1x1 for empty sizes", () => {
    expect(fitCanvasBitmap(0, 0, MAX)).toEqual({
      bitmapWidth: 1,
      bitmapHeight: 1,
      scaleX: 1,
      scaleY: 1,
    })
  })
})

describe("generateMask guard", () => {
  it("throws when 2d context is unavailable", () => {
    // jsdom 没装 canvas 后端时，getContext 会走 notImplemented：先往 virtual console
    // 打一条带调用栈的 "Not implemented" 错误（vitest 转发到 stderr，形似测试崩溃）
    // 再返回 null。这里显式 stub 成 null：既消掉这段误导性输出，也把「取不到 2d ctx」
    // 这个前置条件钉死，测试断言的仍是 generateMask 的空 ctx 防御分支。
    const spy = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue(null)
    try {
      expect(() => generateMask(10, 10, [])).toThrow(
        "could not retrieve mask canvas"
      )
    } finally {
      spy.mockRestore()
    }
  })

  it("requests a software-backed 2d context", () => {
    // 全分辨率掩膜画布必须走 CPU 位图：Chromium 的加速画布把内容放 GPU 进程，
    // GPU 进程崩溃会丢内容，toBlob 出全透明掩膜 → 后端收到空掩膜只重编码原图
    // （用户看到的「能导入、不能 inpaint」）。这里钉住 willReadFrequently，
    // 防止以后被顺手删掉。
    const spy = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue(null)
    try {
      expect(() => generateMask(10, 10, [])).toThrow(
        "could not retrieve mask canvas"
      )
      expect(spy).toHaveBeenCalledWith("2d", { willReadFrequently: true })
    } finally {
      spy.mockRestore()
    }
  })
})

describe("lineGroupToMaskStrokes", () => {
  it("converts image-coordinate points and brush size", () => {
    expect(
      lineGroupToMaskStrokes([
        { size: 12, pts: [{ x: 1.5, y: 2 }, { x: 3, y: 4.25 }] },
      ])
    ).toEqual([{ pts: [[1.5, 2], [3, 4.25]], size: 12 }])
  })

  it("drops entries drawLines would skip", () => {
    expect(
      lineGroupToMaskStrokes([
        { size: 4, pts: [] },
        { size: 0, pts: [{ x: 1, y: 1 }] },
        { pts: [{ x: 1, y: 1 }] },
        { size: 5, pts: [{ x: 2, y: 2 }] },
      ])
    ).toEqual([{ pts: [[2, 2]], size: 5 }])
  })

  it("returns an empty list for an empty group", () => {
    expect(lineGroupToMaskStrokes([])).toEqual([])
  })
})

describe("generateMaskPreview", () => {
  function fakeCtx() {
    const calls: unknown[][] = []
    const ctx = {
      strokeStyle: "",
      lineCap: "",
      lineJoin: "",
      lineWidth: 0,
      setTransform: (...args: number[]) => calls.push(["setTransform", ...args]),
      drawImage: (...args: unknown[]) => calls.push(["drawImage", ...args]),
      beginPath: () => calls.push(["beginPath"]),
      moveTo: (...args: number[]) => calls.push(["moveTo", ...args]),
      lineTo: (...args: number[]) => calls.push(["lineTo", ...args]),
      stroke: () => calls.push(["stroke"]),
    } as unknown as CanvasRenderingContext2D
    return { calls, ctx }
  }

  it("caps the canvas at MASK_CANVAS_MAX_PIXELS and scales the context", () => {
    const { calls, ctx } = fakeCtx()
    const spy = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue(ctx)
    try {
      const canvas = generateMaskPreview(4000, 3000, [
        [{ size: 10, pts: [{ x: 0, y: 0 }, { x: 4000, y: 3000 }] }],
      ])
      // 位图尺寸取整会多出几百像素（4Mi 是近似上限，Editor 的显示画布同理）
      expect(canvas.width * canvas.height).toBeLessThanOrEqual(
        MASK_CANVAS_MAX_PIXELS * 1.001
      )
      expect(calls[0]).toEqual([
        "setTransform",
        canvas.width / 4000,
        0,
        0,
        canvas.height / 3000,
        0,
        0,
      ])
      // 笔迹仍按原图坐标进入绘制路径（由变换矩阵缩放）
      expect(calls).toContainEqual(["moveTo", 0, 0])
      expect(calls).toContainEqual(["lineTo", 4000, 3000])
    } finally {
      spy.mockRestore()
    }
  })

  it("keeps 1:1 for small images", () => {
    const { ctx } = fakeCtx()
    const spy = vi
      .spyOn(HTMLCanvasElement.prototype, "getContext")
      .mockReturnValue(ctx)
    try {
      const canvas = generateMaskPreview(320, 200, [])
      expect(canvas.width).toBe(320)
      expect(canvas.height).toBe(200)
    } finally {
      spy.mockRestore()
    }
  })
})

describe("parseImageSize", () => {
  const ascii = (s: string) => [...s].map((c) => c.charCodeAt(0))

  it("reads PNG IHDR", () => {
    const bytes = new Uint8Array(24)
    bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
    bytes.set([0x49, 0x48, 0x44, 0x52], 12)
    bytes.set([0x00, 0x00, 0x0d, 0x80], 16) // 3456
    bytes.set([0x00, 0x00, 0x07, 0xd0], 20) // 2000
    expect(parseImageSize(bytes)).toEqual({ width: 3456, height: 2000 })
  })

  it("reads JPEG SOF and applies EXIF orientation", () => {
    // SOI + APP1(Exif orientation=6) + SOF0(32x16 raw) + EOI
    const tiff = [
      0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, // II*\0, IFD @8
      0x01, 0x00, // 1 entry
      0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, // tag 0x0112 SHORT 1
      0x06, 0x00, 0x00, 0x00, // orientation 6
      0x00, 0x00, 0x00, 0x00, // next IFD
    ]
    const app1 = [...ascii("Exif"), 0, 0, ...tiff]
    const sof = [0x08, 0x00, 0x10, 0x00, 0x20, 0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]
    const bytes = new Uint8Array([
      0xff, 0xd8,
      0xff, 0xe1, (app1.length + 2) >> 8, (app1.length + 2) & 0xff, ...app1,
      0xff, 0xc0, 0x00, (sof.length + 2), ...sof,
      0xff, 0xd9,
    ])
    expect(parseImageSize(bytes)).toEqual({
      width: 16,
      height: 32,
      orientation: 6,
    })
  })

  it("reads JPEG without EXIF", () => {
    const sof = [0x08, 0x00, 0x10, 0x00, 0x20]
    const bytes = new Uint8Array([
      0xff, 0xd8, 0xff, 0xc0, 0x00, (sof.length + 2), ...sof, 0xff, 0xd9,
    ])
    expect(parseImageSize(bytes)).toEqual({
      width: 32,
      height: 16,
      orientation: 1,
    })
  })

  it("reads GIF / BMP / WebP dimensions", () => {
    const gif = new Uint8Array([
      ...ascii("GIF89a"), 100, 0, 50, 0, 0, 0, 0,
    ])
    expect(parseImageSize(gif)).toEqual({ width: 100, height: 50 })

    const bmp = new Uint8Array(26)
    bmp.set([0x42, 0x4d])
    bmp.set([100, 0, 0, 0], 18)
    bmp.set([0x38, 0xff, 0xff, 0xff], 22) // -200 => 200
    expect(parseImageSize(bmp)).toEqual({ width: 100, height: 200 })

    const webp = new Uint8Array(30)
    webp.set([...ascii("RIFF"), 0, 0, 0, 0, ...ascii("WEBP")])
    webp.set(ascii("VP8X"), 12)
    webp.set([0, 0, 0, 0], 20)
    webp.set([99, 0, 0], 24) // width - 1
    webp.set([49, 0, 0], 27) // height - 1
    expect(parseImageSize(webp)).toEqual({ width: 100, height: 50 })
  })

  it("returns null for unknown/short input", () => {
    expect(parseImageSize(new Uint8Array([1, 2, 3]))).toBeNull()
    expect(parseImageSize(new Uint8Array([0xff, 0xd8]))).toBeNull()
  })
})
