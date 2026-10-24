import { type ClassValue, clsx } from "clsx"
import { SyntheticEvent } from "react"
import { twMerge } from "tailwind-merge"
import { DisplayImage, Line, LineGroup, MaskStroke, Point } from "./types"
import {
  BRUSH_COLOR,
  IMAGE_CANVAS_MAX_PIXELS,
  MASK_CANVAS_MAX_PIXELS,
} from "./const"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

let _keepAliveTimer: ReturnType<typeof setInterval> | null = null

export function keepGUIAlive() {
  async function getRequest(url = "") {
    const response = await fetch(url, {
      method: "GET",
      cache: "no-cache",
    })
    return response.json()
  }

  const keepAliveServer = () => {
    const url = document.location
    const route = "/flaskwebgui-keep-server-alive"
    getRequest(url + route).then((data) => {
      return data
    })
  }

  if (_keepAliveTimer !== null) {
    return
  }
  const intervalRequest = 3 * 1000
  keepAliveServer()
  _keepAliveTimer = setInterval(keepAliveServer, intervalRequest)
}

export function stopGUIAlive() {
  if (_keepAliveTimer !== null) {
    clearInterval(_keepAliveTimer)
    _keepAliveTimer = null
  }
}

export function loadImage(image: HTMLImageElement, src: string) {
  return new Promise((resolve, reject) => {
    const initSRC = image.src
    const img = image
    img.onload = resolve
    img.onerror = (err) => {
      img.src = initSRC
      reject(err)
    }
    img.src = src
  })
}

export async function blobToImage(blob: Blob) {
  const dataURL = URL.createObjectURL(blob)
  const newImage = new Image()
  await loadImage(newImage, dataURL)
  URL.revokeObjectURL(dataURL)
  return newImage
}

/**
 * 把画布变成一个 HTMLImageElement。
 *
 * 原来用的是同步的 ``canvas.toDataURL()``：一张 4096x3072 的掩膜实测
 * 阻塞主线程 66ms（Chromium）/ 54ms（Firefox）/ **952ms（WebKit）**，
 * 而 ``showPrevMask``（悬停「重跑上次掩膜」按钮）和 ``runInpainting``
 * 的重跑分支都会走到这里 —— 鼠标一悬停整页就冻一下。
 * 换成 ``toBlob``：编码在后台线程做，主线程只负责把 blob 包成图。
 *
 * 加载完立刻 revoke：与 runRenderablePlugin / runInteractiveSeg 里处理掩膜
 * 的既有做法一致（Image 拿到的是解码后的位图，之后照常 drawImage），
 * 否则每次悬停都会漏一个 blob URL 出来。
 */
export function canvasToImage(
  canvas: HTMLCanvasElement
): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Expected toBlob() to be defined"))
        return
      }
      const url = URL.createObjectURL(blob)
      const image = new Image()
      image.addEventListener("load", () => {
        URL.revokeObjectURL(url)
        resolve(image)
      })
      image.addEventListener("error", (error) => {
        URL.revokeObjectURL(url)
        reject(error)
      })
      image.src = url
    }, "image/png")
  })
}

// 显示用画布的位图尺寸与绘制缩放（大图会被缩到 maxPixels 以内，见 Editor 的
// IMAGE_CANVAS_MAX_PIXELS）。位图尺寸要取整，而 drawImage 的目标矩形是
// width*scale 这样的分数；若两者用同一个等比 scale，取整后位图可能比目标矩形
// 高出不到 1px（4032x6048 → 位图 5017 行，目标矩形 5016.55 行）。Firefox/WebKit
// 会按覆盖率把这一行画成半透明（实测 alpha 142/128），再被 CSS 拉回原图尺寸，
// 就是底部 1~2px 的透明渐变条。所以缩放按位图实际尺寸反算，保证整块位图被填满。
export function fitCanvasBitmap(
  width: number,
  height: number,
  maxPixels: number
): {
  bitmapWidth: number
  bitmapHeight: number
  scaleX: number
  scaleY: number
} {
  if (width <= 0 || height <= 0) {
    return { bitmapWidth: 1, bitmapHeight: 1, scaleX: 1, scaleY: 1 }
  }
  const pixels = width * height
  const scale = pixels > maxPixels ? Math.sqrt(maxPixels / pixels) : 1
  const bitmapWidth = Math.max(1, Math.round(width * scale))
  const bitmapHeight = Math.max(1, Math.round(height * scale))
  return {
    bitmapWidth,
    bitmapHeight,
    scaleX: bitmapWidth / width,
    scaleY: bitmapHeight / height,
  }
}

// ---------------------------------------------------------------------------
// 图片 header 解析：打开大图时先按显示尺寸解码，必须先知道原图尺寸（和 EXIF
// 方向），而 `createImageBitmap(file, {resizeWidth})` 又需要目标尺寸。这里只读
// 文件头（JPEG SOF / PNG IHDR / WebP / GIF / BMP），不触发整图解码。
// ---------------------------------------------------------------------------

export interface ParsedImageSize {
  width: number
  height: number
  /** EXIF orientation（只有 JPEG 会解析），已作用到 width/height。 */
  orientation?: number
}

function u16be(b: Uint8Array, o: number) {
  return (b[o] << 8) | b[o + 1]
}

function u16le(b: Uint8Array, o: number) {
  return b[o] | (b[o + 1] << 8)
}

function u32be(b: Uint8Array, o: number) {
  return ((b[o] << 24) >>> 0) + (b[o + 1] << 16) + (b[o + 2] << 8) + b[o + 3]
}

function ascii(b: Uint8Array, o: number, len: number) {
  return String.fromCharCode(...b.subarray(o, o + len))
}

function isJpegSof(marker: number) {
  return (
    (marker >= 0xc0 && marker <= 0xc3) ||
    (marker >= 0xc5 && marker <= 0xc7) ||
    (marker >= 0xc9 && marker <= 0xcb) ||
    (marker >= 0xcd && marker <= 0xcf)
  )
}

/** 从 APP1 EXIF 里读 orientation（1~8），没有则返回 undefined。 */
function readExifOrientation(b: Uint8Array, off: number, len: number) {
  if (len < 14 || ascii(b, off, 6) !== "Exif\u0000\u0000") {
    return undefined
  }
  const tiff = off + 6
  const end = Math.min(b.length, off + len)
  const order = ascii(b, tiff, 2)
  if (order !== "II" && order !== "MM") {
    return undefined
  }
  const le = order === "II"
  const read16 = (o: number) => (le ? u16le(b, o) : u16be(b, o))
  const read32 = (o: number) =>
    le
      ? b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | ((b[o + 3] << 24) >>> 0)
      : u32be(b, o)
  if (read16(tiff + 2) !== 42) {
    return undefined
  }
  const ifd = tiff + read32(tiff + 4)
  if (ifd + 2 > end) {
    return undefined
  }
  const count = read16(ifd)
  for (let i = 0; i < count; i++) {
    const entry = ifd + 2 + i * 12
    if (entry + 12 > end) {
      return undefined
    }
    if (read16(entry) === 0x0112) {
      const value = read16(entry + 8)
      return value >= 1 && value <= 8 ? value : undefined
    }
  }
  return undefined
}

function parseJpeg(b: Uint8Array): ParsedImageSize | null {
  let off = 2
  let orientation = 1
  while (off + 4 <= b.length) {
    if (b[off] !== 0xff) {
      return null
    }
    let marker = b[off + 1]
    // 允许 FF 填充字节
    while (marker === 0xff && off + 2 < b.length) {
      off++
      marker = b[off + 1]
    }
    // SOI / TEM / RSTn 没有长度字段
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      off += 2
      continue
    }
    // EOI / SOS 之后不会再有 SOF
    if (marker === 0xd9 || marker === 0xda || off + 4 > b.length) {
      return null
    }
    const len = u16be(b, off + 2)
    if (len < 2) {
      return null
    }
    if (isJpegSof(marker)) {
      if (off + 9 > b.length) {
        return null
      }
      const height = u16be(b, off + 5)
      const width = u16be(b, off + 7)
      if (!width || !height) {
        return null
      }
      // orientation 5~8 把宽高交换（浏览器显示的是旋转后的图）
      const swap = orientation >= 5 && orientation <= 8
      return {
        width: swap ? height : width,
        height: swap ? width : height,
        orientation,
      }
    }
    if (marker === 0xe1) {
      orientation = readExifOrientation(b, off + 4, len - 2) ?? orientation
    }
    off += 2 + len
  }
  return null
}

function parseWebp(b: Uint8Array): ParsedImageSize | null {
  if (b.length < 30 || ascii(b, 8, 4) !== "WEBP") {
    return null
  }
  const chunk = ascii(b, 12, 4)
  if (chunk === "VP8X") {
    const width = 1 + (b[24] | (b[25] << 8) | (b[26] << 16))
    const height = 1 + (b[27] | (b[28] << 8) | (b[29] << 16))
    return { width, height }
  }
  if (chunk === "VP8 " && b.length >= 30) {
    if (b[23] !== 0x9d || b[24] !== 0x01 || b[25] !== 0x2a) {
      return null
    }
    return {
      width: u16le(b, 26) & 0x3fff,
      height: u16le(b, 28) & 0x3fff,
    }
  }
  if (chunk === "VP8L" && b.length >= 25 && b[20] === 0x2f) {
    const bits =
      b[21] | (b[22] << 8) | (b[23] << 16) | ((b[24] << 24) >>> 0)
    return {
      width: 1 + (bits & 0x3fff),
      height: 1 + ((bits >>> 14) & 0x3fff),
    }
  }
  return null
}

/** 解析图片头字节；不认识的格式返回 null。 */
export function parseImageSize(b: Uint8Array): ParsedImageSize | null {
  if (b.length >= 24 && b[0] === 0x89 && ascii(b, 1, 3) === "PNG") {
    return { width: u32be(b, 16), height: u32be(b, 20) }
  }
  if (b.length >= 10 && ascii(b, 0, 3) === "GIF") {
    return { width: u16le(b, 6), height: u16le(b, 8) }
  }
  if (b.length >= 26 && b[0] === 0x42 && b[1] === 0x4d) {
    return {
      width: Math.abs(
        b[18] | (b[19] << 8) | (b[20] << 16) | ((b[21] << 24) >>> 0)
      ),
      height: Math.abs(
        b[22] | (b[23] << 8) | (b[24] << 16) | ((b[25] << 24) >>> 0)
      ),
    }
  }
  if (b.length >= 4 && b[0] === 0xff && b[1] === 0xd8) {
    return parseJpeg(b)
  }
  return parseWebp(b)
}

/** 只读文件头（最多 2MB，找不到 SOF 才回退整文件）拿原图尺寸。 */
export async function readImageSize(blob: Blob): Promise<ParsedImageSize | null> {
  const HEAD_BYTES = 2 * 1024 * 1024
  const head = new Uint8Array(
    await blob.slice(0, Math.min(blob.size, HEAD_BYTES)).arrayBuffer()
  )
  const parsed = parseImageSize(head)
  if (parsed || blob.size <= HEAD_BYTES) {
    return parsed
  }
  return parseImageSize(new Uint8Array(await blob.arrayBuffer()))
}

function loadImageFromSrc(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = reject
    image.src = src
  })
}

function drawToCanvas(
  source: CanvasImageSource,
  width: number,
  height: number
): HTMLCanvasElement {
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext("2d")
  if (!ctx) {
    throw new Error("could not retrieve display canvas")
  }
  ctx.drawImage(source, 0, 0, width, height)
  return canvas
}

/**
 * 按显示位图尺寸解码一张图片。
 *
 * `createImageBitmap(blob, {resizeWidth, resizeHeight})` 在解码阶段就缩放，
 * 不会先落一张全分辨率位图；Safari/旧浏览器不支持时回退到 `<img>` 全解码后
 * 画进目标尺寸的画布（结果一致，只是慢一点）。`src` 保留源字节的 blob URL，
 * 供 revision 失效时回传。
 */
export async function createDisplayImage(
  src: string,
  blob: Blob,
  logicalSize?: { width: number; height: number } | null
): Promise<DisplayImage> {
  const parsed = logicalSize ?? (await readImageSize(blob))
  if (!parsed) {
    const full = await loadImageFromSrc(src)
    const logical = {
      width: full.naturalWidth,
      height: full.naturalHeight,
    }
    const fit = fitCanvasBitmap(
      logical.width,
      logical.height,
      IMAGE_CANVAS_MAX_PIXELS
    )
    return {
      src,
      bitmap: drawToCanvas(full, fit.bitmapWidth, fit.bitmapHeight),
      width: fit.bitmapWidth,
      height: fit.bitmapHeight,
      logicalWidth: logical.width,
      logicalHeight: logical.height,
    }
  }

  const fit = fitCanvasBitmap(
    parsed.width,
    parsed.height,
    IMAGE_CANVAS_MAX_PIXELS
  )
  let bitmap: CanvasImageSource
  if (typeof createImageBitmap === "function") {
    try {
      bitmap = await createImageBitmap(blob, {
        resizeWidth: fit.bitmapWidth,
        resizeHeight: fit.bitmapHeight,
        resizeQuality: "high",
        imageOrientation: "from-image",
      })
    } catch (e) {
      console.warn("createImageBitmap resize failed, falling back", e)
      const full = await loadImageFromSrc(src)
      bitmap = drawToCanvas(full, fit.bitmapWidth, fit.bitmapHeight)
    }
  } else {
    const full = await loadImageFromSrc(src)
    bitmap = drawToCanvas(full, fit.bitmapWidth, fit.bitmapHeight)
  }
  return {
    src,
    bitmap,
    width: fit.bitmapWidth,
    height: fit.bitmapHeight,
    logicalWidth: parsed.width,
    logicalHeight: parsed.height,
  }
}

export function getErrorMessage(e: unknown): string {
  if (e instanceof Error) {
    return e.message
  }
  if (typeof e === "string") {
    return e
  }
  try {
    return JSON.stringify(e)
  } catch {
    return String(e)
  }
}

export function srcToFile(src: string, fileName: string, mimeType: string) {
  return fetch(src)
    .then(function (res) {
      return res.arrayBuffer()
    })
    .then(function (buf) {
      return new File([buf], fileName, { type: mimeType })
    })
}

// ---------------------------------------------------------------------------
// 保存文件名的扩展名（CONS-3）
//
// 输入文件的 `file.type` / `file.name` 来自浏览器按**扩展名**的猜测，而服务端
// 返回的字节格式跟随输入的真实字节（`helper.decode_base64_to_image` 以
// `Image.format` 为准）。两者不一致时（例如改过名的 .png 实为 JPEG），
// 落盘文件就会"名不副实"。所以扩展名一律以响应的 Content-Type 为准。
// ---------------------------------------------------------------------------

const EXT_BY_MIME: Record<string, string> = {
  "image/jpeg": ".jpg",
  "image/jpg": ".jpg",
  "image/png": ".png",
  "image/webp": ".webp",
  "image/gif": ".gif",
  "image/bmp": ".bmp",
  "image/tiff": ".tiff",
}

/** 由 MIME 推扩展名（含点）；未知类型返回 undefined。 */
export function extFromMime(mime?: string | null): string | undefined {
  if (!mime) {
    return undefined
  }
  return EXT_BY_MIME[mime.split(";")[0].trim().toLowerCase()]
}

/**
 * 由 blob: URL 取出实际字节的 MIME（渲染结果的 blob 类型 = 响应 Content-Type）。
 * 拿不到时返回 undefined，调用方回退到原来的扩展名。
 */
export async function mimeFromSrc(src: string): Promise<string | undefined> {
  if (!src.startsWith("blob:")) {
    return undefined
  }
  try {
    const blob = await (await fetch(src)).blob()
    return blob.type || undefined
  } catch {
    return undefined
  }
}

/**
 * 生成保存用文件名：`<stem><suffix><ext>`。
 * `mime` 未给或未知扩展名时保留原扩展名。
 */
export function buildDownloadName(
  filename: string,
  suffix: string,
  mime?: string | null
): string {
  const stem = filename.replace(/\.[\w\d_-]+$/i, "")
  const origExt = filename.match(/\.[\w\d_-]+$/i)?.[0] ?? ""
  const ext = extFromMime(mime) ?? origExt
  return `${stem}${suffix}${ext}`
}

export async function askWritePermission() {
  try {
    // The clipboard-write permission is granted automatically to pages
    // when they are the active tab. So it's not required, but it's more safe.
    const { state } = await navigator.permissions.query({
      name: "clipboard-write" as PermissionName,
    })
    return state === "granted"
  } catch {
    // Browser compatibility / Security error (ONLY HTTPS) ...
    return false
  }
}

export function canvasToBlob(
  canvas: HTMLCanvasElement,
  mime = "image/png"
): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((d) => {
      if (d) {
        resolve(d)
      } else {
        reject(new Error("Expected toBlob() to be defined"))
      }
    }, mime)
  )
}

const setToClipboard = async (blob: Blob) => {
  const data = [new ClipboardItem({ [blob.type]: blob })]
  await navigator.clipboard.write(data)
}

export function isRightClick(ev: SyntheticEvent) {
  const mouseEvent = ev.nativeEvent as MouseEvent
  return mouseEvent.button === 2
}

export function isMidClick(ev: SyntheticEvent) {
  const mouseEvent = ev.nativeEvent as MouseEvent
  return mouseEvent.button === 1
}

export async function copyCanvasImage(canvas: HTMLCanvasElement) {
  const blob = await canvasToBlob(canvas, "image/png")
  try {
    await setToClipboard(blob)
  } catch {
    console.error("Copy image failed!")
  }
}

export function downloadImage(uri: string, name: string) {
  const link = document.createElement("a")
  link.href = uri
  link.download = name
  link.rel = "noopener"
  link.style.display = "none"

  // 锚点必须先挂载到文档中，click() 才会触发下载。
  // 之前用的是「未挂载节点 + 合成 MouseEvent」，Chromium/Firefox 行为不一致，
  // 下载经常不触发或被取消。
  document.body.appendChild(link)
  link.click()

  // 延迟移除，等浏览器真正开始读取 blob 再回收
  setTimeout(() => {
    link.remove()
  }, 1000)
}

// 返回图片坐标系（原图像素）下的鼠标位置。
// 既接受 React 的 SyntheticEvent，也接受原生 MouseEvent（getCoalescedEvents 的结果）。
export function mouseXY(ev: SyntheticEvent | MouseEvent) {
    const mouseEvent =
        'nativeEvent' in ev ? (ev.nativeEvent as MouseEvent) : ev
    // Handle mask drawing coordinate on mobile/tablet devices.
    // On touchend `touches` is empty, so fall back to `changedTouches`.
    if ('touches' in ev) {
        const touchEvent = ev as unknown as TouchEvent
        const touch =
            touchEvent.touches[0] || touchEvent.changedTouches[0]
        if (touch) {
            const target = ev.target as HTMLCanvasElement
            const rect = target.getBoundingClientRect()
            return {
                x: (touch.clientX - rect.x) / rect.width * target.offsetWidth,
                y: (touch.clientY - rect.y) / rect.height * target.offsetHeight,
            }
        }
    }
    return {x: mouseEvent.offsetX, y: mouseEvent.offsetY}
}

// 把某个具体的 Touch 坐标换算成画布（图片）坐标。
// 多指场景下需要根据特定手指（identifier）来绘制，不能用 touches[0]。
export function touchPointXY(touch: Touch, target: HTMLElement): Point {
  const rect = target.getBoundingClientRect()
  return {
    x: ((touch.clientX - rect.x) / rect.width) * target.offsetWidth,
    y: ((touch.clientY - rect.y) / rect.height) * target.offsetHeight,
  }
}

/**
 * 把画布笔迹转成后端 `mask_strokes`。与 drawLines 的过滤条件保持一致：
 * 空笔迹和没笔宽的条目本来就不产生像素。
 */
export function lineGroupToMaskStrokes(lineGroup: LineGroup): MaskStroke[] {
  const strokes: MaskStroke[] = []
  for (const line of lineGroup) {
    if (!line?.pts.length || !line.size || line.size <= 0) {
      continue
    }
    strokes.push({
      pts: line.pts.map((pt) => [pt.x, pt.y]),
      size: line.size,
    })
  }
  return strokes
}

// 描一条笔迹。坐标/线宽都按 ctx 当前的变换矩阵解释，
// 因此显示画布可以用 ctx.setTransform(scale,0,0,scale,0,0) 缩小后仍按原图坐标绘制。
export function drawLines(
  ctx: CanvasRenderingContext2D,
  lines: LineGroup,
  color = BRUSH_COLOR
) {
  ctx.strokeStyle = color
  ctx.lineCap = "round"
  ctx.lineJoin = "round"

  for (const line of lines) {
    if (!line?.pts.length || !line.size) {
      continue
    }
    ctx.lineWidth = line.size
    ctx.beginPath()
    ctx.moveTo(line.pts[0].x, line.pts[0].y)
    for (let i = 1; i < line.pts.length; i++) {
      ctx.lineTo(line.pts[i].x, line.pts[i].y)
    }
    ctx.stroke()
  }
}

/**
 * 只描一条笔迹里「按笔宽外扩后与 rect 相交」的线段，其余掠过。
 *
 * 命中的线段全部拼进同一个 Path（多个 moveTo/lineTo 子路径）再做**一次**
 * stroke()：Canvas2D 的一次 stroke 把整条路径当成一个图形一次性上色，自交/相邻处
 * 只着色一次（正是 drawLines 依赖的性质），所以 rect 内的像素与「整条路径一次
 * stroke」逐像素一致 —— 落到 rect 外面的线段本来就会被调用方 clip 掉，略过它们
 * 不改变任何像素。
 *
 * 用途是绘制中的笔画（flushStroke）：脏矩形通常只盖住笔尖附近的一小段，这样每帧
 * 的路径长度与已画点数无关，消掉“每帧重描整条笔画”的 O(n²)。
 * pts 少于 2 个时退回 drawLines（单点靠圆头笔帽画一个圆点，是既有行为）。
 */
export function drawLineWithinRect(
  ctx: CanvasRenderingContext2D,
  line: Line,
  rect: { x: number; y: number; w: number; h: number },
  pad = 0,
  color = BRUSH_COLOR
) {
  const pts = line.pts
  if (!line.size || pts.length < 2) {
    drawLines(ctx, [line], color)
    return
  }
  ctx.beginPath()
  ctx.strokeStyle = color
  ctx.lineCap = "round"
  ctx.lineJoin = "round"
  ctx.lineWidth = line.size

  const rx0 = rect.x
  const ry0 = rect.y
  const rx1 = rect.x + rect.w
  const ry1 = rect.y + rect.h
  let inRun = false
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1]
    const b = pts[i]
    // 线段外扩 pad 后仍与 rect 无交 => 在 rect 里不可能留下像素，直接跳过
    const hit =
      Math.min(a.x, b.x) - pad <= rx1 &&
      Math.max(a.x, b.x) + pad >= rx0 &&
      Math.min(a.y, b.y) - pad <= ry1 &&
      Math.max(a.y, b.y) + pad >= ry0
    if (hit) {
      if (!inRun) {
        ctx.moveTo(a.x, a.y)
        inRun = true
      }
      ctx.lineTo(b.x, b.y)
    } else {
      inRun = false
    }
  }
  ctx.stroke()
}

// 一条笔迹里所有点的包围盒（笔刷是圆头，调用方需要按 lineWidth/2 外扩）
export function strokeBounds(line: Line, pad = 0) {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const pt of line.pts) {
    if (pt.x < minX) minX = pt.x
    if (pt.y < minY) minY = pt.y
    if (pt.x > maxX) maxX = pt.x
    if (pt.y > maxY) maxY = pt.y
  }
  if (minX === Infinity) {
    return null
  }
  return {
    x: minX - pad,
    y: minY - pad,
    w: maxX - minX + pad * 2,
    h: maxY - minY + pad * 2,
  }
}

export const generateMask = (
  imageWidth: number,
  imageHeight: number,
  lineGroups: LineGroup[],
  maskImages: HTMLImageElement[] = [],
  lineGroupsColor: string = "white"
): HTMLCanvasElement => {
  const maskCanvas = document.createElement("canvas")
  maskCanvas.width = imageWidth
  maskCanvas.height = imageHeight
  // 掩膜要按原图分辨率生成（后端要求 mask 与 image 同尺寸），大图上就是几十 MP
  // 的一块画布（83.6MP ≈ 334MB）。Chromium 的加速 2D 画布把这块位图放在 GPU 进程里：
  // GPU 进程一崩（Linux 上 DRI/驱动加载失败常见，日志 `GPU process exited unexpectedly`
  // / `GPU state invalid after WaitForGetOffsetInRange`），位图连同内容一起丢，
  // toBlob 出来就是全透明掩膜 —— 后端收到空掩膜只会把原图重新编码一遍，
  // 表现成「能导入、不能 inpaint」。willReadFrequently 让这块画布走 CPU 位图：
  // 内容不进 GPU 显存（也不再申请 334MB 纹理），GPU 进程崩了也与它无关。
  const ctx = maskCanvas.getContext("2d", { willReadFrequently: true })
  if (!ctx) {
    throw new Error("could not retrieve mask canvas")
  }

  maskImages.forEach((maskImage) => {
    ctx.drawImage(maskImage, 0, 0, imageWidth, imageHeight)
  })

  lineGroups.forEach((lineGroup) => {
    drawLines(ctx, lineGroup, lineGroupsColor)
  })

  return maskCanvas
}

/**
 * 生成只用于显示的掩膜预览（最多 MASK_CANVAS_MAX_PIXELS）。
 *
 * 坐标与笔宽仍是原图单位，通过 `ctx.scale` 映射到位图；显示画布本来也按同一
 * 上限缩小，所以视觉上与全分辨率掩膜一致。`showPrevMask` 只把它画到屏幕上，
 * 从不发给服务端，没必要生成整张原图大小的画布再 toBlob（WebKit 上 30MP 要 1.6s）。
 */
export const generateMaskPreview = (
  imageWidth: number,
  imageHeight: number,
  lineGroups: LineGroup[],
  maskImages: HTMLImageElement[] = [],
  lineGroupsColor: string = BRUSH_COLOR
): HTMLCanvasElement => {
  const fit = fitCanvasBitmap(
    imageWidth,
    imageHeight,
    MASK_CANVAS_MAX_PIXELS
  )
  const canvas = document.createElement("canvas")
  canvas.width = fit.bitmapWidth
  canvas.height = fit.bitmapHeight
  const ctx = canvas.getContext("2d", { willReadFrequently: true })
  if (!ctx) {
    throw new Error("could not retrieve mask canvas")
  }
  ctx.setTransform(fit.scaleX, 0, 0, fit.scaleY, 0, 0)
  maskImages.forEach((maskImage) => {
    ctx.drawImage(maskImage, 0, 0, imageWidth, imageHeight)
  })
  lineGroups.forEach((lineGroup) => {
    drawLines(ctx, lineGroup, lineGroupsColor)
  })
  return canvas
}

export const convertToBase64 = (fileOrBlob: File | Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (event) => {
      const base64String = event.target?.result as string
      resolve(base64String)
    }
    reader.onerror = (error) => {
      reject(error)
    }
    reader.readAsDataURL(fileOrBlob)
  })
}
