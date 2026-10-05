import { type ClassValue, clsx } from "clsx"
import { SyntheticEvent } from "react"
import { twMerge } from "tailwind-merge"
import { Line, LineGroup, Point } from "./types"
import { BRUSH_COLOR } from "./const"

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

export function canvasToImage(
  canvas: HTMLCanvasElement
): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()

    image.addEventListener("load", () => {
      resolve(image)
    })

    image.addEventListener("error", (error) => {
      reject(error)
    })

    image.src = canvas.toDataURL()
  })
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
  const ctx = maskCanvas.getContext("2d")
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
