import {
  SyntheticEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react"
import { useToast } from "@/components/ui/use-toast"
import {
  ReactZoomPanPinchContentRef,
  TransformComponent,
  TransformWrapper,
} from "react-zoom-pan-pinch"
import { useKeyPressEvent } from "react-use"
import { downloadToOutput, runPlugin } from "@/lib/api"
import { IconButton } from "@/components/ui/button"
import {
  askWritePermission,
  buildDownloadName,
  cn,
  copyCanvasImage,
  downloadImage,
  drawLines,
  generateMask,
  getErrorMessage,
  isMidClick,
  isRightClick,
  mimeFromSrc,
  mouseXY,
  strokeBounds,
  touchPointXY,
} from "@/lib/utils"
import { Eraser, Eye, Redo, Undo, Expand, Download } from "lucide-react"
import { useImage } from "@/hooks/useImage"
import { Slider } from "./ui/slider"
import { Line, PluginName, Point } from "@/lib/types"
import { useStore } from "@/lib/states"
import Cropper from "./Cropper"
import { InteractiveSegPoints } from "./InteractiveSeg"
import useHotKey from "@/hooks/useHotkey"
import Extender from "./Extender"
import {
  MAX_BRUSH_SIZE,
  MIN_BRUSH_SIZE,
  SHORTCUT_KEY_CHANGE_BRUSH_SIZE,
} from "@/lib/const"

const TOOLBAR_HEIGHT = 200
const COMPARE_SLIDER_DURATION_MS = 300
// 触控板捏合 / Ctrl+滚轮的恒定缩放速度：每单位 deltaY 缩放固定倍数
// （指数缩放，几何级数，任何缩放级别下手感一致）
const WHEEL_ZOOM_SPEED = 0.003
// Page Up / Page Down 键盘缩放的每次缩放倍数
const KEY_ZOOM_FACTOR = 1.25
const MAX_SCALE = 50
// 双指中判定“哪根手指在动”的移动阈值（px）
const TOUCH_MOVE_THRESHOLD = 10
// 判定“哪根手指按住了没动”的阈值（px），避免把捏合开始阶段误判成锚定绘制
const TOUCH_ANCHOR_THRESHOLD = 6
// 显示用掩膜画布的像素上限。笔迹是矢量数据，提交时按原图分辨率重新栅格化，
// 所以屏幕上的画布不需要跟原图一样大。如果按原分辨率逐帧
// clearRect + drawImage + stroke()，6000x4000 这种图每帧要处理 2400 万像素，
// 是画笔掉帧的主因（代价随图片分辨率线性增长）。按上限等比缩小后，
// 每帧的重绘量被限制住，画笔手感与图片分辨率无关。
const MASK_CANVAS_MAX_PIXELS = 4 * 1024 * 1024

interface EditorProps {
  file: File
}

// 原生 touch 监听器需要的最新状态/函数快照
type TouchLatestState = {
  context?: CanvasRenderingContext2D
  isProcessing: boolean
  isPanning: boolean
  isInpainting: boolean
  isDraging: boolean
  isOriginalLoaded: boolean
  brushSize: number
  runMannually: boolean
  originalSrc: string
  isInteractiveSeg: boolean
  clicks: number[][]
  startStroke: (pt: Point) => void
  pushStrokePoint: (pt: Point) => void
  commitCurrentStroke: () => void
  cancelCurrentStroke: () => void
  runInteractiveSeg: (clicks: number[][]) => void
  runInpainting: () => Promise<void>
  updateInteractiveSegState: (state: { clicks: number[][] }) => void
  setIsDraging: (value: boolean) => void
}

export default function Editor(props: EditorProps) {
  const { file } = props
  const { toast } = useToast()

  const [
    disableShortCuts,
    windowSize,
    isInpainting,
    imageWidth,
    imageHeight,
    settings,
    enableAutoSaving,
    setImageSize,
    setBaseBrushSize,
    getCurrentTargetFile,
    interactiveSegState,
    updateInteractiveSegState,
    commitStroke,
    undo,
    redo,
    undoDisabled,
    redoDisabled,
    isProcessing,
    updateAppState,
    runMannually,
    runInpainting,
    isCropperExtenderResizing,
    decreaseBaseBrushSize,
    increaseBaseBrushSize,
  ] = useStore((state) => [
    state.disableShortCuts,
    state.windowSize,
    state.isInpainting,
    state.imageWidth,
    state.imageHeight,
    state.settings,
    state.serverConfig.enableAutoSaving,
    state.setImageSize,
    state.setBaseBrushSize,
    state.getCurrentTargetFile,
    state.interactiveSegState,
    state.updateInteractiveSegState,
    state.commitStroke,
    state.undo,
    state.redo,
    state.undoDisabled(),
    state.redoDisabled(),
    state.getIsProcessing(),
    state.updateAppState,
    state.runMannually(),
    state.runInpainting,
    state.isCropperExtenderResizing,
    state.decreaseBaseBrushSize,
    state.increaseBaseBrushSize,
  ])
  const baseBrushSize = useStore((state) => state.editorState.baseBrushSize)
  const brushSize = useStore((state) => state.getBrushSize())
  const renders = useStore((state) => state.editorState.renders)
  const extraMasks = useStore((state) => state.editorState.extraMasks)
  const temporaryMasks = useStore((state) => state.editorState.temporaryMasks)
  const lineGroups = useStore((state) => state.editorState.lineGroups)
  const curLineGroup = useStore((state) => state.editorState.curLineGroup)

  // Local State
  const [showOriginal, setShowOriginal] = useState(false)
  const [original, isOriginalLoaded] = useImage(file)
  const [context, setContext] = useState<CanvasRenderingContext2D>()
  const [imageContext, setImageContext] = useState<CanvasRenderingContext2D>()
  const [showBrush, setShowBrush] = useState(false)
  const [showRefBrush, setShowRefBrush] = useState(false)
  const [isPanning, setIsPanning] = useState<boolean>(false)

  const containerRef = useRef<HTMLDivElement>(null)
  const strokeRef = useRef<Line | null>(null)
  // 当前笔画已经画到第几个点（后面这些点等下一帧再画）
  const strokeDrawnCountRef = useRef(0)
  // 待 flush 的笔画帧 id
  const strokeFrameRef = useRef(0)
  const brushCursorRef = useRef<HTMLDivElement>(null)
  const cursorPosRef = useRef<Point>({ x: -1, y: -1 })
  const cursorFrameRef = useRef<number>(0)
  const timeoutRefs = useRef<Set<ReturnType<typeof setTimeout>>>(new Set())
  const spacePressedRef = useRef(false)

  const trackedTimeout = useCallback((fn: () => void, ms: number) => {
    const id: ReturnType<typeof setTimeout> = setTimeout(() => {
      timeoutRefs.current.delete(id)
      fn()
    }, ms)
    timeoutRefs.current.add(id)
    return id
  }, [])
  // ---- 触屏状态机（原生监听器，绕开 React passive 限制）----
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  // 每根手指上次的屏幕坐标（identifier -> {x,y}）
  const touchPositionsRef = useRef<Map<number, { x: number; y: number }>>(
    new Map()
  )
  // 当前触摸意图：pending=多指刚落下待判断；single=单指绘制；
  // draw=一指定住一指定画（保留触控板/触屏的锚定绘制）；gesture=双指缩放/平移
  const touchModeRef = useRef<"pending" | "single" | "draw" | "gesture" | null>(
    null
  )
  // 正在驱动笔画的指头 identifier
  const drawingTouchIdRef = useRef<number | null>(null)
  // 供原生 touch 监听器读取的最新状态与函数
  const latestRef = useRef<TouchLatestState | null>(null)

  const [scale, setScale] = useState<number>(1)
  const [panned, setPanned] = useState<boolean>(false)
  const [minScale, setMinScale] = useState<number>(1.0)
  const windowCenterX = windowSize.width / 2
  const windowCenterY = windowSize.height / 2
  const viewportRef = useRef<ReactZoomPanPinchContentRef | null>(null)
  // Indicates that the image has been loaded and is centered on first load
  const [initialCentered, setInitialCentered] = useState(false)

  const [isDraging, setIsDraging] = useState(false)

  const [sliderPos, setSliderPos] = useState<number>(0)
  const [isChangingBrushSizeByWheel, setIsChangingBrushSizeByWheel] =
    useState<boolean>(false)

  const hadDrawSomething = useCallback(() => {
    return strokeRef.current !== null || curLineGroup.length !== 0
  }, [curLineGroup])

  useEffect(() => {
    if (
      !imageContext ||
      !isOriginalLoaded ||
      imageWidth === 0 ||
      imageHeight === 0
    ) {
      return
    }
    const render = renders.length === 0 ? original : renders[renders.length - 1]
    imageContext.canvas.width = imageWidth
    imageContext.canvas.height = imageHeight

    imageContext.clearRect(
      0,
      0,
      imageContext.canvas.width,
      imageContext.canvas.height
    )
    imageContext.drawImage(render, 0, 0, imageWidth, imageHeight)
  }, [
    renders,
    original,
    isOriginalLoaded,
    imageContext,
    imageHeight,
    imageWidth,
  ])

  // 显示用掩膜画布的分辨率：不超过 MASK_CANVAS_MAX_PIXELS，坐标仍用原图坐标，
  // 靠 ctx.setTransform 映射。maskScale === 1 时与原来完全一致。
  const maskScale = useMemo(() => {
    if (imageWidth === 0 || imageHeight === 0) {
      return 1
    }
    const pixels = imageWidth * imageHeight
    if (pixels <= MASK_CANVAS_MAX_PIXELS) {
      return 1
    }
    return Math.sqrt(MASK_CANVAS_MAX_PIXELS / pixels)
  }, [imageWidth, imageHeight])
  const maskCanvasWidth = Math.max(1, Math.round(imageWidth * maskScale))
  const maskCanvasHeight = Math.max(1, Math.round(imageHeight * maskScale))
  // 给原生 touch 监听器用的最新快照
  const maskScaleRef = useRef(maskScale)
  maskScaleRef.current = maskScale

  // 给原图分辨率的坐标 -> 画布位图像素 的换算，供绘制路径复用
  const applyMaskTransform = useCallback(
    (ctx: CanvasRenderingContext2D, scale: number) => {
      ctx.setTransform(scale, 0, 0, scale, 0, 0)
    },
    []
  )

  // 已提交内容（已提交的笔迹/掩膜/分割临时掩膜）画在这个“底层”画布上，
  // 显示画布 = 底层画布 + 当前正在画的笔画。
  // 底层画布只在内容真正变化时重建；单纯提交新笔画时只把新增的那一笔补上去。
  const maskBaseCanvasRef = useRef<HTMLCanvasElement | null>(null)
  // 底层画布已经画到 curLineGroup 的第几条（用于增量追加）
  const baseDrawnLinesRef = useRef<Line[]>([])
  // 上一次 redrawMaskCanvas 的“内容类”入参，用来判断能不能只做增量追加
  const baseContentRef = useRef<{
    temporaryMasks: HTMLImageElement[]
    extraMasks: HTMLImageElement[]
    segMask: HTMLImageElement | null
    segClicks: number[][]
    imageWidth: number
    imageHeight: number
    context: CanvasRenderingContext2D | undefined
  } | null>(null)

  const redrawMaskCanvas = useCallback(() => {
    if (
      !context ||
      !isOriginalLoaded ||
      imageWidth === 0 ||
      imageHeight === 0
    ) {
      return
    }
    let base = maskBaseCanvasRef.current
    if (!base) {
      base = document.createElement("canvas")
      maskBaseCanvasRef.current = base
    }
    // 画布位图跟着显示分辨率走，尺寸变化时同步（改 width/height 会清空位图并
    // 重置变换，所以之后必须重新 setTransform）
    const bitmapChanged =
      base.width !== maskCanvasWidth || base.height !== maskCanvasHeight
    if (bitmapChanged) {
      base.width = maskCanvasWidth
      base.height = maskCanvasHeight
      baseDrawnLinesRef.current = []
    }
    const baseCtx = base.getContext("2d")
    if (!baseCtx) {
      return
    }

    const segMask = interactiveSegState.tmpInteractiveSegMask ?? null
    const prev = baseContentRef.current
    // 内容没变、且 curLineGroup 是在已画部分后面追加的 => 只需补画新增的笔画
    const appendable =
      !bitmapChanged &&
      prev !== null &&
      prev.temporaryMasks === temporaryMasks &&
      prev.extraMasks === extraMasks &&
      prev.segMask === segMask &&
      prev.segClicks === interactiveSegState.clicks &&
      prev.imageWidth === imageWidth &&
      prev.imageHeight === imageHeight &&
      prev.context === context &&
      curLineGroup.length >= baseDrawnLinesRef.current.length &&
      baseDrawnLinesRef.current.every(
        (line, i) => curLineGroup[i] === line
      )

    if (appendable) {
      applyMaskTransform(baseCtx, maskScale)
      for (
        let i = baseDrawnLinesRef.current.length;
        i < curLineGroup.length;
        i++
      ) {
        drawLines(baseCtx, [curLineGroup[i]])
      }
      baseDrawnLinesRef.current = curLineGroup.slice()
    } else {
      baseCtx.setTransform(1, 0, 0, 1, 0, 0)
      baseCtx.clearRect(0, 0, base.width, base.height)
      applyMaskTransform(baseCtx, maskScale)
      temporaryMasks.forEach((maskImage) => {
        baseCtx.drawImage(maskImage, 0, 0, imageWidth, imageHeight)
      })
      extraMasks.forEach((maskImage) => {
        baseCtx.drawImage(maskImage, 0, 0, imageWidth, imageHeight)
      })
      if (interactiveSegState.isInteractiveSeg && segMask) {
        baseCtx.drawImage(segMask, 0, 0, imageWidth, imageHeight)
      }
      // 只画当前提交的笔画（curLineGroup）。inpaint 完成后由 runInpainting
      // 清空 curLineGroup，掩膜随之从画布清除（标准行为）。
      drawLines(baseCtx, curLineGroup)
      baseDrawnLinesRef.current = curLineGroup.slice()
    }
    baseContentRef.current = {
      temporaryMasks,
      extraMasks,
      segMask,
      segClicks: interactiveSegState.clicks,
      imageWidth,
      imageHeight,
      context,
    }

    const canvas = context.canvas
    if (canvas.width !== maskCanvasWidth) {
      canvas.width = maskCanvasWidth
    }
    if (canvas.height !== maskCanvasHeight) {
      canvas.height = maskCanvasHeight
    }
    context.setTransform(1, 0, 0, 1, 0, 0)
    context.clearRect(0, 0, maskCanvasWidth, maskCanvasHeight)
    // base 和显示画布位图尺寸相同，这里是 1:1 拷贝，必须在 identity 下做；
    // 缩放变换要等 drawImage 之后再上，否则整块内容会被二次缩小
    context.drawImage(base, 0, 0, maskCanvasWidth, maskCanvasHeight)
    applyMaskTransform(context, maskScale)
    // 把进行中的笔画补上：上面整块重画把它擦掉了，避免绘制过程中出现空白
    if (strokeRef.current) {
      drawLines(context, [strokeRef.current])
      strokeDrawnCountRef.current = strokeRef.current.pts.length
    }
  }, [
    temporaryMasks,
    extraMasks,
    isOriginalLoaded,
    interactiveSegState,
    context,
    curLineGroup,
    imageHeight,
    imageWidth,
    maskCanvasWidth,
    maskCanvasHeight,
    maskScale,
    applyMaskTransform,
  ])

  useEffect(() => {
    redrawMaskCanvas()
  }, [redrawMaskCanvas])

  const hadRunInpainting = () => {
    return renders.length !== 0
  }

  const getCurrentWidthHeight = useCallback(() => {
    let width = 512
    let height = 512
    if (!isOriginalLoaded) {
      return [width, height]
    }
    if (renders.length === 0) {
      width = original.naturalWidth
      height = original.naturalHeight
    } else if (renders.length !== 0) {
      width = renders[renders.length - 1].width
      height = renders[renders.length - 1].height
    }

    return [width, height]
  }, [original, isOriginalLoaded, renders])

  // Draw once the original image is loaded
  useEffect(() => {
    if (!isOriginalLoaded) {
      return
    }

    const [width, height] = getCurrentWidthHeight()
    if (width !== imageWidth || height !== imageHeight) {
      setImageSize(width, height)
    }

    const rW = windowSize.width / width
    const rH = (windowSize.height - TOOLBAR_HEIGHT) / height

    let s = 1.0
    if (rW < 1 || rH < 1) {
      s = Math.min(rW, rH)
    }
    setMinScale(s)
    setScale(s)

    if (context?.canvas) {
      // 位图尺寸用显示分辨率；redrawMaskCanvas 会随后重置变换并重建内容
      if (maskCanvasWidth != context.canvas.width) {
        context.canvas.width = maskCanvasWidth
      }
      if (maskCanvasHeight != context.canvas.height) {
        context.canvas.height = maskCanvasHeight
      }
    }

    if (!initialCentered) {
      // 防止每次擦除以后图片 zoom 还原
      viewportRef.current?.centerView(s, 1)
      setInitialCentered(true)
    }
  }, [
    viewportRef,
    imageHeight,
    imageWidth,
    original,
    isOriginalLoaded,
    windowSize,
    initialCentered,
    getCurrentWidthHeight,
    context?.canvas,
    setImageSize,
    maskCanvasWidth,
    maskCanvasHeight,
  ])

  useEffect(() => {
    // render 改变尺寸以后，undo/redo 重新 center
    viewportRef?.current?.centerView(minScale, 1)
  }, [imageHeight, imageWidth, viewportRef, minScale])

  // Zoom reset
  const resetZoom = useCallback(() => {
    if (!minScale || !windowSize) {
      return
    }
    const viewport = viewportRef.current
    if (!viewport) {
      return
    }
    const offsetX = (windowSize.width - imageWidth * minScale) / 2
    const offsetY = (windowSize.height - imageHeight * minScale) / 2
    viewport.setTransform(offsetX, offsetY, minScale, 200, "easeOutQuad")
    if (viewport.instance.transformState.scale) {
      viewport.instance.transformState.scale = minScale
    }

    setScale(minScale)
    setPanned(false)
  }, [
    viewportRef,
    windowSize,
    imageHeight,
    imageWidth,
    minScale,
  ])

  useEffect(() => {
    const onWindowResize = () => {
      resetZoom()
    }
    window.addEventListener("resize", onWindowResize)
    return () => {
      window.removeEventListener("resize", onWindowResize)
    }
  }, [resetZoom])

  const handleEscPressed = () => {
    if (isProcessing) {
      return
    }

    if (isDraging) {
      setIsDraging(false)
    } else {
      resetZoom()
    }
  }

  useHotKey("Escape", handleEscPressed, [
    isDraging,
    isInpainting,
    resetZoom,
    // drawOnCurrentRender,
  ])

  const updateCursorPos = (x: number, y: number) => {
    cursorPosRef.current = { x, y }
    if (cursorFrameRef.current !== 0) {
      return
    }
    cursorFrameRef.current = requestAnimationFrame(() => {
      cursorFrameRef.current = 0
      const { x: posX, y: posY } = cursorPosRef.current
      const transform = `translate(${posX}px, ${posY}px) translate(-50%, -50%)`
      if (brushCursorRef.current) {
        brushCursorRef.current.style.transform = transform
      }
    })
  }

  useEffect(() => {
    // 同一个 Set 只会增删、不会重新赋值，先取出引用供 cleanup 使用
    const pendingTimeouts = timeoutRefs.current
    return () => {
      if (cursorFrameRef.current !== 0) {
        cancelAnimationFrame(cursorFrameRef.current)
      }
      if (strokeFrameRef.current !== 0) {
        cancelAnimationFrame(strokeFrameRef.current)
      }
      for (const t of pendingTimeouts) {
        clearTimeout(t)
      }
      pendingTimeouts.clear()
    }
  }, [])

  const onMouseMove = (ev: SyntheticEvent) => {
    const mouseEvent = ev.nativeEvent as MouseEvent
    updateCursorPos(mouseEvent.pageX, mouseEvent.pageY)
  }

  const startStroke = (pt: Point) => {
    if (!context?.canvas) {
      return
    }
    strokeDrawnCountRef.current = 0
    strokeRef.current = { size: brushSize, pts: [pt] }
    // 也要 flush 一次，这样“点一下不拖”也能留下一个圆点
    scheduleStrokeFlush()
  }

  // 把当前笔画里“还没画”的新增点画到显示画布上。
  // 只重画新增线段的包围盒（脏矩形）：清掉上一帧画在这里的像素 -> 从底层画布
  // 拷回已提交内容 -> 在同一块区域里重新描一次整条当前笔画。
  // 整条路径在一次 stroke() 内并集覆盖，笔刷的半透明不会自我叠加成实心，
  // 像素结果与“每次全画布重画”完全一致，但每帧的代价只跟笔尖扫过的一小块
  // 区域有关，不再随图片分辨率增长。
  const flushStroke = useCallback(() => {
    strokeFrameRef.current = 0
    const stroke = strokeRef.current
    const ctx = context
    if (!stroke || !ctx) {
      return
    }
    const from = strokeDrawnCountRef.current
    if (from >= stroke.pts.length) {
      return
    }
    const base = maskBaseCanvasRef.current
    if (!base || !ctx.canvas.width || !ctx.canvas.height) {
      return
    }
    const scale = maskScaleRef.current
    // 脏矩形要带上“上一个已画点”：本次新增的第一个线段是 上一点 -> 新点，
    // 只取新点的话，快速甩笔时这段会伸到脏矩形外面，那部分像素就永远补不上
    const added: Line = {
      size: stroke.size,
      pts: stroke.pts.slice(Math.max(0, from - 1)),
    }
    // 圆头笔刷会画到路径外 lineWidth/2（注意 strokeBounds 用的是原图坐标，
    // 这里也要用原图单位；多留 2 个画布像素给抗锯齿和 floor/ceil 取整）
    const pad = (stroke.size ?? brushSize) / 2 + 2 / scale
    const dirty = strokeBounds(added, pad)
    if (!dirty) {
      return
    }
    // 脏矩形换算到画布位图像素，并夹到画布范围内
    const x0 = Math.max(0, Math.floor(dirty.x * scale))
    const y0 = Math.max(0, Math.floor(dirty.y * scale))
    const x1 = Math.min(
      ctx.canvas.width,
      Math.ceil((dirty.x + dirty.w) * scale)
    )
    const y1 = Math.min(
      ctx.canvas.height,
      Math.ceil((dirty.y + dirty.h) * scale)
    )
    const bw = x1 - x0
    const bh = y1 - y0
    if (bw <= 0 || bh <= 0) {
      return
    }
    // clearRect / drawImage / clip 必须用同一个矩形，否则两者之间那圈像素
    // 被清掉又没补描，笔迹上会出现 1px 的缺口
    ctx.save()
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(x0, y0, bw, bh)
    ctx.drawImage(base, x0, y0, bw, bh, x0, y0, bw, bh)
    ctx.beginPath()
    ctx.rect(x0, y0, bw, bh)
    ctx.clip()
    applyMaskTransform(ctx, scale)
    drawLines(ctx, [stroke])
    ctx.restore()
    strokeDrawnCountRef.current = stroke.pts.length
  }, [context, brushSize, applyMaskTransform])

  const scheduleStrokeFlush = useCallback(() => {
    if (strokeFrameRef.current !== 0) {
      return
    }
    // 用 rAF 合帧：一次绘制期间可能有上百个 pointermove，
    // 但每个显示帧只需要把最新的一段画上去
    strokeFrameRef.current = requestAnimationFrame(flushStroke)
  }, [flushStroke])

  const cancelStrokeFlush = useCallback(() => {
    if (strokeFrameRef.current !== 0) {
      cancelAnimationFrame(strokeFrameRef.current)
      strokeFrameRef.current = 0
    }
  }, [])

  // 追加一个绘制点：只记录坐标，实际绘制由 rAF 合帧后统一做（见 flushStroke）
  const pushStrokePoint = useCallback(
    (pt: Point) => {
      const stroke = strokeRef.current
      if (!stroke) {
        return
      }
      const last = stroke.pts[stroke.pts.length - 1]
      if (last && last.x === pt.x && last.y === pt.y) {
        return
      }
      stroke.pts.push(pt)
      scheduleStrokeFlush()
    },
    [scheduleStrokeFlush]
  )

  const commitCurrentStroke = () => {
    const stroke = strokeRef.current
    if (!stroke) {
      return
    }
    // 先把这一帧剩下的点画完再提交，避免最后一小段留白
    cancelStrokeFlush()
    flushStroke()
    strokeRef.current = null
    strokeDrawnCountRef.current = 0
    commitStroke(stroke)
  }

  // 取消进行中的笔画：丢弃 strokeRef 并重绘掩膜画布，移除已画上去的临时笔迹
  const cancelCurrentStroke = useCallback(() => {
    if (!strokeRef.current) {
      return
    }
    cancelStrokeFlush()
    strokeRef.current = null
    strokeDrawnCountRef.current = 0
    setIsDraging(false)
    redrawMaskCanvas()
  }, [redrawMaskCanvas, cancelStrokeFlush])

  const onMouseDrag = (ev: SyntheticEvent) => {
    if (isProcessing) {
      return
    }

    if (interactiveSegState.isInteractiveSeg) {
      return
    }
    if (isPanning) {
      return
    }
    if (!isDraging) {
      return
    }
    if (!strokeRef.current) {
      return
    }
    // getCoalescedEvents 带上浏览器合并掉的中间点，快速划动时笔迹不会断
    const nativeEvent = ev.nativeEvent as MouseEvent
    const withCoalesced = nativeEvent as MouseEvent & {
      getCoalescedEvents?: () => MouseEvent[]
    }
    if (typeof withCoalesced.getCoalescedEvents === "function") {
      const events = withCoalesced.getCoalescedEvents()
      if (events.length > 1) {
        for (const coalesced of events) {
          const xy = mouseXY(coalesced)
          pushStrokePoint(xy)
        }
        return
      }
    }
    pushStrokePoint(mouseXY(nativeEvent))
  }

  const runInteractiveSeg = async (newClicks: number[][]) => {
    updateAppState({ isPluginRunning: true })
    const targetFile = await getCurrentTargetFile()
    try {
      const res = await runPlugin(
        true,
        PluginName.InteractiveSeg,
        targetFile,
        undefined,
        newClicks
      )
      const { blob } = res
      const blobUrl = URL.createObjectURL(blob)
      const img = new Image()
      img.onload = () => {
        URL.revokeObjectURL(blobUrl)
        if (useStore.getState().file !== file) {
          return
        }
        updateInteractiveSegState({ tmpInteractiveSegMask: img })
      }
      img.src = blobUrl
    } catch (e) {
      toast({
        variant: "destructive",
        description: getErrorMessage(e),
      })
    }
    updateAppState({ isPluginRunning: false })
  }

  const onPointerUp = (ev: SyntheticEvent) => {
    if (isMidClick(ev)) {
      setIsPanning(false)
      return
    }
    if (!hadDrawSomething()) {
      return
    }
    if (interactiveSegState.isInteractiveSeg) {
      return
    }
    if (isPanning) {
      return
    }
    if (!original.src) {
      return
    }
    const canvas = context?.canvas
    if (!canvas) {
      return
    }
    if (isInpainting) {
      return
    }
    if (!isDraging) {
      return
    }

    commitCurrentStroke()

    if (runMannually) {
      setIsDraging(false)
    } else {
      runInpainting()
    }
  }

  const onCanvasMouseUp = (ev: SyntheticEvent) => {
    if (interactiveSegState.isInteractiveSeg) {
      const xy = mouseXY(ev)
      const newClicks: number[][] = [...interactiveSegState.clicks]
      if (isRightClick(ev)) {
        newClicks.push([xy.x, xy.y, 0, newClicks.length])
      } else {
        newClicks.push([xy.x, xy.y, 1, newClicks.length])
      }
      runInteractiveSeg(newClicks)
      updateInteractiveSegState({ clicks: newClicks })
    }
  }

  const onMouseDown = (ev: SyntheticEvent) => {
    if (isProcessing) {
      return
    }
    if (interactiveSegState.isInteractiveSeg) {
      return
    }
    if (isPanning) {
      return
    }
    if (!isOriginalLoaded) {
      return
    }
    const canvas = context?.canvas
    if (!canvas) {
      return
    }

    if (isRightClick(ev)) {
      return
    }

    if (isMidClick(ev)) {
      setIsPanning(true)
      return
    }

    setIsDraging(true)
    startStroke(mouseXY(ev))
  }

  // ---- 触屏手势（原生监听器，见下方 useEffect）----
  // 状态机：
  //  - single：单指绘制（或平移模式下交给库平移）
  //  - pending：多指刚落下，等待第一次 touchmove 判断意图
  //  - draw：一指定住、另一指拖动 → 用“在动的那根手指”继续绘制（保留触控板/触屏锚定绘制）
  //  - gesture：两根手指都在动 → 交给 react-zoom-pan-pinch 做缩放/平移
  // 普通滚轮（触控板双指滑动 / 鼠标滚轮）平移画布；Ctrl+滚轮（触控板捏合）由库缩放
  const panBy = useCallback((deltaX: number, deltaY: number) => {
    const viewport = viewportRef.current
    if (!viewport) {
      return
    }
    const { positionX, positionY, scale } = viewport.instance.transformState
    const x = positionX - deltaX
    const y = positionY - deltaY
    if (x === positionX && y === positionY) {
      return
    }
    viewport.instance.setTransformState(scale, x, y)
    setPanned(true)
  }, [])

  // 恒定速度缩放：在指定屏幕坐标锚点（通常为光标/视口中心）缩放固定倍数。
  // 缩放量只与倍数有关，不随缩放级别变化。
  const zoomAt = useCallback(
    (anchorX: number, anchorY: number, factor: number) => {
      const viewport = viewportRef.current
      if (!viewport) {
        return
      }
      const { instance } = viewport
      const content = instance.contentComponent
      if (!content) {
        return
      }
      const { scale, positionX, positionY } = instance.transformState
      const newScale = Math.min(
        Math.max(scale * factor, minScale * 0.3),
        MAX_SCALE
      )
      if (newScale === scale) {
        return
      }
      const rect = content.getBoundingClientRect()
      const anchorXInContent = (anchorX - rect.left) / scale
      const anchorYInContent = (anchorY - rect.top) / scale
      const scaleDiff = newScale - scale
      instance.setTransformState(
        newScale,
        positionX - anchorXInContent * scaleDiff,
        positionY - anchorYInContent * scaleDiff
      )
      setScale(newScale)
      setPanned(true)
    },
    [minScale]
  )

  // 取滚轮主轴位移：覆盖浏览器在部分平台把纵向滚动转成横向（如 Shift+滚轮）
  // 的行为。浏览器/操作系统已按用户的滚动方向设置翻转 delta 符号：
  // deltaY<0 恒等于“用户配置下的向上滚动”手势，因此直接以 delta 为准即可
  // 自动跟随自然滚动 / 经典滚动设置，无需额外检测。
  const getWheelDelta = useCallback((event: WheelEvent) => {
    return Math.abs(event.deltaX) > Math.abs(event.deltaY)
      ? event.deltaX
      : event.deltaY
  }, [])

  // 恒定速度缩放：缩放量只与 deltaY（捏合/滚轮位移）成正比，
  // 不随缩放级别变化，绕光标所在位置缩放。
  const zoomByWheel = useCallback(
    (event: WheelEvent) => {
      const factor = Math.exp(-getWheelDelta(event) * WHEEL_ZOOM_SPEED)
      zoomAt(event.clientX, event.clientY, factor)
    },
    [getWheelDelta, zoomAt]
  )

  // Page Up / Page Down 键盘缩放：绕视口中心缩放
  const zoomByKeys = useCallback(
    (zoomIn: boolean) => {
      zoomAt(
        windowCenterX,
        windowCenterY,
        zoomIn ? KEY_ZOOM_FACTOR : 1 / KEY_ZOOM_FACTOR
      )
    },
    [zoomAt, windowCenterX, windowCenterY]
  )

  useHotKey(
    "PageUp",
    (keyboardEvent) => {
      keyboardEvent.preventDefault()
      zoomByKeys(true)
    },
    [zoomByKeys]
  )

  useHotKey(
    "PageDown",
    (keyboardEvent) => {
      keyboardEvent.preventDefault()
      zoomByKeys(false)
    },
    [zoomByKeys]
  )

  const handleUndo = (keyboardEvent: KeyboardEvent | SyntheticEvent) => {
    keyboardEvent.preventDefault()
    undo()
  }
  useHotKey("meta+z,ctrl+z", handleUndo)

  const handleRedo = (keyboardEvent: KeyboardEvent | SyntheticEvent) => {
    keyboardEvent.preventDefault()
    redo()
  }
  useHotKey("shift+ctrl+z,shift+meta+z", handleRedo)

  useKeyPressEvent(
    "Tab",
    (ev) => {
      ev?.preventDefault()
      ev?.stopPropagation()
      if (hadRunInpainting()) {
        setShowOriginal(() => {
          trackedTimeout(() => {
            setSliderPos(100)
          }, 10)
          return true
        })
      }
    },
    (ev) => {
      ev?.preventDefault()
      ev?.stopPropagation()
      if (hadRunInpainting()) {
        trackedTimeout(() => {
          setSliderPos(0)
        }, 10)
        trackedTimeout(() => {
          setShowOriginal(false)
        }, COMPARE_SLIDER_DURATION_MS)
      }
    }
  )

  const download = useCallback(async () => {
    if (file === undefined) {
      return
    }
    if (enableAutoSaving && renders.length > 0) {
      try {
        const lastRender = renders[renders.length - 1]
        // 保存名的扩展名以渲染结果（服务端响应）的真实类型为准；
        // file.type/file.name 只是浏览器按扩展名的猜测，可能与字节不符（CONS-3）
        const renderMime = (await mimeFromSrc(lastRender.currentSrc)) ?? file.type
        await downloadToOutput(
          lastRender,
          buildDownloadName(file.name, "", renderMime),
          renderMime
        )
        toast({
          description: "Save image success",
        })
      } catch (e) {
        toast({
          variant: "destructive",
          title: "Uh oh! Something went wrong.",
          description: getErrorMessage(e),
        })
      }
      return
    }

    // TODO: download to output directory
    const curRender = renders[renders.length - 1]
    const renderMime = curRender
      ? await mimeFromSrc(curRender.currentSrc)
      : undefined
    const name = buildDownloadName(file.name, "_cleanup", renderMime)
    downloadImage(curRender.currentSrc, name)
    if (settings.enableDownloadMask) {
      // mask 是 0/255 二值图，JPEG 的有损压缩会污染边缘 → 恒定 PNG（CONS-3）
      const maskFileName = buildDownloadName(file.name, "_mask", "image/png")

      const maskCanvas = generateMask(imageWidth, imageHeight, lineGroups)
      // Create a link
      const aDownloadLink = document.createElement("a")
      // Add the name of the file to the link
      aDownloadLink.download = maskFileName
      // Attach the data to the link
      aDownloadLink.href = maskCanvas.toDataURL("image/png")
      aDownloadLink.style.display = "none"
      // 需先挂载再 click，否则部分浏览器不会触发下载
      document.body.appendChild(aDownloadLink)
      aDownloadLink.click()
      setTimeout(() => aDownloadLink.remove(), 1000)
    }
  }, [
    file,
    enableAutoSaving,
    renders,
    settings,
    imageHeight,
    imageWidth,
    lineGroups,
    toast,
  ])

  useHotKey("meta+s,ctrl+s", download)

  const toggleShowBrush = (newState: boolean) => {
    if (newState !== showBrush && !isPanning && !isCropperExtenderResizing) {
      setShowBrush(newState)
    }
  }

  const getCursor = useCallback(() => {
    if (isProcessing) {
      return "default"
    }
    if (isPanning) {
      return "grab"
    }
    if (showBrush && interactiveSegState.isInteractiveSeg) {
      return "crosshair"
    }
    if (showBrush) {
      return "none"
    }
    return undefined
  }, [showBrush, isPanning, isProcessing, interactiveSegState.isInteractiveSeg])

  useHotKey(
    "BracketLeft",
    () => {
      decreaseBaseBrushSize()
    },
    [decreaseBaseBrushSize]
  )

  useHotKey(
    "BracketRight",
    () => {
      increaseBaseBrushSize()
    },
    [increaseBaseBrushSize]
  )

  // Manual Inpainting Hotkey
  useHotKey(
    "shift+r",
    () => {
      if (runMannually && hadDrawSomething()) {
        runInpainting()
      }
    },
    [runMannually, runInpainting, hadDrawSomething]
  )

  useHotKey(
    "ctrl+c,meta+c",
    async () => {
      const hasPermission = await askWritePermission()
      if (hasPermission && renders.length > 0) {
        // 显示画布是按显示分辨率缩小的，这里按原图分辨率重新生成，避免拷到低清掩膜
        if (imageWidth > 0 && imageHeight > 0) {
          const fullRes = generateMask(imageWidth, imageHeight, [
            curLineGroup,
          ])
          await copyCanvasImage(fullRes)
          toast({
            title: "Copy inpainting result to clipboard",
          })
        }
      }
    },
    [renders, imageWidth, imageHeight, curLineGroup]
  )

  // Toggle clean/zoom tool on spacebar.
  useKeyPressEvent(
    " ",
    (ev) => {
      if (!disableShortCuts) {
        ev?.preventDefault()
        ev?.stopPropagation()
        spacePressedRef.current = true
        setShowBrush(false)
        setIsPanning(true)
      }
    },
    (ev) => {
      if (!disableShortCuts) {
        ev?.preventDefault()
        ev?.stopPropagation()
        spacePressedRef.current = false
        setShowBrush(true)
        setIsPanning(false)
      }
    }
  )

  useEffect(() => {
    const handleKeyUp = (ev: KeyboardEvent) => {
      if (ev.key === SHORTCUT_KEY_CHANGE_BRUSH_SIZE) {
        setIsChangingBrushSizeByWheel(false)
      }
      if (ev.key === " ") {
        spacePressedRef.current = false
      }
    }

    const handleBlur = () => {
      setIsChangingBrushSizeByWheel(false)
      spacePressedRef.current = false
    }

    window.addEventListener("keyup", handleKeyUp)
    window.addEventListener("blur", handleBlur)

    return () => {
      window.removeEventListener("keyup", handleKeyUp)
      window.removeEventListener("blur", handleBlur)
    }
  }, [])

  useKeyPressEvent(
    SHORTCUT_KEY_CHANGE_BRUSH_SIZE,
    (ev) => {
      if (!disableShortCuts) {
        ev?.preventDefault()
        ev?.stopPropagation()
        setIsChangingBrushSizeByWheel(true)
      }
    },
    (ev) => {
      if (!disableShortCuts) {
        ev?.preventDefault()
        ev?.stopPropagation()
        setIsChangingBrushSizeByWheel(false)
      }
    }
  )

  const getCurScale = (): number => {
    let s = minScale
    if (viewportRef.current?.instance?.transformState.scale !== undefined) {
      s = viewportRef.current?.instance?.transformState.scale
    }
    return s!
  }

  const getBrushStyle = (_x: number, _y: number) => {
    const curScale = getCurScale()
    return {
      width: `${brushSize * curScale}px`,
      height: `${brushSize * curScale}px`,
      left: `${_x}px`,
      top: `${_y}px`,
      transform: "translate(-50%, -50%)",
    }
  }

  const renderBrush = (style: CSSProperties) => {
    return (
      <div
        className="absolute rounded-[50%] border-[1px] border-[solid] border-[#ffcc00] pointer-events-none bg-[#ffcc00bb]"
        style={style}
      />
    )
  }

  const handleSliderChange = (value: number) => {
    setBaseBrushSize(value)

    if (!showRefBrush) {
      setShowRefBrush(true)
      trackedTimeout(() => {
        setShowRefBrush(false)
      }, 10000)
    }
  }

  const renderCanvas = () => {
    return (
      <TransformWrapper
        ref={(r) => {
          if (r) {
            viewportRef.current = r
          }
        }}
        panning={{ disabled: !isPanning, velocityDisabled: true }}
        // wheel 处理全部交给外部原生监听器（普通滚轮平移、Ctrl+滚轮/捏合恒定速度缩放），
        // 禁用库自带的 wheel 缩放，避免它 stopPropagation 抢占事件
        wheel={{ disabled: true }}
        centerZoomedOut
        alignmentAnimation={{ disabled: true }}
        centerOnInit
        limitToBounds={false}
        doubleClick={{ disabled: true }}
        initialScale={minScale}
        minScale={minScale * 0.3}
        maxScale={50}
        onPanning={() => {
          if (!panned) {
            setPanned(true)
          }
        }}
        onPinching={() => {
          if (!panned) {
            setPanned(true)
          }
        }}
        onZoom={(ref) => {
          setScale(ref.state.scale)
        }}
      >
        <TransformComponent
          contentStyle={{
            visibility: initialCentered ? "visible" : "hidden",
          }}
        >
          <div className="grid [grid-template-areas:'editor-content'] gap-y-4">
            <canvas
              className="[grid-area:editor-content]"
              style={{
                clipPath: `inset(0 ${sliderPos}% 0 0)`,
                transition: `clip-path ${COMPARE_SLIDER_DURATION_MS}ms`,
              }}
              ref={(r) => {
                if (r && !imageContext) {
                  const ctx = r.getContext("2d")
                  if (ctx) {
                    setImageContext(ctx)
                  }
                }
              }}
            />
            <canvas
              className={cn(
                "[grid-area:editor-content]",
                isProcessing
                  ? "pointer-events-none animate-pulse duration-600"
                  : ""
              )}
              style={{
                cursor: getCursor(),
                clipPath: `inset(0 ${sliderPos}% 0 0)`,
                transition: `clip-path ${COMPARE_SLIDER_DURATION_MS}ms`,
                // 位图可能比原图小（见 MASK_CANVAS_MAX_PIXELS），
                // 用 CSS 尺寸把布局盒子撑回原图大小，坐标系和缩放行为保持不变
                width: `${imageWidth}px`,
                height: `${imageHeight}px`,
              }}
              onContextMenu={(e) => {
                e.preventDefault()
              }}
              onMouseOver={() => {
                toggleShowBrush(true)
                setShowRefBrush(false)
              }}
              onFocus={() => toggleShowBrush(true)}
              onMouseLeave={() => toggleShowBrush(false)}
              onMouseDown={onMouseDown}
              onMouseUp={onCanvasMouseUp}
              onMouseMove={onMouseDrag}
              ref={(r) => {
                canvasRef.current = r
                if (r && !context) {
                  // 不要加 desynchronized：Firefox 并没有实现低延迟画布路径，
                  // 只会让这个画布走上一条对大尺寸画布很不利的合成路径
                  const ctx = r.getContext("2d")
                  if (ctx) {
                    setContext(ctx)
                  }
                }
              }}
            />
            <div
              className="[grid-area:editor-content] pointer-events-none grid [grid-template-areas:'original-image-content']"
              style={{
                width: `${imageWidth}px`,
                height: `${imageHeight}px`,
              }}
            >
              {showOriginal && (
                <>
                  <div
                    className="[grid-area:original-image-content] z-10 bg-primary h-full w-[6px] justify-self-end"
                    style={{
                      marginRight: `${sliderPos}%`,
                      transition: `margin-right ${COMPARE_SLIDER_DURATION_MS}ms`,
                    }}
                  />
                  <img
                    className="[grid-area:original-image-content]"
                    src={original.src}
                    alt="original"
                    style={{
                      width: `${imageWidth}px`,
                      height: `${imageHeight}px`,
                    }}
                  />
                </>
              )}
            </div>
          </div>

          <Cropper
            maxHeight={imageHeight}
            maxWidth={imageWidth}
            minHeight={Math.min(512, imageHeight)}
            minWidth={Math.min(512, imageWidth)}
            scale={getCurScale()}
            show={settings.showCropper}
          />

          <Extender
            minHeight={Math.min(512, imageHeight)}
            minWidth={Math.min(512, imageWidth)}
            scale={getCurScale()}
            show={settings.showExtender}
          />

          {interactiveSegState.isInteractiveSeg ? (
            <InteractiveSegPoints />
          ) : (
            <></>
          )}
        </TransformComponent>
      </TransformWrapper>
    )
  }

  // 原生 touch 监听器通过 latestRef 读取最新状态/函数
  latestRef.current = {
    context,
    isProcessing,
    isPanning,
    isInpainting,
    isDraging,
    isOriginalLoaded,
    brushSize,
    runMannually,
    originalSrc: original?.src ?? "",
    isInteractiveSeg: interactiveSegState.isInteractiveSeg,
    clicks: interactiveSegState.clicks,
    startStroke,
    pushStrokePoint,
    commitCurrentStroke,
    cancelCurrentStroke,
    runInteractiveSeg,
    runInpainting,
    updateInteractiveSegState,
    setIsDraging,
  }

  // 触屏手势（原生非 passive 监听器，绕开 React passive 限制）：
  // 支持“一指定住、另一指拖动绘制”（触控板面积不够 / 触屏锚定），
  // 同时保留双指捏合缩放与双指平移。
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const onTouchStart = (event: TouchEvent) => {
      event.preventDefault()
      const latest = latestRef.current
      if (!latest || latest.isProcessing) {
        return
      }
      const { touches } = event
      if (touches.length > 1) {
        // 多指落下：先取消可能已开始的单指笔画（避免误画一个点），
        // 记录所有手指位置，等第一次 touchmove 判断意图（锚定绘制会在此后重新起笔）
        latest.cancelCurrentStroke()
        touchModeRef.current = "pending"
        touchPositionsRef.current.clear()
        for (const t of touches) {
          touchPositionsRef.current.set(t.identifier, {
            x: t.clientX,
            y: t.clientY,
          })
        }
        return
      }
      // 单指
      const t = touches[0]
      touchModeRef.current = "single"
      touchPositionsRef.current.clear()
      touchPositionsRef.current.set(t.identifier, {
        x: t.clientX,
        y: t.clientY,
      })
      drawingTouchIdRef.current = t.identifier
      if (latest.isInteractiveSeg) {
        return
      }
      if (latest.isPanning) {
        return
      }
      if (!latest.isOriginalLoaded) {
        return
      }
      if (!latest.context?.canvas) {
        return
      }
      latest.setIsDraging(true)
      latest.startStroke(touchPointXY(t, canvas))
    }

    const onTouchMove = (event: TouchEvent) => {
      event.preventDefault()
      const latest = latestRef.current
      if (!latest || latest.isProcessing) {
        return
      }
      const { touches } = event
      const mode = touchModeRef.current
      if (mode === "single") {
        const t = touches[0]
        touchPositionsRef.current.set(t.identifier, {
          x: t.clientX,
          y: t.clientY,
        })
        if (latest.isPanning || latest.isInteractiveSeg) {
          return
        }
        if (!latest.isDraging) {
          return
        }
        latest.pushStrokePoint(touchPointXY(t, canvas))
        return
      }
      if (mode === "pending") {
        // 位移始终相对“多指落下的那一刻”计算（不更新基准点），
        // 以便慢速拖动也能累积到阈值，同时避免误判捏合。
        let movingTouch: Touch | null = null
        let movingCount = 0
        let stillCount = 0
        for (const t of touches) {
          const prev = touchPositionsRef.current.get(t.identifier)
          if (prev === undefined) {
            continue
          }
          const dist = Math.hypot(t.clientX - prev.x, t.clientY - prev.y)
          if (dist > TOUCH_MOVE_THRESHOLD) {
            movingCount += 1
            movingTouch = t
          } else if (dist <= TOUCH_ANCHOR_THRESHOLD) {
            stillCount += 1
          }
        }
        const canDraw = !latest.isPanning && !latest.isInteractiveSeg
        if (canDraw && movingCount === 1 && movingTouch && stillCount >= 1) {
          // 一根手指按住不动、另一根拖动 → 视为绘制（保留锚定绘制特性）
          touchModeRef.current = "draw"
          drawingTouchIdRef.current = movingTouch.identifier
          for (const t of touches) {
            touchPositionsRef.current.set(t.identifier, {
              x: t.clientX,
              y: t.clientY,
            })
          }
          // 丢弃以按住指头为起点的那段笔画，从当前移动指头重新开始
          latest.cancelCurrentStroke()
          latest.setIsDraging(true)
          latest.startStroke(touchPointXY(movingTouch, canvas))
          // 阻止库把这次多指当作 pinch
          event.stopPropagation()
          return
        }
        if (movingCount >= 2) {
          // 明显是两根手指一起动 → 手势，交给库处理缩放/平移
          for (const t of touches) {
            touchPositionsRef.current.set(t.identifier, {
              x: t.clientX,
              y: t.clientY,
            })
          }
          touchModeRef.current = "gesture"
          latest.cancelCurrentStroke()
          return
        }
        // 位移还不够明显，继续留在 pending，等下一次 touchmove 再判断
        return
      }
      if (mode === "draw") {
        for (const t of touches) {
          touchPositionsRef.current.set(t.identifier, {
            x: t.clientX,
            y: t.clientY,
          })
        }
        const drawingId = drawingTouchIdRef.current
        const t = Array.from(touches).find(
          (touch) => touch.identifier === drawingId
        )
        if (t && latest.isDraging) {
          latest.pushStrokePoint(touchPointXY(t, canvas))
        }
        event.stopPropagation()
        return
      }
      if (mode === "gesture") {
        for (const t of touches) {
          touchPositionsRef.current.set(t.identifier, {
            x: t.clientX,
            y: t.clientY,
          })
        }
        return
      }
    }

    const onTouchEnd = (event: TouchEvent) => {
      const latest = latestRef.current
      if (!latest) {
        return
      }
      const { touches, changedTouches } = event
      const changed = changedTouches[0]
      const mode = touchModeRef.current
      if (mode === "single") {
        touchPositionsRef.current.delete(changed.identifier)
        if (touches.length === 0) {
          touchModeRef.current = null
          drawingTouchIdRef.current = null
          if (latest.isInteractiveSeg) {
            const xy = touchPointXY(changed, canvas)
            const newClicks = [...latest.clicks]
            newClicks.push([xy.x, xy.y, 1, newClicks.length])
            latest.runInteractiveSeg(newClicks)
            latest.updateInteractiveSegState({ clicks: newClicks })
            return
          }
          if (latest.isPanning || latest.isInpainting) {
            return
          }
          if (!latest.isDraging || !strokeRef.current) {
            return
          }
          if (!latest.originalSrc || !latest.context?.canvas) {
            return
          }
          latest.setIsDraging(false)
          latest.commitCurrentStroke()
          if (!latest.runMannually) {
            latest.runInpainting()
          }
          return
        }
        // 还有手指按着，继续按单指处理
        const remaining = Array.from(touches)[0]
        drawingTouchIdRef.current = remaining.identifier
        touchPositionsRef.current.set(remaining.identifier, {
          x: remaining.clientX,
          y: remaining.clientY,
        })
        return
      }
      if (mode === "pending") {
        touchPositionsRef.current.delete(changed.identifier)
        if (touches.length === 0) {
          touchModeRef.current = null
        } else if (touches.length === 1) {
          touchModeRef.current = "single"
          const remaining = Array.from(touches)[0]
          drawingTouchIdRef.current = remaining.identifier
          touchPositionsRef.current.clear()
          touchPositionsRef.current.set(remaining.identifier, {
            x: remaining.clientX,
            y: remaining.clientY,
          })
        }
        return
      }
      if (mode === "draw") {
        touchPositionsRef.current.delete(changed.identifier)
        if (changed.identifier === drawingTouchIdRef.current) {
          // 绘制手指抬起 → 提交笔画
          touchModeRef.current = null
          drawingTouchIdRef.current = null
          if (
            !latest.isPanning &&
            !latest.isInpainting &&
            latest.isDraging &&
            strokeRef.current
          ) {
            latest.setIsDraging(false)
            latest.commitCurrentStroke()
            if (!latest.runMannually) {
              latest.runInpainting()
            }
          }
        } else {
          // 按住的手指抬起，绘制手指还在 → 用剩下那根手指继续
          const remaining = Array.from(touches).find(
            (t) => t.identifier !== changed.identifier
          )
          if (remaining) {
            drawingTouchIdRef.current = remaining.identifier
            touchPositionsRef.current.set(remaining.identifier, {
              x: remaining.clientX,
              y: remaining.clientY,
            })
          }
        }
        return
      }
      if (mode === "gesture") {
        for (const t of changedTouches) {
          touchPositionsRef.current.delete(t.identifier)
        }
        if (touches.length === 0) {
          touchModeRef.current = null
        }
      }
    }

    const onTouchCancel = () => {
      touchPositionsRef.current.clear()
      touchModeRef.current = null
      drawingTouchIdRef.current = null
      latestRef.current?.cancelCurrentStroke()
    }

    canvas.addEventListener("touchstart", onTouchStart, { passive: false })
    canvas.addEventListener("touchmove", onTouchMove, { passive: false })
    canvas.addEventListener("touchend", onTouchEnd)
    canvas.addEventListener("touchcancel", onTouchCancel)
    return () => {
      canvas.removeEventListener("touchstart", onTouchStart)
      canvas.removeEventListener("touchmove", onTouchMove)
      canvas.removeEventListener("touchend", onTouchEnd)
      canvas.removeEventListener("touchcancel", onTouchCancel)
    }
  }, [])

  // 滚轮处理：普通滚轮（无 Ctrl）→ 平移；Ctrl+滚轮/触控板捏合 → 由库缩放；
  // Alt+滚轮（SHORTCUT_KEY_CHANGE_BRUSH_SIZE 按下）→ 调整画笔大小。
  // 用原生非 passive 监听器，保证 preventDefault 生效（React 的 onWheel 是 passive 的）。
  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }
    const onWheel = (event: WheelEvent) => {
      if (isChangingBrushSizeByWheel) {
        const { deltaY } = event
        if (deltaY > 0) {
          increaseBaseBrushSize()
        } else if (deltaY < 0) {
          decreaseBaseBrushSize()
        }
        event.preventDefault()
        return
      }
      if (spacePressedRef.current) {
        // Space+滚轮：缩放（方向跟随用户滚动设置，见 getWheelDelta 注释）
        zoomByWheel(event)
        event.preventDefault()
        return
      }
      if (event.shiftKey) {
        // Shift+滚轮：横向平移照片（保留浏览器横向滚动的习惯）
        panBy(getWheelDelta(event), 0)
        event.preventDefault()
        return
      }
      if (event.ctrlKey) {
        // 触控板捏合 / Ctrl+滚轮：恒定速度缩放
        zoomByWheel(event)
        event.preventDefault()
        return
      }
      panBy(event.deltaX, event.deltaY)
      event.preventDefault()
    }
    // React 在 root 上绑定的 touchstart/touchmove 是 passive 的，无法 preventDefault。
    // 这里用原生非 passive 监听器阻止浏览器默认行为与兼容 mouse 事件，
    // 避免触屏操作同时触发 onMouseDown/onMouseDrag 造成双重绘制。
    const onNativeTouchStart = (event: TouchEvent) => event.preventDefault()
    const onNativeTouchMove = (event: TouchEvent) => event.preventDefault()

    container.addEventListener("wheel", onWheel, { passive: false })
    container.addEventListener("touchstart", onNativeTouchStart, {
      passive: false,
    })
    container.addEventListener("touchmove", onNativeTouchMove, {
      passive: false,
    })
    return () => {
      container.removeEventListener("wheel", onWheel)
      container.removeEventListener("touchstart", onNativeTouchStart)
      container.removeEventListener("touchmove", onNativeTouchMove)
    }
  }, [
    isChangingBrushSizeByWheel,
    panBy,
    zoomByWheel,
    getWheelDelta,
    increaseBaseBrushSize,
    decreaseBaseBrushSize,
  ])

  return (
    <div
      ref={containerRef}
      className="flex w-screen h-screen justify-center items-center"
      style={{ touchAction: "none" }}
      aria-hidden="true"
      onMouseMove={onMouseMove}
      onMouseUp={onPointerUp}
    >
      {renderCanvas()}
      <div
        ref={brushCursorRef}
        className="absolute rounded-[50%] border-[1px] border-[solid] border-[#ffcc00] pointer-events-none bg-[#ffcc00bb]"
        style={{
          left: 0,
          top: 0,
          width: `${brushSize * getCurScale()}px`,
          height: `${brushSize * getCurScale()}px`,
          opacity:
            showBrush &&
            !isInpainting &&
            !isPanning &&
            !interactiveSegState.isInteractiveSeg
              ? 1
              : 0,
          transform: "translate(-50%, -50%)",
        }}
      />
      {showRefBrush && renderBrush(getBrushStyle(windowCenterX, windowCenterY))}

      <div className="fixed flex bottom-5 border px-4 py-2 rounded-[3rem] gap-8 items-center justify-center backdrop-filter backdrop-blur-md bg-background/70">
        <Slider
          className="w-48"
          defaultValue={[50]}
          min={MIN_BRUSH_SIZE}
          max={MAX_BRUSH_SIZE}
          step={1}
          tabIndex={-1}
          value={[baseBrushSize]}
          onValueChange={(vals) => handleSliderChange(vals[0])}
          onClick={() => setShowRefBrush(false)}
        />
        <div className="flex gap-2">
          <IconButton
            tooltip="Reset zoom & pan"
            disabled={scale === minScale && panned === false}
            onClick={resetZoom}
          >
            <Expand />
          </IconButton>
          <IconButton
            tooltip="Undo"
            onClick={handleUndo}
            disabled={undoDisabled}
          >
            <Undo />
          </IconButton>
          <IconButton
            tooltip="Redo"
            onClick={handleRedo}
            disabled={redoDisabled}
          >
            <Redo />
          </IconButton>
          <IconButton
            tooltip="Show original image"
            onPointerDown={(ev) => {
              ev.preventDefault()
              setShowOriginal(() => {
                trackedTimeout(() => {
                  setSliderPos(100)
                }, 10)
                return true
              })
            }}
            onPointerUp={() => {
              trackedTimeout(() => {
                // 防止快速点击 show original image 按钮时图片消失
                setSliderPos(0)
              }, 10)

              trackedTimeout(() => {
                setShowOriginal(false)
              }, COMPARE_SLIDER_DURATION_MS)
            }}
            disabled={renders.length === 0}
          >
            <Eye />
          </IconButton>
          <IconButton
            tooltip="Save Image"
            disabled={!renders.length}
            onClick={download}
          >
            <Download />
          </IconButton>

          {settings.enableManualInpainting &&
          settings.model.model_type === "inpaint" ? (
            <IconButton
              tooltip="Run Inpainting"
              disabled={
                isProcessing || (!hadDrawSomething() && extraMasks.length === 0)
              }
              onClick={() => {
                runInpainting()
              }}
            >
              <Eraser />
            </IconButton>
          ) : (
            <></>
          )}
        </div>
      </div>
    </div>
  )
}
