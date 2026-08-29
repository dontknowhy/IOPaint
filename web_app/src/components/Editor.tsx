import {
  SyntheticEvent,
  useCallback,
  useEffect,
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
  cn,
  copyCanvasImage,
  downloadImage,
  drawLines,
  generateMask,
  getErrorMessage,
  isMidClick,
  isRightClick,
  mouseXY,
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
  BRUSH_COLOR,
  MAX_BRUSH_SIZE,
  MIN_BRUSH_SIZE,
  SHORTCUT_KEY_CHANGE_BRUSH_SIZE,
} from "@/lib/const"

const TOOLBAR_HEIGHT = 200
const COMPARE_SLIDER_DURATION_MS = 300
// 触控板捏合 / Ctrl+滚轮的恒定缩放速度：每单位 deltaY 缩放固定倍数
// （指数缩放，几何级数，任何缩放级别下手感一致）
const WHEEL_ZOOM_SPEED = 0.003
const MAX_SCALE = 50
// 双指中判定“哪根手指在动”的移动阈值（px）
const TOUCH_MOVE_THRESHOLD = 10
// 判定“哪根手指按住了没动”的阈值（px），避免把捏合开始阶段误判成锚定绘制
const TOUCH_ANCHOR_THRESHOLD = 6

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
  const brushCursorRef = useRef<HTMLDivElement>(null)
  const cursorPosRef = useRef<Point>({ x: -1, y: -1 })
  const cursorFrameRef = useRef<number>(0)
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

  // 已提交内容（已提交的笔迹/掩膜/分割临时掩膜）画在这个“底层”画布上，
  // 绘制进行中的笔画时只需把它原样拷贝到显示画布上，再一次性画当前笔画，
  // 避免每帧重复重绘所有已提交内容，也避免重复 stroke() 同一条累计路径。
  const maskBaseCanvasRef = useRef<HTMLCanvasElement | null>(null)

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
    // 始终与当前图片尺寸对齐：切换图片后尺寸变化时同步 base 画布，
    // 避免绘制/提交时用旧的画布尺寸导致笔画错位或“消失”
    if (base.width !== imageWidth || base.height !== imageHeight) {
      base.width = imageWidth
      base.height = imageHeight
    }
    const baseCtx = base.getContext("2d")
    if (!baseCtx) {
      return
    }
    baseCtx.clearRect(0, 0, base.width, base.height)
    temporaryMasks.forEach((maskImage) => {
      baseCtx.drawImage(maskImage, 0, 0, imageWidth, imageHeight)
    })
    extraMasks.forEach((maskImage) => {
      baseCtx.drawImage(maskImage, 0, 0, imageWidth, imageHeight)
    })

    if (
      interactiveSegState.isInteractiveSeg &&
      interactiveSegState.tmpInteractiveSegMask
    ) {
      baseCtx.drawImage(
        interactiveSegState.tmpInteractiveSegMask,
        0,
        0,
        imageWidth,
        imageHeight
      )
    }
    // 只画当前提交的笔画（curLineGroup）。inpaint 完成后由 runInpainting
    // 清空 curLineGroup，掩膜随之从画布清除（标准行为）。
    drawLines(baseCtx, curLineGroup)

    context.canvas.width = imageWidth
    context.canvas.height = imageHeight
    context.clearRect(0, 0, context.canvas.width, context.canvas.height)
    context.drawImage(base, 0, 0, imageWidth, imageHeight)
  }, [
    temporaryMasks,
    extraMasks,
    isOriginalLoaded,
    interactiveSegState,
    context,
    curLineGroup,
    imageHeight,
    imageWidth,
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
      if (width != context.canvas.width) {
        context.canvas.width = width
      }
      if (height != context.canvas.height) {
        context.canvas.height = height
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
    return () => {
      if (cursorFrameRef.current !== 0) {
        cancelAnimationFrame(cursorFrameRef.current)
      }
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
    strokeRef.current = { size: brushSize, pts: [pt] }
  }

  // 追加一个绘制点：把“已提交内容(底层画布) + 当前笔画(一次性重画)”合成到显示画布。
  // 整个路径只 stroke() 一次，笔刷透明度不会因为反复描边而自我叠加成实心，
  // 也不会随着笔画变长而不断重复重绘旧段落（避免掉帧）。
  const pushStrokePoint = useCallback(
    (pt: Point) => {
      const stroke = strokeRef.current
      const ctx = context
      if (!stroke || !ctx) {
        return
      }
      stroke.pts.push(pt)
      const base = maskBaseCanvasRef.current
      if (!base) {
        return
      }
      const canvas = ctx.canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(base, 0, 0, imageWidth, imageHeight)
      ctx.strokeStyle = BRUSH_COLOR
      ctx.lineCap = "round"
      ctx.lineJoin = "round"
      ctx.lineWidth = stroke.size ?? brushSize
      ctx.beginPath()
      ctx.moveTo(stroke.pts[0].x, stroke.pts[0].y)
      for (let i = 1; i < stroke.pts.length; i++) {
        ctx.lineTo(stroke.pts[i].x, stroke.pts[i].y)
      }
      ctx.stroke()
    },
    [context, imageWidth, imageHeight, brushSize]
  )

  const commitCurrentStroke = () => {
    const stroke = strokeRef.current
    if (!stroke) {
      return
    }
    strokeRef.current = null
    commitStroke(stroke)
  }

  // 取消进行中的笔画：丢弃 strokeRef 并重绘掩膜画布，移除已画上去的临时笔迹
  const cancelCurrentStroke = useCallback(() => {
    if (!strokeRef.current) {
      return
    }
    strokeRef.current = null
    setIsDraging(false)
    redrawMaskCanvas()
  }, [redrawMaskCanvas])

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
    pushStrokePoint(mouseXY(ev))
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
      const img = new Image()
      img.onload = () => {
        if (useStore.getState().file !== file) {
          return
        }
        updateInteractiveSegState({ tmpInteractiveSegMask: img })
      }
      img.src = blob
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

  // 恒定速度缩放：缩放量只与 deltaY（捏合/滚轮位移）成正比，
  // 不随缩放级别变化，绕光标所在位置缩放。
  const zoomByWheel = useCallback(
    (event: WheelEvent) => {
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
      const factor = Math.exp(-event.deltaY * WHEEL_ZOOM_SPEED)
      const newScale = Math.min(
        Math.max(scale * factor, minScale * 0.3),
        MAX_SCALE
      )
      if (newScale === scale) {
        return
      }
      const rect = content.getBoundingClientRect()
      const mouseX = (event.clientX - rect.left) / scale
      const mouseY = (event.clientY - rect.top) / scale
      const scaleDiff = newScale - scale
      instance.setTransformState(
        newScale,
        positionX - mouseX * scaleDiff,
        positionY - mouseY * scaleDiff
      )
      setScale(newScale)
      setPanned(true)
    },
    [minScale]
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
          window.setTimeout(() => {
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
        window.setTimeout(() => {
          setSliderPos(0)
        }, 10)
        window.setTimeout(() => {
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
        await downloadToOutput(
          renders[renders.length - 1],
          file.name,
          file.type
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
    const name = file.name.replace(/(\.[\w\d_-]+)$/i, "_cleanup$1")
    const curRender = renders[renders.length - 1]
    downloadImage(curRender.currentSrc, name)
    if (settings.enableDownloadMask) {
      let maskFileName = file.name.replace(/(\.[\w\d_-]+)$/i, "_mask$1")
      maskFileName = maskFileName.replace(/\.[^/.]+$/, ".jpg")

      const maskCanvas = generateMask(imageWidth, imageHeight, lineGroups)
      // Create a link
      const aDownloadLink = document.createElement("a")
      // Add the name of the file to the link
      aDownloadLink.download = maskFileName
      // Attach the data to the link
      aDownloadLink.href = maskCanvas.toDataURL("image/jpeg")
      // Get the code to click the download link
      aDownloadLink.click()
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
        if (context?.canvas) {
          await copyCanvasImage(context?.canvas)
          toast({
            title: "Copy inpainting result to clipboard",
          })
        }
      }
    },
    [renders, context]
  )

  // Toggle clean/zoom tool on spacebar.
  useKeyPressEvent(
    " ",
    (ev) => {
      if (!disableShortCuts) {
        ev?.preventDefault()
        ev?.stopPropagation()
        setShowBrush(false)
        setIsPanning(true)
      }
    },
    (ev) => {
      if (!disableShortCuts) {
        ev?.preventDefault()
        ev?.stopPropagation()
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
    }

    const handleBlur = () => {
      setIsChangingBrushSizeByWheel(false)
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
      window.setTimeout(() => {
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
                  const ctx = r.getContext("2d", { desynchronized: true })
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
                window.setTimeout(() => {
                  setSliderPos(100)
                }, 10)
                return true
              })
            }}
            onPointerUp={() => {
              window.setTimeout(() => {
                // 防止快速点击 show original image 按钮时图片消失
                setSliderPos(0)
              }, 10)

              window.setTimeout(() => {
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
