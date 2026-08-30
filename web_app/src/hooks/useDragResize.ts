import { useEffect, useState } from "react"
import { Rect } from "@/lib/types"

interface EVData {
  initX: number
  initY: number
  initHeight: number
  initWidth: number
  startResizeX: number
  startResizeY: number
  ord: string
}

interface Options {
  active: boolean
  setActive: (value: boolean) => void
  scale: number
  isInpainting: boolean
  applyRect: (updates: Partial<Rect>) => void
  clampRect: (proposed: Rect, ord: string, init: Rect) => Rect
}

const DOC_MOVE_OPTS = { capture: true, passive: false }

const useDragResize = ({
  active,
  setActive,
  scale,
  isInpainting,
  applyRect,
  clampRect,
}: Options) => {
  const [evData, setEVData] = useState<EVData>({
    initX: 0,
    initY: 0,
    initHeight: 0,
    initWidth: 0,
    startResizeX: 0,
    startResizeY: 0,
    ord: "top",
  })

  const startResize = (
    rect: Rect,
    startResizeX: number,
    startResizeY: number,
    ord: string
  ) => {
    setEVData({
      initX: rect.x,
      initY: rect.y,
      initHeight: rect.height,
      initWidth: rect.width,
      startResizeX,
      startResizeY,
      ord,
    })
    setActive(true)
  }

  useEffect(() => {
    if (!active) {
      return
    }
    const onPointerMove = (e: PointerEvent) => {
      if (isInpainting) {
        return
      }
      const offsetX = Math.round((e.clientX - evData.startResizeX) / scale)
      const offsetY = Math.round((e.clientY - evData.startResizeY) / scale)
      const init = {
        x: evData.initX,
        y: evData.initY,
        width: evData.initWidth,
        height: evData.initHeight,
      }
      const { ord } = evData
      const proposed = { ...init }
      if (ord.includes("left")) {
        proposed.width = init.width - offsetX
        proposed.x = init.x + offsetX
      }
      if (ord.includes("right")) {
        proposed.width = init.width + offsetX
      }
      if (ord.includes("top")) {
        proposed.height = init.height - offsetY
        proposed.y = init.y + offsetY
      }
      if (ord.includes("bottom")) {
        proposed.height = init.height + offsetY
      }
      const clamped = clampRect(proposed, ord, init)
      applyRect({
        x: clamped.x,
        y: clamped.y,
        width: clamped.width,
        height: clamped.height,
      })
    }

    const onPointerDone = () => {
      setActive(false)
    }

    document.addEventListener("pointermove", onPointerMove, DOC_MOVE_OPTS)
    document.addEventListener("pointerup", onPointerDone, DOC_MOVE_OPTS)
    document.addEventListener("pointercancel", onPointerDone, DOC_MOVE_OPTS)
    return () => {
      document.removeEventListener("pointermove", onPointerMove, DOC_MOVE_OPTS)
      document.removeEventListener("pointerup", onPointerDone, DOC_MOVE_OPTS)
      document.removeEventListener("pointercancel", onPointerDone, DOC_MOVE_OPTS)
    }
  }, [active, isInpainting, scale, evData, clampRect, applyRect, setActive])

  return { evData, startResize }
}

export default useDragResize
