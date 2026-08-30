import { useStore } from "@/lib/states"
import { useEffect, useState } from "react"
import useDragResize from "@/hooks/useDragResize"
import { Rect } from "@/lib/types"
import {
  ResizeBorder,
  ResizeEdges,
  ResizeInfoBar,
} from "@/components/ui/resize"

interface Props {
  maxHeight: number
  maxWidth: number
  scale: number
  minHeight: number
  minWidth: number
  show: boolean
}

const clamp = (
  newPos: number,
  newLength: number,
  oldPos: number,
  oldLength: number,
  minLength: number,
  maxLength: number
): [number, number] => {
  if (newPos !== oldPos && newLength === oldLength) {
    if (newPos < 0) {
      return [0, oldLength]
    }
    if (newPos + newLength > maxLength) {
      return [maxLength - oldLength, oldLength]
    }
  } else {
    if (newLength < minLength) {
      if (newPos === oldPos) {
        return [newPos, minLength]
      }
      return [newPos + newLength - minLength, minLength]
    }
    if (newPos < 0) {
      return [0, newPos + newLength]
    }
    if (newPos + newLength > maxLength) {
      return [newPos, maxLength - newPos]
    }
  }

  return [newPos, newLength]
}

const Cropper = (props: Props) => {
  const { minHeight, minWidth, maxHeight, maxWidth, scale, show } = props

  const [
    imageWidth,
    imageHeight,
    isInpainting,
    isSD,
    { x, y, width, height },
    setCropperState,
    isResizing,
    setIsResizing,
  ] = useStore((state) => [
    state.imageWidth,
    state.imageHeight,
    state.isInpainting,
    state.isSD(),
    state.cropperState,
    state.setCropperState,
    state.isCropperExtenderResizing,
    state.setIsCropperExtenderResizing,
  ])

  const [isMoving, setIsMoving] = useState(false)

  useEffect(() => {
    setCropperState({
      x: Math.round((maxWidth - 512) / 2),
      y: Math.round((maxHeight - 512) / 2),
    })
    // TODO: 换了一张较小的图片，cropper 的起始位置和边界要修改
    // TODO: 一开始的 scale 不对
  }, [maxHeight, maxWidth, imageWidth, imageHeight, setCropperState])

  const clampRect = (proposed: Rect, _ord: string, init: Rect): Rect => {
    const [clampedX, clampedWidth] = clamp(
      proposed.x,
      proposed.width,
      init.x,
      init.width,
      minWidth,
      maxWidth
    )
    const [clampedY, clampedHeight] = clamp(
      proposed.y,
      proposed.height,
      init.y,
      init.height,
      minHeight,
      maxHeight
    )
    return {
      x: clampedX,
      y: clampedY,
      width: clampedWidth,
      height: clampedHeight,
    }
  }

  const applyRect = (updates: Partial<Rect>) => {
    setCropperState(updates)
  }

  const resize = useDragResize({
    active: isResizing,
    setActive: setIsResizing,
    scale,
    isInpainting,
    applyRect,
    clampRect,
  })

  const move = useDragResize({
    active: isMoving,
    setActive: setIsMoving,
    scale,
    isInpainting,
    applyRect,
    clampRect,
  })

  const onCropPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const { ord } = (e.target as HTMLElement).dataset
    if (ord) {
      resize.startResize({ x, y, width, height }, e.clientX, e.clientY, ord)
    }
  }

  const onInfoBarPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    move.startResize({ x, y, width, height }, e.clientX, e.clientY, "")
  }

  if (show === false || !isSD) {
    return null
  }

  return (
    <div className="absolute h-full w-full overflow-hidden pointer-events-none z-[2]">
      <div
        className="relative pointer-events-none z-[2] [box-shadow:0_0_0_9999px_rgba(0,_0,_0,_0.5)]"
        style={{ height, width, left: x, top: y }}
      >
        <ResizeBorder width={width} height={height} scale={scale} />
        <ResizeInfoBar
          width={width}
          height={height}
          scale={scale}
          onPointerDown={onInfoBarPointerDown}
          className="hover:cursor-move"
        />
        <ResizeEdges
          active={{ top: true, right: true, bottom: true, left: true }}
          scale={scale}
          onPointerDown={onCropPointerDown}
        />
      </div>
    </div>
  )
}

export default Cropper
