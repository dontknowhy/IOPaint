import { useStore } from "@/lib/states"
import { ExtenderDirection } from "@/lib/types"
import useDragResize, { Rect } from "@/hooks/useDragResize"
import {
  ResizeBorder,
  ResizeEdges,
  ResizeInfoBar,
} from "@/components/ui/resize"

interface Props {
  scale: number
  minHeight: number
  minWidth: number
  show: boolean
}

const clamp = (
  newPos: number,
  newLength: number,
  oldPos: number,
  minLength: number
): [number, number] => {
  if (newLength < minLength) {
    if (newPos === oldPos) {
      return [newPos, minLength]
    }
    return [newPos + newLength - minLength, minLength]
  }

  return [newPos, newLength]
}

const Extender = (props: Props) => {
  const { minHeight, minWidth, scale, show } = props

  const [
    isInpainting,
    imageHeight,
    imageWidth,
    isSD,
    { x, y, width, height },
    setExtenderState,
    extenderDirection,
    isResizing,
    setIsResizing,
  ] = useStore((state) => [
    state.isInpainting,
    state.imageHeight,
    state.imageWidth,
    state.isSD(),
    state.extenderState,
    state.setExtenderState,
    state.settings.extenderDirection,
    state.isCropperExtenderResizing,
    state.setIsCropperExtenderResizing,
  ])

  const clampRect = (proposed: Rect, ord: string, init: Rect): Rect => {
    const result = {
      x: proposed.x,
      y: proposed.y,
      width: proposed.width,
      height: proposed.height,
    }
    const clampLeftRight = (
      newPos: number,
      newLength: number,
      oldPos: number
    ): [number, number] => clamp(newPos, newLength, oldPos, minWidth)
    const clampTopBottom = (
      newPos: number,
      newLength: number,
      oldPos: number
    ): [number, number] => clamp(newPos, newLength, oldPos, minHeight)

    const moveTop = () => {
      if (extenderDirection === ExtenderDirection.xy) {
        if (result.y > 0) {
          result.y = 0
          result.height = init.height - Math.abs(init.y)
        }
      } else {
        const [clampedY, clampedHeight] = clampTopBottom(
          result.y,
          result.height,
          init.y
        )
        result.y = clampedY
        result.height = clampedHeight
      }
    }

    const moveBottom = () => {
      const [clampedY, clampedHeight] = clampTopBottom(
        init.y,
        result.height,
        init.y
      )
      result.y = clampedY
      result.height = clampedHeight
      if (
        extenderDirection === ExtenderDirection.xy &&
        result.height < Math.abs(result.y) + imageHeight
      ) {
        result.height = Math.abs(result.y) + imageHeight
      }
    }

    const moveLeft = () => {
      if (extenderDirection === ExtenderDirection.xy) {
        if (result.x > 0) {
          result.x = 0
          result.width = init.width - Math.abs(init.x)
        }
      } else {
        const [clampedX, clampedWidth] = clampLeftRight(
          result.x,
          result.width,
          init.x
        )
        result.x = clampedX
        result.width = clampedWidth
      }
    }

    const moveRight = () => {
      const [clampedX, clampedWidth] = clampLeftRight(
        init.x,
        result.width,
        init.x
      )
      result.x = clampedX
      result.width = clampedWidth
      if (
        extenderDirection === ExtenderDirection.xy &&
        result.width < Math.abs(result.x) + imageWidth
      ) {
        result.width = Math.abs(result.x) + imageWidth
      }
    }

    if (ord.includes("top")) {
      moveTop()
    }
    if (ord.includes("bottom")) {
      moveBottom()
    }
    if (ord.includes("left")) {
      moveLeft()
    }
    if (ord.includes("right")) {
      moveRight()
    }
    return result
  }

  const applyRect = (updates: Partial<Rect>) => {
    setExtenderState(updates)
  }

  const resize = useDragResize({
    active: isResizing,
    setActive: setIsResizing,
    scale,
    isInpainting,
    applyRect,
    clampRect,
  })

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const { ord } = (e.target as HTMLElement).dataset
    if (ord) {
      resize.startResize({ x, y, width, height }, e.clientX, e.clientY, ord)
    }
  }

  if (show === false || !isSD) {
    return null
  }

  return (
    <div className="absolute h-full w-full pointer-events-none z-[2]">
      <div
        className="relative pointer-events-none z-[2] [box-shadow:0_0_0_9999px_rgba(0,_0,_0,_0.5)]"
        style={{ height, width, left: x, top: y }}
      >
        <ResizeBorder width={width} height={height} scale={scale} />
        <ResizeInfoBar width={width} height={height} scale={scale} />
        <ResizeEdges
          active={{
            top:
              extenderDirection === ExtenderDirection.y ||
              extenderDirection === ExtenderDirection.xy,
            bottom:
              extenderDirection === ExtenderDirection.y ||
              extenderDirection === ExtenderDirection.xy,
            left:
              extenderDirection === ExtenderDirection.x ||
              extenderDirection === ExtenderDirection.xy,
            right:
              extenderDirection === ExtenderDirection.x ||
              extenderDirection === ExtenderDirection.xy,
          }}
          scale={scale}
          onPointerDown={onPointerDown}
        />
      </div>
    </div>
  )
}

export default Extender
