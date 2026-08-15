import React from "react"
import { cn } from "@/lib/utils"
import { twMerge } from "tailwind-merge"

const DRAG_HANDLE_BORDER = 2

const DragHandle = ({
  cursor,
  side1,
  side2,
  scale,
}: {
  cursor: string
  side1: string
  side2: string
  scale: number
}) => {
  const sideLength = 12
  const halfSideLength = sideLength / 2

  let xTrans = "0"
  let yTrans = "0"

  let side2Key = side2
  let side2Val = `${-halfSideLength}px`
  if (side2 === "") {
    side2Val = "50%"
    if (side1 === "left" || side1 === "right") {
      side2Key = "top"
      yTrans = "-50%"
    } else {
      side2Key = "left"
      xTrans = "-50%"
    }
  }

  return (
    <div
      className={cn(
        "w-3 h-3 z-[4] absolute content-[''] block border-2 border-primary pointer-events-auto hover:bg-primary",
        cursor
      )}
      style={{
        [side1]: -halfSideLength,
        [side2Key]: side2Val,
        transform: `translate(${xTrans}, ${yTrans}) scale(${1 / scale})`,
      }}
      data-ord={side1 + side2}
      aria-label={side1 + side2}
      tabIndex={-1}
      role="button"
    />
  )
}

const ResizeEdges = ({
  active,
  scale,
  onPointerDown,
}: {
  active: { top: boolean; right: boolean; bottom: boolean; left: boolean }
  scale: number
  onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void
}) => {
  return (
    <div onPointerDown={onPointerDown} className="absolute top-0 h-full w-full">
      {active.top ? (
        <div
          className="absolute pointer-events-auto top-0 left-0 w-full cursor-ns-resize h-3 mt-[-6px]"
          data-ord="top"
        />
      ) : (
        <></>
      )}
      {active.right ? (
        <div
          className="absolute pointer-events-auto top-0 right-0 h-full cursor-ew-resize w-3 mr-[-6px]"
          data-ord="right"
        />
      ) : (
        <></>
      )}
      {active.bottom ? (
        <div
          className="absolute pointer-events-auto bottom-0 left-0 w-full cursor-ns-resize h-3 mb-[-6px]"
          data-ord="bottom"
        />
      ) : (
        <></>
      )}
      {active.left ? (
        <div
          className="absolute pointer-events-auto top-0 left-0 h-full cursor-ew-resize w-3 ml-[-6px]"
          data-ord="left"
        />
      ) : (
        <></>
      )}
      {active.top && active.left ? (
        <DragHandle cursor="cursor-nw-resize" side1="top" side2="left" scale={scale} />
      ) : (
        <></>
      )}
      {active.top && active.right ? (
        <DragHandle cursor="cursor-ne-resize" side1="top" side2="right" scale={scale} />
      ) : (
        <></>
      )}
      {active.bottom && active.left ? (
        <DragHandle cursor="cursor-sw-resize" side1="bottom" side2="left" scale={scale} />
      ) : (
        <></>
      )}
      {active.bottom && active.right ? (
        <DragHandle cursor="cursor-se-resize" side1="bottom" side2="right" scale={scale} />
      ) : (
        <></>
      )}
      {active.top ? (
        <DragHandle cursor="cursor-ns-resize" side1="top" side2="" scale={scale} />
      ) : (
        <></>
      )}
      {active.bottom ? (
        <DragHandle cursor="cursor-ns-resize" side1="bottom" side2="" scale={scale} />
      ) : (
        <></>
      )}
      {active.left ? (
        <DragHandle cursor="cursor-ew-resize" side1="left" side2="" scale={scale} />
      ) : (
        <></>
      )}
      {active.right ? (
        <DragHandle cursor="cursor-ew-resize" side1="right" side2="" scale={scale} />
      ) : (
        <></>
      )}
    </div>
  )
}

const ResizeInfoBar = ({
  width,
  height,
  scale,
  onPointerDown,
  className,
}: {
  width: number
  height: number
  scale: number
  onPointerDown?: (e: React.PointerEvent<HTMLDivElement>) => void
  className?: string
}) => {
  return (
    <div
      className={twMerge(
        "border absolute pointer-events-auto px-2 py-1 rounded-full bg-background",
        "origin-top-left top-0 left-0",
        className
      )}
      style={{
        transform: `scale(${(1 / scale) * 0.8})`,
      }}
      onPointerDown={onPointerDown}
    >
      {width} x {height}
    </div>
  )
}

const ResizeBorder = ({
  width,
  height,
  scale,
}: {
  width: number
  height: number
  scale: number
}) => {
  return (
    <div
      className="outline-dashed outline-primary"
      style={{
        height,
        width,
        outlineWidth: `${(DRAG_HANDLE_BORDER / scale) * 1.3}px`,
      }}
    />
  )
}

export { DragHandle, ResizeBorder, ResizeEdges, ResizeInfoBar }
