// 触屏多指手势判定。Editor 的原生 touch 监听器只负责执行副作用，
// 这里保持纯函数：给同样的坐标/时间戳一定得到同样的结论（可单测）。
export type TouchPoint = {
  id: number
  x: number
  y: number
}

export type TouchMode = "pending" | "draw" | "gesture"

export type TouchGestureInput = {
  /** 当前所处状态；pending 表示多指刚落下、意图未定 */
  mode: TouchMode
  /** 多指落下时的基准坐标（pending 期间不更新） */
  startPoints: TouchPoint[]
  currentPoints: TouchPoint[]
  drawingId?: number | null
  /** 进入 draw 时锚定指的起点，用于识别「其实是捏合」的漂移 */
  anchorStart?: TouchPoint | null
  /** 第一次满足绘制条件的时间戳，未满足时为 null */
  pendingSince: number | null
  now: number
  /** 是否允许绘制（平移/交互分割时为 false） */
  canDraw: boolean
  moveThreshold: number
  anchorThreshold: number
  dwellMs: number
}

export type TouchGestureDecision = {
  mode: TouchMode
  drawingId: number | null
  anchorStart: TouchPoint | null
  pendingSince: number | null
}

const NO_DRAW = {
  drawingId: null,
  anchorStart: null,
  pendingSince: null,
} as const

/**
 * 多指状态下每一帧的意图判定：
 * - 只有一根手指移动、另一根基本没动，且这个状态连续保持 dwellMs → draw；
 * - 两根手指一起移动 → gesture（交给缩放/平移）；
 * - 已进入 draw 后锚定指漂移超过 moveThreshold → 判定为捏合，回 gesture。
 * 单指不属于捏合场景，直接返回 draw（由调用方的 single 分支负责落笔）。
 */
export function decideTouchGesture(
  input: TouchGestureInput
): TouchGestureDecision {
  const {
    mode,
    startPoints,
    currentPoints,
    drawingId = null,
    anchorStart = null,
    pendingSince,
    now,
    canDraw,
    moveThreshold,
    anchorThreshold,
    dwellMs,
  } = input

  if (currentPoints.length <= 1) {
    return {
      mode: "draw",
      drawingId: currentPoints[0]?.id ?? drawingId,
      anchorStart: null,
      pendingSince: null,
    }
  }

  if (mode === "gesture") {
    return { mode: "gesture", ...NO_DRAW }
  }

  if (mode === "draw") {
    if (!anchorStart) {
      // 锚定指已经抬起（只剩绘制指），不存在捏合风险
      return { mode: "draw", drawingId, anchorStart: null, pendingSince: null }
    }
    const anchor = currentPoints.find((point) => point.id === anchorStart.id)
    if (!anchor) {
      // 锚定指这一帧不在 touches 里（touchend 后只剩绘制指）
      return { mode: "draw", drawingId, anchorStart: null, pendingSince: null }
    }
    const drift = Math.hypot(anchor.x - anchorStart.x, anchor.y - anchorStart.y)
    if (drift > moveThreshold) {
      return { mode: "gesture", ...NO_DRAW }
    }
    return { mode: "draw", drawingId, anchorStart, pendingSince: null }
  }

  const startById = new Map(startPoints.map((point) => [point.id, point]))
  let moving: TouchPoint | null = null
  let movingCount = 0
  let stillCount = 0
  let stillPoint: TouchPoint | null = null
  for (const point of currentPoints) {
    const start = startById.get(point.id)
    if (!start) {
      continue
    }
    const dist = Math.hypot(point.x - start.x, point.y - start.y)
    if (dist > moveThreshold) {
      movingCount += 1
      moving = point
    } else if (dist <= anchorThreshold) {
      stillCount += 1
      stillPoint = point
    }
  }

  if (movingCount >= 2) {
    return { mode: "gesture", ...NO_DRAW }
  }
  if (canDraw && movingCount === 1 && moving && stillCount >= 1) {
    const since = pendingSince ?? now
    if (now - since >= dwellMs) {
      return {
        mode: "draw",
        drawingId: moving.id,
        anchorStart: stillPoint,
        pendingSince: null,
      }
    }
    return { mode: "pending", ...NO_DRAW, pendingSince: since }
  }
  return { mode: "pending", ...NO_DRAW }
}

/**
 * 从 changedTouches 里挑出驱动笔画的那根手指（identifier 匹配）。
 * touchend 的坐标可能比最后一次 touchmove 更靠前，commit 前要用它补最后一个点。
 */
export function lastTouchPoint<T extends { identifier: number }>(
  changedTouches: ArrayLike<T>,
  drawingId: number | null
): T | null {
  if (drawingId === null) {
    return null
  }
  for (let i = 0; i < changedTouches.length; i += 1) {
    const touch = changedTouches[i]
    if (touch.identifier === drawingId) {
      return touch
    }
  }
  return null
}
