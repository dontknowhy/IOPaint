// 触屏多指判定纯函数单测：锚定绘制的 dwell、捏合识别、锚定指漂移取消。
import { describe, expect, it } from "vitest"
import {
  decideTouchGesture,
  lastTouchPoint,
  type TouchGestureInput,
} from "./touchGesture"

const MOVE = 10
const ANCHOR = 6
const DWELL = 120

// 两根手指：id 1 是锚定指，id 2 是绘制指
const startPoints = [
  { id: 1, x: 0, y: 0 },
  { id: 2, x: 0, y: 0 },
]

function input(overrides: Partial<TouchGestureInput>): TouchGestureInput {
  return {
    mode: "pending",
    startPoints,
    currentPoints: startPoints,
    drawingId: null,
    anchorStart: null,
    pendingSince: null,
    now: 0,
    canDraw: true,
    moveThreshold: MOVE,
    anchorThreshold: ANCHOR,
    dwellMs: DWELL,
    ...overrides,
  }
}

describe("decideTouchGesture", () => {
  it("single finger goes straight to draw", () => {
    const decision = decideTouchGesture(
      input({
        startPoints: [{ id: 7, x: 0, y: 0 }],
        currentPoints: [{ id: 7, x: 12, y: 34 }],
      })
    )
    expect(decision.mode).toBe("draw")
    expect(decision.drawingId).toBe(7)
  })

  it("one finger 2px, the other 30px for 80ms is still pending", () => {
    const currentPoints = [
      { id: 1, x: 2, y: 0 },
      { id: 2, x: 30, y: 0 },
    ]
    const first = decideTouchGesture(
      input({ currentPoints, now: 0, pendingSince: null })
    )
    expect(first.mode).toBe("pending")
    expect(first.pendingSince).toBe(0)

    const second = decideTouchGesture(
      input({ currentPoints, now: 80, pendingSince: first.pendingSince })
    )
    expect(second.mode).toBe("pending")
    expect(second.pendingSince).toBe(0)
  })

  it("the same anchor + moving finger held for 150ms becomes draw", () => {
    const decision = decideTouchGesture(
      input({
        currentPoints: [
          { id: 1, x: 2, y: 0 },
          { id: 2, x: 30, y: 0 },
        ],
        pendingSince: 0,
        now: 150,
      })
    )
    expect(decision.mode).toBe("draw")
    expect(decision.drawingId).toBe(2)
    expect(decision.anchorStart?.id).toBe(1)
  })

  it("drops the dwell timer when the condition is not continuous", () => {
    const decision = decideTouchGesture(
      input({
        currentPoints: [
          { id: 1, x: 2, y: 0 },
          { id: 2, x: 2, y: 0 },
        ],
        pendingSince: 0,
        now: 80,
      })
    )
    expect(decision.mode).toBe("pending")
    expect(decision.pendingSince).toBeNull()
  })

  it("both fingers moving is a gesture even before the dwell elapses", () => {
    const decision = decideTouchGesture(
      input({
        currentPoints: [
          { id: 1, x: 30, y: 0 },
          { id: 2, x: 30, y: 0 },
        ],
        now: 40,
      })
    )
    expect(decision.mode).toBe("gesture")
  })

  it("returns to gesture when the anchor finger drifts after entering draw", () => {
    const decision = decideTouchGesture(
      input({
        mode: "draw",
        drawingId: 2,
        anchorStart: { id: 1, x: 0, y: 0 },
        currentPoints: [
          { id: 1, x: 40, y: 0 },
          { id: 2, x: 60, y: 0 },
        ],
        now: 200,
      })
    )
    expect(decision.mode).toBe("gesture")
    expect(decision.drawingId).toBeNull()
  })

  it("keeps drawing after the anchor finger lifts (only the drawing finger left)", () => {
    const decision = decideTouchGesture(
      input({
        mode: "draw",
        drawingId: 2,
        anchorStart: { id: 1, x: 0, y: 0 },
        currentPoints: [{ id: 2, x: 60, y: 0 }],
        now: 200,
      })
    )
    expect(decision.mode).toBe("draw")
    expect(decision.drawingId).toBe(2)
    expect(decision.anchorStart).toBeNull()
  })

  it("keeps drawing while the anchor finger only jitters below the threshold", () => {
    const decision = decideTouchGesture(
      input({
        mode: "draw",
        drawingId: 2,
        anchorStart: { id: 1, x: 0, y: 0 },
        currentPoints: [
          { id: 1, x: 3, y: 0 },
          { id: 2, x: 60, y: 0 },
        ],
        now: 200,
      })
    )
    expect(decision.mode).toBe("draw")
    expect(decision.drawingId).toBe(2)
  })
})

describe("lastTouchPoint", () => {
  const touches = [
    { identifier: 11, clientX: 1, clientY: 2 },
    { identifier: 22, clientX: 3, clientY: 4 },
  ]

  it("picks the drawing finger and ignores the others", () => {
    expect(lastTouchPoint(touches, 22)).toBe(touches[1])
  })

  it("returns null when the drawing finger is not in changedTouches", () => {
    expect(lastTouchPoint(touches, 99)).toBeNull()
    expect(lastTouchPoint(touches, null)).toBeNull()
  })
})
