// states.ts::runInpainting 单测（vitest + jsdom）。
// 只覆盖「掩膜来源」这一条判定：extraMasks（交互分割/插件/调整掩膜）不带笔迹，
// 也必须能发出 /inpaint 请求；完全没掩膜时仍然拦住并提示。
// jsdom 没有真实 canvas/编码器，generateMask/canvasToBlob/createDisplayImage
// 用桩替换（同 utils.test.ts 里 stub getContext 的做法）。
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { DisplayImage } from "./types"

const { inpaintMock, toastMock, httpStatusMock } = vi.hoisted(() => ({
  inpaintMock: vi.fn(),
  toastMock: vi.fn(),
  httpStatusMock: vi.fn(),
}))

vi.mock("@/lib/api", () => ({
  default: inpaintMock,
  exportImage: vi.fn(),
  getGenInfo: vi.fn(),
  httpStatus: httpStatusMock,
  postAdjustMask: vi.fn(),
  runPlugin: vi.fn(),
}))

vi.mock("@/components/ui/use-toast", () => ({
  toast: toastMock,
  useToast: () => ({ toast: toastMock, dismiss: vi.fn(), toasts: [] }),
}))

vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./utils")>()
  return {
    ...actual,
    generateMask: vi.fn(() => document.createElement("canvas")),
    canvasToBlob: vi.fn(async () => new Blob(["mask"], { type: "image/png" })),
    createDisplayImage: vi.fn(async (src: string) => ({
      src,
      bitmap: document.createElement("canvas"),
      width: 64,
      height: 64,
      logicalWidth: 64,
      logicalHeight: 64,
    })),
    srcToFile: vi.fn(
      async () => new File(["render"], "render.png", { type: "image/png" })
    ),
  }
})

import { useStore } from "./states"

const FILE = new File(["pixels"], "test.png", { type: "image/png" })

function resetStore(editorState: {
  extraMasks?: HTMLImageElement[]
  renders?: DisplayImage[]
}) {
  useStore.setState({
    file: FILE,
    imageWidth: 64,
    imageHeight: 64,
    isInpainting: false,
  })
  useStore.getState().updateEditorState({
    renders: editorState.renders ?? [],
    lineGroups: [],
    lastLineGroup: [],
    curLineGroup: [],
    extraMasks: editorState.extraMasks ?? [],
    prevExtraMasks: [],
    temporaryMasks: [],
  })
}

// jsdom 不实现 object URL（不是可 spy 的既有方法），直接装两个桩。
const createObjectURL = vi.fn(() => "blob:render")
const revokeObjectURL = vi.fn()

beforeEach(() => {
  inpaintMock.mockReset()
  toastMock.mockReset()
  httpStatusMock.mockReset()
  createObjectURL.mockClear()
  revokeObjectURL.mockClear()
  httpStatusMock.mockImplementation(
    (error) => (error as { status?: number } | null)?.status
  )
  Object.defineProperty(URL, "createObjectURL", {
    value: createObjectURL,
    writable: true,
    configurable: true,
  })
  Object.defineProperty(URL, "revokeObjectURL", {
    value: revokeObjectURL,
    writable: true,
    configurable: true,
  })
})

describe("runInpainting mask source", () => {
  it("sends the request when the mask only lives in extraMasks", async () => {
    resetStore({ extraMasks: [new Image()] })
    inpaintMock.mockResolvedValue({
      blob: new Blob(["result"], { type: "image/png" }),
      revision: "rev-1",
      width: 64,
      height: 64,
    })

    await useStore.getState().runInpainting()

    expect(toastMock).not.toHaveBeenCalled()
    expect(inpaintMock).toHaveBeenCalledTimes(1)
    const [, , , , maskBlob, strokes] = inpaintMock.mock.calls[0]
    expect(maskBlob).toBeInstanceOf(Blob)
    // extraMasks 必须走整图 PNG 分支，而不是空笔迹
    expect(strokes).toBeNull()
  })

  it("still refuses and prompts when there is no mask at all", async () => {
    resetStore({})

    await useStore.getState().runInpainting()

    expect(inpaintMock).not.toHaveBeenCalled()
    expect(toastMock).toHaveBeenCalledTimes(1)
    expect(toastMock.mock.calls[0][0].description).toBe(
      "Please draw mask on picture"
    )
  })
})

// 410 回退：render 的字节来源决定能否重传。显示预览是有损缩小版，重传会
// 被服务端当成新源图（永久降分辨率），只能报错；原始字节的 render 仍照旧重试。
function seededRender(preview: boolean): DisplayImage {
  return {
    src: "blob:render",
    bitmap: document.createElement("canvas"),
    width: 64,
    height: 64,
    logicalWidth: 4200,
    logicalHeight: 4200,
    preview,
  }
}

// 先跑一次成功请求，把 "blob:render" → revision 登记进内部映射（没有导出
// 的写入口），再换成待测的 render 状态；计数清零，断言只看 410 这一次。
async function seedExpiredRevision(preview: boolean) {
  resetStore({ extraMasks: [new Image()] })
  inpaintMock.mockResolvedValueOnce({
    blob: new Blob(["result"], { type: "image/jpeg" }),
    revision: "rev-1",
    width: 4200,
    height: 4200,
    preview,
  })
  await useStore.getState().runInpainting()
  inpaintMock.mockReset()
  useStore.getState().updateEditorState({
    renders: [seededRender(preview)],
    extraMasks: [new Image()],
  })
}

describe("runInpainting revision expiry", () => {
  it("does not re-upload the display preview when the revision expired", async () => {
    await seedExpiredRevision(true)
    inpaintMock.mockRejectedValue(
      Object.assign(new Error("gone"), { status: 410 })
    )

    await useStore.getState().runInpainting()

    expect(inpaintMock).toHaveBeenCalledTimes(1)
    expect(toastMock).toHaveBeenCalledTimes(1)
    expect(toastMock.mock.calls[0][0].description).toContain(
      "edit session has expired"
    )
  })

  it("still re-uploads a non-preview render exactly once", async () => {
    await seedExpiredRevision(false)
    inpaintMock
      .mockRejectedValueOnce(Object.assign(new Error("gone"), { status: 410 }))
      .mockResolvedValueOnce({
        blob: new Blob(["result"], { type: "image/png" }),
        revision: "rev-2",
        width: 64,
        height: 64,
        preview: false,
      })

    await useStore.getState().runInpainting()

    expect(inpaintMock).toHaveBeenCalledTimes(2)
    expect(toastMock).not.toHaveBeenCalled()
  })
})

// 迟到的响应不能把请求期间发生的 undo 盖回去：请求开始时的 renders/lineGroups
// 快照已经是过去式，再写回 state 等于静默撤销了用户的撤销。
describe("runInpainting stale response", () => {
  it("drops the response when the user undid while it was in flight", async () => {
    resetStore({
      extraMasks: [new Image()],
      renders: [seededRender(false)],
    })
    useStore.getState().updateEditorState({
      lineGroups: [[{ size: 10, pts: [{ x: 1, y: 1 }] }]],
    })

    let resolveInpaint!: (value: unknown) => void
    inpaintMock.mockReturnValue(
      new Promise((resolve) => {
        resolveInpaint = resolve
      })
    )

    const pending = useStore.getState().runInpainting()
    await vi.waitFor(() => expect(inpaintMock).toHaveBeenCalledTimes(1))

    useStore.getState().undo()
    expect(useStore.getState().editorState.renders).toHaveLength(0)
    expect(useStore.getState().editorState.lineGroups).toHaveLength(0)

    resolveInpaint({
      blob: new Blob(["late"], { type: "image/png" }),
      revision: "rev-late",
      width: 64,
      height: 64,
    })
    await pending

    // 修复前这里会变成 1（快照 + 新结果），撤销被重新盖回来
    expect(useStore.getState().editorState.renders).toHaveLength(0)
    expect(useStore.getState().editorState.lineGroups).toHaveLength(0)
    // redo 栈保留用户刚撤销的那一步，没有被 resetRedoState 清掉
    expect(useStore.getState().editorState.redoRenders).toHaveLength(1)
    expect(useStore.getState().editorState.redoLineGroups).toHaveLength(1)
  })
})

describe("runInpainting extender", () => {
  it("toasts instead of silently doing nothing when the extender equals the image", async () => {
    const originalSettings = useStore.getState().settings
    try {
      resetStore({ extraMasks: [new Image()] })
      useStore.getState().updateSettings({
        showExtender: true,
        model: {
          ...originalSettings.model,
          support_outpainting: true,
        },
      })
      useStore.setState({
        extenderState: { x: 0, y: 0, width: 64, height: 64 },
      })

      await useStore.getState().runInpainting()

      expect(inpaintMock).not.toHaveBeenCalled()
      expect(toastMock).toHaveBeenCalledTimes(1)
      expect(toastMock.mock.calls[0][0].description).toBe(
        "The extender area is the same as the image. Drag the extender box out first."
      )
    } finally {
      useStore.setState({ settings: originalSettings })
    }
  })
})

describe("clearMask", () => {
  it("also clears the last-run mask snapshot so the next Run does not reuse it", async () => {
    resetStore({})
    useStore.getState().updateEditorState({
      lastLineGroup: [{ size: 10, pts: [{ x: 1, y: 1 }] }],
      prevExtraMasks: [new Image()],
    })

    useStore.getState().clearMask()

    expect(useStore.getState().editorState.lastLineGroup).toHaveLength(0)
    expect(useStore.getState().editorState.prevExtraMasks).toHaveLength(0)

    await useStore.getState().runInpainting()

    // 修复前 lastLineGroup 还在，runInpainting 会重跑上一张掩膜
    expect(inpaintMock).not.toHaveBeenCalled()
    expect(toastMock.mock.calls[0][0].description).toBe(
      "Please draw mask on picture"
    )
  })
})

describe("brush size steps", () => {
  const getSize = () => useStore.getState().editorState.baseBrushSize
  const setSize = (size: number) =>
    useStore.getState().setBaseBrushSize(size)

  it("steps by 1 inside the first 20% of the slider range", () => {
    setSize(39)
    useStore.getState().increaseBaseBrushSize()
    expect(getSize()).toBe(40)

    setSize(39)
    useStore.getState().decreaseBaseBrushSize()
    expect(getSize()).toBe(38)
  })

  it("steps by 4 between 20% and 50% of the slider range", () => {
    setSize(41)
    useStore.getState().increaseBaseBrushSize()
    expect(getSize()).toBe(45)

    setSize(99)
    useStore.getState().increaseBaseBrushSize()
    expect(getSize()).toBe(103)

    setSize(99)
    useStore.getState().decreaseBaseBrushSize()
    expect(getSize()).toBe(95)
  })

  it("steps by 16 from 50% of the slider range upward", () => {
    setSize(100)
    useStore.getState().increaseBaseBrushSize()
    expect(getSize()).toBe(116)

    setSize(116)
    useStore.getState().decreaseBaseBrushSize()
    expect(getSize()).toBe(100)
  })

  it("returns to the same size after a step up and down at a tier boundary", () => {
    for (const size of [40, 97, 100]) {
      setSize(size)
      useStore.getState().increaseBaseBrushSize()
      useStore.getState().decreaseBaseBrushSize()
      expect(getSize()).toBe(size)
    }
  })

  it("clamps to the 1..200 slider range", () => {
    setSize(200)
    useStore.getState().increaseBaseBrushSize()
    expect(getSize()).toBe(200)

    setSize(1)
    useStore.getState().decreaseBaseBrushSize()
    expect(getSize()).toBe(1)
  })
})

describe("persist options", () => {
  it("merge tolerates a null persisted state", () => {
    const options = useStore.persist.getOptions()
    const current = useStore.getState()

    const merged = options.merge!(null, current)

    expect(merged.settings).toEqual(current.settings)
    expect(merged.fileManagerState).toEqual(current.fileManagerState)
    // 新对象，不是把 current 原样返回
    expect(merged.settings).not.toBe(current.settings)
  })

  it("merge fills missing settings from current defaults", () => {
    const options = useStore.persist.getOptions()
    const current = useStore.getState()

    const merged = options.merge!(
      {
        fileManagerState: { layout: "rows" },
        settings: {
          prompt: "old prompt",
          model: { name: "persisted-model" },
        },
      },
      current
    )

    expect(merged.settings.prompt).toBe("old prompt")
    expect(merged.settings.ldmSteps).toBe(current.settings.ldmSteps)
    // model 永远来自运行时，不被持久化数据覆盖
    expect(merged.settings.model).toBe(current.settings.model)
    expect(merged.fileManagerState.layout).toBe("rows")
    expect(merged.fileManagerState.sortBy).toBe(current.fileManagerState.sortBy)
  })

  it("migrate normalizes an older shape and drops unknown keys", () => {
    const options = useStore.persist.getOptions()
    const migrated = options.migrate!(
      {
        fileManagerState: { layout: "rows", legacy: 1 },
        settings: { prompt: "old prompt", sdMaskBlur: 4 },
      },
      1
    ) as {
      fileManagerState: Record<string, unknown>
      settings: Record<string, unknown>
    }

    expect(migrated.settings.prompt).toBe("old prompt")
    expect(migrated.settings.ldmSteps).toBe(
      useStore.getState().settings.ldmSteps
    )
    expect("sdMaskBlur" in migrated.settings).toBe(false)
    expect(migrated.fileManagerState.layout).toBe("rows")
    expect("legacy" in migrated.fileManagerState).toBe(false)
  })
})
