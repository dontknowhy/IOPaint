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
