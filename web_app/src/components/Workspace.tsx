import { lazy, Suspense, useEffect } from "react"
import Editor from "./Editor"
import { currentModel } from "@/lib/api"
import { useStore } from "@/lib/states"
import ImageSize from "./ImageSize"
import Plugins from "./Plugins"
import { InteractiveSeg } from "./InteractiveSeg"
import SidePanel from "./SidePanel"

// 进度条要连 socket.io（engine.io + parser 加起来上百 KB），但它只在真正开始
// 出图后才用得上 —— 按需加载，首屏不必先背这段。
const DiffusionProgress = lazy(() => import("./DiffusionProgress"))

const Workspace = () => {
  const [file, updateSettings] = useStore((state) => [
    state.file,
    state.updateSettings,
  ])

  useEffect(() => {
    const fetchCurrentModel = async () => {
      const model = await currentModel()
      updateSettings({ model })
    }
    fetchCurrentModel()
  }, [updateSettings])

  return (
    <>
      <div className="flex gap-3 absolute top-[68px] left-[24px] items-center">
        <Plugins />
        <ImageSize />
      </div>
      <InteractiveSeg />
      <Suspense fallback={null}>
        <DiffusionProgress />
      </Suspense>
      <SidePanel />
      {file ? <Editor file={file} /> : <></>}
    </>
  )
}

export default Workspace
