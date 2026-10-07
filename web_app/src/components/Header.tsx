import { IconButton, ImageUploadButton } from "@/components/ui/button"
import Shortcuts from "@/components/Shortcuts"

import { lazy, Suspense } from "react"
import PromptInput from "./PromptInput"
import { RotateCw, Image } from "lucide-react"
import { getMediaBlob, getMediaFile } from "@/lib/api"
import { MASK_TAB } from "@/lib/const"
import { useStore } from "@/lib/states"
import { getErrorMessage } from "@/lib/utils"
import Coffee from "./Coffee"
import { useToast } from "./ui/use-toast"

// 设置面板（zod + react-hook-form + react-query）和文件管理器（fuse.js +
// react-photo-album + lodash）都是点开才用得上的，而且里面还各带一个 Dialog。
// 静态引入时它们全算进首屏 chunk（Firefox 上 DCL 就要 370ms），改成按需加载
// 后这部分在 first paint 之后才下载，不挡首屏。
const FileManager = lazy(() => import("./FileManager"))
const SettingsDialog = lazy(() => import("./Settings"))

const Header = () => {
  const [
    file,
    isInpainting,
    serverConfig,
    model,
    setFile,
    runInpainting,
    showPrevMask,
    hidePrevMask,
    handleFileManagerMaskSelect,
  ] = useStore((state) => [
    state.file,
    state.isInpainting,
    state.serverConfig,
    state.settings.model,
    state.setFile,
    state.runInpainting,
    state.showPrevMask,
    state.hidePrevMask,
    state.handleFileManagerMaskSelect,
  ])

  const { toast } = useToast()

  const handleRerunLastMask = () => {
    runInpainting()
  }

  const onRerunMouseEnter = () => {
    showPrevMask()
  }

  const onRerunMouseLeave = () => {
    hidePrevMask()
  }

  const handleOnPhotoClick = async (tab: string, filename: string) => {
    try {
      if (tab === MASK_TAB) {
        const maskBlob = await getMediaBlob(tab, filename)
        handleFileManagerMaskSelect(maskBlob)
      } else {
        const newFile = await getMediaFile(tab, filename)
        setFile(newFile)
      }
    } catch (e) {
      toast({
        variant: "destructive",
        description: getErrorMessage(e),
      })
      return
    }
  }

  return (
    <header className="h-[60px] px-6 py-4 absolute top-[0] flex justify-between items-center w-full z-20 border-b backdrop-filter backdrop-blur-md bg-background/70">
      <div className="flex items-center gap-1">
        {serverConfig.enableFileManager ? (
          <Suspense fallback={null}>
            <FileManager photoWidth={512} onPhotoClick={handleOnPhotoClick} />
          </Suspense>
        ) : (
          <></>
        )}

        <ImageUploadButton
          disabled={isInpainting}
          tooltip="Upload image"
          onFileUpload={(file) => {
            setFile(file)
          }}
        >
          <Image />
        </ImageUploadButton>

        {file && !model.need_prompt ? (
          <IconButton
            disabled={isInpainting}
            tooltip="Rerun previous mask"
            onClick={handleRerunLastMask}
            onMouseEnter={onRerunMouseEnter}
            onMouseLeave={onRerunMouseLeave}
          >
            <RotateCw />
          </IconButton>
        ) : (
          <></>
        )}
      </div>

      {model.need_prompt ? <PromptInput /> : <></>}

      <div className="flex gap-1">
        <Coffee />
        <Shortcuts />
        {serverConfig.disableModelSwitch ? <></> : (
          <Suspense fallback={null}>
            <SettingsDialog />
          </Suspense>
        )}
      </div>
    </header>
  )
}

export default Header
