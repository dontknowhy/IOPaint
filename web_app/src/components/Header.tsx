import { IconButton, ImageUploadButton } from "@/components/ui/button"
import Shortcuts from "@/components/Shortcuts"

import PromptInput from "./PromptInput"
import { RotateCw, Image } from "lucide-react"
import FileManager, { MASK_TAB } from "./FileManager"
import { getMediaBlob, getMediaFile } from "@/lib/api"
import { useStore } from "@/lib/states"
import { getErrorMessage } from "@/lib/utils"
import SettingsDialog from "./Settings"
import Coffee from "./Coffee"
import { useToast } from "./ui/use-toast"

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
          <FileManager photoWidth={512} onPhotoClick={handleOnPhotoClick} />
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
        {serverConfig.disableModelSwitch ? <></> : <SettingsDialog />}
      </div>
    </header>
  )
}

export default Header
