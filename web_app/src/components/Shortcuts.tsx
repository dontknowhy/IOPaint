import { Keyboard } from "lucide-react"
import { useTranslation } from "react-i18next"
import { IconButton } from "@/components/ui/button"
import { useToggle } from "@uidotdev/usehooks"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog"
import useHotKey from "@/hooks/useHotkey"

interface ShortcutProps {
  content: string
  keys: string[]
}

function ShortCut(props: ShortcutProps) {
  const { content, keys } = props

  return (
    <div className="flex justify-between">
      <div>{content}</div>
      <div className="flex gap-[8px]">
        {keys.map((k) => (
          // TODO: 优化快捷键显示
          <div className="border px-2 py-1 rounded-lg" key={k}>
            {k}
          </div>
        ))}
      </div>
    </div>
  )
}

const isMac = function () {
  return /macintosh|mac os x/i.test(navigator.userAgent)
}

const CmdOrCtrl = () => {
  return isMac() ? "Cmd" : "Ctrl"
}

export function Shortcuts() {
  const [open, toggleOpen] = useToggle(false)
  const { t } = useTranslation()

  useHotKey("h", () => {
    toggleOpen()
  })

  return (
    <Dialog open={open} onOpenChange={toggleOpen}>
      <DialogTrigger asChild>
        <IconButton tooltip={t("shortcuts.tooltip")}>
          <Keyboard />
        </IconButton>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t("shortcuts.title")}</DialogTitle>
          <div className="flex gap-2 flex-col pt-4">
            <ShortCut content={t("shortcuts.pan")} keys={["Space + Drag"]} />
            <ShortCut content={t("shortcuts.resetZoomPan")} keys={["Esc"]} />
            <ShortCut content={t("shortcuts.zoomIn")} keys={["PageUp"]} />
            <ShortCut content={t("shortcuts.zoomOut")} keys={["PageDown"]} />
            <ShortCut
              content={t("shortcuts.zoomInOut")}
              keys={["Space + Scroll"]}
            />
            <ShortCut
              content={t("shortcuts.panHorizontally")}
              keys={["Shift + Scroll"]}
            />
            <ShortCut content={t("shortcuts.decreaseBrushSize")} keys={["["]} />
            <ShortCut content={t("shortcuts.increaseBrushSize")} keys={["]"]} />
            <ShortCut
              content={t("shortcuts.viewOriginalImage")}
              keys={["Hold Tab"]}
            />

            <ShortCut content={t("shortcuts.undo")} keys={[CmdOrCtrl(), "Z"]} />
            <ShortCut
              content={t("shortcuts.redo")}
              keys={[CmdOrCtrl(), "Shift", "Z"]}
            />
            <ShortCut
              content={t("shortcuts.copyResult")}
              keys={[CmdOrCtrl(), "C"]}
            />
            <ShortCut
              content={t("shortcuts.pasteImage")}
              keys={[CmdOrCtrl(), "V"]}
            />
            <ShortCut
              content={t("shortcuts.triggerManuallyInpainting")}
              keys={["Shift", "R"]}
            />
            <ShortCut content={t("shortcuts.toggleHotkeysDialog")} keys={["H"]} />
            <ShortCut
              content={t("shortcuts.toggleSettingsDialog")}
              keys={["S"]}
            />
            <ShortCut content={t("shortcuts.toggleFileManager")} keys={["F"]} />
          </div>
        </DialogHeader>
      </DialogContent>
    </Dialog>
  )
}

export default Shortcuts
