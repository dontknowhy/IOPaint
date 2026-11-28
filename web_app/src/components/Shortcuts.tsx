import { Fragment, type ReactNode } from "react"
import { Keyboard } from "lucide-react"
import { useTranslation } from "react-i18next"
import { IconButton } from "@/components/ui/button"
import { useToggle } from "@uidotdev/usehooks"
import { cn } from "@/lib/utils"
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
  /** 键位之间的连接符；组合键用 +，可替代键位用 / */
  separator?: string
  /** 键位前的说明文字，例如 Hold */
  prefix?: string
}

/** 键帽：mono 字体 + 上下边框阴影，让按键和普通文字一眼可分 */
function KeyCap({ children }: { children: ReactNode }) {
  return (
    <kbd
      className={cn(
        "inline-flex h-6 min-w-6 items-center justify-center rounded-md",
        "border border-foreground/15 bg-foreground/10 px-1.5 font-mono text-xs font-medium text-foreground",
        "shadow-[0_2px_0_0_hsl(var(--foreground)/0.15)]"
      )}
    >
      {children}
    </kbd>
  )
}

function ShortCut(props: ShortcutProps) {
  const { content, keys, separator = "+", prefix } = props

  return (
    <div className="flex items-center justify-between gap-4">
      <div className="text-sm">{content}</div>
      <div className="flex shrink-0 items-center gap-1.5 whitespace-nowrap">
        {prefix && (
          <span className="text-xs text-muted-foreground">{prefix}</span>
        )}
        {keys.map((k, i) => (
          <Fragment key={k}>
            {i > 0 && (
              <span className="text-xs text-muted-foreground">{separator}</span>
            )}
            <KeyCap>{k}</KeyCap>
          </Fragment>
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
      <DialogContent className="max-h-[calc(100vh-2rem)]">
        <DialogHeader className="min-h-0">
          <DialogTitle>{t("shortcuts.title")}</DialogTitle>
          <div className="flex gap-2 flex-col pt-4 overflow-y-auto min-h-0 pr-1">
            <ShortCut content={t("shortcuts.pan")} keys={["Space", "Drag"]} />
            <ShortCut content={t("shortcuts.resetZoomPan")} keys={["Esc"]} />
            <ShortCut content={t("shortcuts.zoomIn")} keys={["PageUp"]} />
            <ShortCut content={t("shortcuts.zoomOut")} keys={["PageDown"]} />
            <ShortCut
              content={t("shortcuts.zoomInOut")}
              keys={["Space", "Scroll"]}
            />
            <ShortCut
              content={t("shortcuts.panHorizontally")}
              keys={["Shift", "Scroll"]}
            />
            <ShortCut
              content={t("shortcuts.decreaseBrushSize")}
              keys={["[", "←", "↑"]}
              separator="/"
            />
            <ShortCut
              content={t("shortcuts.increaseBrushSize")}
              keys={["]", "→", "↓"]}
              separator="/"
            />
            <ShortCut
              content={t("shortcuts.viewOriginalImage")}
              keys={["Tab"]}
              prefix="Hold"
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
