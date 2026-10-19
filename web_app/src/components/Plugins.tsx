import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu"
import { Button } from "./ui/button"
import {
  Blocks,
  Fullscreen,
  MousePointerClick,
  Slice,
  Smile,
} from "lucide-react"
import { useStore } from "@/lib/states"
import { useTranslation } from "react-i18next"
import { PluginInfo, PluginName } from "@/lib/types"
import Spinner from "@/components/ui/spinner"

// TODO: get plugin config from server and using form-render??
const pluginMap = {
  [PluginName.RemoveBG]: {
    IconClass: Slice,
    showNameKey: "plugins.removeBgName",
  },
  [PluginName.AnimeSeg]: {
    IconClass: Slice,
    showNameKey: "plugins.animeSegName",
  },
  [PluginName.RealESRGAN]: {
    IconClass: Fullscreen,
    showNameKey: "plugins.realesrganName",
  },
  [PluginName.GFPGAN]: {
    IconClass: Smile,
    showNameKey: "plugins.gfpganName",
  },
  [PluginName.RestoreFormer]: {
    IconClass: Smile,
    showNameKey: "plugins.restoreFormerName",
  },
  [PluginName.InteractiveSeg]: {
    IconClass: MousePointerClick,
    showNameKey: "plugins.interactiveSegName",
  },
} as const

const Plugins = () => {
  const { t } = useTranslation()
  const [
    file,
    plugins,
    isPluginRunning,
    updateInteractiveSegState,
    runRenderablePlugin,
  ] = useStore((state) => [
    state.file,
    state.serverConfig.plugins,
    state.isPluginRunning,
    state.updateInteractiveSegState,
    state.runRenderablePlugin,
  ])
  const disabled = !file

  if (plugins.length === 0) {
    return null
  }

  const onPluginClick = (genMask: boolean, pluginName: string) => {
    if (pluginName === PluginName.InteractiveSeg) {
      updateInteractiveSegState({ isInteractiveSeg: true })
    } else {
      runRenderablePlugin(genMask, pluginName)
    }
  }

  const renderRealESRGANPlugin = () => {
    return (
      <DropdownMenuSub key="RealESRGAN">
        <DropdownMenuSubTrigger disabled={disabled}>
          <div className="flex gap-2 items-center">
            <Fullscreen />
            {t("plugins.realesrganName")}
          </div>
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent>
          <DropdownMenuItem
            onClick={() =>
              runRenderablePlugin(false, PluginName.RealESRGAN, { upscale: 2 })
            }
          >
            {t("plugins.upscale2x")}
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() =>
              runRenderablePlugin(false, PluginName.RealESRGAN, { upscale: 4 })
            }
          >
            {t("plugins.upscale4x")}
          </DropdownMenuItem>
        </DropdownMenuSubContent>
      </DropdownMenuSub>
    )
  }

  const renderGenImageAndMaskPlugin = (plugin: PluginInfo) => {
    const { IconClass, showNameKey } = pluginMap[plugin.name as PluginName]
    return (
      <DropdownMenuSub key={plugin.name}>
        <DropdownMenuSubTrigger disabled={disabled}>
          <div className="flex gap-2 items-center">
            <IconClass className="p-1" />
            {t(showNameKey)}
          </div>
        </DropdownMenuSubTrigger>
        <DropdownMenuSubContent>
          <DropdownMenuItem onClick={() => onPluginClick(false, plugin.name)}>
            {t("plugins.removeBackground")}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => onPluginClick(true, plugin.name)}>
            {t("plugins.generateMask")}
          </DropdownMenuItem>
        </DropdownMenuSubContent>
      </DropdownMenuSub>
    )
  }

  const renderPlugins = () => {
    return plugins.map((plugin: PluginInfo) => {
      const { IconClass, showNameKey } = pluginMap[plugin.name as PluginName]
      if (plugin.name === PluginName.RealESRGAN) {
        return renderRealESRGANPlugin()
      }
      if (
        plugin.name === PluginName.RemoveBG ||
        plugin.name === PluginName.AnimeSeg
      ) {
        return renderGenImageAndMaskPlugin(plugin)
      }
      return (
        <DropdownMenuItem
          key={plugin.name}
          onClick={() => onPluginClick(false, plugin.name)}
          disabled={disabled}
        >
          <div className="flex gap-2 items-center">
            <IconClass className="p-1" />
            {t(showNameKey)}
          </div>
        </DropdownMenuItem>
      )
    })
  }

  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger
        className="border rounded-lg z-10 bg-background outline-none"
        tabIndex={-1}
      >
        <Button variant="ghost" size="icon" asChild className="p-1.5">
          {isPluginRunning ? (
            <div role="status">
              <Spinner className="w-5 h-5" />
            </div>
          ) : (
            <Blocks strokeWidth={1} />
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent side="bottom" align="start">
        {renderPlugins()}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export default Plugins
