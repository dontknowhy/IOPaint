import { AxiosError } from "axios"
import axios from "axios"
import {
  Filename,
  GenInfo,
  ModelInfo,
  PowerPaintTask,
  Rect,
  ServerConfig,
} from "@/lib/types"
import { Settings } from "@/lib/states"
import { convertToBase64, srcToFile } from "@/lib/utils"
import {
  HD_STRATEGY,
  HD_STRATEGY_CROP_MARGIN,
  HD_STRATEGY_CROP_TRIGGER_SIZE,
  HD_STRATEGY_RESIZE_LIMIT,
} from "@/lib/const"

export const API_ENDPOINT = import.meta.env.DEV
  ? import.meta.env.VITE_BACKEND + "/api/v1"
  : "/api/v1"

const api = axios.create({
  baseURL: API_ENDPOINT,
})

api.interceptors.response.use(
  (res) => res,
  async (error: AxiosError) => {
    let detail = ""
    try {
      const data = error.response?.data
      if (data instanceof Blob) {
        detail = (JSON.parse(await data.text()) as { errors?: string }).errors ?? ""
      } else if (data && typeof data === "object") {
        const body = data as { errors?: string; detail?: string }
        detail = body.errors ?? body.detail ?? ""
      } else if (typeof data === "string") {
        detail = data
      }
    } catch {
      // ignore parsing errors, fall back to the axios error message
    }
    throw new Error(
      `${detail || error.message}\nPlease take a screenshot of the detailed error message in your terminal`
    )
  }
)

export default async function inpaint(
  imageFile: File,
  settings: Settings,
  croperRect: Rect,
  extenderState: Rect,
  mask: File | Blob,
  paintByExampleImage: File | null = null
) {
  const imageBase64 = await convertToBase64(imageFile)
  const maskBase64 = await convertToBase64(mask)
  const exampleImageBase64 = paintByExampleImage
    ? await convertToBase64(paintByExampleImage)
    : null

  const res = await api.post(
    "/inpaint",
    {
      image: imageBase64,
      mask: maskBase64,
      ldm_steps: settings.ldmSteps,
      ldm_sampler: settings.ldmSampler,
      zits_wireframe: settings.zitsWireframe,
      cv2_flag: settings.cv2Flag,
      cv2_radius: settings.cv2Radius,
      hd_strategy: HD_STRATEGY,
      hd_strategy_crop_trigger_size: HD_STRATEGY_CROP_TRIGGER_SIZE,
      hd_strategy_crop_margin: HD_STRATEGY_CROP_MARGIN,
      hd_strategy_resize_limit: HD_STRATEGY_RESIZE_LIMIT,
      prompt: settings.prompt,
      negative_prompt: settings.negativePrompt,
      use_croper: settings.showCropper,
      croper_x: croperRect.x,
      croper_y: croperRect.y,
      croper_height: croperRect.height,
      croper_width: croperRect.width,
      use_extender: settings.showExtender,
      extender_x: extenderState.x,
      extender_y: extenderState.y,
      extender_height: extenderState.height,
      extender_width: extenderState.width,
      sd_mask_blur: settings.sdMaskBlur,
      sd_strength: settings.sdStrength,
      sd_steps: settings.sdSteps,
      sd_guidance_scale: settings.sdGuidanceScale,
      sd_sampler: settings.sdSampler,
      sd_seed: settings.seedFixed ? settings.seed : -1,
      sd_match_histograms: settings.sdMatchHistograms,
      sd_lcm_lora: settings.enableLCMLora,
      paint_by_example_example_image: exampleImageBase64,
      p2p_image_guidance_scale: settings.p2pImageGuidanceScale,
      enable_controlnet: settings.enableControlnet,
      controlnet_conditioning_scale: settings.controlnetConditioningScale,
      controlnet_method: settings.controlnetMethod
        ? settings.controlnetMethod
        : "",
      enable_brushnet: settings.enableBrushNet,
      brushnet_method: settings.brushnetMethod ? settings.brushnetMethod : "",
      brushnet_conditioning_scale: settings.brushnetConditioningScale,
      enable_powerpaint_v2: settings.enablePowerPaintV2,
      powerpaint_task: settings.showExtender
        ? PowerPaintTask.outpainting
        : settings.powerpaintTask,
    },
    { responseType: "blob" }
  )
  return {
    blob: res.data as Blob,
    seed: res.headers["x-seed"] as string | undefined,
  }
}

export async function getServerConfig(): Promise<ServerConfig> {
  const res = await api.get(`/server-config`)
  return res.data
}

export async function switchModel(name: string): Promise<ModelInfo> {
  const res = await api.post(`/model`, { name })
  return res.data
}

export async function switchPluginModel(
  plugin_name: string,
  model_name: string
) {
  return api.post(`/switch_plugin_model`, { plugin_name, model_name })
}

export async function currentModel(): Promise<ModelInfo> {
  const res = await api.get("/model")
  return res.data
}

export async function runPlugin(
  genMask: boolean,
  name: string,
  imageFile: File,
  upscale?: number,
  clicks?: number[][]
) {
  const imageBase64 = await convertToBase64(imageFile)
  const p = genMask ? "run_plugin_gen_mask" : "run_plugin_gen_image"
  const res = await api.post(
    `/${p}`,
    {
      name,
      image: imageBase64,
      scale: upscale,
      clicks,
    },
    { responseType: "blob" }
  )
  return { blob: res.data as Blob }
}

async function getMedia(tab: string, filename: string): Promise<Blob> {
  const res = await api.get("/media_file", {
    params: { tab, filename },
    responseType: "blob",
  })
  return res.data
}

export async function getMediaFile(tab: string, filename: string) {
  const blob = await getMedia(tab, filename)
  return new File([blob], filename, {
    type: blob.type || "image/png",
  })
}

export async function getMediaBlob(tab: string, filename: string) {
  return getMedia(tab, filename)
}

export async function getMedias(tab: string): Promise<Filename[]> {
  const res = await api.get(`medias`, { params: { tab } })
  return res.data
}

export async function downloadToOutput(
  image: HTMLImageElement,
  filename: string,
  mimeType: string
) {
  const file = await srcToFile(image.src, filename, mimeType)
  const fd = new FormData()
  fd.append("file", file)
  try {
    await api.post("/save_image", fd)
  } catch (error) {
    throw new Error(`Something went wrong: ${error}`)
  }
}

export async function getGenInfo(file: File): Promise<GenInfo> {
  const fd = new FormData()
  fd.append("file", file)
  const res = await api.post(`/gen-info`, fd)
  return res.data
}

export async function postAdjustMask(
  mask: File | Blob,
  operate: "expand" | "shrink" | "reverse",
  kernel_size: number
) {
  const maskBase64 = await convertToBase64(mask)
  const res = await api.post(
    "/adjust_mask",
    {
      mask: maskBase64,
      operate: operate,
      kernel_size: kernel_size,
    },
    { responseType: "blob" }
  )
  return res.data
}
