export const ACCENT_COLOR = "#ffcc00bb"
export const DEFAULT_BRUSH_SIZE = 40
export const MIN_BRUSH_SIZE = 1
export const MAX_BRUSH_SIZE = 200
export const MODEL_TYPE_INPAINT = "inpaint"
export const MODEL_TYPE_DIFFUSERS_SD = "diffusers_sd"
export const MODEL_TYPE_DIFFUSERS_SDXL = "diffusers_sdxl"
export const MODEL_TYPE_DIFFUSERS_SD_INPAINT = "diffusers_sd_inpaint"
export const MODEL_TYPE_DIFFUSERS_SDXL_INPAINT = "diffusers_sdxl_inpaint"
export const MODEL_TYPE_OTHER = "diffusers_other"
export const BRUSH_COLOR = "#ffcc00bb"

// 显示用掩膜画布的像素上限。笔迹是矢量数据，提交时后端按原图分辨率重新
// 栅格化，所以屏幕上的画布不需要跟原图一样大。如果按原分辨率逐帧
// clearRect + drawImage + stroke()，6000x4000 这种图每帧要处理 2400 万像素，
// 是画笔掉帧的主因（代价随图片分辨率线性增长）。按上限等比缩小后，
// 每帧的重绘量被限制住，画笔手感与图片分辨率无关。
export const MASK_CANVAS_MAX_PIXELS = 4 * 1024 * 1024
// 显示用原图/渲染画布（非掩膜）的像素上限，也是 `/inpaint` 预览响应的上限。
// 这张画布的内容绘制时并不变化，但它每帧都会被合成为一张 GPU 共享纹理
// （trace 里的 Canvas2DResourceProviderSharedImage::ProduceCanvasResource），
// 代价随位图面积增长：6000x6000（36M px / 144MB）时主线程 Commit 约 19ms/帧，
// 画笔画 120 帧有 21 帧掉到 33ms 以上，平移/缩放同样卡；位图压到 16M px 以内后
// 掉帧归零。坐标仍用原图单位（scaleX/scaleY 经 ctx.setTransform 映射），配合
// CSS width/height 撑回原图大小，布局盒子与缩放行为都不变，掩膜那套同理。
export const IMAGE_CANVAS_MAX_PIXELS = 16 * 1024 * 1024

export const LDM = "ldm"
export const CV2 = "cv2"

export const PAINT_BY_EXAMPLE = "Fantasy-Studio/Paint-by-Example"
export const INSTRUCT_PIX2PIX = "timbrooks/instruct-pix2pix"
export const KANDINSKY_2_2 = "kandinsky-community/kandinsky-2-2-decoder-inpaint"
export const POWERPAINT = "Sanster/PowerPaint-V1-stable-diffusion-inpainting"
export const ANYTEXT = "Sanster/AnyText"

export const DEFAULT_NEGATIVE_PROMPT =
  "out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature"

export const SHORTCUT_KEY_CHANGE_BRUSH_SIZE = "Alt"

/** FileManager 对话框里「掩膜目录」这个 tab 的标识，Header 选文件时要认它 */
export const MASK_TAB = "mask"

export const HD_STRATEGY = "Crop"
export const HD_STRATEGY_CROP_TRIGGER_SIZE = 640
export const HD_STRATEGY_CROP_MARGIN = 128
export const HD_STRATEGY_RESIZE_LIMIT = 2048
