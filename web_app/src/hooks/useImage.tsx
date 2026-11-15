import { useEffect, useRef, useState } from "react"
import { createDisplayImage, releaseDisplayImage } from "@/lib/utils"
import { DisplayImage } from "@/lib/types"

function useImage(file: File | null): [HTMLImageElement, boolean] {
  const [image] = useState(new Image())
  const [isLoaded, setIsLoaded] = useState(false)
  const blobUrlRef = useRef<string | null>(null)

  useEffect(() => {
    if (!file) {
      return
    }
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current)
      blobUrlRef.current = null
    }
    image.onload = () => {
      setIsLoaded(true)
    }
    setIsLoaded(false)
    const url = URL.createObjectURL(file)
    blobUrlRef.current = url
    image.src = url
    return () => {
      image.onload = null
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current)
        blobUrlRef.current = null
      }
    }
  }, [file, image])

  return [image, isLoaded]
}

/**
 * 原图按显示位图尺寸解码：解析 header 拿逻辑尺寸后直接
 * `createImageBitmap(file, {resizeWidth, resizeHeight})`，避免为 12/30MP 图
 * 先落一张全分辨率位图再缩进 16Mi 画布。
 */
function useDisplayImage(
  file: File | null
): [DisplayImage | null, boolean] {
  const [image, setImage] = useState<DisplayImage | null>(null)
  const [isLoaded, setIsLoaded] = useState(false)
  const urlRef = useRef<string | null>(null)
  const imageRef = useRef<DisplayImage | null>(null)

  // 卸载时归还最后一张图的像素内存（ImageBitmap 的 64MB 不在 JS 堆里，
  // 没人 close 就会一直留在 renderer 进程）。
  useEffect(() => {
    return () => {
      releaseDisplayImage(imageRef.current)
      imageRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!file) {
      releaseDisplayImage(imageRef.current)
      imageRef.current = null
      setImage(null)
      setIsLoaded(false)
      return
    }
    let cancelled = false
    const url = URL.createObjectURL(file)
    urlRef.current = url
    setIsLoaded(false)
    createDisplayImage(url, file)
      .then((displayImage) => {
        if (cancelled) {
          // 解码完成时这张图已经被下一张取代，没人会绘制它。
          releaseDisplayImage(displayImage)
          return
        }
        // 新图已就绪，旧图从这一帧起不会再被画到画布上（Editor 下一帧
        // 拿到的 original 就是新图），在这里归还它的位图存储。
        releaseDisplayImage(imageRef.current)
        imageRef.current = displayImage
        setImage(displayImage)
        setIsLoaded(true)
      })
      .catch((e) => {
        console.error(e)
        if (!cancelled) {
          releaseDisplayImage(imageRef.current)
          imageRef.current = null
          setImage(null)
          setIsLoaded(false)
        }
      })
    return () => {
      cancelled = true
      if (urlRef.current === url) {
        URL.revokeObjectURL(url)
        urlRef.current = null
      }
    }
  }, [file])

  return [image, isLoaded]
}

export { useImage, useDisplayImage }
