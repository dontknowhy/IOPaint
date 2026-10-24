import { useEffect, useRef, useState } from "react"
import { createDisplayImage } from "@/lib/utils"
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

  useEffect(() => {
    if (!file) {
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
          return
        }
        setImage(displayImage)
        setIsLoaded(true)
      })
      .catch((e) => {
        console.error(e)
        if (!cancelled) {
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
