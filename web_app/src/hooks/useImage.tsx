import { useEffect, useRef, useState } from "react"

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

export { useImage }
