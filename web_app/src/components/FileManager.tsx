import {
  useEffect,
  useState,
  useCallback,
  useRef,
  FormEvent,
} from "react"
import orderBy from "lodash/orderBy"
import { useTranslation } from "react-i18next"
import PhotoAlbum, { RenderImageProps } from "react-photo-album"
import "react-photo-album/styles.css"
import { BarsArrowDownIcon, BarsArrowUpIcon } from "@heroicons/react/24/outline"
import {
  MagnifyingGlassIcon,
  ViewHorizontalIcon,
  ViewGridIcon,
} from "@radix-ui/react-icons"
import { useToggle } from "react-use"
import { useDebounce } from "@uidotdev/usehooks"
import Fuse from "fuse.js"
import { useToast } from "@/components/ui/use-toast"
import { API_ENDPOINT, getMedias } from "@/lib/api"
import { MASK_TAB } from "@/lib/const"
import { getErrorMessage } from "@/lib/utils"
import { IconButton } from "./ui/button"
import { Input } from "./ui/input"
import { Dialog, DialogContent, DialogTitle } from "./ui/dialog"
import { Tabs, TabsList, TabsTrigger } from "./ui/tabs"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select"
import { ScrollArea } from "./ui/scroll-area"
import { DialogTrigger } from "@radix-ui/react-dialog"
import { useStore } from "@/lib/states"
import { Filename, SortBy, SortOrder } from "@/lib/types"
import { FolderClosed } from "lucide-react"
import useHotKey from "@/hooks/useHotkey"

interface Photo {
  src: string
  height: number
  width: number
  name: string
}

// <img> 没有 src 时浏览器会画破图图标，占位用 1x1 透明 GIF（内联，不发请求）
const PLACEHOLDER_IMAGE =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

// 网格里一次性存在全部已加载的 <img>，但只有接近视口的才真正发起缩略图请求：
// 3948 张壁纸一张缩略图约 277ms（24MP 全图解码），全量打开要十几分钟。
// 原生 loading="lazy" 在嵌套滚动容器里会提前加载上百张，这里用观察器精确控制。
function LazyPhotoImage({
  src,
  renderedWidth,
  renderedHeight,
  ...props
}: RenderImageProps & {
  renderedWidth: number
  renderedHeight: number
}) {
  const ref = useRef<HTMLImageElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const element = ref.current
    if (!element || visible) {
      return
    }
    // 上下各预加载两行（左右各两列同理）：用户往回滚是常态，只往前预取
    // 会在反向滚动时露出空白。按这张图实际渲染尺寸算，不用固定像素——
    // 窗口大小和 rows/masonry 布局不同，固定值不是预取太多就是不够。
    // root 必须给真正的滚动容器：root 是浏览器视口时，祖先的 overflow
    // 会先把元素裁掉，rootMargin 扩出来的区域等于不存在。
    const root = element.closest<HTMLElement>(
      "[data-radix-scroll-area-viewport]"
    )
    const marginY = Math.ceil(renderedHeight * 2)
    const marginX = Math.ceil(renderedWidth * 2)
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setVisible(true)
          observer.disconnect()
        }
      },
      { root, rootMargin: `${marginY}px ${marginX}px` }
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [visible, renderedWidth, renderedHeight])

  return <img ref={ref} src={visible ? src : PLACEHOLDER_IMAGE} {...props} />
}

const IMAGE_TAB = "input"
const OUTPUT_TAB = "output"
// 一页 100 张：滚动到接近底部再拉下一页，避免一次为整个目录生成缩略图
const PAGE_SIZE = 100

// Select 的 value 保持稳定的枚举值，显示名走语言包。
const SORT_BY_LABEL_KEYS = {
  [SortBy.NAME]: "fileManager.sortName",
  [SortBy.CTIME]: "fileManager.sortCreatedTime",
  [SortBy.MTIME]: "fileManager.sortModifiedTime",
} as const

interface Props {
  onPhotoClick(tab: string, filename: string): void
  photoWidth: number
}

export default function FileManager(props: Props) {
  const { onPhotoClick, photoWidth } = props
  const [open, toggleOpen] = useToggle(false)

  const [fileManagerState, updateFileManagerState, updateAppState] = useStore(
    (state) => [
      state.fileManagerState,
      state.updateFileManagerState,
      state.updateAppState,
    ]
  )

  const { toast } = useToast()
  const { t } = useTranslation()
  const [scrollTop, setScrollTop] = useState(0)
  const [closeScrollTop, setCloseScrollTop] = useState(0)

  const ref = useRef(null)
  const debouncedSearchText = useDebounce(fileManagerState.searchText, 300)
  const [tab, setTab] = useState(IMAGE_TAB)
  // 文件名和它所属的 tab 绑在一起存：tab 切换后 Effect 里拿到的 filenames
  // 还是上一个 tab 的，直接用会渲染出 404 缩略图、还能点到新目录里不存在的文件
  const [fileList, setFileList] = useState<{
    tab: string
    items: Filename[]
    total: number
  }>({
    tab: IMAGE_TAB,
    items: [],
    total: 0,
  })
  // 搜索沿用全量 Fuse：输入搜索词时才把整份名单拉回来（不含缩略图）
  const [allItems, setAllItems] = useState<{ tab: string; items: Filename[] }>({
    tab: IMAGE_TAB,
    items: [],
  })
  const [photos, setPhotos] = useState<Photo[]>([])
  const [photoIndex, setPhotoIndex] = useState(0)
  // FE-3/D-19：列表拉取的请求序号，只认最新一次
  const listReqSeqRef = useRef(0)
  const loadingMoreRef = useRef(false)
  const loadMoreRef = useRef<() => Promise<void>>(async () => {})
  const scrollRootRef = useRef<HTMLDivElement | null>(null)
  // 打开时要把上次的位置滚回来，但名单是分页异步到的，得等照片渲染出来再滚
  const pendingRestoreRef = useRef(0)

  useHotKey("f", () => {
    toggleOpen()
  })

  useHotKey(
    "left",
    () => {
      // FE-2/D-14：列表为空（尚未加载/目录无文件）时直接忽略，
      // 否则 photos[newIndex].name 会抛 TypeError
      if (photos.length === 0) {
        return
      }
      const current = Math.min(photoIndex, photos.length - 1)
      const newIndex = current > 0 ? current - 1 : 0
      setPhotoIndex(newIndex)
      onPhotoClick(tab, photos[newIndex].name)
    },
    [photoIndex, photos]
  )

  useHotKey(
    "right",
    () => {
      // 同上：空列表忽略；同时把越界的 photoIndex 夹回新列表范围
      if (photos.length === 0) {
        return
      }
      const current = Math.min(photoIndex, photos.length - 1)
      const newIndex =
        current < photos.length - 1 ? current + 1 : photos.length - 1
      if (current >= photos.length - 3) {
        // 已加载的这一页到头了，先拉下一页，继续按右键才不会卡在边界
        void loadMoreRef.current()
      }
      setPhotoIndex(newIndex)
      onPhotoClick(tab, photos[newIndex].name)
    },
    [photoIndex, photos]
  )

  useEffect(() => {
    if (!open) {
      setCloseScrollTop(scrollTop)
    }
  }, [open, scrollTop])

  const onRefChange = useCallback(
    (node: HTMLDivElement | null) => {
      // Radix 的 ScrollArea 把 ref 转发到根节点，真正滚动的是内部的 viewport
      scrollRootRef.current = node
      if (node && open) {
        pendingRestoreRef.current = closeScrollTop
      }
    },
    [open, closeScrollTop]
  )

  useEffect(() => {
    // FE-3/D-19：快速切换 tab 时旧请求可能晚到，会把上一个 tab 的列表盖上去。
    // 双保险：AbortController 直接取消在途请求，请求序号兜底丢弃过期响应。
    const controller = new AbortController()
    const seq = ++listReqSeqRef.current
    const fetchData = async () => {
      try {
        const { items, total } = await getMedias(
          tab,
          {
            limit: PAGE_SIZE,
            sort: fileManagerState.sortBy,
            order: fileManagerState.sortOrder,
          },
          controller.signal
        )
        if (seq !== listReqSeqRef.current) {
          return // 过期响应，丢弃
        }
        setFileList({ tab, items, total })
      } catch (e) {
        // 主动 abort 不是错误，别弹 toast（拦截器会把它包成普通 Error）
        if (controller.signal.aborted) {
          return
        }
        toast({
          variant: "destructive",
          title: t("common.somethingWentWrong"),
          description: getErrorMessage(e),
        })
      }
    }
    fetchData()
    return () => controller.abort()
  }, [tab, fileManagerState.sortBy, fileManagerState.sortOrder, toast, t])

  const loadMore = useCallback(async () => {
    if (!open || loadingMoreRef.current) {
      return
    }
    if (fileList.tab !== tab || fileList.items.length >= fileList.total) {
      return
    }
    loadingMoreRef.current = true
    const seq = listReqSeqRef.current
    try {
      const { items, total } = await getMedias(tab, {
        limit: PAGE_SIZE,
        offset: fileList.items.length,
        sort: fileManagerState.sortBy,
        order: fileManagerState.sortOrder,
      })
      if (seq !== listReqSeqRef.current) {
        return // 列表已经按新的 tab/排序重建
      }
      setFileList((prev) => {
        if (prev.tab !== tab) {
          return prev
        }
        // 两页请求可能并发返回同一批文件，按名字去重
        const known = new Set(prev.items.map((item) => item.name))
        return {
          tab,
          items: [...prev.items, ...items.filter((item) => !known.has(item.name))],
          total,
        }
      })
    } catch {
      // 下一页失败就停在这里，继续滚动会再试
    } finally {
      loadingMoreRef.current = false
    }
  }, [
    open,
    tab,
    fileList,
    fileManagerState.sortBy,
    fileManagerState.sortOrder,
  ])

  loadMoreRef.current = loadMore

  useEffect(() => {
    // 搜索要覆盖整份名单，不能在已加载的 100 条里找；输入搜索词时补拉一次全量
    if (!open || !debouncedSearchText || allItems.tab === tab) {
      return
    }
    const controller = new AbortController()
    getMedias(tab, { limit: 0 }, controller.signal)
      .then(({ items }) => setAllItems({ tab, items }))
      .catch((e) => {
        if (!controller.signal.aborted) {
          toast({
            variant: "destructive",
            title: t("common.somethingWentWrong"),
            description: getErrorMessage(e),
          })
        }
      })
    return () => controller.abort()
  }, [open, debouncedSearchText, tab, allItems, toast, t])

  useEffect(() => {
    if (!open) {
      return
    }
    const fetchData = async () => {
      try {
        // 上一个 tab 的名单还没到（或已作废）时按空列表处理，不拿旧 tab 的
        // 文件名去拼新 tab 的缩略图 URL
        let filteredFilenames = debouncedSearchText
          ? allItems.tab === tab
            ? allItems.items
            : []
          : fileList.tab === tab
            ? fileList.items
            : []
        if (debouncedSearchText) {
          const fuse = new Fuse(filteredFilenames, {
            keys: ["name"],
          })
          const items = fuse.search(debouncedSearchText)
          filteredFilenames = items.map(
            (item) => filteredFilenames[item.refIndex]
          )
        }

        filteredFilenames = orderBy(
          filteredFilenames,
          fileManagerState.sortBy,
          fileManagerState.sortOrder
        )

        const newPhotos = filteredFilenames.map((filename: Filename) => {
          const width = photoWidth
          const height = filename.height * (width / filename.width)
          const src = `${API_ENDPOINT}/media_thumbnail_file?tab=${tab}&filename=${encodeURIComponent(
            filename.name
          )}&width=${Math.ceil(width)}&height=${Math.ceil(height)}`
          return { src, height, width, name: filename.name }
        })
        setPhotos(newPhotos)
      } catch (e) {
        toast({
          variant: "destructive",
          title: t("common.somethingWentWrong"),
          description: getErrorMessage(e),
        })
      }
    }
    fetchData()
  }, [fileList, allItems, debouncedSearchText, fileManagerState, photoWidth, open, tab, toast, t])

  // scroll 事件不冒泡，ScrollArea 根节点上的 onScroll 收不到内部 viewport 的滚动，
  // 所以直接监听 viewport：记录位置（关闭时恢复）并在接近底部时拉下一页
  useEffect(() => {
    const root = scrollRootRef.current
    if (!open || !root) {
      return
    }
    const viewport = root.querySelector<HTMLElement>(
      "[data-radix-scroll-area-viewport]"
    )
    if (!viewport) {
      return
    }
    const onScroll = () => {
      setScrollTop(viewport.scrollTop)
      if (
        viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight <
        800
      ) {
        void loadMoreRef.current()
      }
    }
    viewport.addEventListener("scroll", onScroll, { passive: true })
    let frame = 0
    if (pendingRestoreRef.current > 0) {
      const target = pendingRestoreRef.current
      frame = requestAnimationFrame(() => {
        viewport.scrollTo({ top: target })
        if (Math.abs(viewport.scrollTop - target) < 2) {
          pendingRestoreRef.current = 0
        }
      })
    }
    return () => {
      cancelAnimationFrame(frame)
      viewport.removeEventListener("scroll", onScroll)
    }
  }, [open, photos])

  // 搜索框 focus 会把 disableShortCuts 置 true，但关闭对话框时输入框直接卸载，
  // 收不到 blur，全局快捷键会一直停在禁用状态。关闭时统一兜底复位。
  const onOpenChange = (nextOpen: boolean) => {
    toggleOpen(nextOpen)
    if (!nextOpen) {
      updateAppState({ disableShortCuts: false })
    }
  }

  const onClick = ({ index }: { index: number }) => {
    onOpenChange(false)
    setPhotoIndex(index)
    onPhotoClick(tab, photos[index].name)
  }

  const renderTitle = () => {
    const count = debouncedSearchText
      ? photos.length
      : fileList.tab === tab
        ? fileList.total
        : 0
    return (
      <div className="flex justify-start items-center gap-[12px]">
        <div>{t("fileManager.imagesCount", { count })}</div>
        <div className="flex">
          <IconButton
            tooltip={t("fileManager.rowsLayout")}
            onClick={() => {
              updateFileManagerState({ layout: "rows" })
            }}
          >
            <ViewHorizontalIcon
              className={fileManagerState.layout !== "rows" ? "opacity-50" : ""}
            />
          </IconButton>
          <IconButton
            tooltip={t("fileManager.gridLayout")}
            onClick={() => {
              updateFileManagerState({ layout: "masonry" })
            }}
          >
            <ViewGridIcon
              className={
                fileManagerState.layout !== "masonry" ? "opacity-50" : ""
              }
            />
          </IconButton>
        </div>
      </div>
    )
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <IconButton tooltip={t("fileManager.fileManager")}>
          <FolderClosed />
        </IconButton>
      </DialogTrigger>
      <DialogContent className="h-4/5 max-w-6xl">
        <DialogTitle>{renderTitle()}</DialogTitle>
        <div className="flex justify-between gap-8 items-center">
          <div className="flex relative justify-start items-center">
            <MagnifyingGlassIcon className="absolute left-[8px]" />
            <Input
              ref={ref}
              value={fileManagerState.searchText}
              className="w-[250px] pl-[30px]"
              tabIndex={-1}
              onInput={(evt: FormEvent<HTMLInputElement>) => {
                evt.preventDefault()
                evt.stopPropagation()
                const target = evt.target as HTMLInputElement
                updateFileManagerState({ searchText: target.value })
              }}
              placeholder={t("fileManager.searchPlaceholder")}
            />
          </div>

          <Tabs defaultValue={tab} onValueChange={(val) => setTab(val)}>
            <TabsList aria-label="Manage your account">
              <TabsTrigger value={IMAGE_TAB}>
                {t("fileManager.imageDirectory")}
              </TabsTrigger>
              <TabsTrigger value={OUTPUT_TAB}>
                {t("fileManager.outputDirectory")}
              </TabsTrigger>
              <TabsTrigger value={MASK_TAB}>
                {t("fileManager.maskDirectory")}
              </TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="flex gap-2">
            <div className="flex gap-1">
              <Select
                value={fileManagerState.sortBy}
                onValueChange={(val) => {
                  updateFileManagerState({ sortBy: val as SortBy })
                }}
              >
                <SelectTrigger className="w-[140px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.values(SortBy).map((sortBy) => {
                    return (
                      <SelectItem value={sortBy} key={sortBy}>
                        {t(SORT_BY_LABEL_KEYS[sortBy])}
                      </SelectItem>
                    )
                  })}
                </SelectContent>
              </Select>

              {fileManagerState.sortOrder === SortOrder.DESCENDING ? (
                <IconButton
                  tooltip={t("fileManager.descendingOrder")}
                  onClick={() => {
                    updateFileManagerState({ sortOrder: SortOrder.ASCENDING })
                  }}
                >
                  <BarsArrowDownIcon />
                </IconButton>
              ) : (
                <IconButton
                  tooltip={t("fileManager.ascendingOrder")}
                  onClick={() => {
                    updateFileManagerState({ sortOrder: SortOrder.DESCENDING })
                  }}
                >
                  <BarsArrowUpIcon />
                </IconButton>
              )}
            </div>
          </div>
        </div>

        <ScrollArea className="w-full h-full rounded-md" ref={onRefChange}>
          <PhotoAlbum
            layout={fileManagerState.layout}
            render={{
              image: (props, { width, height }) => (
                <LazyPhotoImage
                  {...props}
                  renderedWidth={width}
                  renderedHeight={height}
                />
              ),
            }}
            photos={photos}
            spacing={12}
            padding={0}
            onClick={onClick}
          />
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}
