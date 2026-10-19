// i18n 基建单测（vitest + jsdom）：资源自动注册、默认语言回退、语言切换、语言包 key 对齐。
import { afterEach, beforeAll, describe, expect, it } from "vitest"
import i18n, {
  DEFAULT_LOCALE,
  SUPPORTED_LOCALES,
  localeDisplayName,
} from "./i18n"
import en from "../locales/en.json"

// 遍历 src/locales/*.json：新增一门语言会自动进入下面的对齐检查。
const catalogs = import.meta.glob("../locales/*.json", {
  eager: true,
  import: "default",
}) as Record<string, Record<string, unknown>>

function localeFromPath(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1, -".json".length)
}

// 把嵌套 JSON 展平成点分 key 集合，用于比对不同语言包的 key 是否一一对应。
function flatten(value: Record<string, unknown>, prefix = ""): string[] {
  return Object.entries(value).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key
    if (child && typeof child === "object" && !Array.isArray(child)) {
      return flatten(child as Record<string, unknown>, path)
    }
    return [path]
  })
}

describe("i18n catalogs", () => {
  it("registers every locale file in src/locales", () => {
    expect(SUPPORTED_LOCALES).toContain(DEFAULT_LOCALE)
    expect(SUPPORTED_LOCALES).toContain("zh-CN")
    expect(SUPPORTED_LOCALES).toEqual(
      Object.keys(catalogs)
        .map(localeFromPath)
        .sort()
    )
  })

  it("keeps every locale's keys aligned with the canonical en catalog", () => {
    const enKeys = flatten(en).sort()
    for (const [path, catalog] of Object.entries(catalogs)) {
      const locale = localeFromPath(path)
      if (locale === DEFAULT_LOCALE) {
        continue
      }
      expect(flatten(catalog).sort(), locale).toEqual(enKeys)
    }
  })
})

describe("i18n runtime", () => {
  beforeAll(async () => {
    await i18n.changeLanguage(DEFAULT_LOCALE)
  })

  afterEach(async () => {
    await i18n.changeLanguage(DEFAULT_LOCALE)
  })

  it("falls back to English for unsupported browser languages", () => {
    expect(i18n.resolvedLanguage).toBe(DEFAULT_LOCALE)
    expect(i18n.t("shortcuts.tooltip")).toBe("Hotkeys")
  })

  it("translates after switching locale and syncs <html lang>", async () => {
    await i18n.changeLanguage("zh-CN")
    expect(i18n.t("shortcuts.tooltip")).toBe("快捷键")
    expect(document.documentElement.lang).toBe("zh-CN")
  })

  it("shows a locale's native display name", () => {
    expect(localeDisplayName("en")).toBe("English")
    expect(localeDisplayName("zh-CN")).toContain("中文")
  })
})
