import i18n from "i18next"
import LanguageDetector from "i18next-browser-languagedetector"
import { initReactI18next } from "react-i18next"

// 可扩展点：往 `src/locales/` 放一个 `<locale>.json`（BCP 47，例如 `ja.json`）就会自动
// 注册成一门语言，并出现在 Settings 的语言下拉里，无需改动这个文件。
const catalogs = import.meta.glob("../locales/*.json", {
  eager: true,
  import: "default",
}) as Record<string, Record<string, unknown>>

function localeFromPath(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1, -".json".length)
}

const resources: Record<string, { translation: Record<string, unknown> }> = {}
for (const [path, translation] of Object.entries(catalogs)) {
  resources[localeFromPath(path)] = { translation }
}

export const SUPPORTED_LOCALES = Object.keys(resources).sort()
export const DEFAULT_LOCALE = "en"
export const LOCALE_STORAGE_KEY = "IOPAINT_LOCALE"

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources,
    supportedLngs: SUPPORTED_LOCALES,
    fallbackLng: DEFAULT_LOCALE,
    // 所有 zh 变体（zh、zh-Hans、zh-Hant…）都归到已提供的 zh-CN；其余语言回退英文。
    detection: {
      order: ["localStorage", "navigator"],
      caches: ["localStorage"],
      lookupLocalStorage: LOCALE_STORAGE_KEY,
      convertDetectedLanguage: (lng) =>
        lng.toLowerCase().startsWith("zh") ? "zh-CN" : lng,
    },
    interpolation: { escapeValue: false },
    returnNull: false,
  })

i18n.on("languageChanged", (lng) => {
  document.documentElement.lang = lng
})

// 语言下拉的显示名：用该语言自己的名字（中文用户看到「中文（中国）」），
// 这样新加一门语言也不必维护任何元数据文件。
export function localeDisplayName(locale: string): string {
  try {
    return (
      new Intl.DisplayNames([locale], { type: "language" }).of(locale) ?? locale
    )
  } catch {
    return locale
  }
}

export default i18n
