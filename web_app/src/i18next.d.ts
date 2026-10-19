import type en from "./locales/en.json"

// 让 `t("shortcuts.undo")` 这类调用拿到 key 的编译期检查：
// `en.json` 是规范源，拼接错的 key 会在 `tsc` 阶段报错。
declare module "i18next" {
  interface CustomTypeOptions {
    defaultNS: "translation"
    resources: {
      translation: typeof en
    }
  }
}
