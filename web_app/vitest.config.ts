import path from "path"
import { defineConfig } from "vitest/config"

// 与 vite.config.ts 保持一致的 alias；环境用 jsdom（utils 里有 document/navigator 引用）。
export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts"],
    // 前端单测只跑纯逻辑，不碰网络/后端；coverage 报告目录已被 .gitignore 忽略
    coverage: {
      provider: "v8",
      include: ["src/lib/**"],
      // api.ts/states.ts 依赖 axios/zustand 运行时时序，交给 playwright webui 覆盖
      exclude: ["src/lib/api.ts", "src/lib/states.ts", "src/lib/types.ts"],
    },
  },
})
