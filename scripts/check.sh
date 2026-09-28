#!/usr/bin/env bash
# 一次性检查整棵树是否健康：前端 lint/build → 装进 python 包 → 后端快测（含 webui）。
# 和 .github/workflows/ci.yml 跑的是同一套命令；改了一边记得改另一边。
#
# 前端放在 pytest 之前构建，这样 test_webui（真浏览器打真 UI）能真的跑到。
# 只想要「不起 Node 的后端快测」就直接跑： pytest -m "not slow" -q
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"

echo "=== frontend: npm run lint ==="
(cd web_app && npm run lint)

echo ""
echo "=== frontend: npm run build ==="
(cd web_app && npm run build)

echo ""
echo "=== 装进 python 包（iopaint/web_app 是 gitignore 的构建产物） ==="
rm -rf iopaint/web_app && cp -r web_app/dist iopaint/web_app

echo ""
echo "=== backend: pytest -m 'not slow' ==="
# test_webui 需要 playwright，没装就自己 skip；端口走 conftest 的 bind(("",0))，不碰 8080。
"${PYTHON}" -m pytest -m "not slow" -q

echo ""
echo "OK. 启动： iopaint start --model lama --port 8080"
