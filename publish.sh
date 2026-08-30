#!/usr/bin/env bash
# Build and publish IOPaint to PyPI.
#
# Prerequisites:
#   - Node.js and npm (for frontend build)
#   - Python build module:  pip install build
#
# Usage:
#   bash publish.sh          # build sdist + wheel
#   bash publish.sh upload   # build and upload to PyPI
set -euo pipefail

# ── 1. Build frontend ──────────────────────────────────────────────────
pushd ./web_app >/dev/null
rm -rf dist
npm run build
popd >/dev/null

# ── 2. Copy frontend dist into the Python package ─────────────────────
rm -rf ./iopaint/web_app
cp -r web_app/dist ./iopaint/web_app

# ── 3. Clean old build artefacts ──────────────────────────────────────
rm -rf dist build iopaint.egg-info

# ── 4. Build sdist + wheel using the modern PEP 517 build frontend ───
python3 -m build

if [ "${1:-}" = "upload" ]; then
  echo "Uploading to PyPI …"
  python3 -m twine upload dist/*
fi

echo ""
echo "Build complete.  Artefacts are in ./dist/"
ls -lh dist/
