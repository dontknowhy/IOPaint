#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# 真正的 env 文件在仓库根目录；scripts/environment.yaml 是过期的 lama-cleaner 残留
ENV_FILE="${REPO_ROOT}/environment.yml"
ENV_NAME="iopaint"

echo "=== IOPaint Environment Setup ==="

# Check if conda is available
if ! command -v conda &>/dev/null; then
    echo "Error: conda is not installed."
    echo "Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
    echo "Error: ${ENV_FILE} not found."
    exit 1
fi

# Create or update conda environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Updating..."
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
    echo "Creating environment '${ENV_NAME}'..."
    conda env create -f "${ENV_FILE}"
fi

echo ""
echo "=== Environment ready ==="
echo "Activate with:  conda activate ${ENV_NAME}"
echo "Start IOPaint: iopaint start --model lama"
