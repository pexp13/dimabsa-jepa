#!/usr/bin/env bash
# Скачивает официальный репозиторий DimABSA2026 (данные и скрипт оценки) в data/.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d data/DimABSA2026/.git ]; then
  git -C data/DimABSA2026 pull --ff-only
else
  mkdir -p data
  git clone --depth 1 https://github.com/DimABSA/DimABSA2026 data/DimABSA2026
fi
