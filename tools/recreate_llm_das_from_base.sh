#!/usr/bin/env bash
set -euo pipefail

BASE_ENV="${1:-base}"
TARGET_ENV="${2:-llm_das}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"
  exit 2
fi

echo "[1/4] Removing target env if exists: ${TARGET_ENV}"
conda env remove -n "${TARGET_ENV}" -y >/dev/null 2>&1 || true

echo "[2/4] Cloning ${BASE_ENV} -> ${TARGET_ENV}"
conda create -n "${TARGET_ENV}" --clone "${BASE_ENV}" -y

echo "[3/4] Checking version parity"
python tools/check_env_parity.py --base_env "${BASE_ENV}" --target_env "${TARGET_ENV}"

echo "[4/4] Done. Activate with: conda activate ${TARGET_ENV}"
