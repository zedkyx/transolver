#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${TRANSOLVER_CONDA_ENV:-transolver}"
PYTHON_RUNNER=(python)
if command -v conda >/dev/null 2>&1; then
  PYTHON_RUNNER=(conda run -n "${CONDA_ENV}" python)
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}}"
export PYTHONDONTWRITEBYTECODE=1

DATA_PATH="${ROOT_DIR}/data/transolver_2d_irregular_n128.pt"
RUN_DIR="${ROOT_DIR}/runs/synth_2d_n128_q"
SAVE_NAME="synth_2d_n128_q"
BEST_CKPT="${RUN_DIR}/checkpoints/${SAVE_NAME}_best_eval.pt"
CONFIG_PATH="${ROOT_DIR}/tests/configs/train_eval_2d_irregular.yaml"

mkdir -p "${RUN_DIR}"
cd "${ROOT_DIR}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "missing data: ${DATA_PATH}" >&2
  echo "generate it with: ${PYTHON_RUNNER[*]} tests/make_transolver_2d_irregular_pt.py" >&2
  exit 1
fi

echo "[1/2] train 2D irregular synthetic cache with quadrature weights"
"${PYTHON_RUNNER[@]}" -m scripts.transolver.public.train_dp \
  --config "${CONFIG_PATH}"

echo "[2/2] eval best checkpoint"
"${PYTHON_RUNNER[@]}" -m scripts.transolver.public.evaluate \
  --config "${CONFIG_PATH}" \
  --load_ckpt "${BEST_CKPT}"

echo "done: ${RUN_DIR}"
