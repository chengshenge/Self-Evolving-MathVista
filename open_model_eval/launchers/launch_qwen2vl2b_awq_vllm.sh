#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_BIN="${VLLM_BIN:-$REPO_ROOT/open_model_eval/.venv-vllm/bin/vllm}"

MODEL="Qwen/Qwen2-VL-2B-Instruct-AWQ"
PORT="${PORT:-8001}"
MAX_LEN="${MAX_LEN:-1024}"
GPU_UTIL="${GPU_UTIL:-0.98}"

exec "$VLLM_BIN" serve "$MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --dtype auto \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt '{"image":1}'
