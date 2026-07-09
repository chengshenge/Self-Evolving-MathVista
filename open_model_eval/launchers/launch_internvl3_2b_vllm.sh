#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_BIN="${VLLM_BIN:-$REPO_ROOT/open_model_eval/.venv-vllm/bin/vllm}"

MODEL="OpenGVLab/InternVL3-2B"
PORT="${PORT:-8002}"
DTYPE="${DTYPE:-float16}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.70}"

exec "$VLLM_BIN" serve "$MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt '{"image":1}'
