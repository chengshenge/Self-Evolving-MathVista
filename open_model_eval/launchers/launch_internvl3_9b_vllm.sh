#!/usr/bin/env bash
set -euo pipefail

MODEL="OpenGVLab/InternVL3-9B"
PORT="${PORT:-8002}"
DTYPE="${DTYPE:-bfloat16}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.92}"

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --trust-remote-code \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt '{"image":1}'
