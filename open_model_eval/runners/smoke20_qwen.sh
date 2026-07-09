#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

bash open_model_eval/runners/run_with_env.sh \
  open_model_eval/envs/qwen25vl7b.env \
  --model-alias qwen25vl7b \
  --hf-split testmini \
  --seed-count 10 \
  --eval-count 20 \
  --offset 0 \
  --workers 10 \
  --output-dir results/open_model_qwen25vl7b_smoke20
