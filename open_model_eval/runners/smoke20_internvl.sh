#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

bash open_model_eval/runners/run_with_env.sh \
  open_model_eval/envs/internvl3_9b.env \
  --model-alias internvl3_9b \
  --hf-split testmini \
  --seed-count 10 \
  --eval-count 20 \
  --offset 0 \
  --workers 10 \
  --output-dir results/open_model_internvl3_9b_smoke20
