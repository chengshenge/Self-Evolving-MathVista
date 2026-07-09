#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <env-file> [args passed to run_open_model_baseline_evolution.py]" >&2
  exit 1
fi

ENV_FILE="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

sed -i 's/\r$//' "$ENV_FILE"
set -a
source "$ENV_FILE"
set +a

export VISION_BASE_URL="${VISION_BASE_URL:-$BASE_URL}"
export VISION_API_KEY="${VISION_API_KEY:-${API_KEY:-$OPENAI_API_KEY}}"
export VISION_MODEL="${VISION_MODEL:-$MODEL}"
export REASONING_BASE_URL="${REASONING_BASE_URL:-$BASE_URL}"
export REASONING_API_KEY="${REASONING_API_KEY:-${API_KEY:-$OPENAI_API_KEY}}"
export REASONING_MODEL="${REASONING_MODEL:-$MODEL}"
export ORCHESTRATOR_BASE_URL="${ORCHESTRATOR_BASE_URL:-$BASE_URL}"
export ORCHESTRATOR_API_KEY="${ORCHESTRATOR_API_KEY:-${API_KEY:-$OPENAI_API_KEY}}"
export ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-$MODEL}"
export OPEN_MODEL_MAX_TOKENS="${OPEN_MODEL_MAX_TOKENS:-128}"

python open_model_eval/runners/run_open_model_baseline_evolution.py "$@"
