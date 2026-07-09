# MUSE

MUSE is a multimodal reasoning pipeline for MathVista-style tasks. It solves an image-question pair through staged visual grounding, mathematical reasoning, verification, answer normalization, and reusable skill synthesis.

This repository currently publishes the reproducible MUSE pipeline. Paper result tables and curated raw artifacts will be added after the final result set is selected.

## Pipeline

For each task, MUSE runs:

1. `visual_detail_agent`: extracts grounded visual facts from the image.
2. `math_reason_agent`: reasons from the task and structured evidence.
3. `multimodal_verifier`: checks whether the candidate answer is grounded and canonical.
4. `answer_normalizer`: converts the final answer into the expected format.
5. Skill synthesis: successful seed traces can be distilled into reusable narrow skills under the generated-skill root (`MUSE_GENERATED_SKILLS_ROOT`, default `skills/subagents/generated/`, created at runtime).
6. Skill reuse: later tasks retrieve candidate skills first, verify their outputs, and fall back to the base pipeline when reuse is unsafe.

See [docs/PIPELINE.md](docs/PIPELINE.md) for the full reproduction workflow.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The default `.env.example` enables `MOCK_MODE=1`, which is only for installation and pipeline smoke tests. For real model calls, set `MOCK_MODE=0` and fill in an OpenAI-compatible endpoint:

```bash
BASE_URL=https://api.openai.com/v1
API_KEY=your_api_key
MODEL=your_model_name
MOCK_MODE=0
```

You can also configure per-stage models with `VISION_*`, `REASONING_*`, and `ORCHESTRATOR_*` variables.

## Smoke Test

This runs without API calls and verifies imports, task loading, trace writing, answer normalization, and evaluation.

```bash
MOCK_MODE=1 python run_mathvista.py \
  --question-file data/demo/muse_smoke.jsonl \
  --limit 2 \
  --disable-save \
  --output /tmp/muse_smoke_predictions.jsonl

python eval_results.py --predictions /tmp/muse_smoke_predictions.jsonl
```

Expected smoke-test accuracy is `2/2`; this uses gold answers only because `MOCK_MODE=1` is a wiring test, not an evaluation result.

## Run MathVista

A small real-model debug run:

```bash
MOCK_MODE=0 python run_mathvista.py \
  --hf-split testmini \
  --limit 20 \
  --experiment-tag debug20 \
  --output results/mathvista_debug20_predictions.jsonl

python eval_results.py --predictions results/mathvista_debug20_predictions.jsonl
```

A seed-plus-eval comparison run matching the main MUSE workflow shape:

```bash
MOCK_MODE=0 python run_compare_mathvista_parallel.py \
  --hf-split testmini \
  --seed-count 20 \
  --eval-count 980 \
  --workers 10 \
  --output-dir results/mathvista_muse_s20_e980
```

`results/`, `workspace/`, and `trajectory/` are intentionally ignored by git. They are local run outputs and can be archived or published separately.

## Repository Map

- `muse/`: public Python package entry point.
- `run_mathvista.py`: single-branch MUSE runner for local JSONL or Hugging Face MathVista splits.
- `run_compare_mathvista_parallel.py`: seed, baseline, seeded-no-reuse, and evolved comparison runner.
- `run_multidataset_matrix.py`: optional multi-dataset experiment runner.
- `experiments/baseline_variants/`: direct-model baseline variants used by multi-dataset experiments.
- `baseline_ablation/`: signal-source ablation scripts.
- `continual_learning/`: continual-learning smoke/formal experiment scripts.
- `open_model_eval/`: helpers for local/open VLM serving and evaluation.
- Generated skills: written at runtime to `MUSE_GENERATED_SKILLS_ROOT` (default `skills/subagents/generated/`).
- `docs/PIPELINE.md`: detailed reproducibility notes.

## Notes

- Do not commit `.env`, API keys, local caches, or raw `results/` directories.
- Use `MUSE_ENV_FILE=/path/to/env` when running with a non-default environment file.
- Use `MUSE_GENERATED_SKILLS_ROOT=/path/to/generated_skills` to isolate generated skill libraries between experiments.

## Citation

Citation information will be added with the camera-ready paper metadata.
