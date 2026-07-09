# MUSE Pipeline Reproduction

This document describes how to reproduce the MUSE pipeline without relying on local-only state.

## 1. Environment

Create a Python environment and install the repository dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Important environment variables:

| Variable | Purpose |
|---|---|
| `MOCK_MODE` | `1` uses deterministic mock stages for smoke tests; `0` calls real models. |
| `BASE_URL`, `API_KEY`, `MODEL` | OpenAI-compatible endpoint used as the default model backend. |
| `VISION_*` | Optional visual-stage endpoint override. |
| `REASONING_*` | Optional reasoning-stage endpoint override. |
| `ORCHESTRATOR_*` | Optional verifier/reflection/synthesis endpoint override. |
| `MODEL_PROTOCOL` | Request protocol; default is `OPENAI_STYLE`. |
| `SAVE_GENERATED_SKILLS` | Set `0` to disable writing new generated skills. |
| `MAX_RECHECKS` | Number of visual recheck rounds after verifier feedback. |
| `MUSE_ENV_FILE` | Optional path to an alternate env file. |
| `MUSE_GENERATED_SKILLS_ROOT` | Optional path for an isolated generated-skill library. |

Use the `MUSE_*` names for public reproduction scripts; older local aliases are not part of the public reproduction contract.

## 2. Task Format

Local JSONL records should contain fields compatible with `TaskPacket`:

```json
{"pid":"demo_001","question":"What is 2 + 3?","image_path":"","choices":["4","5","6"],"question_type":"multi_choice","answer_type":"text","answer":"5","metadata":{"context":"text only","task":"arithmetic smoke"}}
```

Useful fields:

- `pid` or `sample_id`: stable sample identifier.
- `question`: question text.
- `image_path` or `image_paths`: local image path(s); empty is allowed for text-only tests.
- `choices`: optional multiple-choice candidates.
- `question_type`: usually `multi_choice` or `free_form`.
- `answer_type`: `text`, `integer`, or `float`.
- `precision`, `unit`: optional numeric normalization metadata.
- `answer`: optional gold answer used only for evaluation and skill-save gating.
- `metadata`: dataset/task descriptors used by skill retrieval.

## 3. Smoke Test

Run this first after cloning:

```bash
MOCK_MODE=1 python run_mathvista.py \
  --question-file data/demo/muse_smoke.jsonl \
  --limit 2 \
  --disable-save \
  --output /tmp/muse_smoke_predictions.jsonl

python eval_results.py --predictions /tmp/muse_smoke_predictions.jsonl
```

The smoke test should print `Accuracy: 2/2 = 1.000`.

A minimal compare-workflow smoke test can also run on the local demo file. Pass an empty HF split so the runner does not load MathVista from Hugging Face:

```bash
MOCK_MODE=1 SAVE_GENERATED_SKILLS=0 python run_compare_mathvista_parallel.py \
  --question-file data/demo/muse_smoke.jsonl \
  --hf-split "" \
  --seed-count 1 \
  --eval-count 1 \
  --workers 1 \
  --output-dir /tmp/muse_compare_smoke
```

## 4. Single-Branch Runs

Use `run_mathvista.py` for quick debugging on either local JSONL or Hugging Face MathVista splits.

Local JSONL:

```bash
MOCK_MODE=0 python run_mathvista.py \
  --question-file /path/to/tasks.jsonl \
  --image-root /path/to/images \
  --limit 20 \
  --output results/local_debug_predictions.jsonl
```

Hugging Face MathVista:

```bash
MOCK_MODE=0 python run_mathvista.py \
  --hf-split testmini \
  --limit 20 \
  --output results/mathvista_testmini_debug20.jsonl
```

## 5. Main Seed/Eval Workflow

`run_compare_mathvista_parallel.py` runs the main comparison workflow:

- `baseline_model`: direct multimodal baseline.
- `seed_install`: first `seed_count` tasks can save reusable skills.
- `eval_seeded_no_reuse`: evaluates without reusing generated skills.
- `eval_with_evolution`: evaluates with retrieval, verification, and reuse of saved skills.

```bash
MOCK_MODE=0 python run_compare_mathvista_parallel.py \
  --hf-split testmini \
  --seed-count 20 \
  --eval-count 980 \
  --workers 10 \
  --output-dir results/mathvista_muse_s20_e980 \
  --rewrite-readable
```

Outputs in the run directory include:

- `summary.json`: aggregate accuracy and reuse statistics.
- `baseline_model.jsonl`: direct baseline predictions.
- `seed_install.jsonl`: seed-stage predictions and save decisions.
- `eval_seeded_no_reuse.jsonl`: no-reuse eval predictions.
- `eval_with_evolution.jsonl`: final MUSE predictions.
- `generated_after_seed/`: skill library saved after seed tasks.
- `reasoning_details/`: optional readable per-sample traces.

## 6. Isolating Generated Skills

Use a separate generated-skill root when running multiple experiments:

```bash
MUSE_GENERATED_SKILLS_ROOT=results/run_a/generated_skills \
MOCK_MODE=0 python run_compare_mathvista_parallel.py \
  --hf-split testmini \
  --seed-count 20 \
  --eval-count 980 \
  --workers 10 \
  --output-dir results/run_a
```

This prevents one experiment's generated library from contaminating another run.

## 7. Optional Experiment Entrypoints

- `run_multidataset_matrix.py`: M3CoT, MathVerse, and MMMU-Pro matrix experiments.
- `experiments/baseline_variants/`: direct-model baseline variants used by multi-dataset experiments.
- `baseline_ablation/`: signal-source ablations over a fixed seed candidate pool.
- `continual_learning/`: phase-based continual-learning probes.
- `open_model_eval/`: vLLM/OpenAI-compatible local model launch and evaluation helpers.

These scripts write outputs under `results/`, which is intentionally ignored by git.
