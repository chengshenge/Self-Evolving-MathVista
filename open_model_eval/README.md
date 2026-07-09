# Open-Model Evaluation Pipeline

This folder is designed to live under your repo root:

```
~/merge/MUSE/open_model_eval/
```

It is a small MUSE extension for local OpenAI-compatible VLM servers. It adds:

- model-specific `.env` files
- launcher scripts for local OpenAI-compatible servers (vLLM)
- runner scripts for baseline/evolution evaluation on MathVista testmini

## Recommended models

1. `Qwen/Qwen2.5-VL-7B-Instruct`
2. `OpenGVLab/InternVL3-9B`

## Folder layout

- `envs/`
  - `qwen25vl7b.env`
  - `internvl3_9b.env`
- `launchers/`
  - `launch_qwen25vl7b_vllm.sh`
  - `launch_internvl3_9b_vllm.sh`
- `runners/`
  - `run_open_model_baseline_evolution.py`
  - `run_with_env.sh`
  - `smoke20_qwen.sh`
  - `smoke20_internvl.sh`

## Typical workflow

### 1) Start one local server

Example for Qwen2.5-VL-7B:

```bash
cd ~/merge/MUSE
bash open_model_eval/launchers/launch_qwen25vl7b_vllm.sh
```

Example for InternVL3-9B:

```bash
cd ~/merge/MUSE
bash open_model_eval/launchers/launch_internvl3_9b_vllm.sh
```

### 2) Smoke test (baseline + evolution only)

```bash
cd ~/merge/MUSE
bash open_model_eval/runners/smoke20_qwen.sh
```

or

```bash
cd ~/merge/MUSE
bash open_model_eval/runners/smoke20_internvl.sh
```

### 3) Full testmini run (20 seed + 980 eval)

```bash
cd ~/merge/MUSE
bash open_model_eval/runners/run_with_env.sh \
  open_model_eval/envs/qwen25vl7b.env \
  --hf-split testmini \
  --seed-count 20 \
  --eval-count 980 \
  --offset 0 \
  --output-dir results/open_model_qwen25vl7b_testmini
```

Swap the env file for InternVL3-9B if needed.

## Notes

- These runners intentionally evaluate only `baseline_model` and `eval_with_evolution` to save time.
- `seed_install` still runs internally because evolution needs a seeded skill library.
- `eval_with_evolution` is executed in the same way as your current pipeline: reuse is on, save is off.
- Results are written in the same JSONL style as your existing compare outputs so you can analyze them easily.
