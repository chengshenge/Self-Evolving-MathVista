# MUSE

MUSE 是一个面向 MathVista 风格图文数学推理任务的多模态 pipeline。它把一道题拆成视觉证据抽取、数学推理、跨模态验证、答案规范化，并把成功 seed 轨迹蒸馏成可复用的窄技能。

这个仓库当前先发布可复现的 MUSE pipeline。论文最终结果表和整理后的结果文件，等你确认后再补进来。

## Pipeline 概览

每道题会经过这些阶段：

1. `visual_detail_agent`：从图片中抽取有依据的视觉事实。
2. `math_reason_agent`：基于题目和结构化视觉证据进行推理。
3. `multimodal_verifier`：检查候选答案是否被证据支撑、格式是否合理。
4. `answer_normalizer`：把最终答案转成题目要求的标准格式。
5. 技能蒸馏：seed 阶段做对的轨迹会被写到生成技能目录（`MUSE_GENERATED_SKILLS_ROOT`，默认运行时创建 `skills/subagents/generated/`）。
6. 技能复用：后续题目会先检索候选技能，经过验证后再决定是否复用；不可靠时回退到基础 pipeline。

更完整的复现说明见 [docs/PIPELINE.md](docs/PIPELINE.md)。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env.example` 默认开启 `MOCK_MODE=1`，只用于无 API 的 smoke test。真实模型实验请改成：

```bash
BASE_URL=https://api.openai.com/v1
API_KEY=your_api_key
MODEL=your_model_name
MOCK_MODE=0
```

如果不同阶段要用不同模型，可以配置 `VISION_*`、`REASONING_*`、`ORCHESTRATOR_*`。

## 无 API Smoke Test

```bash
MOCK_MODE=1 python run_mathvista.py \
  --question-file data/demo/muse_smoke.jsonl \
  --limit 2 \
  --disable-save \
  --output /tmp/muse_smoke_predictions.jsonl

python eval_results.py --predictions /tmp/muse_smoke_predictions.jsonl
```

预期输出是 `2/2`。这是连通性测试，不是论文实验结果。

## 跑 MathVista

小规模真实模型调试：

```bash
MOCK_MODE=0 python run_mathvista.py \
  --hf-split testmini \
  --limit 20 \
  --experiment-tag debug20 \
  --output results/mathvista_debug20_predictions.jsonl

python eval_results.py --predictions results/mathvista_debug20_predictions.jsonl
```

主流程形态的 seed/eval 对比：

```bash
MOCK_MODE=0 python run_compare_mathvista_parallel.py \
  --hf-split testmini \
  --seed-count 20 \
  --eval-count 980 \
  --workers 10 \
  --output-dir results/mathvista_muse_s20_e980
```

`results/`、`workspace/`、`trajectory/` 默认不进 git，后续可以单独整理成论文结果包。

## 目录说明

- `muse/`：公开 Python 包入口。
- `run_mathvista.py`：单分支 MUSE 运行入口。
- `run_compare_mathvista_parallel.py`：seed、baseline、seeded-no-reuse、evolved 对比入口。
- `run_multidataset_matrix.py`：多数据集实验入口。
- `experiments/baseline_variants/`：多数据集实验使用的直接模型 baseline 变体。
- `baseline_ablation/`：信号来源消融脚本。
- `continual_learning/`：持续学习实验脚本。
- `open_model_eval/`：本地/open VLM 服务与评测辅助脚本。
- 生成技能：运行时写入 `MUSE_GENERATED_SKILLS_ROOT`（默认 `skills/subagents/generated/`）。
- `docs/PIPELINE.md`：详细复现说明。

## 注意

- 不要提交 `.env`、API key、本地缓存和原始 `results/`。
- 用 `MUSE_ENV_FILE=/path/to/env` 指定非默认环境文件。
- 用 `MUSE_GENERATED_SKILLS_ROOT=/path/to/generated_skills` 隔离不同实验的生成技能库。
