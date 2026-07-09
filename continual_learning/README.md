# Continual Learning / Forgetting Smoke Experiment

This directory implements a minimal executable 6.6 continual-learning smoke experiment on M3CoT.

## Phases

- Phase 1: Commonsense, `P1Train_CS`
- Phase 2: Science, `P2Train_SCI`
- Phase 3: Math, `P3Train_MATH`

Smoke defaults:

- 10 train samples per domain.
- 100 probe samples per domain.

Each phase learns only one domain. The three domains are not mixed during training.

## Checkpoints

- `ckpt0`: empty skill library / initial state.
- `ckpt1`: after Phase 1.
- `ckpt2`: after Phase 2.
- `ckpt3`: after Phase 3.

Every checkpoint is evaluated on all probes:

- `Probe_CS`
- `Probe_SCI`
- `Probe_MATH`

Probe evaluation is always `save=False`; probes never update the skill library.

## Systems

- `S0 = no_evolution`: sequentially sees all train phases, but `save=False` throughout. It is a continual control with no skill library growth.
- `S1 = frozen_after_p1`: Phase 1 can save skills. After Phase 1 the library is frozen; Phase 2 and Phase 3 can reuse P1 skills but cannot save new skills.
- `S2 = full_continual`: all three train phases can save new skills and reuse existing skills.

## Outputs

`build_phase_splits.py` writes:

- `P1Train_CS.jsonl`, `P2Train_SCI.jsonl`, `P3Train_MATH.jsonl`
- `Probe_CS.jsonl`, `Probe_SCI.jsonl`, `Probe_MATH.jsonl`
- `manifest.json`

`run_stream.py` writes, for each system:

- `library_ckpt_0_empty/`
- `library_ckpt_1_after_p1/`
- `library_ckpt_2_after_p2/`
- `library_ckpt_3_after_p3/`
- `eval/ckpt*/summary.json`
- `eval/ckpt*/probe_summary.csv`
- per-probe prediction JSONLs under each checkpoint eval directory
- `train/*.jsonl`
- `stream_summary.json`
- `skill_phase_map.json`

`eval_checkpoint.py` can evaluate any checkpoint library on all probes.

`report_continual.py` writes:

- `success_curve.csv`
- `forward_transfer.csv`
- `forgetting.csv`
- `skill_pollution.csv`
- `summary.md`

## Metrics

Supported in the minimal implementation:

- Success curve: `Acc(Probe_* | ckpt_t)` for all systems and checkpoints.
- Forward transfer:
  - `FWT_CS_to_SCI = Acc(Probe_SCI|ckpt1) - Acc(Probe_SCI|ckpt0)`
  - `FWT_CS_to_MATH = Acc(Probe_MATH|ckpt1) - Acc(Probe_MATH|ckpt0)`
  - `FWT_CS_SCI_to_MATH = Acc(Probe_MATH|ckpt2) - Acc(Probe_MATH|ckpt0)`
- Forgetting:
  - `Forget_CS_after_SCI = Acc(Probe_CS|ckpt1) - Acc(Probe_CS|ckpt2)`
  - `Forget_CS_after_MATH = Acc(Probe_CS|ckpt1) - Acc(Probe_CS|ckpt3)`
  - `Forget_SCI_after_MATH = Acc(Probe_SCI|ckpt2) - Acc(Probe_SCI|ckpt3)`
- Skill pollution:
  - reuse accuracy
  - unique reused skills
  - reuse entropy
  - top-1 and top-3 skill share
  - cross-phase toxic reuse count and accuracy
  - library size per checkpoint

## Smoke Run

Build splits:

```bash
python continual_learning/build_phase_splits.py \
  --train-count 10 \
  --probe-count 100 \
  --output-dir results/continual_learning_smoke/splits
```

Run all three systems:

```bash
MODEL=your_model_name \
python continual_learning/run_stream.py \
  --system S0 \
  --split-dir results/continual_learning_smoke/splits \
  --output-dir results/continual_learning_smoke/S0 \
  --workers 20

MODEL=your_model_name \
python continual_learning/run_stream.py \
  --system S1 \
  --split-dir results/continual_learning_smoke/splits \
  --output-dir results/continual_learning_smoke/S1 \
  --workers 20

MODEL=your_model_name \
python continual_learning/run_stream.py \
  --system S2 \
  --split-dir results/continual_learning_smoke/splits \
  --output-dir results/continual_learning_smoke/S2 \
  --workers 20
```

Build the report:

```bash
python continual_learning/report_continual.py \
  --run-root results/continual_learning_smoke
```

## Minimal Implementation Notes

- The scripts use an isolated `MUSE_GENERATED_SKILLS_ROOT` per stream output directory.
- Training is sequential inside each stream to preserve phase order.
- Probe eval may use process workers but always runs with `save=False`.
- The current implementation computes toxic reuse by skill phase provenance stored in `skill_phase_map.json`.
- It intentionally does not change default behavior of the existing compare/evolution scripts.
