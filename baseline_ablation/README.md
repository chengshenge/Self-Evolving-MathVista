# Section 6.4 Signal-Source Ablations

This directory contains a minimal executable implementation for the evolution signal-source ablation.

## Definitions

G0 is the existing system result and is not regenerated in this ablation pipeline.

Current G0 semantics:

- Oracle-gated saving plus current step-level-verifier-enhanced candidate generation.
- The seed stage generates candidates through the normal current agent chain.
- The current verifier/agent chain may refine candidate generation.
- Final skill saving is still primarily gated by `trace.correct == True`.

The scripts here only implement G1/G2/G3 and offline candidate-level reliability.

## Experimental Principle

All signal sources must share one fixed seed candidate pool:

1. Build `candidate_pool.jsonl` once with the current full system.
2. Score the same candidates with different signals.
3. Materialize skills from candidates selected by each signal.
4. Run eval with each materialized library.

G1/G2/G3 differ only in the signal that decides whether a candidate enters the skill library. Retrieval and execution should stay as close as possible to the existing system.

## Groups

G1: Oracle + outcome-only verifier

- Uses the fixed candidate pool.
- The verifier sees task without gold, image, final answer, final normalized answer, and a very short rationale.
- The verifier does not see visual facts, focus answers, math rounds, verify issues, or step-level trace.
- A candidate is materialized only if `verifier.keep == true` and `trace.correct == true`.

G2: Environment reward only

- No verifier and no self-critique.
- Binary exact match for multiple-choice/text tasks.
- Numeric partial reward: `max(0, 1 - abs(pred-gold)/(abs(gold)+tau))`.
- Candidate is materialized when reward passes threshold.

G3: Noisy reward

- Same as G2, but reward is corrupted by `--reward-flip-prob`.
- Binary reward is flipped directly.
- Numeric reward is thresholded first and then flipped.

## Candidate-Level Reliability

The offline reliability table compares:

- C1 `self_critique`: actor self-evaluation; no gold; no external verifier conclusion.
- C2 `verifier_outcome`: same outcome-only verifier as G1.
- C3 `verifier_step`: independent verifier with step-level evidence.
- C4 `oracle`: direct `trace.correct` reference.

Self-critique and verifier are intentionally separated. Both may use an LLM, but self-critique is the actor judging itself, while verifier is an independent scorer.

## Typical Run

Set `MODEL` (and optionally `VISION_MODEL` / `ORCHESTRATOR_MODEL`) from your `.env` or shell. The typical MathVista `testmini` run uses seed count 20 and eval count 980.

Build the shared candidate pool:

```bash
MODEL=your_model_name \
python baseline_ablation/build_candidate_pool.py \
  --hf-split testmini \
  --seed-count 20 \
  --offset 0 \
  --output-dir results/baseline_ablation/pool_main
```

Score the pool:

```bash
python baseline_ablation/score_candidate_pool.py results/baseline_ablation/pool_main/candidate_pool.jsonl \
  --signal-mode verifier_outcome \
  --output-dir results/baseline_ablation/g1_scores

python baseline_ablation/score_candidate_pool.py results/baseline_ablation/pool_main/candidate_pool.jsonl \
  --signal-mode reward_only \
  --output-dir results/baseline_ablation/g2_scores

python baseline_ablation/score_candidate_pool.py results/baseline_ablation/pool_main/candidate_pool.jsonl \
  --signal-mode noisy_reward \
  --reward-flip-prob 0.1 \
  --output-dir results/baseline_ablation/g3_scores
```

Materialize libraries:

```bash
python baseline_ablation/materialize_library.py \
  results/baseline_ablation/pool_main/candidate_pool.jsonl \
  results/baseline_ablation/g1_scores/signal_scores.jsonl \
  --output-library-dir results/baseline_ablation/g1_library
```

Run eval with a fixed library:

```bash
python baseline_ablation/run_eval_with_library.py \
  --library-dir results/baseline_ablation/g1_library \
  --hf-split testmini \
  --offset 20 \
  --eval-count 980 \
  --output-dir results/baseline_ablation/g1_eval
```

Offline reliability:

```bash
python baseline_ablation/report_signal_reliability.py \
  results/baseline_ablation/pool_main/candidate_pool.jsonl \
  results/baseline_ablation/self_critique_scores/signal_scores.jsonl \
  results/baseline_ablation/g1_scores/signal_scores.jsonl \
  results/baseline_ablation/step_verifier_scores/signal_scores.jsonl \
  results/baseline_ablation/oracle_scores/signal_scores.jsonl \
  --output-dir results/baseline_ablation/reliability_report
```

Generate the G0 MUSE reference run with the main compare runner:

```bash
MODEL=your_model_name python run_compare_mathvista_parallel.py \
  --hf-split testmini \
  --seed-count 20 \
  --eval-count 980 \
  --workers 10 \
  --output-dir results/baseline_ablation/g0_muse
```

Downstream ablation report:

```bash
python baseline_ablation/report_signal_ablation.py \
  --g0-dir results/baseline_ablation/g0_muse \
  --g1-dir results/baseline_ablation/g1_eval \
  --g2-dir results/baseline_ablation/g2_eval \
  --g3-dir results/baseline_ablation/g3_eval \
  --output-dir results/baseline_ablation/downstream_report
```
