#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_ablation.common import (  # noqa: E402
    append_jsonl,
    corrupt_reward,
    llm_score_candidate,
    read_jsonl,
    reward_for_candidate,
    threshold_for_candidate,
    write_csv,
)


def score_one(candidate, mode: str, *, reward_threshold, reward_flip_prob: float, tau: float, rng: random.Random):
    oracle_correct = bool(candidate.get("trace_correct") is True)
    if mode == "oracle":
        return {
            "pid": candidate["pid"],
            "signal_mode": mode,
            "keep": oracle_correct,
            "score": 1.0 if oracle_correct else 0.0,
            "reason": "oracle trace.correct",
            "oracle_correct": oracle_correct,
        }
    if mode == "reward_only":
        reward, reason = reward_for_candidate(candidate, tau=tau)
        threshold = threshold_for_candidate(candidate, reward_threshold)
        return {
            "pid": candidate["pid"],
            "signal_mode": mode,
            "keep": reward >= threshold,
            "score": reward,
            "reason": f"{reason}; threshold={threshold}",
            "oracle_correct": oracle_correct,
        }
    if mode == "noisy_reward":
        reward, reason = reward_for_candidate(candidate, tau=tau)
        threshold = threshold_for_candidate(candidate, reward_threshold)
        corrupted, flipped = corrupt_reward(candidate, reward, flip_prob=reward_flip_prob, threshold=threshold, rng=rng)
        return {
            "pid": candidate["pid"],
            "signal_mode": mode,
            "keep": corrupted >= 1.0,
            "score": corrupted,
            "reason": f"{reason}; threshold={threshold}; flip_prob={reward_flip_prob}; flipped={flipped}",
            "oracle_correct": oracle_correct,
            "raw_reward": reward,
            "flipped": flipped,
        }
    if mode in {"self_critique", "verifier_outcome", "verifier_step"}:
        scored = llm_score_candidate(candidate, mode=mode)
        return {
            "pid": candidate["pid"],
            "signal_mode": mode,
            "keep": bool(scored["keep"]),
            "score": scored["score"],
            "reason": scored["reason"],
            "oracle_correct": oracle_correct,
        }
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a fixed candidate pool with one signal source.")
    parser.add_argument("candidate_pool")
    parser.add_argument("--signal-mode", required=True, choices=[
        "oracle", "reward_only", "noisy_reward", "self_critique", "verifier_outcome", "verifier_step"
    ])
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--reward-flip-prob", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1e-6)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "signal_scores.jsonl"
    if out_path.exists():
        out_path.unlink()

    candidates = read_jsonl(args.candidate_pool)
    rng = random.Random(args.seed)
    rows = []
    for candidate in tqdm(candidates, desc=args.signal_mode):
        row = score_one(
            candidate,
            args.signal_mode,
            reward_threshold=args.reward_threshold,
            reward_flip_prob=args.reward_flip_prob,
            tau=args.tau,
            rng=rng,
        )
        rows.append(row)
        append_jsonl(out_path, row)

    write_csv(out_dir / "signal_scores.csv", rows, ["pid", "signal_mode", "keep", "score", "reason", "oracle_correct"])
    print(f"Saved signal scores to: {out_path}")


if __name__ == "__main__":
    main()
