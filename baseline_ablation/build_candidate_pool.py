#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse.compose import build_profile_from_trace  # noqa: E402
from muse.io_utils import save_json, save_jsonl  # noqa: E402
from muse.orchestrator import MultimodalMetaAgent  # noqa: E402
from baseline_ablation.common import (  # noqa: E402
    REPO_ROOT,
    clear_directory,
    load_env,
    load_mathvista_tasks,
    task_without_gold,
    trace_to_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed seed candidate pool for signal-source ablations.")
    parser.add_argument("--hf-split", default="testmini")
    parser.add_argument("--hf-cache-dir", default=str(REPO_ROOT / ".cache_hf_runtime" / "datasets"))
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    load_env()
    model = os.getenv("MODEL") or os.getenv("DEFAULT_MODEL")
    if model:
        os.environ.setdefault("MODEL", model)
        os.environ.setdefault("DEFAULT_MODEL", model)
        os.environ.setdefault("VISION_MODEL", model)
        os.environ.setdefault("ORCHESTRATOR_MODEL", model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_root = out_dir / "generated_skills_work"
    clear_directory(generated_root)
    os.environ["MUSE_GENERATED_SKILLS_ROOT"] = str(generated_root.resolve())

    tasks = load_mathvista_tasks(args.hf_split, args.hf_cache_dir, args.offset, args.seed_count)
    save_jsonl(out_dir / "seed_tasks.jsonl", [task.to_dict() for task in tasks])
    save_json(out_dir / "manifest.json", {
        "hf_split": args.hf_split,
        "seed_count": args.seed_count,
        "offset": args.offset,
        "backbone_model": os.getenv("MODEL"),
        "multimodal_setting": "current_full",
        "generated_skills_root": str(generated_root.resolve()),
    })

    agent = MultimodalMetaAgent(
        allow_generated_skill_reuse=True,
        allow_save_generated_skills=False,
        experiment_tag="baseline_ablation_candidate_pool",
    )
    rows = []
    for task in tqdm(tasks, desc="candidate_pool"):
        trace = agent.solve(task)
        profile = build_profile_from_trace(trace, REPO_ROOT)
        profile.setdefault("provenance", {})
        profile["provenance"].update({
            "sample_id": task.sample_id,
            "source_workspace": trace.workspace,
            "generation_mode": profile.get("hint_generation_mode", "candidate_pool_profile"),
            "final_answer_normalized": trace.final_answer_normalized,
        })
        candidate = trace_to_candidate(trace, task, profile)
        candidate["task_without_gold"] = task_without_gold(task)
        rows.append(candidate)

    save_jsonl(out_dir / "candidate_pool.jsonl", rows)
    print(f"Saved candidate pool to: {out_dir / 'candidate_pool.jsonl'}")


if __name__ == "__main__":
    main()
