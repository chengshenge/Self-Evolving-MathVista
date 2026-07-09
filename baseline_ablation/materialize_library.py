#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse.compose import write_skill_from_profile  # noqa: E402
from muse.schemas import TaskPacket  # noqa: E402
from baseline_ablation.common import clear_directory, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize kept candidates into a generated-skill library.")
    parser.add_argument("candidate_pool")
    parser.add_argument("signal_scores")
    parser.add_argument("--output-library-dir", required=True)
    parser.add_argument(
        "--require-oracle-correct",
        action="store_true",
        help="Only materialize candidates that are both keep=True and oracle_correct=True. Use for G1.",
    )
    args = parser.parse_args()

    candidates = {str(row["pid"]): row for row in read_jsonl(args.candidate_pool)}
    scores = read_jsonl(args.signal_scores)
    out_dir = Path(args.output_library_dir)
    clear_directory(out_dir)

    registry_rows = []
    for score in scores:
        if not score.get("keep"):
            continue
        if args.require_oracle_correct and score.get("oracle_correct") is not True:
            continue
        pid = str(score.get("pid"))
        candidate = candidates.get(pid)
        if not candidate:
            continue
        profile = dict(candidate.get("candidate_profile") or {})
        if not profile:
            continue
        task = TaskPacket.from_dict(candidate.get("task") or {})
        name = str(profile.get("name") or f"candidate_{pid}").strip()
        if not name:
            name = f"candidate_{pid}"
        skill_dir = write_skill_from_profile(
            profile,
            out_dir / name,
            task=task,
            extra_note=f"\nMaterialized by baseline_ablation using signal={score.get('signal_mode')} score={score.get('score')}.\n",
        )
        registry_rows.append({
            "skill_name": skill_dir.name,
            "sample_id": pid,
            "source_workspace": candidate.get("workspace"),
            "generation_mode": f"baseline_ablation_{score.get('signal_mode')}",
            "reflection": False,
            "signal_score": score.get("score"),
            "oracle_correct": score.get("oracle_correct"),
        })

    with (out_dir / "_generated_skill_registry.jsonl").open("w", encoding="utf-8") as f:
        for row in registry_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Materialized {len(registry_rows)} skills to: {out_dir}")


if __name__ == "__main__":
    main()
