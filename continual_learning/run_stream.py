#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse.io_utils import save_json  # noqa: E402
from continual_learning.common import (  # noqa: E402
    PHASES,
    clear_dir,
    copy_library,
    library_skill_names,
    load_env,
    load_tasks_jsonl,
    read_jsonl,
    run_train_phase,
    summarize_rows,
)


SYSTEMS = {"S0": "no_evolution", "S1": "frozen_after_p1", "S2": "full_continual"}


def should_save(system: str, phase_index: int) -> bool:
    if system == "S0":
        return False
    if system == "S1":
        return phase_index == 1
    if system == "S2":
        return True
    raise ValueError(system)


def update_skill_phase_map(library_dir: Path, skill_phase_map: dict, phase_key: str) -> dict:
    for name in library_skill_names(library_dir):
        skill_phase_map.setdefault(name, phase_key)
    return skill_phase_map


def run_eval(
    split_dir: Path,
    library_dir: Path,
    system: str,
    checkpoint: str,
    out_dir: Path,
    workers: int,
    skill_phase_map_path: Path,
    *,
    reuse_min_score: float,
    reuse_top_k: int,
    reuse_accept_conf: float,
) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "eval_checkpoint.py"),
        "--split-dir", str(split_dir),
        "--library-dir", str(library_dir),
        "--checkpoint", checkpoint,
        "--system", system,
        "--output-dir", str(out_dir),
        "--workers", str(workers),
        "--skill-phase-map", str(skill_phase_map_path),
        "--reuse-min-score", str(reuse_min_score),
        "--reuse-top-k", str(reuse_top_k),
        "--reuse-accept-conf", str(reuse_accept_conf),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one continual-learning stream system.")
    parser.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--reuse-min-score", type=float, default=float(os.getenv("CONTINUAL_REUSE_MIN_SCORE", "0.0")))
    parser.add_argument("--reuse-top-k", type=int, default=int(os.getenv("CONTINUAL_REUSE_TOP_K", "5")))
    parser.add_argument("--reuse-accept-conf", type=float, default=float(os.getenv("CONTINUAL_REUSE_ACCEPT_CONF", "0.70")))
    args = parser.parse_args()

    load_env()
    model = os.getenv("MODEL") or os.getenv("DEFAULT_MODEL")
    if model:
        os.environ.setdefault("MODEL", model)
        os.environ.setdefault("DEFAULT_MODEL", model)
        os.environ.setdefault("VISION_MODEL", model)
        os.environ.setdefault("ORCHESTRATOR_MODEL", model)

    split_dir = Path(args.split_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_library = out_dir / "work_generated_skills"
    clear_dir(work_library)
    os.environ["MUSE_GENERATED_SKILLS_ROOT"] = str(work_library.resolve())

    skill_phase_map = {}
    skill_phase_map_path = out_dir / "skill_phase_map.json"
    save_json(skill_phase_map_path, skill_phase_map)

    ckpt0 = out_dir / "library_ckpt_0_empty"
    copy_library(work_library, ckpt0)
    run_eval(
        split_dir,
        ckpt0,
        args.system,
        "ckpt0",
        out_dir / "eval" / "ckpt0",
        args.workers,
        skill_phase_map_path,
        reuse_min_score=args.reuse_min_score,
        reuse_top_k=args.reuse_top_k,
        reuse_accept_conf=args.reuse_accept_conf,
    )

    train_summaries = {}
    for phase_index, (phase_key, _domain, split_name) in enumerate(PHASES, start=1):
        tasks = load_tasks_jsonl(split_dir / f"{split_name}.jsonl")
        rows = run_train_phase(
            tasks,
            generated_root=work_library,
            save=should_save(args.system, phase_index),
            reuse=args.system != "S0",
            phase=phase_key,
            out_path=out_dir / "train" / f"{split_name}.jsonl",
            experiment_tag=("continual_seed_train_" if should_save(args.system, phase_index) else "continual_train_") + phase_key,
            suppress_reuse_registry=args.system == "S1" and phase_index > 1,
        )
        train_summaries[split_name] = summarize_rows(rows)
        if should_save(args.system, phase_index):
            skill_phase_map = update_skill_phase_map(work_library, skill_phase_map, f"p{phase_index}_{phase_key}")
            save_json(skill_phase_map_path, skill_phase_map)
        ckpt_dir = out_dir / f"library_ckpt_{phase_index}_after_p{phase_index}"
        copy_library(work_library, ckpt_dir)
        run_eval(
            split_dir,
            ckpt_dir,
            args.system,
            f"ckpt{phase_index}",
            out_dir / "eval" / f"ckpt{phase_index}",
            args.workers,
            skill_phase_map_path,
            reuse_min_score=args.reuse_min_score,
            reuse_top_k=args.reuse_top_k,
            reuse_accept_conf=args.reuse_accept_conf,
        )

    save_json(out_dir / "stream_summary.json", {
        "system": args.system,
        "system_name": SYSTEMS[args.system],
        "split_dir": str(split_dir.resolve()),
        "train_summaries": train_summaries,
        "num_final_skills": len(library_skill_names(work_library)),
        "skill_phase_map": skill_phase_map,
    })
    print(f"Saved stream outputs to: {out_dir}")


if __name__ == "__main__":
    main()
