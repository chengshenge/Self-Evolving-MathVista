#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muse.io_utils import save_json, save_jsonl  # noqa: E402
from continual_learning.common import (  # noqa: E402
    PROBES,
    compute_reuse_stats,
    eval_worker,
    library_skill_names,
    load_env,
    load_skill_phase_map,
    load_tasks_jsonl,
    read_jsonl,
    summarize_rows,
    write_csv,
)


def eval_probe(
    tasks,
    *,
    generated_root: Path,
    out_path: Path,
    workers: int,
    tag: str,
    reuse_enabled: bool,
    reuse_min_score: float,
    reuse_top_k: int,
    reuse_accept_conf: float,
):
    existing = {str(row.get("pid")): row for row in read_jsonl(out_path)}
    pending = [task for task in tasks if str(task.sample_id) not in existing]
    if pending:
        with cf.ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = {
                ex.submit(
                    eval_worker,
                    task.to_dict(),
                    str(generated_root.resolve()),
                    tag,
                    reuse_enabled=reuse_enabled,
                    reuse_min_score=reuse_min_score,
                    reuse_top_k=reuse_top_k,
                    reuse_accept_conf=reuse_accept_conf,
                ): task
                for task in pending
            }
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc=tag):
                row = fut.result()
                existing[str(row.get("pid"))] = row
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = [existing[str(task.sample_id)] for task in tasks if str(task.sample_id) in existing]
    save_jsonl(out_path, rows)
    return rows


def accepted_reuse_rows(rows, *, system: str, checkpoint: str, probe: str, domain: str):
    out = []
    for row in rows:
        for attempt in row.get("reuse_attempts") or []:
            if not attempt.get("accepted"):
                continue
            skill_name = attempt.get("skill_name")
            out.append({
                "system": system,
                "checkpoint": checkpoint,
                "probe": probe,
                "domain": domain,
                "pid": row.get("pid"),
                "skill_name": skill_name,
                "used_generated_skill": row.get("used_generated_skill"),
                "accepted_but_not_used": bool(skill_name and row.get("used_generated_skill") != skill_name),
                "correct": row.get("correct"),
                "score": attempt.get("score"),
                "normalized_answer": attempt.get("normalized_answer"),
                "verifier_decision": attempt.get("verifier_decision"),
                "effective_verifier_decision": attempt.get("effective_verifier_decision"),
                "effective_verifier_confidence": attempt.get("effective_verifier_confidence"),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one continual-learning checkpoint on all probes.")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--library-dir", required=True)
    parser.add_argument("--checkpoint", required=True, choices=["ckpt0", "ckpt1", "ckpt2", "ckpt3"])
    parser.add_argument("--system", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--skill-phase-map", default=None)
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
    library_dir = Path(args.library_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skill_phase_map = load_skill_phase_map(Path(args.skill_phase_map)) if args.skill_phase_map else {}
    skill_names = library_skill_names(library_dir)
    probe_reuse_enabled = args.system != "S0"
    audit = {
        "checkpoint_name": args.checkpoint,
        "system": args.system,
        "library_dir": str(library_dir.resolve()),
        "generated_dir_exists": library_dir.exists(),
        "num_skill_dirs": len(skill_names),
        "skill_names": skill_names,
        "probe_eval": {
            "reuse": probe_reuse_enabled,
            "save": False,
            "reuse_min_score": args.reuse_min_score,
            "reuse_top_k": args.reuse_top_k,
            "reuse_accept_conf": args.reuse_accept_conf,
        },
    }
    save_json(out_dir / "probe_library_audit.json", audit)

    checkpoint_index = int(args.checkpoint[-1])
    summaries = {}
    pollution_rows = []
    accepted_rows = []
    for probe_key, domain, probe_name in PROBES:
        tasks = load_tasks_jsonl(split_dir / f"{probe_name}.jsonl")
        rows = eval_probe(
            tasks,
            generated_root=library_dir,
            out_path=out_dir / f"{probe_name}.jsonl",
            workers=args.workers,
            tag=f"{args.system}_{args.checkpoint}_{probe_key}",
            reuse_enabled=probe_reuse_enabled,
            reuse_min_score=args.reuse_min_score,
            reuse_top_k=args.reuse_top_k,
            reuse_accept_conf=args.reuse_accept_conf,
        )
        summary = summarize_rows(rows)
        reuse_stats = compute_reuse_stats(
            rows,
            skill_phase_map=skill_phase_map,
            old_domain=domain,
            current_phase_index=checkpoint_index,
        )
        summary.update(reuse_stats)
        summary["probe"] = probe_name
        summary["domain"] = domain
        summaries[probe_name] = summary
        accepted_rows.extend(accepted_reuse_rows(rows, system=args.system, checkpoint=args.checkpoint, probe=probe_name, domain=domain))
        pollution_rows.append({
            "system": args.system,
            "checkpoint": args.checkpoint,
            "probe": probe_name,
            "domain": domain,
            **summary,
        })

    summary = {
        "system": args.system,
        "checkpoint": args.checkpoint,
        "library_dir": str(library_dir.resolve()),
        "num_saved_skills": len(skill_names),
        "probe_library_audit": audit,
        "probes": summaries,
    }
    save_json(out_dir / "summary.json", summary)
    write_csv(
        out_dir / "probe_summary.csv",
        pollution_rows,
        [
            "system", "checkpoint", "probe", "domain", "num_samples", "num_scored", "num_correct",
            "accuracy", "num_failures", "num_reused_generated_skills", "num_unique_reused_skills",
            "num_reuse_candidates_total", "num_reuse_attempts_total", "num_reuse_accepts_total",
            "num_used_generated_skills_total", "reuse_accuracy", "old_domain_reuse_precision",
            "reuse_entropy", "top1_skill_share", "top3_skill_share", "cross_phase_reuse_count",
            "cross_phase_toxic_reuse_count", "cross_phase_toxic_reuse_ratio", "cross_phase_toxic_reuse_accuracy",
        ],
    )
    write_csv(
        out_dir / "accepted_reuse.csv",
        accepted_rows,
        [
            "system", "checkpoint", "probe", "domain", "pid", "skill_name", "used_generated_skill",
            "accepted_but_not_used", "correct", "score", "normalized_answer", "verifier_decision",
            "effective_verifier_decision", "effective_verifier_confidence",
        ],
    )
    print(f"Saved checkpoint eval to: {out_dir}")


if __name__ == "__main__":
    main()
