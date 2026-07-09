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
from baseline_ablation.common import (  # noqa: E402
    REPO_ROOT,
    load_env,
    load_mathvista_tasks,
    read_jsonl,
    run_eval_task,
    slice_name,
    summarize_rows,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MathVista eval using a fixed generated-skill library.")
    parser.add_argument("--library-dir", required=True)
    parser.add_argument("--hf-split", default="testmini")
    parser.add_argument("--hf-cache-dir", default=str(REPO_ROOT / ".cache_hf_runtime" / "datasets"))
    parser.add_argument("--eval-count", type=int, default=980)
    parser.add_argument("--offset", type=int, default=20, help="Default skips the 20 seed examples.")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    load_env()
    model = os.getenv("MODEL") or os.getenv("DEFAULT_MODEL")
    if model:
        os.environ.setdefault("MODEL", model)
        os.environ.setdefault("DEFAULT_MODEL", model)
        os.environ.setdefault("VISION_MODEL", model)
        os.environ.setdefault("ORCHESTRATOR_MODEL", model)
    os.environ["MUSE_GENERATED_SKILLS_ROOT"] = str(Path(args.library_dir).resolve())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_with_library.jsonl"
    tasks = load_mathvista_tasks(args.hf_split, args.hf_cache_dir, args.offset, args.eval_count)
    save_jsonl(out_dir / "eval_tasks.jsonl", [task.to_dict() for task in tasks])

    existing = {str(r.get("pid")): r for r in read_jsonl(out_path)}
    pending = [task for task in tasks if str(task.sample_id) not in existing]
    if pending:
        with cf.ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {
                ex.submit(run_eval_task, task.to_dict(), str(Path(args.library_dir).resolve()), "baseline_ablation_eval_with_library"): task
                for task in pending
            }
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="eval_with_library"):
                row = fut.result()
                existing[str(row.get("pid"))] = row
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = [existing[str(task.sample_id)] for task in tasks if str(task.sample_id) in existing]
    save_jsonl(out_path, rows)
    summary = summarize_rows(rows)
    summary["config"] = {
        "library_dir": str(Path(args.library_dir).resolve()),
        "hf_split": args.hf_split,
        "offset": args.offset,
        "eval_count": len(tasks),
        "workers": args.workers,
        "env": {k: os.getenv(k) for k in ["BASE_URL", "MODEL", "DEFAULT_MODEL", "VISION_MODEL", "ORCHESTRATOR_MODEL", "MODEL_PROTOCOL"]},
    }
    save_json(out_dir / "summary.json", summary)

    task_by_pid = {str(task.sample_id): task.to_dict() for task in tasks}
    per_pid = []
    slice_stats = {}
    buckets = {"overall": rows}
    for row in rows:
        sl = slice_name(task_by_pid.get(str(row.get("pid")), {}))
        if sl:
            buckets.setdefault(sl, []).append(row)
        per_pid.append({
            "pid": row.get("pid"),
            "correct": row.get("correct"),
            "prediction": row.get("prediction"),
            "gold": row.get("gold"),
            "used_generated_skill": row.get("used_generated_skill"),
            "error": row.get("error"),
            "slice": sl or "",
        })
    for name, bucket in buckets.items():
        slice_stats[name] = summarize_rows(bucket)
    save_json(out_dir / "slice_stats.json", slice_stats)
    write_csv(out_dir / "per_pid_eval.csv", per_pid, ["pid", "correct", "prediction", "gold", "used_generated_skill", "error", "slice"])
    print(f"Saved eval outputs to: {out_dir}")


if __name__ == "__main__":
    main()
