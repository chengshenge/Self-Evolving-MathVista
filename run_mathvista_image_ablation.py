from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm  # noqa: E402

from muse.compare_exports import export_reasoning_details  # noqa: E402
from muse.compare_reports import rebuild_compare_outputs  # noqa: E402
from muse.io_utils import save_json, save_jsonl  # noqa: E402
from muse.orchestrator import MultimodalMetaAgent  # noqa: E402
from muse.schemas import TaskPacket  # noqa: E402
from run_compare_mathvista_parallel import (  # noqa: E402
    _baseline_row_from_trace,
    _load_env,
    _row_from_trace,
    _row_looks_unusable_for_evolved_floor,
    _summarize,
    _worker_run_baseline,
    _worker_run_task,
)
from run_mathvista import load_tasks  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"


def _generated_root() -> Path:
    configured = os.getenv("MUSE_GENERATED_SKILLS_ROOT")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        path = PROJECT_ROOT / "skills" / "subagents" / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_remove_generated_contents() -> None:
    root = _generated_root()
    for child in root.iterdir():
        if child.name.startswith("_"):
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _copy_generated(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)


def _restore_generated(src: Path) -> None:
    root = _generated_root()
    _safe_remove_generated_contents()
    if src.exists():
        for child in src.iterdir():
            target = root / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)


def _without_images(task: TaskPacket) -> TaskPacket:
    payload = task.to_dict()
    payload["image_path"] = ""
    payload["image_paths"] = []
    return TaskPacket.from_dict(payload)


def _tasks_for_visibility(tasks: List[TaskPacket], *, no_image: bool) -> List[TaskPacket]:
    return [_without_images(task) for task in tasks] if no_image else list(tasks)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_process_incremental(
    name: str,
    path: Path,
    tasks: List[TaskPacket],
    *,
    mode: str,
    workers: int,
    retry_errors: bool = False,
) -> List[Dict[str, Any]]:
    existing = _read_jsonl(path)
    rows_by_pid = {str(row.get("pid")): row for row in existing}
    pending = [
        task
        for task in tasks
        if str(task.sample_id) not in rows_by_pid
        or (retry_errors and rows_by_pid[str(task.sample_id)].get("error") is not None)
    ]
    if pending:
        task_dicts = [task.to_dict() for task in pending]
        if workers <= 1:
            for task_dict in tqdm(task_dicts, desc=name):
                if mode == "baseline":
                    row = _worker_run_baseline(task_dict, name)
                elif mode == "evolved":
                    row = _worker_run_task(task_dict, True, False, name, True)
                else:
                    raise ValueError(mode)
                rows_by_pid[str(row.get("pid"))] = row
                _append_jsonl(path, row)
        else:
            with cf.ProcessPoolExecutor(max_workers=workers) as ex:
                future_to_idx = {}
                for idx, task_dict in enumerate(task_dicts):
                    if mode == "baseline":
                        fut = ex.submit(_worker_run_baseline, task_dict, name)
                    elif mode == "evolved":
                        fut = ex.submit(_worker_run_task, task_dict, True, False, name, True)
                    else:
                        raise ValueError(mode)
                    future_to_idx[fut] = idx
                for fut in tqdm(cf.as_completed(future_to_idx), total=len(future_to_idx), desc=name):
                    row = fut.result()
                    rows_by_pid[str(row.get("pid"))] = row
                    _append_jsonl(path, row)
    ordered = [rows_by_pid[str(task.sample_id)] for task in tasks if str(task.sample_id) in rows_by_pid]
    save_jsonl(path, ordered)
    return ordered


def _run_seed_install_incremental(path: Path, seed_tasks: List[TaskPacket]) -> List[Dict[str, Any]]:
    existing = _read_jsonl(path)
    rows_by_pid = {str(row.get("pid")): row for row in existing}
    pending = [task for task in seed_tasks if str(task.sample_id) not in rows_by_pid]
    if pending:
        agent = MultimodalMetaAgent(
            allow_generated_skill_reuse=True,
            allow_save_generated_skills=True,
            experiment_tag="seed_install",
        )
        for task in tqdm(pending, desc="seed_install"):
            trace = agent.solve(task)
            row = _row_from_trace(task, trace)
            rows_by_pid[str(task.sample_id)] = row
            _append_jsonl(path, row)
    ordered = [rows_by_pid[str(task.sample_id)] for task in seed_tasks if str(task.sample_id) in rows_by_pid]
    save_jsonl(path, ordered)
    return ordered


def _merge_evolved_with_base_floor(evolved_rows: List[Dict[str, Any]], base_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_by_pid = {str(row.get("pid")): row for row in base_rows}
    merged: List[Dict[str, Any]] = []
    for row in evolved_rows:
        base = base_by_pid.get(str(row.get("pid")))
        if base is not None and _row_looks_unusable_for_evolved_floor(row):
            out = dict(base)
            out["fallback_from_base"] = True
            out["evolution_workspace_raw"] = row.get("workspace")
            out["evolution_error_raw"] = row.get("error")
            out["evolution_prediction_raw"] = row.get("prediction")
        else:
            out = dict(row)
            out["fallback_from_base"] = False
        merged.append(out)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="MathVista image visibility ablation for Base+Evolution.")
    parser.add_argument("--hf-split", default="testmini")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--eval-count", type=int, default=980)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed-no-image", action="store_true")
    parser.add_argument("--eval-no-image", action="store_true")
    parser.add_argument("--evolution-only", action="store_true", help="Skip base evaluation and only run seed_install plus evolution eval.")
    parser.add_argument("--retry-errors", action="store_true", help="Re-run rows that already exist but have a non-null error.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rewrite-readable", action="store_true")
    args = parser.parse_args()

    _load_env(PROJECT_ROOT)

    tasks = load_tasks(None, None, args.hf_split, args.hf_cache_dir)
    tasks = tasks[args.offset:]
    seed_tasks_original = tasks[: args.seed_count]
    eval_tasks_original = tasks[args.seed_count : args.seed_count + args.eval_count]
    if not seed_tasks_original or not eval_tasks_original:
        raise SystemExit("No seed/eval tasks selected.")

    seed_tasks = _tasks_for_visibility(seed_tasks_original, no_image=args.seed_no_image)
    eval_tasks = _tasks_for_visibility(eval_tasks_original, no_image=args.eval_no_image)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        name = []
        name.append("seed_no_image" if args.seed_no_image else "seed_with_image")
        name.append("eval_no_image" if args.eval_no_image else "eval_with_image")
        out_dir = RESULTS_ROOT / f"mathvista_muse_image_ablation_{'_'.join(name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    isolated_generated_root = out_dir / "generated_skills"
    os.environ["MUSE_GENERATED_SKILLS_ROOT"] = str(isolated_generated_root.resolve())

    manifest = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "hf_split": args.hf_split,
        "offset": args.offset,
        "seed_count": len(seed_tasks),
        "eval_count": len(eval_tasks),
        "workers": args.workers,
        "seed_no_image": args.seed_no_image,
        "eval_no_image": args.eval_no_image,
        "evolution_only": args.evolution_only,
        "retry_errors": args.retry_errors,
        "generated_skills_root": str(isolated_generated_root.resolve()),
        "env": {k: os.getenv(k) for k in ["BASE_URL", "MODEL", "DEFAULT_MODEL", "VISION_MODEL", "ORCHESTRATOR_MODEL", "MODEL_PROTOCOL"]},
    }
    save_json(out_dir / "manifest.json", manifest)
    save_jsonl(out_dir / "seed_tasks_original.jsonl", [task.to_dict() for task in seed_tasks_original])
    save_jsonl(out_dir / "eval_tasks_original.jsonl", [task.to_dict() for task in eval_tasks_original])
    save_jsonl(out_dir / "seed_tasks_used.jsonl", [task.to_dict() for task in seed_tasks])
    save_jsonl(out_dir / "eval_tasks_used.jsonl", [task.to_dict() for task in eval_tasks])

    base_rows: List[Dict[str, Any]] = []
    if not args.evolution_only:
        print(f"[ablation] base on {len(eval_tasks)} eval tasks, workers={args.workers}, eval_no_image={args.eval_no_image}")
        base_rows = _run_process_incremental(
            "base",
            out_dir / "base.jsonl",
            eval_tasks,
            mode="baseline",
            workers=args.workers,
            retry_errors=args.retry_errors,
        )

    seeded_library = out_dir / "generated_after_seed"
    if not seeded_library.exists():
        _safe_remove_generated_contents()
        print(f"[ablation] seed_install on {len(seed_tasks)} seed tasks, seed_no_image={args.seed_no_image}")
        seed_rows = _run_seed_install_incremental(out_dir / "seed_install.jsonl", seed_tasks)
        _copy_generated(isolated_generated_root, seeded_library)
    else:
        seed_rows = _read_jsonl(out_dir / "seed_install.jsonl")
        _restore_generated(seeded_library)

    _restore_generated(seeded_library)
    print(f"[ablation] base_evolution on {len(eval_tasks)} eval tasks, workers={args.workers}, eval_no_image={args.eval_no_image}")
    evolved_raw = _run_process_incremental(
        "base_evolution_raw",
        out_dir / "base_evolution_raw.jsonl",
        eval_tasks,
        mode="evolved",
        workers=args.workers,
        retry_errors=args.retry_errors,
    )
    evolved = evolved_raw if args.evolution_only else _merge_evolved_with_base_floor(evolved_raw, base_rows)
    save_jsonl(out_dir / "base_evolution.jsonl", evolved)

    summary = {
        "config": manifest,
        "seed_install": _summarize(seed_rows),
        "base_evolution_raw": _summarize(evolved_raw),
        "base_evolution": _summarize(evolved),
    }
    if not args.evolution_only:
        summary["base"] = _summarize(base_rows)
        base_acc = summary["base"]["accuracy"]
        evo_acc = summary["base_evolution"]["accuracy"]
        summary["delta_accuracy_base_evolution_minus_base"] = None if base_acc is None or evo_acc is None else evo_acc - base_acc
    save_json(out_dir / "summary.json", summary)

    export_reasoning_details(out_dir)
    rebuild_compare_outputs(out_dir)

    if args.rewrite_readable:
        rewrite_path = PROJECT_ROOT / "rewrite_reasoning_details_readable.py"
        if rewrite_path.exists():
            import subprocess

            subprocess.run([sys.executable, str(rewrite_path), "--compare-dir", str(out_dir)], check=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved image ablation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
