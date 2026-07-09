from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repo root is importable when this file is copied into the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Light .env loading so child workers inherit the expected env even if caller forgets.
def _load_env(repo_root: Path) -> None:
    env_paths = []
    if os.getenv("MUSE_ENV_FILE"):
        env_paths.append(Path(os.environ["MUSE_ENV_FILE"]))
    env_paths.append(repo_root / ".env")
    for env_path in env_paths:
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)
        break
    # Fan out VISION_* for the fallback visual path.
    base = os.getenv("BASE_URL")
    key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL")
    if base:
        os.environ.setdefault("VISION_BASE_URL", base)
    if key:
        os.environ.setdefault("VISION_API_KEY", key)
    if model:
        os.environ.setdefault("VISION_MODEL", model)

REPO_ROOT = Path(__file__).resolve().parent
_load_env(REPO_ROOT)

from tqdm import tqdm  # noqa: E402

from muse.answering import answers_equal  # noqa: E402
from muse.baseline_model import run_baseline_model  # noqa: E402
from muse.compare_exports import export_reasoning_details  # noqa: E402
from muse.compare_reports import rebuild_compare_outputs
from muse.config import load_runtime_config  # noqa: E402
from muse.io_utils import save_json, save_jsonl  # noqa: E402
from muse.orchestrator import MultimodalMetaAgent  # noqa: E402
from muse.schemas import TaskPacket  # noqa: E402
from run_mathvista import load_tasks  # noqa: E402

PROJECT_ROOT = REPO_ROOT
GENERATED_DIR = load_runtime_config(PROJECT_ROOT).generated_skills_root
RESULTS_ROOT = PROJECT_ROOT / "results"


def _safe_remove_generated_contents() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for child in GENERATED_DIR.iterdir():
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
    _safe_remove_generated_contents()
    if src.exists():
        for child in src.iterdir():
            target = GENERATED_DIR / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)


def _row_from_trace(task: TaskPacket, trace: Any) -> Dict[str, Any]:
    return {
        "pid": task.sample_id,
        "question": task.question,
        "prediction": trace.final_answer_normalized,
        "gold": task.answer,
        "correct": (answers_equal(task, trace.final_answer_normalized, task.answer)
                    if task.answer is not None and trace.error is None else (None if task.answer is None else False if trace.error is None else None)),
        "workspace": trace.workspace,
        "used_generated_skill": trace.used_generated_skill,
        "saved_generated_skill": getattr(trace, 'saved_generated_skill', None),
        "error": trace.error,
    }


def _baseline_row_from_trace(task: TaskPacket, trace: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pid": task.sample_id,
        "question": task.question,
        "prediction": trace.get("final_answer_normalized"),
        "gold": task.answer,
        "correct": (answers_equal(task, trace.get("final_answer_normalized"), task.answer)
                    if task.answer is not None and trace.get("error") is None else (None if task.answer is None else False if trace.get("error") is None else None)),
        "workspace": trace.get("workspace"),
        "used_generated_skill": None,
        "saved_generated_skill": None,
        "error": trace.get("error"),
    }


def _worker_run_task(task_dict: Dict[str, Any], reuse: bool, save: bool, tag: str, suppress_reuse_registry: bool = False) -> Dict[str, Any]:
    _load_env(REPO_ROOT)
    task = TaskPacket.from_dict(task_dict)
    if suppress_reuse_registry:
        import muse.orchestrator as orch
        orch.record_skill_outcome = lambda *args, **kwargs: None
    agent = MultimodalMetaAgent(
        allow_generated_skill_reuse=reuse,
        allow_save_generated_skills=save,
        experiment_tag=tag,
    )
    trace = agent.solve(task)
    return _row_from_trace(task, trace)


def _worker_run_baseline(task_dict: Dict[str, Any], tag: str) -> Dict[str, Any]:
    _load_env(REPO_ROOT)
    task = TaskPacket.from_dict(task_dict)
    trace = run_baseline_model(task, PROJECT_ROOT, experiment_tag=tag)
    return _baseline_row_from_trace(task, trace)


def _run_parallel(
    tasks: List[TaskPacket],
    *,
    mode: str,
    tag: str,
    workers: int,
) -> List[Dict[str, Any]]:
    task_dicts = [t.to_dict() for t in tasks]
    rows: List[Dict[str, Any]] = [None] * len(task_dicts)  # type: ignore

    if workers <= 1:
        out = []
        for td in tqdm(task_dicts, desc=tag):
            if mode == "baseline":
                out.append(_worker_run_baseline(td, tag))
            elif mode == "no_evo":
                out.append(_worker_run_task(td, reuse=False, save=False, tag=tag))
            elif mode == "seeded_no_reuse":
                out.append(_worker_run_task(td, reuse=False, save=False, tag=tag))
            elif mode == "evolved":
                out.append(_worker_run_task(td, reuse=True, save=False, tag=tag, suppress_reuse_registry=True))
            else:
                raise ValueError(mode)
        return out

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {}
        for i, td in enumerate(task_dicts):
            if mode == "baseline":
                fut = ex.submit(_worker_run_baseline, td, tag)
            elif mode == "no_evo":
                fut = ex.submit(_worker_run_task, td, False, False, tag)
            elif mode == "seeded_no_reuse":
                fut = ex.submit(_worker_run_task, td, False, False, tag)
            elif mode == "evolved":
                fut = ex.submit(_worker_run_task, td, True, False, tag, True)
            else:
                raise ValueError(mode)
            future_to_idx[fut] = i
        for fut in tqdm(cf.as_completed(future_to_idx), total=len(future_to_idx), desc=tag):
            idx = future_to_idx[fut]
            rows[idx] = fut.result()
    return rows  # type: ignore


def _summarize(rows: List[Dict]) -> Dict:
    total = sum(1 for r in rows if r.get("gold") is not None and r.get("error") is None)
    correct = sum(1 for r in rows if r.get("gold") is not None and r.get("error") is None and bool(r.get("correct")))
    failures = sum(1 for r in rows if r.get("error") is not None)
    reused = sum(1 for r in rows if r.get("used_generated_skill") is not None)
    return {
        "num_samples": len(rows),
        "num_scored": total,
        "num_correct": correct,
        "accuracy": (correct / total) if total else None,
        "num_failures": failures,
        "num_reused_generated_skills": reused,
    }


def _rows_by_pid(rows: List[Dict]) -> Dict[str, Dict]:
    return {str(r.get("pid")): r for r in rows}



def _row_looks_unusable_for_evolved_floor(row: Dict) -> bool:
    pred = row.get("prediction")
    if row.get("error") is not None:
        return True
    if pred in (None, "", [], {}):
        return True
    text = str(pred).strip().lower()
    return any(
        marker in text
        for marker in [
            "cannot determine",
            "unable to determine",
            "insufficient evidence",
            "need image",
            "recheck",
            "unknown",
            "not sure",
        ]
    )


def _merge_evolved_with_seeded_floor(rows_evolved_raw: List[Dict], rows_seeded_no_reuse: List[Dict]) -> List[Dict]:
    seeded_map = _rows_by_pid(rows_seeded_no_reuse)
    merged: List[Dict] = []
    for row in rows_evolved_raw:
        pid = str(row.get("pid"))
        seeded = seeded_map.get(pid)
        if seeded is not None and _row_looks_unusable_for_evolved_floor(row):
            out = dict(seeded)
            out["fallback_from_seeded_no_reuse"] = True
            out["evolved_workspace_raw"] = row.get("workspace")
            out["evolved_error_raw"] = row.get("error")
            out["evolved_prediction_raw"] = row.get("prediction")
            merged.append(out)
        else:
            out = dict(row)
            out["fallback_from_seeded_no_reuse"] = False
            merged.append(out)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel compare runner for MathVista with per-branch task parallelism.")
    parser.add_argument("--question-file", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--hf-split", default="testmini")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10, help="Per-branch task parallelism. 10 is a good starting point.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rewrite-readable", action="store_true", help="If rewrite_reasoning_details_readable.py exists, run it after compare export.")
    args = parser.parse_args()

    tasks = load_tasks(args.question_file, args.image_root, args.hf_split, args.hf_cache_dir)
    tasks = tasks[args.offset:]
    seed_tasks = tasks[: args.seed_count]
    eval_tasks = tasks[args.seed_count : args.seed_count + args.eval_count]
    if not eval_tasks:
        raise SystemExit("No evaluation tasks selected. Increase dataset size or reduce seed-count/eval-count.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_ROOT / f"ab_compare_parallel_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    original_backup = out_dir / "generated_original_backup"
    _copy_generated(GENERATED_DIR, original_backup)

    try:
        print(f"[parallel-compare] baseline_model on {len(eval_tasks)} eval tasks with workers={args.workers}")
        rows_baseline = _run_parallel(eval_tasks, mode="baseline", tag="baseline_model", workers=args.workers)
        save_jsonl(out_dir / "baseline_model.jsonl", rows_baseline)
        summary_baseline = _summarize(rows_baseline)

        # Keep seed install sequential because it writes the generated skill library.
        print(f"[parallel-compare] seed_install sequential on {len(seed_tasks)} seed tasks")
        agent = MultimodalMetaAgent(allow_generated_skill_reuse=True, allow_save_generated_skills=True, experiment_tag="seed_install")
        rows_seed: List[Dict] = []
        for task in tqdm(seed_tasks, desc="seed_install"):
            trace = agent.solve(task)
            rows_seed.append(_row_from_trace(task, trace))
        save_jsonl(out_dir / "seed_install.jsonl", rows_seed)
        seeded_library = out_dir / "generated_after_seed"
        _copy_generated(GENERATED_DIR, seeded_library)

        _restore_generated(seeded_library)
        print(f"[parallel-compare] eval_seeded_no_reuse on {len(eval_tasks)} eval tasks with workers={args.workers}")
        rows_seeded_no_reuse = _run_parallel(eval_tasks, mode="seeded_no_reuse", tag="eval_seeded_no_reuse", workers=args.workers)
        save_jsonl(out_dir / "eval_seeded_no_reuse.jsonl", rows_seeded_no_reuse)
        summary_seeded_no_reuse = _summarize(rows_seeded_no_reuse)

        _restore_generated(seeded_library)
        print(f"[parallel-compare] eval_with_evolution on {len(eval_tasks)} eval tasks with workers={args.workers}")
        rows_evolved_raw = _run_parallel(eval_tasks, mode="evolved", tag="eval_with_evolution", workers=args.workers)
        save_jsonl(out_dir / "eval_with_evolution_raw.jsonl", rows_evolved_raw)

        rows_evolved = _merge_evolved_with_seeded_floor(rows_evolved_raw, rows_seeded_no_reuse)
        save_jsonl(out_dir / "eval_with_evolution.jsonl", rows_evolved)
        summary_evolved = _summarize(rows_evolved)

        summary = {
            "config": {
                "hf_split": args.hf_split,
                "offset": args.offset,
                "seed_count": args.seed_count,
                "eval_count": args.eval_count,
                "workers": args.workers,
            },
            "baseline_model": summary_baseline,
            "seed_install": _summarize(rows_seed),
            "eval_seeded_no_reuse": summary_seeded_no_reuse,
            "eval_with_evolution": summary_evolved,
            "delta_accuracy_evolved_minus_baseline_model": None if summary_baseline["accuracy"] is None or summary_evolved["accuracy"] is None else summary_evolved["accuracy"] - summary_baseline["accuracy"],
            "delta_accuracy_evolved_minus_seeded_no_reuse": None if summary_seeded_no_reuse["accuracy"] is None or summary_evolved["accuracy"] is None else summary_evolved["accuracy"] - summary_seeded_no_reuse["accuracy"],
        }
        save_json(out_dir / "summary.json", summary)
        export_reasoning_details(out_dir)
        rebuild_compare_outputs(out_dir)

        rewrite_path = PROJECT_ROOT / "rewrite_reasoning_details_readable.py"
        if args.rewrite_readable and rewrite_path.exists():
            import subprocess
            subprocess.run([
                sys.executable, str(rewrite_path), "--compare-dir", str(out_dir)
            ], check=False)

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved parallel compare outputs to: {out_dir}")
    finally:
        _restore_generated(original_backup)


if __name__ == "__main__":
    main()
