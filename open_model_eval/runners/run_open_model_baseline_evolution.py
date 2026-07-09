#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Allow running from open_model_eval/runners inside the repo.
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

from muse.answering import answers_equal  # noqa: E402
from muse.baseline_model import run_baseline_model  # noqa: E402
from muse.compare_exports import export_reasoning_details  # noqa: E402
from muse.compare_reports import rebuild_compare_outputs  # noqa: E402
from muse.io_utils import save_json, save_jsonl  # noqa: E402
from muse.config import load_runtime_config  # noqa: E402
from muse.orchestrator import MultimodalMetaAgent  # noqa: E402
from run_mathvista import load_tasks  # noqa: E402
from muse import llm_clients as _muse_llm_clients  # noqa: E402


def _install_open_model_token_cap() -> None:
    if getattr(_muse_llm_clients.OpenAIStyleClient, "_open_model_cap_installed", False):
        return

    original_chat_create = _muse_llm_clients.OpenAIStyleClient._chat_create

    def capped_chat_create(self, messages, *, temperature, max_tokens, response_format=None, reasoning_effort=None):
        raw_cap = os.getenv("OPEN_MODEL_MAX_TOKENS", "128").strip()
        try:
            cap = max(16, int(raw_cap))
        except Exception:
            cap = 128
        return original_chat_create(
            self,
            messages,
            temperature=temperature,
            max_tokens=min(max_tokens, cap),
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )

    _muse_llm_clients.OpenAIStyleClient._chat_create = capped_chat_create
    _muse_llm_clients.OpenAIStyleClient._open_model_cap_installed = True


_install_open_model_token_cap()

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


def _row_from_trace(task, trace: Any) -> Dict[str, Any]:
    pred = trace.final_answer_normalized
    gold = task.answer
    err = trace.error
    correct = None
    if gold is not None and err is None:
        correct = answers_equal(task, pred, gold)
    return {
        "pid": task.sample_id,
        "question": task.question,
        "prediction": pred,
        "gold": gold,
        "correct": correct,
        "workspace": trace.workspace,
        "used_generated_skill": trace.used_generated_skill,
        "saved_generated_skill": getattr(trace, "saved_generated_skill", None),
        "error": err,
    }


def _baseline_worker(task, tag: str) -> Dict[str, Any]:
    trace = run_baseline_model(task, PROJECT_ROOT, experiment_tag=tag)
    pred = trace.get("final_answer_normalized")
    gold = task.answer
    err = trace.get("error")
    correct = None
    if gold is not None and err is None:
        correct = answers_equal(task, pred, gold)
    return {
        "pid": task.sample_id,
        "question": task.question,
        "prediction": pred,
        "gold": gold,
        "correct": correct,
        "workspace": trace.get("workspace"),
        "used_generated_skill": None,
        "saved_generated_skill": None,
        "error": err,
    }


def _evo_worker(task, *, reuse: bool, save: bool, tag: str) -> Dict[str, Any]:
    agent = MultimodalMetaAgent(
        allow_generated_skill_reuse=reuse,
        allow_save_generated_skills=save,
        experiment_tag=tag,
    )
    trace = agent.solve(task)
    return _row_from_trace(task, trace)


def _run_parallel(tasks, fn, *, desc: str, workers: int) -> List[Dict[str, Any]]:
    rows: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        future_to_idx = {ex.submit(fn, task): idx for idx, task in enumerate(tasks)}
        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=desc):
            idx = future_to_idx[fut]
            rows[idx] = fut.result()
    return [r for r in rows if r is not None]


def _run_seed_install(seed_tasks, *, tag: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    agent = MultimodalMetaAgent(
        allow_generated_skill_reuse=True,
        allow_save_generated_skills=True,
        experiment_tag=tag,
    )
    for task in tqdm(seed_tasks, desc=tag):
        trace = agent.solve(task)
        rows.append(_row_from_trace(task, trace))
    return rows


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(1 for r in rows if r.get("gold") is not None and r.get("error") is None)
    correct = sum(1 for r in rows if r.get("gold") is not None and r.get("error") is None and bool(r.get("correct")))
    failures = sum(1 for r in rows if r.get("error") is not None)
    reused = sum(1 for r in rows if r.get("used_generated_skill") is not None)
    saved = sum(1 for r in rows if r.get("saved_generated_skill") is not None)
    return {
        "num_samples": len(rows),
        "num_scored": total,
        "num_correct": correct,
        "accuracy": (correct / total) if total else None,
        "num_failures": failures,
        "num_reused_generated_skills": reused,
        "num_saved_generated_skills": saved,
    }


def _rows_by_pid(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("pid")): r for r in rows}


def _is_unusable_evolved_row(row: Dict[str, Any]) -> bool:
    pred = row.get("prediction")
    err = row.get("error")
    if err:
        return True
    if pred in (None, ""):
        return True
    text = str(pred).strip().lower()
    return any(marker in text for marker in [
        "cannot determine", "unable to determine", "insufficient evidence",
        "need image", "recheck", "cannot answer"
    ])


def _merge_evolved_with_seeded_floor(rows_evolved_raw: List[Dict[str, Any]], rows_seeded_no_reuse: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seeded_map = _rows_by_pid(rows_seeded_no_reuse)
    merged: List[Dict[str, Any]] = []
    for row in rows_evolved_raw:
        pid = str(row.get("pid"))
        seeded = seeded_map.get(pid)
        if seeded is not None and _is_unusable_evolved_row(row):
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
    parser = argparse.ArgumentParser(description="Isolated open-model MathVista baseline/evolution runner.")
    parser.add_argument("--question-file", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--hf-split", default="testmini")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-alias", default=None, help="Optional tag like qwen25vl7b or internvl3_9b")
    args = parser.parse_args()

    tasks = load_tasks(args.question_file, args.image_root, args.hf_split, args.hf_cache_dir)
    tasks = tasks[args.offset:]
    seed_tasks = tasks[: args.seed_count]
    eval_tasks = tasks[args.seed_count : args.seed_count + args.eval_count]
    if not eval_tasks:
        raise SystemExit("No evaluation tasks selected. Increase dataset size or reduce seed-count/eval-count.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.model_alias}" if args.model_alias else ""
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_ROOT / f"open_model_compare_{timestamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": timestamp,
        "repo_root": str(PROJECT_ROOT),
        "hf_split": args.hf_split,
        "seed_count": args.seed_count,
        "eval_count": args.eval_count,
        "offset": args.offset,
        "workers": args.workers,
        "model_alias": args.model_alias,
        "env": {
            k: os.getenv(k)
            for k in [
                "BASE_URL", "MODEL", "API_KEY", "VISION_BASE_URL", "VISION_MODEL",
                "REASONING_BASE_URL", "REASONING_MODEL", "ORCHESTRATOR_BASE_URL", "ORCHESTRATOR_MODEL"
            ]
        }
    }
    save_json(out_dir / "manifest.json", manifest)

    original_backup = out_dir / "generated_original_backup"
    _copy_generated(GENERATED_DIR, original_backup)

    try:
        # baseline on eval tasks
        rows_baseline = _run_parallel(
            eval_tasks,
            lambda task: _baseline_worker(task, tag=f"baseline_model{suffix}"),
            desc="baseline_model",
            workers=args.workers,
        )
        save_jsonl(out_dir / "baseline_model.jsonl", rows_baseline)

        # seed install (serial, because it writes new skills)
        _safe_remove_generated_contents()
        rows_seed = _run_seed_install(seed_tasks, tag=f"seed_install{suffix}")
        save_jsonl(out_dir / "seed_install.jsonl", rows_seed)
        seeded_library = out_dir / "generated_after_seed"
        _copy_generated(GENERATED_DIR, seeded_library)

        # seeded-no-reuse (to preserve the exact semantics of final evolved floor)
        _restore_generated(seeded_library)
        rows_seeded_no_reuse = _run_parallel(
            eval_tasks,
            lambda task: _evo_worker(task, reuse=False, save=False, tag=f"eval_seeded_no_reuse{suffix}"),
            desc="eval_seeded_no_reuse",
            workers=args.workers,
        )
        save_jsonl(out_dir / "eval_seeded_no_reuse.jsonl", rows_seeded_no_reuse)

        # evolved raw
        _restore_generated(seeded_library)
        rows_evolved_raw = _run_parallel(
            eval_tasks,
            lambda task: _evo_worker(task, reuse=True, save=False, tag=f"eval_with_evolution{suffix}"),
            desc="eval_with_evolution_raw",
            workers=args.workers,
        )
        save_jsonl(out_dir / "eval_with_evolution_raw.jsonl", rows_evolved_raw)

        # evolved final (with gated seeded floor)
        rows_evolved = _merge_evolved_with_seeded_floor(rows_evolved_raw, rows_seeded_no_reuse)
        save_jsonl(out_dir / "eval_with_evolution.jsonl", rows_evolved)

        summary = {
            "config": {
                "hf_split": args.hf_split,
                "offset": args.offset,
                "seed_count": args.seed_count,
                "eval_count": args.eval_count,
                "workers": args.workers,
                "model_alias": args.model_alias,
            },
            "baseline_model": _summarize(rows_baseline),
            "seed_install": _summarize(rows_seed),
            "eval_seeded_no_reuse": _summarize(rows_seeded_no_reuse),
            "eval_with_evolution_raw": _summarize(rows_evolved_raw),
            "eval_with_evolution": _summarize(rows_evolved),
        }
        bacc = summary["baseline_model"]["accuracy"]
        eacc = summary["eval_with_evolution"]["accuracy"]
        eracc = summary["eval_with_evolution_raw"]["accuracy"]
        sacc = summary["eval_seeded_no_reuse"]["accuracy"]
        summary["delta_accuracy_evolved_minus_baseline_model"] = None if bacc is None or eacc is None else eacc - bacc
        summary["delta_accuracy_evolved_raw_minus_baseline_model"] = None if bacc is None or eracc is None else eracc - bacc
        summary["delta_accuracy_evolved_minus_seeded_no_reuse"] = None if sacc is None or eacc is None else eacc - sacc
        save_json(out_dir / "summary.json", summary)

        # Export reasoning details / comparison reports if available in the repo.
        try:
            export_reasoning_details(out_dir)
        except Exception as exc:
            save_json(out_dir / "reasoning_details_export_error.json", {"error": f"{type(exc).__name__}: {exc}"})
        try:
            rebuild_compare_outputs(out_dir)
        except Exception as exc:
            save_json(out_dir / "compare_reports_error.json", {"error": f"{type(exc).__name__}: {exc}"})

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved outputs to: {out_dir}")
    finally:
        _restore_generated(original_backup)


if __name__ == "__main__":
    main()
