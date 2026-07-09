from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

import os
from pathlib import Path
try:
    from dotenv import dotenv_values
    for _k, _v in dotenv_values(Path(__file__).resolve().parent / ".env").items():
        if _v is not None and os.getenv(_k) is None:
            os.environ[_k] = _v
except Exception:
    pass

from muse.answering import answers_equal
from muse.baseline_model import run_baseline_model
from muse.compare_reports import rebuild_compare_outputs
from muse.config import load_runtime_config
from muse.io_utils import save_json, save_jsonl
from muse.orchestrator import MultimodalMetaAgent
from run_mathvista import load_tasks

from muse.compare_exports import export_reasoning_details

PROJECT_ROOT = Path(__file__).resolve().parent
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


def _run_tasks(tasks, *, reuse: bool, save: bool, tag: str) -> List[Dict]:
    agent = MultimodalMetaAgent(
        allow_generated_skill_reuse=reuse,
        allow_save_generated_skills=save,
        experiment_tag=tag,
    )
    rows: List[Dict] = []
    for task in tqdm(tasks, desc=tag):
        trace = agent.solve(task)
        rows.append(
            {
                "pid": task.sample_id,
                "question": task.question,
                "prediction": trace.final_answer_normalized,
                "gold": task.answer,
                "correct": (answers_equal(task, trace.final_answer_normalized, task.answer) if task.answer is not None and trace.error is None else (None if task.answer is None else False if trace.error is None else None)),
                "workspace": trace.workspace,
                "used_generated_skill": trace.used_generated_skill,
                "saved_generated_skill": getattr(trace, 'saved_generated_skill', None),
                "error": trace.error,
            }
        )
    return rows


def _run_baseline(tasks, *, tag: str) -> List[Dict]:
    rows: List[Dict] = []
    for task in tqdm(tasks, desc=tag):
        trace = run_baseline_model(task, PROJECT_ROOT, experiment_tag=tag)
        rows.append(
            {
                "pid": task.sample_id,
                "question": task.question,
                "prediction": trace.get("final_answer_normalized"),
                "gold": task.answer,
                "correct": (answers_equal(task, trace.get("final_answer_normalized"), task.answer) if task.answer is not None and trace.get("error") is None else (None if task.answer is None else False if trace.get("error") is None else None)),
                "workspace": trace.get("workspace"),
                "used_generated_skill": None,
                "saved_generated_skill": None,
                "error": trace.get("error"),
            }
        )
    return rows


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
    parser = argparse.ArgumentParser(description="Compare MathVista accuracy across baseline / no-evolution / seeded / evolved settings.")
    parser.add_argument("--question-file", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--hf-split", default="testmini")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--eval-count", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    tasks = load_tasks(args.question_file, args.image_root, args.hf_split, args.hf_cache_dir)
    tasks = tasks[args.offset:]
    seed_tasks = tasks[: args.seed_count]
    eval_tasks = tasks[args.seed_count : args.seed_count + args.eval_count]
    if not eval_tasks:
        raise SystemExit("No evaluation tasks selected. Increase dataset size or reduce seed-count/eval-count.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_ROOT / f"ab_compare_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    original_backup = out_dir / "generated_original_backup"
    _copy_generated(GENERATED_DIR, original_backup)

    try:
        rows_baseline = _run_baseline(eval_tasks, tag="baseline_model")
        save_jsonl(out_dir / "baseline_model.jsonl", rows_baseline)
        summary_baseline = _summarize(rows_baseline)

        _safe_remove_generated_contents()
        rows_no_evo = _run_tasks(eval_tasks, reuse=False, save=False, tag="eval_no_evolution")
        save_jsonl(out_dir / "eval_no_evolution.jsonl", rows_no_evo)
        summary_no_evo = _summarize(rows_no_evo)

        _safe_remove_generated_contents()
        rows_seed = _run_tasks(seed_tasks, reuse=True, save=True, tag="seed_install")
        save_jsonl(out_dir / "seed_install.jsonl", rows_seed)
        seeded_library = out_dir / "generated_after_seed"
        _copy_generated(GENERATED_DIR, seeded_library)

        _restore_generated(seeded_library)
        rows_seeded_no_reuse = _run_tasks(eval_tasks, reuse=False, save=False, tag="eval_seeded_no_reuse")
        save_jsonl(out_dir / "eval_seeded_no_reuse.jsonl", rows_seeded_no_reuse)
        summary_seeded_no_reuse = _summarize(rows_seeded_no_reuse)

        _restore_generated(seeded_library)
        rows_evolved_raw = _run_tasks(eval_tasks, reuse=True, save=False, tag="eval_with_evolution")
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
            },
            "baseline_model": summary_baseline,
            "seed_install": _summarize(rows_seed),
            "eval_no_evolution": summary_no_evo,
            "eval_seeded_no_reuse": summary_seeded_no_reuse,
            "eval_with_evolution": summary_evolved,
            "delta_accuracy_evolved_minus_baseline_model": None if summary_baseline["accuracy"] is None or summary_evolved["accuracy"] is None else summary_evolved["accuracy"] - summary_baseline["accuracy"],
            "delta_accuracy_evolved_minus_no_evolution": None if summary_no_evo["accuracy"] is None or summary_evolved["accuracy"] is None else summary_evolved["accuracy"] - summary_no_evo["accuracy"],
            "delta_accuracy_evolved_minus_seeded_no_reuse": None if summary_seeded_no_reuse["accuracy"] is None or summary_evolved["accuracy"] is None else summary_evolved["accuracy"] - summary_seeded_no_reuse["accuracy"],
        }
        save_json(out_dir / "summary.json", summary)
        export_reasoning_details(out_dir)
        rebuild_compare_outputs(out_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Saved A/B comparison outputs to: {out_dir}")
    finally:
        _restore_generated(original_backup)


if __name__ == "__main__":
    main()
