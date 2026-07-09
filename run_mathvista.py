from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

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

from muse.io_utils import load_jsonl, materialize_hf_image, save_jsonl, task_from_mathvista_record
from muse.answering import answers_equal
from muse.orchestrator import MultimodalMetaAgent
from muse.schemas import TaskPacket


def _load_from_hf(split: str, cache_dir: str | None = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("AI4Math/MathVista", split=split, cache_dir=cache_dir)
    rows: List[Dict[str, Any]] = []
    temp_img_root = Path(cache_dir or ".cache_mathvista_images")
    for record in ds:
        row = dict(record)
        row["image_path"] = materialize_hf_image(row, temp_img_root / split)
        rows.append(row)
    return rows


def load_tasks(question_file: str | None, image_root: str | None, hf_split: str | None, hf_cache_dir: str | None) -> List[TaskPacket]:
    if not question_file and not hf_split:
        raise SystemExit("Provide either --question-file or --hf-split.")

    tasks: List[TaskPacket] = []
    if hf_split:
        rows = _load_from_hf(hf_split, hf_cache_dir)
        tasks.extend(TaskPacket.from_dict(row) for row in rows)
    if question_file:
        rows = load_jsonl(question_file)
        tasks.extend(task_from_mathvista_record(row, image_root) for row in rows)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MUSE multimodal skill-evolution pipeline on MathVista-style data.")
    parser.add_argument("--question-file", default=None, help="Local JSONL in MathVista format.")
    parser.add_argument("--image-root", default=None, help="Optional image root for local JSONL.")
    parser.add_argument("--hf-split", default=None, help="Optional Hugging Face split, e.g. testmini.")
    parser.add_argument("--hf-cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--offset", type=int, default=0, help="Start index after loading tasks.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for fast debugging.")
    parser.add_argument("--disable-reuse", action="store_true", help="Disable generated-skill reuse.")
    parser.add_argument("--disable-save", action="store_true", help="Disable saving new generated skills.")
    parser.add_argument("--experiment-tag", default=None, help="Optional tag recorded in traces/logs.")
    parser.add_argument("--output", default="results/mathvista_predictions.jsonl", help="Output JSONL path.")
    args = parser.parse_args()

    tasks = load_tasks(args.question_file, args.image_root, args.hf_split, args.hf_cache_dir)
    if args.offset:
        tasks = tasks[args.offset:]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    agent = MultimodalMetaAgent(
        allow_generated_skill_reuse=not args.disable_reuse,
        allow_save_generated_skills=not args.disable_save,
        experiment_tag=args.experiment_tag,
    )
    outputs: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    failures = 0

    for task in tqdm(tasks, desc="Running MathVista"):
        trace = agent.solve(task)
        row_correct = (
            answers_equal(task, trace.final_answer_normalized, task.answer)
            if task.answer is not None and trace.error is None
            else None
        )
        row = {
            "pid": task.sample_id,
            "question": task.question,
            "prediction": trace.final_answer_normalized,
            "gold": task.answer,
            "correct": row_correct,
            "workspace": trace.workspace,
            "used_generated_skill": trace.used_generated_skill,
            "error": trace.error,
        }
        outputs.append(row)
        if trace.error:
            failures += 1
        if task.answer is not None and trace.error is None:
            total += 1
            correct += int(bool(row_correct))

    save_jsonl(args.output, outputs)

    if total:
        print(f"Accuracy: {correct}/{total} = {correct / total:.3f}")
    if failures:
        print(f"Failures: {failures}")
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
