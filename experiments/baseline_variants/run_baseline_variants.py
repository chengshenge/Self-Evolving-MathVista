#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from muse.answering import answers_equal, normalize_answer  # noqa: E402
from muse.baseline_model import SYSTEM_PROMPT as BASE_SYSTEM_PROMPT  # noqa: E402
from muse.baseline_model import _build_user_prompt  # noqa: E402
from muse.config import load_runtime_config  # noqa: E402
from muse.io_utils import materialize_hf_image, save_json, save_jsonl  # noqa: E402
from muse.llm_clients import OpenAIStyleClient  # noqa: E402
from muse.schemas import TaskPacket  # noqa: E402
from run_mathvista import load_tasks  # noqa: E402


RESULTS_ROOT = REPO_ROOT / "results"


def _load_tasks_with_cache_fallback(
    question_file: Optional[str],
    image_root: Optional[str],
    hf_split: Optional[str],
    hf_cache_dir: Optional[str],
) -> List[TaskPacket]:
    try:
        return load_tasks(question_file, image_root, hf_split, hf_cache_dir)
    except Exception:
        if question_file or not hf_split or not hf_cache_dir:
            raise
        cache_root = Path(hf_cache_dir) / "datasets" / "AI4Math___math_vista" / "default" / "0.0.0"
        versions = [p for p in cache_root.iterdir() if p.is_dir()] if cache_root.exists() else []
        if not versions:
            raise
        arrow_name = f"math_vista-{hf_split}.arrow"
        for version_dir in sorted(versions, key=lambda p: p.stat().st_mtime, reverse=True):
            arrow_path = version_dir / arrow_name
            if not arrow_path.exists():
                continue
            from datasets import Dataset

            rows = []
            image_dir = Path(hf_cache_dir) / "mathvista_cached_images" / hf_split
            for record in Dataset.from_file(str(arrow_path)):
                row = dict(record)
                row["image_path"] = materialize_hf_image(row, image_dir)
                rows.append(TaskPacket.from_dict(row))
            return rows
        raise


def _pick_model_config(config):
    if config.orchestrator_model and config.orchestrator_model.enabled:
        return config.orchestrator_model
    return config.reasoning_model


def _client() -> OpenAIStyleClient:
    cfg = load_runtime_config(REPO_ROOT)
    return OpenAIStyleClient(_pick_model_config(cfg))


def _collapse_steps(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _tokenize(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        parts: List[str] = []
        for v in value.values():
            parts.extend(_tokenize(v))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts = []
        for v in value:
            parts.extend(_tokenize(v))
        return parts
    return [t for t in re.findall(r"[a-z0-9]+", str(value).lower()) if len(t) > 1]


def _task_text_for_retrieval(task: TaskPacket) -> str:
    meta = task.metadata or {}
    return " ".join(
        str(x)
        for x in [
            task.question,
            task.question_type,
            task.answer_type,
            meta.get("context"),
            meta.get("task"),
            meta.get("category"),
            meta.get("source"),
            " ".join(str(s) for s in meta.get("skills", []) or []),
        ]
        if x
    )


def _call_direct_json(
    client: OpenAIStyleClient,
    task: TaskPacket,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    image_paths = task.existing_image_paths()
    if image_paths:
        raw = client.complete_multimodal_json(
            system_prompt,
            user_prompt,
            image_paths,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raw = client.complete_json(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return raw if isinstance(raw, dict) else {}


def _row(
    task: TaskPacket,
    *,
    variant: str,
    prediction_raw: Any,
    reasoning_steps: List[str],
    confidence: Any,
    workspace: Path,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = normalize_answer(task, prediction_raw)
    correct = answers_equal(task, normalized, task.answer) if task.answer is not None and error is None else None
    trace = {
        "variant": variant,
        "task": task.to_dict(),
        "reasoning_steps": reasoning_steps,
        "confidence": confidence,
        "final_answer_raw": prediction_raw,
        "final_answer_normalized": normalized,
        "correct": correct,
        "error": error,
        **(extra or {}),
    }
    workspace.mkdir(parents=True, exist_ok=True)
    save_json(workspace / "trace.json", trace)
    save_json(workspace / "task.json", task.to_dict())
    return {
        "pid": task.sample_id,
        "question": task.question,
        "prediction": normalized,
        "gold": task.answer,
        "correct": correct,
        "workspace": str(workspace),
        "variant": variant,
        "error": error,
        **(extra or {}),
    }


def _task_payload_without_gold(task: TaskPacket) -> Dict[str, Any]:
    payload = task.to_dict()
    payload.pop("answer", None)
    return payload


def _initial_payload_without_gold(row: Dict[str, Any]) -> Dict[str, Any]:
    blocked = {"gold", "correct"}
    payload = {k: v for k, v in row.items() if k not in blocked}
    task_payload = payload.get("task")
    if isinstance(task_payload, dict):
        task_payload = dict(task_payload)
        task_payload.pop("answer", None)
        payload["task"] = task_payload
    return payload


def _run_base_call(task: TaskPacket, *, variant: str, workspace: Path, temperature: Optional[float] = None, extra_prompt: str = "") -> Dict[str, Any]:
    client = _client()
    user_prompt = _build_user_prompt(task)
    if extra_prompt:
        user_prompt = f"{extra_prompt}\n\n{user_prompt}"
    try:
        data = _call_direct_json(
            client,
            task,
            system_prompt=BASE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1600,
            temperature=temperature,
        )
        return _row(
            task,
            variant=variant,
            prediction_raw=data.get("answer"),
            reasoning_steps=_collapse_steps(data.get("reasoning_steps")),
            confidence=data.get("confidence"),
            workspace=workspace,
        )
    except Exception as exc:
        return _row(
            task,
            variant=variant,
            prediction_raw=None,
            reasoning_steps=[],
            confidence=None,
            workspace=workspace,
            error=f"{type(exc).__name__}: {exc}",
        )


def _vote_answer(task: TaskPacket, attempts: List[Dict[str, Any]]) -> tuple[Any, Any]:
    valid = [a for a in attempts if a.get("normalized") not in (None, "")]
    if not valid:
        return None, None
    counts = Counter(str(a["normalized"]) for a in valid)
    best_count = max(counts.values())
    tied = [answer for answer, count in counts.items() if count == best_count]
    if len(tied) == 1:
        chosen = tied[0]
    else:
        def score(answer: str) -> float:
            vals = []
            for item in valid:
                if str(item.get("normalized")) == answer:
                    try:
                        vals.append(float(item.get("confidence") or 0.0))
                    except Exception:
                        vals.append(0.0)
            return sum(vals) / max(1, len(vals))
        chosen = max(tied, key=score)
    raw = next((a.get("raw") for a in valid if str(a.get("normalized")) == chosen), chosen)
    return raw, chosen


def run_rollout5(task: TaskPacket, *, out_dir: Path, rollouts: int, temperature: float) -> Dict[str, Any]:
    workspace = out_dir / "workspaces" / f"{task.sample_id}_base_rollout{rollouts}"
    attempts = []
    for idx in range(rollouts):
        row = _run_base_call(
            task,
            variant=f"base_rollout_attempt_{idx + 1}",
            workspace=workspace / f"attempt_{idx + 1}",
            temperature=temperature,
        )
        attempts.append({
            "idx": idx + 1,
            "raw": row.get("prediction"),
            "normalized": row.get("prediction"),
            "confidence": None,
            "error": row.get("error"),
            "workspace": row.get("workspace"),
        })
    raw_answer, _ = _vote_answer(task, attempts)
    return _row(
        task,
        variant=f"base_rollout{rollouts}",
        prediction_raw=raw_answer,
        reasoning_steps=[f"Majority vote over {rollouts} direct baseline rollouts."],
        confidence=None,
        workspace=workspace,
        extra={"rollouts": attempts},
    )


REFLECTION_SYSTEM_PROMPT = """You are a reflection baseline for a multimodal QA model.
You receive the original task and the model's initial direct answer.
Critique only that answer, then either keep it or revise it.
Do not use external memory, generated skills, tools, or hidden gold labels.
Return JSON only with keys:
- critique: short explanation
- revised_answer: final answer
- confidence: number between 0 and 1
"""


def run_reflection1(task: TaskPacket, *, out_dir: Path) -> Dict[str, Any]:
    workspace = out_dir / "workspaces" / f"{task.sample_id}_base_reflection1"
    first = _run_base_call(task, variant="base_reflection_initial", workspace=workspace / "initial")
    if first.get("error"):
        return _row(
            task,
            variant="base_reflection1",
            prediction_raw=None,
            reasoning_steps=[],
            confidence=None,
            workspace=workspace,
            error=first.get("error"),
            extra={"initial": first},
        )

    client = _client()
    prompt = (
        f"Task:\n{json.dumps(_task_payload_without_gold(task), ensure_ascii=False)}\n\n"
        f"Initial answer:\n{json.dumps(_initial_payload_without_gold(first), ensure_ascii=False)}\n\n"
        "Return JSON only."
    )
    try:
        data = _call_direct_json(
            client,
            task,
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=1000,
            temperature=0.0,
        )
        return _row(
            task,
            variant="base_reflection1",
            prediction_raw=data.get("revised_answer", data.get("answer")),
            reasoning_steps=_collapse_steps(data.get("critique")),
            confidence=data.get("confidence"),
            workspace=workspace,
            extra={"initial": first},
        )
    except Exception as exc:
        return _row(
            task,
            variant="base_reflection1",
            prediction_raw=first.get("prediction"),
            reasoning_steps=["Reflection call failed; falling back to initial direct answer."],
            confidence=None,
            workspace=workspace,
            error=None,
            extra={"initial": first, "reflection_error": f"{type(exc).__name__}: {exc}"},
        )


def _memory_strategy(task: TaskPacket) -> str:
    meta = task.metadata or {}
    context = str(meta.get("context") or "").lower()
    q = str(task.question or "").lower()
    if "geometry" in context:
        return "Extract given labels and relations first, then solve algebraically; verify the final choice against all options."
    if "chart" in context or "plot" in context:
        return "Read axes, labels, legends, and exact plotted values before computing; avoid estimating when a value is explicitly marked."
    if "synthetic scene" in context or "subtract all" in q:
        return "Count total objects and each descriptor-specific set separately, then compute remaining objects carefully."
    if "table" in context:
        return "Use table headers and row/column alignment; compute only from the relevant cells."
    return "Identify the decisive visual facts and match the final answer to the requested answer type."


def build_memory(seed_tasks: List[TaskPacket], *, out_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    memory = []
    for task in seed_tasks:
        row = {
            "pid": task.sample_id,
            "question": task.question,
            "gold": task.answer,
            "metadata": task.metadata,
            "answer_type": task.answer_type,
            "question_type": task.question_type,
            "memory_text": _memory_strategy(task),
            "retrieval_tokens": _tokenize(_task_text_for_retrieval(task)),
        }
        rows.append(row)
        memory.append({
            "pid": task.sample_id,
            "text": (
                f"Seed task context={task.metadata.get('context') if task.metadata else None}; "
                f"task={task.metadata.get('task') if task.metadata else None}; "
                f"answer_type={task.answer_type}. Strategy: {_memory_strategy(task)}"
            ),
            "tokens": row["retrieval_tokens"],
        })
    save_jsonl(out_dir / "memory_seed.jsonl", rows)
    save_json(out_dir / "memory_store.json", memory)
    return memory


def retrieve_memory(task: TaskPacket, memory: List[Dict[str, Any]], *, top_k: int) -> List[Dict[str, Any]]:
    query_tokens = set(_tokenize(_task_text_for_retrieval(task)))
    scored = []
    for item in memory:
        tokens = set(item.get("tokens") or [])
        overlap = len(query_tokens & tokens)
        scored.append((overlap, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k] if score > 0]


def run_memory_topk(task: TaskPacket, *, out_dir: Path, memory: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    retrieved = retrieve_memory(task, memory, top_k=top_k)
    memory_text = "\n".join(f"- {item['text']}" for item in retrieved)
    extra_prompt = (
        "Relevant textual memories from seed tasks. Use these only as high-level strategy hints; "
        "do not copy any seed answer.\n"
        f"{memory_text if memory_text else '- No relevant memory retrieved.'}"
    )
    row = _run_base_call(
        task,
        variant=f"base_memory_top{top_k}",
        workspace=out_dir / "workspaces" / f"{task.sample_id}_base_memory_top{top_k}",
        temperature=0.0,
        extra_prompt=extra_prompt,
    )
    row["retrieved_memory_pids"] = [item["pid"] for item in retrieved]
    return row


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = sum(1 for r in rows if r.get("gold") is not None and r.get("error") is None)
    correct = sum(1 for r in rows if r.get("gold") is not None and r.get("error") is None and bool(r.get("correct")))
    failures = sum(1 for r in rows if r.get("error") is not None)
    return {
        "num_samples": len(rows),
        "num_scored": total,
        "num_correct": correct,
        "accuracy": (correct / total) if total else None,
        "num_failures": failures,
    }


def _run_incremental(
    name: str,
    path: Path,
    tasks: List[TaskPacket],
    fn: Callable[[TaskPacket], Dict[str, Any]],
    *,
    workers: int,
) -> List[Dict[str, Any]]:
    existing = _read_jsonl(path)
    done = {str(r.get("pid")) for r in existing}
    rows_by_pid = {str(r.get("pid")): r for r in existing if r.get("pid") is not None}
    pending = [task for task in tasks if str(task.sample_id) not in done]
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            future_to_task = {ex.submit(fn, task): task for task in pending}
            for fut in tqdm(as_completed(future_to_task), total=len(future_to_task), desc=name):
                task = future_to_task[fut]
                row = fut.result()
                rows_by_pid[str(task.sample_id)] = row
                _append_jsonl(path, row)
    ordered = [rows_by_pid[str(task.sample_id)] for task in tasks if str(task.sample_id) in rows_by_pid]
    save_jsonl(path, ordered)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct-baseline comparison variants on MathVista eval tasks.")
    parser.add_argument("--question-file", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--hf-split", default="testmini")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--memory-seed-count", type=int, default=20)
    parser.add_argument("--eval-count", type=int, default=980)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--rollout-temperature", type=float, default=0.7)
    parser.add_argument("--memory-top-k", type=int, default=3)
    parser.add_argument("--variants", default="rollout,reflection,memory")
    parser.add_argument("--model-alias", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    tasks = _load_tasks_with_cache_fallback(args.question_file, args.image_root, args.hf_split, args.hf_cache_dir)
    tasks = tasks[args.offset:]
    seed_tasks = tasks[: args.memory_seed_count]
    eval_tasks = tasks[args.memory_seed_count : args.memory_seed_count + args.eval_count]
    if not eval_tasks:
        raise SystemExit("No evaluation tasks selected.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.model_alias}" if args.model_alias else ""
    out_dir = Path(args.output_dir).resolve() if args.output_dir else RESULTS_ROOT / f"baseline_variants_{timestamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    manifest = {
        "created_at": timestamp,
        "hf_split": args.hf_split,
        "offset": args.offset,
        "memory_seed_count": args.memory_seed_count,
        "eval_count": args.eval_count,
        "workers": args.workers,
        "rollouts": args.rollouts,
        "memory_top_k": args.memory_top_k,
        "variants": sorted(variants),
        "model_alias": args.model_alias,
        "env": {k: os.getenv(k) for k in ["BASE_URL", "MODEL", "VISION_MODEL", "ORCHESTRATOR_MODEL"]},
    }
    save_json(out_dir / "manifest.json", manifest)

    summary: Dict[str, Any] = {"config": manifest}
    memory: List[Dict[str, Any]] = []
    if "memory" in variants:
        memory = build_memory(seed_tasks, out_dir=out_dir)

    if "rollout" in variants:
        rows = _run_incremental(
            f"base_rollout{args.rollouts}",
            out_dir / f"base_rollout{args.rollouts}.jsonl",
            eval_tasks,
            lambda task: run_rollout5(task, out_dir=out_dir, rollouts=args.rollouts, temperature=args.rollout_temperature),
            workers=args.workers,
        )
        summary[f"base_rollout{args.rollouts}"] = _summarize(rows)

    if "reflection" in variants:
        rows = _run_incremental(
            "base_reflection1",
            out_dir / "base_reflection1.jsonl",
            eval_tasks,
            lambda task: run_reflection1(task, out_dir=out_dir),
            workers=args.workers,
        )
        summary["base_reflection1"] = _summarize(rows)

    if "memory" in variants:
        rows = _run_incremental(
            f"base_memory_top{args.memory_top_k}",
            out_dir / f"base_memory_top{args.memory_top_k}.jsonl",
            eval_tasks,
            lambda task: run_memory_topk(task, out_dir=out_dir, memory=memory, top_k=args.memory_top_k),
            workers=args.workers,
        )
        summary[f"base_memory_top{args.memory_top_k}"] = _summarize(rows)

    save_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved baseline variant outputs to: {out_dir}")


if __name__ == "__main__":
    main()
