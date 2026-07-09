from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from muse.answering import answers_equal, normalize_answer  # noqa: E402
from muse.config import load_runtime_config  # noqa: E402
from muse.io_utils import save_json, save_jsonl  # noqa: E402
from muse.llm_clients import OpenAIStyleClient  # noqa: E402
from muse.orchestrator import MultimodalMetaAgent  # noqa: E402
from muse.schemas import TaskPacket  # noqa: E402
from run_compare_mathvista_parallel import _row_from_trace, _worker_run_task  # noqa: E402
from run_mathvista import load_tasks  # noqa: E402


OUTCOME_VERIFIER_PROMPT = """You are an independent outcome-only verifier for multimodal QA candidates.
You must judge whether the final answer is acceptable from only the task, image, final answer, normalized answer, and a very short rationale.
Do not use gold answers. Do not inspect step-level trace evidence.
Return JSON only:
{"keep": true|false, "score": float in [0,1], "reason": "..."}"""

STEP_VERIFIER_PROMPT = """You are an independent step-level verifier for multimodal QA candidates.
Judge whether the final answer is supported by the task, image, visual facts, focus answers, math reasoning summary, and verifier issues.
Do not use gold answers. Do not use the actor self-critique.
Return JSON only:
{"keep": true|false, "score": float in [0,1], "reason": "..."}"""

SELF_CRITIQUE_PROMPT = """You are the original actor model self-critiquing its own multimodal QA candidate.
Use only the task, image, final answer, normalized answer, and candidate trace summary.
Do not use gold answers. Do not use any external verifier conclusion.
Return JSON only:
{"keep": true|false, "score": float in [0,1], "reason": "..."}"""


def load_env(repo_root: Path = REPO_ROOT) -> None:
    env_paths: List[Path] = []
    if os.getenv("MUSE_ENV_FILE"):
        env_paths.append(Path(os.environ["MUSE_ENV_FILE"]))
    env_paths.append(repo_root / ".env")
    for env_path in env_paths:
        if not env_path.exists():
            continue
        override = os.getenv("MUSE_ENV_FILE") and env_path == Path(os.environ["MUSE_ENV_FILE"])
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if override or key not in os.environ:
                os.environ[key] = value
        break
    base = os.getenv("BASE_URL")
    key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL")
    if base:
        os.environ.setdefault("VISION_BASE_URL", base)
    if key:
        os.environ.setdefault("VISION_API_KEY", key)
    if model:
        os.environ.setdefault("VISION_MODEL", model)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def task_without_gold(task: TaskPacket | Dict[str, Any]) -> Dict[str, Any]:
    payload = task.to_dict() if hasattr(task, "to_dict") else dict(task)
    payload.pop("answer", None)
    return payload


def load_mathvista_tasks(hf_split: str, hf_cache_dir: Optional[str], offset: int, count: int) -> List[TaskPacket]:
    load_env()
    tasks = load_tasks(None, None, hf_split, hf_cache_dir)
    return tasks[offset : offset + count]


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if r.get("error") is None and r.get("correct") is not None]
    correct = sum(1 for r in scored if r.get("correct") is True)
    failures = sum(1 for r in rows if r.get("error") is not None)
    reused = sum(1 for r in rows if r.get("used_generated_skill"))
    return {
        "num_samples": len(rows),
        "num_scored": len(scored),
        "num_correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "num_failures": failures,
        "num_reused_generated_skills": reused,
    }


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def is_numeric_task(task: Dict[str, Any]) -> bool:
    return str(task.get("answer_type") or "").lower() in {"integer", "float", "number", "numeric"}


def reward_for_candidate(candidate: Dict[str, Any], *, tau: float) -> tuple[float, str]:
    task = candidate.get("task") or {}
    pred = candidate.get("final_answer_normalized")
    gold = candidate.get("gold_answer")
    if is_numeric_task(task):
        pred_num = safe_float(pred)
        gold_num = safe_float(gold)
        if pred_num is None or gold_num is None:
            return (0.0, "numeric_parse_failed")
        denom = abs(gold_num) + tau
        reward = max(0.0, 1.0 - abs(pred_num - gold_num) / denom)
        return (float(min(1.0, reward)), "numeric_partial_reward")

    task_obj = TaskPacket.from_dict(task)
    try:
        correct = answers_equal(task_obj, pred, gold)
    except Exception:
        correct = str(pred).strip().lower() == str(gold).strip().lower()
    return (1.0 if correct else 0.0, "binary_exact_match")


def threshold_for_candidate(candidate: Dict[str, Any], explicit_threshold: Optional[float]) -> float:
    if explicit_threshold is not None:
        return explicit_threshold
    return 0.95 if is_numeric_task(candidate.get("task") or {}) else 1.0


def corrupt_reward(candidate: Dict[str, Any], reward: float, *, flip_prob: float, threshold: float, rng: random.Random) -> tuple[float, bool]:
    binary = reward >= threshold
    flipped = rng.random() < flip_prob
    if flipped:
        binary = not binary
    return (1.0 if binary else 0.0, flipped)


def llm_score_candidate(candidate: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
    load_env()
    cfg = load_runtime_config(REPO_ROOT)
    client = OpenAIStyleClient(cfg.orchestrator_model if cfg.orchestrator_model.enabled else cfg.reasoning_model)
    task = TaskPacket.from_dict(candidate.get("task") or {})
    short_rationale = short_math_rationale(candidate)
    if mode == "self_critique":
        system = SELF_CRITIQUE_PROMPT
        payload = {
            "task": candidate.get("task_without_gold") or task_without_gold(task),
            "final_answer_raw": candidate.get("final_answer_raw"),
            "final_answer_normalized": candidate.get("final_answer_normalized"),
            "candidate_trace_summary": {
                "visual_facts": candidate.get("visual_facts", [])[:5],
                "math_rounds_summary": candidate.get("math_rounds_summary", [])[:2],
                "verify_rounds_summary": candidate.get("verify_rounds_summary", [])[:2],
                "short_rationale": short_rationale,
            },
        }
    elif mode == "verifier_outcome":
        system = OUTCOME_VERIFIER_PROMPT
        payload = {
            "task": candidate.get("task_without_gold") or task_without_gold(task),
            "final_answer_raw": candidate.get("final_answer_raw"),
            "final_answer_normalized": candidate.get("final_answer_normalized"),
            "very_short_rationale": short_rationale,
        }
    elif mode == "verifier_step":
        system = STEP_VERIFIER_PROMPT
        payload = {
            "task": candidate.get("task_without_gold") or task_without_gold(task),
            "final_answer_raw": candidate.get("final_answer_raw"),
            "final_answer_normalized": candidate.get("final_answer_normalized"),
            "visual_facts": candidate.get("visual_facts", [])[:10],
            "focus_answers": candidate.get("focus_answers", [])[:8],
            "math_rounds_summary": candidate.get("math_rounds_summary", [])[:3],
            "verify_rounds_summary": candidate.get("verify_rounds_summary", [])[:3],
        }
    else:
        raise ValueError(mode)

    image_paths = task.existing_image_paths()
    try:
        if image_paths:
            raw = client.complete_multimodal_json(system, json.dumps(payload, ensure_ascii=False), image_paths, max_tokens=800, temperature=0.0)
        else:
            raw = client.complete_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=800, temperature=0.0)
    except Exception as exc:
        return {"keep": False, "score": 0.0, "reason": f"{type(exc).__name__}: {exc}"}
    if not isinstance(raw, dict):
        raw = {}
    score = raw.get("score", 0.0)
    try:
        score = max(0.0, min(1.0, float(score)))
    except Exception:
        score = 0.0
    keep = bool(raw.get("keep", score >= 0.5))
    return {"keep": keep, "score": score, "reason": str(raw.get("reason") or "")[:1200]}


def short_math_rationale(candidate: Dict[str, Any]) -> str:
    rounds = candidate.get("math_rounds_summary") or []
    if not rounds:
        return ""
    last = rounds[-1] if isinstance(rounds[-1], dict) else {}
    steps = last.get("reasoning_steps") or []
    if isinstance(steps, list):
        return " ".join(str(x) for x in steps[:3])[:1000]
    return str(steps)[:1000]


def trace_to_candidate(trace: Any, task: TaskPacket, profile: Dict[str, Any]) -> Dict[str, Any]:
    data = trace.to_dict()
    evidence = data.get("evidence") or {}
    return {
        "pid": task.sample_id,
        "sample_id": task.sample_id,
        "task": task.to_dict(),
        "task_without_gold": task_without_gold(task),
        "gold_answer": task.answer,
        "trace_correct": data.get("correct"),
        "final_answer_raw": data.get("final_answer_raw"),
        "final_answer_normalized": data.get("final_answer_normalized"),
        "visual_facts": evidence.get("visual_facts") or [],
        "focus_answers": evidence.get("focus_answers") or [],
        "math_rounds_summary": data.get("math_rounds") or [],
        "verify_rounds_summary": data.get("verify_rounds") or [],
        "candidate_profile": profile,
        "workspace": data.get("workspace"),
        "backbone_model": os.getenv("MODEL") or os.getenv("DEFAULT_MODEL") or "",
        "multimodal_setting": "current_full",
        "error": data.get("error"),
    }


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_library_to_generated(library_dir: Path, generated_root: Path) -> None:
    clear_directory(generated_root)
    if not library_dir.exists():
        return
    for child in library_dir.iterdir():
        target = generated_root / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def run_eval_task(task_dict: Dict[str, Any], generated_root: str, tag: str) -> Dict[str, Any]:
    load_env()
    os.environ["MUSE_GENERATED_SKILLS_ROOT"] = generated_root
    return _worker_run_task(task_dict, True, False, tag, True)


def slice_name(task: Dict[str, Any]) -> Optional[str]:
    q = str(task.get("question") or "").lower()
    meta = task.get("metadata") or {}
    context = str(meta.get("context") or "").lower()
    source = str(meta.get("source") or "").lower()
    task_name = str(meta.get("task") or "").lower()
    if "age gap" in q or "born after the end of world war ii" in q or ("kvqa" in source and ("age" in q or "born" in q)):
        return "identity_age_gap"
    if context == "geometry diagram" or "unigeo" in source or "geometry" in task_name:
        return "geometry"
    if context == "bar chart" or "chartqa" in source or "bar chart" in q:
        return "bar_chart"
    if "clevr" in source or "synthetic scene" in context or ("subtract all" in q):
        return "synthetic_counting"
    return None


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def ranks(values: Sequence[float]) -> List[float]:
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        out = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                out[indexed[k][0]] = rank
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)
