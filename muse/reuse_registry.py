from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .generated_paths import resolved_generated_skills_root

from .schemas import TaskPacket


REGISTRY_FILENAME = "_reuse_registry.json"


def _registry_path(project_root: Optional[str | Path] = None) -> Path:
    base = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    path = resolved_generated_skills_root(base) / REGISTRY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _bucket_key(task: TaskPacket) -> str:
    meta = task.metadata or {}
    context = str(meta.get("context") or "unknown")
    task_name = str(meta.get("task") or "unknown")
    return "|".join([
        context.strip().lower(),
        task_name.strip().lower(),
        str(task.question_type).strip().lower(),
        str(task.answer_type).strip().lower(),
    ])


def _empty_counter() -> Dict[str, int]:
    return {"success": 0, "total": 0}


def load_registry(project_root: Optional[str | Path] = None) -> Dict[str, Any]:
    path = _registry_path(project_root)
    if not path.exists():
        return {"skills": {}, "updated_at": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"skills": {}, "updated_at": None}


def save_registry(data: Dict[str, Any], project_root: Optional[str | Path] = None) -> None:
    path = _registry_path(project_root)
    data = dict(data)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_skill_stats(project_root: Optional[str | Path], skill_name: str, task: Optional[TaskPacket] = None) -> Dict[str, Any]:
    registry = load_registry(project_root)
    skills = registry.get("skills", {})
    info = skills.get(skill_name, {})
    global_stats = info.get("global", _empty_counter())

    result: Dict[str, Any] = {
        "global_success": int(global_stats.get("success", 0)),
        "global_total": int(global_stats.get("total", 0)),
        "global_accuracy": 0.0,
        "bucket_key": None,
        "bucket_success": 0,
        "bucket_total": 0,
        "bucket_accuracy": 0.0,
    }
    if result["global_total"] > 0:
        result["global_accuracy"] = result["global_success"] / result["global_total"]

    if task is not None:
        bucket_key = _bucket_key(task)
        bucket_stats = info.get("buckets", {}).get(bucket_key, _empty_counter())
        result.update(
            {
                "bucket_key": bucket_key,
                "bucket_success": int(bucket_stats.get("success", 0)),
                "bucket_total": int(bucket_stats.get("total", 0)),
            }
        )
        if result["bucket_total"] > 0:
            result["bucket_accuracy"] = result["bucket_success"] / result["bucket_total"]

    return result


def record_skill_outcome(project_root: Optional[str | Path], skill_name: str, task: TaskPacket, correct: bool) -> None:
    registry = load_registry(project_root)
    skills = registry.setdefault("skills", {})
    info = skills.setdefault(skill_name, {"global": _empty_counter(), "buckets": {}})

    global_stats = info.setdefault("global", _empty_counter())
    global_stats["total"] = int(global_stats.get("total", 0)) + 1
    global_stats["success"] = int(global_stats.get("success", 0)) + int(bool(correct))

    bucket_key = _bucket_key(task)
    bucket_stats = info.setdefault("buckets", {}).setdefault(bucket_key, _empty_counter())
    bucket_stats["total"] = int(bucket_stats.get("total", 0)) + 1
    bucket_stats["success"] = int(bucket_stats.get("success", 0)) + int(bool(correct))

    save_registry(registry, project_root)
