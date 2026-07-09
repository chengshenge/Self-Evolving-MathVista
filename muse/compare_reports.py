from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from muse.answering import answers_equal
from muse.schemas import TaskPacket

BRANCH_FILES = {
    "baseline_model": "baseline_model.jsonl",
    "seed_install": "seed_install.jsonl",
    "eval_no_evolution": "eval_no_evolution.jsonl",
    "eval_seeded_no_reuse": "eval_seeded_no_reuse.jsonl",
    "eval_with_evolution": "eval_with_evolution.jsonl",
}

BRANCH_NAMES = list(BRANCH_FILES.keys())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            writer = csv.writer(f)
            writer.writerow(fieldnames or ["empty"])
            return
        keys = fieldnames or list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _reasoning_files(compare_dir: Path) -> Iterable[Path]:
    rd = compare_dir / "reasoning_details"
    if not rd.exists():
        return []
    return sorted(rd.glob("PID*/reasoning_details.json"))


def _get_branch_blob(data: Dict[str, Any], branch_name: str) -> Dict[str, Any]:
    branches = data.get("branches")
    if isinstance(branches, dict):
        return branches.get(branch_name, {}) or {}
    return data.get(branch_name, {}) or {}


def _resolve_workspace(compare_dir: Path, workspace_value: Any) -> Optional[Path]:
    if not workspace_value:
        return None
    raw = Path(str(workspace_value))
    # Prefer packaged workspace in compare_dir/workspaces/<basename>
    packaged = compare_dir / "workspaces" / raw.name
    if packaged.exists():
        return packaged
    if raw.exists():
        return raw
    return None


def _load_trace(compare_dir: Path, branch_blob: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    trace = branch_blob.get("trace")
    if isinstance(trace, dict) and trace:
        return trace

    trace_path = branch_blob.get("trace_path")
    if trace_path:
        tp = Path(str(trace_path))
        packaged_tp = compare_dir / "workspaces" / tp.parent.name / tp.name
        for candidate in [packaged_tp, tp]:
            if candidate.exists():
                try:
                    return _read_json(candidate)
                except Exception:
                    pass

    ws = _resolve_workspace(compare_dir, branch_blob.get("workspace"))
    if ws is None:
        return None
    for name in ["trace.json", "baseline_trace.json"]:
        tp = ws / name
        if tp.exists():
            try:
                return _read_json(tp)
            except Exception:
                continue
    return None


def _task_from_branch(compare_dir: Path, data: Dict[str, Any], branch_blob: Dict[str, Any]) -> Optional[TaskPacket]:
    trace = _load_trace(compare_dir, branch_blob)
    if isinstance(trace, dict) and isinstance(trace.get("task"), dict):
        try:
            return TaskPacket.from_dict(trace["task"])
        except Exception:
            pass

    # Fallback: reconstruct a minimal task from visible fields.
    question = data.get("question")
    gold = data.get("gold")
    if question is None:
        return None
    return TaskPacket.from_dict(
        {
            "sample_id": str(data.get("pid") or "unknown"),
            "question": question,
            "image_path": "",
            "choices": [],
            "question_type": "free_form",
            "answer_type": "text",
            "answer": gold,
            "metadata": {},
        }
    )


def _recompute_correct(compare_dir: Path, data: Dict[str, Any], branch_blob: Dict[str, Any]) -> Optional[bool]:
    if not branch_blob.get("present"):
        return None
    gold = branch_blob.get("gold", data.get("gold"))
    pred = branch_blob.get("prediction")
    if gold is None:
        return None
    if pred is None:
        return False
    task = _task_from_branch(compare_dir, data, branch_blob)
    if task is None:
        return branch_blob.get("correct")
    task.answer = gold
    try:
        return bool(answers_equal(task, pred, gold))
    except Exception:
        return branch_blob.get("correct")


def _task_metadata(compare_dir: Path, data: Dict[str, Any], branch_blob: Dict[str, Any]) -> Dict[str, Any]:
    trace = _load_trace(compare_dir, branch_blob) or {}
    task = trace.get("task") or {}
    meta = task.get("metadata") or branch_blob.get("task_metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _question_for(data: Dict[str, Any], branch_blob: Dict[str, Any]) -> str:
    if branch_blob.get("question"):
        return str(branch_blob.get("question"))
    return str(data.get("question") or "")


def _is_identity_age_gap(question: str, meta: Dict[str, Any]) -> bool:
    q = question.lower()
    src = str(meta.get("source", "")).lower()
    return (
        "age gap" in q
        or "born after the end of world war ii" in q
        or ("kvqa" in src and ("age" in q or "born after" in q))
    )


def _is_bar_chart(question: str, meta: Dict[str, Any]) -> bool:
    ctx = str(meta.get("context", "")).lower()
    src = str(meta.get("source", "")).lower()
    return ctx == "bar chart" or "chartqa" in src or "bar chart" in question.lower()


def _is_synthetic_counting(question: str, meta: Dict[str, Any]) -> bool:
    q = question.lower()
    ctx = str(meta.get("context", "")).lower()
    src = str(meta.get("source", "")).lower()
    return (
        ctx == "synthetic scene"
        and (
            "subtract all" in q
            or "how many objects are left" in q
            or "fewer" in q
            or "left side of" in q
            or "clevr-math" in src
            or "super-clevr" in src
        )
    )


def _is_geometry(question: str, meta: Dict[str, Any]) -> bool:
    ctx = str(meta.get("context", "")).lower()
    src = str(meta.get("source", "")).lower()
    return ctx == "geometry diagram" or "unigeo" in src


SLICE_FNS = {
    "identity_age_gap": _is_identity_age_gap,
    "bar_chart": _is_bar_chart,
    "synthetic_counting": _is_synthetic_counting,
    "geometry": _is_geometry,
}


def rebuild_compare_outputs(compare_dir: Path) -> Dict[str, Any]:
    compare_dir = Path(compare_dir)
    rd_files = list(_reasoning_files(compare_dir))
    if not rd_files:
        raise FileNotFoundError(f"No reasoning_details found under {compare_dir}")

    accepted_rows: List[Dict[str, Any]] = []
    per_pid_rows: List[Dict[str, Any]] = []
    slice_stats: Dict[str, Dict[str, Dict[str, Any]]] = {
        s: {
            b: {"num_samples": 0, "num_scored": 0, "num_correct": 0, "accuracy": None}
            for b in ["baseline_model", "eval_no_evolution", "eval_seeded_no_reuse", "eval_with_evolution"]
        }
        for s in SLICE_FNS
    }

    for f in rd_files:
        data = _read_json(f)
        row: Dict[str, Any] = {
            "pid": data.get("pid"),
            "question": data.get("question"),
            "gold": data.get("gold"),
        }

        for branch_name in BRANCH_NAMES:
            branch = _get_branch_blob(data, branch_name)
            if not branch:
                continue
            corrected = _recompute_correct(compare_dir, data, branch)
            row[f"{branch_name}_present"] = branch.get("present")
            row[f"{branch_name}_prediction"] = branch.get("prediction")
            row[f"{branch_name}_correct"] = corrected
            row[f"{branch_name}_used_generated_skill"] = branch.get("used_generated_skill")
            row[f"{branch_name}_error"] = branch.get("error")

            if branch_name == "eval_with_evolution":
                trace = _load_trace(compare_dir, branch) or {}
                attempts = trace.get("reuse_attempts") or []
                if branch.get("used_generated_skill") and not attempts:
                    accepted_rows.append(
                        {
                            "pid": data.get("pid"),
                            "question": data.get("question"),
                            "gold": data.get("gold"),
                            "prediction": branch.get("prediction"),
                            "correct": corrected,
                            "skill_name": branch.get("used_generated_skill"),
                            "score": trace.get("reuse_selected_score"),
                            "would_be_correct": corrected,
                            "verifier_decision": None,
                            "verifier_confidence": None,
                            "reason": "accepted_used_generated_skill_without_recorded_reuse_attempt",
                        }
                    )
                for att in attempts:
                    if att.get("accepted"):
                        accepted_rows.append(
                            {
                                "pid": data.get("pid"),
                                "question": data.get("question"),
                                "gold": data.get("gold"),
                                "prediction": branch.get("prediction"),
                                "correct": corrected,
                                "skill_name": att.get("skill_name") or branch.get("used_generated_skill"),
                                "score": att.get("score"),
                                "would_be_correct": att.get("would_be_correct"),
                                "verifier_decision": att.get("effective_verifier_decision", att.get("verifier_decision")),
                                "verifier_confidence": att.get("effective_verifier_confidence", att.get("verifier_confidence")),
                                "reason": att.get("reason"),
                            }
                        )

            if branch_name in slice_stats and False:
                pass

            if branch_name in ["baseline_model", "eval_no_evolution", "eval_seeded_no_reuse", "eval_with_evolution"] and branch.get("present"):
                meta = _task_metadata(compare_dir, data, branch)
                question = _question_for(data, branch)
                for sname, fn in SLICE_FNS.items():
                    if fn(question, meta):
                        slice_stats[sname][branch_name]["num_samples"] += 1
                        if corrected is not None:
                            slice_stats[sname][branch_name]["num_scored"] += 1
                            slice_stats[sname][branch_name]["num_correct"] += int(bool(corrected))

        per_pid_rows.append(row)

    for sname in slice_stats:
        for bname in slice_stats[sname]:
            scored = slice_stats[sname][bname]["num_scored"]
            correct = slice_stats[sname][bname]["num_correct"]
            slice_stats[sname][bname]["accuracy"] = (correct / scored) if scored else None

    accepted_fieldnames = [
        "pid",
        "question",
        "gold",
        "prediction",
        "correct",
        "skill_name",
        "score",
        "would_be_correct",
        "verifier_decision",
        "verifier_confidence",
        "reason",
    ]
    _write_csv(compare_dir / "accepted_reuse.csv", accepted_rows, accepted_fieldnames)
    if per_pid_rows:
        fieldnames = list(per_pid_rows[0].keys())
    else:
        fieldnames = ["pid", "question", "gold"]
    _write_csv(compare_dir / "per_pid_eval.csv", per_pid_rows, fieldnames)
    _write_json(compare_dir / "slice_stats.json", slice_stats)

    return {
        "accepted_reuse_count": len(accepted_rows),
        "per_pid_count": len(per_pid_rows),
        "slice_stats": slice_stats,
    }
