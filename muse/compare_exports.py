from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BRANCH_FILES = {
    "baseline_model": "baseline_model.jsonl",
    "seed_install": "seed_install.jsonl",
    "eval_no_evolution": "eval_no_evolution.jsonl",
    "eval_seeded_no_reuse": "eval_seeded_no_reuse.jsonl",
    "eval_with_evolution": "eval_with_evolution.jsonl",
}


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



def _branch_rows(compare_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    return {
        branch: _read_jsonl(compare_dir / filename)
        for branch, filename in BRANCH_FILES.items()
    }



def _trace_bundle_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    workspace = row.get("workspace")
    if not workspace:
        return {
            "workspace": None,
            "trace_path": None,
            "trace_exists": False,
            "trace": None,
        }

    workspace_path = Path(workspace)
    trace_path = workspace_path / "trace.json"
    trace = None
    if trace_path.exists():
        try:
            trace = _read_json(trace_path)
        except Exception as exc:  # pragma: no cover
            trace = {"_trace_load_error": f"{type(exc).__name__}: {exc}"}

    return {
        "workspace": str(workspace_path),
        "trace_path": str(trace_path),
        "trace_exists": trace_path.exists(),
        "trace": trace,
    }



def _pid_dirname(pid: str) -> str:
    return f"PID{pid.zfill(4)}" if pid.isdigit() else f"PID_{pid}"



def _blank_branch(branch: str, note: str | None = None) -> Dict[str, Any]:
    return {
        "branch": branch,
        "present": False,
        "note": note or "pid not present in this branch",
        "prediction": None,
        "gold": None,
        "correct": None,
        "error": None,
        "used_generated_skill": None,
        "workspace": None,
        "trace_path": None,
        "trace": None,
        "final_answer_raw": None,
        "final_answer_normalized": None,
    }



def _filled_branch(branch: str, row: Dict[str, Any]) -> Dict[str, Any]:
    bundle = _trace_bundle_from_row(row)
    trace = bundle.get("trace") or {}
    task = trace.get("task", {}) if isinstance(trace, dict) else {}
    return {
        "branch": branch,
        "present": True,
        "note": None,
        "prediction": row.get("prediction"),
        "gold": row.get("gold"),
        "correct": row.get("correct"),
        "error": row.get("error"),
        "used_generated_skill": row.get("used_generated_skill"),
        "workspace": bundle["workspace"],
        "trace_path": bundle["trace_path"],
        "trace": trace,
        "final_answer_raw": trace.get("final_answer_raw") if isinstance(trace, dict) else None,
        "final_answer_normalized": trace.get("final_answer_normalized") if isinstance(trace, dict) else None,
        "question": row.get("question") or task.get("question"),
        "answer_type": task.get("answer_type"),
        "question_type": task.get("question_type"),
        "task_metadata": task.get("metadata"),
    }



def build_reasoning_details(compare_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    compare_dir = Path(compare_dir)
    branch_rows = _branch_rows(compare_dir)

    all_pids = set()
    for rows in branch_rows.values():
        for row in rows:
            all_pids.add(str(row.get("pid")))

    per_pid: Dict[str, Dict[str, Any]] = {}
    matrix: Dict[str, Any] = {}

    for pid in sorted(all_pids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
        details = {
            "pid": pid,
            "question": None,
            "gold": None,
            "branches": {},
        }
        matrix[pid] = {}

        for branch, rows in branch_rows.items():
            row = next((r for r in rows if str(r.get("pid")) == pid), None)
            if row is None:
                branch_payload = _blank_branch(branch)
            else:
                branch_payload = _filled_branch(branch, row)
                if details["question"] is None:
                    details["question"] = branch_payload.get("question")
                if details["gold"] is None:
                    details["gold"] = branch_payload.get("gold")
            details["branches"][branch] = branch_payload
            matrix[pid][branch] = {
                "present": branch_payload["present"],
                "correct": branch_payload["correct"],
                "prediction": branch_payload["prediction"],
                "used_generated_skill": branch_payload["used_generated_skill"],
                "error": branch_payload["error"],
            }

        per_pid[pid] = details

    return per_pid, {
        "num_pids": len(per_pid),
        "branch_files": BRANCH_FILES,
        "pid_matrix": matrix,
    }



def export_reasoning_details(compare_dir: str | Path, output_root: str | Path | None = None) -> Path:
    compare_dir = Path(compare_dir)
    per_pid, meta = build_reasoning_details(compare_dir)

    output_root = Path(output_root) if output_root else compare_dir / "reasoning_details"
    output_root.mkdir(parents=True, exist_ok=True)

    index = {
        "compare_dir": str(compare_dir),
        "num_pids": len(per_pid),
        "pids": [],
    }

    for pid, payload in per_pid.items():
        pid_dir = output_root / _pid_dirname(pid)
        pid_dir.mkdir(parents=True, exist_ok=True)
        out_path = pid_dir / "reasoning_details.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        index["pids"].append({
            "pid": pid,
            "question": payload.get("question"),
            "path": str(out_path),
        })

    (output_root / "reasoning_details_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "pid_branch_matrix.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_root
