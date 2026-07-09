from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from muse.answering import answers_equal  # noqa: E402
from muse.io_utils import save_json, save_jsonl  # noqa: E402
from muse.orchestrator import MultimodalMetaAgent  # noqa: E402
from muse.schemas import TaskPacket  # noqa: E402
from run_multidataset_matrix import DATASET_SPECS, _load_hf_rows, _row_to_task  # noqa: E402

PHASES = [
    ("p1_cs", "commonsense", "P1Train_CS"),
    ("p2_sci", "science", "P2Train_SCI"),
    ("p3_math", "mathematics", "P3Train_MATH"),
]
PROBES = [
    ("probe_cs", "commonsense", "Probe_CS"),
    ("probe_sci", "science", "Probe_SCI"),
    ("probe_math", "mathematics", "Probe_MATH"),
]


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
    rows = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_library(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)


def domain_of(row_or_task: Dict[str, Any]) -> str:
    meta = row_or_task.get("metadata") or {}
    return str(meta.get("domain") or row_or_task.get("domain") or "").strip().lower()


def load_m3cot_rows(split: str) -> List[Dict[str, Any]]:
    load_env()
    spec = DATASET_SPECS["m3cot"]
    return _load_hf_rows(spec, split)


def rows_to_tasks(rows: Sequence[Dict[str, Any]]) -> List[TaskPacket]:
    spec = DATASET_SPECS["m3cot"]
    return [_row_to_task(row, spec) for row in rows]


def save_tasks(path: Path, tasks: Sequence[TaskPacket]) -> None:
    save_jsonl(path, [task.to_dict() for task in tasks])


def load_tasks_jsonl(path: str | Path) -> List[TaskPacket]:
    return [TaskPacket.from_dict(row) for row in read_jsonl(path)]


def row_from_trace(task: TaskPacket, trace: Any, *, phase: Optional[str] = None) -> Dict[str, Any]:
    reuse_candidates = list(getattr(trace, "reuse_candidates", []) or [])
    reuse_attempts = list(getattr(trace, "reuse_attempts", []) or [])
    accepted_attempts = [a for a in reuse_attempts if a.get("accepted")]
    correct = (
        answers_equal(task, trace.final_answer_normalized, task.answer)
        if task.answer is not None and trace.error is None
        else None
    )
    return {
        "pid": task.sample_id,
        "question": task.question,
        "domain": task.metadata.get("domain"),
        "prediction": trace.final_answer_normalized,
        "gold": task.answer,
        "correct": correct,
        "workspace": trace.workspace,
        "used_generated_skill": trace.used_generated_skill,
        "saved_generated_skill": getattr(trace, "saved_generated_skill", None),
        "reuse_enabled": bool(reuse_candidates or reuse_attempts or getattr(trace, "used_generated_skill", None)),
        "reuse_candidates": reuse_candidates,
        "reuse_attempts": reuse_attempts,
        "reuse_fallback_reason": getattr(trace, "reuse_fallback_reason", None),
        "reuse_selected_score": getattr(trace, "reuse_selected_score", None),
        "num_reuse_candidates": len(reuse_candidates),
        "reuse_candidate_names": [str(c.get("name")) for c in reuse_candidates if c.get("name")],
        "reuse_attempted": bool(reuse_attempts),
        "num_reuse_attempts": len(reuse_attempts),
        "reuse_accepted": bool(accepted_attempts),
        "num_reuse_accepts": len(accepted_attempts),
        "phase": phase,
        "error": trace.error,
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if r.get("error") is None and r.get("correct") is not None]
    correct = sum(1 for r in scored if r.get("correct") is True)
    failures = sum(1 for r in rows if r.get("error") is not None)
    reused = [r for r in rows if r.get("used_generated_skill")]
    return {
        "num_samples": len(rows),
        "num_scored": len(scored),
        "num_correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "num_failures": failures,
        "num_reused_generated_skills": len(reused),
    }


def run_train_phase(
    tasks: Sequence[TaskPacket],
    *,
    generated_root: Path,
    save: bool,
    reuse: bool,
    phase: str,
    out_path: Path,
    experiment_tag: Optional[str] = None,
    suppress_reuse_registry: bool = False,
) -> List[Dict[str, Any]]:
    os.environ["MUSE_GENERATED_SKILLS_ROOT"] = str(generated_root.resolve())
    existing = {str(row.get("pid")): row for row in read_jsonl(out_path)}
    pending = [task for task in tasks if str(task.sample_id) not in existing]
    if pending:
        if suppress_reuse_registry:
            import muse.orchestrator as orch

            orch.record_skill_outcome = lambda *args, **kwargs: None
        agent = MultimodalMetaAgent(
            allow_generated_skill_reuse=reuse,
            allow_save_generated_skills=save,
            experiment_tag=experiment_tag or f"continual_train_{phase}",
        )
        for task in pending:
            trace = agent.solve(task)
            row = row_from_trace(task, trace, phase=phase)
            existing[str(task.sample_id)] = row
            append_jsonl(out_path, row)
    ordered = [existing[str(task.sample_id)] for task in tasks if str(task.sample_id) in existing]
    save_jsonl(out_path, ordered)
    return ordered


def eval_worker(
    task_dict: Dict[str, Any],
    generated_root: str,
    tag: str,
    *,
    reuse_enabled: bool = True,
    reuse_min_score: float = 0.0,
    reuse_top_k: int = 5,
    reuse_accept_conf: float = 0.70,
) -> Dict[str, Any]:
    load_env()
    os.environ["MUSE_GENERATED_SKILLS_ROOT"] = generated_root
    task = TaskPacket.from_dict(task_dict)

    if reuse_enabled:
        # Continual probes must exercise checkpoint libraries. The mainline retriever
        # is conservative for production compare/evolution, so relax it only here.
        import muse.orchestrator as orch
        from muse.evolution_policy import ReusePolicyResult

        original_get_candidates = orch.get_generated_skill_candidates
        original_decide_reuse_acceptance = orch.decide_reuse_acceptance

        def continual_candidates(candidate_task, project_root=None, *, top_k=3, min_score=4.5):
            return original_get_candidates(
                candidate_task,
                project_root,
                top_k=max(int(top_k or 0), int(reuse_top_k)),
                min_score=float(reuse_min_score),
            )

        def continual_acceptance(candidate_task, candidate, verifier_decision, verifier_confidence, looks_canonical):
            policy = original_decide_reuse_acceptance(
                candidate_task,
                candidate,
                verifier_decision,
                verifier_confidence,
                looks_canonical,
            )
            if policy.accepted:
                return policy
            if (
                verifier_decision == "accept"
                and looks_canonical
                and float(verifier_confidence or 0.0) >= float(reuse_accept_conf)
            ):
                return ReusePolicyResult(
                    accepted=True,
                    trust_level="continual_probe_relaxed",
                    threshold=float(reuse_accept_conf),
                    reasons=["continual_probe_relaxed_acceptance"],
                )
            return policy

        orch.get_generated_skill_candidates = continual_candidates
        orch.decide_reuse_acceptance = continual_acceptance
        orch.record_skill_outcome = lambda *args, **kwargs: None

    agent = MultimodalMetaAgent(
        allow_generated_skill_reuse=reuse_enabled,
        allow_save_generated_skills=False,
        experiment_tag=tag,
    )
    trace = agent.solve(task)
    row = row_from_trace(task, trace)
    row["reuse_enabled"] = bool(reuse_enabled)
    row["continual_reuse_min_score"] = float(reuse_min_score)
    row["continual_reuse_top_k"] = int(reuse_top_k)
    row["continual_reuse_accept_conf"] = float(reuse_accept_conf)
    return row


def library_skill_names(library_dir: Path) -> List[str]:
    if not library_dir.exists():
        return []
    return sorted(child.name for child in library_dir.iterdir() if child.is_dir() and not child.name.startswith("_"))


def load_skill_phase_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_number(phase_label: str) -> Optional[int]:
    if len(phase_label) >= 2 and phase_label[0] == "p" and phase_label[1].isdigit():
        return int(phase_label[1])
    return None


def compute_reuse_stats(rows: Sequence[Dict[str, Any]], *, skill_phase_map: Dict[str, str], old_domain: Optional[str] = None, current_phase_index: Optional[int] = None) -> Dict[str, Any]:
    reused = [row for row in rows if row.get("used_generated_skill")]
    counts = Counter(str(row.get("used_generated_skill")) for row in reused)
    total_reuse = sum(counts.values())
    correct_reuse = sum(1 for row in reused if row.get("correct") is True)
    total_candidates = sum(int(row.get("num_reuse_candidates") or len(row.get("reuse_candidates") or [])) for row in rows)
    total_attempts = sum(int(row.get("num_reuse_attempts") or len(row.get("reuse_attempts") or [])) for row in rows)
    total_accepts = sum(int(row.get("num_reuse_accepts") or sum(1 for a in (row.get("reuse_attempts") or []) if a.get("accepted"))) for row in rows)
    entropy = 0.0
    if total_reuse:
        for count in counts.values():
            p = count / total_reuse
            entropy -= p * math.log(p, 2)
    top_counts = sorted(counts.values(), reverse=True)
    cross_phase = []
    toxic_wrong = []
    domain_phase = {"commonsense": 1, "science": 2, "mathematics": 3}.get(str(old_domain or "").lower())
    if domain_phase is not None:
        for row in reused:
            phase = skill_phase_map.get(str(row.get("used_generated_skill")), "")
            phase_num = _phase_number(phase)
            if phase_num is not None and phase_num > domain_phase:
                cross_phase.append(row)
                if row.get("correct") is False:
                    toxic_wrong.append(row)
    return {
        "num_reuse": total_reuse,
        "num_reuse_candidates_total": total_candidates,
        "num_reuse_attempts_total": total_attempts,
        "num_reuse_accepts_total": total_accepts,
        "num_used_generated_skills_total": total_reuse,
        "num_unique_reused_skills": len(counts),
        "reuse_accuracy": correct_reuse / len(reused) if reused else None,
        "old_domain_reuse_precision": correct_reuse / len(reused) if reused else None,
        "reuse_entropy": entropy,
        "top1_skill_share": top_counts[0] / total_reuse if total_reuse else 0.0,
        "top3_skill_share": sum(top_counts[:3]) / total_reuse if total_reuse else 0.0,
        "cross_phase_reuse_count": len(cross_phase),
        "cross_phase_toxic_reuse_count": len(toxic_wrong),
        "cross_phase_toxic_reuse_ratio": len(toxic_wrong) / len(cross_phase) if cross_phase else None,
        "cross_phase_toxic_reuse_accuracy": summarize_rows(cross_phase)["accuracy"] if cross_phase else None,
    }
