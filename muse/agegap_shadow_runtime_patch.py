
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .answering import answers_equal, normalize_answer
from .baseline_model import run_baseline_model
from .io_utils import save_json


_BAD_SKILL_MARKERS = (
    "kitchen",
    "baking",
    "countertop",
    "measuring",
    "cup",
    "utensil",
)

_PERSON_ENTITY_MARKERS = (
    "person",
    "people",
    "portrait",
    "face",
    "identity",
    "entity",
    "celebrity",
    "public",
    "figure",
    "birth",
    "age",
    "man",
    "woman",
    "histor",
    "vintage",
)


def _norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip().lower())


def _is_identity_age_gap_task(task: Any) -> bool:
    q = _norm(getattr(task, "question", None) or (task.get("question") if isinstance(task, dict) else ""))
    meta = getattr(task, "metadata", None) or (task.get("metadata") if isinstance(task, dict) else {}) or {}
    src = _norm(meta.get("source"))
    return (
        "age gap" in q
        or "difference in age" in q
        or "older than" in q
        or "younger than" in q
        or "born after the end of world war ii" in q
        or ("kvqa" in src and ("age" in q or "born after" in q or "how many years" in q))
    )


def _candidate_blob(c: Dict[str, Any]) -> str:
    parts = [
        c.get("name"),
        " ".join(c.get("matched_fields") or []),
        " ".join(c.get("scene_overlap") or []),
        " ".join(c.get("lexical_overlap") or []),
        c.get("description"),
        c.get("directory"),
    ]
    profile = c.get("profile") or {}
    try:
        parts.append(str(profile))
    except Exception:
        pass
    return _norm(" ".join(str(p or "") for p in parts))


def _filter_agegap_candidates(task: Any, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not _is_identity_age_gap_task(task):
        return candidates
    kept: List[Dict[str, Any]] = []
    for c in candidates:
        blob = _candidate_blob(c)
        if any(m in blob for m in _BAD_SKILL_MARKERS):
            continue
        if not any(m in blob for m in _PERSON_ENTITY_MARKERS):
            # Drop generic natural-image skills without person/entity signal.
            continue
        kept.append(c)
    return kept


def _trace_dict(trace: Any) -> Dict[str, Any]:
    if trace is None:
        return {}
    if isinstance(trace, dict):
        return trace
    if hasattr(trace, "to_dict"):
        try:
            return trace.to_dict()
        except Exception:
            pass
    return {}


def _trace_has_grounding(trace: Any) -> bool:
    td = _trace_dict(trace)
    blob = _norm(str(td))
    if any(m in blob for m in ["recognized_entity", "explicit_text", "birth year", "born in", "wikidata", "public figure"]):
        return True
    if re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", blob):
        return True
    return False


def _looks_unreasonable_agegap_answer(task: Any, value: Any) -> bool:
    q = _norm(getattr(task, "question", "") if not isinstance(task, dict) else task.get("question", ""))
    if value in (None, "", [], {}):
        return True
    s = str(value).strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            x = float(m.group(0))
        except Exception:
            return False
        if "age gap" in q or "difference in age" in q or "older than" in q or "younger than" in q:
            if x < 0 or x > 40:
                return True
        if "born after the end of world war ii" in q:
            if x > 120:
                return True
    return False


def _uses_ungrounded_agegap_estimator(trace: Any) -> bool:
    td = _trace_dict(trace)
    rounds = td.get("math_rounds") or []
    if not rounds:
        return False
    last = rounds[-1] if isinstance(rounds[-1], dict) else {}
    notes = _norm(last.get("normalization_notes"))
    if "age_gap_estimator" not in notes:
        return False
    return not _trace_has_grounding(td)


def _baseline_candidate_reasonable(task: Any, value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if _looks_unreasonable_agegap_answer(task, value):
        return False
    return True


def _append_shadow_note(trace: Any, baseline_trace: Dict[str, Any], reason: str) -> None:
    note = {
        "source": "agegap_shadow_baseline",
        "decision": "accept",
        "issues": [reason],
        "revised_answer": baseline_trace.get("final_answer_raw"),
        "follow_up_visual_questions": [],
        "confidence": baseline_trace.get("confidence", 0.66),
    }
    if hasattr(trace, "verify_rounds"):
        trace.verify_rounds.append(note)


def _maybe_shadow_with_baseline(self, task: Any, trace: Any) -> Any:
    if not _is_identity_age_gap_task(task):
        return trace

    current_norm = getattr(trace, "final_answer_normalized", None)
    current_err = getattr(trace, "error", None)

    should_shadow = False
    why: List[str] = []

    if current_err is not None:
        should_shadow = True
        why.append("pipeline_error")
    if current_norm in (None, ""):
        should_shadow = True
        why.append("empty_final_answer")
    if _looks_unreasonable_agegap_answer(task, current_norm):
        should_shadow = True
        why.append("unreasonable_age_gap_answer")
    if _uses_ungrounded_agegap_estimator(trace):
        should_shadow = True
        why.append("ungrounded_age_gap_estimator")

    if not should_shadow:
        return trace

    base = run_baseline_model(task, self.project_root, experiment_tag=(self.experiment_tag or "") + "_agegap_shadow")
    base_raw = base.get("final_answer_raw")
    base_norm = normalize_answer(task, base.get("final_answer_normalized") if base.get("final_answer_normalized") is not None else base_raw)

    if not _baseline_candidate_reasonable(task, base_norm):
        return trace

    trace.final_answer_raw = base_raw if base_raw is not None else base_norm
    trace.final_answer_normalized = base_norm
    trace.error = None
    if getattr(task, "answer", None) is not None:
        trace.correct = answers_equal(task, trace.final_answer_normalized, task.answer)
    _append_shadow_note(trace, base, "shadow baseline replaced unsupported or unreasonable age-gap result: " + ", ".join(why))

    try:
        ws = Path(str(trace.workspace))
        save_json(ws / "trace.json", trace.to_dict())
    except Exception:
        pass
    return trace


def apply_agegap_shadow_runtime_patch(cls, module_globals: Dict[str, Any]) -> None:
    # Patch candidate retrieval used inside orchestrator._attempt_reuse by monkeypatching module globals.
    orig_get = module_globals.get("get_generated_skill_candidates")
    if callable(orig_get) and not getattr(orig_get, "_agegap_filtered", False):
        def _filtered(task, project_root=None, *, top_k=3, min_score=4.5):
            candidates = orig_get(task, project_root, top_k=top_k, min_score=min_score)
            return _filter_agegap_candidates(task, candidates)
        _filtered._agegap_filtered = True
        module_globals["get_generated_skill_candidates"] = _filtered

    # Disable risky arbitration for age-gap / KVQA tasks unless backup is clearly supported.
    orig_arb = getattr(cls, "_arbitrate_backup")
    if not getattr(orig_arb, "_agegap_guard_patched", False):
        def _arbitrate_backup_patched(self, task, workspace, trace, backup):
            if _is_identity_age_gap_task(task):
                if not backup:
                    return trace
                skill_name = _norm((backup or {}).get("skill_name"))
                matched_fields = _norm(" ".join((backup or {}).get("matched_fields") or []))
                verifier_conf = float(((backup or {}).get("verifier") or {}).get("confidence", 0.0) or 0.0)
                if any(m in skill_name for m in _BAD_SKILL_MARKERS):
                    return trace
                if verifier_conf < 0.85 and not any(m in matched_fields for m in ["identity", "entity", "birth", "age", "portrait", "person"]):
                    return trace
            return orig_arb(self, task, workspace, trace, backup)
        _arbitrate_backup_patched._agegap_guard_patched = True
        cls._arbitrate_backup = _arbitrate_backup_patched

    # Apply a shadow baseline after solve() when the age-gap result is empty/unreasonable/ungrounded.
    orig_solve = getattr(cls, "solve")
    if not getattr(orig_solve, "_agegap_shadow_patched", False):
        def solve_patched(self, task):
            trace = orig_solve(self, task)
            try:
                trace = _maybe_shadow_with_baseline(self, task, trace)
                if hasattr(self, "_maybe_apply_answer_reflection"):
                    trace = self._maybe_apply_answer_reflection(task, trace, Path(str(trace.workspace)))
                try:
                    self._persist_trace(trace)
                except Exception:
                    pass
            except Exception:
                pass
            return trace
        solve_patched._agegap_shadow_patched = True
        cls.solve = solve_patched
