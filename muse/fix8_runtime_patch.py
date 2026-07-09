
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .answering import answers_equal
from .compose import save_composed_skill_from_trace
from .io_utils import save_json
from .reflection_agent import reflect_failed_trace


def _is_seed_context(agent) -> bool:
    tag = str(getattr(agent, "experiment_tag", "") or "").lower()
    return "seed" in tag


def _merge_retry_profile(base: Optional[Dict[str, object]], reflection: Dict[str, object]) -> Dict[str, object]:
    out = dict(base or {})
    for key in ["visual_hint", "reasoning_hint", "verifier_hint", "retrieval_text_bge", "retrieval_text_clip"]:
        if reflection.get(key):
            out[key] = reflection[key]
    merged_keywords: List[str] = []
    for src in [out.get("keywords", []), reflection.get("keywords", [])]:
        if isinstance(src, list):
            for item in src:
                t = str(item).strip()
                if t and t not in merged_keywords:
                    merged_keywords.append(t)
    if merged_keywords:
        out["keywords"] = merged_keywords
    return out


def _maybe_run_seed_reflection(self, task, trace):
    if not self.allow_save_generated_skills:
        return trace
    if not _is_seed_context(self):
        return trace
    if getattr(trace, "error", None) is not None:
        return trace
    if getattr(trace, "used_generated_skill", None):
        return trace
    if getattr(task, "answer", None) is None:
        return trace
    if bool(getattr(trace, "correct", False)):
        return trace

    original_trace = trace
    current_failed_trace = trace
    history: List[Dict[str, object]] = []

    for round_index in range(1, 3):
        gold_answer = task.answer if round_index >= 2 else None
        reflection = reflect_failed_trace(
            current_failed_trace,
            self.project_root,
            prior_profile=(self.profile or {}),
            round_index=round_index,
            max_rounds=2,
            gold_answer=gold_answer,
        )
        history.append(reflection)
        if not reflection.get("should_retry", True):
            break

        retry_profile = _merge_retry_profile(self.profile, reflection)
        retry_agent = self.__class__(
            profile=retry_profile,
            allow_generated_skill_reuse=False,
            allow_save_generated_skills=False,
            experiment_tag=f"{self.experiment_tag or 'seed'}|reflection_{round_index}",
        )
        retry_trace = retry_agent.solve(task)
        retry_trace.experiment_tag = getattr(self, "experiment_tag", None)

        try:
            if getattr(retry_trace, "error", None) is None and getattr(task, "answer", None) is not None:
                retry_trace.correct = answers_equal(task, retry_trace.final_answer_normalized, task.answer)
        except Exception:
            pass

        if bool(getattr(retry_trace, "correct", False)):
            retry_trace.reflection_profile_override = retry_profile
            retry_trace.reflection_history = history
            saved_path = save_composed_skill_from_trace(retry_trace, self.project_root)
            if saved_path is not None:
                retry_trace.saved_generated_skill = Path(saved_path).name
            try:
                save_json(Path(retry_trace.workspace) / "trace.json", retry_trace.to_dict())
            except Exception:
                pass
            try:
                self._persist_trace(retry_trace)
            except Exception:
                pass
            return retry_trace

        current_failed_trace = retry_trace

    try:
        original_trace.reflection_history = history
    except Exception:
        pass
    return original_trace


def apply_fix8_seed_reflection_patch(MultimodalMetaAgent):
    if getattr(MultimodalMetaAgent, "_fix8_seed_reflection_patch_applied", False):
        return

    original_solve = MultimodalMetaAgent.solve

    def patched_solve(self, task):
        trace = original_solve(self, task)
        try:
            return _maybe_run_seed_reflection(self, task, trace)
        except Exception:
            return trace

    MultimodalMetaAgent.solve = patched_solve
    MultimodalMetaAgent._fix8_seed_reflection_patch_applied = True
