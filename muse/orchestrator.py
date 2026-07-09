from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .answering import answers_equal, normalize_answer
from .compose import save_composed_skill_from_trace
from .config import load_runtime_config
from .evolution_policy import decide_reuse_acceptance
from .io_utils import save_json
from .reuse_registry import record_skill_outcome
from .runtime import run_python_file
from .schemas import EvidenceBundle, MathReasoningResult, SolveTrace, TaskPacket, VerificationResult
from .skills_utils import get_generated_skill_candidates, list_subagent_skills
from .fallback_subagents import call_builtin_skill
from .llm_clients import OpenAIStyleClient
from .public_figure_reasoner import maybe_override_with_public_figure_grounding


class MultimodalMetaAgent:
    REUSE_TOP_K = 3
    REUSE_MIN_SCORE = 4.5
    ANSWER_REFLECTION_SYSTEM_PROMPT = """You are a reflection baseline for a multimodal QA model.
You receive the original task and the agent's initial final answer.
Critique only that answer, then either keep it or revise it.
Do not use external memory, generated skills, tools, or hidden gold labels.
Return JSON only with keys:
- critique: short explanation
- revised_answer: final answer
- confidence: number between 0 and 1
"""

    def __init__(
        self,
        *,
        profile: Optional[Dict[str, Any]] = None,
        allow_generated_skill_reuse: bool = True,
        allow_save_generated_skills: Optional[bool] = None,
        experiment_tag: Optional[str] = None,
    ):
        self.project_root = Path(__file__).resolve().parent.parent
        self.config = load_runtime_config(self.project_root)
        self.profile = profile or {}
        self.allow_generated_skill_reuse = allow_generated_skill_reuse
        self.allow_save_generated_skills = self.config.save_generated_skills if allow_save_generated_skills is None else bool(allow_save_generated_skills)
        self.experiment_tag = experiment_tag

    def _create_workspace(self, task: TaskPacket) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace = self.config.workspace_root / f"{stamp}_{task.sample_id}"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _skill_path(self, skill_name: str) -> Path:
        for item in list_subagent_skills():
            if item["name"] == skill_name:
                candidate = Path(item["directory"]) / item["entry_file"]
                if candidate.exists():
                    return candidate

        search_roots = [
            self.project_root / "skills" / "subagents",
            self.config.generated_skills_root,
            self.project_root / "skills" / "meta",
            self.project_root / "skills" / "tools",
        ]
        direct_names = [f"{skill_name}.py", "solver.py", "main.py"]

        for root in search_roots:
            if not root.exists():
                continue

            direct_dir = root / skill_name
            for filename in direct_names:
                candidate = direct_dir / filename
                if candidate.exists():
                    return candidate

            for filename in direct_names:
                hits = sorted(root.rglob(filename))
                for hit in hits:
                    if hit.parent.name == skill_name:
                        return hit

            named_hits = sorted(root.rglob(f"{skill_name}.py"))
            if named_hits:
                return named_hits[0]

        raise FileNotFoundError(f"Skill not found: {skill_name}")


    def _call_skill(self, skill_name: str, payload: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
        try:
            path = self._skill_path(skill_name)
        except FileNotFoundError:
            fallback = call_builtin_skill(skill_name, payload, workspace, self.project_root)
            if fallback is not None:
                return fallback
            raise
        return run_python_file(path, json.dumps(payload, ensure_ascii=False), work_dir=workspace)

    def _normalize_via_skill(self, task: TaskPacket, candidate_answer: Any, workspace: Path) -> Any:
        normalize_payload = {"task": self._task_payload_without_gold(task), "candidate_answer": candidate_answer}
        skill_answer = candidate_answer
        try:
            normalize_result = self._call_skill("answer_normalizer", normalize_payload, workspace)
            if normalize_result.get("success", False):
                skill_answer = normalize_result.get("answer")
        except Exception:
            skill_answer = candidate_answer

        normalized_from_skill = normalize_answer(task, skill_answer)
        normalized_from_raw = normalize_answer(task, candidate_answer)

        # Keep answer_normalizer in the loop, but for numeric+unit questions
        # prefer normalization from the raw answer so units like "0.01155 m"
        # can still be converted to the requested task unit.
        if task.answer_type in {"integer", "float"} and getattr(task, "unit", None) not in (None, "", "none", "None"):
            return normalized_from_raw if normalized_from_raw is not None else normalized_from_skill
        return normalized_from_skill if normalized_from_skill is not None else normalized_from_raw

    def _recompute_correct(self, task: TaskPacket, trace: SolveTrace) -> None:
        if task.answer is None:
            trace.correct = None
            return
        if trace.error is not None:
            trace.correct = None
            return
        trace.correct = answers_equal(task, trace.final_answer_normalized, task.answer)

    def _numeric_like(self, value: Any) -> bool:
        if value is None:
            return False
        return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(value).strip()))

    def _canonical_like(self, task: TaskPacket, value: Any) -> bool:
        if value in (None, ""):
            return False
        if task.answer_type in {"float", "integer"}:
            return self._numeric_like(value)
        if task.question_type == "multi_choice" and task.choices:
            return str(value).strip() in set(task.choices)
        return True

    def _parse_result_summary(self, summary: Any) -> Dict[str, Any]:
        if summary is None:
            return {}
        text = str(summary).strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"summary_text": text}

    def _load_trace_from_workspace(self, workspace_like: Any) -> Dict[str, Any]:
        if not workspace_like:
            return {}
        try:
            ws = Path(str(workspace_like))
            trace_path = ws / "trace.json"
            if trace_path.exists():
                return json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {}

    def _last_nonempty_math_result(self, trace_obj: Dict[str, Any], fallback_answer: Any, summary_text: str) -> Dict[str, Any]:
        math_rounds = trace_obj.get("math_rounds") or []
        for item in reversed(math_rounds):
            if not isinstance(item, dict):
                continue
            candidate = item.get("candidate_answer")
            steps = item.get("reasoning_steps") or []
            if candidate not in (None, "") or steps:
                out = dict(item)
                if out.get("candidate_answer") in (None, "") and fallback_answer not in (None, ""):
                    out["candidate_answer"] = fallback_answer
                return out
        return {
            "reasoning_steps": [f"Saved generated skill proposed an answer.{(' ' + summary_text[:500]) if summary_text else ''}".strip()],
            "candidate_answer": fallback_answer,
            "answer_confidence": 0.55,
            "needs_visual_recheck": False,
            "focus_questions": [],
            "normalization_notes": summary_text[:1200],
        }

    def _verifier_empty(self, verify_data: VerificationResult) -> bool:
        return any("verifier returned empty json" in str(issue).lower() for issue in (verify_data.issues or []))

    def _canonical_accept(self, task: TaskPacket, normalized: Any) -> bool:
        return self._canonical_like(task, normalized) and normalized not in (None, "")

    def _pick_reflection_model_config(self):
        if self.config.orchestrator_model and self.config.orchestrator_model.enabled:
            return self.config.orchestrator_model
        return self.config.reasoning_model

    def _answer_reflection_enabled(self) -> bool:
        if os.getenv("MUSE_EVOLUTION_REFLECTION1", "0").strip() not in {"1", "true", "TRUE", "yes", "YES"}:
            return False
        tag = str(self.experiment_tag or "").lower()
        return "eval_with_evolution" in tag or "evolved" in tag

    def _task_payload_without_gold(self, task: TaskPacket) -> Dict[str, Any]:
        payload = task.to_dict()
        payload.pop("answer", None)
        return payload

    def _task_query_without_gold(self, task: TaskPacket) -> str:
        return json.dumps(self._task_payload_without_gold(task), ensure_ascii=False)

    def _maybe_apply_answer_reflection(self, task: TaskPacket, trace: SolveTrace, workspace: Path) -> SolveTrace:
        if not self._answer_reflection_enabled():
            return trace
        if trace.error is not None:
            return trace
        if trace.final_answer_normalized in (None, "") and trace.final_answer_raw in (None, ""):
            return trace

        initial = {
            "final_answer_raw": trace.final_answer_raw,
            "final_answer_normalized": trace.final_answer_normalized,
            "used_generated_skill": trace.used_generated_skill,
            "reuse_fallback_reason": trace.reuse_fallback_reason,
            "last_verify_round": trace.verify_rounds[-1] if trace.verify_rounds else None,
            "last_math_round": trace.math_rounds[-1] if trace.math_rounds else None,
        }
        prompt = (
            f"Task:\n{json.dumps(self._task_payload_without_gold(task), ensure_ascii=False)}\n\n"
            f"Initial answer:\n{json.dumps(initial, ensure_ascii=False)}\n\n"
            "Return JSON only."
        )

        record: Dict[str, Any] = {
            "enabled": True,
            "source": "answer_reflection1",
            "initial_answer_raw": trace.final_answer_raw,
            "initial_answer_normalized": trace.final_answer_normalized,
        }
        try:
            client = OpenAIStyleClient(self._pick_reflection_model_config())
            image_paths = task.existing_image_paths()
            if image_paths:
                data = client.complete_multimodal_json(
                    self.ANSWER_REFLECTION_SYSTEM_PROMPT,
                    prompt,
                    image_paths,
                    max_tokens=1000,
                    temperature=0.0,
                )
            else:
                data = client.complete_json(
                    self.ANSWER_REFLECTION_SYSTEM_PROMPT,
                    prompt,
                    max_tokens=1000,
                    temperature=0.0,
                )
            if not isinstance(data, dict):
                data = {}
            revised = data.get("revised_answer", data.get("answer"))
            revised_norm = self._normalize_via_skill(task, revised, workspace)
            record.update({
                "critique": data.get("critique"),
                "revised_answer_raw": revised,
                "revised_answer_normalized": revised_norm,
                "confidence": data.get("confidence"),
            })
            if self._canonical_accept(task, revised_norm):
                trace.final_answer_raw = revised
                trace.final_answer_normalized = revised_norm
                record["applied"] = True
            else:
                record["applied"] = False
                record["skip_reason"] = "reflection_revised_answer_not_canonical"
        except Exception as exc:
            record.update({
                "applied": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

        trace.answer_reflection = record
        if task.answer is not None and trace.error is None:
            trace.correct = answers_equal(task, trace.final_answer_normalized, task.answer)
        return trace

    def _is_abstention_like(self, value: Any) -> bool:
        if value in (None, "", [], {}):
            return True
        text = str(value).strip().lower()
        return any(marker in text for marker in ["cannot determine", "unable to determine", "insufficient evidence", "need image", "recheck"])


    def _is_identity_age_gap_task(self, task: TaskPacket) -> bool:
        q = str(getattr(task, "question", "") or "").lower()
        meta = getattr(task, "metadata", {}) or {}
        src = str(meta.get("source", "")).lower()
        return (
            "age gap" in q
            or "born after the end of world war ii" in q
            or ("kvqa" in src and ("age" in q or "born after" in q))
        )

    def _candidate_looks_grounded_for_identity_age_gap(self, candidate_like: Dict[str, Any]) -> bool:
        name = str(candidate_like.get("name") or candidate_like.get("skill_name") or "").lower()
        matched_fields = " ".join(str(x).lower() for x in (candidate_like.get("matched_fields") or []))
        hay = f"{name} {matched_fields}"
        strong_tokens = [
            "identity",
            "entity",
            "public",
            "figure",
            "birth",
            "born",
            "age-gap",
            "age_gap",
            "kvqa",
            "portrait",
            "celebrity",
            "biograph",
            "name",
        ]
        if any(tok in hay for tok in strong_tokens):
            return True
        profile = candidate_like.get("profile") or {}
        extra = str(profile).lower()
        return any(tok in extra for tok in strong_tokens)

    def _should_skip_reuse_candidate_for_task(self, task: TaskPacket, candidate: Dict[str, Any]) -> Optional[str]:
        if self._is_identity_age_gap_task(task) and not self._candidate_looks_grounded_for_identity_age_gap(candidate):
            return "identity_age_gap_filtered_generic_skill"
        return None

    def _should_allow_backup_arbitration(self, task: TaskPacket, trace: SolveTrace, backup: Dict[str, Any]) -> bool:
        if not backup:
            return False
        if not self._is_identity_age_gap_task(task):
            return True
        if not self._candidate_looks_grounded_for_identity_age_gap(backup):
            return False
        policy = backup.get("policy", {}) or {}
        if not bool(policy.get("accepted", False)) and str(policy.get("trust_level", "")).lower() == "low":
            return False
        conf = float((backup.get("verifier", {}) or {}).get("confidence", 0.0) or 0.0)
        if conf < 0.75:
            return False
        base_verify = trace.verify_rounds[-1] if trace.verify_rounds else {}
        base_conf = float(base_verify.get("confidence", 0.0) or 0.0)
        if trace.final_answer_normalized not in (None, "") and conf <= base_conf + 0.05:
            return False
        return True

    def _assess_reused_answer(self, task: TaskPacket, candidate: Dict[str, Any], result: Dict[str, Any], workspace: Path) -> Tuple[VerificationResult, Any]:
        parsed_summary = self._parse_result_summary(result.get("summary"))
        inner_trace = self._load_trace_from_workspace(parsed_summary.get("workspace"))
        inner_evidence = inner_trace.get("evidence") or {
            "scene_type": str(task.metadata.get("context") or "unknown"),
            "visual_facts": [],
            "uncertainties": [],
            "focus_answers": [],
            "raw_payloads": [],
        }
        inner_math = self._last_nonempty_math_result(inner_trace, result.get("answer"), str(result.get("summary") or ""))
        verify_payload = {
            "task": self._task_payload_without_gold(task),
            "evidence": inner_evidence,
            "math_result": inner_math,
            "profile": {
                "reuse_mode": True,
                "generated_skill": candidate["name"],
                "candidate_score": candidate.get("score"),
                "candidate_stats": candidate.get("stats", {}),
                "candidate_matched_fields": candidate.get("matched_fields", []),
            },
        }
        verify_result = self._call_skill("multimodal_verifier", verify_payload, workspace)
        if not verify_result["success"]:
            raise RuntimeError(verify_result["error"])
        verify_data = VerificationResult.from_dict(json.loads(verify_result.get("answer") or "{}"))
        normalized = self._normalize_via_skill(task, verify_data.revised_answer if verify_data.revised_answer is not None else result.get("answer"), workspace)
        return verify_data, normalized

    def _run_base_pipeline(self, task: TaskPacket, workspace: Path, trace: SolveTrace) -> SolveTrace:
        evidence = EvidenceBundle()
        pending_focus_questions: List[str] = []
        max_rounds = max(1, self.config.max_rechecks + 1)
        best_round: Optional[Dict[str, Any]] = None

        for round_idx in range(max_rounds):
            visual_payload = {"task": self._task_payload_without_gold(task), "focus_questions": pending_focus_questions, "profile": self.profile}
            visual_result = self._call_skill("visual_detail_agent", visual_payload, workspace)
            if not visual_result["success"]:
                raise RuntimeError(visual_result["error"])
            visual_data = json.loads(visual_result.get("answer") or "{}")
            evidence.merge_visual_payload(visual_data)
            trace.visual_rounds.append(visual_data)
            trace.evidence = evidence

            math_payload = {"task": self._task_payload_without_gold(task), "evidence": evidence.to_dict(), "profile": self.profile}
            math_result = self._call_skill("math_reason_agent", math_payload, workspace)
            if not math_result["success"]:
                raise RuntimeError(math_result["error"])
            math_data = MathReasoningResult.from_dict(json.loads(math_result.get("answer") or "{}"))
            override = maybe_override_with_public_figure_grounding(task, evidence.to_dict(), math_data.to_dict(), self.project_root)
            if override is not None:
                math_data = MathReasoningResult.from_dict(override)
            trace.math_rounds.append(math_data.to_dict())

            verify_payload = {"task": self._task_payload_without_gold(task), "evidence": evidence.to_dict(), "math_result": math_data.to_dict(), "profile": self.profile}
            verify_result = self._call_skill("multimodal_verifier", verify_payload, workspace)
            if not verify_result["success"]:
                raise RuntimeError(verify_result["error"])
            verify_data = VerificationResult.from_dict(json.loads(verify_result.get("answer") or "{}"))
            trace.verify_rounds.append(verify_data.to_dict())

            raw_answer = verify_data.revised_answer if verify_data.revised_answer is not None else math_data.candidate_answer
            normalized_here = self._normalize_via_skill(task, raw_answer, workspace)
            conf_here = max(float(math_data.answer_confidence or 0.0), float(verify_data.confidence or 0.0))
            if self._canonical_accept(task, normalized_here):
                if best_round is None or conf_here > best_round["confidence"]:
                    best_round = {
                        "raw": raw_answer,
                        "normalized": normalized_here,
                        "confidence": conf_here,
                        "round_index": round_idx,
                    }

            if (
                round_idx < max_rounds - 1
                and (math_data.needs_visual_recheck or verify_data.decision == "recheck")
                and (math_data.focus_questions or verify_data.follow_up_visual_questions)
            ):
                pending_focus_questions = verify_data.follow_up_visual_questions or math_data.focus_questions
                continue

            trace.final_answer_raw = raw_answer
            break

        normalized_answer = self._normalize_via_skill(task, trace.final_answer_raw, workspace)
        if not self._canonical_accept(task, normalized_answer) and best_round is not None:
            trace.final_answer_raw = best_round["raw"]
            trace.final_answer_normalized = best_round["normalized"]
            trace.verify_rounds.append({
                "source": "best_round_salvage",
                "decision": "accept",
                "issues": [f"Recovered the strongest canonical candidate from round {best_round['round_index'] + 1} after later rechecks degraded to abstention or non-canonical output."],
                "revised_answer": best_round["raw"],
                "follow_up_visual_questions": [],
                "confidence": best_round["confidence"],
            })
        else:
            trace.final_answer_normalized = normalized_answer

        if task.answer is not None:
            trace.correct = answers_equal(task, trace.final_answer_normalized, task.answer)
        return trace

    def _maybe_backup_candidate(self, task: TaskPacket, candidate: Dict[str, Any], verify_data: VerificationResult, normalized: Any, result: Dict[str, Any], policy) -> Optional[Dict[str, Any]]:
        if normalized in (None, ""):
            return None
        if not self._canonical_like(task, normalized):
            return None
        # Only retain plausible proposals; ignore cold low-confidence junk.
        if policy.trust_level == "low" and verify_data.confidence < 0.40:
            return None
        if verify_data.decision not in {"accept", "recheck"}:
            return None
        return {
            "skill_name": candidate["name"],
            "score": float(candidate.get("score", 0.0)),
            "normalized_answer": normalized,
            "raw_answer": result.get("answer"),
            "verifier": verify_data.to_dict(),
            "matched_fields": list(candidate.get("matched_fields", [])),
            "stats": candidate.get("stats", {}),
            "policy": {
                "accepted": policy.accepted,
                "trust_level": policy.trust_level,
                "threshold": policy.threshold,
                "reasons": list(policy.reasons),
            },
        }

    def _attempt_reuse(self, task: TaskPacket, workspace: Path, trace: SolveTrace) -> Tuple[Optional[SolveTrace], Optional[Dict[str, Any]]]:
        if not self.allow_generated_skill_reuse:
            return None, None


        raw_candidates = get_generated_skill_candidates(task, self.project_root, top_k=self.REUSE_TOP_K, min_score=self.REUSE_MIN_SCORE)
        candidates: List[Dict[str, Any]] = []
        trace.reuse_candidates = []
        for c in raw_candidates:
            payload = {
                "name": c["name"],
                "score": c["score"],
                "matched_fields": c["matched_fields"],
                "stats": c.get("stats", {}),
            }
            skip_reason = self._should_skip_reuse_candidate_for_task(task, c)
            if skip_reason:
                payload["filtered_out"] = True
                payload["filter_reason"] = skip_reason
                trace.reuse_candidates.append(payload)
                continue
            payload["filtered_out"] = False
            trace.reuse_candidates.append(payload)
            candidates.append(c)

        if raw_candidates and not candidates:
            trace.reuse_fallback_reason = "all_reuse_candidates_filtered"

        best_backup: Optional[Dict[str, Any]] = None
        for candidate in candidates:
            path = Path(candidate["directory"]) / candidate["entry_file"]
            result = run_python_file(path, self._task_query_without_gold(task), work_dir=workspace)
            attempt: Dict[str, Any] = {
                "skill_name": candidate["name"],
                "score": candidate["score"],
                "matched_fields": list(candidate.get("matched_fields", [])),
                "stats": candidate.get("stats", {}),
                "success": bool(result.get("success", False)),
                "raw_answer": result.get("answer"),
                "summary": str(result.get("summary") or "")[:1000],
            }
            if not result.get("success", False):
                attempt.update({"accepted": False, "reason": result.get("error", "generated skill failed")})
                trace.reuse_attempts.append(attempt)
                continue

            verify_data, normalized = self._assess_reused_answer(task, candidate, result, workspace)
            effective_decision = verify_data.decision
            effective_confidence = verify_data.confidence
            if self._verifier_empty(verify_data):
                effective_decision = "recheck"
                effective_confidence = min(effective_confidence, 0.05)
            attempt.update(
                {
                    "verifier_decision": verify_data.decision,
                    "verifier_confidence": verify_data.confidence,
                    "effective_verifier_decision": effective_decision,
                    "effective_verifier_confidence": effective_confidence,
                    "verifier_issues": list(verify_data.issues),
                    "normalized_answer": normalized,
                }
            )

            if task.answer is not None:
                attempt_correct = answers_equal(task, normalized, task.answer)
                attempt["would_be_correct"] = attempt_correct
                record_skill_outcome(self.project_root, candidate["name"], task, attempt_correct)

            looks_canonical = self._canonical_like(task, normalized)
            policy = decide_reuse_acceptance(task, candidate, effective_decision, effective_confidence, looks_canonical)
            attempt["accepted"] = policy.accepted
            attempt["trust_level"] = policy.trust_level
            attempt["threshold"] = policy.threshold

            if not policy.accepted:
                attempt["reason"] = ", ".join(policy.reasons) or "reuse_rejected"
                backup = self._maybe_backup_candidate(task, candidate, verify_data, normalized, result, policy)
                if backup is not None:
                    attempt["backup_candidate"] = True
                    if best_backup is None or (backup["verifier"].get("confidence", 0.0), backup["score"]) > (best_backup["verifier"].get("confidence", 0.0), best_backup["score"]):
                        best_backup = backup
                trace.reuse_attempts.append(attempt)
                continue

            trace.reuse_attempts.append(attempt)
            trace.used_generated_skill = candidate["name"]
            trace.reuse_selected_score = float(candidate["score"])
            trace.final_answer_raw = verify_data.revised_answer if verify_data.revised_answer is not None else result.get("answer")
            trace.final_answer_normalized = normalized
            trace.verify_rounds.append({"source": "generated_skill_reuse", "skill_name": candidate["name"], **verify_data.to_dict()})
            if task.answer is not None:
                trace.correct = answers_equal(task, normalized, task.answer)
            return trace, None

        if candidates:
            last_reason = trace.reuse_attempts[-1].get("reason") if trace.reuse_attempts else "all_reuse_attempts_failed"
            trace.reuse_fallback_reason = last_reason
        return None, best_backup

    def _arbitrate_backup(self, task: TaskPacket, workspace: Path, trace: SolveTrace, backup: Optional[Dict[str, Any]]) -> SolveTrace:
        if not backup:
            return trace
        if not self._should_allow_backup_arbitration(task, trace, backup):
            trace.reuse_fallback_reason = trace.reuse_fallback_reason or "backup_arbitration_filtered"
            return trace

        base_norm = trace.final_answer_normalized
        base_verify = trace.verify_rounds[-1] if trace.verify_rounds else {}
        base_decision = str(base_verify.get("decision", "")).lower()
        base_conf = float(base_verify.get("confidence", 0.0) or 0.0)
        backup_norm = backup.get("normalized_answer")

        need_arbitration = False
        if base_norm in (None, ""):
            need_arbitration = True
        elif base_norm != backup_norm and (base_decision == "recheck" or base_conf < 0.60):
            need_arbitration = True

        if not need_arbitration:
            return trace

        pseudo_math = {
            "reasoning_steps": [f"Candidate answer proposed by saved skill {backup['skill_name']}."],
            "candidate_answer": backup_norm,
            "answer_confidence": float(backup.get("verifier", {}).get("confidence", 0.0) or 0.0),
            "needs_visual_recheck": False,
            "focus_questions": [],
            "normalization_notes": "Arbitration between fallback base-pipeline answer and saved-skill proposal.",
        }
        verify_payload = {
            "task": self._task_payload_without_gold(task),
            "evidence": trace.evidence.to_dict() if hasattr(trace.evidence, "to_dict") else trace.evidence,
            "math_result": pseudo_math,
            "profile": {
                "arbitration_mode": True,
                "generated_skill": backup["skill_name"],
                "candidate_score": backup.get("score"),
                "candidate_stats": backup.get("stats", {}),
                "candidate_matched_fields": backup.get("matched_fields", []),
            },
        }
        verify_result = self._call_skill("multimodal_verifier", verify_payload, workspace)
        if not verify_result["success"]:
            return trace
        verify_data = VerificationResult.from_dict(json.loads(verify_result.get("answer") or "{}"))
        normalized = self._normalize_via_skill(task, verify_data.revised_answer if verify_data.revised_answer is not None else backup_norm, workspace)
        if (not self._verifier_empty(verify_data)) and verify_data.decision == "accept" and self._canonical_like(task, normalized) and float(verify_data.confidence or 0.0) >= 0.45:
            trace.used_generated_skill = backup["skill_name"]
            trace.reuse_selected_score = float(backup.get("score", 0.0))
            trace.final_answer_raw = verify_data.revised_answer if verify_data.revised_answer is not None else backup_norm
            trace.final_answer_normalized = normalized
            trace.verify_rounds.append({"source": "generated_skill_arbitration", "skill_name": backup["skill_name"], **verify_data.to_dict()})
            if task.answer is not None:
                trace.correct = answers_equal(task, normalized, task.answer)
        return trace

    def _cross_check_reuse_with_base(self, task: TaskPacket, workspace: Path, reuse_trace: SolveTrace) -> SolveTrace:
        shadow = SolveTrace(task=task, evidence=EvidenceBundle(), workspace=str(workspace))
        if self.experiment_tag:
            shadow.experiment_tag = self.experiment_tag
        try:
            shadow = self._run_base_pipeline(task, workspace, shadow)
        except Exception:
            return reuse_trace

        reuse_norm = reuse_trace.final_answer_normalized
        base_norm = shadow.final_answer_normalized
        if not self._canonical_accept(task, reuse_norm):
            return shadow if self._canonical_accept(task, base_norm) else reuse_trace
        if not self._canonical_accept(task, base_norm):
            return reuse_trace
        if reuse_norm == base_norm:
            return reuse_trace

        base_verify = shadow.verify_rounds[-1] if shadow.verify_rounds else {}
        if str(base_verify.get("decision", "")).lower() == "accept" or not self._is_abstention_like(base_norm):
            shadow.used_generated_skill = None
            shadow.reuse_candidates = getattr(reuse_trace, 'reuse_candidates', [])
            shadow.reuse_attempts = getattr(reuse_trace, 'reuse_attempts', [])
            shadow.reuse_fallback_reason = "base_pipeline_overrode_conflicting_generated_skill"
            return shadow
        return reuse_trace

    def solve(self, task: TaskPacket) -> SolveTrace:
        workspace = self._create_workspace(task)
        trace = SolveTrace(task=task, evidence=EvidenceBundle(), workspace=str(workspace))
        if self.experiment_tag:
            trace.experiment_tag = self.experiment_tag
        save_json(workspace / "task.json", task.to_dict())

        try:
            reuse_trace, backup = self._attempt_reuse(task, workspace, trace)
            if reuse_trace is not None and reuse_trace.used_generated_skill:
                reuse_trace = self._cross_check_reuse_with_base(task, workspace, reuse_trace)
                reuse_trace = self._refresh_trace_outputs(reuse_trace, workspace)
                reuse_trace = self._maybe_apply_answer_reflection(task, reuse_trace, workspace)
                self._recompute_correct(task, reuse_trace)
                save_json(workspace / "trace.json", reuse_trace.to_dict())
                self._persist_trace(reuse_trace)
                return reuse_trace

            trace = self._run_base_pipeline(task, workspace, trace)
            trace = self._arbitrate_backup(task, workspace, trace, backup)
            trace = self._refresh_trace_outputs(trace, workspace)
            trace = self._maybe_apply_answer_reflection(task, trace, workspace)

            should_save_skill = False
            if self.allow_save_generated_skills and not trace.error and not trace.used_generated_skill:
                if task.answer is not None:
                    should_save_skill = bool(trace.correct)
                else:
                    should_save_skill = True
            if should_save_skill:
                saved_path = save_composed_skill_from_trace(trace, self.project_root)
                if saved_path is not None:
                    trace.saved_generated_skill = Path(saved_path).name

            self._recompute_correct(task, trace)
            save_json(workspace / "trace.json", trace.to_dict())
            self._persist_trace(trace)
            return trace

        except Exception as exc:
            trace.error = f"{type(exc).__name__}: {exc}"
            save_json(workspace / "trace.json", trace.to_dict())
            (workspace / "error.txt").write_text(trace.error + "\n", encoding="utf-8")
            self._persist_trace(trace)
            return trace

    def _persist_trace(self, trace: SolveTrace) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_json(self.config.trajectory_root / f"{ts}_{trace.task.sample_id}.json", trace.to_dict())


# === AB_SMOKE20_FIX3_ORCHESTRATOR ===
def _af_fix3_refresh_trace_outputs(self, *args, **kwargs):
    trace = kwargs.get("trace")
    task = kwargs.get("task")
    workspace = kwargs.get("workspace")

    for obj in args:
        if trace is None and hasattr(obj, "final_answer_raw") and hasattr(obj, "workspace"):
            trace = obj
        if task is None and hasattr(obj, "question") and hasattr(obj, "answer_type"):
            task = obj
        if workspace is None and isinstance(obj, (str, Path)):
            workspace = obj

    if trace is None:
        return None

    if task is None:
        task = getattr(trace, "task", None)
    if workspace is None:
        workspace = getattr(trace, "workspace", None)

    if task is None:
        return trace

    raw = getattr(trace, "final_answer_raw", None)

    try:
        ws = Path(str(workspace)) if workspace not in (None, "") else Path(str(getattr(trace, "workspace", ".")))
        normalized = self._normalize_via_skill(task, raw, ws)
    except Exception:
        try:
            normalized = normalize_answer(task, raw)
        except Exception:
            normalized = getattr(trace, "final_answer_normalized", None)

    try:
        trace.final_answer_normalized = normalized
    except Exception:
        pass

    try:
        if getattr(trace, "error", None) is None and getattr(task, "answer", None) is not None:
            trace.correct = answers_equal(task, normalized, task.answer)
    except Exception:
        pass

    return trace

if not hasattr(MultimodalMetaAgent, "_refresh_trace_outputs"):
    MultimodalMetaAgent._refresh_trace_outputs = _af_fix3_refresh_trace_outputs
from .agegap_shadow_runtime_patch import apply_agegap_shadow_runtime_patch as _apply_agegap_shadow_runtime_patch
_apply_agegap_shadow_runtime_patch(MultimodalMetaAgent, globals())



# --- fix7 runtime patch bootstrap ---
try:
    from .fix7_runtime_patch import apply_fix7_runtime_patch as _af_apply_fix7_runtime_patch
    _af_apply_fix7_runtime_patch(MultimodalMetaAgent)
except Exception:
    pass
# --- end fix7 runtime patch bootstrap ---
# --- fix8 seed reflection runtime patch bootstrap ---
try:
    from .fix8_runtime_patch import apply_fix8_seed_reflection_patch as _af_apply_fix8_seed_reflection_patch
    _af_apply_fix8_seed_reflection_patch(MultimodalMetaAgent)
except Exception:
    pass
# --- end fix8 seed reflection runtime patch bootstrap ---
