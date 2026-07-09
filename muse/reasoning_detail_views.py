
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_MAX_TEXT = 280
_MAX_STEPS = 6
_MAX_FACTS = 5
_MAX_ISSUES = 5
_MAX_QUESTIONS = 5


def _short_text(value: Any, limit: int = _MAX_TEXT) -> Optional[str]:
    if value is None:
        return None
    text = str(value).replace("\n", " ").strip()
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _short_list(items: Any, limit_items: int = 5, limit_text: int = _MAX_TEXT) -> List[str]:
    if not items:
        return []
    out: List[str] = []
    for item in list(items)[:limit_items]:
        t = _short_text(item, limit_text)
        if t:
            out.append(t)
    return out


def _focus_questions_for_visual_round(trace: Dict[str, Any], round_idx: int) -> List[str]:
    if round_idx <= 0:
        return []
    verify_rounds = trace.get("verify_rounds") or []
    math_rounds = trace.get("math_rounds") or []
    if round_idx - 1 < len(verify_rounds):
        fq = verify_rounds[round_idx - 1].get("follow_up_visual_questions") or []
        if fq:
            return [str(x) for x in fq]
    if round_idx - 1 < len(math_rounds):
        fq = math_rounds[round_idx - 1].get("focus_questions") or []
        if fq:
            return [str(x) for x in fq]
    return []


def _visual_summary(visual_round: Dict[str, Any]) -> Dict[str, Any]:
    facts = visual_round.get("visual_facts") or []
    uncertainties = visual_round.get("uncertainties") or []
    focus_answers = visual_round.get("focus_answers") or []
    scene_type = visual_round.get("scene_type")
    focus_summary = []
    for item in focus_answers[:3]:
        if isinstance(item, dict):
            focus_summary.append({
                "question": _short_text(item.get("question")),
                "answer": item.get("answer"),
                "confidence": item.get("confidence"),
                "evidence_type": item.get("evidence_type"),
            })
        else:
            focus_summary.append(_short_text(item))
    fact_texts = []
    for fact in facts[:_MAX_FACTS]:
        if isinstance(fact, dict):
            fact_texts.append(_short_text(fact.get("fact") or fact.get("value") or fact))
        else:
            fact_texts.append(_short_text(fact))
    return {
        "scene_type": _short_text(scene_type),
        "visual_fact_count": len(facts),
        "visual_facts_sample": [x for x in fact_texts if x],
        "uncertainty_count": len(uncertainties),
        "uncertainties_sample": _short_list(uncertainties, _MAX_FACTS),
        "focus_answer_count": len(focus_answers),
        "focus_answers_sample": focus_summary,
    }


def _math_summary(math_round: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_answer": math_round.get("candidate_answer"),
        "answer_confidence": math_round.get("answer_confidence"),
        "needs_visual_recheck": math_round.get("needs_visual_recheck"),
        "reasoning_steps": _short_list(math_round.get("reasoning_steps") or [], _MAX_STEPS),
        "focus_questions": _short_list(math_round.get("focus_questions") or [], _MAX_QUESTIONS),
        "normalization_notes": _short_text(math_round.get("normalization_notes")),
    }


def _verify_summary(verify_round: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "decision": verify_round.get("decision"),
        "revised_answer": verify_round.get("revised_answer"),
        "confidence": verify_round.get("confidence"),
        "issues": _short_list(verify_round.get("issues") or [], _MAX_ISSUES),
        "follow_up_visual_questions": _short_list(verify_round.get("follow_up_visual_questions") or [], _MAX_QUESTIONS),
        "source": verify_round.get("source"),
        "skill_name": verify_round.get("skill_name"),
    }


def _reuse_candidates_summary(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for item in (trace.get("reuse_candidates") or [])[:10]:
        out.append({
            "skill_name": item.get("name"),
            "score": item.get("score"),
            "matched_fields": list(item.get("matched_fields") or [])[:8],
        })
    return out


def _reuse_attempt_summary(attempt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "skill_name": attempt.get("skill_name"),
        "score": attempt.get("score"),
        "success": attempt.get("success"),
        "accepted": attempt.get("accepted"),
        "normalized_answer": attempt.get("normalized_answer"),
        "raw_answer": _short_text(attempt.get("raw_answer")),
        "verifier_decision": attempt.get("effective_verifier_decision", attempt.get("verifier_decision")),
        "verifier_confidence": attempt.get("effective_verifier_confidence", attempt.get("verifier_confidence")),
        "reason": _short_text(attempt.get("reason")),
        "matched_fields": list(attempt.get("matched_fields") or [])[:8],
        "verifier_issues": _short_list(attempt.get("verifier_issues") or [], _MAX_ISSUES),
    }


def build_branch_readable(branch_name: str, branch_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not branch_payload.get("present"):
        return {
            "branch_summary": {
                "status": "missing",
                "note": branch_payload.get("note"),
            },
            "action_timeline": [],
            "action_timeline_text": [],
        }

    trace = branch_payload.get("trace") or {}
    actions: List[Dict[str, Any]] = []

    def add_action(actor: str, action_type: str, title: str, request: Any = None, response: Any = None, result: Any = None, notes: Any = None) -> None:
        idx = len(actions) + 1
        actions.append({
            "action": f"action{idx}",
            "actor": actor,
            "action_type": action_type,
            "title": title,
            "request": request,
            "response": response,
            "result": result,
            "notes": notes,
        })

    if trace.get("reuse_candidates"):
        add_action(
            "orchestrator",
            "reuse_search",
            "Scored generated-skill reuse candidates",
            request={"allow_generated_skill_reuse": True},
            response={"reuse_candidates": _reuse_candidates_summary(trace)},
            result={"selected_score": trace.get("reuse_selected_score")},
        )

    for attempt in trace.get("reuse_attempts") or []:
        add_action(
            "orchestrator",
            "reuse_attempt",
            f"Attempted generated skill: {attempt.get('skill_name')}",
            request={"matched_fields": list(attempt.get("matched_fields") or [])[:8], "score": attempt.get("score")},
            response=_reuse_attempt_summary(attempt),
            result={
                "accepted": attempt.get("accepted"),
                "normalized_answer": attempt.get("normalized_answer"),
                "reason": _short_text(attempt.get("reason")),
            },
        )

    if branch_name == "baseline_model" or trace.get("branch_type") == "baseline_model":
        add_action(
            "baseline_model",
            "direct_model_call",
            "Direct multimodal baseline call",
            request={
                "model_name": trace.get("model_name"),
                "question_type": (trace.get("task") or {}).get("question_type"),
                "answer_type": (trace.get("task") or {}).get("answer_type"),
            },
            response={
                "reasoning_steps": _short_list(trace.get("reasoning_steps") or [], _MAX_STEPS),
                "confidence": trace.get("confidence"),
            },
            result={
                "final_answer_raw": trace.get("final_answer_raw"),
                "final_answer_normalized": trace.get("final_answer_normalized"),
                "correct": trace.get("correct"),
            },
        )
    else:
        visual_rounds = trace.get("visual_rounds") or []
        math_rounds = trace.get("math_rounds") or []
        verify_rounds = trace.get("verify_rounds") or []
        max_rounds = max(len(visual_rounds), len(math_rounds), len(verify_rounds))

        for round_idx in range(max_rounds):
            if round_idx < len(visual_rounds):
                add_action(
                    "visual_detail_agent",
                    "subagent_call",
                    f"Visual round {round_idx + 1}",
                    request={"focus_questions": _short_list(_focus_questions_for_visual_round(trace, round_idx), _MAX_QUESTIONS)},
                    response=_visual_summary(visual_rounds[round_idx]),
                    result=None,
                )
            if round_idx < len(math_rounds):
                add_action(
                    "math_reason_agent",
                    "subagent_call",
                    f"Math round {round_idx + 1}",
                    request=None,
                    response=_math_summary(math_rounds[round_idx]),
                    result={
                        "candidate_answer": math_rounds[round_idx].get("candidate_answer"),
                        "needs_visual_recheck": math_rounds[round_idx].get("needs_visual_recheck"),
                    },
                )
            if round_idx < len(verify_rounds):
                add_action(
                    "multimodal_verifier",
                    "subagent_call",
                    f"Verifier round {round_idx + 1}",
                    request=None,
                    response=_verify_summary(verify_rounds[round_idx]),
                    result={
                        "decision": verify_rounds[round_idx].get("decision"),
                        "revised_answer": verify_rounds[round_idx].get("revised_answer"),
                    },
                )

    add_action(
        "orchestrator",
        "finalize",
        "Finalize branch output",
        request=None,
        response={
            "used_generated_skill": branch_payload.get("used_generated_skill"),
            "saved_generated_skill": trace.get("saved_generated_skill"),
            "reuse_fallback_reason": trace.get("reuse_fallback_reason"),
        },
        result={
            "prediction": branch_payload.get("prediction"),
            "final_answer_raw": branch_payload.get("final_answer_raw"),
            "final_answer_normalized": branch_payload.get("final_answer_normalized"),
            "correct": branch_payload.get("correct"),
            "error": branch_payload.get("error"),
        },
    )

    timeline_text: List[str] = []
    for item in actions:
        line = f"{item['action']} [{item['actor']}] {item['title']}"
        result = item.get("result") or {}
        if isinstance(result, dict):
            if result.get("prediction") is not None:
                line += f" -> prediction={result.get('prediction')!r}"
            elif result.get("final_answer_normalized") is not None:
                line += f" -> normalized={result.get('final_answer_normalized')!r}"
            elif result.get("candidate_answer") is not None:
                line += f" -> candidate={result.get('candidate_answer')!r}"
            elif result.get("decision") is not None:
                line += f" -> decision={result.get('decision')!r}"
        timeline_text.append(line)

    summary = {
        "status": "ok" if not branch_payload.get("error") else "error",
        "prediction": branch_payload.get("prediction"),
        "correct": branch_payload.get("correct"),
        "error": branch_payload.get("error"),
        "used_generated_skill": branch_payload.get("used_generated_skill"),
        "saved_generated_skill": trace.get("saved_generated_skill"),
        "num_actions": len(actions),
        "num_visual_rounds": len(trace.get("visual_rounds") or []),
        "num_math_rounds": len(trace.get("math_rounds") or []),
        "num_verify_rounds": len(trace.get("verify_rounds") or []),
        "num_reuse_candidates": len(trace.get("reuse_candidates") or []),
        "num_reuse_attempts": len(trace.get("reuse_attempts") or []),
        "reuse_fallback_reason": trace.get("reuse_fallback_reason"),
    }

    return {
        "branch_summary": summary,
        "action_timeline": actions,
        "action_timeline_text": timeline_text,
    }


def make_pid_markdown(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# PID {data.get('pid')}")
    lines.append("")
    lines.append(f"**Question:** {data.get('question')}")
    lines.append("")
    lines.append(f"**Gold:** {data.get('gold')}")
    lines.append("")
    branches = data.get("branches") or {}
    for branch_name, branch in branches.items():
        lines.append(f"## {branch_name}")
        readable = {
            "branch_summary": branch.get("branch_summary"),
            "action_timeline": branch.get("action_timeline"),
        }
        summary = readable["branch_summary"] or {}
        lines.append(f"- prediction: {summary.get('prediction')!r}")
        lines.append(f"- correct: {summary.get('correct')}")
        lines.append(f"- used_generated_skill: {summary.get('used_generated_skill')!r}")
        lines.append(f"- error: {summary.get('error')!r}")
        lines.append("")
        for action in readable["action_timeline"] or []:
            lines.append(f"### {action.get('action')} · {action.get('actor')} · {action.get('title')}")
            req = action.get("request")
            resp = action.get("response")
            res = action.get("result")
            if req not in (None, {}, []):
                lines.append("**request**")
                lines.append("```json")
                lines.append(json.dumps(req, ensure_ascii=False, indent=2))
                lines.append("```")
            if resp not in (None, {}, []):
                lines.append("**response**")
                lines.append("```json")
                lines.append(json.dumps(resp, ensure_ascii=False, indent=2))
                lines.append("```")
            if res not in (None, {}, []):
                lines.append("**result**")
                lines.append("```json")
                lines.append(json.dumps(res, ensure_ascii=False, indent=2))
                lines.append("```")
            notes = action.get("notes")
            if notes not in (None, {}, []):
                lines.append("**notes**")
                lines.append("```json")
                lines.append(json.dumps(notes, ensure_ascii=False, indent=2))
                lines.append("```")
            lines.append("")
    return "\n".join(lines)
