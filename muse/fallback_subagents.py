from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .age_gap_reasoner import try_age_gap_answer
from .answering import normalize_answer
from .config import load_runtime_config
from .function_plot_reasoner import try_function_plot_answer
from .llm_clients import OpenAIStyleClient
from .public_figure_reasoner import maybe_public_figure_grounding, try_conservative_post_wwii_count
from .schemas import TaskPacket
from .synthetic_scene_reasoner import solve_synthetic_subtraction


def _pick_model_config(config):
    if getattr(config, "orchestrator_model", None) and getattr(config.orchestrator_model, "enabled", False):
        return config.orchestrator_model
    return config.reasoning_model


def _client(project_root: Path) -> OpenAIStyleClient:
    cfg = load_runtime_config(project_root)
    return OpenAIStyleClient(_pick_model_config(cfg))


def _task(payload: Dict[str, Any]) -> TaskPacket:
    return TaskPacket.from_dict(payload["task"])


def _task_payload_without_gold(task: TaskPacket) -> Dict[str, Any]:
    payload = task.to_dict()
    payload.pop("answer", None)
    return payload


def _safe_json_text(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _as_list_str(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _sanitize_visual_focus_answers(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        supporting = _as_list_str(item.get("supporting_clues"))
        observations = _as_list_str(item.get("visual_observations"))
        if not observations:
            observations = supporting
        if not observations:
            rationale = str(item.get("rationale") or "").strip()
            observations = [rationale] if rationale else []
        out.append({
            "question": str(item.get("question") or "").strip(),
            "answer": None,
            "confidence": _coerce_confidence(item.get("confidence"), 0.0),
            "evidence_type": str(item.get("evidence_type") or "visual_evidence"),
            "visual_observations": observations,
            "supporting_clues": supporting,
            "person_age_ranges": item.get("person_age_ranges") if isinstance(item.get("person_age_ranges"), list) else [],
            "relative_order": item.get("relative_order") if isinstance(item.get("relative_order"), list) else [],
        })
    return out




def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return float(default)
    s = str(value).strip().lower()
    mapping = {
        "very high": 0.95,
        "high": 0.85,
        "medium": 0.60,
        "moderate": 0.60,
        "low": 0.35,
        "very low": 0.15,
    }
    if s in mapping:
        return mapping[s]
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return float(default)
    return float(default)


def _visual_complete_with_fallback(client: OpenAIStyleClient, system_prompt: str, prompt: str, task: TaskPacket):
    image_paths = task.existing_image_paths()
    if image_paths:
        try:
            return client.complete_multimodal_json(system_prompt, prompt, image_paths, max_tokens=2200)
        except Exception as exc:
            msg = str(exc).lower()
            if ("parse the json body" in msg) or ("invalid_request_error" in msg) or ("badrequesterror" in msg):
                retry_prompt = (
                    prompt
                    + "\n\nImage attachment transport failed. "
                    + "If the question is answerable from the textual passage/choices alone, answer conservatively from text only. "
                    + "Otherwise return cannot_determine."
                )
                return client.complete_json(system_prompt, retry_prompt, max_tokens=1800)
            raise
    return client.complete_json(system_prompt, prompt, max_tokens=1600)

def _call_visual_detail_agent(payload: Dict[str, Any], workspace: Path, project_root: Path) -> Dict[str, Any]:
    task = _task(payload)
    focus_questions = [str(x).strip() for x in (payload.get("focus_questions") or []) if str(x).strip()]
    cfg = load_runtime_config(project_root)
    if getattr(cfg, "mock_mode", False):
        data = {
            "scene_type": str((task.metadata or {}).get("context") or "unknown"),
            "visual_facts": ["MOCK_MODE enabled; no visual extraction performed."],
            "uncertainties": [],
            "focus_answers": [{"question": q, "answer": None, "confidence": 0.0, "evidence_type": "mock"} for q in focus_questions],
        }
        return {"success": True, "answer": _safe_json_text(data), "summary": "mock visual_detail_agent"}

    system_prompt = (
        "You are the visual_detail_agent for a MathVista-style multimodal pipeline. "
        "Look at the image and extract only grounded visual evidence. Return JSON only with keys: "
        "scene_type, visual_facts, uncertainties, focus_answers.\n"
        "Rules:\n"
        "- Prefer explicit visible text, numbers, labels, flags, logos, jersey names/numbers, podium seals, event branding, charts, and object counts.\n"
        "- For public figures, do NOT rely on face recognition alone. Only mention identity hypotheses when supported by visible text or strong non-facial contextual clues already in the image.\n"
        "- For age questions, provide only coarse age ranges with clear uncertainty if exact ages are not visible.\n"
        "- If a focus question cannot be answered from the image, say so explicitly.\n"
        "- Do NOT solve the problem or provide a final answer/choice in the visual stage. The answer field in focus_answers must be null.\n"
        "- Put only visible observations in visual_facts/supporting_clues; do not perform algebra, geometry solving, or pattern completion.\n"
        "Each focus_answers item should be an object with keys: question, answer, confidence, evidence_type, visual_observations, supporting_clues, person_age_ranges, relative_order."
    )
    user_parts = [
        f"Question: {task.question}",
        f"Question type: {task.question_type}",
        f"Answer type: {task.answer_type}",
        f"Scene/context hint: {(task.metadata or {}).get('context')}",
    ]
    if task.choices:
        user_parts.append("Choices:\n" + "\n".join(f"({chr(65+i)}) {choice}" for i, choice in enumerate(task.choices)))
    if focus_questions:
        user_parts.append("Focus questions to answer individually:\n" + "\n".join(f"- {q}" for q in focus_questions))
    else:
        user_parts.append("No follow-up focus questions were provided. Still extract decisive facts and any relevant age/context clues.")
    user_parts.append("Return JSON only.")
    prompt = "\n\n".join(user_parts)

    client = _client(project_root)
    raw = _visual_complete_with_fallback(client, system_prompt, prompt, task)

    data = raw if isinstance(raw, dict) else {}
    out = {
        "scene_type": str(data.get("scene_type") or (task.metadata or {}).get("context") or "unknown"),
        "visual_facts": _as_list_str(data.get("visual_facts")),
        "uncertainties": _as_list_str(data.get("uncertainties")),
        "focus_answers": _sanitize_visual_focus_answers(data.get("focus_answers")),
    }
    if focus_questions and not out["focus_answers"]:
        out["focus_answers"] = [{"question": q, "answer": None, "confidence": 0.0, "evidence_type": "cannot_determine", "visual_observations": ["Model did not return grounded visual observations."], "supporting_clues": []} for q in focus_questions]
    return {"success": True, "answer": _safe_json_text(out), "summary": "fallback visual_detail_agent"}


def _call_math_reason_agent(payload: Dict[str, Any], workspace: Path, project_root: Path) -> Dict[str, Any]:
    task = _task(payload)
    evidence = payload.get("evidence") or {}

    for fn in (solve_synthetic_subtraction, try_function_plot_answer):
        result = fn(task, evidence)
        if result is not None:
            return {"success": True, "answer": _safe_json_text(result), "summary": f"{fn.__name__}"}

    result = maybe_public_figure_grounding(task, evidence, project_root, workspace)
    if result is not None:
        return {"success": True, "answer": _safe_json_text(result), "summary": "public_figure_grounding"}

    result = try_conservative_post_wwii_count(task, evidence)
    if result is not None:
        return {"success": True, "answer": _safe_json_text(result), "summary": "post_wwii_conservative_count"}

    result = try_age_gap_answer(task, evidence)
    if result is not None:
        return {"success": True, "answer": _safe_json_text(result), "summary": "age_gap_reasoner"}

    cfg = load_runtime_config(project_root)
    if getattr(cfg, "mock_mode", False):
        mock_answer = (task.metadata or {}).get("mock_answer")
        if mock_answer is None and task.choices:
            mock_answer = task.choices[0]
        data = {
            "reasoning_steps": ["MOCK_MODE enabled.", "Returning the deterministic mock answer from task metadata."],
            "candidate_answer": mock_answer,
            "answer_confidence": 1.0,
            "needs_visual_recheck": False,
            "focus_questions": [],
            "normalization_notes": "mock_mode_math_reason_agent",
        }
        return {"success": True, "answer": _safe_json_text(data), "summary": "mock math_reason_agent"}

    system_prompt = (
        "You are the math_reason_agent in a multimodal QA pipeline. You receive a task and structured visual evidence, and must return JSON only. "
        "Required keys: reasoning_steps (list), candidate_answer, answer_confidence, needs_visual_recheck, focus_questions, normalization_notes.\n"
        "Be concise and canonical. For integer/float questions, candidate_answer should be a scalar or null. If evidence is insufficient, set candidate_answer=null and ask targeted focus_questions.\n"
        "Treat focus_answers from the visual stage as observations only, not as proposed answers. If your reasoning derives a value or option, candidate_answer must exactly match that derivation."
    )
    user_prompt = (
        f"Task:\n{json.dumps(_task_payload_without_gold(task), ensure_ascii=False)}\n\n"
        f"Structured visual evidence:\n{json.dumps(evidence, ensure_ascii=False)[:14000]}\n\n"
        "Return JSON only."
    )
    client = _client(project_root)
    raw = client.complete_json(system_prompt, user_prompt, max_tokens=1600)
    data = raw if isinstance(raw, dict) else {}
    out = {
        "reasoning_steps": _as_list_str(data.get("reasoning_steps")),
        "candidate_answer": data.get("candidate_answer", data.get("answer")),
        "answer_confidence": _coerce_confidence(data.get("answer_confidence", data.get("confidence", 0.0)), 0.0),
        "needs_visual_recheck": bool(data.get("needs_visual_recheck", False)),
        "focus_questions": _as_list_str(data.get("focus_questions")),
        "normalization_notes": str(data.get("normalization_notes") or "llm_fallback_math_reason_agent"),
    }
    return {"success": True, "answer": _safe_json_text(out), "summary": "fallback math_reason_agent"}


def _call_multimodal_verifier(payload: Dict[str, Any], workspace: Path, project_root: Path) -> Dict[str, Any]:
    task = _task(payload)
    evidence = payload.get("evidence") or {}
    math_result = payload.get("math_result") or {}
    candidate = math_result.get("candidate_answer")
    normalized = normalize_answer(task, candidate)
    notes = str(math_result.get("normalization_notes") or "")
    steps_text = " ".join(_as_list_str(math_result.get("reasoning_steps"))).lower()
    issues: List[str] = []
    follow_up = _as_list_str(math_result.get("focus_questions"))
    conf = _coerce_confidence(math_result.get("answer_confidence", 0.0), 0.0)

    decision = "recheck"
    revised_answer = candidate
    needs_llm_verification = "llm_fallback" in notes.lower()

    if normalized is None:
        issues.append("No canonical candidate answer was provided.")
    elif any(tag in notes for tag in [
        "public_figure_grounding:",
        "post_wwii_conservative_count",
        "synthetic_subtraction_rule_based",
        "function_plot_discrete",
        "age_gap_estimator",
    ]):
        decision = "accept"
        conf = max(conf, 0.68)
    elif ("birth year" in steps_text or "wikidata" in notes.lower()) and normalized not in (None, ""):
        decision = "accept"
        conf = max(conf, 0.62)
    elif not needs_llm_verification and task.answer_type in {"integer", "float"} and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(normalized)):
        decision = "accept"
        conf = max(conf, 0.55)
    elif not needs_llm_verification and normalized not in (None, ""):
        decision = "accept"
        conf = max(conf, 0.5)

    if decision != "accept" or needs_llm_verification:
        cfg = load_runtime_config(project_root)
        if not getattr(cfg, "mock_mode", False):
            try:
                system_prompt = (
                    "You are the multimodal_verifier. Decide whether the candidate answer is grounded enough to accept. "
                    "Return JSON only with keys: decision, issues, revised_answer, follow_up_visual_questions, confidence.\n"
                    "Rules:\n"
                    "- Check internal consistency before accepting. If reasoning_steps derive one value/choice but candidate_answer is different, do not accept the candidate.\n"
                    "- If the derivation clearly supports another listed choice or scalar, set decision='accept' and revised_answer to the derived value.\n"
                    "- If the evidence is only a visual observation and does not support the reasoning, set decision='recheck' with targeted follow_up_visual_questions.\n"
                    "- Do not accept answers justified by contradictory language such as 'therefore X' followed by an unsupported 'however Y'."
                )
                user_prompt = (
                    f"Task:\n{json.dumps(_task_payload_without_gold(task), ensure_ascii=False)}\n\n"
                    f"Evidence:\n{json.dumps(evidence, ensure_ascii=False)[:12000]}\n\n"
                    f"Math result:\n{json.dumps(math_result, ensure_ascii=False)}\n\n"
                    "Return JSON only."
                )
                raw = _client(project_root).complete_json(system_prompt, user_prompt, max_tokens=900)
                if isinstance(raw, dict):
                    decision = str(raw.get("decision") or decision)
                    issues = _as_list_str(raw.get("issues")) or issues
                    revised_answer = raw.get("revised_answer", revised_answer)
                    follow_up = _as_list_str(raw.get("follow_up_visual_questions")) or follow_up
                    conf = _coerce_confidence(raw.get("confidence", conf), conf)
            except Exception:
                pass

    if decision != "accept" and not issues:
        issues.append("Evidence was insufficient for a confident acceptance.")

    out = {
        "decision": decision,
        "issues": issues,
        "revised_answer": revised_answer if decision == "accept" else revised_answer,
        "follow_up_visual_questions": follow_up,
        "confidence": conf,
    }
    return {"success": True, "answer": _safe_json_text(out), "summary": "fallback multimodal_verifier"}


def _call_answer_normalizer(payload: Dict[str, Any], workspace: Path, project_root: Path) -> Dict[str, Any]:
    task = _task(payload)
    normalized = normalize_answer(task, payload.get("candidate_answer"))
    return {"success": True, "answer": normalized, "summary": "fallback answer_normalizer"}


def call_builtin_skill(skill_name: str, payload: Dict[str, Any], workspace: Path, project_root: Path) -> Optional[Dict[str, Any]]:
    handlers = {
        "visual_detail_agent": _call_visual_detail_agent,
        "math_reason_agent": _call_math_reason_agent,
        "multimodal_verifier": _call_multimodal_verifier,
        "answer_normalizer": _call_answer_normalizer,
    }
    handler = handlers.get(skill_name)
    if handler is None:
        return None
    try:
        return handler(payload, workspace, project_root)
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}
