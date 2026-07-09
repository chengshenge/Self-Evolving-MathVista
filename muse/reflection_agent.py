
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_runtime_config
from .llm_clients import OpenAIStyleClient
from .schemas import SolveTrace


REFLECTION_SYSTEM_PROMPT = """You are reflection_agent, a narrow-expert synthesis subagent for a multimodal reasoning system.
You will read a FAILED seed-stage trace and propose a better narrow expert profile for a retry.
Your job is to diagnose why the current pipeline failed and produce transferable hints, not to memorize the sample.
Return JSON only with keys:
- failure_hypothesis: short explanation of the likely failure mode
- visual_hint: 1-2 concise sentences telling the visual stage what to prioritize next time
- reasoning_hint: 1-2 concise sentences telling the math/reasoning stage how to avoid the failure
- verifier_hint: 1-2 concise sentences telling the verifier what to reject or require
- keywords: list[str] of concise reusable keywords
- retrieval_text_bge: one short paragraph for semantic text retrieval of this expert
- retrieval_text_clip: one short paragraph that visually describes what kinds of images/questions this expert applies to
- should_retry: boolean
- confidence: float in [0,1]
Do NOT include the final gold answer verbatim inside any hint.
Do NOT produce sample-specific memorization like names, exact dates, or 'the answer is ...' unless they are clearly reusable category-level cues.
"""


def _pick_model_config(cfg):
    if getattr(cfg, "orchestrator_model", None) and getattr(cfg.orchestrator_model, "enabled", False):
        return cfg.orchestrator_model
    return cfg.reasoning_model


def _truncate_list(xs: Any, n: int = 8) -> List[Any]:
    if not isinstance(xs, list):
        return []
    return xs[:n]


def _safe_trace_dict(trace: SolveTrace) -> Dict[str, Any]:
    if hasattr(trace, "to_dict"):
        try:
            data = trace.to_dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _remove_answer_field(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    task_dict = dict(task_dict or {})
    task_dict.pop("answer", None)
    return task_dict


def _summarize_for_reflection(trace: SolveTrace, gold_answer: Optional[str], round_index: int) -> Dict[str, Any]:
    data = _safe_trace_dict(trace)
    task = _remove_answer_field(data.get("task") or {})
    evidence = data.get("evidence") or {}
    visual_rounds = data.get("visual_rounds") or []
    math_rounds = data.get("math_rounds") or []
    verify_rounds = data.get("verify_rounds") or []

    payload: Dict[str, Any] = {
        "round_index": round_index,
        "task": task,
        "final_answer_raw": data.get("final_answer_raw"),
        "final_answer_normalized": data.get("final_answer_normalized"),
        "correct": data.get("correct"),
        "used_generated_skill": data.get("used_generated_skill"),
        "scene_type": evidence.get("scene_type"),
        "visual_facts": _truncate_list(evidence.get("visual_facts"), 10),
        "uncertainties": _truncate_list(evidence.get("uncertainties"), 10),
        "focus_answers": _truncate_list(evidence.get("focus_answers"), 6),
        "visual_rounds": _truncate_list(visual_rounds, 2),
        "math_rounds": _truncate_list(math_rounds, 2),
        "verify_rounds": _truncate_list(verify_rounds, 2),
    }
    if gold_answer is not None:
        payload["teacher_signal"] = {
            "gold_answer": str(gold_answer),
            "note": "Use only to diagnose the failure and improve future transferable hints. Do not copy it into hints."
        }
    return payload


def _sentences(text: str, max_sentences: int = 2) -> str:
    parts = re.split(r'(?<=[.!?])\s+', str(text).strip())
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:max_sentences]).strip()


def _sanitize_hint(text: Any, *, gold_answer: Optional[str]) -> str:
    s = str(text or "").strip()
    s = re.sub(r"(?i)\b(the\s+)?correct answer is\b.*", "", s)
    s = re.sub(r"(?i)\btherefore\b.*", "", s)
    if gold_answer:
        try:
            s = re.sub(re.escape(str(gold_answer).strip()), "[ANSWER]", s, flags=re.IGNORECASE)
        except Exception:
            pass
    s = _sentences(s, max_sentences=2)
    return s


def _sanitize_keywords(keywords: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    if not isinstance(keywords, list):
        return out
    for item in keywords:
        t = str(item).strip().lower()
        if not t or len(t) > 64:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:16]


def _fallback_profile(summary: Dict[str, Any]) -> Dict[str, Any]:
    task = summary.get("task", {})
    meta = task.get("metadata", {}) if isinstance(task, dict) else {}
    q = str(task.get("question") or "").lower()
    context = str(meta.get("context") or "").lower()
    source = str(meta.get("source") or "").lower()

    if "age gap" in q or ("kvqa" in source and ("age" in q or "born after" in q)):
        return {
            "failure_hypothesis": "The pipeline over-trusted appearance-based age estimation without strong grounding.",
            "visual_hint": "Prioritize readable names, inscriptions, captions, jersey text, or surrounding context before estimating age from appearance.",
            "reasoning_hint": "Avoid midpoint arithmetic over broad visual age ranges unless identity grounding or explicit age evidence is present.",
            "verifier_hint": "Reject precise integer age-gap answers derived only from uncertain appearance ranges.",
            "keywords": ["age gap", "identity grounding", "public figure", "kvqa", "appearance-only evidence"],
            "retrieval_text_bge": "Natural-image identity or age-gap questions where the correct answer depends on grounding people via contextual clues instead of pure appearance-based midpoint estimation.",
            "retrieval_text_clip": "A natural image with one or more people where names, handwriting, inscriptions, or contextual visual cues may help identify them better than facial age guessing.",
            "should_retry": True,
            "confidence": 0.72,
        }
    if "subtract all" in q or ("synthetic scene" in context and "how many objects are left" in q):
        return {
            "failure_hypothesis": "The pipeline relied on an unstable parsed inventory instead of targeted counting.",
            "visual_hint": "Count the total visible objects first, then count each subtraction descriptor explicitly, and keep the counts separate.",
            "reasoning_hint": "Prefer targeted descriptor counts over inventory reconstruction when computing the final remaining-object answer.",
            "verifier_hint": "Reject subtraction answers if the total count and descriptor counts are not explicitly grounded.",
            "keywords": ["synthetic counting", "clevr-math", "descriptor count", "remaining objects"],
            "retrieval_text_bge": "Synthetic-scene subtraction or counting questions where the answer should come from total count plus descriptor-specific counts rather than a noisy parsed inventory.",
            "retrieval_text_clip": "A synthetic tabletop scene with many colored objects where the task asks how many remain after subtracting certain described objects.",
            "should_retry": True,
            "confidence": 0.68,
        }
    if any(x in q for x in ["odd", "even", "continuous", "discontinuous"]) or "functionqa" in source:
        return {
            "failure_hypothesis": "The pipeline collapsed a graph-property question into a yes/no answer instead of preserving the property label.",
            "visual_hint": "Focus on visual graph properties like symmetry, continuity breaks, and parity-relevant structure before answering.",
            "reasoning_hint": "Preserve property labels such as odd, even, or neither instead of translating them into generic yes/no text.",
            "verifier_hint": "Reject yes/no answers when the task expects a graph property label.",
            "keywords": ["function plot", "odd even", "continuity", "graph property"],
            "retrieval_text_bge": "Function or plot questions where the system must identify a property like odd, even, or continuous rather than answer with a generic yes/no.",
            "retrieval_text_clip": "A graph or function plot where visual symmetry or discontinuities determine the correct property label.",
            "should_retry": True,
            "confidence": 0.66,
        }
    return {
        "failure_hypothesis": "The pipeline likely emphasized generic descriptions instead of the decisive evidence needed for this task.",
        "visual_hint": "Extract the smallest set of decisive visual facts that directly constrain the final answer.",
        "reasoning_hint": "Use only answer-relevant facts and avoid verbose or weakly grounded intermediate reasoning.",
        "verifier_hint": "Prefer concise canonical answers and reject unsupported elaboration.",
        "keywords": ["narrow expert", "decisive visual facts", "canonical answer"],
        "retrieval_text_bge": "A narrow expert for multimodal questions where success depends on extracting decisive evidence and avoiding weakly grounded reasoning.",
        "retrieval_text_clip": "An image-question pair where a few decisive visual cues determine the answer and verbose generic descriptions are harmful.",
        "should_retry": True,
        "confidence": 0.55,
    }


def reflect_failed_trace(
    trace: SolveTrace,
    project_root: str | Path,
    *,
    prior_profile: Optional[Dict[str, Any]] = None,
    round_index: int = 1,
    max_rounds: int = 2,
    gold_answer: Optional[str] = None,
) -> Dict[str, Any]:
    project_root = Path(project_root)
    cfg = load_runtime_config(project_root)
    client = OpenAIStyleClient(_pick_model_config(cfg))

    payload = {
        "failed_trace_summary": _summarize_for_reflection(trace, gold_answer=gold_answer, round_index=round_index),
        "prior_profile": prior_profile or {},
        "instruction": (
            f"Generate a better narrow-expert profile for retry round {round_index} of {max_rounds}. "
            "The output should be reusable for similar tasks and should not memorize this specific sample."
        ),
    }

    raw: Dict[str, Any] = {}
    try:
        raw = client.complete_json(
            REFLECTION_SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False, indent=2),
            max_tokens=1200,
        )
    except Exception:
        raw = {}

    if not isinstance(raw, dict) or not raw:
        raw = _fallback_profile(payload["failed_trace_summary"])

    fallback = _fallback_profile(payload["failed_trace_summary"])
    visual_hint = _sanitize_hint(raw.get("visual_hint"), gold_answer=gold_answer) or fallback["visual_hint"]
    reasoning_hint = _sanitize_hint(raw.get("reasoning_hint"), gold_answer=gold_answer) or fallback["reasoning_hint"]
    verifier_hint = _sanitize_hint(raw.get("verifier_hint"), gold_answer=gold_answer) or fallback["verifier_hint"]

    task = getattr(trace, "task", None)
    meta = getattr(task, "metadata", {}) if task else {}
    scene = getattr(getattr(trace, "evidence", None), "scene_type", None) or str(meta.get("context") or "unknown")
    task_name = str(meta.get("task") or meta.get("context") or "general")
    keywords = _sanitize_keywords(raw.get("keywords"))
    keywords = list(dict.fromkeys(keywords + [scene.lower(), task_name.lower(), str(meta.get("source") or "").lower()]))

    profile = {
        "visual_hint": visual_hint,
        "reasoning_hint": reasoning_hint,
        "verifier_hint": verifier_hint,
        "keywords": [k for k in keywords if k],
        "retrieval_text_bge": _sanitize_hint(raw.get("retrieval_text_bge"), gold_answer=gold_answer)
            or fallback["retrieval_text_bge"],
        "retrieval_text_clip": _sanitize_hint(raw.get("retrieval_text_clip"), gold_answer=gold_answer)
            or fallback["retrieval_text_clip"],
        "reflection_round": round_index,
        "failure_hypothesis": _sanitize_hint(raw.get("failure_hypothesis"), gold_answer=gold_answer)
            or fallback["failure_hypothesis"],
        "should_retry": bool(raw.get("should_retry", True)),
        "reflection_confidence": float(raw.get("confidence", 0.55) or 0.55),
    }
    return profile
