from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .schemas import TaskPacket

ABSTAIN_PATTERNS = [
    r"^cannot[_\s-]*determine",
    r"^unable[_\s-]*to[_\s-]*determine",
    r"^insufficient\s+evidence",
    r"^insufficient\s+visual\s+evidence",
    r"^not\s+enough\s+information",
    r"^unknown$",
    r"^none$",
    r"^null$",
    r"^n/?a$",
    r"^$",
]

DIRECT_ANSWER_KEYS = {
    "answer",
    "direct_answer",
    "final_answer",
    "candidate_answer",
    "result",
    "value",
    "count",
    "conclusion",
    "remaining_objects_after_subtraction",
}
IGNORE_KEYS = {
    "question",
    "rationale",
    "reason",
    "evidence",
    "explanation",
    "confidence",
    "status",
    "support",
    "supporting_evidence",
    "notes",
}


def _text(value: Any) -> str:
    return str(value).strip()


def is_abstention(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict)):
        return False
    text = _text(value).lower()
    return any(re.match(pat, text) for pat in ABSTAIN_PATTERNS)


def looks_like_question_text(task: TaskPacket, value: Any) -> bool:
    if value is None:
        return False
    text = re.sub(r"\s+", " ", _text(value)).strip().lower()
    if not text:
        return False
    q = re.sub(r"\s+", " ", task.question).strip().lower()
    if text == q:
        return True
    if q and text in q and len(text) > 16:
        return True
    if text.endswith("?") and len(text) > 16:
        return True
    return False


def _extract_number_text(text: str) -> Optional[str]:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return m.group(0) if m else None


def coerce_candidate_for_task(task: TaskPacket, value: Any) -> Any:
    if value is None:
        return None
    if is_abstention(value):
        return None
    if looks_like_question_text(task, value):
        return None

    if task.question_type == "multi_choice" and task.choices:
        text = _text(value)
        if not text or is_abstention(text):
            return None
        # exact choice
        choices_norm = {c.strip().lower(): c for c in task.choices}
        if text.strip().lower() in choices_norm:
            return choices_norm[text.strip().lower()]
        # option letter
        m = re.fullmatch(r"[\(\[]?([A-Za-z])[\)\]]?", text)
        if m:
            idx = ord(m.group(1).upper()) - ord("A")
            if 0 <= idx < len(task.choices):
                return task.choices[idx]
        # choice text embedded in answer
        lowered = text.lower()
        for choice in task.choices:
            if choice.lower() in lowered:
                return choice
        return None

    if task.answer_type == "integer":
        if isinstance(value, (int, float)):
            return int(round(float(value)))
        num = _extract_number_text(_text(value))
        if num is None:
            return None
        return int(round(float(num)))

    if task.answer_type == "float":
        if isinstance(value, (int, float)):
            return float(value)
        num = _extract_number_text(_text(value))
        if num is None:
            return None
        return float(num)

    # free-form text answer: require a short scalar-like answer, not a sentence of observations
    text = _text(value)
    if len(text) > 120 and task.answer_type == "text":
        return None
    if task.answer_type == "text" and task.question_type == "free_form":
        # reject observational prose that clearly does not answer the question
        lowered = text.lower()
        bad_markers = [
            "no visible",
            "not visible",
            "caption",
            "metadata",
            "date inscriptions",
            "no information",
        ]
        if any(m in lowered for m in bad_markers) and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:°|%|cm|mm|m|kg)?", text):
            return None
    return text


def _iter_focus_items(focus_answers: Any) -> Iterable[Any]:
    if isinstance(focus_answers, list):
        for item in focus_answers:
            yield item
    elif focus_answers is not None:
        yield focus_answers


def extract_direct_answer_from_focus(task: TaskPacket, focus_answers: Any) -> Any:
    # list of answer objects
    for item in _iter_focus_items(focus_answers):
        if isinstance(item, dict):
            status = str(item.get("status", "")).strip().lower()
            if status.startswith("cannot_determine") or status.startswith("insufficient"):
                continue
            if "answer" in item:
                cand = coerce_candidate_for_task(task, item.get("answer"))
                if cand is not None:
                    return cand
            # fallback: pick only trusted direct-answer keys that are scalar
            for key, val in item.items():
                if key in IGNORE_KEYS or key not in DIRECT_ANSWER_KEYS:
                    continue
                cand = coerce_candidate_for_task(task, val)
                if cand is not None:
                    return cand
        elif isinstance(item, (str, int, float)):
            cand = coerce_candidate_for_task(task, item)
            if cand is not None:
                return cand

    # plain dict mapping of direct answers
    if isinstance(focus_answers, dict):
        for key, val in focus_answers.items():
            if key in IGNORE_KEYS or key not in DIRECT_ANSWER_KEYS:
                continue
            cand = coerce_candidate_for_task(task, val)
            if cand is not None:
                return cand
    return None


def has_grounded_commonsense_signal(task: TaskPacket, visual_facts: List[Dict[str, Any]]) -> bool:
    if not visual_facts:
        return False
    q = task.question.lower()
    facts = " ".join(str(v.get("fact", "")) for v in visual_facts).lower()
    # named-entity / age-gap style
    if any(token in q for token in ["age gap", "age difference", "born after"]):
        if any(token in facts for token in ["vintage", "old photograph", "black-and-white", "early 20th", "king ", "queen ", "anne", "richard"]):
            return True
    return False
