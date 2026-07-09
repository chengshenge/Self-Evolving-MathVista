from __future__ import annotations

import json
import math
import re
from typing import Any, Optional, Tuple

from .answer_policy import coerce_candidate_for_task, is_abstention
from .schemas import TaskPacket


_ABSTENTION_MARKERS = [
    "cannot determine",
    "can't determine",
    "unable to determine",
    "insufficient evidence",
    "not enough information",
    "cannot answer",
    "can't answer",
    "image not accessible",
    "image is not accessible",
    "cannot view the image",
    "can't view the image",
    "cannot see the image",
    "can't see the image",
    "please upload",
    "need a clearer image",
    "need a clearer photo",
    "unable to access the image",
    "i cannot access the image",
]

_UNIT_ALIASES = {
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "mg": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
}

_UNIT_FACTORS_TO_BASE = {
    # base: meter
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    # base: gram
    "mg": 0.001,
    "g": 1.0,
    "kg": 1000.0,
    # base: second
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
}


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _coerce_precision(value: Any, default: int = 2) -> int:
    if value is None:
        return default
    try:
        return max(0, int(round(float(value))))
    except Exception:
        return default


def _looks_abstention_text(value: Any) -> bool:
    if value is None:
        return True if value in (None,) else False
    text = _norm_text(value)
    if not text:
        return False
    if any(marker in text for marker in _ABSTENTION_MARKERS):
        return True
    return False


def _extract_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(value).replace(",", ""))
    return float(m.group(0)) if m else None


def _normalize_unit_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "null"}:
        return None
    text = text.replace("²", "^2").replace("³", "^3")
    text = re.sub(r"[\[\]\(\),]", " ", text)
    # keep only plain scalar units; skip composite/power units to avoid incorrect conversions
    if any(tok in text for tok in ["/", "^", "2", "3", "%", "per "]):
        return None
    token = re.sub(r"[^a-z]", "", text)
    return _UNIT_ALIASES.get(token)


def _extract_number_and_unit(value: Any) -> Tuple[Optional[float], Optional[str]]:
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return _extract_number(value), None
    text = str(value).strip().replace(",", "")
    if not text:
        return None, None
    m = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-z]+)?", text)
    if not m:
        return None, None
    try:
        num = float(m.group(1))
    except Exception:
        return None, None
    unit = _normalize_unit_token(m.group(2)) if m.group(2) else None
    return num, unit


def _convert_number_to_task_unit(task: TaskPacket, raw_value: Any) -> Optional[float]:
    target_unit = _normalize_unit_token(task.unit)
    if target_unit is None:
        return None
    num, source_unit = _extract_number_and_unit(raw_value)
    if num is None or source_unit is None:
        return None
    if source_unit == target_unit:
        return num
    if source_unit not in _UNIT_FACTORS_TO_BASE or target_unit not in _UNIT_FACTORS_TO_BASE:
        return None
    base = num * _UNIT_FACTORS_TO_BASE[source_unit]
    return base / _UNIT_FACTORS_TO_BASE[target_unit]



def normalize_answer(task: TaskPacket, candidate: Any) -> Any:
    if is_abstention(candidate) or _looks_abstention_text(candidate):
        return None

    cand = coerce_candidate_for_task(task, candidate)
    if cand is None or is_abstention(cand) or _looks_abstention_text(cand):
        return None

    if task.question_type == "multi_choice" and task.choices:
        return str(cand)

    if task.answer_type == "integer":
        num = _convert_number_to_task_unit(task, candidate)
        if num is None:
            num = _extract_number(cand)
        return None if num is None else str(int(round(num)))

    if task.answer_type == "float":
        num = _convert_number_to_task_unit(task, candidate)
        if num is None:
            num = _extract_number(cand)
        if num is None:
            return None
        p = _coerce_precision(task.precision, 2)
        return f"{round(num, p):.{p}f}"

    if task.answer_type == "list":
        if isinstance(cand, list):
            return [str(x).strip() for x in cand]
        text = str(cand).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass
        return [part.strip() for part in re.split(r"[,;]", text) if part.strip()]

    return str(cand).strip()



def answers_equal(task: TaskPacket, prediction: Any, gold: Any) -> bool:
    pred = normalize_answer(task, prediction)
    goldn = normalize_answer(task, gold)
    if pred is None or goldn is None:
        return False

    if task.answer_type == "integer":
        return int(_extract_number(pred)) == int(_extract_number(goldn))
    if task.answer_type == "float":
        p = _coerce_precision(task.precision, 2)
        return round(float(_extract_number(pred)), p) == round(float(_extract_number(goldn)), p)
    if task.answer_type == "list":
        return [_norm_text(x) for x in pred] == [_norm_text(x) for x in goldn]
    return _norm_text(pred) == _norm_text(goldn)


# === AB_SMOKE20_FIX3_ANSWERING ===
_ORIG_NORMALIZE_ANSWER_FIX3 = normalize_answer
_ORIG_ANSWERS_EQUAL_FIX3 = answers_equal

_MCQ_ABSTAIN_MARKERS_FIX3 = [
    "cannot determine",
    "can not determine",
    "unable to determine",
    "insufficient evidence",
    "not enough evidence",
    "image not accessible",
    "image unavailable",
    "cannot access the image",
    "cannot see the image",
    "need the image",
    "need image",
    "recheck",
]

def _fix3_choice_norm(text: Any) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[–—]", "-", s)
    s = re.sub(r"[\(\)\[\]\{\}:;,_\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _fix3_is_abstention(text: Any) -> bool:
    s = str(text or "").strip().lower()
    return any(m in s for m in _MCQ_ABSTAIN_MARKERS_FIX3)

def _fix3_letter_to_choice(task: TaskPacket, letter: str) -> Optional[str]:
    if not getattr(task, "choices", None):
        return None
    letter = str(letter or "").strip().upper()
    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        return None
    idx = ord(letter) - ord("A")
    if 0 <= idx < len(task.choices):
        return str(task.choices[idx]).strip()
    return None

def _fix3_strip_markers(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"^\(?\s*([A-Za-z])\s*[\)\].:\-]\s*", "", s)
    s = re.sub(r"^\s*(?:option|choice)\s+([A-Za-z])\s*[\)\].:\-]?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*\(?\s*[A-Za-z]\s*\)?\s*$", "", s)
    return s.strip()

def _fix3_try_numeric_choice_match(task: TaskPacket, text: str) -> Optional[str]:
    cand_num = _extract_number(text)
    if cand_num is None:
        return None
    hits = []
    for choice in getattr(task, "choices", []) or []:
        num = _extract_number(choice)
        if num is None:
            continue
        try:
            if float(num) == float(cand_num):
                hits.append(str(choice).strip())
        except Exception:
            pass
    if len(hits) == 1:
        return hits[0]
    return None

def _fix3_mcq_canonicalize(task: TaskPacket, candidate: Any) -> Any:
    if candidate is None:
        return None
    raw = str(candidate).strip()
    if not raw:
        return None
    if _fix3_is_abstention(raw):
        return None

    m = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?", raw)
    if m:
        mapped = _fix3_letter_to_choice(task, m.group(1))
        if mapped is not None:
            return mapped

    m = re.match(r"^\s*(?:option|choice)?\s*\(?\s*([A-Za-z])\s*\)?[\s:.\-]*", raw, flags=re.I)
    if m:
        mapped = _fix3_letter_to_choice(task, m.group(1))
        stripped = _fix3_strip_markers(raw)
        if stripped:
            for choice in task.choices or []:
                if _fix3_choice_norm(stripped) == _fix3_choice_norm(choice):
                    return str(choice).strip()
        if mapped is not None and not stripped:
            return mapped

    stripped = _fix3_strip_markers(raw)

    for choice in task.choices or []:
        if _fix3_choice_norm(stripped) == _fix3_choice_norm(choice):
            return str(choice).strip()

    norm_stripped = _fix3_choice_norm(stripped)
    hits = []
    for choice in task.choices or []:
        norm_choice = _fix3_choice_norm(choice)
        if norm_choice and norm_choice in norm_stripped:
            hits.append(str(choice).strip())
    if len(hits) == 1:
        return hits[0]

    numeric_hit = _fix3_try_numeric_choice_match(task, stripped)
    if numeric_hit is not None:
        return numeric_hit

    return stripped if stripped else raw

def normalize_answer(task: TaskPacket, candidate: Any) -> Any:  # type: ignore[override]
    if getattr(task, "question_type", None) == "multi_choice" and getattr(task, "choices", None):
        return _fix3_mcq_canonicalize(task, candidate)
    return _ORIG_NORMALIZE_ANSWER_FIX3(task, candidate)

def answers_equal(task: TaskPacket, prediction: Any, gold: Any) -> bool:  # type: ignore[override]
    pred = normalize_answer(task, prediction)
    goldn = normalize_answer(task, gold)
    if pred is None or goldn is None:
        return False
    if getattr(task, "question_type", None) == "multi_choice" and getattr(task, "choices", None):
        return _norm_text(pred) == _norm_text(goldn)
    return _ORIG_ANSWERS_EQUAL_FIX3(task, prediction, gold)


# === AB_SMOKE20_FIX4_MCQ_PREFIX_REPAIR ===
_PREV_NORMALIZE_ANSWER_FIX4 = normalize_answer
_PREV_ANSWERS_EQUAL_FIX4 = answers_equal

_FIX4_ABSTAIN_MARKERS = [
    "cannot determine",
    "can not determine",
    "unable to determine",
    "insufficient evidence",
    "not enough evidence",
    "image not accessible",
    "image unavailable",
    "cannot access the image",
    "cannot see the image",
    "need the image",
    "need image",
    "recheck",
]

def _fix4_choice_norm(value: Any) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\u2013\u2014]", "-", s)
    s = re.sub(r"[\(\)\[\]\{\}:;,_\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _fix4_is_abstention(value: Any) -> bool:
    s = str(value or "").strip().lower()
    return any(m in s for m in _FIX4_ABSTAIN_MARKERS)

def _fix4_letter_to_choice(task: TaskPacket, letter: str) -> Optional[str]:
    if not getattr(task, "choices", None):
        return None
    letter = str(letter or "").strip().upper()
    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        return None
    idx = ord(letter) - ord("A")
    if 0 <= idx < len(task.choices):
        return str(task.choices[idx]).strip()
    return None

def _fix4_safe_strip_wrappers(text: Any) -> str:
    s = str(text or "").strip()
    s = re.sub(r"^\s*(?:option|choice)\s+[A-Za-z]\s*[\)\].:\-]?\s*", "", s, flags=re.I)
    s = re.sub(r"^\(?\s*[A-Za-z]\s*[\)\].:\-]\s*", "", s)
    s = re.sub(r"\s*\(\s*[A-Za-z]\s*\)\s*$", "", s)
    s = re.sub(r"\s*[,;:/-]\s*[A-Za-z]\s*$", "", s)
    s = re.sub(r"\s*(?:option|choice)\s+[A-Za-z]\s*$", "", s, flags=re.I)
    return s.strip()

def _fix4_exact_choice(task: TaskPacket, text: Any) -> Optional[str]:
    s = str(text or "").strip()
    if not s:
        return None
    for choice in getattr(task, "choices", []) or []:
        if _fix4_choice_norm(s) == _fix4_choice_norm(choice):
            return str(choice).strip()
    return None

def _fix4_numeric_choice_match(task: TaskPacket, text: Any) -> Optional[str]:
    cand_num = _extract_number(text)
    if cand_num is None:
        return None
    hits = []
    for choice in getattr(task, "choices", []) or []:
        num = _extract_number(choice)
        if num is None:
            continue
        try:
            if float(num) == float(cand_num):
                hits.append(str(choice).strip())
        except Exception:
            pass
    if len(hits) == 1:
        return hits[0]
    return None

def _fix4_repair_mcq_output(task: TaskPacket, candidate: Any, prev_normalized: Any) -> Any:
    if candidate is None:
        return prev_normalized

    raw = str(candidate).strip()
    prev = None if prev_normalized is None else str(prev_normalized).strip()

    if not raw:
        return prev_normalized

    if _fix4_is_abstention(raw):
        return None

    # Preserve any already-exact current normalization result.
    exact_prev = _fix4_exact_choice(task, prev)
    if exact_prev is not None:
        return exact_prev

    # Single-letter answer like "A" or "(B)".
    m = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?", raw)
    if m:
        mapped = _fix4_letter_to_choice(task, m.group(1))
        if mapped is not None:
            return mapped

    # Direct exact match on raw string first.
    exact_raw = _fix4_exact_choice(task, raw)
    if exact_raw is not None:
        return exact_raw

    # Remove only explicit wrapper markers such as "(A)" or ", A" and try again.
    stripped = _fix4_safe_strip_wrappers(raw)
    exact_stripped = _fix4_exact_choice(task, stripped)
    if exact_stripped is not None:
        return exact_stripped

    # Numeric unique-choice repair like "6" -> "6cm".
    numeric_hit = _fix4_numeric_choice_match(task, stripped or raw)
    if numeric_hit is not None:
        return numeric_hit

    # Prefix-repair only if the previous normalized string is a strict prefix of exactly one choice.
    if prev:
        prev_n = _fix4_choice_norm(prev)
        raw_n = _fix4_choice_norm(raw)
        stripped_n = _fix4_choice_norm(stripped)
        candidates = []
        for choice in getattr(task, "choices", []) or []:
            ch = str(choice).strip()
            ch_n = _fix4_choice_norm(ch)
            if prev_n and ch_n.startswith(prev_n) and ch_n != prev_n:
                candidates.append(ch)

        if len(candidates) == 1:
            ch = candidates[0]
            ch_n = _fix4_choice_norm(ch)
            if ch_n and (ch_n in raw_n or ch_n in stripped_n or raw_n.startswith(prev_n) or stripped_n.startswith(prev_n)):
                return ch

    return prev_normalized

def normalize_answer(task: TaskPacket, candidate: Any) -> Any:  # type: ignore[override]
    prev = _PREV_NORMALIZE_ANSWER_FIX4(task, candidate)
    if getattr(task, "question_type", None) == "multi_choice" and getattr(task, "choices", None):
        repaired = _fix4_repair_mcq_output(task, candidate, prev)
        return repaired
    return prev

def answers_equal(task: TaskPacket, prediction: Any, gold: Any) -> bool:  # type: ignore[override]
    pred = normalize_answer(task, prediction)
    goldn = normalize_answer(task, gold)
    if pred is None or goldn is None:
        return False
    if getattr(task, "question_type", None) == "multi_choice" and getattr(task, "choices", None):
        return _norm_text(pred) == _norm_text(goldn)
    return _PREV_ANSWERS_EQUAL_FIX4(task, prediction, gold)
