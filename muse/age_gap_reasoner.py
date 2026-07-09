from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .answer_policy import coerce_candidate_for_task
from .schemas import TaskPacket

AGE_GAP_PATTERNS = [
    'age gap', 'how many years', 'older than', 'younger than',
    'difference in age', 'difference of age'
]


def _norm(x: Any) -> str:
    return re.sub(r'\s+', ' ', str(x or '').strip().lower())


def is_age_gap_task(task: TaskPacket) -> bool:
    if task.answer_type != 'integer':
        return False
    q = _norm(task.question)
    return any(p in q for p in AGE_GAP_PATTERNS)


def _iter_focus_items(focus: Any) -> Iterable[Any]:
    if isinstance(focus, list):
        for item in focus:
            yield item
    elif focus is not None:
        yield focus


def collect_focus_items(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for payload in reversed(evidence.get('raw_payloads', []) or []):
        fa = payload.get('focus_answers')
        for item in _iter_focus_items(fa):
            if isinstance(item, dict):
                items.append(item)
    fa = evidence.get('focus_answers')
    for item in _iter_focus_items(fa):
        if isinstance(item, dict):
            items.append(item)
    return items


def _coerce_range(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            a = float(value[0])
            b = float(value[1])
            if math.isnan(a) or math.isnan(b):
                return None
            return (min(a, b), max(a, b))
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip().replace('–', '-').replace('—', '-')
        m = re.search(r'(\d{1,2})\s*[-]\s*(\d{1,2})', s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            return (min(a, b), max(a, b))
    return None


def _decade_token_to_range(prefix: str, decade: int) -> Tuple[int, int]:
    prefix = prefix.strip().lower()
    if prefix == 'early':
        return decade, decade + 4
    if prefix == 'mid':
        return decade + 3, decade + 7
    if prefix == 'late':
        return decade + 6, decade + 9
    return decade, decade + 9


def _extract_ranges_from_text(text: str) -> List[Tuple[float, float]]:
    text = text.replace('–', '-').replace('—', '-').lower()
    out: List[Tuple[float, float]] = []
    pattern = re.compile(r'(?:(early|mid|late)\s+)?(\d{2})s(?:\s*[-]\s*(?:(early|mid|late)\s+)?(\d{2})s)?')
    for m in pattern.finditer(text):
        p1, d1, p2, d2 = m.group(1) or '', m.group(2), m.group(3) or '', m.group(4)
        r1 = _decade_token_to_range(p1, int(d1))
        if d2 is None:
            out.append((float(r1[0]), float(r1[1])))
        else:
            r2 = _decade_token_to_range(p2, int(d2))
            out.append((float(min(r1[0], r2[0])), float(max(r1[1], r2[1]))))
    return out


def _collect_person_ranges(items: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    for item in items:
        # 1) explicit list of per-person ranges
        pr = item.get('person_age_ranges')
        if isinstance(pr, list):
            for obj in pr:
                if not isinstance(obj, dict):
                    continue
                r = _coerce_range(obj.get('estimated_age_range') or obj.get('age_range'))
                if r is not None:
                    ranges.append(r)
        # 2) explicit person_1/person_2 keys
        for key in ('person_1_age_range', 'person_2_age_range', 'person1_age_range', 'person2_age_range'):
            r = _coerce_range(item.get(key))
            if r is not None:
                ranges.append(r)
        # 3) parse from supporting clues / rationale text
        for field in ('supporting_clues', 'rationale', 'explanation', 'evidence'):
            v = item.get(field)
            if isinstance(v, list):
                texts = [str(x) for x in v]
            elif v is None:
                texts = []
            else:
                texts = [str(v)]
            for t in texts:
                ranges.extend(_extract_ranges_from_text(t))
    # Deduplicate approximately and keep first two strongest distinct ranges
    dedup: List[Tuple[float, float]] = []
    for r in ranges:
        if not any(abs(r[0]-d[0]) < 0.5 and abs(r[1]-d[1]) < 0.5 for d in dedup):
            dedup.append(r)
    return dedup[:2]


def _ranges_support_precise_gap(r1: Tuple[float, float], r2: Tuple[float, float]) -> bool:
    w1 = r1[1] - r1[0]
    w2 = r2[1] - r2[0]
    if w1 > 14 or w2 > 14:
        return False
    m1 = (r1[0] + r1[1]) / 2
    m2 = (r2[0] + r2[1]) / 2
    gap = abs(m1 - m2)
    # if ranges overlap heavily and gap is small, abstain
    overlap = max(0.0, min(r1[1], r2[1]) - max(r1[0], r2[0]))
    if overlap > 4 and gap < 6:
        return False
    return gap >= 4


def _rounded_gap(r1: Tuple[float, float], r2: Tuple[float, float]) -> int:
    m1 = (r1[0] + r1[1]) / 2
    m2 = (r2[0] + r2[1]) / 2
    return int(round(abs(m1 - m2)))


def _has_named_entity_signal(evidence: Dict[str, Any]) -> bool:
    text_chunks: List[str] = []
    for fact in evidence.get('visual_facts', []) or []:
        if isinstance(fact, dict):
            text_chunks.append(str(fact.get('fact', '')))
        else:
            text_chunks.append(str(fact))
    for item in collect_focus_items(evidence):
        text_chunks.append(str(item))
    text = _norm(' '.join(text_chunks))
    return 'named_entities_text:' in text or 'king ' in text or 'queen ' in text or 'anne ' in text or 'richard ' in text


def _reasoning_mentions_grounding_steps(reasoning_steps: List[str]) -> bool:
    text = ' '.join(reasoning_steps or []).lower()
    # explicit birth years/dates or age years in reasoning
    year_hits = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', text)
    if len(year_hits) >= 2:
        return True
    if 'birth year' in text or 'born in' in text or 'age gap' in text and 'using' in text:
        return True
    return False


def try_age_gap_answer(task: TaskPacket, evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not is_age_gap_task(task):
        return None

    items = collect_focus_items(evidence)

    # Never fabricate a precise integer merely because named entities exist.
    # If we have a direct numeric answer from focus, only trust it when explicit support exists.
    for item in items:
        ans = item.get('answer', item.get('direct_answer', item.get('result')))
        cand = coerce_candidate_for_task(task, ans)
        if cand is None:
            continue
        et = _norm(item.get('evidence_type') or '')
        if et in {'explicit_text', 'recognized_entity'}:
            return {
                'reasoning_steps': [
                    'A direct numeric answer was provided by the visual extractor for this age-gap question.',
                    'The answer is supported by explicit text or strong recognized-entity evidence.',
                ],
                'candidate_answer': int(cand),
                'answer_confidence': max(0.72, float(item.get('confidence', 0.0) or 0.0)),
                'needs_visual_recheck': False,
                'focus_questions': [],
                'normalization_notes': 'age_gap_direct_focus_answer',
            }

    ranges = _collect_person_ranges(items)
    if len(ranges) >= 2:
        r1, r2 = ranges[0], ranges[1]
        if _ranges_support_precise_gap(r1, r2):
            gap = _rounded_gap(r1, r2)
            # calibrated confidence: narrower ranges and larger separation => higher confidence
            w1 = r1[1] - r1[0]
            w2 = r2[1] - r2[0]
            conf = 0.55 + max(0.0, 10 - (w1 + w2) / 2) * 0.02
            conf = min(0.82, max(0.58, conf))
            return {
                'reasoning_steps': [
                    f'Estimated age range for person 1: [{r1[0]:.0f}, {r1[1]:.0f}] years.',
                    f'Estimated age range for person 2: [{r2[0]:.0f}, {r2[1]:.0f}] years.',
                    'Using the midpoints of these ranges to estimate the most likely integer age gap.',
                    f'Estimated integer age gap ≈ {gap} years.',
                ],
                'candidate_answer': gap,
                'answer_confidence': conf,
                'needs_visual_recheck': False,
                'focus_questions': [],
                'normalization_notes': 'age_gap_estimator',
            }

    return None


def gate_age_gap_candidate(task: TaskPacket, evidence: Dict[str, Any], candidate: Any, math_result: Optional[Dict[str, Any]] = None) -> Any:
    if not is_age_gap_task(task):
        return candidate
    cand = coerce_candidate_for_task(task, candidate)
    if cand is None:
        return None
    notes = _norm((math_result or {}).get('normalization_notes') or '')
    reasoning_steps = [str(x) for x in ((math_result or {}).get('reasoning_steps') or [])]
    if 'age_gap_estimator' in notes:
        return int(cand)
    if _has_named_entity_signal(evidence) and _reasoning_mentions_grounding_steps(reasoning_steps):
        return int(cand)
    return None


def verifier_supports_age_gap_accept(task: TaskPacket, evidence: Dict[str, Any], candidate: Any, math_result: Optional[Dict[str, Any]] = None) -> bool:
    return gate_age_gap_candidate(task, evidence, candidate, math_result) is not None
