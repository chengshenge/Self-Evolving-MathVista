from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from .answer_policy import coerce_candidate_for_task
from .schemas import TaskPacket

PLOT_QUESTION_MARKERS = [
    'continuous', 'discontinuous', 'function', 'graph', 'plot', 'curve', 'odd', 'even',
    'increasing', 'decreasing', 'symmetric', 'y-axis', 'x-axis', 'origin'
]


def _norm(x: Any) -> str:
    return re.sub(r'\s+', ' ', str(x or '').strip().lower())


def _iter_focus_items(focus: Any) -> Iterable[Any]:
    if isinstance(focus, list):
        for item in focus:
            yield item
    elif focus is not None:
        yield focus


def _collect_focus_items(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for payload in reversed(evidence.get('raw_payloads', []) or []):
        fa = payload.get('focus_answers')
        for item in _iter_focus_items(fa):
            if isinstance(item, dict):
                items.append(item)
    for item in _iter_focus_items(evidence.get('focus_answers')):
        if isinstance(item, dict):
            items.append(item)
    return items


def _evidence_text(evidence: Dict[str, Any]) -> str:
    chunks: List[str] = []
    for fact in evidence.get('visual_facts', []) or []:
        chunks.append(str(fact.get('fact') if isinstance(fact, dict) else fact))
    for item in evidence.get('uncertainties', []) or []:
        chunks.append(str(item))
    for item in _collect_focus_items(evidence):
        for k, v in item.items():
            chunks.append(f'{k}: {v}')
    return _norm(' '.join(chunks))


def is_function_plot_task(task: TaskPacket) -> bool:
    context = _norm((task.metadata or {}).get('context'))
    source = _norm((task.metadata or {}).get('source'))
    q = _norm(task.question)
    if 'functionqa' in source or 'function qa' in source:
        return True
    if any(m in context for m in ['plot', 'function', 'graph', 'line chart', 'line graph']):
        return True
    return any(m in q for m in PLOT_QUESTION_MARKERS)


def _best_focus_answer(task: TaskPacket, evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for item in _collect_focus_items(evidence):
        ans = item.get('answer', item.get('direct_answer'))
        cand = coerce_candidate_for_task(task, ans)
        if cand is None:
            continue
        try:
            conf = float(item.get('confidence', 0.0))
        except Exception:
            conf = 0.0
        rationale = str(item.get('rationale', '') or item.get('supporting_features', '') or item.get('evidence', ''))
        return {'candidate': cand, 'confidence': conf, 'rationale': rationale}
    return None


def try_function_plot_answer(task: TaskPacket, evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not is_function_plot_task(task):
        return None
    txt = _evidence_text(evidence)
    direct = _best_focus_answer(task, evidence)
    q = _norm(task.question)

    # Prefer a direct focused answer when the supporting graph features are present.
    if direct and direct['confidence'] >= 0.6:
        return {
            'reasoning_steps': [
                'The visual extractor provided a direct graph-property answer with supporting graph features.',
                f"Using direct plot answer: {direct['candidate']}.",
            ],
            'candidate_answer': direct['candidate'],
            'answer_confidence': max(0.75, direct['confidence']),
            'needs_visual_recheck': False,
            'focus_questions': [],
            'normalization_notes': 'function_plot_discrete: direct focus answer',
        }

    # Simple discrete heuristics for yes/no continuity and symmetry.
    if 'continuous' in q:
        if any(m in txt for m in ['hole', 'open circle', 'jump', 'break', 'gap', 'discontinuity', 'vertical asymptote']):
            return {
                'reasoning_steps': [
                    'The graph evidence mentions a hole/jump/break/asymptote, which implies the graph is not continuous.',
                ],
                'candidate_answer': 'No',
                'answer_confidence': 0.8,
                'needs_visual_recheck': False,
                'focus_questions': [],
                'normalization_notes': 'function_plot_discrete: discontinuity marker',
            }
        if any(m in txt for m in ['continuous curve', 'no breaks', 'unbroken curve', 'connected graph']):
            return {
                'reasoning_steps': [
                    'The graph evidence indicates a continuous/unbroken curve.',
                ],
                'candidate_answer': 'Yes',
                'answer_confidence': 0.78,
                'needs_visual_recheck': False,
                'focus_questions': [],
                'normalization_notes': 'function_plot_discrete: continuity marker',
            }

    if 'even' in q and 'symmetric about the y-axis' in txt:
        return {
            'reasoning_steps': ['The graph is symmetric about the y-axis, so the function is even.'],
            'candidate_answer': 'even' if task.question_type == 'free_form' else 'Yes',
            'answer_confidence': 0.78,
            'needs_visual_recheck': False,
            'focus_questions': [],
            'normalization_notes': 'function_plot_discrete: y-axis symmetry',
        }

    if 'odd' in q and 'symmetric about the origin' in txt:
        return {
            'reasoning_steps': ['The graph is symmetric about the origin, so the function is odd.'],
            'candidate_answer': 'odd' if task.question_type == 'free_form' else 'Yes',
            'answer_confidence': 0.78,
            'needs_visual_recheck': False,
            'focus_questions': [],
            'normalization_notes': 'function_plot_discrete: origin symmetry',
        }

    return None


def verifier_supports_plot_accept(task: TaskPacket, evidence: Dict[str, Any], candidate: Any, math_result: Optional[Dict[str, Any]] = None) -> bool:
    cand = coerce_candidate_for_task(task, candidate)
    if cand is None:
        return False
    txt = _evidence_text(evidence)
    q = _norm(task.question)
    if 'continuous' in q:
        if str(cand).lower() in {'no', 'yes'}:
            if str(cand).lower() == 'no' and any(m in txt for m in ['hole', 'open circle', 'jump', 'break', 'gap', 'discontinuity', 'vertical asymptote']):
                return True
            if str(cand).lower() == 'yes' and any(m in txt for m in ['continuous curve', 'no breaks', 'unbroken curve', 'connected graph']):
                return True
    direct = _best_focus_answer(task, evidence)
    if direct and direct['candidate'] == cand and direct['confidence'] >= 0.6:
        return True
    return False
