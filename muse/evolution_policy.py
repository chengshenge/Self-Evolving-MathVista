from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass
class ReusePolicyResult:
    accepted: bool
    trust_level: str
    threshold: float
    reasons: List[str]


def _has_exact_match(matched_fields: Iterable[str], prefix: str) -> bool:
    for field in matched_fields:
        if str(field).startswith(prefix):
            return True
    return False


def decide_reuse_acceptance(task: Any, candidate: Dict[str, Any], verifier_decision: str, verifier_confidence: float, looks_canonical: bool) -> ReusePolicyResult:
    """
    Reuse policy after the router shortlist.

    Goals:
    - keep reuse conservative for cold / weak skills
    - allow genuinely strong exact-bucket specialists to be used without near-perfect verifier scores
    - avoid accepting non-canonical answers
    """
    stats = candidate.get("stats", {}) or {}
    matched_fields = list(candidate.get("matched_fields", []) or [])

    bucket_total = int(stats.get("bucket_total", 0) or 0)
    bucket_accuracy = float(stats.get("bucket_accuracy", 0.0) or 0.0)
    global_total = int(stats.get("global_total", 0) or 0)
    global_accuracy = float(stats.get("global_accuracy", 0.0) or 0.0)

    has_task_exact = _has_exact_match(matched_fields, "task:exact")
    has_context_exact = _has_exact_match(matched_fields, "context:exact") or _has_exact_match(matched_fields, "scene/context:exact")
    exact_bucket = has_task_exact and has_context_exact

    reasons: List[str] = []
    if verifier_decision != "accept":
        reasons.append(f"verifier_decision={verifier_decision}")
    if not looks_canonical:
        reasons.append("non_canonical_answer")

    if exact_bucket and bucket_total >= 3 and bucket_accuracy >= 0.80:
        trust_level = "high"
        threshold = 0.58
    elif exact_bucket and bucket_total >= 2 and bucket_accuracy >= 0.67:
        trust_level = "high"
        threshold = 0.64
    elif exact_bucket and bucket_total >= 1 and bucket_accuracy >= 1.0:
        trust_level = "medium"
        threshold = 0.72
    elif exact_bucket and global_total >= 4 and global_accuracy >= 0.75:
        trust_level = "medium"
        threshold = 0.78
    else:
        trust_level = "low"
        threshold = 0.92
        reasons.append("insufficient_validated_support")

    if verifier_confidence < threshold:
        reasons.append(f"low_verify_conf={verifier_confidence:.2f}< {threshold:.2f}")

    accepted = verifier_decision == "accept" and looks_canonical and verifier_confidence >= threshold and trust_level != "low"
    return ReusePolicyResult(
        accepted=accepted,
        trust_level=trust_level,
        threshold=threshold,
        reasons=reasons,
    )
