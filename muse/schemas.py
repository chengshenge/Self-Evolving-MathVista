from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _clean_unit(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return str(value)


def _clean_precision(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(round(value))
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _normalize_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items = [str(x) for x in value if x is not None]
        stripped = [x for x in items if x != ""]
        if stripped and all(len(x) <= 1 for x in stripped):
            joined = "".join(stripped).strip()
            return [joined] if joined else []
        out: List[str] = []
        for item in items:
            text = item.strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


@dataclass
class TaskPacket:
    sample_id: str
    question: str
    image_path: str
    image_paths: List[str] = field(default_factory=list)
    choices: List[str] = field(default_factory=list)
    question_type: str = "free_form"
    answer_type: str = "text"
    precision: Optional[int] = None
    unit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[Any] = None
    query: Optional[str] = None

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "TaskPacket":
        pid = str(record.get("pid") or record.get("sample_id") or record.get("id") or "unknown")
        choices = record.get("choices") or []
        if isinstance(choices, str) and choices.lower() == "none":
            choices = []
        image_paths = record.get("image_paths") or []
        if isinstance(image_paths, str):
            image_paths = [image_paths] if image_paths else []
        image_path = str(record.get("image_path") or record.get("image") or "")
        if not image_path and image_paths:
            image_path = str(image_paths[0])
        return cls(
            sample_id=pid,
            question=record.get("question") or record.get("query") or "",
            image_path=image_path,
            image_paths=[str(x) for x in image_paths],
            choices=[str(x) for x in choices],
            question_type=str(record.get("question_type") or "free_form"),
            answer_type=str(record.get("answer_type") or "text"),
            precision=_clean_precision(record.get("precision")),
            unit=_clean_unit(record.get("unit")),
            metadata=record.get("metadata") or {},
            answer=record.get("answer"),
            query=record.get("query"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_query_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_query_json(cls, query: str) -> "TaskPacket":
        return cls.from_dict(json.loads(query))

    def existing_image_paths(self) -> List[str]:
        paths: List[str] = []
        for value in [self.image_path, *self.image_paths]:
            if not value:
                continue
            path = Path(str(value))
            if path.exists():
                resolved = str(path.resolve())
                if resolved not in paths:
                    paths.append(resolved)
        return paths


@dataclass
class EvidenceFact:
    fact: str
    confidence: float = 0.5
    bbox: Optional[List[int]] = None
    source: str = "vision"
    value: Optional[Any] = None

    @classmethod
    def from_any(cls, item: Any) -> "EvidenceFact":
        if isinstance(item, cls):
            return item
        if isinstance(item, str):
            return cls(fact=item)
        if isinstance(item, dict):
            return cls(
                fact=str(item.get("fact") or item.get("text") or item.get("observation") or ""),
                confidence=float(item.get("confidence", 0.5)),
                bbox=item.get("bbox"),
                source=str(item.get("source", "vision")),
                value=item.get("value"),
            )
        raise TypeError(f"Unsupported evidence fact: {type(item)}")


@dataclass
class EvidenceBundle:
    scene_type: str = "unknown"
    visual_facts: List[EvidenceFact] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    focus_answers: List[Any] = field(default_factory=list)
    raw_payloads: List[Dict[str, Any]] = field(default_factory=list)

    def merge_visual_payload(self, payload: Dict[str, Any]) -> None:
        scene = payload.get("scene_type")
        if scene and self.scene_type == "unknown":
            self.scene_type = str(scene)

        for item in payload.get("visual_facts", []):
            try:
                fact = EvidenceFact.from_any(item)
            except Exception:
                continue
            if fact.fact:
                self.visual_facts.append(fact)

        for item in payload.get("uncertainties", []):
            text = str(item).strip()
            if text and text not in self.uncertainties:
                self.uncertainties.append(text)

        focus = payload.get("focus_answers", [])
        if isinstance(focus, list):
            for item in focus:
                self.focus_answers.append(item)
        elif focus:
            self.focus_answers.append(focus)

        self.raw_payloads.append(payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_type": self.scene_type,
            "visual_facts": [asdict(x) for x in self.visual_facts],
            "uncertainties": list(self.uncertainties),
            "focus_answers": list(self.focus_answers),
            "raw_payloads": list(self.raw_payloads),
        }


@dataclass
class MathReasoningResult:
    reasoning_steps: List[str] = field(default_factory=list)
    candidate_answer: Any = None
    answer_confidence: float = 0.0
    needs_visual_recheck: bool = False
    focus_questions: List[str] = field(default_factory=list)
    normalization_notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MathReasoningResult":
        return cls(
            reasoning_steps=_normalize_str_list(data.get("reasoning_steps", [])),
            candidate_answer=data.get("candidate_answer"),
            answer_confidence=float(data.get("answer_confidence", 0.0)),
            needs_visual_recheck=bool(data.get("needs_visual_recheck", False)),
            focus_questions=_normalize_str_list(data.get("focus_questions", [])),
            normalization_notes=str(data.get("normalization_notes", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    decision: str = "accept"
    issues: List[str] = field(default_factory=list)
    revised_answer: Any = None
    follow_up_visual_questions: List[str] = field(default_factory=list)
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        return cls(
            decision=str(data.get("decision", "accept")),
            issues=_normalize_str_list(data.get("issues", [])),
            revised_answer=data.get("revised_answer"),
            follow_up_visual_questions=_normalize_str_list(data.get("follow_up_visual_questions", [])),
            confidence=float(data.get("confidence", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolveTrace:
    task: TaskPacket
    used_generated_skill: Optional[str] = None
    saved_generated_skill: Optional[str] = None
    reuse_strategy: Optional[str] = None
    reuse_candidates: List[Dict[str, Any]] = field(default_factory=list)
    reuse_attempts: List[Dict[str, Any]] = field(default_factory=list)
    reuse_fallback_reason: Optional[str] = None
    reuse_selected_score: Optional[float] = None
    evidence: EvidenceBundle = field(default_factory=EvidenceBundle)
    visual_rounds: List[Dict[str, Any]] = field(default_factory=list)
    math_rounds: List[Dict[str, Any]] = field(default_factory=list)
    verify_rounds: List[Dict[str, Any]] = field(default_factory=list)
    answer_reflection: Optional[Dict[str, Any]] = None
    final_answer_raw: Any = None
    final_answer_normalized: Any = None
    correct: Optional[bool] = None
    workspace: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "used_generated_skill": self.used_generated_skill,
            "saved_generated_skill": self.saved_generated_skill,
            "reuse_strategy": self.reuse_strategy,
            "reuse_candidates": self.reuse_candidates,
            "reuse_attempts": self.reuse_attempts,
            "reuse_fallback_reason": self.reuse_fallback_reason,
            "reuse_selected_score": self.reuse_selected_score,
            "evidence": self.evidence.to_dict(),
            "visual_rounds": self.visual_rounds,
            "math_rounds": self.math_rounds,
            "verify_rounds": self.verify_rounds,
            "answer_reflection": self.answer_reflection,
            "final_answer_raw": self.final_answer_raw,
            "final_answer_normalized": self.final_answer_normalized,
            "correct": self.correct,
            "workspace": self.workspace,
            "error": self.error,
        }


def slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return value or "skill"
