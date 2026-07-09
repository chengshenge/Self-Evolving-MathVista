from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .schemas import TaskPacket


COLOR_WORDS = {
    "red", "green", "blue", "yellow", "purple", "brown", "gray", "grey",
    "gold", "golden", "teal", "cyan", "orange", "beige", "pink"
}
SIZE_MAP = {
    "big": "large",
    "large": "large",
    "small": "small",
    "tiny": "small",
}
SHAPE_ALIASES = {
    "sphere": "sphere",
    "spheres": "sphere",
    "ball": "sphere",
    "balls": "sphere",
    "cylinder": "cylinder",
    "cylinders": "cylinder",
    "cube": "cube",
    "cubes": "cube",
    "block": "cuboid",
    "blocks": "cuboid",
    "cuboid": "cuboid",
    "cuboids": "cuboid",
    "box": "cube",
    "boxes": "cube",
    "bus": "bus",
    "buss": "bus",
    "double bus": "bus",
    "double buss": "bus",
    "fighter": "fighter",
    "fighters": "fighter",
    "jet": "fighter",
    "airplane": "fighter",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "car": "car",
    "cars": "car",
    "sedan": "car",
    "thing": None,
    "things": None,
    "object": None,
    "objects": None,
}
MATERIAL_WORDS = {"rubber", "metal", "metallic"}
FINISH_WORDS = {"shiny", "matte", "glossy", "reflective"}


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    text = _to_text(value)
    m = re.search(r"-?\d+", text)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def _is_subtraction_question(task: TaskPacket) -> bool:
    q = _norm(task.question)
    return "subtract all" in q and "how many objects are left" in q


def _flatten_focus_answers(focus: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if focus is None:
        return out
    if isinstance(focus, list):
        for item in focus:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"value": item})
        return out
    if isinstance(focus, dict):
        for k, v in focus.items():
            out.append({"key": k, "value": v})
        return out
    out.append({"value": focus})
    return out


def _parse_descriptor(text: str) -> Dict[str, Any]:
    t = _norm(text)
    desc: Dict[str, Any] = {
        "colors": set(),
        "size": None,
        "shape": None,
        "materials": set(),
        "finishes": set(),
        "raw": t,
    }

    for token in re.split(r"[^a-zA-Z]+", t):
        if not token:
            continue
        if token in SIZE_MAP:
            desc["size"] = SIZE_MAP[token]
        if token in COLOR_WORDS:
            desc["colors"].add("gold" if token == "golden" else token)
        if token in MATERIAL_WORDS:
            desc["materials"].add("metallic" if token in {"metal", "metallic"} else token)
        if token in FINISH_WORDS:
            desc["finishes"].add("shiny" if token in {"glossy", "reflective"} else token)

    if "double bus" in t or "double buss" in t:
        desc["shape"] = "bus"
    else:
        for token in sorted(SHAPE_ALIASES.keys(), key=len, reverse=True):
            if token and token in t and SHAPE_ALIASES[token] is not None:
                desc["shape"] = SHAPE_ALIASES[token]
                break

    return desc


def _parse_subtract_descriptors(question: str) -> List[Dict[str, Any]]:
    q = question.strip()
    clauses = re.findall(r"Subtract all\s+([^\.]+)", q, flags=re.IGNORECASE)
    return [_parse_descriptor(c) for c in clauses]


def _extract_total_from_payloads(evidence: Dict[str, Any]) -> Optional[int]:
    for payload in evidence.get("raw_payloads", []):
        focus = payload.get("focus_answers")
        for entry in _flatten_focus_answers(focus):
            key = _norm(_to_text(entry.get("key")))
            value = entry.get("value", entry.get("answer"))
            if key in {"total_visible_objects", "total_object_count", "object_count", "total_count"}:
                iv = _extract_int(value)
                if iv is not None:
                    return iv

    for payload in evidence.get("raw_payloads", []):
        for fact in payload.get("visual_facts", []):
            val = fact.get("value") if isinstance(fact, dict) else None
            if isinstance(val, int) and "total" in _norm(_to_text(fact.get("fact") if isinstance(fact, dict) else fact)):
                return val
            txt = _norm(_to_text(fact.get("fact") if isinstance(fact, dict) else fact))
            m = re.search(r"there (?:are|is)\s+(\d+)\s+(?:distinct\s+)?objects", txt)
            if m:
                return int(m.group(1))
    return None


def _extract_count_answers_from_followups(evidence: Dict[str, Any]) -> Dict[str, Optional[int]]:
    counts: Dict[str, Optional[int]] = {}
    total: Optional[int] = None
    for payload in evidence.get("raw_payloads", []):
        for entry in _flatten_focus_answers(payload.get("focus_answers")):
            qtext = _norm(_to_text(entry.get("question")))
            status = _norm(_to_text(entry.get("status")))
            answer = entry.get("answer", entry.get("value"))
            if "how many visible objects are there in total" in qtext or "total visible object count" in qtext:
                if status in {"answered", "answer_provided", "answered_from_image"}:
                    iv = _extract_int(answer)
                    if iv is not None:
                        total = iv
            if "how many visible objects match descriptor" in qtext:
                m = re.search(r"descriptor '([^']+)'", qtext)
                if m:
                    desc = _norm(m.group(1))
                    counts[desc] = _extract_int(answer) if status in {"answered", "answer_provided", "answered_from_image"} else None
    if total is not None:
        counts["__total__"] = total
    return counts


def _parse_object_from_fact_text(text: str) -> Optional[Dict[str, Any]]:
    t = _norm(text)
    if not t:
        return None
    if any(phrase in t for phrase in ["count", "objects visible", "sum of", "remaining"]):
        if not re.match(r"there (?:is|are) (?:a|an|one)\b", t):
            return None

    desc = _parse_descriptor(t)
    if not desc["shape"] and not desc["colors"]:
        return None

    obj = {
        "id": None,
        "shape": desc["shape"],
        "size": desc["size"],
        "material": next(iter(desc["materials"])) if desc["materials"] else None,
        "finish": next(iter(desc["finishes"])) if desc["finishes"] else None,
        "colors": sorted(desc["colors"]),
        "raw": text,
    }
    return obj


def _parse_inventory_from_payloads(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    inventory: List[Dict[str, Any]] = []
    for payload in evidence.get("raw_payloads", []):
        for fact in payload.get("visual_facts", []):
            fact_text = _to_text(fact.get("fact") if isinstance(fact, dict) else fact)
            obj = _parse_object_from_fact_text(fact_text)
            if obj:
                obj["id"] = obj["id"] or f"obj_{len(inventory)+1}"
                inventory.append(obj)
    return inventory


def _material_matches(desc_mat: str, obj: Dict[str, Any]) -> bool:
    material = _norm(_to_text(obj.get("material")))
    finish = _norm(_to_text(obj.get("finish")))
    raw = _norm(_to_text(obj.get("raw")))
    if desc_mat == "rubber":
        return any(x in {material, finish} for x in ["rubber", "matte"]) or "rubber" in raw or "matte" in raw
    if desc_mat == "metallic":
        return any(x in {material, finish} for x in ["metallic", "shiny"]) or any(tok in raw for tok in ["metallic", "shiny", "reflective", "glossy"])
    return desc_mat in {material, finish} or desc_mat in raw


def _finish_matches(desc_fin: str, obj: Dict[str, Any]) -> bool:
    finish = _norm(_to_text(obj.get("finish")))
    raw = _norm(_to_text(obj.get("raw")))
    if desc_fin == "shiny":
        return finish == "shiny" or any(tok in raw for tok in ["shiny", "metallic", "reflective", "glossy"])
    if desc_fin == "matte":
        return finish == "matte" or any(tok in raw for tok in ["matte", "rubber"])
    return desc_fin == finish or desc_fin in raw


def _matches_descriptor(obj: Dict[str, Any], desc: Dict[str, Any]) -> bool:
    shape = obj.get("shape")
    colors = set(obj.get("colors") or [])
    size = obj.get("size")
    raw = _norm(_to_text(obj.get("raw")))

    if desc["shape"] and shape != desc["shape"]:
        if not (desc["shape"] == "sphere" and shape == "sphere"):
            return False
    if desc["size"] and size != desc["size"]:
        return False
    if desc["colors"] and not (desc["colors"] & colors):
        if not any(c in raw for c in desc["colors"]):
            return False
    for mat in desc["materials"]:
        if not _material_matches(mat, obj):
            return False
    for fin in desc["finishes"]:
        if not _finish_matches(fin, obj):
            return False
    return True


def _union_count_for_descriptors(inventory: List[Dict[str, Any]], descriptors: List[Dict[str, Any]]) -> int:
    matched_ids = set()
    for desc in descriptors:
        for obj in inventory:
            if _matches_descriptor(obj, desc):
                matched_ids.add(obj["id"])
    return len(matched_ids)


def solve_synthetic_subtraction(task: TaskPacket, evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _is_subtraction_question(task):
        return None

    descriptors = _parse_subtract_descriptors(task.question)
    followup_counts = _extract_count_answers_from_followups(evidence)
    total = followup_counts.get("__total__") or _extract_total_from_payloads(evidence)

    # Preferred path: use explicit targeted descriptor counts returned by stage-2 visual QA.
    if total is not None and descriptors:
        keys = [_norm(d["raw"]) for d in descriptors]
        if all(k in followup_counts and followup_counts[k] is not None for k in keys):
            subtract_n = sum(int(followup_counts[k] or 0) for k in keys)
            remaining = total - subtract_n
            return {
                "reasoning_steps": [
                    f"Used targeted follow-up counts from the visual extractor. Total visible objects = {total}.",
                    *[
                        f"Objects matching descriptor '{k}' = {int(followup_counts[k] or 0)}."
                        for k in keys
                    ],
                    f"Computed remaining objects = {total} - {subtract_n} = {remaining}.",
                ],
                "candidate_answer": remaining,
                "answer_confidence": 0.93,
                "needs_visual_recheck": False,
                "focus_questions": [],
                "normalization_notes": "synthetic_subtraction_rule_based: targeted descriptor counts",
            }

    # If we do not yet have targeted descriptor counts, ask for them.
    if descriptors and not any(k for k in followup_counts.keys() if k != "__total__"):
        focus_questions = ["How many visible objects are there in total? Return integer only."]
        for d in descriptors:
            focus_questions.append(
                f"How many visible objects match descriptor '{d['raw']}'? Return integer only. Count each matching object once."
            )
        return {
            "reasoning_steps": [
                "Detected a synthetic subtraction/counting question.",
                "Visual object facts are present, but the direct remaining-count field is not trusted because it has been inconsistent with its own explanation in this slice.",
                "Requesting targeted counts for the total object count and for each subtraction descriptor.",
            ],
            "candidate_answer": None,
            "answer_confidence": 0.0,
            "needs_visual_recheck": True,
            "focus_questions": focus_questions,
            "normalization_notes": "synthetic_subtraction_rule_based: request targeted total/descriptors only",
        }

    # Fallback: deterministic parse from inventory if targeted counts are unavailable after recheck.
    inventory = _parse_inventory_from_payloads(evidence)
    if total is None and inventory:
        total = len(inventory)
    if total is not None and inventory and descriptors:
        subtract_n = _union_count_for_descriptors(inventory, descriptors)
        remaining = total - subtract_n
        return {
            "reasoning_steps": [
                f"Built an object inventory with {len(inventory)} parsed objects; using total visible objects = {total}.",
                *[
                    f"Parsed descriptor '{d['raw']}' as shape={d['shape']}, size={d['size']}, colors={sorted(d['colors'])}, materials={sorted(d['materials'])}, finishes={sorted(d['finishes'])}."
                    for d in descriptors
                ],
                f"Subtracted the union of all matching objects ({subtract_n}) and computed remaining objects = {remaining}.",
                "This is a fallback because explicit targeted descriptor counts were unavailable.",
            ],
            "candidate_answer": remaining,
            "answer_confidence": 0.58,
            "needs_visual_recheck": False,
            "focus_questions": [],
            "normalization_notes": "synthetic_subtraction_rule_based: parsed inventory fallback",
        }

    # Last resort: still request targeted counts.
    focus_questions = ["How many visible objects are there in total? Return integer only."]
    for d in descriptors:
        focus_questions.append(
            f"How many visible objects match descriptor '{d['raw']}'? Return integer only. Count each matching object once."
        )
    return {
        "reasoning_steps": [
            "Detected a synthetic subtraction/counting question.",
            "Current evidence is insufficient to compute a reliable remaining count deterministically.",
        ],
        "candidate_answer": None,
        "answer_confidence": 0.0,
        "needs_visual_recheck": True,
        "focus_questions": focus_questions,
        "normalization_notes": "synthetic_subtraction_rule_based: final targeted count request",
    }
