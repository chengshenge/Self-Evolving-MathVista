from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .generated_paths import resolved_generated_skills_root
from .schemas import SolveTrace, TaskPacket


SUCCESS_SYNTHESIS_SYSTEM_PROMPT = """You are success_synthesizer, a narrow-expert profile writer for a multimodal reasoning system.
You will read a SUCCESSFUL solve trace and produce a reusable narrow expert profile.
Your goal is to summarize transferable experience, not to memorize the sample.
Return JSON only with keys:
- visual_hint: 1-2 concise sentences telling the visual stage what to prioritize for similar tasks
- reasoning_hint: 1-2 concise sentences telling the reasoning stage how to solve similar tasks
- verifier_hint: 1-2 concise sentences telling the verifier what must be grounded before accepting an answer
- keywords: list[str] of concise reusable keywords
- retrieval_text_bge: one short paragraph for semantic text retrieval of this expert
- retrieval_text_clip: one short paragraph that visually describes what kinds of images/questions this expert applies to
- confidence: float in [0,1]
Do NOT include the final answer verbatim inside any hint unless it is a category-level label like odd/even/yes/no that is itself the general skill target.
Do NOT memorize sample-specific names, dates, exact measurements, or unique labels unless they are clearly reusable category cues.
Prefer category-level strategies such as what visual evidence matters, how to structure the math, and how to verify the final answer form.
"""


def solve_with_profile(query: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    from .orchestrator import MultimodalMetaAgent

    task = TaskPacket.from_query_json(query)
    agent = MultimodalMetaAgent(profile=profile, allow_generated_skill_reuse=False)
    result = agent.solve(task)
    return {
        "answer": result.final_answer_normalized,
        "summary": json.dumps(
            {
                "used_profile": profile.get("name"),
                "correct": result.correct,
                "workspace": result.workspace,
            },
            ensure_ascii=False,
        ),
    }


def _pick_model_config(cfg):
    if getattr(cfg, "orchestrator_model", None) and getattr(cfg.orchestrator_model, "enabled", False):
        return cfg.orchestrator_model
    return cfg.reasoning_model


def _dedupe_keywords(parts: List[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple, set)):
            items = list(part)
        else:
            items = [part]
        for item in items:
            text = str(item).strip().lower()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


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


def _truncate_list(xs: Any, n: int = 8) -> List[Any]:
    if not isinstance(xs, list):
        return []
    return xs[:n]


def _sentences(text: Any, max_sentences: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:max_sentences]).strip()


def _generic_category_answer(value: str) -> bool:
    v = str(value or "").strip().lower()
    return v in {"yes", "no", "odd", "even", "neither", "true", "false", "a", "b", "c", "d", "e"}


def _protectable_value(value: Any) -> Optional[str]:
    v = str(value or "").strip()
    if not v:
        return None
    if _generic_category_answer(v):
        return None
    compact = re.sub(r"\s+", " ", v)
    alnum = re.sub(r"[^0-9A-Za-z]+", "", compact)
    if len(alnum) <= 2:
        return None
    return compact


def _replace_protected_value(text: str, value: str) -> str:
    token = _protectable_value(value)
    if token is None:
        return text
    escaped = re.escape(token)
    if re.fullmatch(r"[0-9A-Za-z ._:/+\-]+", token):
        pattern = rf"(?<![0-9A-Za-z]){escaped}(?![0-9A-Za-z])"
    else:
        pattern = escaped
    return re.sub(pattern, "[ANSWER]", text, flags=re.IGNORECASE)


def _collapse_answer_placeholders(text: str) -> str:
    s = text
    s = s.replace("[ANSW[ANSWER]R]", "[ANSWER]")
    s = re.sub(r"(?:\[ANSWER\]){2,}", "[ANSWER]", s)
    s = re.sub(r"\s*\[ANSWER\]\s*", " [ANSWER] ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_hint(text: Any, *, protected_values: Optional[List[str]] = None) -> str:
    original = str(text or "").strip()
    s = original
    s = re.sub(r"(?i)\b(the\s+)?correct answer is\b.*", "", s)
    s = re.sub(r"(?i)\btherefore\b.*", "", s)
    s = re.sub(r"(?i)\bfor this (image|question|sample)\b", "", s)
    for value in protected_values or []:
        s = _replace_protected_value(s, str(value or ""))
    s = _collapse_answer_placeholders(s)
    if s.count("[ANSWER]") >= 3:
        s = original
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


def _scene_and_task(trace: SolveTrace) -> tuple[str, str]:
    task = trace.task
    scene = (
        trace.evidence.scene_type
        if trace.evidence and trace.evidence.scene_type != "unknown"
        else str(task.metadata.get("context") or "unknown")
    )
    task_name = str(task.metadata.get("task") or task.metadata.get("context") or "general")
    return scene, task_name


def _fallback_profile_from_trace(trace: SolveTrace) -> Dict[str, Any]:
    task = trace.task
    meta = task.metadata or {}
    q = str(task.question or "").lower()
    context = str(meta.get("context") or "").lower()
    source = str(meta.get("source") or "").lower()
    scene, task_name = _scene_and_task(trace)

    specialization = {
        "scene_type": scene,
        "task": meta.get("task"),
        "context": meta.get("context"),
        "category": meta.get("category"),
        "source": meta.get("source"),
        "skills": meta.get("skills", []),
        "question_type": task.question_type,
        "answer_type": task.answer_type,
    }
    keywords = _dedupe_keywords(
        [
            scene,
            meta.get("task"),
            meta.get("context"),
            meta.get("category"),
            meta.get("source"),
            task.question_type,
            task.answer_type,
            meta.get("skills", []),
        ]
    )

    if context == "geometry diagram" or "unigeo" in source:
        visual_hint = (
            "Identify the labeled geometric relations first: given angles, equal-length marks, parallel lines, and intersecting segments. "
            "Extract only the constraints that directly determine the target quantity."
        )
        reasoning_hint = (
            "Use structured geometric relations such as angle chasing, triangle similarity, parallel-line angle rules, or polygon decomposition instead of free-form description. "
            "Prefer the shortest derivation that lands on a canonical choice or scalar answer."
        )
        verifier_hint = (
            "Accept a geometry answer only when the decisive relation is explicit and the final quantity matches the requested answer format."
        )
        bge = "Geometry-diagram expert for labeled figures requiring angle chasing, length relations, similarity, or direct extraction of a canonical multiple-choice or numeric answer."
        clip = "A geometry diagram with labeled points, angle marks, parallel lines, or intersecting segments where a small set of geometric constraints determines the answer."
    elif ("subtract all" in q) or ("synthetic scene" in context and "how many" in q) or ("clevr" in source):
        visual_hint = (
            "Count the total visible objects first, then count each requested descriptor explicitly and keep those counts separate. "
            "Prefer targeted descriptor counts over noisy full-scene inventory reconstruction."
        )
        reasoning_hint = (
            "For subtraction and counting questions, compute from grounded counts only: total count, descriptor-specific counts, then the final arithmetic. "
            "Do not trust an inventory fallback if it conflicts with targeted counts or yields unstable object totals."
        )
        verifier_hint = (
            "Reject a synthetic counting answer unless the total and every descriptor-specific count are grounded in the visual evidence."
        )
        bge = "Synthetic-scene counting expert for CLEVR-style math word problems where the answer depends on total object count, descriptor filtering, and explicit arithmetic over grounded counts."
        clip = "A synthetic tabletop scene with colored geometric objects where the task asks for counts, subtraction, or comparison after filtering by attributes like color, size, shape, or material."
    elif context == "bar chart" or "chartqa" in source or "bar chart" in q:
        visual_hint = (
            "Read the chart title, axes, category labels, and exact bar values before reasoning. "
            "Align the compared bars or categories explicitly instead of inferring from rough visual height alone."
        )
        reasoning_hint = (
            "Reduce chart questions to a small table of grounded values, then do only the requested comparison or arithmetic. "
            "Be careful about largest-smallest wording and whether the task expects a number, a category, or a yes/no answer."
        )
        verifier_hint = (
            "Accept a chart answer only if the referenced bar values or categories are explicitly grounded and the final format matches the question."
        )
        bge = "Bar-chart reasoning expert for questions that require reading exact values, comparing categories, or computing aggregates from visually grounded chart elements."
        clip = "A bar chart or simple chart with labeled categories and numeric values where exact value reading and careful comparison determine the answer."
    elif "clock" in q or "clock" in context or "clock" in source:
        visual_hint = (
            "Identify the hour and minute hands separately, then determine the exact minute position before estimating any derived quantity."
        )
        reasoning_hint = (
            "Translate the clock face into a structured time representation first, then perform any angle or elapsed-time calculation from that representation."
        )
        verifier_hint = (
            "Reject a clock answer if the hand interpretation or the requested time/angle conversion is ambiguous."
        )
        bge = "Analog-clock expert for reading hand positions and converting them into exact times, elapsed durations, or clock-angle calculations."
        clip = "An analog clock face where the answer depends on accurately reading the hour and minute hands and converting them into time or angle information."
    elif ("age gap" in q) or ("kvqa" in source and ("age" in q or "born after" in q)):
        visual_hint = (
            "Prioritize readable names, captions, inscriptions, jersey text, or contextual cues before estimating age from appearance. "
            "If identity is not grounded, do not over-commit to precise age ranges."
        )
        reasoning_hint = (
            "Do not replace weak appearance-based evidence with aggressive midpoint arithmetic over broad age ranges. "
            "Prefer grounded identity facts, explicit textual evidence, or abstain rather than fabricate a precise gap."
        )
        verifier_hint = (
            "Reject precise age-gap answers unless they are supported by grounded identity evidence or explicit age/birth information."
        )
        bge = "Identity and age-gap expert for natural-image questions where the answer depends on grounding people using contextual clues instead of pure facial-age guessing."
        clip = "A natural image with one or more people where names, captions, uniforms, inscriptions, or scene context may matter more than appearance-only age estimation."
    elif ("functionqa" in source) or any(x in q for x in ["odd", "even", "continuous", "discontinuous", "plot", "graph"]):
        visual_hint = (
            "Inspect the graph for decisive visual properties such as symmetry, continuity breaks, holes, jumps, or monotonic segments before answering."
        )
        reasoning_hint = (
            "Preserve the required graph-property label, such as odd, even, continuous, or neither, instead of collapsing the answer into generic yes/no language."
        )
        verifier_hint = (
            "Reject graph answers that are not in the expected property form or are unsupported by explicit visual graph evidence."
        )
        bge = "Function-plot expert for graph questions that ask for parity, continuity, or other structural properties instead of generic textual descriptions."
        clip = "A function plot or graph where symmetry, continuity breaks, or other visual graph properties determine the correct property label."
    elif any(x in q for x in ["force", "spring", "mass", "velocity", "acceleration"]) or "scienceqa" in source or "textbook" in source or "physics" in context:
        visual_hint = (
            "Extract the labeled quantities, symbols, and diagram annotations before reasoning; separate decorative text from the values that actually constrain the answer."
        )
        reasoning_hint = (
            "Convert the diagram into a compact structured representation, then apply the smallest relevant rule or formula needed for the requested quantity."
        )
        verifier_hint = (
            "Accept a science-diagram answer only if the used quantities are explicitly grounded in the figure and the final unit/format is correct."
        )
        bge = "Scientific-figure expert for textbook-style diagrams where success depends on extracting labeled quantities and applying a small number of grounded rules or formulas."
        clip = "A scientific or textbook-style diagram with labels, arrows, quantities, or symbols where the answer depends on grounded extraction of the relevant annotations."
    else:
        visual_hint = (
            f"For scene={scene} and task={task_name}, extract the smallest set of decisive visual facts first and ignore weakly related details."
        )
        reasoning_hint = (
            "Reason from grounded facts only, prefer concise structured intermediate steps, and return the most canonical answer form for the task."
        )
        verifier_hint = (
            "Accept the answer only if the decisive evidence is grounded and the final output matches the required answer type."
        )
        bge = f"Narrow expert for scene={scene} and task={task_name}, focused on decisive visual evidence, grounded reasoning, and canonical final answers."
        clip = f"An image-question pair in scene={scene} for task={task_name}, where a small set of decisive visual cues determines the answer."

    from .schemas import slugify

    return {
        "name": slugify(f"{scene}_{task_name}_solver"),
        "visual_hint": visual_hint,
        "reasoning_hint": reasoning_hint,
        "verifier_hint": verifier_hint,
        "retrieval_text_bge": bge,
        "retrieval_text_clip": clip,
        "keywords": keywords,
        "specialization": specialization,
    }


def _success_summary_for_synthesis(trace: SolveTrace) -> Dict[str, Any]:
    data = _safe_trace_dict(trace)
    task = _remove_answer_field(data.get("task") or {})
    evidence = data.get("evidence") or {}
    visual_rounds = data.get("visual_rounds") or []
    math_rounds = data.get("math_rounds") or []
    verify_rounds = data.get("verify_rounds") or []

    payload: Dict[str, Any] = {
        "task": task,
        "scene_type": evidence.get("scene_type"),
        "visual_facts": _truncate_list(evidence.get("visual_facts"), 10),
        "uncertainties": _truncate_list(evidence.get("uncertainties"), 8),
        "focus_answers": _truncate_list(evidence.get("focus_answers"), 6),
        "visual_rounds": _truncate_list(visual_rounds, 2),
        "math_rounds": _truncate_list(math_rounds, 2),
        "verify_rounds": _truncate_list(verify_rounds, 2),
        "final_answer_raw": data.get("final_answer_raw"),
        "final_answer_normalized": data.get("final_answer_normalized"),
        "used_generated_skill": data.get("used_generated_skill"),
        "saved_generated_skill": data.get("saved_generated_skill"),
    }
    return payload


def _synthesize_success_profile(trace: SolveTrace, project_root: str | Path) -> Dict[str, Any]:
    project_root = Path(project_root)
    fallback = _fallback_profile_from_trace(trace)

    try:
        from .config import load_runtime_config
        from .llm_clients import OpenAIStyleClient

        cfg = load_runtime_config(project_root)
        client = OpenAIStyleClient(_pick_model_config(cfg))
        raw = client.complete_json(
            SUCCESS_SYNTHESIS_SYSTEM_PROMPT,
            json.dumps({
                "successful_trace_summary": _success_summary_for_synthesis(trace),
                "instruction": (
                    "Generate a reusable narrow-expert profile for future similar tasks. "
                    "Summarize what visual evidence mattered, what reasoning pattern succeeded, and how the verifier should validate the answer."
                ),
            }, ensure_ascii=False, indent=2),
            max_tokens=1200,
        )
    except Exception:
        raw = {}

    if not isinstance(raw, dict) or not raw:
        raw = {}

    protected_values = [getattr(trace, "final_answer_normalized", None), getattr(trace, "final_answer_raw", None)]
    visual_hint = _sanitize_hint(raw.get("visual_hint"), protected_values=protected_values) or fallback["visual_hint"]
    reasoning_hint = _sanitize_hint(raw.get("reasoning_hint"), protected_values=protected_values) or fallback["reasoning_hint"]
    verifier_hint = _sanitize_hint(raw.get("verifier_hint"), protected_values=protected_values) or fallback["verifier_hint"]

    keywords = _sanitize_keywords(raw.get("keywords"))
    keywords = _dedupe_keywords([fallback.get("keywords", []), keywords])

    return {
        "visual_hint": visual_hint,
        "reasoning_hint": reasoning_hint,
        "verifier_hint": verifier_hint,
        "retrieval_text_bge": _sanitize_hint(raw.get("retrieval_text_bge"), protected_values=protected_values)
            or fallback["retrieval_text_bge"],
        "retrieval_text_clip": _sanitize_hint(raw.get("retrieval_text_clip"), protected_values=protected_values)
            or fallback["retrieval_text_clip"],
        "keywords": keywords,
        "hint_generation_mode": "success_synthesis_llm" if raw else "success_synthesis_fallback",
        "hint_generation_confidence": float(raw.get("confidence", 0.6) or 0.6) if raw else 0.45,
    }


def _merge_profile(base_profile: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not override:
        return base_profile

    merged = dict(base_profile)
    for key, value in override.items():
        if value is None:
            continue
        if key == "keywords":
            merged["keywords"] = _dedupe_keywords([base_profile.get("keywords", []), value])
        elif key == "specialization" and isinstance(value, dict):
            spec = dict(base_profile.get("specialization", {}))
            spec.update({k: v for k, v in value.items() if v is not None})
            merged["specialization"] = spec
        elif key == "name":
            merged["name"] = str(value)
        else:
            merged[key] = value
    return merged


def _registry_path(project_root: Path) -> Path:
    return resolved_generated_skills_root(project_root) / "_generated_skill_registry.jsonl"


def _read_registry(project_root: Path) -> List[Dict[str, Any]]:
    path = _registry_path(project_root)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except Exception:
            continue
    return rows


def _append_registry(project_root: Path, row: Dict[str, Any]) -> None:
    path = _registry_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _existing_skill_for_trace(trace: SolveTrace, project_root: Path) -> Optional[Path]:
    workspace = str(getattr(trace, "workspace", "") or "")
    sample_id = str(getattr(getattr(trace, "task", None), "sample_id", "") or "")
    base = resolved_generated_skills_root(project_root)
    for row in reversed(_read_registry(project_root)):
        skill_name = row.get("skill_name")
        if not skill_name:
            continue
        p = base / skill_name
        if not p.exists():
            continue
        if workspace and row.get("source_workspace") == workspace:
            return p
        if sample_id and workspace and row.get("sample_id") == sample_id and row.get("source_workspace") == workspace:
            return p
    return None


def build_profile_from_trace(
    trace: SolveTrace,
    project_root: str | Path,
    reflected_profile_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    project_root = Path(project_root)
    base_profile = _fallback_profile_from_trace(trace)
    synthesized = _synthesize_success_profile(trace, project_root)
    profile = _merge_profile(base_profile, synthesized)

    override = reflected_profile_override
    if override is None:
        override = getattr(trace, "reflection_profile_override", None)
    if override:
        profile = _merge_profile(profile, override)
        profile["hint_generation_mode"] = "reflection_success"
    return profile


def write_skill_from_profile(
    profile: Dict[str, Any],
    dst_dir: str | Path,
    *,
    task: Optional[TaskPacket] = None,
    extra_note: str = "",
) -> Path:
    skill_dir = Path(dst_dir)
    skill_dir.mkdir(parents=True, exist_ok=True)
    profile = dict(profile)
    profile["name"] = skill_dir.name

    (skill_dir / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    spec = profile.get("specialization", {}) if isinstance(profile.get("specialization"), dict) else {}
    scene = spec.get("scene_type", "unknown")
    task_name = str(spec.get("task") or spec.get("context") or "general")
    question_type = task.question_type if task is not None else str(spec.get("question_type") or "unknown")
    answer_type = task.answer_type if task is not None else str(spec.get("answer_type") or "unknown")

    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {profile['name']}
description: Generated multimodal solver specialized for scene={scene}; task={task_name}; qtype={question_type}; atype={answer_type}.
entry_file: solver.py
---
This generated skill was created from a successful multimodal trace. It reuses the same visual -> math -> verify -> normalize pipeline, but injects extra prompt hints for tasks similar to: scene={scene}, task={task_name}, qtype={question_type}, atype={answer_type}. Use it as a narrow, descriptive expert for closely related tasks.
{extra_note}
""",
        encoding="utf-8",
    )

    solver_code = (
        "from muse.compose import solve_with_profile\n"
        "import json\n\n"
        f"PROFILE = json.loads('''{json.dumps(profile, ensure_ascii=False)}''')\n\n\n"
        "def main(query: str):\n"
        "    return solve_with_profile(query, PROFILE)\n"
    )
    (skill_dir / "solver.py").write_text(solver_code, encoding="utf-8")
    return skill_dir


def save_composed_skill_from_trace(trace: SolveTrace, project_root: str | Path) -> Optional[Path]:
    project_root = Path(project_root)
    if not trace.correct:
        return None
    if trace.used_generated_skill:
        return None

    profile = build_profile_from_trace(trace, project_root)
    override = getattr(trace, "reflection_profile_override", None)
    existing = _existing_skill_for_trace(trace, project_root)
    if existing is not None:
        profile["name"] = existing.name
        return existing

    base = resolved_generated_skills_root(project_root)
    skill_dir = base / profile["name"]
    skill_dir.mkdir(parents=True, exist_ok=True)
    profile["name"] = skill_dir.name

    provenance = {
        "sample_id": getattr(getattr(trace, "task", None), "sample_id", None),
        "source_workspace": getattr(trace, "workspace", None),
        "used_generated_skill": getattr(trace, "used_generated_skill", None),
        "generation_mode": profile.get("hint_generation_mode", "success_synthesis_fallback"),
        "final_answer_normalized": getattr(trace, "final_answer_normalized", None),
    }
    profile["provenance"] = provenance

    extra = ""
    if override:
        extra = (
            "\nThis generated skill was refined via reflection on a failed seed trace and stores reflection-derived hints in addition to synthesized success hints.\n"
        )
    elif profile.get("hint_generation_mode") == "success_synthesis_llm":
        extra = (
            "\nThis generated skill uses LLM-synthesized visual, reasoning, and verifier hints distilled from a successful seed trace.\n"
        )
    else:
        extra = (
            "\nThis generated skill uses fallback differentiated hints derived from the successful trace metadata and task family.\n"
        )

    write_skill_from_profile(profile, skill_dir, task=trace.task, extra_note=extra)

    _append_registry(project_root, {
        "skill_name": skill_dir.name,
        "sample_id": getattr(getattr(trace, "task", None), "sample_id", None),
        "source_workspace": getattr(trace, "workspace", None),
        "generation_mode": profile.get("hint_generation_mode"),
        "reflection": bool(override),
    })
    return skill_dir
