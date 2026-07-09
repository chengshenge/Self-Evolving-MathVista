from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .generated_paths import resolved_generated_skills_root
from .reuse_registry import get_skill_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = PROJECT_ROOT / "skills"
SUBAGENTS_ROOT = SKILLS_ROOT / "subagents"
GENERATED_ROOT = SUBAGENTS_ROOT / "generated"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "by", "from", "at",
    "is", "are", "was", "were", "be", "this", "that", "these", "those", "it", "its", "into", "then",
    "when", "what", "which", "how", "why", "who", "whom", "whose", "about", "please", "provide",
    "using", "use", "give", "find", "compute", "calculate", "daily", "life", "grade", "english",
    "hint", "question", "answer", "answers", "choice", "choices", "option", "options", "letter",
}

# Weak / generic routing tokens. These should contribute little to matching because they cause
# kitchen / tabletop / generic visual solvers to match unrelated natural-image tasks.
_GENERIC_ROUTING_TOKENS = {
    "general", "visual", "question", "answering", "vqa", "reasoning", "problem", "solving", "scene",
    "image", "images", "figure", "free", "form", "multi", "choice", "text", "integer", "float",
    "natural", "synthetic", "math", "targeted", "commonsense", "arithmetic", "algebraic",
}


def parse_skill_md(path: str | Path) -> Optional[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return None

    frontmatter = match.group(1)
    body = match.group(2).strip()
    data: Dict[str, Any] = {"instructions": body}
    for line in frontmatter.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def list_subagent_skill_dirs() -> List[Path]:
    if not SUBAGENTS_ROOT.exists():
        return []
    explicit_root = resolved_generated_skills_root(PROJECT_ROOT)
    default_generated = GENERATED_ROOT
    skip_default_generated = explicit_root.resolve() != default_generated.resolve()
    items: List[Path] = []
    for path in SUBAGENTS_ROOT.rglob("SKILL.md"):
        if skip_default_generated:
            try:
                path.parent.resolve().relative_to(default_generated.resolve())
                continue
            except ValueError:
                pass
        items.append(path.parent)
    return sorted(items)


def _guess_skill_entry_file(skill_dir: Path) -> Optional[str]:
    for name in [f"{skill_dir.name}.py", "solver.py", "main.py"]:
        if (skill_dir / name).exists():
            return name

    py_files = sorted(
        p.name
        for p in skill_dir.iterdir()
        if p.is_file() and p.suffix == ".py" and not p.name.startswith("__")
    ) if skill_dir.exists() else []
    if len(py_files) == 1:
        return py_files[0]

    for name in py_files:
        lower = name.lower()
        if any(tok in lower for tok in ["agent", "reason", "verify", "normal", "solver"]):
            return name
    return None


def _fallback_skill_dirs() -> List[Path]:
    if not SUBAGENTS_ROOT.exists():
        return []
    explicit_root = resolved_generated_skills_root(PROJECT_ROOT)
    skip_default_generated = explicit_root.resolve() != GENERATED_ROOT.resolve()
    out: List[Path] = []
    for child in sorted(SUBAGENTS_ROOT.iterdir()):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        if child.name == "generated":
            if skip_default_generated:
                continue
            for gchild in sorted(child.iterdir()):
                if gchild.is_dir() and not gchild.name.startswith("__"):
                    out.append(gchild)
            continue
        out.append(child)
    return out


def list_subagent_skills() -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen_dirs: Set[str] = set()

    for skill_dir in list_subagent_skill_dirs():
        info = parse_skill_md(skill_dir / "SKILL.md")
        if not info:
            continue
        entry_file = info.get("entry_file", "solver.py")
        result.append(
            {
                "name": info.get("name", skill_dir.name),
                "description": info.get("description", ""),
                "entry_file": entry_file,
                "directory": str(skill_dir),
            }
        )
        seen_dirs.add(str(skill_dir.resolve()))

    for skill_dir in _fallback_skill_dirs():
        key = str(skill_dir.resolve())
        if key in seen_dirs:
            continue
        entry_file = _guess_skill_entry_file(skill_dir)
        if not entry_file:
            continue
        result.append(
            {
                "name": skill_dir.name,
                "description": "fallback_discovered_without_skill_md",
                "entry_file": entry_file,
                "directory": str(skill_dir),
            }
        )
        seen_dirs.add(key)

    explicit_root = resolved_generated_skills_root(PROJECT_ROOT)
    default_generated = PROJECT_ROOT / "skills" / "subagents" / "generated"
    if explicit_root.resolve() != default_generated.resolve() and explicit_root.exists():
        for skill_dir in sorted(explicit_root.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
                continue
            key = str(skill_dir.resolve())
            if key in seen_dirs:
                continue
            entry_file = _guess_skill_entry_file(skill_dir)
            if not entry_file:
                continue
            result.append(
                {
                    "name": skill_dir.name,
                    "description": "generated_skill_explicit_root",
                    "entry_file": entry_file,
                    "directory": str(skill_dir),
                }
            )
            seen_dirs.add(key)

    return result



def find_skill_by_name(skill_name: str) -> Optional[Dict[str, Any]]:
    for item in list_subagent_skills():
        if item["name"] == skill_name or Path(item["directory"]).name == skill_name:
            return item
    return None


def get_skill_profile(skill_dir: str | Path) -> Dict[str, Any]:
    profile_path = Path(skill_dir) / "profile.json"
    if profile_path.exists():
        return json.loads(profile_path.read_text(encoding="utf-8"))
    return {}


def _tokenize(parts: Iterable[Any]) -> Set[str]:
    tokens: Set[str] = set()
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple, set)):
            tokens |= _tokenize(part)
            continue
        if isinstance(part, dict):
            tokens |= _tokenize(part.values())
            continue
        text = str(part).strip().lower()
        if not text:
            continue
        for token in re.findall(r"[a-z0-9]+", text):
            if len(token) <= 1:
                continue
            if token in _STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def _task_keywords(task: Any) -> Set[str]:
    if hasattr(task, "to_dict"):
        task = task.to_dict()
    meta = task.get("metadata") or {}
    parts: List[Any] = [
        task.get("question"),
        task.get("query"),
        task.get("choices", []),
        task.get("question_type"),
        task.get("answer_type"),
        meta.get("context"),
        meta.get("task"),
        meta.get("category"),
        meta.get("source"),
        meta.get("skills", []),
    ]
    return _tokenize(parts)


def _skill_keywords(skill_info: Dict[str, Any], profile: Dict[str, Any]) -> Set[str]:
    spec = profile.get("specialization", {}) if isinstance(profile, dict) else {}
    parts: List[Any] = [
        skill_info.get("name"),
        skill_info.get("description"),
        profile.get("keywords", []),
        spec.get("scene_type"),
        spec.get("task"),
        spec.get("context"),
        spec.get("category"),
        spec.get("source"),
        spec.get("skills", []),
        spec.get("question_type"),
        spec.get("answer_type"),
    ]
    return _tokenize(parts)


def _content_tokens(tokens: Set[str]) -> Set[str]:
    return {t for t in tokens if t not in _GENERIC_ROUTING_TOKENS}


def _scene_signature_tokens(skill_info: Dict[str, Any], profile: Dict[str, Any]) -> Set[str]:
    spec = profile.get("specialization", {}) if isinstance(profile, dict) else {}
    parts: List[Any] = [
        skill_info.get("name"),
        spec.get("scene_type"),
        spec.get("context"),
        spec.get("task"),
        profile.get("keywords", []),
    ]
    tokens = _content_tokens(_tokenize(parts))
    # Drop purely generic geometry/math tokens from scene signature so they don't over-match.
    return {t for t in tokens if t not in {"geometry", "diagram", "visual", "question", "answering", "solver", "problem", "solving"}}


def _task_signature_tokens(task: Any) -> Set[str]:
    tokens = _content_tokens(_task_keywords(task))
    # Keep chart / geometry / scene nouns, drop answer-format words.
    return {t for t in tokens if t not in {"integer", "float", "text", "choice", "choices", "option", "options"}}



def _is_identity_age_gap_task(task: Any) -> bool:
    task_dict = task.to_dict() if hasattr(task, "to_dict") else dict(task)
    q = str(task_dict.get("question") or "").strip().lower()
    meta = task_dict.get("metadata") or {}
    src = str(meta.get("source") or "").strip().lower()
    return (
        "age gap" in q
        or "born after the end of world war ii" in q
        or ("kvqa" in src and ("age" in q or "born after" in q))
    )


def _skill_is_agegap_relevant(skill_info: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    spec = profile.get("specialization", {}) if isinstance(profile, dict) else {}
    parts: List[Any] = [
        skill_info.get("name"),
        skill_info.get("description"),
        profile.get("keywords", []),
        spec.get("scene_type"),
        spec.get("task"),
        spec.get("context"),
        spec.get("category"),
        spec.get("source"),
        spec.get("skills", []),
    ]
    text = " ".join(sorted(_tokenize(parts))).lower()
    keep_markers = {
        "age", "gap", "identity", "entity", "historical", "history", "public", "figure",
        "portrait", "person", "people", "celebrity", "birth", "born", "wikidata",
        "king", "queen", "monarch", "leader", "politician", "actor", "actress",
        "singer", "musician", "royal", "vintage", "world", "war", "ii",
    }
    return any(marker in text for marker in keep_markers)


def get_generated_skill_candidates(
    task: Any,
    project_root: Optional[str | Path] = None,
    *,
    top_k: int = 3,
    min_score: float = 4.5,
) -> List[Dict[str, Any]]:
    task_dict = task.to_dict() if hasattr(task, "to_dict") else dict(task)
    meta = task_dict.get("metadata") or {}
    task_keywords = _task_keywords(task_dict)
    task_signature = _task_signature_tokens(task_dict)
    task_skills = {str(x).strip().lower() for x in (meta.get("skills") or []) if str(x).strip()}

    age_gap_task = _is_identity_age_gap_task(task_dict)

    candidates: List[Dict[str, Any]] = []
    explicit_root = resolved_generated_skills_root(project_root or PROJECT_ROOT).resolve()
    for item in list_subagent_skills():
        directory = Path(item["directory"])
        in_explicit_root = False
        try:
            directory.resolve().relative_to(explicit_root)
            in_explicit_root = True
        except ValueError:
            pass
        if not in_explicit_root and "generated" not in directory.parts:
            continue

        profile = get_skill_profile(directory)
        if age_gap_task and not _skill_is_agegap_relevant(item, profile):
            continue
        spec = profile.get("specialization", {}) if isinstance(profile, dict) else {}
        score = 0.0
        matched_fields: List[str] = []

        spec_task = str(spec.get("task") or "").strip().lower()
        spec_context = str(spec.get("context") or "").strip().lower()
        spec_scene = str(spec.get("scene_type") or "").strip().lower()
        task_name = str(meta.get("task") or "").strip().lower()
        context = str(meta.get("context") or "").strip().lower()
        exact_bucket = False

        if spec_task and task_name and spec_task == task_name:
            score += 4.0
            matched_fields.append("task:exact")
        if spec_context and context and spec_context == context:
            score += 3.0
            matched_fields.append("context:exact")
        if spec_scene and context and spec_scene == context:
            score += 2.0
            matched_fields.append("scene/context:exact")
        if str(spec.get("question_type") or "").strip().lower() == str(task_dict.get("question_type") or "").strip().lower():
            score += 1.0
            matched_fields.append("question_type")
        if str(spec.get("answer_type") or "").strip().lower() == str(task_dict.get("answer_type") or "").strip().lower():
            score += 1.0
            matched_fields.append("answer_type")
        exact_bucket = (spec_task == task_name and spec_context == context and str(spec.get("question_type") or "").strip().lower() == str(task_dict.get("question_type") or "").strip().lower() and str(spec.get("answer_type") or "").strip().lower() == str(task_dict.get("answer_type") or "").strip().lower())

        spec_skills = {str(x).strip().lower() for x in (spec.get("skills") or []) if str(x).strip()}
        shared_skills = sorted(spec_skills & task_skills)
        if shared_skills:
            bonus = min(1.5, 0.5 * len(shared_skills))
            score += bonus
            matched_fields.append(f"shared_skills:{','.join(shared_skills)}")

        keywords = _skill_keywords(item, profile)
        lexical_overlap = sorted(_content_tokens(task_keywords) & _content_tokens(keywords))
        if lexical_overlap:
            lexical_bonus = min(1.6, 0.25 * len(lexical_overlap))
            score += lexical_bonus
            matched_fields.append(f"lexical_overlap:{','.join(lexical_overlap[:8])}")

        scene_tokens = _scene_signature_tokens(item, profile)
        scene_overlap = sorted(task_signature & scene_tokens)
        if scene_overlap:
            scene_bonus = min(4.0, 1.2 * len(scene_overlap))
            score += scene_bonus
            matched_fields.append(f"scene_overlap:{','.join(scene_overlap[:8])}")

        stats = get_skill_stats(project_root or PROJECT_ROOT, item["name"], task)
        bucket_total = int(stats.get("bucket_total", 0) or 0)
        bucket_accuracy = float(stats.get("bucket_accuracy", 0.0) or 0.0)
        global_total = int(stats.get("global_total", 0) or 0)
        global_accuracy = float(stats.get("global_accuracy", 0.0) or 0.0)

        # Strong positive signal for repeatedly successful exact buckets.
        if bucket_total >= 2 and bucket_accuracy >= 0.75:
            score += 2.0 + 1.5 * (bucket_accuracy - 0.75)
            matched_fields.append(f"bucket_acc:{stats['bucket_success']}/{stats['bucket_total']}")
        elif bucket_total >= 1 and bucket_accuracy <= 0.34:
            score -= 4.0
            matched_fields.append(f"bucket_bad:{stats['bucket_success']}/{stats['bucket_total']}")
        elif global_total >= 4 and global_accuracy >= 0.75 and exact_bucket:
            score += 1.0
            matched_fields.append(f"global_acc:{stats['global_success']}/{stats['global_total']}")
        elif global_total >= 2 and global_accuracy < 0.4:
            score -= 1.75
            matched_fields.append(f"global_bad:{stats['global_success']}/{stats['global_total']}")

        # Cold-start generic skills with no scene overlap should not crowd the shortlist.
        cold_unproven = bucket_total == 0 and global_total < 2
        if cold_unproven and not scene_overlap and not exact_bucket:
            score -= 4.5
            matched_fields.append("cold_start_scene_miss")

        # Even for exact task/context matches, generic natural-image VQA skills should not rank high
        # unless they have either scene-token overlap or validated history on the exact bucket.
        if exact_bucket and bucket_total == 0 and not scene_overlap and context in {"natural image", "synthetic scene"}:
            score -= 2.5
            matched_fields.append("generic_exact_without_scene_support")

        candidate = dict(item)
        candidate.update(
            {
                "profile": profile,
                "score": round(score, 3),
                "matched_fields": matched_fields,
                "stats": stats,
                "scene_overlap": scene_overlap,
                "lexical_overlap": lexical_overlap,
            }
        )
        candidates.append(candidate)

    candidates.sort(
        key=lambda x: (
            x["score"],
            len(x.get("scene_overlap") or []),
            x["stats"].get("bucket_accuracy", 0.0),
            x["stats"].get("bucket_total", 0),
            x["stats"].get("global_accuracy", 0.0),
            x["stats"].get("global_total", 0),
        ),
        reverse=True,
    )
    return [c for c in candidates if c["score"] >= min_score][:top_k]
