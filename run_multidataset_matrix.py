#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_env(repo_root: Path) -> None:
    env_paths = []
    if os.getenv("MUSE_ENV_FILE"):
        env_paths.append(Path(os.environ["MUSE_ENV_FILE"]))
    env_paths.append(repo_root / ".env")
    for env_path in env_paths:
        if not env_path.exists():
            continue
        override = env_path == Path(os.environ["MUSE_ENV_FILE"]) if os.getenv("MUSE_ENV_FILE") else False
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if override or key not in os.environ:
                os.environ[key] = value
        break
    base = os.getenv("BASE_URL")
    key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL")
    if base:
        os.environ.setdefault("VISION_BASE_URL", base)
    if key:
        os.environ.setdefault("VISION_API_KEY", key)
    if model:
        os.environ.setdefault("VISION_MODEL", model)


REPO_ROOT = Path(__file__).resolve().parent
_load_env(REPO_ROOT)

from muse.answering import answers_equal  # noqa: E402
from muse.baseline_model import run_baseline_model  # noqa: E402
from muse.io_utils import save_json, save_jsonl  # noqa: E402
from muse.orchestrator import MultimodalMetaAgent  # noqa: E402
from muse.schemas import TaskPacket  # noqa: E402
from experiments.baseline_variants.run_baseline_variants import (  # noqa: E402
    _run_base_call,
    _run_incremental,
    _summarize,
    build_memory,
    run_memory_topk,
    run_reflection1,
    run_rollout5,
)

PROJECT_ROOT = REPO_ROOT
RESULTS_ROOT = PROJECT_ROOT / "results" / "multidataset_matrix"
GENERATED_DIR = PROJECT_ROOT / "skills" / "subagents" / "generated"
GENERATED_LOCK_PATH = PROJECT_ROOT / ".cache_hf_runtime" / "generated_skills.lock"
HF_CACHE_DIR = PROJECT_ROOT / ".cache_hf_runtime" / "datasets"
IMAGE_CACHE_DIR = PROJECT_ROOT / ".cache_hf_runtime" / "materialized_images"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    hf_id: str
    config: str
    split: str
    eval_count: int
    seed_count: int
    seed_split: Optional[str] = None
    eval_take_first: Optional[int] = None
    version_filter: Optional[str] = None


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "m3cot": DatasetSpec(
        key="m3cot",
        title="M3CoT test first 1200, validation first 20 seed",
        hf_id="LightChen2333/M3CoT",
        config="default",
        split="test",
        seed_split="validation",
        seed_count=20,
        eval_count=1200,
    ),
    "mathverse_text_dominant": DatasetSpec(
        key="mathverse_text_dominant",
        title="MathVerse testmini Text Dominant first 220",
        hf_id="AI4Math/MathVerse",
        config="testmini",
        split="testmini",
        seed_count=20,
        eval_count=200,
        eval_take_first=220,
        version_filter="Text Dominant",
    ),
    "mathverse_text_lite": DatasetSpec(
        key="mathverse_text_lite",
        title="MathVerse testmini Text Lite first 220",
        hf_id="AI4Math/MathVerse",
        config="testmini",
        split="testmini",
        seed_count=20,
        eval_count=200,
        eval_take_first=220,
        version_filter="Text Lite",
    ),
    "mathverse_vision_intensive": DatasetSpec(
        key="mathverse_vision_intensive",
        title="MathVerse testmini Vision Intensive first 220",
        hf_id="AI4Math/MathVerse",
        config="testmini",
        split="testmini",
        seed_count=20,
        eval_count=200,
        eval_take_first=220,
        version_filter="Vision Intensive",
    ),
    "mathverse_vision_dominant": DatasetSpec(
        key="mathverse_vision_dominant",
        title="MathVerse testmini Vision Dominant first 220",
        hf_id="AI4Math/MathVerse",
        config="testmini",
        split="testmini",
        seed_count=20,
        eval_count=200,
        eval_take_first=220,
        version_filter="Vision Dominant",
    ),
    "mathverse_vision_only": DatasetSpec(
        key="mathverse_vision_only",
        title="MathVerse testmini Vision Only first 220",
        hf_id="AI4Math/MathVerse",
        config="testmini",
        split="testmini",
        seed_count=20,
        eval_count=200,
        eval_take_first=220,
        version_filter="Vision Only",
    ),
    "mmmu_pro_standard_10": DatasetSpec(
        key="mmmu_pro_standard_10",
        title="MMMU-Pro standard (10 options) first 1200",
        hf_id="MMMU/MMMU_Pro",
        config="standard (10 options)",
        split="test",
        seed_count=20,
        eval_count=1180,
        eval_take_first=1200,
    ),
}


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "sample")).strip("_")
    return text[:160] or "sample"


def _parse_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [text]
    return [str(value)]


def _save_pil_image(image: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL image, got {type(image).__name__}")
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    image.save(path)
    return str(path.resolve())


def _materialize_single_image(row: Dict[str, Any], dataset_key: str, sample_id: str) -> str:
    image = row.get("image")
    if image is None:
        return ""
    if isinstance(image, str):
        candidate = Path(image)
        return str(candidate.resolve()) if candidate.exists() else ""
    if isinstance(image, Image.Image):
        return _save_pil_image(image, IMAGE_CACHE_DIR / dataset_key / f"{_safe_name(sample_id)}.png")
    if isinstance(image, dict) and image.get("bytes"):
        out = IMAGE_CACHE_DIR / dataset_key / f"{_safe_name(sample_id)}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(image["bytes"])
        return str(out.resolve())
    return ""


def _materialize_mmmu_images(row: Dict[str, Any], dataset_key: str, sample_id: str) -> List[str]:
    paths: List[str] = []
    for idx in range(1, 8):
        image = row.get(f"image_{idx}")
        if isinstance(image, Image.Image):
            paths.append(_save_pil_image(image, IMAGE_CACHE_DIR / dataset_key / f"{_safe_name(sample_id)}_image_{idx}.png"))
    return paths


def _m3cot_task(row: Dict[str, Any], spec: DatasetSpec) -> TaskPacket:
    sample_id = f"{spec.key}_{row.get('id')}"
    choices = _parse_list(row.get("choices"))
    context = str(row.get("context") or "").strip()
    question = str(row.get("question") or "").strip()
    if context:
        question = f"Context: {context}\nQuestion: {question}"
    return TaskPacket(
        sample_id=sample_id,
        question=question,
        image_path=_materialize_single_image(row, spec.key, sample_id),
        choices=choices,
        question_type="multi_choice" if choices else "free_form",
        answer_type="text",
        metadata={
            "dataset": "M3CoT",
            "category": row.get("category"),
            "domain": row.get("domain"),
            "topic": row.get("topic"),
            "split": row.get("split"),
            "image_id": row.get("image_id"),
        },
        answer=row.get("answer"),
    )


def _mathverse_task(row: Dict[str, Any], spec: DatasetSpec) -> TaskPacket:
    sample_id = f"{spec.key}_{row.get('sample_index')}"
    question = str(row.get("question") or row.get("question_for_eval") or "").strip()
    query = str(row.get("query_cot") or row.get("query_wo") or "").strip() or None
    return TaskPacket(
        sample_id=sample_id,
        question=question,
        image_path=_materialize_single_image(row, spec.key, sample_id),
        choices=[],
        question_type="multi_choice",
        answer_type="text",
        metadata={
            "dataset": "MathVerse",
            "problem_index": row.get("problem_index"),
            "problem_version": row.get("problem_version"),
            "metadata": row.get("metadata"),
        },
        answer=row.get("answer"),
        query=query,
    )


def _mmmu_task(row: Dict[str, Any], spec: DatasetSpec) -> TaskPacket:
    sample_id = f"{spec.key}_{row.get('id')}"
    options = _parse_list(row.get("options"))
    image_paths = _materialize_mmmu_images(row, spec.key, sample_id)
    return TaskPacket(
        sample_id=sample_id,
        question=str(row.get("question") or "").strip(),
        image_path=image_paths[0] if image_paths else "",
        image_paths=image_paths,
        choices=options,
        question_type="multi_choice",
        answer_type="text",
        metadata={
            "dataset": "MMMU-Pro",
            "config": spec.config,
            "img_type": row.get("img_type"),
            "topic_difficulty": row.get("topic_difficulty"),
            "subject": row.get("subject"),
        },
        answer=row.get("answer"),
    )


def _row_to_task(row: Dict[str, Any], spec: DatasetSpec) -> TaskPacket:
    if spec.key == "m3cot":
        return _m3cot_task(row, spec)
    if spec.key.startswith("mathverse_"):
        return _mathverse_task(row, spec)
    if spec.key == "mmmu_pro_standard_10":
        return _mmmu_task(row, spec)
    raise ValueError(f"Unsupported dataset spec: {spec.key}")


def _load_hf_rows(spec: DatasetSpec, split: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(spec.hf_id, spec.config, split=split, cache_dir=str(HF_CACHE_DIR))
    rows: List[Dict[str, Any]] = [dict(row) for row in dataset]
    if spec.version_filter:
        rows = [row for row in rows if str(row.get("problem_version")) == spec.version_filter]
    return rows


def load_dataset_tasks(spec: DatasetSpec, *, max_seed_count: Optional[int] = None, max_eval_count: Optional[int] = None) -> Tuple[List[TaskPacket], List[TaskPacket]]:
    if spec.seed_split:
        seed_rows = _load_hf_rows(spec, spec.seed_split)[: spec.seed_count]
        eval_rows = _load_hf_rows(spec, spec.split)[: spec.eval_count]
    else:
        rows = _load_hf_rows(spec, spec.split)
        if spec.eval_take_first is not None:
            rows = rows[: spec.eval_take_first]
        seed_rows = rows[: spec.seed_count]
        eval_rows = rows[spec.seed_count : spec.seed_count + spec.eval_count]

    if max_seed_count is not None:
        seed_rows = seed_rows[:max_seed_count]
    if max_eval_count is not None:
        eval_rows = eval_rows[:max_eval_count]

    seed_tasks = [_row_to_task(row, spec) for row in seed_rows]
    eval_tasks = [_row_to_task(row, spec) for row in eval_rows]
    if not eval_tasks:
        raise SystemExit(f"No evaluation tasks selected for {spec.key}.")
    return seed_tasks, eval_tasks


def _row_from_trace(task: TaskPacket, trace: Any) -> Dict[str, Any]:
    return {
        "pid": task.sample_id,
        "question": task.question,
        "prediction": trace.final_answer_normalized,
        "gold": task.answer,
        "correct": (
            answers_equal(task, trace.final_answer_normalized, task.answer)
            if task.answer is not None and trace.error is None
            else (None if task.answer is None else None)
        ),
        "workspace": trace.workspace,
        "used_generated_skill": trace.used_generated_skill,
        "saved_generated_skill": getattr(trace, "saved_generated_skill", None),
        "error": trace.error,
    }


def _baseline_row_from_trace(task: TaskPacket, trace: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pid": task.sample_id,
        "question": task.question,
        "prediction": trace.get("final_answer_normalized"),
        "gold": task.answer,
        "correct": (
            answers_equal(task, trace.get("final_answer_normalized"), task.answer)
            if task.answer is not None and trace.get("error") is None
            else (None if task.answer is None else None)
        ),
        "workspace": trace.get("workspace"),
        "used_generated_skill": None,
        "saved_generated_skill": None,
        "error": trace.get("error"),
    }


def _worker_run_baseline(task_dict: Dict[str, Any], tag: str) -> Dict[str, Any]:
    _load_env(REPO_ROOT)
    task = TaskPacket.from_dict(task_dict)
    trace = run_baseline_model(task, PROJECT_ROOT, experiment_tag=tag)
    return _baseline_row_from_trace(task, trace)


def _worker_run_agent(task_dict: Dict[str, Any], reuse: bool, save: bool, tag: str, suppress_reuse_registry: bool = False) -> Dict[str, Any]:
    _load_env(REPO_ROOT)
    task = TaskPacket.from_dict(task_dict)
    if suppress_reuse_registry:
        import muse.orchestrator as orch

        orch.record_skill_outcome = lambda *args, **kwargs: None
    agent = MultimodalMetaAgent(
        allow_generated_skill_reuse=reuse,
        allow_save_generated_skills=save,
        experiment_tag=tag,
    )
    trace = agent.solve(task)
    return _row_from_trace(task, trace)


def _safe_remove_generated_contents() -> None:
    root = Path(os.getenv("MUSE_GENERATED_SKILLS_ROOT") or GENERATED_DIR)
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.name.startswith("_"):
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _copy_generated(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)


def _restore_generated(src: Path) -> None:
    root = Path(os.getenv("MUSE_GENERATED_SKILLS_ROOT") or GENERATED_DIR)
    _safe_remove_generated_contents()
    if src.exists():
        for child in src.iterdir():
            target = root / child.name
            if child.is_dir():
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)


@contextmanager
def _generated_library_lock():
    GENERATED_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GENERATED_LOCK_PATH.open("a+", encoding="utf-8") as fh:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _run_process_incremental(
    name: str,
    path: Path,
    tasks: Sequence[TaskPacket],
    *,
    workers: int,
    mode: str,
) -> List[Dict[str, Any]]:
    existing = _read_jsonl(path)
    done = {str(r.get("pid")) for r in existing}
    rows_by_pid = {str(r.get("pid")): r for r in existing if r.get("pid") is not None}
    pending = [task for task in tasks if str(task.sample_id) not in done]
    if pending:
        with cf.ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            future_to_task = {}
            for task in pending:
                td = task.to_dict()
                if mode == "baseline":
                    fut = ex.submit(_worker_run_baseline, td, name)
                elif mode == "evolution":
                    fut = ex.submit(_worker_run_agent, td, True, False, name, True)
                else:
                    raise ValueError(mode)
                future_to_task[fut] = task
            for fut in tqdm(cf.as_completed(future_to_task), total=len(future_to_task), desc=name):
                task = future_to_task[fut]
                row = fut.result()
                rows_by_pid[str(task.sample_id)] = row
                _append_jsonl(path, row)
    ordered = [rows_by_pid[str(task.sample_id)] for task in tasks if str(task.sample_id) in rows_by_pid]
    save_jsonl(path, ordered)
    return ordered


def _run_seed_install(seed_tasks: Sequence[TaskPacket], out_dir: Path) -> List[Dict[str, Any]]:
    path = out_dir / "seed_install.jsonl"
    existing = _read_jsonl(path)
    rows_by_pid = {str(r.get("pid")): r for r in existing if r.get("pid") is not None}
    pending = [task for task in seed_tasks if str(task.sample_id) not in rows_by_pid]
    if pending:
        agent = MultimodalMetaAgent(allow_generated_skill_reuse=True, allow_save_generated_skills=True, experiment_tag="seed_install")
        for task in tqdm(pending, desc="seed_install"):
            trace = agent.solve(task)
            row = _row_from_trace(task, trace)
            rows_by_pid[str(task.sample_id)] = row
            _append_jsonl(path, row)
    ordered = [rows_by_pid[str(task.sample_id)] for task in seed_tasks if str(task.sample_id) in rows_by_pid]
    save_jsonl(path, ordered)
    return ordered


def _row_looks_unusable(row: Dict[str, Any]) -> bool:
    pred = row.get("prediction")
    if row.get("error") is not None:
        return True
    if pred in (None, "", [], {}):
        return True
    text = str(pred).strip().lower()
    return any(marker in text for marker in ["cannot determine", "unable to determine", "insufficient evidence", "need image", "unknown", "not sure"])


def _merge_evolution_with_base_floor(evolved_rows: List[Dict[str, Any]], base_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_by_pid = {str(row.get("pid")): row for row in base_rows}
    merged = []
    for row in evolved_rows:
        base = base_by_pid.get(str(row.get("pid")))
        if base is not None and _row_looks_unusable(row):
            out = dict(base)
            out["fallback_from_base"] = True
            out["evolution_workspace_raw"] = row.get("workspace")
            out["evolution_error_raw"] = row.get("error")
            out["evolution_prediction_raw"] = row.get("prediction")
            merged.append(out)
        else:
            out = dict(row)
            out["fallback_from_base"] = False
            merged.append(out)
    return merged


def _write_summary(out_dir: Path, config: Dict[str, Any], branch_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"config": config}
    for name, rows in branch_rows.items():
        summary[name] = _summarize(rows)
    base_acc = summary.get("base", {}).get("accuracy")
    for name, item in list(summary.items()):
        if name == "config" or name == "base":
            continue
        acc = item.get("accuracy") if isinstance(item, dict) else None
        summary[f"delta_accuracy_{name}_minus_base"] = None if base_acc is None or acc is None else acc - base_acc
    save_json(out_dir / "summary.json", summary)
    return summary


def _parse_variants(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return ["base", "rollout", "reflection", "memory", "evolution"]
    aliases = {
        "base": "base",
        "base+rollouts3": "rollout",
        "base+rollout3": "rollout",
        "rollout": "rollout",
        "rollouts": "rollout",
        "reflection": "reflection",
        "base+reflection": "reflection",
        "memory": "memory",
        "memory3": "memory",
        "base+memory3": "memory",
        "evolution": "evolution",
        "base+evolution": "evolution",
    }
    variants = []
    for part in raw.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise SystemExit(f"Unknown variant: {part}")
        value = aliases[key]
        if value not in variants:
            variants.append(value)
    return variants


def run_dataset(
    spec: DatasetSpec,
    *,
    output_root: Path,
    model_alias: Optional[str],
    variants: Sequence[str],
    workers: int,
    evolution_workers: int,
    max_seed_count: Optional[int],
    max_eval_count: Optional[int],
    rollouts: int,
) -> Dict[str, Any]:
    seed_tasks, eval_tasks = load_dataset_tasks(spec, max_seed_count=max_seed_count, max_eval_count=max_eval_count)
    suffix = f"__{model_alias}" if model_alias else ""
    out_dir = output_root / f"{spec.key}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    isolated_generated_root = out_dir / "generated_skills"
    os.environ["MUSE_GENERATED_SKILLS_ROOT"] = str(isolated_generated_root.resolve())

    manifest = {
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset_key": spec.key,
        "dataset_title": spec.title,
        "hf_id": spec.hf_id,
        "config": spec.config,
        "split": spec.split,
        "seed_split": spec.seed_split,
        "version_filter": spec.version_filter,
        "seed_count": len(seed_tasks),
        "eval_count": len(eval_tasks),
        "workers": workers,
        "evolution_workers": evolution_workers,
        "rollouts": rollouts,
        "variants": list(variants),
        "model_alias": model_alias,
        "generated_skills_root": str(isolated_generated_root.resolve()),
        "env": {k: os.getenv(k) for k in ["BASE_URL", "MODEL", "DEFAULT_MODEL", "VISION_MODEL", "ORCHESTRATOR_MODEL", "MODEL_PROTOCOL"]},
    }
    save_json(out_dir / "manifest.json", manifest)
    save_jsonl(out_dir / "seed_tasks.jsonl", [task.to_dict() for task in seed_tasks])
    save_jsonl(out_dir / "eval_tasks.jsonl", [task.to_dict() for task in eval_tasks])

    branch_rows: Dict[str, List[Dict[str, Any]]] = {}
    if "base" in variants or "evolution" in variants:
        rows = _run_process_incremental(
            "base",
            out_dir / "base.jsonl",
            eval_tasks,
            workers=evolution_workers,
            mode="baseline",
        )
        branch_rows["base"] = rows
        _write_summary(out_dir, manifest, branch_rows)

    if "rollout" in variants:
        rows = _run_incremental(
            f"base_rollout{rollouts}",
            out_dir / f"base_rollout{rollouts}.jsonl",
            eval_tasks,
            lambda task: run_rollout5(task, out_dir=out_dir, rollouts=rollouts, temperature=0.7),
            workers=workers,
        )
        branch_rows[f"base_rollout{rollouts}"] = rows
        _write_summary(out_dir, manifest, branch_rows)

    if "reflection" in variants:
        rows = _run_incremental(
            "base_reflection1",
            out_dir / "base_reflection1.jsonl",
            eval_tasks,
            lambda task: run_reflection1(task, out_dir=out_dir),
            workers=workers,
        )
        branch_rows["base_reflection1"] = rows
        _write_summary(out_dir, manifest, branch_rows)

    if "memory" in variants:
        memory = build_memory(seed_tasks, out_dir=out_dir)
        rows = _run_incremental(
            "base_memory_top3",
            out_dir / "base_memory_top3.jsonl",
            eval_tasks,
            lambda task: run_memory_topk(task, out_dir=out_dir, memory=memory, top_k=3),
            workers=workers,
        )
        branch_rows["base_memory_top3"] = rows
        _write_summary(out_dir, manifest, branch_rows)

    if "evolution" in variants:
        seeded_library = out_dir / "generated_after_seed"
        if not seeded_library.exists():
            _safe_remove_generated_contents()
            seed_rows = _run_seed_install(seed_tasks, out_dir)
            branch_rows["seed_install"] = seed_rows
            _copy_generated(isolated_generated_root, seeded_library)
        else:
            seed_rows = _read_jsonl(out_dir / "seed_install.jsonl")
            branch_rows["seed_install"] = seed_rows
            _restore_generated(seeded_library)
        evolved_raw = _run_process_incremental(
            "base_evolution_raw",
            out_dir / "base_evolution_raw.jsonl",
            eval_tasks,
            workers=evolution_workers,
            mode="evolution",
        )
        base_rows = branch_rows.get("base") or _read_jsonl(out_dir / "base.jsonl")
        evolved = _merge_evolution_with_base_floor(evolved_raw, base_rows)
        save_jsonl(out_dir / "base_evolution.jsonl", evolved)
        branch_rows["base_evolution"] = evolved
        _write_summary(out_dir, manifest, branch_rows)

    return _write_summary(out_dir, manifest, branch_rows)


def _parse_datasets(raw: str) -> List[DatasetSpec]:
    if raw.strip().lower() == "all":
        return [DATASET_SPECS[key] for key in DATASET_SPECS]
    specs = []
    for part in raw.split(","):
        key = part.strip()
        if not key:
            continue
        if key not in DATASET_SPECS:
            raise SystemExit(f"Unknown dataset key: {key}. Available: {', '.join(DATASET_SPECS)}")
        specs.append(DATASET_SPECS[key])
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run base/rollout/reflection/memory/evolution on M3CoT, MathVerse, and MMMU-Pro.")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset keys, or all.")
    parser.add_argument("--variants", default="all", help="Comma-separated variants, or all.")
    parser.add_argument("--workers", type=int, default=20, help="Thread workers for rollout/reflection/memory.")
    parser.add_argument("--evolution-workers", type=int, default=20, help="Process workers for base/evolution branches.")
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--model-alias", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-seed-count", type=int, default=None, help="Smoke-test override.")
    parser.add_argument("--max-eval-count", type=int, default=None, help="Smoke-test override.")
    parser.add_argument("--dry-run-load", action="store_true", help="Only load and convert tasks; do not call models.")
    args = parser.parse_args()

    variants = _parse_variants(args.variants)
    specs = _parse_datasets(args.datasets)
    output_root = Path(args.output_root).resolve() if args.output_root else RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Any] = {}
    for spec in specs:
        print(f"[multidataset] running {spec.key}: {spec.title}")
        if args.dry_run_load:
            seed_tasks, eval_tasks = load_dataset_tasks(spec, max_seed_count=args.max_seed_count, max_eval_count=args.max_eval_count)
            sample = eval_tasks[0].to_dict() if eval_tasks else None
            summaries[spec.key] = {
                "title": spec.title,
                "seed_count": len(seed_tasks),
                "eval_count": len(eval_tasks),
                "first_eval_task": sample,
            }
            save_json(output_root / "dry_run_load_summary.json", summaries)
            continue
        summaries[spec.key] = run_dataset(
            spec,
            output_root=output_root,
            model_alias=args.model_alias,
            variants=variants,
            workers=args.workers,
            evolution_workers=args.evolution_workers,
            max_seed_count=args.max_seed_count,
            max_eval_count=args.max_eval_count,
            rollouts=args.rollouts,
        )
        save_json(output_root / "summary_all.json", summaries)

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved multidataset matrix outputs to: {output_root}")


if __name__ == "__main__":
    main()
