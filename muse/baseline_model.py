from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .answering import answers_equal, normalize_answer
from .config import load_runtime_config
from .io_utils import save_json
from .llm_clients import OpenAIStyleClient
from .schemas import TaskPacket

SYSTEM_PROMPT = """You are the raw baseline multimodal model branch.
Solve the task directly from the original image and question.
Do not simulate or call subagents. Do not mention visual_detail_agent, math_reason_agent, verifier, or any pipeline.
Return JSON only with keys:
- reasoning_steps: a list of short natural-language steps (3-6 items)
- answer: the final answer
- confidence: a number between 0 and 1
For multi-choice questions, answer may be the option text or option letter.
For numeric questions, answer should be the final scalar only if possible.
"""


def _collapse_steps(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        if value and all(isinstance(x, str) for x in value):
            stripped = [x for x in value if x != ""]
            if stripped and all(len(x) <= 1 for x in stripped):
                joined = "".join(value).strip()
                return [joined] if joined else []
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return [str(value).strip()]


def _build_user_prompt(task: TaskPacket) -> str:
    parts = [
        f"Question: {task.question}",
        f"Question type: {task.question_type}",
        f"Answer type: {task.answer_type}",
    ]
    if task.choices:
        choice_lines = "\n".join(f"({chr(65+i)}) {choice}" for i, choice in enumerate(task.choices))
        parts.append("Choices:\n" + choice_lines)
    if task.precision is not None:
        parts.append(f"Requested precision: {task.precision}")
    if task.unit not in (None, "", "none", "None"):
        parts.append(f"Requested unit: {task.unit}")
    if task.query:
        parts.append(f"Original query hint: {task.query}")
    parts.append("Return JSON only.")
    return "\n\n".join(parts)


def _pick_model_config(config):
    if config.orchestrator_model and config.orchestrator_model.enabled:
        return config.orchestrator_model
    return config.reasoning_model


def run_baseline_model(task: TaskPacket, project_root: str | Path, experiment_tag: Optional[str] = None) -> Dict[str, Any]:
    project_root = Path(project_root)
    config = load_runtime_config(project_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = config.workspace_root / f"{timestamp}_{task.sample_id}_baseline_model"
    workspace.mkdir(parents=True, exist_ok=True)
    save_json(workspace / "task.json", task.to_dict())

    trace: Dict[str, Any] = {
        "branch_type": "baseline_model",
        "task": task.to_dict(),
        "experiment_tag": experiment_tag,
        "workspace": str(workspace),
        "used_generated_skill": None,
        "saved_generated_skill": None,
        "reasoning_steps": [],
        "model_name": None,
        "confidence": None,
        "final_answer_raw": None,
        "final_answer_normalized": None,
        "correct": None,
        "error": None,
    }

    try:
        model_cfg = _pick_model_config(config)
        trace["model_name"] = model_cfg.model

        if config.mock_mode:
            raw_answer = task.answer
            reasoning_steps = ["MOCK_MODE is enabled.", "Returning the gold answer as the baseline prediction."]
            confidence = 1.0
        else:
            client = OpenAIStyleClient(model_cfg)
            prompt = _build_user_prompt(task)
            image_paths = task.existing_image_paths()
            if image_paths:
                response = client.complete_multimodal_json(SYSTEM_PROMPT, prompt, image_paths, max_tokens=1600)
            else:
                response = client.complete_json(SYSTEM_PROMPT, prompt, max_tokens=1200)
            reasoning_steps = _collapse_steps(response.get("reasoning_steps"))
            raw_answer = response.get("answer")
            confidence = response.get("confidence")

        normalized = normalize_answer(task, raw_answer)
        correct = answers_equal(task, normalized, task.answer) if task.answer is not None else None

        trace.update(
            {
                "reasoning_steps": reasoning_steps,
                "confidence": confidence,
                "final_answer_raw": raw_answer,
                "final_answer_normalized": normalized,
                "correct": correct,
            }
        )
        save_json(workspace / "trace.json", trace)
        return trace
    except Exception as exc:
        trace["error"] = f"{type(exc).__name__}: {exc}"
        save_json(workspace / "trace.json", trace)
        return trace
