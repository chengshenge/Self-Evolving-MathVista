from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .generated_paths import resolved_generated_skills_root

import os

_ENV_ALIAS_PREFIXES = ("REASONING", "ORCHESTRATOR", "VERIFIER", "BASELINE", "VISUAL", "MATH")

def _af_strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v

def _af_load_simple_env_file(path: Path, *, override: bool = False) -> bool:
    if not path.exists():
        return False
    loaded = False
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _af_strip_quotes(value)
        if key and (override or key not in os.environ):
            os.environ[key] = value
            loaded = True
    return loaded

def _af_fanout_base_env_aliases() -> None:
    base_url = os.getenv("BASE_URL")
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL")

    if api_key and not os.getenv("API_KEY"):
        os.environ["API_KEY"] = api_key
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key

    for prefix in _ENV_ALIAS_PREFIXES:
        if base_url and not os.getenv(f"{prefix}_BASE_URL"):
            os.environ[f"{prefix}_BASE_URL"] = base_url
        if api_key and not os.getenv(f"{prefix}_API_KEY"):
            os.environ[f"{prefix}_API_KEY"] = api_key
        if model and not os.getenv(f"{prefix}_MODEL"):
            os.environ[f"{prefix}_MODEL"] = model

def _af_ensure_repo_env_loaded(project_root) -> None:
    candidates = []
    explicit_env = os.getenv("MUSE_ENV_FILE")
    if explicit_env:
        candidates.append(Path(explicit_env))
    candidates.extend([
        Path(project_root) / ".env",
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ])
    chosen = None
    for p in candidates:
        if p.exists():
            _af_load_simple_env_file(p, override=bool(explicit_env) and p == Path(explicit_env))
            chosen = p
            break
    _af_fanout_base_env_aliases()

    if os.getenv("RUNTIME_CONFIG_DEBUG") == "1":
        def flag(k: str) -> str:
            return "SET" if os.getenv(k) else "MISSING"
        print(
            "[runtime-config-debug]",
            {
                "dotenv": str(chosen) if chosen else None,
                "MUSE_ENV_FILE": flag("MUSE_ENV_FILE"),
                "BASE_URL": flag("BASE_URL"),
                "MODEL": flag("MODEL"),
                "API_KEY": flag("API_KEY"),
                "OPENAI_API_KEY": flag("OPENAI_API_KEY"),
                "REASONING_BASE_URL": flag("REASONING_BASE_URL"),
                "REASONING_MODEL": flag("REASONING_MODEL"),
                "REASONING_API_KEY": flag("REASONING_API_KEY"),
                "ORCHESTRATOR_BASE_URL": flag("ORCHESTRATOR_BASE_URL"),
                "ORCHESTRATOR_MODEL": flag("ORCHESTRATOR_MODEL"),
                "ORCHESTRATOR_API_KEY": flag("ORCHESTRATOR_API_KEY"),
                "VERIFIER_BASE_URL": flag("VERIFIER_BASE_URL"),
                "VERIFIER_MODEL": flag("VERIFIER_MODEL"),
                "VERIFIER_API_KEY": flag("VERIFIER_API_KEY"),
            }
        )


load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    protocol: str = "OPENAI_STYLE"
    timeout: int = 120
    temperature: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


@dataclass(frozen=True)
class RuntimeConfig:
    mock_mode: bool
    save_generated_skills: bool
    max_rechecks: int
    workspace_root: Path
    trajectory_root: Path
    generated_skills_root: Path
    vision_model: ModelConfig
    reasoning_model: ModelConfig
    orchestrator_model: Optional[ModelConfig] = None


def _read_model(prefix: str, fallback_prefix: Optional[str] = None) -> ModelConfig:
    fallback_prefix = fallback_prefix or prefix
    return ModelConfig(
        base_url=os.getenv(f"{prefix}_BASE_URL", os.getenv(f"{fallback_prefix}_BASE_URL", "")),
        api_key=os.getenv(f"{prefix}_API_KEY", os.getenv(f"{fallback_prefix}_API_KEY", "")),
        model=os.getenv(f"{prefix}_MODEL", os.getenv(f"{fallback_prefix}_MODEL", "")),
        protocol=os.getenv(f"{prefix}_PROTOCOL", os.getenv(f"{fallback_prefix}_PROTOCOL", os.getenv("MODEL_PROTOCOL", "OPENAI_STYLE"))),
        timeout=int(os.getenv(f"{prefix}_TIMEOUT", os.getenv(f"{fallback_prefix}_TIMEOUT", "120"))),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", os.getenv(f"{fallback_prefix}_TEMPERATURE", "0.0"))),
    )


def load_runtime_config(project_root: Optional[Path] = None) -> RuntimeConfig:
    if project_root is not None:
        project_root = Path(project_root)
    _af_ensure_repo_env_loaded(project_root)
    project_root = project_root or Path(__file__).resolve().parent.parent
    workspace_root = project_root / "workspace"
    trajectory_root = project_root / "trajectory"
    generated_skills_root = resolved_generated_skills_root(project_root)

    for path in (workspace_root, trajectory_root, generated_skills_root):
        path.mkdir(parents=True, exist_ok=True)

    vision_model = _read_model("VISION", fallback_prefix="DEFAULT")
    reasoning_model = _read_model("REASONING", fallback_prefix="DEFAULT")
    orchestrator_model = _read_model("ORCHESTRATOR", fallback_prefix="REASONING")

    return RuntimeConfig(
        mock_mode=os.getenv("MOCK_MODE", "0").strip() == "1",
        save_generated_skills=os.getenv("SAVE_GENERATED_SKILLS", "1").strip() != "0",
        max_rechecks=int(os.getenv("MAX_RECHECKS", "1")),
        workspace_root=workspace_root,
        trajectory_root=trajectory_root,
        generated_skills_root=generated_skills_root,
        vision_model=vision_model,
        reasoning_model=reasoning_model,
        orchestrator_model=orchestrator_model,
    )
