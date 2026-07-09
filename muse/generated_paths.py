from __future__ import annotations

import os
from pathlib import Path


def resolved_generated_skills_root(project_root: str | Path) -> Path:
    project_root = Path(project_root)
    configured = os.getenv("MUSE_GENERATED_SKILLS_ROOT")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = project_root / path
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = project_root / "skills" / "subagents" / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path
