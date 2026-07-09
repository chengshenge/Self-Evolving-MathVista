from __future__ import annotations

import importlib.util
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _infer_project_root(path: Path) -> Path:
    current = path.resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "muse").exists():
            return parent
    return current


def run_python_file(path: str | Path, query: str, *, timeout: int = 300, work_dir: str | Path | None = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"success": False, "error": f"File not found: {path}"}

    module_name = path.stem + "_dynamic"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        return {"success": False, "error": f"Could not load module: {path}"}

    module = importlib.util.module_from_spec(spec)
    project_root = _infer_project_root(path)

    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))

    try:
        if work_dir:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            os.chdir(work_dir)
        else:
            os.chdir(path.parent)

        spec.loader.exec_module(module)

        if not hasattr(module, "main"):
            return {"success": False, "error": f"{path} has no main(query) function"}

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout}s")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        try:
            result = module.main(query)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        if not isinstance(result, dict):
            return {"success": False, "error": f"main(query) must return dict, got {type(result).__name__}"}
        return {"success": True, **result}
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
