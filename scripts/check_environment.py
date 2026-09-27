"""Check installed development tools without downloading or loading models."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_python(component: Path, modules: list[str]) -> dict:
    python = component / ".venv/Scripts/python.exe"
    code = (
        "import importlib,json,sys; "
        f"[importlib.import_module(m) for m in {modules!r}]; "
        "print(json.dumps({'python':sys.version.split()[0],"
        "'executable':sys.executable,'base_prefix':sys.base_prefix}))"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", code], cwd=component, text=True,
        capture_output=True, check=True, timeout=120,
    )
    info = json.loads(result.stdout.splitlines()[-1])
    if not Path(info["base_prefix"]).resolve().is_relative_to(component.resolve()):
        raise RuntimeError(f"Python is outside its component: {info['base_prefix']}")
    return info


def main() -> int:
    checks: dict[str, dict] = {}
    components = {
        "coordinator": (ROOT, ["fastapi", "uvicorn", "sqlalchemy", "alembic", "sqlite3"]),
        "worker": (ROOT / "services/worker", ["httpx", "pydantic_settings", "psutil"]),
        "diffusers": (ROOT / "services/worker/runtimes/diffusers", ["torch", "diffusers"]),
        "irodori": (ROOT / "services/worker/runtimes/irodori", ["torch", "soundfile"]),
    }
    for name, (component, modules) in components.items():
        try:
            checks[name] = {"ok": True, **run_python(component, modules)}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            checks[name] = {"ok": False, "error": str(exc)}

    for name, path in {
        "web_build": ROOT / "apps/web/dist/index.html",
        "llama_server": ROOT / "services/worker/runtimes/llama_cpp/bin/llama-server.exe",
        "tyrano": ROOT / "tyranoscript/index.html",
    }.items():
        checks[name] = {"ok": path.is_file(), "path": str(path)}

    for name, path in {
        "irodori_git": ROOT / "services/worker/runtimes/irodori/source",
        "diffusers_git": ROOT / "services/worker/runtimes/diffusers/source",
        "llama_cpp_git": ROOT / "services/worker/runtimes/llama_cpp/source",
        "tyrano_git": ROOT / "tyranoscript",
    }.items():
        try:
            if not (path / ".git").exists():
                raise RuntimeError("Repository was not cloned")
            result = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                text=True, capture_output=True, check=True, timeout=10,
            )
            checks[name] = {"ok": True, "commit": result.stdout.strip()}
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            checks[name] = {"ok": False, "error": str(exc)}

    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if all(item["ok"] for item in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())


