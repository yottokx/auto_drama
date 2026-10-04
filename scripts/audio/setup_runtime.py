"""Prepare the isolated audio runtime from a GUI without requiring PowerShell."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

try:
    from .engine import write_json
except ImportError:
    from engine import write_json

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "services/worker/runtimes/stable_audio3"
PYTHON_VERSION = "3.12.13"


def find_uv() -> str:
    """Explorer-launched GUIs may have an older PATH than interactive shells."""
    located = shutil.which("uv")
    if located:
        return located
    candidates = [
        Path.home() / ".local/bin/uv.exe",
        Path(sys.executable).with_name("uv.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "uv が見つかりません。uv のインストール先をPATHに追加してGUIを再起動してください。"
    )


def runtime_environment(runtime: Path, inherited: Mapping[str, str]) -> dict[str, str]:
    environment = dict(inherited)
    environment.update({
        "UV_CACHE_DIR": str(ROOT / ".cache/uv"),
        "UV_PYTHON_INSTALL_DIR": str(runtime / ".python"),
        "UV_PROJECT_ENVIRONMENT": str(runtime / ".venv"),
        "UV_PYTHON_PREFERENCE": "only-managed",
        "PYTHONUTF8": "1",
    })
    return environment


def setup_runtime(runtime: Path = RUNTIME, *, status_dir: Path | None = None) -> None:
    uv = find_uv()
    environment = runtime_environment(runtime, os.environ)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def run(command: list[str]) -> None:
        # Lists preserve paths containing spaces; no shell or PATH edits needed.
        subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                       creationflags=flags, check=True)

    def notify(message: str) -> None:
        print(message, flush=True)
        if status_dir:
            write_json(status_dir / "status.json", {"phase": "preparing", "message": message})

    notify("音楽生成用Pythonを準備しています。")
    run([uv, "python", "install", PYTHON_VERSION, "--no-bin", "--no-registry"])
    notify("Stable Audio / ACE-Step の依存パッケージを準備しています。")
    command = [uv, "sync", "--python", PYTHON_VERSION, "--project", str(runtime)]
    if (runtime / "uv.lock").is_file():
        command.append("--locked")
    run(command)
    notify("StableAudio3Pipeline・AceStepPipeline・CUDAを確認しています。")
    run([str(runtime / ".venv/Scripts/python.exe"), "-c",
         ('from diffusers import StableAudio3Pipeline, AceStepPipeline; import torch, soundfile; '
          'print("Stable Audio 3 / ACE-Step 1.5 ready / CUDA:", torch.cuda.is_available())')])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stable Audio 3 / ACE-Step 1.5 専用実行環境の準備")
    parser.add_argument("--status-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        setup_runtime(status_dir=args.status_dir)
    except (OSError, subprocess.CalledProcessError) as exc:
        error = f"Stable Audioの環境準備に失敗しました: {exc}"
        print(error, file=sys.stderr, flush=True)
        if args.status_dir:
            write_json(args.status_dir / "status.json", {"phase": "error", "message": error})
            write_json(args.status_dir / "result.json", {"ok": False, "error": error})
        return 1
    if args.status_dir:
        write_json(args.status_dir / "status.json", {"phase": "done", "message": "実行環境の準備が完了しました。"})
        write_json(args.status_dir / "result.json", {"ok": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
