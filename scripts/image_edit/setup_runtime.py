"""Prepare only the dedicated Qwen Image 2.1 runtime from the experiment UI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.audio.setup_runtime import find_uv, runtime_environment
from scripts.image_edit.common import ROOT, RUNTIME, write_json

PYTHON_VERSION = "3.12.13"


def setup_runtime(runtime: Path = RUNTIME, *, status_dir: Path | None = None) -> None:
    uv = find_uv()
    environment = runtime_environment(runtime, os.environ)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def notify(message: str) -> None:
        print(message, flush=True)
        if status_dir:
            write_json(status_dir / "status.json", {"phase": "preparing", "message": message})

    def run(command: list[str]) -> None:
        subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                       creationflags=flags, check=True)

    notify("Qwen Image 2.1用Pythonを準備しています。")
    run([uv, "python", "install", PYTHON_VERSION, "--no-bin", "--no-registry"])
    notify("Qwen Image 2.1の専用依存パッケージを準備しています。")
    command = [uv, "sync", "--python", PYTHON_VERSION, "--project", str(runtime)]
    if (runtime / "uv.lock").is_file():
        command.append("--locked")
    run(command)
    notify("QwenImage21PipelineとCUDAを確認しています。")
    run([str(runtime / ".venv/Scripts/python.exe"), "-X", "utf8", "-c",
         ('from diffusers import QwenImage21Pipeline; import torch; '
          'print("Qwen Image 2.1 ready / CUDA:", torch.cuda.is_available())')])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen Image 2.1専用実行環境の準備")
    parser.add_argument("--status-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        setup_runtime(status_dir=args.status_dir)
    except (OSError, subprocess.CalledProcessError) as exc:
        error = f"Qwen Image 2.1の環境準備に失敗しました: {exc}"
        print(error, file=sys.stderr, flush=True)
        if args.status_dir:
            write_json(args.status_dir / "status.json", {"phase": "error", "message": error})
            write_json(args.status_dir / "result.json", {"ok": False, "error": error})
        return 1
    if args.status_dir:
        write_json(args.status_dir / "status.json", {
            "phase": "done", "message": "Qwen Image 2.1の実行環境の準備が完了しました。"})
        write_json(args.status_dir / "result.json", {"ok": True})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
