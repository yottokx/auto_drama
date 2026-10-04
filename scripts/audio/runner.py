"""Subprocess entry point for the Stable Audio 3 test GUI."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.audio.engine import (
    GenerationCancelled,
    explain_error,
    generate_audio,
    utc_timestamp,
    validate_request,
    write_json,
)
from scripts.audio.resources import cancellation_watcher, music_gpu_scope


def run_request(request_path: Path, output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    # A GUI retry must select a fresh directory to preserve earlier experiments.
    if result_path.exists():
        print(f"Existing result will not be overwritten: {result_path}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    steps = 0

    def status(data: dict) -> None:
        write_json(output_dir / "status.json", data)

    try:
        data = json.loads(request_path.read_text(encoding="utf-8-sig"))
        request = validate_request(data)
        steps = request.steps
        with cancellation_watcher(output_dir) as token:
            token.check()
            release = data.get("llm_release")
            if release is not None:
                from scripts.audio.llm_runtime import release_ollama

                if (not isinstance(release, dict) or not isinstance(release.get("base_url"), str)
                        or not isinstance(release.get("model"), str)):
                    raise ValueError("音楽生成前のLLM解放設定を読み込めません。")
                status({"phase": "releasing", "message": "音楽生成前にOllamaの指定モデルを解放しています。"})
                release_ollama(release["base_url"], release["model"])
            status({"phase": "waiting", "message": "他のGPU処理の終了を待っています。"})
            with music_gpu_scope(Path(__file__).resolve().parents[2], request.device):
                token.check()
                result = generate_audio(
                    request,
                    output_dir,
                    progress=status,
                    cancelled=lambda: (output_dir / "stop.request").exists(),
                )
                token.check()
        write_json(result_path, result)
        return 0
    except Exception as exc:  # noqa: BLE001 - report subprocess failures to the GUI
        is_cancelled = isinstance(exc, GenerationCancelled) or (output_dir / "stop.request").exists()
        message = explain_error(exc)
        result = {
            "ok": False,
            "cancelled": is_cancelled,
            "error": message,
            "error_type": type(exc).__name__,
        }
        write_json(result_path, result)
        status({
            "phase": "cancelled" if is_cancelled else "error",
            "message": message,
            "step": 0,
            "steps": steps,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "updated_at": utc_timestamp(),
        })
        if not is_cancelled:
            traceback.print_exc()
        return 130 if is_cancelled else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate local Stable Audio 3 stereo music.")
    parser.add_argument("--request", type=Path, required=True, help="Generation settings JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh experiment directory")
    arguments = parser.parse_args(argv)
    return run_request(arguments.request, arguments.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
