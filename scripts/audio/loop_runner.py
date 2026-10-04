"""Subprocess entry point for cancellable standalone loop experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from scripts.audio.engine import GenerationCancelled, explain_error, utc_timestamp
from scripts.audio.json_io import write_json
from scripts.audio.loops import AI_METHODS, create_loops, validate_loop_request
from scripts.audio.resources import cancellation_watcher, music_gpu_scope


def run_request(request_path: Path, output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        print(f"Existing result will not be overwritten: {result_path}", file=sys.stderr)
        return 2
    started = time.perf_counter()

    def status(data):
        write_json(output_dir / "status.json", {
            **data, "elapsed_seconds": round(time.perf_counter() - started, 3),
            "updated_at": utc_timestamp(),
        })

    def cancelled():
        return (output_dir / "stop.request").exists()

    try:
        data = json.loads(request_path.read_text(encoding="utf-8-sig"))
        request = validate_loop_request(data)
        with cancellation_watcher(output_dir) as token:
            token.check()
            if request.method in AI_METHODS:
                release = data.get("llm_release")
                if release is not None:
                    from scripts.audio.llm_runtime import release_ollama

                    if (not isinstance(release, dict) or not isinstance(release.get("base_url"), str)
                            or not isinstance(release.get("model"), str)):
                        raise ValueError("AI修復前のLLM解放設定を読み込めません。")
                    status({"phase": "releasing", "message": "AI修復前に指定したOllamaモデルを解放しています。"})
                    release_ollama(release["base_url"], release["model"])
                status({"phase": "waiting", "message": "他のGPU処理の終了を待っています。"})
            lease = music_gpu_scope(ROOT, request.device) if request.method in AI_METHODS else nullcontext()
            with lease:
                token.check()
                result = create_loops(request, output_dir, progress=status, cancelled=cancelled)
                token.check()
            token.check()
        write_json(result_path, result)
        return 0
    except Exception as exc:  # noqa: BLE001 - subprocess reports failures to the GUI
        was_cancelled = isinstance(exc, GenerationCancelled) or cancelled()
        message = "ループ実験を停止しました。" if was_cancelled else explain_error(exc)
        failure = {"ok": False, "cancelled": was_cancelled, "error": message,
                   "error_type": type(exc).__name__}
        # A cancellation after candidate export must not leave discoverable success
        # records in a GUI that scans candidate subdirectories recursively.
        if was_cancelled:
            for candidate_result in output_dir.glob("candidate-*/result.json"):
                write_json(candidate_result, failure)
        write_json(result_path, failure)
        status({"phase": "cancelled" if was_cancelled else "error", "message": message})
        if not was_cancelled:
            traceback.print_exc()
        return 130 if was_cancelled else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create Stable Audio test loop candidates")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_request(args.request, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
