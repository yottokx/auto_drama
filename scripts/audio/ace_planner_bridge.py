"""Run the optional ACE LM to completion before importing/loading Diffusers.

The runner already holds the music GPU lease. The owned child uses that lease,
exits, and releases its entire CUDA context before DiT is allowed to load.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .ace_backend import LYRICS, VOCAL_LANGUAGE, instrumental_caption
from .engine import AudioGenerationError, GenerateRequest, GenerationCancelled, write_json
from .process_tree import WindowsChildJob

_CODES = re.compile(r"(?:<\|audio_code_\d{1,8}\|>)+")


def validate_result(result: Any, request: GenerateRequest) -> dict[str, Any]:
    """Fail closed on incomplete, rewritten, or non-audio planner results."""
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = result.get("error", "結果がありません。") if isinstance(result, dict) else "結果がありません。"
        raise AudioGenerationError(f"専用plannerに失敗しました: {detail}")
    codes = result.get("audio_codes")
    count = math.ceil(request.duration * 5)
    if (not isinstance(codes, str) or not _CODES.fullmatch(codes)
            or codes.count("<|audio_code_") != count or type(result.get("code_count")) is not int
            or result["code_count"] != count):
        raise AudioGenerationError("専用plannerの音楽コードの形式・秒数が正しくありません。")
    if any(int(value) > 63999 for value in re.findall(r"<\|audio_code_(\d+)\|>", codes)):
        raise AudioGenerationError("専用plannerの音楽コードは 0〜63999 の範囲である必要があります。")
    if result.get("actual_device") not in {"cpu", "cuda"} or (
        request.device != "auto" and result["actual_device"] != request.device
    ):
        raise AudioGenerationError("専用plannerの実行デバイスが指定と一致しません。")
    if result.get("planner_released") is not True:
        raise AudioGenerationError("専用plannerのモデル解放を確認できません。")
    meta, provenance = result.get("metadata"), result.get("planner_metadata")
    if not isinstance(meta, dict) or not isinstance(provenance, dict):
        raise AudioGenerationError("専用plannerの生成記録が不完全です。")
    for key, expected in (("duration", request.duration), ("bpm", request.bpm),
                          ("keyscale", request.keyscale), ("timesignature", request.timesignature)):
        if expected is not None and meta.get(key) != expected:
            raise AudioGenerationError(f"専用plannerが指定した {key} を変更しました。")
    for key, expected in (("caption", instrumental_caption(request.prompt, continuous=request.ace_continuous)), ("lyrics", LYRICS),
                          ("vocal_language", VOCAL_LANGUAGE), ("seed", request.seed)):
        if provenance.get(key) != expected:
            raise AudioGenerationError(f"専用plannerのインスト生成条件を確認できません ({key})。")
    return result


def run_planner(
    request: GenerateRequest, output_dir: Path, *, report: Callable[..., None],
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Start one local planner, wait for exit, then validate its saved result."""
    directory = output_dir / "planner"
    directory.mkdir(exist_ok=False)
    request_path = directory / "request.json"
    write_json(request_path, {
        "prompt": instrumental_caption(request.prompt, continuous=request.ace_continuous), "duration": request.duration,
        "bpm": request.bpm, "keyscale": request.keyscale, "timesignature": request.timesignature,
        "seed": request.seed, "device": request.device, "planner_model_path": request.planner_model_path,
    })
    report("planning", "専用plannerで音楽の構成・メロディー用コードを生成しています。")
    command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).with_name("ace_planner.py")),
               "--request", str(request_path), "--output-dir", str(directory)]
    process = None
    stop_started = None
    last_message = None
    with (directory / "process.log").open("w", encoding="utf-8") as log, WindowsChildJob() as job:
        try:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            job.assign(process)
            while process.poll() is None:
                if cancelled is not None and cancelled():
                    if stop_started is None:
                        (directory / "stop.request").touch()
                        stop_started = time.monotonic()
                    elif time.monotonic() - stop_started > 3:
                        process.terminate()
                try:
                    state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    state = {}
                message = state.get("message") if isinstance(state, dict) else None
                if isinstance(message, str) and message and message != last_message:
                    report("planning", message)
                    last_message = message
                time.sleep(0.1)
            process.wait()
            if stop_started is not None or (cancelled is not None and cancelled()):
                raise GenerationCancelled("専用plannerの生成を中止しました。")
            try:
                result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise AudioGenerationError(f"専用plannerの結果を読めません。ログ: {directory / 'process.log'}") from exc
            if process.returncode != 0:
                detail = result.get("error", f"終了コード {process.returncode}") if isinstance(result, dict) else "結果が不正です。"
                raise AudioGenerationError(f"専用plannerに失敗しました: {detail}\nログ: {directory / 'process.log'}")
            result = validate_result(result, request)
            result["result_path"] = str(directory / "result.json")
            report("releasing", "専用plannerの終了・GPU解放を確認しました。Diffusersへ切り替えます。")
            return result
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
