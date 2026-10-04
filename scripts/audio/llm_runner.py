"""Produce a scene prompt and release its LLM before reporting success."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from scripts.audio.catalog import Scene
from scripts.audio.engine import utc_timestamp
from scripts.audio.json_io import write_json
from scripts.audio.prompts import (
    _finalize_ace_prompt,
    normalize_ace_metadata,
    request_llm_prompt,
    validate_music_backend,
)
from scripts.audio.resources import cancellation_watcher, music_gpu_scope


def _normalize_prompt_result(value: object, *, music_backend: str = "stable_audio3") -> dict:
    """Validate final output fields without exposing a model's reasoning."""
    legacy = isinstance(value, str)
    if legacy:
        prompt, interpretation = value, ""
    elif isinstance(value, dict):
        prompt = value.get("prompt")
        interpretation = value.get("scene_interpretation")
    else:
        raise ValueError("LLMの回答から音楽プロンプトとシーン解釈を読み込めません。")
    if not isinstance(prompt, str):
        raise ValueError("LLMが有効な英語の音楽プロンプトを返しませんでした。")  # noqa: TRY004 - response validation
    prompt = prompt.strip()
    japanese = r"[\u3040-\u30ff\u3400-\u9fff]"
    if (not prompt or len(prompt) > 2500 or re.search(japanese, prompt)
            or len(re.findall(r"[A-Za-z]{3,}", prompt)) < 3):
        raise ValueError("LLMが有効な英語の音楽プロンプトを返しませんでした。")
    if not isinstance(interpretation, str):
        raise ValueError("LLMが日本語の短いシーン解釈を返しませんでした。")  # noqa: TRY004 - response validation
    interpretation = interpretation.strip()
    if not legacy and (not interpretation or len(interpretation) > 600
                       or not re.search(japanese, interpretation)):
        raise ValueError("LLMが日本語の短いシーン解釈を返しませんでした。")
    result = {"prompt": prompt, "scene_interpretation": interpretation}
    if music_backend == "ace_step15":
        if legacy:
            raise ValueError("ACE-StepのLLM応答には英語captionに加えて BPM・調・拍子が必要です。")
        result["prompt"] = _finalize_ace_prompt(prompt)
        result["ace_metadata"] = normalize_ace_metadata(value.get("ace_metadata"))
    return result


def run_request(request_path: Path, output_dir: Path) -> int:
    from scripts.audio.llm_runtime import is_ollama_url, managed_llama, release_ollama

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        print(f"Existing result will not be overwritten: {result_path}", file=sys.stderr)
        return 2
    started = time.monotonic()

    def status(value: dict):
        write_json(output_dir / "status.json", {
            **value, "elapsed_seconds": round(time.monotonic() - started, 3),
            "updated_at": utc_timestamp(),
        })

    def cancelled():
        return (output_dir / "stop.request").exists()

    try:
        data = json.loads(request_path.read_text(encoding="utf-8-sig"))
        backend = data.get("backend", "llama_cpp")
        if backend not in {"llama_cpp", "ollama", "openai"}:
            raise ValueError("LLMの実行方法を選択してください。")
        source = data["scene"]
        scene = Scene(source["id"], source.get("label", ""), source["context"], source.get("preview", ""))
        options = data.get("options", {})
        if not isinstance(options, dict) or set(options) - {"style", "mood", "tempo", "music_backend"}:
            raise ValueError("LLMの音楽設定を読み込めません。")
        options = dict(options)
        music_backend = validate_music_backend(options.pop("music_backend", "stable_audio3"))
        if music_backend == "ace_step15":
            options["music_backend"] = music_backend
        base_url, model = data.get("base_url", ""), data.get("model", "")
        with cancellation_watcher(output_dir, worker_helpers=True) as token:
            token.check()
            if backend == "llama_cpp":
                if is_ollama_url(base_url) and model:
                    status({"phase": "releasing", "message": "以前指定したOllamaモデルを解放しています。"})
                    release_ollama(base_url, model)
                with managed_llama(ROOT, data.get("local_model", ""), output_dir,
                                   status=status, cancelled=cancelled) as (local_url, local_alias):
                    token.check()
                    status({"phase": "prompt", "message": "llama.cppで英語の音楽プロンプトを作成しています。"})
                    details = request_llm_prompt(
                        scene, local_url, local_alias, **options,
                        request_options={"chat_template_kwargs": {"enable_thinking": False}},
                        return_details=True,
                    )
                status({"phase": "releasing", "message": "LLMを終了し、GPUメモリーを解放しました。"})
            elif backend == "ollama":
                status({"phase": "waiting", "message": "他のGPU処理の終了を待っています。"})
                with music_gpu_scope(ROOT, "cuda"):
                    try:
                        token.check()
                        status({"phase": "prompt", "message": "Ollamaで英語の音楽プロンプトを作成しています。"})
                        details = request_llm_prompt(scene, base_url, model, **options,
                                                     request_options={"reasoning_effort": "none"},
                                                     return_details=True)
                    finally:
                        try:
                            status({"phase": "releasing", "message": "Ollamaの指定モデルを解放しています。"})
                        finally:
                            release_ollama(base_url, model)
            else:
                status({"phase": "prompt", "message": "外部APIで英語の音楽プロンプトを作成しています。"})
                details = request_llm_prompt(scene, base_url, model, **options, return_details=True)
            token.check()
            details = _normalize_prompt_result(details, music_backend=music_backend)
        write_json(result_path, {
            "ok": True, **details, "backend": backend,
            "local_model": data.get("local_model"),
            "llm_released": backend in {"llama_cpp", "ollama"},
            **({"music_backend": music_backend, "native_planner_used": False}
               if music_backend == "ace_step15" else {}),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        status({"phase": "done", "message": "プロンプト作成が完了しました。音楽を生成できます。"})
        return 0
    except Exception as exc:  # noqa: BLE001 - report subprocess failures to the GUI
        was_cancelled = cancelled()
        message = "プロンプト作成を停止しました。" if was_cancelled else str(exc)
        write_json(result_path, {"ok": False, "cancelled": was_cancelled,
                                 "error": message, "error_type": type(exc).__name__})
        status({"phase": "cancelled" if was_cancelled else "error", "message": message})
        if not was_cancelled:
            traceback.print_exc()
        return 130 if was_cancelled else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create an English BGM prompt with a bounded LLM lifetime")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_request(args.request, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
