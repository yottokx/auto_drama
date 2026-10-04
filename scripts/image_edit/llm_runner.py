"""Create a scene image prompt and release the owned LLM before publication."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from scripts.audio.engine import utc_timestamp
from scripts.audio.json_io import write_json
from scripts.audio.resources import cancellation_watcher, music_gpu_scope
from scripts.image_edit.llm_prompts import normalize_scene_prompt, request_scene_prompt


def run_request(request_path: Path, output_dir: Path) -> int:
    from scripts.audio.llm_runtime import (
        default_native_model,
        is_ollama_url,
        managed_llama,
        release_ollama,
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        print(f"Existing result will not be overwritten: {result_path}", file=sys.stderr)
        return 2
    started = time.monotonic()
    backend, local_model, released = None, None, False

    def status(value: dict):
        write_json(output_dir / "status.json", {
            **value, "elapsed_seconds": round(time.monotonic() - started, 3),
            "updated_at": utc_timestamp(),
        })

    def cancelled():
        return (output_dir / "stop.request").exists()

    try:
        data = json.loads(request_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise TypeError("LLMの依頼はJSONオブジェクトで指定してください。")
        backend = data.get("backend", "llama_cpp")
        if backend not in {"llama_cpp", "ollama", "openai"}:
            raise ValueError("LLMの実行方法を選択してください。")
        scene, references = data["scene"], data["references"]
        instruction = data.get("instruction", "")
        base_url, model = data.get("base_url", ""), data.get("model", "")
        local_model = data.get("local_model", "")
        options = {"api_key": data.get("api_key"), "timeout": data.get("timeout", 120)}
        request_options = data.get("request_options") or {}
        if not isinstance(request_options, dict):
            raise TypeError("LLMの追加設定はJSONオブジェクトで指定してください。")
        with cancellation_watcher(output_dir, worker_helpers=True) as token:
            token.check()
            if backend == "llama_cpp":
                if is_ollama_url(base_url) and model:
                    status({"phase": "releasing", "message": "以前指定したOllamaモデルを解放しています。"})
                    release_ollama(base_url, model)
                local_model = local_model or default_native_model(ROOT)

                def native_status(value: dict):
                    if value.get("phase") == "ready_llm":
                        value = {**value, "message": "場面を解釈して一枚絵の指示を作成しています。"}
                    status(value)

                prompt_error = None
                with managed_llama(ROOT, local_model, output_dir, status=native_status,
                                   cancelled=cancelled) as (local_url, local_alias):
                    token.check()
                    status({"phase": "prompt", "message": "ローカルLLMで場面と構図を解釈しています。"})
                    try:
                        details = request_scene_prompt(
                            scene, references, instruction, local_url, local_alias, **options,
                            request_options={**request_options,
                                             "chat_template_kwargs": {"enable_thinking": False}},
                        )
                    except Exception as exc:  # noqa: BLE001 - publish failures after confirming release
                        prompt_error = exc
                released = True
                status({"phase": "releasing", "message": "LLMを終了し、GPUメモリーを解放しました。"})
                if prompt_error is not None:
                    raise prompt_error
            elif backend == "ollama":
                status({"phase": "waiting", "message": "他のGPU処理の終了を待っています。"})
                with music_gpu_scope(ROOT, "cuda"):
                    try:
                        token.check()
                        status({"phase": "prompt", "message": "Ollamaで場面と構図を解釈しています。"})
                        details = request_scene_prompt(
                            scene, references, instruction, base_url, model, **options,
                            request_options={**request_options, "reasoning_effort": "none"},
                        )
                    finally:
                        try:
                            status({"phase": "releasing", "message": "Ollamaの指定モデルを解放しています。"})
                        finally:
                            release_ollama(base_url, model)
                            released = True
            else:
                status({"phase": "prompt", "message": "外部LLMで場面と構図を解釈しています。"})
                details = request_scene_prompt(scene, references, instruction, base_url, model,
                                               **options, request_options=request_options)
            token.check()
            details = normalize_scene_prompt(details, len(references))
        write_json(result_path, {
            "ok": True, **details, "backend": backend, "local_model": local_model,
            "llm_released": released, "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        status({"phase": "done", "message": "場面解釈と画像の指示ができました。内容を確認できます。"})
        return 0
    except Exception as exc:  # noqa: BLE001 - subprocess protocol reports failures to the GUI
        was_cancelled = cancelled()
        message = "場面の指示作成を停止しました。" if was_cancelled else str(exc)
        write_json(result_path, {
            "ok": False, "cancelled": was_cancelled, "error": message,
            "error_type": type(exc).__name__, "backend": backend, "local_model": local_model,
            "llm_released": released, "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        status({"phase": "cancelled" if was_cancelled else "error", "message": message})
        if not was_cancelled:
            traceback.print_exc()
        return 130 if was_cancelled else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Interpret a selected scene as one image prompt")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_request(args.request, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
