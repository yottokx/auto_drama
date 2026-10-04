"""Borrow local LLM assets for one audio prompt, then release their GPU memory.

The worker's model registry and process owners are reused without changing their
configuration or resident sessions. Ollama release targets only the selected model.
"""

from __future__ import annotations

import errno
import json
import math
import socket
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

from .engine import GenerationCancelled

_STARTUP_TIMEOUT = 300.0
_TOTAL_TIMEOUT = 600.0
_CONTEXT_SIZE = 16384
_ALIAS = "bgm-local"
_MAX_RESPONSE_BYTES = 1024 * 1024


def check_cancelled() -> None:
    """Import worker dependencies only in the coordinator's LLM subprocess."""
    from services.worker.generation.cancellation import check_cancelled as worker_check

    worker_check()


def gpu_lock(path: Path, timeout: float):
    from .resources import gpu_lock as audio_lock

    return audio_lock(path, timeout)


def owned_process(command: list[str], log: Path, *, cwd: Path, timeout: float):
    from services.worker.generation.processes import owned_process as worker_owned

    return worker_owned(command, log, cwd=cwd, timeout=timeout)


def _model_config():
    # The inference runtime can use Ollama release without importing Pydantic.
    from services.worker import model_config

    return model_config


def list_native_models(root: Path) -> list[str]:
    """List valid local GGUFs using the worker's existing deployment registry."""
    registry = _model_config()
    models = []
    for model_id, entry in registry.entries(root).items():
        try:
            registry.resolve_entry(root, model_id, entry)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        models.append(model_id)
    return sorted(models, key=str.casefold)


def default_native_model(root: Path) -> str:
    """Prefer the configured worker default when that model is available."""
    models = list_native_models(root)
    if not models:
        raise ValueError("利用できるローカルGGUFモデルとllama-serverがありません。")
    config = json.loads((root / "config/m2-generation.json").read_text(encoding="utf-8"))
    selected = config.get("llm", {}).get("model_id")
    return selected if selected in models else models[0]


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    check_cancelled()
    if cancelled is not None and cancelled():
        raise GenerationCancelled("LLMのプロンプト生成を中止しました。")


def _report(status: Callable[[dict], None] | None, phase: str, message: str) -> None:
    if status is not None:
        status({"phase": phase, "message": message})


def _json_request(opener, url: str, *, timeout: float, payload: dict | None = None) -> dict:
    request = Request(
        url,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ValueError("ローカルLLMの応答が1MiBを超えています。")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("ローカルLLMの応答がJSONオブジェクトではありません。")  # noqa: TRY004 - protocol error
    return result


@contextmanager
def managed_llama(root: Path, model_id: str, output: Path, *,
                  status: Callable[[dict], None] | None = None,
                  cancelled: Callable[[], bool] | None = None):
    """Own one ephemeral llama-server and hold the worker's shared GPU lease.

    The caller's cancellation_scope also interrupts GPU-lock waits and an active
    request. The optional callback handles standalone startup cancellation.
    The yielded URL includes /v1 and is ready for request_llm_prompt.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    _check_cancelled(cancelled)
    registry = _model_config()
    entries = registry.entries(root)
    if model_id not in entries:
        raise ValueError(f"このWorkerにローカルモデルがありません: {model_id}")
    base, _settings = registry.resolve_entry(root, model_id, entries[model_id])
    model = (root / base["model"]["relative_path"]).resolve(strict=True)
    server = (root / base["server"]["executable"]).resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _TOTAL_TIMEOUT
    opener = build_opener(ProxyHandler({}))
    _report(status, "waiting_gpu", "他のGPU処理の終了を待っています。")
    with gpu_lock(root / "services/worker/cache/m2/gpu.lock", _TOTAL_TIMEOUT):
        _check_cancelled(cancelled)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("LLMのGPU待機時間が上限に達しました。")
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        command = [str(server), "--model", str(model), "--host", "127.0.0.1",
                   "--port", str(port), "--ctx-size", str(_CONTEXT_SIZE),
                   "--n-gpu-layers", "all", "--parallel", "1", "--jinja",
                   "--reasoning-format", "deepseek", "--reasoning-budget", "0",
                   "--alias", _ALIAS]
        _report(status, "loading_llm", "ローカルLLMを読み込んでいます。")
        with owned_process(command, output / "llama-server.log", cwd=server.parent,
                           timeout=remaining) as process:
            try:
                health_deadline = min(deadline, time.monotonic() + _STARTUP_TIMEOUT)
                while True:
                    _check_cancelled(cancelled)
                    if process.poll() is not None:
                        raise RuntimeError("llama-serverの起動に失敗しました。llama-server.logを確認してください。")
                    remaining = health_deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("ローカルLLMの準備が制限時間内に完了しませんでした。")
                    try:
                        health = _json_request(opener, base_url + "/health",
                                               timeout=min(2.0, remaining))
                        if health.get("status") == "ok":
                            break
                    except (OSError, URLError, ValueError):
                        pass
                    time.sleep(min(0.1, max(0, health_deadline - time.monotonic())))
                _check_cancelled(cancelled)
                _report(status, "ready_llm", "シーンに合う英語BGMプロンプトを生成しています。")
                yield base_url + "/v1", _ALIAS
                _check_cancelled(cancelled)
            finally:
                _report(status, "releasing_llm", "LLMを終了してGPUメモリを解放しています。")


def is_ollama_url(base_url: str) -> bool:
    """Recognize the known local Ollama port; other endpoints require selection."""
    try:
        parsed = urlsplit(base_url)
        return (parsed.scheme in {"http", "https"}
                and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
                and parsed.port == 11434)
    except ValueError:
        return False


def _ollama_root(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OllamaのURLにはhttp://またはhttps://を指定してください。")
    path = parsed.path.rstrip("/").removesuffix("/v1")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _model_name(model: str) -> str:
    # Ollama's omitted tag means :latest; named tags remain distinct.
    return model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"


def _selected_loaded(result: dict, model: str) -> bool:
    rows = result.get("models")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Ollamaの常駐モデル一覧を確認できません。")
    expected = _model_name(model)
    return any(_model_name(str(row.get("name") or row.get("model") or "")) == expected
               for row in rows)


def _connection_refused(error: OSError) -> bool:
    reason = error.reason if isinstance(error, URLError) else error
    return (isinstance(reason, ConnectionRefusedError)
            or (isinstance(reason, OSError)
                and (reason.errno in {errno.ECONNREFUSED, 10061}
                     or getattr(reason, "winerror", None) == 10061)))


def release_ollama(base_url: str, model: str, *, timeout: float = 30) -> None:
    """Unload only the chosen Ollama model and confirm its memory release.

    An absent server or already-unloaded selected model is a no-op. Once the
    model is known to be loaded, any failed release stays visible to callers.
    """
    model = model.strip()
    if not model:
        raise ValueError("解放するOllamaモデル名を指定してください。")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Ollamaの解放タイムアウトは0より大きくしてください。")
    root = _ollama_root(base_url)
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    try:
        state = _json_request(opener, root + "/api/ps", timeout=min(3.0, timeout))
    except HTTPError as exc:
        raise RuntimeError(f"Ollamaの常駐確認に失敗しました（HTTP {exc.code}）。") from exc
    except (URLError, OSError) as exc:
        if _connection_refused(exc):
            return
        raise RuntimeError(f"Ollamaの常駐状態を確認できません: {exc}") from exc
    if not _selected_loaded(state, model):
        return
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Ollamaの解放時間が上限に達しました。")
        _json_request(opener, root + "/api/generate", timeout=remaining,
                      payload={"model": model, "keep_alive": 0, "stream": False})
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Ollamaモデルが制限時間内に解放されませんでした。")
            state = _json_request(opener, root + "/api/ps", timeout=min(3.0, remaining))
            if not _selected_loaded(state, model):
                return
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    except (OSError, URLError, ValueError) as exc:
        raise RuntimeError(f"Ollamaモデル {model} のGPU解放に失敗しました: {exc}") from exc
