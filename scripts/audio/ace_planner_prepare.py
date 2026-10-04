"""Download the official ACE-Step 1.5 1.7B planner without executing Hub code."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.audio.prepare import PreparationCancelled, check_cancel, update_status, write_json

REPO_ID = "ACE-Step/Ace-Step1.5"
REPO_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
MODEL_SUBFOLDER = "acestep-5Hz-lm-1.7B"
MODEL_KEY = "ace15_planner_1_7b"
MODEL_FILES = (
    "config.json", "model.safetensors", "added_tokens.json", "chat_template.jinja",
    "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
)
SOURCE_FILES = tuple(f"{MODEL_SUBFOLDER}/{name}" for name in MODEL_FILES)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "services/worker/runtimes/stable_audio3/models" / MODEL_KEY
)


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"planner の設定は JSON オブジェクトである必要があります: {path.name}")
    return value


def validate_config(config: dict) -> None:
    expected = {
        "model_type": "qwen3", "hidden_size": 2048, "intermediate_size": 6144,
        "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
        "head_dim": 128, "vocab_size": 217204, "tie_word_embeddings": True,
    }
    if any(config.get(key) != value for key, value in expected.items()) or config.get("auto_map"):
        raise ValueError("公式 ACE-Step 1.5 planner 1.7B の Qwen3 構造と一致しません。")


def validate_model(path: Path) -> dict:
    if not path.is_dir() or any(not (path / name).is_file() or (path / name).stat().st_size == 0
                                  for name in MODEL_FILES):
        raise ValueError("ACE planner のモデルが未準備または不完全です。planner モデルの準備を実行してください。")
    validate_config(read_object(path / "config.json"))
    manifest = read_object(path / "ace_planner_preparation.json")
    expected = {"model": MODEL_KEY, "repo_id": REPO_ID, "repo_revision": REPO_REVISION,
                "model_subfolder": MODEL_SUBFOLDER, "remote_code": False}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("ACE planner の準備記録が公式 1.7B モデルと一致しません。")
    return manifest


def download_checkpoint(cache_dir: Path, status_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    cache = str(cache_dir / "huggingface")
    check_cancel(status_dir)
    update_status(status_dir, "preparing", "公式 ACE planner 1.7B の構造を確認しています。")
    config_path = Path(hf_hub_download(
        REPO_ID, f"{MODEL_SUBFOLDER}/config.json", revision=REPO_REVISION,
        cache_dir=cache, token=False,
    ))
    validate_config(read_object(config_path))
    check_cancel(status_dir)
    update_status(status_dir, "preparing", "公式 ACE planner 1.7B を取得しています。初回は約 3.76 GB です。")
    snapshot = Path(snapshot_download(
        REPO_ID, revision=REPO_REVISION, allow_patterns=list(SOURCE_FILES),
        cache_dir=cache, token=False, max_workers=4,
    ))
    check_cancel(status_dir)
    source = snapshot / MODEL_SUBFOLDER
    if any(not (source / name).is_file() for name in MODEL_FILES):
        raise RuntimeError("公式 ACE planner の必要なファイルが不足しています。")
    return source


def preparation_worker(job_path: Path) -> None:
    job = read_object(job_path)
    status_dir = Path(job["status_dir"])
    stage = Path(job["stage_dir"])
    source = download_checkpoint(Path(job["cache_dir"]), status_dir)
    for name in MODEL_FILES:
        check_cancel(status_dir)
        shutil.copy2(source / name, stage / name)
    check_cancel(status_dir)
    write_json(stage / "ace_planner_preparation.json", {
        "model": MODEL_KEY, "repo_id": REPO_ID, "repo_revision": REPO_REVISION,
        "model_subfolder": MODEL_SUBFOLDER, "source_files": list(SOURCE_FILES),
        "remote_code": False, "sample_rate_hz": 5, "audio_code_min": 0, "audio_code_max": 63999,
    })
    validate_model(stage)


def run_child(job_path: Path, status_dir: Path) -> None:
    child = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--internal-job", str(job_path)],
        stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        while child.poll() is None:
            check_cancel(status_dir)
            time.sleep(0.1)
        check_cancel(status_dir)
        if child.returncode:
            details = status_dir / "child_error.json"
            message = read_object(details).get("error") if details.is_file() else None
            raise RuntimeError(message or f"ACE planner の取得に失敗しました ({child.returncode})。")
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)


def prepare_model(output_dir: Path, cache_dir: Path, status_dir: Path) -> Path:
    output, cache, status = output_dir.resolve(), cache_dir.resolve(), status_dir.resolve()
    check_cancel(status)
    if output.exists() and (output.is_file() or any(output.iterdir())):
        validate_model(output)
        update_status(status, "done", "準備済みの ACE planner 1.7B を使用します。", model_path=str(output))
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    try:
        job = status / "prepare_planner_job.json"
        write_json(job, {"stage_dir": str(stage), "cache_dir": str(cache), "status_dir": str(status)})
        run_child(job, status)
        check_cancel(status)
        validate_model(stage)
        if output.exists():
            if output.is_file() or any(output.iterdir()):
                raise ValueError("準備中に出力先が変更されました。既存のファイルは保持します。")
            output.rmdir()
        stage.replace(output)
        update_status(status, "done", "ACE planner 1.7B の準備が完了しました。", model_path=str(output))
        return output
    finally:
        if stage.exists():
            # stage is a fresh owned directory under the resolved model parent.
            if stage.parent != output.parent or not stage.name.startswith(f".{output.name}.prepare-"):
                raise RuntimeError("planner の一時フォルダーを確認できません。")
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the official ACE-Step 1.5 1.7B planner.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MODEL_PATH.parent.parent / "cache")
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--internal-job", type=Path)
    args = parser.parse_args(argv)
    if args.internal_job:
        try:
            preparation_worker(args.internal_job)
            return 0
        except Exception as exc:
            job = read_object(args.internal_job)
            write_json(Path(job["status_dir"]) / "child_error.json", {"error": str(exc)})
            raise
    if args.status_dir is None:
        parser.error("--status-dir is required")
    args.status_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = prepare_model(args.output_dir, args.cache_dir, args.status_dir)
        write_json(args.status_dir / "result.json", {"ok": True, "model_path": str(path), "model": MODEL_KEY})
        return 0
    except Exception as exc:  # noqa: BLE001 - subprocess reports all failures to its GUI caller.
        cancelled = isinstance(exc, PreparationCancelled) or (args.status_dir / "stop.request").exists()
        write_json(args.status_dir / "result.json", {"ok": False, "cancelled": cancelled, "error": str(exc)})
        update_status(args.status_dir, "cancelled" if cancelled else "error", str(exc))
        return 130 if cancelled else 1


if __name__ == "__main__":
    raise SystemExit(main())
