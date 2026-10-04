"""Fetch a pinned official Qwen Image 2.1 snapshot and publish verified files.

The coordinator may import validation helpers without importing the Hub or torch.
Authentication is inherited from the environment or the existing Hub login; no
token is written to a job, model manifest, status file, or result record.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.image_edit.common import (
    MODEL_DIR,
    MODEL_ID,
    RUNTIME,
    hub_environment,
    read_json,
    write_json,
)

COMPONENTS = ("processor", "scheduler", "text_encoder", "transformer", "vae")
COMPONENT_CLASSES = {
    "processor": ["transformers", "Qwen3VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage21"],
}
MANIFEST = "model-manifest.json"
REQUIRED_FILES = (
    "model_index.json", "processor/preprocessor_config.json", "processor/tokenizer.json",
    "processor/tokenizer_config.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "transformer/config.json", "vae/config.json",
)


class PreparationCancelled(Exception):
    pass


def update_status(status_dir: Path, phase: str, message: str, **details) -> None:
    write_json(status_dir / "status.json", {"phase": phase, "message": message, **details})
    print(message, flush=True)


def check_cancel(status_dir: Path) -> None:
    if (status_dir / "stop.request").exists():
        raise PreparationCancelled("Qwen Image 2.1のモデル準備を中止しました。")


def safe_error(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    # A library exception must never expose an inherited credential in GUI logs.
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(name)
        if token:
            message = message.replace(token, "[credential]")
    message = re.sub(r"hf_[A-Za-z0-9]+", "[credential]", message)
    response = getattr(exc, "response", None)
    http_status = getattr(response, "status_code", None)
    if http_status in {401, 403} or any(x in message.lower() for x in ("gated", "401", "403")):
        return (
            "公式モデルへのアクセスが拒否されました。"
            f"https://huggingface.co/{MODEL_ID} で利用条件・アクセス権を確認し、"
            "同じアカウントで hf auth login を実行してください。"
            "制限付きトークンにはこのモデルの読み取り権限が必要です。\n" + message
        )
    return message


def model_files(directory: Path) -> list[Path]:
    """Hub local_dir transfer metadata is not part of the model snapshot."""
    return sorted(path for path in directory.rglob("*") if path.is_file()
                  and path.relative_to(directory).parts[0] != ".cache"
                  and path.name != MANIFEST)


def contained_file(directory: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute() or ":" in name
            or any(part in {"..", "."} for part in relative.parts)):
        raise ValueError("モデルのファイル一覧に不正なパスがあります。")
    path = directory.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
        raise ValueError("モデルのファイルがモデルフォルダーの外を参照しています。")
    return path


def validate_components(directory: Path) -> None:
    index = read_json(directory / "model_index.json")
    if index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("QwenImage21Pipelineのmodel_index.jsonを確認できません。")
    for component in COMPONENTS:
        definition = index.get(component)
        if definition != COMPONENT_CLASSES[component]:
            raise ValueError(f"公式モデルの構成と一致しません: {component}")
    for name in REQUIRED_FILES:
        if not contained_file(directory, name).is_file():
            raise ValueError(f"モデルの必須ファイルがありません: {name}")
    for component in ("text_encoder", "transformer", "vae"):
        folder = directory / component
        if not list(folder.glob("*.safetensors")):
            raise ValueError(f"モデル重みがありません: {component}")
        for index_path in folder.glob("*.safetensors.index.json"):
            weights = read_json(index_path).get("weight_map")
            if not isinstance(weights, dict) or not weights:
                raise ValueError(f"分割重みの一覧が不正です: {component}")
            for name in set(weights.values()):
                if not isinstance(name, str) or not contained_file(folder, name).is_file():
                    raise ValueError(f"分割重みが不足しています: {component}")


def file_digest(path: Path, status_dir: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            check_cancel(status_dir)
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(directory: Path, status_dir: Path) -> dict:
    """Rehash all saved files; never trust merely having model_index.json."""
    validate_components(directory)
    manifest = read_json(directory / MANIFEST)
    if (manifest.get("schema_version") != 1 or manifest.get("model_id") != MODEL_ID
            or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("revision", "")))):
        raise ValueError("準備済みモデルの公式リビジョン・検証記録を確認できません。")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("準備済みモデルのファイル検証記録がありません。")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise TypeError("準備済みモデルのファイル検証記録が不正です。")
        name = entry["path"]
        if name in seen or name == MANIFEST or name.startswith(".cache/"):
            raise ValueError("準備済みモデルのファイル検証記録が重複しています。")
        seen.add(name)
        path = contained_file(directory, name)
        expected_size = entry.get("size_bytes")
        expected_hash = entry.get("sha256")
        if (type(expected_size) is not int or expected_size < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash or ""))
                or not path.is_file() or path.stat().st_size != expected_size):
            raise ValueError(f"モデルファイルのサイズ・検証記録が一致しません: {name}")
        update_status(status_dir, "validating", f"モデルを検証しています: {name}",
                      current_file=name, revision=manifest["revision"])
        if file_digest(path, status_dir) != expected_hash:
            raise ValueError(f"モデルファイルのSHA256が一致しません: {name}")
    actual = {path.relative_to(directory).as_posix() for path in model_files(directory)}
    if actual != seen:
        raise ValueError("モデルのファイル一覧と検証記録が一致しません。")
    return manifest


def snapshot_entries(info) -> list[dict]:
    entries = []
    for item in info.siblings:
        name = item.rfilename
        if name not in {"model_index.json", "LICENSE", "README.md"} and name.split("/")[0] not in COMPONENTS:
            continue
        if type(item.size) is not int or item.size < 0:
            raise ValueError(f"公式モデルのファイルサイズを確定できません: {name}")
        entry = {"path": name, "size_bytes": item.size}
        lfs = getattr(item, "lfs", None)
        lfs_hash = getattr(lfs, "sha256", None)
        if lfs_hash:
            entry["sha256"] = lfs_hash
        entries.append(entry)
    if not set(REQUIRED_FILES).issubset({entry["path"] for entry in entries}):
        raise ValueError("公式リポジトリのモデル構成が想定と異なります。")
    return sorted(entries, key=lambda entry: entry["path"])


def download_worker(job_path: Path) -> None:
    job = read_json(job_path)
    status = Path(job["status_dir"])
    stage = Path(job["stage_dir"])
    check_cancel(status)
    from huggingface_hub import HfApi, snapshot_download

    update_status(status, "preparing", "公式モデルのリビジョンと容量を確認しています。")
    info = HfApi().model_info(MODEL_ID, revision=job.get("revision") or "main", files_metadata=True)
    revision = info.sha
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise ValueError("公式モデルの固定リビジョンを確定できません。")
    entries = snapshot_entries(info)
    total = sum(entry["size_bytes"] for entry in entries)
    write_json(status / "snapshot.json", {
        "model_id": MODEL_ID, "revision": revision, "gated": info.gated,
        "snapshot_required_bytes": total, "files": entries,
    })
    check_cancel(status)
    update_status(status, "downloading", "公式Qwen Image 2.1のモデルを取得しています。", 
                  revision=revision, gated=info.gated, total_bytes=total, downloaded_bytes=0)
    snapshot_download(MODEL_ID, revision=revision, local_dir=stage,
                      allow_patterns=[entry["path"] for entry in entries], max_workers=3)
    check_cancel(status)
    validate_components(stage)
    verified = []
    for entry in entries:
        name = entry["path"]
        path = contained_file(stage, name)
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"取得した公式ファイルのサイズが一致しません: {name}")
        update_status(status, "validating", f"SHA256を検証しています: {name}",
                      current_file=name, revision=revision, total_bytes=total,
                      downloaded_bytes=total)
        digest = file_digest(path, status)
        if entry.get("sha256") and digest != entry["sha256"]:
            raise ValueError(f"公式重みのSHA256と一致しません: {name}")
        verified.append({"path": name, "size_bytes": entry["size_bytes"], "sha256": digest})
    write_json(stage / MANIFEST, {
        "schema_version": 1, "model_id": MODEL_ID, "revision": revision,
        "gated": info.gated, "snapshot_required_bytes": total, "files": verified,
    })


def downloaded_bytes(stage: Path) -> int:
    total = 0
    for path in stage.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(stage)
        if relative.parts[0] != ".cache" or path.name.endswith(".incomplete"):
            try:
                total += path.stat().st_size
            except OSError:
                pass  # A completed transfer may be renamed during observation.
    return total


def run_child(job_path: Path, status_dir: Path) -> None:
    """Cancellation reaps only the downloader child created by this command."""
    job = read_json(job_path)
    process = subprocess.Popen(
        [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
         "--internal-job", str(job_path)],
        stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    last_update = 0.0
    try:
        while process.poll() is None:
            check_cancel(status_dir)
            now = time.monotonic()
            if now - last_update >= 2:
                current = read_json(status_dir / "status.json")
                snapshot = read_json(status_dir / "snapshot.json")
                if current.get("phase") == "downloading" and snapshot:
                    total = snapshot["snapshot_required_bytes"]
                    actual = min(downloaded_bytes(Path(job["stage_dir"])), total)
                    write_json(status_dir / "status.json", {
                        **current, "total_bytes": total, "downloaded_bytes": actual,
                        "progress": actual / total if total else 0,
                        "message": f"モデル取得中: {actual / 1e9:.2f} / {total / 1e9:.2f} GB",
                    })
                last_update = now
            time.sleep(0.1)
        check_cancel(status_dir)
        if process.returncode:
            error = read_json(status_dir / "child_error.json").get("error")
            raise RuntimeError(error or f"公式モデル取得に失敗しました (終了コード {process.returncode})。")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def prepare_model(model_dir: Path, status_dir: Path, revision: str | None = None) -> dict:
    output = model_dir.expanduser().resolve()
    status = status_dir.expanduser().resolve()
    update_status(status, "preparing", "Qwen Image 2.1のモデル準備を開始しています。")
    check_cancel(status)
    if output.exists() and (output.is_file() or any(output.iterdir())):
        if not output.is_dir() or not (output / MANIFEST).is_file():
            raise ValueError("モデル出力先に既存ファイルがあります。検証記録のあるモデルか空のフォルダーを選択してください。")
        manifest = validate_manifest(output, status)
        if revision and manifest["revision"] != revision:
            raise ValueError("指定リビジョンと準備済みモデルのリビジョンが一致しません。")
        update_status(status, "done", "準備済み公式モデルの検証が完了しました。", model_path=str(output))
        return {"model_path": str(output), **manifest, "reused": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Inherit the model parent's ACL. tempfile.mkdtemp uses mode 0700, which on
    # Windows prevents a later sandboxed GUI/test from reading an elevated download.
    # UUID plus an exclusive mkdir retains collision-safe ownership for cleanup.
    stage = output.parent / f".{output.name}.prepare-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        job = status / "prepare_job.json"
        write_json(job, {"stage_dir": str(stage), "status_dir": str(status), "revision": revision})
        # Remove only earlier job errors in this status directory.
        (status / "child_error.json").unlink(missing_ok=True)
        run_child(job, status)
        check_cancel(status)
        manifest = read_json(stage / MANIFEST)
        validate_components(stage)
        if not manifest or manifest.get("model_id") != MODEL_ID:
            raise RuntimeError("公式モデルの検証記録が作成されませんでした。")
        # The child verified each digest; avoid hashing 33 GB twice on first setup.
        if output.exists():
            if output.is_file() or any(output.iterdir()):
                raise ValueError("準備中にモデル出力先が変更されました。既存ファイルを保持します。")
            output.rmdir()
        stage.rename(output)
    finally:
        if (stage.exists() and stage.resolve().parent == output.parent
                and stage.name.startswith(f".{output.name}.prepare-")):
            shutil.rmtree(stage)
    update_status(status, "done", "Qwen Image 2.1の公式モデル準備が完了しました。",
                  model_path=str(output), revision=manifest["revision"],
                  snapshot_required_bytes=manifest["snapshot_required_bytes"])
    return {"model_path": str(output), **manifest, "reused": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="公式Qwen Image 2.1モデルの取得・検証")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--internal-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.environ.update(hub_environment(RUNTIME / "cache/huggingface", os.environ))
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if args.internal_job:
        job = read_json(args.internal_job)
        try:
            download_worker(args.internal_job)
            return 0
        except PreparationCancelled:
            return 0
        except Exception as exc:  # noqa: BLE001 - CLI reports failures without a traceback.
            error = safe_error(exc)
            write_json(Path(job["status_dir"]) / "child_error.json", {"error": error})
            print(error, file=sys.stderr, flush=True)
            return 1
    if args.status_dir is None:
        parser.error("--status-dir が必要です")
    try:
        details = prepare_model(args.model_dir, args.status_dir, args.revision)
        result = {"ok": True, "cancelled": False, **details}
        code = 0
    except PreparationCancelled as exc:
        update_status(args.status_dir, "cancelled", str(exc))
        result = {"ok": False, "cancelled": True, "error": str(exc)}
        code = 0
    except Exception as exc:  # noqa: BLE001 - Always complete the GUI result contract.
        error = safe_error(exc)
        update_status(args.status_dir, "error", error)
        result = {"ok": False, "cancelled": False, "error": error}
        code = 1
    write_json(args.status_dir / "result.json", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
