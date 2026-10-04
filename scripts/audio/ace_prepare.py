"""Prepare the official ACE-Step 1.5 Turbo checkpoint for Diffusers.

Only the DiT, VAE and required Qwen embedding/tokenizer files are downloaded.
The separate ACE language-model planner is never downloaded or executed.
Conversion uses a reviewed, hash-pinned official script and local files only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

try:
    from .prepare import PreparationCancelled, check_cancel, update_status, write_json
except ImportError:
    from prepare import PreparationCancelled, check_cancel, update_status, write_json

MODEL_KEY = "ace15_turbo"
REPO_ID = "ACE-Step/Ace-Step1.5"
REPO_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
DIT_CONFIG = "acestep-v15-turbo"
CONVERTER_REVISION = "8b33bfc04b6b5e8bb58a58e55f68746c1bbee4cd"
CONVERTER_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/"
    f"{CONVERTER_REVISION}/scripts/convert_ace_step_to_diffusers.py"
)
CONVERTER_SHA256 = "63791b98c24ba17e1c8f24437fc3f7ac7e1ea4d0fd1c2e3d136476ae148507c5"
DIFFUSERS_VERSION = "0.40.0"
SOURCE_FILES = (
    f"{DIT_CONFIG}/config.json",
    f"{DIT_CONFIG}/model.safetensors",
    f"{DIT_CONFIG}/silence_latent.pt",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "Qwen3-Embedding-0.6B/config.json",
    "Qwen3-Embedding-0.6B/model.safetensors",
    "Qwen3-Embedding-0.6B/added_tokens.json",
    "Qwen3-Embedding-0.6B/chat_template.jinja",
    "Qwen3-Embedding-0.6B/merges.txt",
    "Qwen3-Embedding-0.6B/special_tokens_map.json",
    "Qwen3-Embedding-0.6B/tokenizer.json",
    "Qwen3-Embedding-0.6B/tokenizer_config.json",
    "Qwen3-Embedding-0.6B/vocab.json",
)
COMPONENTS = (
    "transformer", "condition_encoder", "vae", "text_encoder", "tokenizer",
    "scheduler", "audio_tokenizer", "audio_token_detokenizer",
)
DTYPES = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"設定ファイルが JSON オブジェクトではありません: {path.name}")
    return value


def validate_reference_config(config: dict) -> None:
    expected = {
        "is_turbo": True, "model_version": "turbo", "hidden_size": 2048,
        "intermediate_size": 6144, "num_hidden_layers": 24,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "in_channels": 192, "audio_acoustic_hidden_dim": 64,
        "text_hidden_dim": 1024, "timbre_hidden_dim": 64,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("公式 ACE-Step 1.5 Turbo (2B) の構造と一致しません。")


def _weights_complete(folder: Path) -> bool:
    weights = list(folder.glob("*.safetensors"))
    if not weights or any(weight.stat().st_size == 0 for weight in weights):
        return False
    for index in folder.glob("*.safetensors.index.json"):
        mapping = _read_json(index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            return False
        for name in set(mapping.values()):
            if (not isinstance(name, str) or Path(name).name != name
                    or not (folder / name).is_file() or (folder / name).stat().st_size == 0):
                return False
    return True


def is_complete_model(path: Path) -> bool:
    """Require every component written by the official converter, including codecs."""
    try:
        index = _read_json(path / "model_index.json")
        if index.get("_class_name") != "AceStepPipeline":
            return False
        for component in COMPONENTS:
            declaration = index.get(component)
            folder = path / component
            if (not isinstance(declaration, list) or len(declaration) != 2
                    or not all(isinstance(item, str) and item for item in declaration)
                    or not folder.is_dir()):
                return False
            if (component not in {"tokenizer", "scheduler"}
                    and (not (folder / "config.json").is_file() or not _weights_complete(folder))):
                return False
        transformer = _read_json(path / "transformer/config.json")
        if (transformer.get("is_turbo") is not True
                or transformer.get("model_version") != "turbo"
                or transformer.get("hidden_size") != 2048
                or transformer.get("num_hidden_layers") != 24):
            return False
        scheduler = _read_json(path / "scheduler/scheduler_config.json")
        if scheduler.get("num_train_timesteps") != 1 or scheduler.get("shift") != 1.0:
            return False
        return ((path / "tokenizer/tokenizer_config.json").is_file()
                and (path / "tokenizer/tokenizer.json").is_file())
    except (OSError, ValueError, TypeError):
        return False


def _valid_manifest(path: Path) -> bool:
    try:
        manifest = _read_json(path / "ace_preparation.json")
        expected = {
            "model": MODEL_KEY, "repo_id": REPO_ID, "repo_revision": REPO_REVISION,
            "dit_config": DIT_CONFIG, "converter_sha256": CONVERTER_SHA256,
            "diffusers_version": DIFFUSERS_VERSION, "planner_enabled": False,
            "planner_downloaded": False, "silence_latent_verified": True,
        }
        return (all(manifest.get(key) == value for key, value in expected.items())
                and manifest.get("dtype") in DTYPES)
    except (OSError, ValueError, TypeError):
        return False


def fetch_converter(cache_dir: Path) -> Path:
    directory = cache_dir / "converter"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"convert_ace_step_{CONVERTER_REVISION}.py"
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == CONVERTER_SHA256:
        return path
    with urllib.request.urlopen(CONVERTER_URL, timeout=30) as response:
        source = response.read(2_000_000)
    if hashlib.sha256(source).hexdigest() != CONVERTER_SHA256:
        raise ValueError("公式 ACE-Step 変換スクリプトのチェックサムが一致しません。")
    temporary = path.with_suffix(".download")
    temporary.write_bytes(source)
    os.replace(temporary, path)
    return path


def download_checkpoint(cache_dir: Path, status_dir: Path) -> tuple[Path, dict]:
    from huggingface_hub import hf_hub_download, snapshot_download

    cache = str(cache_dir / "huggingface")
    check_cancel(status_dir)
    update_status(status_dir, "preparing", "公式 ACE-Step 1.5 Turbo の構造を確認しています。")
    config_path = Path(hf_hub_download(
        REPO_ID, f"{DIT_CONFIG}/config.json", revision=REPO_REVISION,
        cache_dir=cache, token=False,
    ))
    config = _read_json(config_path)
    validate_reference_config(config)
    check_cancel(status_dir)
    update_status(status_dir, "preparing", "ACE-Step 1.5 Turbo・VAE・必須の Qwen 埋め込みを取得しています。約 6.34 GB です。")
    snapshot = Path(snapshot_download(
        REPO_ID, revision=REPO_REVISION, allow_patterns=list(SOURCE_FILES),
        cache_dir=cache, token=False, max_workers=4,
    ))
    check_cancel(status_dir)
    if any(not (snapshot / name).is_file() for name in SOURCE_FILES):
        raise RuntimeError("公式 ACE-Step の必要なファイルが不足しています。")
    return snapshot, config


def validate_silence_buffer(output: Path, checkpoint: Path, dtype: str) -> None:
    """Prevent the missing/transposed-silence defect that produces drone audio."""
    import torch
    from safetensors import safe_open

    source = torch.load(checkpoint / DIT_CONFIG / "silence_latent.pt", weights_only=True, map_location="cpu")
    expected = source.transpose(1, 2).to(getattr(torch, dtype)).contiguous()
    if tuple(expected.shape) != (1, 15000, 64) or not torch.isfinite(expected).all():
        raise ValueError("公式の無音潜在表現の形状または値が不正です。")
    found = None
    for weights in (output / "condition_encoder").glob("*.safetensors"):
        with safe_open(weights, framework="pt", device="cpu") as tensors:
            if "silence_latent" in tensors.keys():  # noqa: SIM118 - safe_open is not a mapping.
                found = tensors.get_tensor("silence_latent")
                break
    if found is None or not torch.equal(found, expected) or not torch.count_nonzero(found):
        raise ValueError("必須の無音潜在表現が正しく変換されていません。")


def conversion_worker(job_path: Path) -> None:
    import diffusers

    if diffusers.__version__ != DIFFUSERS_VERSION:
        raise RuntimeError("ACE-Step のモデル準備には diffusers==0.40.0 が必要です。環境準備を実行してください。")
    job = _read_json(job_path)
    status = Path(job["status_dir"])
    cache = Path(job["cache_dir"])
    converter_path = fetch_converter(cache)
    checkpoint, config = download_checkpoint(cache, status)
    check_cancel(status)
    spec = importlib.util.spec_from_file_location("ace_official_converter", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    update_status(status, "preparing", "公式 ACE-Step 1.5 Turbo を Diffusers 形式へ変換しています。LM planner は使いません。")
    # All inputs are local at this point. Prevent accidental extra Hub downloads
    # in a future converter/Transformers code path inside this isolated child.
    offline_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in offline_keys}
    try:
        os.environ.update({key: "1" for key in offline_keys})
        converter.convert_ace_step_weights(
            checkpoint_dir=str(checkpoint), dit_config=DIT_CONFIG,
            output_dir=job["stage_dir"], dtype_str=DTYPES[job["dtype"]],
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    check_cancel(status)
    output = Path(job["stage_dir"])
    if not is_complete_model(output):
        raise RuntimeError("変換後の ACE-Step 構成ファイルまたはスケジューラーが不正です。")
    validate_silence_buffer(output, checkpoint, job["dtype"])
    write_json(output / "ace_preparation.json", {
        "model": MODEL_KEY, "repo_id": REPO_ID, "repo_revision": REPO_REVISION,
        "dit_config": DIT_CONFIG, "converter_url": CONVERTER_URL,
        "converter_sha256": CONVERTER_SHA256, "converter_revision": CONVERTER_REVISION,
        "diffusers_version": DIFFUSERS_VERSION, "dtype": job["dtype"],
        "planner_enabled": False, "planner_downloaded": False,
        "text_encoder": "Qwen3-Embedding-0.6B", "source_files": list(SOURCE_FILES),
        "silence_latent_verified": True, "reference_config": config,
    })


def run_child(job: Path, status_dir: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--internal-job", str(job)],
        stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        while process.poll() is None:
            check_cancel(status_dir)
            time.sleep(0.1)
        check_cancel(status_dir)
        if process.returncode:
            details = status_dir / "child_error.json"
            if details.is_file():
                error = _read_json(details).get("error")
                if error:
                    raise RuntimeError(error)
            raise RuntimeError(f"ACE-Step のモデル変換に失敗しました (終了コード {process.returncode})。")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def prepare_model(model: str, output_dir: Path, cache_dir: Path, status_dir: Path,
                  dtype: str = "bfloat16") -> Path:
    if model != MODEL_KEY or dtype not in DTYPES:
        raise ValueError("ACE-Step 1.5 Turbo と有効な保存精度を指定してください。")
    output, cache, status = output_dir.resolve(), cache_dir.resolve(), status_dir.resolve()
    update_status(status, "preparing", "ACE-Step 1.5 Turbo のモデルを準備しています。")
    check_cancel(status)
    if is_complete_model(output):
        if not _valid_manifest(output):
            raise ValueError("準備済み ACE-Step モデルの生成記録を確認できません。")
        update_status(status, "done", "準備済みの ACE-Step 1.5 Turbo を使用します。", model_path=str(output))
        return output
    if output.exists() and (output.is_file() or any(output.iterdir())):
        raise ValueError("出力先に既存のファイルがあります。空のモデルフォルダーを選択してください。")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    try:
        job = status / "prepare_job.json"
        write_json(job, {"model": model, "stage_dir": str(stage), "cache_dir": str(cache),
                         "status_dir": str(status), "dtype": dtype})
        run_child(job, status)
        check_cancel(status)
        if not is_complete_model(stage) or not _valid_manifest(stage):
            raise RuntimeError("ACE-Step の準備の検証に失敗しました。")
        if output.exists():
            if output.is_file() or any(output.iterdir()):
                raise ValueError("モデルの準備中に出力先が変更されました。")
            output.rmdir()
        stage.rename(output)
    finally:
        # This resolved direct child was made by mkdtemp above. Never remove a
        # command-line-supplied directory or a replacement symbolic link.
        if (stage.exists() and not stage.is_symlink() and stage.resolve().parent == output.parent
                and stage.name.startswith(f".{output.name}.prepare-")):
            shutil.rmtree(stage)
    update_status(status, "done", "ACE-Step 1.5 Turbo の準備が完了しました。", model_path=str(output))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACE-Step 1.5 Turbo を Diffusers 形式へ準備")
    parser.add_argument("--model", choices=(MODEL_KEY,), default=MODEL_KEY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--internal-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    if args.internal_job:
        job = _read_json(args.internal_job)
        try:
            conversion_worker(args.internal_job)
            return 0
        except PreparationCancelled:
            return 0
        except Exception as exc:  # noqa: BLE001 - child errors must reach the GUI.
            error = str(exc).strip() or type(exc).__name__
            write_json(Path(job["status_dir"]) / "child_error.json", {"error": error})
            print(error, file=sys.stderr, flush=True)
            return 1
    if not all((args.output_dir, args.cache_dir, args.status_dir)):
        parser.error("--output-dir, --cache-dir, --status-dir が必要です")
    try:
        output = prepare_model(args.model, args.output_dir, args.cache_dir, args.status_dir, args.dtype)
        result, code = {"ok": True, "cancelled": False, "model_path": str(output)}, 0
    except PreparationCancelled as exc:
        update_status(args.status_dir, "cancelled", str(exc))
        result, code = {"ok": False, "cancelled": True, "error": str(exc)}, 0
    except Exception as exc:  # noqa: BLE001 - CLI reports every preparation failure.
        error = str(exc).strip() or type(exc).__name__
        update_status(args.status_dir, "error", error)
        result, code = {"ok": False, "cancelled": False, "error": error}, 1
    write_json(args.status_dir / "result.json", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
