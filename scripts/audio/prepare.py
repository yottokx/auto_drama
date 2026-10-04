"""Download official SA3 checkpoints and atomically prepare a Diffusers model.

Run with the dedicated Stable Audio 3 runtime, not the coordinator environment.
HF authentication comes only from HF_TOKEN or `hf auth login`; tokens are never
written to manifests, job files or status files. The two gated repositories
require the user to accept their model terms on Hugging Face first.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

try:
    from .json_io import write_json as atomic_write_json
    from .same_compat import install_same_compat, same_config_fields
except ImportError:
    from json_io import write_json as atomic_write_json
    from same_compat import install_same_compat, same_config_fields


MODEL_REPOS = {
    "small": "stabilityai/stable-audio-3-small-music",
    "medium": "stabilityai/stable-audio-3-medium",
}
CONVERTER_VERSION = "v0.40.0"
CONVERTER_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/"
    "v0.40.0/scripts/convert_stable_audio_3_to_diffusers.py"
)
CONVERTER_SHA256 = "ee782fa2dfacc637251d00add7cee93f6c3aafcbed6abf111ff209947d98267f"
COMPONENTS = ("vae", "duration_embedder", "transformer", "text_encoder", "tokenizer", "scheduler")


class PreparationCancelled(Exception):
    pass


def write_json(path: Path, value: dict):
    atomic_write_json(path, value)


def update_status(status_dir: Path, phase: str, message: str, **details):
    write_json(status_dir / "status.json", {"phase": phase, "message": message, **details})
    print(message, flush=True)


def check_cancel(status_dir: Path):
    if (status_dir / "stop.request").exists():
        raise PreparationCancelled("モデルの準備を中止しました。")


def is_complete_model(path: Path) -> bool:
    """Reject partial converter output even if model_index.json already exists."""
    try:
        index = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
        if index.get("_class_name") != "StableAudio3Pipeline":
            return False
        if any(not isinstance(index.get(component), list) or len(index[component]) != 2
               for component in COMPONENTS):
            return False
        if any(not (path / component).is_dir() for component in COMPONENTS):
            return False
        for component in ("vae", "duration_embedder", "transformer", "text_encoder"):
            folder = path / component
            if not (folder / "config.json").is_file() or not list(folder.glob("*.safetensors")):
                return False
        return ((path / "scheduler" / "scheduler_config.json").is_file()
                and (path / "tokenizer" / "tokenizer_config.json").is_file())
    except (OSError, ValueError, TypeError):
        return False


def fetch_converter(cache_dir: Path) -> Path:
    """Only execute the reviewed release file if its pinned digest matches."""
    directory = cache_dir / "converter"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"convert_stable_audio_3_{CONVERTER_VERSION}.py"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == CONVERTER_SHA256:
        return path
    with urllib.request.urlopen(CONVERTER_URL, timeout=30) as response:
        source = response.read(2_000_000)
    if hashlib.sha256(source).hexdigest() != CONVERTER_SHA256:
        raise ValueError("公式 Diffusers 変換スクリプトのチェックサムが一致しません。")
    temporary = path.with_suffix(".download")
    temporary.write_bytes(source)
    os.replace(temporary, path)
    return path


def cached_repo_revision(config_path: Path) -> str:
    """hf_hub_download returns snapshots/<commit>/model_config.json."""
    revision = config_path.parent.name
    if config_path.parent.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("公式モデルのキャッシュからリビジョンを確定できませんでした。")
    return revision


def run_child(job: Path, status_dir: Path):
    """Wait responsively and reap the conversion child on cancellation/error."""
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
            details_path = status_dir / "child_error.json"
            if details_path.is_file():
                details = json.loads(details_path.read_text(encoding="utf-8"))
                if details.get("error"):
                    raise RuntimeError(details["error"])
            raise RuntimeError(f"モデル変換に失敗しました (終了コード {process.returncode})。ログを確認してください。")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def _validate_reference_config(config: dict, model: str):
    if config.get("sample_rate") != 44100 or config.get("audio_channels") != 2:
        raise ValueError("Unexpected Stable Audio 3 audio format")
    diffusion = config["model"]["diffusion"]
    if diffusion.get("diffusion_objective") != "rf_denoiser":
        raise ValueError("Expected the post-trained SA3 checkpoint, not a base checkpoint")
    actual = diffusion["config"]
    expected = {"small": (1024, 20, False), "medium": (1536, 24, True)}[model]
    if (actual["embed_dim"], actual["depth"], actual["attn_kwargs"]["differential"]) != expected:
        raise ValueError("Unexpected Stable Audio 3 DiT architecture")
    fields = same_config_fields(config)
    if model == "small" and (
        fields["decoder_chunk_size"] != 32
        or not fields["decoder_chunk_midpoint_shift"]
        or fields["decoder_mapping_kernel"] != 3
    ):
        raise ValueError("Unexpected Stable Audio 3 Small SAME architecture")


def _validated_same_state_dict(model, state_dict: dict) -> dict:
    """Drop only SAME RoPE buffers reproduced exactly by the native model.

    The reference persists each transformer's ``rope.inv_freq``. Diffusers
    nests that module under ``attn`` and recomputes its nonpersistent buffer.
    Validate the frequencies before removing those extra checkpoint keys;
    all learned weights and any unrecognized keys still require strict load.
    DiT's separate ``rotary_pos_emb.inv_freq`` is persistent and is retained.
    """
    import torch

    result = state_dict.copy()
    persistent = set(model.state_dict())
    pattern = r"((?:encoder|decoder)\.blocks\.\d+\.transformers\.\d+)\.attn\.rope\.inv_freq"
    for native_key, expected in model.named_buffers():
        match = re.fullmatch(pattern, native_key)
        if match is None:
            continue
        key = match[1] + ".rope.inv_freq"
        if key not in state_dict or native_key in persistent:
            continue
        value = state_dict[key]
        expected = expected.to(device=value.device, dtype=value.dtype)
        if (value.shape != expected.shape or not value.is_floating_point()
                or not torch.allclose(value, expected, rtol=1e-5, atol=1e-7)):
            raise ValueError(f"SAME rotary frequencies differ from the native architecture: {key}")
        del result[key]
    return result


def conversion_worker(job_path: Path):
    """Load the fixed converter in memory and add missing SAME-S config fields."""
    job = json.loads(job_path.read_text(encoding="utf-8"))
    import diffusers
    from huggingface_hub import hf_hub_download, snapshot_download

    if diffusers.__version__ != "0.40.0":
        raise RuntimeError("モデル準備には diffusers==0.40.0 が必要です。GUI の環境準備を実行してください。")
    status = Path(job["status_dir"])
    converter_path = fetch_converter(Path(job["cache_dir"]))
    check_cancel(status)
    repo = MODEL_REPOS[job["model"]]
    cache = str(Path(job["cache_dir"]) / "huggingface")
    update_status(status, "preparing", "公式モデルの設定を取得しています。")
    config_path = hf_hub_download(repo, "model_config.json", cache_dir=cache)
    revision = cached_repo_revision(Path(config_path))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _validate_reference_config(config, job["model"])
    check_cancel(status)
    update_status(status, "preparing", "公式の音声モデルをダウンロードしています。初回は時間がかかります。")
    checkpoint = hf_hub_download(repo, "model.safetensors", revision=revision, cache_dir=cache)
    check_cancel(status)
    update_status(status, "preparing", "同じ公式リポジトリの T5Gemma を取得しています。")
    snapshot = snapshot_download(
        repo, revision=revision, allow_patterns=["t5gemma-b-b-ul2/*"], cache_dir=cache
    )
    encoder_path = Path(snapshot) / "t5gemma-b-b-ul2"
    if not (encoder_path / "config.json").is_file():
        raise RuntimeError("公式モデルに同梱された T5Gemma が見つかりません。")
    check_cancel(status)
    install_same_compat()
    spec = importlib.util.spec_from_file_location("sa3_official_converter", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    original_infer = converter._infer_vae_config

    def infer_compatible_vae(weights, model_config=None):
        if model_config is None:
            raise ValueError("SAME conversion requires the original model_config.json")
        return {**original_infer(weights, model_config), **same_config_fields(model_config)}

    converter._infer_vae_config = infer_compatible_vae
    # Fail instead of publishing a partially initialized component. The upstream
    # converter otherwise logs missing/unexpected keys and continues silently.
    for name in ("AutoencoderSAME", "StableAudio3DiTModel", "StableAudio3DurationEmbedder"):
        model_class = getattr(diffusers, name)
        original_loader = model_class.load_state_dict

        def strict_loader(self, state_dict, strict=True, assign=False,
                          _loader=original_loader, _name=name):
            if _name == "AutoencoderSAME":
                state_dict = _validated_same_state_dict(self, state_dict)
            return _loader(self, state_dict, strict=True, assign=assign)

        model_class.load_state_dict = strict_loader
    update_status(status, "preparing", "Small / Medium の構造を保持して Diffusers 形式へ変換しています。")
    converter.convert(SimpleNamespace(
        checkpoint_path=checkpoint, model_config_path=config_path,
        output_dir=job["stage_dir"], text_encoder_repo=str(encoder_path),
        dtype=job["dtype"], skip_sanity_check=True,
    ))
    check_cancel(status)
    output = Path(job["stage_dir"])
    if not is_complete_model(output):
        raise RuntimeError("変換されたモデルの構成ファイルが不足しています。")
    scheduler = json.loads((output / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
    if not scheduler.get("stochastic_sampling"):
        raise RuntimeError("SA3 post-trained の ping-pong スケジューラーが設定されていません。")
    write_json(output / "sa3_preparation.json", {
        "model": job["model"], "repo_id": repo, "repo_revision": revision,
        "converter_url": CONVERTER_URL,
        "converter_sha256": CONVERTER_SHA256, "diffusers_version": "0.40.0",
        "dtype": job["dtype"], "same_compat_version": 1,
        "reference_config": config,
    })


def prepare_model(model: str, output_dir: Path, cache_dir: Path, status_dir: Path, dtype="float32"):
    if model == "ace15_turbo":
        try:
            from .ace_prepare import PreparationCancelled as AcePreparationCancelled
            from .ace_prepare import prepare_model as prepare_ace_model
        except ImportError:
            from ace_prepare import PreparationCancelled as AcePreparationCancelled
            from ace_prepare import prepare_model as prepare_ace_model
        try:
            return prepare_ace_model(model, output_dir, cache_dir, status_dir, dtype)
        except AcePreparationCancelled as exc:
            # A direct prepare.py launch loads this file as __main__, while
            # ACE imports it as prepare. Normalize the cancellation class.
            raise PreparationCancelled(str(exc)) from exc
    output = output_dir.resolve()
    cache = cache_dir.resolve()
    status = status_dir.resolve()
    update_status(status, "preparing", "Stable Audio 3 のモデルを準備しています。")
    check_cancel(status)
    if is_complete_model(output):
        manifest = output / "sa3_preparation.json"
        if manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("model") != model:
            raise ValueError("選択したモデルと準備済みフォルダーの Small / Medium が一致しません。")
        update_status(status, "done", "準備済みのモデルを使用します。", model_path=str(output))
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
        if not is_complete_model(stage):
            raise RuntimeError("準備の検証に失敗しました。既存のモデルは変更されていません。")
        if output.exists():
            if any(output.iterdir()):
                raise ValueError("モデルの準備中に出力先が変更されました。")
            output.rmdir()  # An empty directory only; never overwrite a model.
        stage.rename(output)
    finally:
        # stage is the verified direct-child directory created by mkdtemp above.
        # Do not recursively delete anything supplied through the command line.
        if stage.exists() and stage.parent == output.parent and stage.name.startswith(f".{output.name}.prepare-"):
            shutil.rmtree(stage)
    update_status(status, "done", "モデルの準備が完了しました。", model_path=str(output))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="音楽モデルを Diffusers 形式へ準備")
    parser.add_argument("--model", choices=(*MODEL_REPOS, "ace15_turbo"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--internal-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    if args.internal_job:
        job = json.loads(args.internal_job.read_text(encoding="utf-8"))
        try:
            conversion_worker(args.internal_job)
            return 0
        except PreparationCancelled:
            return 0
        except Exception as exc:  # noqa: BLE001 - preserve child failures for the parent GUI.
            error = str(exc).strip() or type(exc).__name__
            if any(marker in error.lower() for marker in ("gated", "401", "403")):
                repo = MODEL_REPOS[job["model"]]
                error = (
                    f"https://huggingface.co/{repo} で利用条件に同意し、同じアカウントで "
                    "HF_TOKEN または専用環境の hf auth login を設定して再実行してください。\n" + error
                )
            write_json(Path(job["status_dir"]) / "child_error.json", {"error": error})
            print(error, file=sys.stderr, flush=True)
            return 1
    if not all((args.model, args.output_dir, args.cache_dir, args.status_dir)):
        parser.error("--model, --output-dir, --cache-dir, --status-dir が必要です")
    try:
        output = prepare_model(args.model, args.output_dir, args.cache_dir, args.status_dir, args.dtype)
        result = {"ok": True, "cancelled": False, "model_path": str(output)}
        code = 0
    except PreparationCancelled as exc:
        update_status(args.status_dir, "cancelled", str(exc))
        result = {"ok": False, "cancelled": True, "error": str(exc)}
        code = 0
    except Exception as exc:  # noqa: BLE001 - CLI must report any preparation failure to the GUI.
        error = str(exc)
        update_status(args.status_dir, "error", error)
        result = {"ok": False, "cancelled": False, "error": error}
        code = 1
    write_json(args.status_dir / "result.json", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
