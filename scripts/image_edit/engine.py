"""Offline Qwen Image 2.1 experiments with immutable inputs and partial results.

Importing this module never imports torch, diffusers, or transformers. The GUI
can validate its request without loading an inference runtime.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import io
import json
import os
import re
import secrets
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .backend import load_pipeline, render_image
from .common import MODEL_DIR, MODEL_ID, ROOT, RUNTIME, read_json, write_json
from .resources import GenerationCancelled, StopToken, gpu_lock

MAX_REFERENCE_BYTES = 32 * 1024 * 1024
MAX_REFERENCES = 10
MAX_SEEDS = 8
MAX_SEED = 2**63 - 1
COMPONENTS = ("processor", "scheduler", "text_encoder", "transformer", "vae")
COMPONENT_CLASSES = {
    "processor": ["transformers", "Qwen3VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage21"],
}
REQUIRED_FILES = (
    "model_index.json", "scheduler/scheduler_config.json", "processor/preprocessor_config.json",
    "processor/tokenizer.json", "processor/tokenizer_config.json", "text_encoder/config.json",
    "transformer/config.json", "vae/config.json",
)


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _integer(value, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label}は{minimum}〜{maximum}の整数を指定してください。")
    return value


def validate_request(data: dict) -> dict:
    """Return a normalized JSON request using only lightweight dependencies."""
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data.get("schema_version") != 1):
        raise ValueError("画像編集リクエストの形式が正しくありません。")
    mode = data.get("mode")
    if mode not in {"portrait", "scene"}:
        raise ValueError("立ち絵の差分または一枚絵を選んでください。")
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 30_000:
        raise ValueError("生成指示には1〜30000文字のテキストを入力してください。")
    references = data.get("references")
    if not isinstance(references, list) or not 1 <= len(references) <= MAX_REFERENCES:
        raise ValueError("参照立ち絵は1〜10枚選んでください。")
    if mode == "portrait" and len(references) != 1:
        raise ValueError("立ち絵の差分には基準となる1枚を選んでください。")
    normalized_refs = []
    for index, reference in enumerate(references, 1):
        if not isinstance(reference, dict):
            raise TypeError(f"参照画像{index}の形式が正しくありません。")
        path = reference.get("path")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"参照画像{index}には絶対パスを指定してください。")
        name = reference.get("name", f"参照{index}")
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("参照画像の名前は1〜200文字で指定してください。")
        source = reference.get("source", {})
        if not isinstance(source, dict):
            raise TypeError("参照画像の出典情報はオブジェクトで指定してください。")
        identifier = reference.get("character_id")
        if identifier is not None and (not isinstance(identifier, str) or len(identifier) > 128):
            raise ValueError("参照画像の人物IDが正しくありません。")
        normalized_refs.append({**reference, "path": str(Path(path).resolve()),
                                "name": name, "source": source})
    model_path = data.get("model_path", str(MODEL_DIR))
    if not isinstance(model_path, str) or not Path(model_path).is_absolute():
        raise ValueError("ローカルモデルには絶対パスを指定してください。")
    width = _integer(data.get("width", 768), "幅", 32, 4096)
    height = _integer(data.get("height", 1152), "高さ", 32, 4096)
    if width % 32 or height % 32:
        raise ValueError("出力の幅と高さは32px単位で指定してください。")
    steps = _integer(data.get("steps", 40), "ステップ数", 1, 100)
    seed = _integer(data.get("seed", -1), "シード", -1, MAX_SEED)
    seeds = data.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, list) or not 1 <= len(seeds) <= MAX_SEEDS:
            raise ValueError("比較用シードは1〜8個のリストで指定してください。")
        seeds = [_integer(value, "比較用シード", 0, MAX_SEED) for value in seeds]
    dtype = data.get("dtype", "bfloat16")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("計算精度はbfloat16・float16・float32から選んでください。")
    booleans = {}
    for key, default in (("cpu_offload", True), ("use_kv_cache", True), ("transparent", False)):
        value = data.get(key, default)
        if type(value) is not bool:
            raise ValueError(f"{key}にはtrueまたはfalseを指定してください。")
        booleans[key] = value
    resolution = _integer(data.get("reference_resolution", 1024), "参照解像度", 256, 2048)
    if resolution % 32:
        raise ValueError("参照解像度は32px単位で指定してください。")
    context = data.get("context", {})
    if not isinstance(context, dict):
        raise TypeError("場面情報はオブジェクトで指定してください。")
    result = {**data, "schema_version": 1, "mode": mode, "prompt": prompt,
              "references": normalized_refs, "model_path": str(Path(model_path).resolve()),
              "width": width, "height": height, "steps": steps, "seed": seed,
              "dtype": dtype, "reference_resolution": resolution, "context": context,
              **booleans}
    if seeds is not None:
        result["seeds"] = seeds
    # Reject non-JSON/NaN metadata before a subprocess loads weights.
    json.dumps(result, allow_nan=False)
    return result


def resolved_seeds(request: dict) -> list[int]:
    if request.get("seeds") is not None:
        return list(request["seeds"])
    return [secrets.randbelow(MAX_SEED + 1) if request["seed"] == -1 else request["seed"]]


def digest(path: Path, *, token: StopToken | None = None) -> str:
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            if token is not None:
                token.check()
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_model(model_path: Path, *, token: StopToken | None = None, progress=None) -> dict:
    """Require a complete locally pinned checkpoint, including every shard hash."""
    root = Path(model_path).resolve()
    manifest_path = root / "model-manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("schema_version") != 1 or manifest.get("model_id") != MODEL_ID
            or not re.fullmatch(r"[a-f0-9]{40}", str(manifest.get("revision", "")))):
        raise ValueError("モデルの固定リビジョン記録がありません。先にモデルを準備してください。")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("モデルの検証対象ファイルが記録されていません。")
    names = set()
    for index, record in enumerate(records, 1):
        if token is not None:
            token.check()
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise TypeError("モデルのファイル記録が正しくありません。")
        relative = PurePosixPath(record["path"])
        name = relative.as_posix()
        target = (root / relative).resolve()
        if (not record["path"] or "\\" in record["path"] or ":" in record["path"]
                or relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(root)
                or target.is_symlink() or name in names or name == "model-manifest.json"
                or name.startswith(".cache/")):
            raise ValueError("モデルのファイルパスが正しくありません。")
        names.add(name)
        expected_size = record.get("size_bytes", record.get("size"))
        expected_hash = record.get("sha256")
        if (type(expected_size) is not int or expected_size < 0
                or not isinstance(expected_hash, str)
                or not re.fullmatch(r"[a-f0-9]{64}", expected_hash)):
            raise ValueError("モデルのファイルサイズまたはSHA256記録が正しくありません。")
        if progress:
            progress({"phase": "verifying", "message": f"モデルを検証中: {name}",
                      "step": 0, "steps": 0, "file": index, "files": len(records)})
        if (not target.is_file() or target.stat().st_size != expected_size
                or digest(target, token=token) != expected_hash):
            raise ValueError(f"モデルのファイルが固定記録と一致しません: {name}")
    if any(name not in names for name in REQUIRED_FILES):
        raise ValueError("モデルの必須設定・処理器ファイルが不足しています。")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.relative_to(root).parts[0] != ".cache"
              and path.name != "model-manifest.json"}
    if actual != names:
        raise ValueError("モデルのファイル一覧と固定検証記録が一致しません。")
    for component in ("text_encoder", "transformer", "vae"):
        if not any(name.startswith(component + "/") and name.endswith(".safetensors")
                   for name in names):
            raise ValueError(f"モデルの重みが不足しています: {component}")
    model_index = read_json(root / "model_index.json")
    if model_index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("Qwen Image 2.1の公式パイプライン形式ではありません。")
    if any(model_index.get(name) != definition for name, definition in COMPONENT_CLASSES.items()):
        raise ValueError("公式モデルの構成要素・クラスが一致しません。")
    # Shard index files must not name weights absent from the verified manifest.
    for name in names:
        if name.endswith(".safetensors.index.json"):
            index = read_json(root / name)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"モデルの分割重み索引が正しくありません: {name}")
            for shard in weight_map.values():
                if not isinstance(shard, str):
                    raise TypeError("モデルの分割重み名が正しくありません。")
                shard_name = (Path(name).parent / shard).as_posix()
                if shard_name not in names:
                    raise ValueError(f"未検証の分割重みが指定されています: {shard_name}")
    return {"model_id": MODEL_ID, "revision": manifest["revision"],
            "manifest_sha256": digest(manifest_path), "verified_files": len(names),
            "model_path": str(root)}


def atomic_image(image, destination: Path) -> None:
    """Publish an entire PNG atomically, refusing to replace an existing image."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".image-", suffix=".png",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            image.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link creates a new name atomically and fails when it already exists.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def normalize_references(references: list[dict], output_dir: Path, *, token=None):
    """Freeze source bytes once, honoring EXIF and retaining native alpha/order."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    images, records = [], []
    inputs = Path(output_dir) / "inputs"
    inputs.mkdir(exist_ok=False)
    for index, reference in enumerate(references, 1):
        if token is not None:
            token.check()
        path = Path(reference["path"])
        with path.open("rb") as stream:
            data = stream.read(MAX_REFERENCE_BYTES + 1)
        if not data or len(data) > MAX_REFERENCE_BYTES:
            raise ValueError("参照画像は空でない32MiB以内のファイルを選んでください。")
        try:
            with Image.open(io.BytesIO(data)) as source:
                if (source.format not in {"PNG", "WEBP", "JPEG"}
                        or getattr(source, "is_animated", False)):
                    raise ValueError("参照画像は静止PNG・WebP・JPEGを選んでください。")
                if max(source.size) > 4096:
                    raise ValueError("参照画像の縦横はそれぞれ4096px以内にしてください。")
                source.load()
                source_format = source.format
                image = ImageOps.exif_transpose(source).convert("RGBA")
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError("参照画像を読み取れません。") from exc
        alpha = image.getchannel("A")
        if alpha.getbbox() is None:
            raise ValueError("参照画像に表示できる画素がありません。")
        frozen = inputs / f"reference-{index:02d}.png"
        atomic_image(image, frozen)
        images.append(image)
        records.append({**reference, "order": index, "original_path": str(path),
                        "original_sha256": hashlib.sha256(data).hexdigest(),
                        "original_size_bytes": len(data), "source_format": source_format,
                        "normalized_path": str(frozen.resolve()), "normalized_sha256": digest(frozen),
                        "width": image.width, "height": image.height,
                        "alpha_range": list(alpha.getextrema()), "normalization": "oriented-rgba-png/1"})
    return images, records


def step_callback(token: StopToken, status, *, steps: int, seed: int, batch: int, batches: int):
    def callback(_pipe, index, _timestep, callback_kwargs):
        token.check()
        status({"phase": "generating", "message": f"画像 {batch}/{batches} を生成中",
                "step": index + 1, "steps": steps, "seed": seed,
                "batch": batch, "batches": batches})
        return callback_kwargs
    return callback


def _alpha_metrics(image) -> dict:
    has_alpha = "A" in image.getbands()
    alpha_range = list(image.getchannel("A").getextrema()) if has_alpha else None
    return {"image_mode": image.mode, "has_alpha": has_alpha, "alpha_range": alpha_range,
            "opaque": not has_alpha or alpha_range[0] == 255,
            "fully_transparent": has_alpha and alpha_range[1] == 0}


def _versions() -> dict:
    versions = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _claim_output(output_dir: Path) -> None:
    if not output_dir.is_absolute() or not output_dir.is_dir():
        raise ValueError("出力先には新しい実験フォルダーを作成してください。")
    reserved = ("result.json", "generation.json", "inputs", "outputs", ".engine-started")
    if any((output_dir / name).exists() for name in reserved):
        raise FileExistsError("既存の実験は上書きしません。新しい出力先を指定してください。")
    with (output_dir / ".engine-started").open("xb") as stream:
        stream.write(b"Qwen Image 2.1 experiment\n")


@contextmanager
def _cleanup_before_release(cleanup):
    try:
        yield
    except BaseException as exc:
        import traceback

        # An inference exception retains self/weights in unwound pipeline frames.
        # Preserve the diagnostic first, then clear those references before the
        # CUDA cleanup and before the shared GPU lease can be released.
        traceback.print_exception(exc)
        pending, seen = [exc], set()
        while pending:
            error = pending.pop()
            if id(error) in seen:
                continue
            seen.add(id(error))
            traceback.clear_frames(error.__traceback__)
            pending.extend(value for value in (error.__cause__, error.__context__)
                           if value is not None)
        cleanup()
        raise
    else:
        cleanup()


def generate(request_data: dict, output_dir: Path) -> dict:
    """Execute a seed batch once per seed; preserve completed images on failure."""
    output_dir = Path(output_dir).resolve()
    _claim_output(output_dir)
    started = time.perf_counter()
    token = StopToken(output_dir)
    try:
        json.dumps(request_data, allow_nan=False)
        initial_request = request_data if isinstance(request_data, dict) else {}
    except (TypeError, ValueError):
        initial_request = {"invalid_request": True}
    report = {"schema_version": 1, "created_at": utc_timestamp(), "request": initial_request,
              "outputs": [], "references": [], "status": "starting", "actual_seeds": [],
              "inference_verified": False, "gpu_process_exit_required_for_context_release": True}
    result = {"ok": False, "outputs": [], "cancelled": False}
    pipe = torch = None

    def status(data: dict) -> None:
        write_json(output_dir / "status.json", {**data, "updated_at": utc_timestamp(),
                   "elapsed_seconds": round(time.perf_counter() - started, 3)})

    def publish_partial() -> None:
        write_json(output_dir / "generation.json", report)
        write_json(output_dir / "result.json", {**result, "partial": bool(result["outputs"])})

    def cleanup() -> None:
        nonlocal pipe
        pipe = None
        gc.collect()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001 - preserve results after CUDA failures
                report.setdefault("cleanup_errors", []).append(str(exc))

    try:
        request = validate_request(request_data)
        report["request"] = request
        seeds = resolved_seeds(request)
        report["actual_seeds"] = seeds
        report["requested_dimensions"] = {"width": request["width"], "height": request["height"]}
        token.check()
        status({"phase": "preparing", "message": "参照画像を固定しています。", "step": 0,
                "steps": request["steps"]})
        images, references = normalize_references(request["references"], output_dir, token=token)
        report["references"] = references
        check_started = time.perf_counter()
        report["model"] = verify_model(Path(request["model_path"]), token=token, progress=status)
        report["verification_seconds"] = round(time.perf_counter() - check_started, 3)
        publish_partial()
        status({"phase": "waiting", "message": "他のGPU処理の終了を待っています。", "step": 0,
                "steps": request["steps"]})
        with (gpu_lock(ROOT / "services/worker/cache/m2/gpu.lock", token=token),
              _cleanup_before_release(cleanup)):
            token.check()
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                              HF_HUB_DISABLE_TELEMETRY="1", HF_HOME=str(RUNTIME / "cache/huggingface"))
            import torch as torch_module

            torch = torch_module
            if not torch.cuda.is_available():
                raise RuntimeError("Qwen Image 2.1のローカル生成にはCUDA GPUが必要です。")
            status({"phase": "loading", "message": "Qwen Image 2.1を読み込んでいます。",
                    "step": 0, "steps": request["steps"]})
            load_started = time.perf_counter()
            pipe = load_pipeline(request, torch)
            report.update(library_versions=_versions(), gpu=torch.cuda.get_device_name(),
                          initialization_seconds=round(time.perf_counter() - load_started, 3),
                          pipeline="QwenImage21Pipeline", true_cfg_scale=1.0, negative_prompt=None,
                          reference_resolution_parameter="output_resolution")
            token.check()
            (output_dir / "outputs").mkdir(exist_ok=False)
            for batch, seed in enumerate(seeds, 1):
                token.check()
                torch.cuda.reset_peak_memory_stats()
                generation_started = time.perf_counter()
                status({"phase": "generating", "message": f"画像 {batch}/{len(seeds)} を生成中",
                        "step": 0, "steps": request["steps"], "seed": seed,
                        "batch": batch, "batches": len(seeds)})
                generated = render_image(pipe, torch, request, images, seed,
                    step_callback(token, status, steps=request["steps"],
                        seed=seed, batch=batch, batches=len(seeds)))
                # Never publish the decoded incomplete image after a stop request.
                token.check()
                filename = output_dir / "outputs" / f"image-{batch:02d}-seed-{seed}.png"
                atomic_image(generated, filename)
                metrics = _alpha_metrics(generated)
                warnings = []
                if request["transparent"] and metrics["opaque"]:
                    warnings.append("透明背景の指示に対して不透明な画像が生成されました。")
                if metrics["fully_transparent"]:
                    warnings.append("全画素が透明です。表示できる人物を確認してください。")
                output = {"path": str(filename.resolve()), "seed": seed,
                          "width": generated.width, "height": generated.height,
                          "sha256": digest(filename), "batch": batch,
                          "elapsed_seconds": round(time.perf_counter() - generation_started, 3),
                          "gpu_peak_allocated": torch.cuda.max_memory_allocated(),
                          "gpu_peak_reserved": torch.cuda.max_memory_reserved(),
                          "warnings": warnings, **metrics}
                result["outputs"].append(output)
                report["outputs"].append(output)
                report["status"] = "partial"
                report["inference_verified"] = True
                publish_partial()
            token.check()
            result["ok"] = True
            result["partial"] = False
            report["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 - preserve an experiment and its completed images
        result.update(cancelled=isinstance(exc, GenerationCancelled), error=str(exc),
                      error_type=type(exc).__name__, partial=bool(result["outputs"]))
        report["status"] = "cancelled" if result["cancelled"] else "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if not result["cancelled"]:
            import traceback

            traceback.print_exc()
    finally:
        cleanup()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        report["finished_at"] = utc_timestamp()
        write_json(output_dir / "generation.json", report)
        write_json(output_dir / "result.json", result)
        status({"phase": "complete" if result["ok"] else report["status"],
                "message": "生成が完了しました。" if result["ok"] else result.get("error", "停止しました。"),
                "step": report.get("request", {}).get("steps", 0) if result["ok"] else 0,
                "steps": report.get("request", {}).get("steps", 0),
                "completed_images": len(result["outputs"])})
    return result
