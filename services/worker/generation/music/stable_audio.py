"""Stable Audio 3 core shared by production worker and experimental GUI.

Inference dependencies are loaded only in the dedicated music process.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import re
import tempfile
import time
import wave
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .json_io import write_json as atomic_write_json

MODEL_SPECS = {
    "small": {"repo_id": "stabilityai/stable-audio-3-small-music", "max_duration": 120.0},
    "medium": {"repo_id": "stabilityai/stable-audio-3-medium", "max_duration": 380.0},
}
MIN_DIFFUSERS_VERSION = (0, 40, 0)
MP3_BITRATES = (192, 256, 320)



class AudioGenerationError(RuntimeError):
    """A generation error with instructions suitable for the test GUI."""


class GenerationCancelled(AudioGenerationError):
    """Raised between inference steps or before saving when cancellation is requested."""


@dataclass(frozen=True)
class GenerateRequest:
    model: str
    model_path: str
    prompt: str
    duration: float = 120.0
    steps: int = 8
    seed: int = 0
    device: str = "cuda"
    dtype: str = "float32"
    cpu_offload: bool = False
    context: dict = field(default_factory=dict)
    output_format: str = "mp3"
    keep_wav: bool = False
    mp3_bitrate: int = 192

    def to_dict(self):
        return asdict(self)


def validate_request(data):
    if not isinstance(data, Mapping):
        raise ValueError("Music settings must be a JSON object.")  # noqa: TRY004
    model = data.get("model", "medium")
    if model not in MODEL_SPECS:
        raise ValueError("Unsupported Stable Audio 3 model.")
    prompt, model_path = data.get("prompt"), data.get("model_path")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(model_path, str) or not model_path:
        raise ValueError("A prompt and prepared local model are required.")
    duration = _number(data.get("duration", 120), "duration")
    steps, seed = _integer(data.get("steps", 8), "steps"), _integer(data.get("seed", 0), "seed")
    if not 0 < duration <= MODEL_SPECS[model]["max_duration"] or not 1 <= steps <= 1000 or not 0 <= seed < 2**32:
        raise ValueError("Invalid music duration, steps or seed.")
    device, dtype = data.get("device", "cuda"), data.get("dtype", "float32")
    if device not in {"auto", "cuda", "cpu"} or dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("Invalid music device or precision.")
    offload = data.get("cpu_offload", False)
    if type(offload) is not bool or device == "cpu" and offload:
        raise ValueError("Invalid music offload setting.")
    context = json.loads(json.dumps(data.get("context", {}), ensure_ascii=False, allow_nan=False))
    return GenerateRequest(model, model_path, prompt.strip(), duration, steps, seed, device, dtype, offload, context,
        data.get("output_format", "mp3"), data.get("keep_wav", False), data.get("mp3_bitrate", 192))



def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} は数値で指定してください。")  # noqa: TRY004 - UI validation contract
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} は数値で指定してください。") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} は有限の数値で指定してください。")
    return result


def _integer(value: Any, name: str) -> int:
    number = _number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} は整数で指定してください。")
    return int(number)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically replace status/result files while the GUI polls them."""
    atomic_write_json(path, data)


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def validate_model_directory(model_path: str, model: str = "small") -> Path:
    """Never fetch weights implicitly; preparation is a separate GUI action."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise AudioGenerationError(
            f"モデルフォルダーがありません: {path}\nGUI の「モデル準備」を実行してください。"
        )
    index_path = path / "model_index.json"
    if not index_path.is_file():
        raise AudioGenerationError(
            f"変換済み Diffusers モデルではありません: {path}\n"
            "元の Hugging Face checkpoint を直接読み込むことはできません。"
            "GUI の「モデル準備」で Stable Audio 3 用の変換を実行してください。"
        )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AudioGenerationError(f"model_index.json を読み込めません: {index_path}") from exc
    if not isinstance(index, dict) or index.get("_class_name") != "StableAudio3Pipeline":
        raise AudioGenerationError("指定したモデルは StableAudio3Pipeline 用ではありません。")
    for component in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler", "duration_embedder"):
        if not (path / component).is_dir():
            raise AudioGenerationError(
                f"変換済みモデルが不完全です ({component} がありません)。モデル準備を再実行してください。"
            )
    return path


def resolve_device_dtype(torch: Any, request: GenerateRequest) -> tuple[str, str, Any]:
    available = bool(torch.cuda.is_available())
    device = "cuda" if request.device == "auto" and available else request.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not available:
        raise AudioGenerationError(
            "CUDA を利用できません。CUDA 対応 PyTorch / NVIDIA ドライバーを確認するか CPU を選択してください。"
        )
    dtype = "float32" if request.dtype == "auto" else request.dtype
    if device == "cpu" and dtype != "float32":
        raise AudioGenerationError("CPU 生成では float32 を選択してください。")
    if request.cpu_offload and device != "cuda":
        raise AudioGenerationError("CPU オフロードには CUDA GPU が必要です。")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise AudioGenerationError("この CUDA GPU は bfloat16 に対応していません。float32 を選択してください。")
    return device, dtype, getattr(torch, dtype)


def validate_model_identity(path: Path, requested_model: str) -> dict[str, Any]:
    """Prevent a browsed Medium directory being evaluated as Small, or vice versa."""
    provenance = {}
    manifest_path = path / "sa3_preparation.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AudioGenerationError("モデルの準備記録を読めません。モデル準備を再実行してください。") from exc
        if not isinstance(manifest, dict) or manifest.get("model") not in MODEL_SPECS:
            raise AudioGenerationError("モデルの準備記録に Small / Medium の識別情報がありません。")
        model = manifest["model"]
        source = "sa3_preparation.json"
        repo = manifest.get("repo_id")
        if repo and repo != MODEL_SPECS[model]["repo_id"]:
            raise AudioGenerationError("モデルの準備記録と公式モデル ID が一致しません。")
        provenance = {key: manifest[key] for key in (
            "repo_revision", "converter_url", "converter_sha256", "same_compat_version",
            "diffusers_version", "dtype"
        ) if key in manifest}
    else:
        try:
            config = json.loads((path / "transformer" / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AudioGenerationError(
                "変換済みモデルの種類を判定できません。transformer/config.json を確認するかモデル準備を実行してください。"
            ) from exc
        if not isinstance(config, dict):
            raise AudioGenerationError("transformer/config.json は JSON オブジェクトで指定してください。")
        signatures = {
            (1024, 20, 16, False): "small",
            (1536, 24, 24, True): "medium",
        }
        signature = tuple(config.get(key) for key in (
            "embed_dim", "depth", "num_heads", "use_differential_attention"
        ))
        try:
            model = signatures.get(signature)
        except TypeError:
            model = None
        if model is None:
            raise AudioGenerationError("指定したモデルは対応する Stable Audio 3 Small / Medium ではありません。")
        source = "transformer/config.json"
        repo = None  # Architecture alone cannot identify the original Hugging Face repository.
    if model != requested_model:
        raise AudioGenerationError(
            f"選択したモデルは {requested_model} ですが、フォルダーは {model} 用です。モデルの選択とパスを合わせてください。"
        )
    return {"model": model, "repo_id": repo, "verified_by": source, **provenance}


def inspect_pcm16_wav(path: Path, expected_sample_rate: int) -> dict[str, Any]:
    """Verify the saved preview independently of NumPy/Torch serialization."""
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        samples = source.getnframes()
        if channels != 2 or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
            raise AudioGenerationError("生成 WAV が stereo PCM16 形式ではありません。")
        if sample_rate != expected_sample_rate or samples <= 0:
            raise AudioGenerationError("生成 WAV のサンプルレートまたは長さが不正です。")
        remaining = samples * channels * 2
        while remaining:
            chunk = source.readframes(min(65536, (remaining + channels * 2 - 1) // (channels * 2)))
            if not chunk:
                raise AudioGenerationError("生成 WAV の音声データが途中で切れています。")
            remaining -= len(chunk)
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_count": samples,
        "duration_seconds": samples / sample_rate,
        "preview_format": "PCM16",
    }


def _audio_files() -> Any:
    try:
        from . import audio_files
    except ImportError:
        import audio_files
    return audio_files


def _check_mp3_tools() -> None:
    files = _audio_files()
    try:
        files.find_ffmpeg()
        files.find_ffprobe()
    except files.AudioFileError as exc:
        raise AudioGenerationError(str(exc)) from exc


def _temporary_audio(destination: Path, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.stem}-", suffix=suffix, delete=False,
    ) as temporary:
        return Path(temporary.name)


def _publish_audio(temporary: Path, destination: Path) -> None:
    # A link publishes the completed file atomically and refuses a destination
    # created by another process between our initial guard and this operation.
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise AudioGenerationError(f"音声ファイルが既にあります: {destination}") from exc


def _save_audio(
    audios: Any,
    sample_rate: int,
    destination: Path,
    np: Any,
    *,
    keep_wav: bool = False,
    mp3_bitrate: int = 192,
    cancelled: Callable[[], bool] | None = None,
    analyze_silence: bool = False,
) -> dict[str, Any]:
    if hasattr(audios, "detach"):
        audios = audios.detach().cpu().float().numpy()
    audio = np.asarray(audios)
    if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] != 2 or audio.shape[2] == 0:
        raise AudioGenerationError("モデルの出力が空か stereo の [1, 2, samples] 形式ではありません。")
    audio = audio[0].T.astype(np.float32)
    if not np.isfinite(audio).all():
        raise AudioGenerationError(
            "生成音声に NaN / Inf が含まれています。float32 を選択して再生成してください。"
        )
    magnitude = np.abs(audio)
    peak = float(np.max(magnitude))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    clipping_fraction = float(np.count_nonzero(magnitude > 1.0) / audio.size)
    stats = {
        "peak": peak,
        "rms": rms,
        "silent": peak < 1e-5,
        "clipping_fraction": clipping_fraction,
        "normalized": False,
    }
    if analyze_silence:
        from .audio_quality import analyze_silence as silence_metrics

        stats["silence_analysis"] = silence_metrics(audio, sample_rate, np=np)
    output_format = destination.suffix.lower().lstrip(".")
    if output_format not in {"mp3", "wav"}:
        raise AudioGenerationError("保存先の拡張子は .mp3 / .wav にしてください。")
    # Explicit WAV output preserves the established PCM16 + raw FLOAT32 pair.
    retain_wav = output_format == "wav" or keep_wav
    pcm_destination = destination.with_suffix(".wav")
    float_destination = destination.with_name(f"{destination.stem}-float.wav")
    outputs = [destination]
    if retain_wav:
        outputs.extend((pcm_destination, float_destination))
    if any(path.exists() for path in outputs):
        raise AudioGenerationError("出力先に既存の音声があります。新しい実行フォルダーを指定してください。")

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise GenerationCancelled("音声の保存を中止しました。")

    check_cancelled()
    # Preserve low-level outputs, but attenuate over-range previews uniformly instead
    # of distorting their peaks. The raw float WAV and its quality metrics stay intact.
    preview_gain = 0.99 / peak if peak > 1.0 else 1.0
    pcm = np.rint(np.clip(audio * preview_gain, -1.0, 1.0) * 32767).astype("<i2")
    stats.update({
        "preview_gain": preview_gain,
        "preview_peak": float(np.max(np.abs(pcm))) / 32767,
        "preview_normalized": preview_gain < 1.0,
        "preview_normalization": (
            "ピークが 1.0 を超える場合のみ 0.99 まで一律に減衰します。"
            "自動増幅は行わず、原音の品質指標は変更しません。"
            "原音 FLOAT32 WAV は WAV 保存を選択した場合のみ残します。"
        ),
    })
    temporary = _temporary_audio(pcm_destination, ".wav")
    float_temporary: Path | None = None
    published: list[Path] = []
    try:
        with wave.open(str(temporary), "wb") as target:
            target.setnchannels(2)
            target.setsampwidth(2)
            target.setframerate(sample_rate)
            target.writeframes(pcm.tobytes())
        stats.update(inspect_pcm16_wav(temporary, sample_rate))
        stats.update({
            "source_sample_count": stats["sample_count"],
            "source_duration_seconds": stats["duration_seconds"],
            "raw_audio_retained": retain_wav,
        })
        if retain_wav:
            # Values above 1.0 survive in the optional original waveform.
            sf = importlib.import_module("soundfile")
            float_temporary = _temporary_audio(float_destination, ".wav")
            sf.write(str(float_temporary), audio, sample_rate, subtype="FLOAT", format="WAV")
            float_info = sf.info(str(float_temporary))
            if (
                float_info.channels != 2
                or float_info.frames != audio.shape[0]
                or float_info.samplerate != sample_rate
                or float_info.subtype != "FLOAT"
            ):
                raise AudioGenerationError("原音の float32 WAV を保存できませんでした。")
        check_cancelled()
        if output_format == "mp3":
            files = _audio_files()
            try:
                encoded = files.encode_mp3(
                    temporary, destination, bitrate_kbps=mp3_bitrate,
                    sample_rate=sample_rate, cancelled=cancelled,
                )
            except files.ExportCancelled as exc:
                raise GenerationCancelled(str(exc)) from exc
            except files.AudioFileError as exc:
                raise AudioGenerationError(str(exc)) from exc
            published.append(destination)
            stats.update(encoded)
            stats["preview_format"] = "MP3"
        if retain_wav:
            assert float_temporary is not None
            _publish_audio(float_temporary, float_destination)
            published.append(float_destination)
            _publish_audio(temporary, pcm_destination)
            published.append(pcm_destination)
            stats.update({
                "float_audio_path": str(float_destination), "float_format": "FLOAT32",
                "wav_audio_path": str(pcm_destination),
            })
        if output_format == "wav":
            stats["codec"] = "pcm_s16le"
        stats["audio_bytes"] = destination.stat().st_size
        stats["audio_path"] = str(destination)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        if float_temporary is not None:
            float_temporary.unlink(missing_ok=True)
    return stats


@dataclass(frozen=True)
class LoadedPipeline:
    """One prepared model instance shared by generation and experimental inpainting."""

    pipeline: Any
    torch: Any
    np: Any
    device: str
    dtype_name: str
    model_path: Path
    model_identity: dict[str, Any]
    load_seconds: float
    diffusers_version: str


def infer_pcm(loaded, request, *, cancelled=None, progress=None):
    """Preserve the GUI's tested latent decode, global RNG and SAME settings."""
    def check():
        if cancelled is not None and cancelled():
            raise GenerationCancelled("生成を中止しました。")

    pipeline, torch = loaded.pipeline, loaded.torch
    started = time.perf_counter()
    check()
    if loaded.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device=loaded.device).manual_seed(request.seed)
    sample_rate = int(pipeline.vae.config.sampling_rate)

    def on_step_end(pipe, step, timestep, callback_kwargs):
        check()
        if progress:
            progress(step + 1)
        return callback_kwargs

    with torch.random.fork_rng(devices=[0] if loaded.device == "cuda" else []), torch.inference_mode():
        torch.manual_seed(request.seed)
        audios = decode_latents(pipeline, request, generator, on_step_end, check)
        audio = audios.detach().cpu().float().numpy()[0].T.copy()
    if loaded.device == "cuda":
        torch.cuda.synchronize()
    check()
    return audio, sample_rate, {"inference_seconds": time.perf_counter() - started,
        "load_seconds": loaded.load_seconds,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if loaded.device == "cuda" else None}


def decode_latents(pipeline, request, generator, on_step_end, check_cancelled, *, decoding=None):
    """Shared SA3 inference: no pipeline clamping, padding or changed CFG."""
    output = pipeline(prompt=request.prompt, duration=request.duration,
        num_inference_steps=request.steps, guidance_scale=1.0, silence_padding_duration=0.0,
        generator=generator, callback_on_step_end=on_step_end, output_type="latent")
    check_cancelled()
    if decoding:
        decoding()
    try:
        return pipeline.vae.decode(output.audios).sample[:, :, :int(request.duration * pipeline.vae.config.sampling_rate)]
    finally:
        pipeline.maybe_free_model_hooks()


def configure_pipeline(diffusers, model_path, request, dtype, device, *, chunked_decode=None):
    if chunked_decode is None:
        from .same_compat import enable_chunked_decode

        chunked_decode = enable_chunked_decode
    pipeline = diffusers.StableAudio3Pipeline.from_pretrained(str(model_path),
        torch_dtype=dtype, local_files_only=True, use_safetensors=True)
    chunked_decode(pipeline.vae, chunk_size=128, overlap=32)
    pipeline.set_progress_bar_config(disable=True)
    if request.cpu_offload:
        pipeline.enable_model_cpu_offload(gpu_id=0)
    else:
        pipeline.to(device)
    return pipeline


def load_pipeline(
    request: GenerateRequest | Mapping[str, Any],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> LoadedPipeline:
    """Load one local model with the same identity, precision, and decode settings.

    The caller owns the GPU lease and process lifecycle. No network download or
    second model is started here; an inpaint pipeline can reuse the components.
    """
    request = validate_request(request.to_dict() if isinstance(request, GenerateRequest) else request)

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise GenerationCancelled("生成を中止しました。")

    check_cancelled()
    model_path = validate_model_directory(request.model_path, request.model)
    model_identity = validate_model_identity(model_path, request.model)
    try:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise AudioGenerationError(
            "Stable Audio 3 専用ランタイムが未準備です。GUI の「環境セットアップ」を実行してください。\n"
            + str(exc)
        ) from exc
    version = str(getattr(diffusers, "__version__", "0"))
    parsed_version = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    version_tuple = tuple(int(item or 0) for item in parsed_version.groups()) if parsed_version else (0,)
    if version_tuple < MIN_DIFFUSERS_VERSION or not hasattr(diffusers, "StableAudio3Pipeline"):
        raise AudioGenerationError(
            f"StableAudio3Pipeline には Diffusers 0.40.0 以降が必要です (現在 {version})。"
            "GUI の環境セットアップを再実行してください。"
        )
    from .same_compat import enable_chunked_decode, install_same_compat

    install_same_compat()
    device, dtype_name, dtype = resolve_device_dtype(torch, request)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    pipeline = configure_pipeline(diffusers, model_path, request, dtype, device,
        chunked_decode=enable_chunked_decode)
    load_seconds = time.perf_counter() - load_started
    check_cancelled()
    return LoadedPipeline(
        pipeline=pipeline, torch=torch, np=np, device=device, dtype_name=dtype_name,
        model_path=model_path, model_identity=model_identity,
        load_seconds=load_seconds, diffusers_version=version,
    )
