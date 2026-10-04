"""Local Diffusers music inference, isolated from the coordinator runtime.

This module intentionally imports only the standard library at module scope. The GUI
and validation tests can import it without installing torch, numpy, or Diffusers.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import secrets
import tempfile
import time
import wave
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .json_io import write_json as atomic_write_json
except ImportError:
    from json_io import write_json as atomic_write_json

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "small": {
        "label": "Small Music (BGM / 最大 120 秒)",
        "repo_id": "stabilityai/stable-audio-3-small-music",
        "max_duration": 120.0,
        "default_steps": 8,
    },
    "medium": {
        "label": "Medium (音楽・音響 / 最大 380 秒)",
        "repo_id": "stabilityai/stable-audio-3-medium",
        "max_duration": 380.0,
        "default_steps": 8,
    },
    "ace15_turbo": {
        "label": "ACE-Step 1.5 Turbo (インスト / 10〜600 秒)",
        "repo_id": "ACE-Step/Ace-Step1.5",
        "dit_config": "acestep-v15-turbo",
        "min_duration": 10.0,
        "max_duration": 600.0,
        "default_steps": 8,
        "sample_rate": 48000,
    },
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
    duration: float = 30.0
    steps: int = 8
    seed: int = -1
    device: str = "auto"
    dtype: str = "auto"
    cpu_offload: bool = False
    context: dict[str, Any] = field(default_factory=dict)
    output_format: str = "mp3"
    keep_wav: bool = False
    mp3_bitrate: int = 192
    bpm: int | None = None
    keyscale: str | None = None
    timesignature: str | None = None
    ace_planner: bool = False
    planner_model_path: str = ""
    ace_continuous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def validate_request(data: Mapping[str, Any]) -> GenerateRequest:
    """Validate UI/JSON settings without importing any inference dependencies."""
    if not isinstance(data, Mapping):
        raise ValueError("生成設定は JSON オブジェクトで指定してください。")  # noqa: TRY004
    model = str(data.get("model", "small")).strip().lower()
    if model not in MODEL_SPECS:
        raise ValueError("モデルは small / medium / ace15_turbo を選択してください。")
    model_path = data.get("model_path", "")
    if not isinstance(model_path, str):
        raise ValueError("変換済みモデルのフォルダーは文字列で指定してください。")  # noqa: TRY004
    model_path = model_path.strip()
    if not model_path:
        raise ValueError("変換済みモデルのフォルダーを指定してください。")
    prompt = data.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValueError("生成プロンプトは文字列で指定してください。")  # noqa: TRY004
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("生成プロンプトを入力してください。")
    duration = _number(data.get("duration", 30), "生成秒数")
    maximum = MODEL_SPECS[model]["max_duration"]
    minimum = MODEL_SPECS[model].get("min_duration", 0.0)
    if minimum and not minimum <= duration <= maximum:
        raise ValueError(f"{model} の生成秒数は {minimum:g}〜{maximum:g} 秒です。")
    if not minimum and not 0 < duration <= maximum:
        raise ValueError(f"{model} の生成秒数は 0 より大きく {maximum:g} 秒以下です。")
    steps = _integer(data.get("steps", 8), "ステップ数")
    if not 1 <= steps <= 1000:
        raise ValueError("ステップ数は 1〜1000 の範囲で指定してください。標準は 8 です。")
    seed = _integer(data.get("seed", -1), "シード")
    if not -1 <= seed <= 2**32 - 1:
        raise ValueError("シードは -1 (ランダム) または 0〜4294967295 です。")
    device = str(data.get("device", "auto")).strip().lower()
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("デバイスは auto / cuda / cpu から選択してください。")
    dtype = str(data.get("dtype", "auto")).strip().lower()
    if dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("精度は auto / float32 / float16 / bfloat16 から選択してください。")
    cpu_offload = data.get("cpu_offload", False)
    if not isinstance(cpu_offload, bool):
        raise ValueError("CPU オフロードは true / false で指定してください。")  # noqa: TRY004
    if device == "cpu" and cpu_offload:
        raise ValueError("CPU オフロードには CUDA GPU が必要です。CPU では無効にしてください。")
    if device == "cpu" and dtype not in {"auto", "float32"}:
        raise ValueError("CPU 生成では float32 を選択してください。")
    output_format = data.get("output_format", "mp3")
    if not isinstance(output_format, str):
        raise ValueError("保存形式は mp3 / wav から選択してください。")  # noqa: TRY004
    output_format = output_format.strip().lower()
    if output_format not in {"mp3", "wav"}:
        raise ValueError("保存形式は mp3 / wav から選択してください。")
    keep_wav = data.get("keep_wav", False)
    if not isinstance(keep_wav, bool):
        raise ValueError("WAV を残す設定は true / false で指定してください。")  # noqa: TRY004
    mp3_bitrate = _integer(data.get("mp3_bitrate", 192), "MP3 ビットレート")
    if mp3_bitrate not in MP3_BITRATES:
        raise ValueError("MP3 ビットレートは 192 / 256 / 320 kbps から選択してください。")
    context = data.get("context", {})
    if not isinstance(context, dict):
        raise ValueError("シーン情報 context は JSON オブジェクトで指定してください。")  # noqa: TRY004
    try:
        # Round-trip also isolates mutable caller-owned nested scene information.
        context = json.loads(json.dumps(context, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("シーン情報 context を JSON に保存できません。") from exc
    ace_planner = data.get("ace_planner", False)
    if not isinstance(ace_planner, bool):
        raise ValueError("専用plannerは true / false で指定してください。")  # noqa: TRY004
    planner_model_path = data.get("planner_model_path", "")
    if not isinstance(planner_model_path, str):
        raise ValueError("plannerモデルのフォルダーは文字列で指定してください。")  # noqa: TRY004
    planner_model_path = planner_model_path.strip()
    if ace_planner and model != "ace15_turbo":
        raise ValueError("専用plannerは ACE-Step 1.5 専用です。")
    if ace_planner and not planner_model_path:
        raise ValueError("専用plannerのモデルフォルダーを指定してください。")
    ace_continuous = data.get("ace_continuous", False)
    if not isinstance(ace_continuous, bool):
        raise ValueError("長い無音を抑える設定は true / false で指定してください。")  # noqa: TRY004
    if ace_continuous and model != "ace15_turbo":
        raise ValueError("連続伴奏の指定は ACE-Step 1.5 専用です。")
    bpm, keyscale, timesignature = None, None, None
    if model == "ace15_turbo":
        try:
            from .ace_backend import validate_instrumental_options
        except ImportError:
            from ace_backend import validate_instrumental_options
        bpm, keyscale, timesignature = validate_instrumental_options(data)
    return GenerateRequest(
        model=model,
        model_path=model_path,
        prompt=prompt,
        duration=duration,
        steps=steps,
        seed=seed,
        device=device,
        dtype=dtype,
        cpu_offload=cpu_offload,
        context=context,
        output_format=output_format,
        keep_wav=keep_wav,
        mp3_bitrate=mp3_bitrate,
        bpm=bpm,
        keyscale=keyscale,
        timesignature=timesignature,
        ace_planner=ace_planner,
        planner_model_path=planner_model_path,
        ace_continuous=ace_continuous,
    )


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
        if model == "ace15_turbo":
            try:
                from .ace_backend import validate_directory
            except ImportError:
                from ace_backend import validate_directory
            return validate_directory(path, index)
        raise AudioGenerationError("指定したモデルは StableAudio3Pipeline 用ではありません。")
    if model == "ace15_turbo":
        raise AudioGenerationError("ACE-Step 1.5 を選択していますが、フォルダーは Stable Audio 3 用です。")
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
    # Stable Audio stays fp32; ACE-Step's official CUDA examples use bf16.
    dtype = (
        "bfloat16" if request.model == "ace15_turbo" and device == "cuda" else "float32"
    ) if request.dtype == "auto" else request.dtype
    if device == "cpu" and dtype != "float32":
        raise AudioGenerationError("CPU 生成では float32 を選択してください。")
    if request.cpu_offload and device != "cuda":
        raise AudioGenerationError("CPU オフロードには CUDA GPU が必要です。")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise AudioGenerationError("この CUDA GPU は bfloat16 に対応していません。float32 を選択してください。")
    return device, dtype, getattr(torch, dtype)


def validate_model_identity(path: Path, requested_model: str) -> dict[str, Any]:
    """Prevent a browsed Medium directory being evaluated as Small, or vice versa."""
    if requested_model == "ace15_turbo":
        try:
            from .ace_backend import validate_identity
        except ImportError:
            from ace_backend import validate_identity
        return validate_identity(path)
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


def explain_error(exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    lowered = message.lower()
    if "out of memory" in lowered:
        return (
            "GPU メモリーが不足しました。Small を選ぶ、生成秒数を短くする、CPU オフロードを有効にする、"
            "または CPU で再試行してください。\n" + message
        )
    if "gated" in lowered or "401" in lowered or "403" in lowered:
        return "モデルの利用申請と Hugging Face のログインを確認してモデル準備を再実行してください。\n" + message
    return message


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
    if request.model == "ace15_turbo":
        try:
            from .ace_backend import load_pipeline as load_ace_pipeline
        except ImportError:
            from ace_backend import load_pipeline as load_ace_pipeline
        return load_ace_pipeline(request, cancelled=cancelled)
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
    from scripts.audio.same_compat import enable_chunked_decode, install_same_compat

    install_same_compat()
    device, dtype_name, dtype = resolve_device_dtype(torch, request)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    from services.worker.generation.music.stable_audio import configure_pipeline

    pipeline = configure_pipeline(diffusers, model_path, request, dtype, device,
        chunked_decode=enable_chunked_decode)
    load_seconds = time.perf_counter() - load_started
    check_cancelled()
    return LoadedPipeline(
        pipeline=pipeline, torch=torch, np=np, device=device, dtype_name=dtype_name,
        model_path=model_path, model_identity=model_identity,
        load_seconds=load_seconds, diffusers_version=version,
    )


def generate_audio(
    request: GenerateRequest | Mapping[str, Any],
    output_dir: Path | str,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Generate one stereo clip using only an already converted local model."""
    if isinstance(request, GenerateRequest):
        request = validate_request(request.to_dict())
    else:
        request = validate_request(request)
    started = time.perf_counter()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    audio_path = directory / f"output.{request.output_format}"
    metadata_path = directory / "generation.json"
    if metadata_path.exists() or any((directory / name).exists() for name in (
        "output.mp3", "output.wav", "output-float.wav",
    )):
        raise AudioGenerationError("出力先に既存の生成結果があります。新しい実行フォルダーを指定してください。")

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise GenerationCancelled("生成を中止しました。")

    def report(phase: str, message: str, *, step: int = 0) -> None:
        if progress is not None:
            progress({
                "phase": phase,
                "message": message,
                "step": step,
                "steps": request.steps,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "updated_at": utc_timestamp(),
            })

    requested_seed = request.seed
    seed = secrets.randbelow(2**32) if request.seed == -1 else request.seed
    planner_result = None
    check_cancelled()
    if request.output_format == "mp3":
        _check_mp3_tools()
    if request.ace_planner:
        from .ace_planner_bridge import run_planner

        planner_result = run_planner(
            replace(request, seed=seed), directory, report=report, cancelled=cancelled,
        )
    check_cancelled()
    report("loading", "ローカルモデルと推論ランタイムを読み込んでいます。")
    loaded = load_pipeline(request, cancelled=cancelled)
    pipeline, torch, np = loaded.pipeline, loaded.torch, loaded.np
    model_path, model_identity = loaded.model_path, loaded.model_identity
    device, dtype_name = loaded.device, loaded.dtype_name
    load_seconds, version = loaded.load_seconds, loaded.diffusers_version
    check_cancelled()
    report("generating", "音楽を生成しています。")

    def on_step_end(pipe: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:
        check_cancelled()
        report("generating", f"音楽を生成しています ({step + 1}/{request.steps})。", step=step + 1)
        return callback_kwargs

    inference_started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(seed)
    sample_rate = int(pipeline.sample_rate) if request.model == "ace15_turbo" else int(
        pipeline.vae.config.sampling_rate
    )
    # SAME injects mask noise through the global RNG, separately from the latent
    # Generator passed to the pipeline. Seed both for repeatable A/B evaluation.
    with torch.random.fork_rng(devices=[0] if device == "cuda" else []), torch.inference_mode():
        torch.manual_seed(seed)
        if request.model == "ace15_turbo":
            try:
                from .ace_backend import infer_audio
            except ImportError:
                from ace_backend import infer_audio
            audios = infer_audio(
                loaded, request, generator=generator, on_step_end=on_step_end,
                check_cancelled=check_cancelled, report=report,
                audio_codes=planner_result["audio_codes"] if planner_result else None,
            )
        else:
            from services.worker.generation.music.stable_audio import decode_latents

            audios = decode_latents(pipeline, request, generator, on_step_end, check_cancelled,
                decoding=lambda: report("decoding", "音声を復元しています。", step=request.steps))
    if device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    check_cancelled()
    report(
        "saving", f"音声を検証し、{request.output_format.upper()} と生成記録を保存しています。",
        step=request.steps,
    )
    stats = _save_audio(
        audios, sample_rate, audio_path, np, keep_wav=request.keep_wav,
        mp3_bitrate=request.mp3_bitrate, cancelled=cancelled,
        analyze_silence=request.model == "ace15_turbo",
    )
    metadata = {
        "created_at": utc_timestamp(),
        "audio_path": str(audio_path),
        "model_repo": model_identity["repo_id"],
        "selected_model_repo": MODEL_SPECS[request.model]["repo_id"],
        "model_identity": model_identity,
        "vae_decode": (
            {"pipeline": "AceStepPipeline", "pipeline_peak_normalization_dbfs": -1.0,
             "duration_cropped": True, "global_rng_seed": seed}
            if request.model == "ace15_turbo" else
            {"chunk_size": 128, "overlap": 32, "global_rng_seed": seed, "clamp_output": False}
        ),
        "model_path": str(model_path),
        "settings": {**request.to_dict(), "seed": seed, "device": device, "dtype": dtype_name},
        "requested_seed": requested_seed,
        "context": request.context,
        **stats,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "versions": {"torch": str(torch.__version__), "diffusers": version, "numpy": str(np.__version__)},
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
    }
    if request.model == "ace15_turbo":
        try:
            from .ace_backend import generation_contract
        except ImportError:
            from ace_backend import generation_contract
        metadata["ace_step"] = generation_contract(request, planner_result=planner_result)
    write_json(metadata_path, metadata)
    report("done", "生成が完了しました。", step=request.steps)
    return {
        "ok": True,
        "audio_path": str(audio_path),
        "float_audio_path": stats.get("float_audio_path"),
        "wav_audio_path": stats.get("wav_audio_path"),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }
