"""ACE-Step 1.5 Turbo through Diffusers, with fixed instrumental conditioning.

The scene LLM supplies a caption and musical metadata. An optional native ACE
planner runs separately and supplies 5 Hz semantic codes to Diffusers.
Heavy dependencies are imported only when loading the locally prepared model.
"""

from __future__ import annotations

import importlib
import json
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .engine import (
    MIN_DIFFUSERS_VERSION,
    MODEL_SPECS,
    AudioGenerationError,
    GenerateRequest,
    GenerationCancelled,
    LoadedPipeline,
    _integer,
    resolve_device_dtype,
    validate_model_directory,
)

MODEL_KEY = "ace15_turbo"
LYRICS = "[Instrumental]"
VOCAL_LANGUAGE = "unknown"
SHIFT = 3.0
CONTINUITY = (
    "Keep any lead melody prominent, with supportive accompaniment through natural phrasing, "
    "brief rests and transitions. Avoid extended silent breaks or premature endings."
)
_STABLE_TAGS = re.compile(
    r"\b(?:TrackType\s*:\s*(?:Music|SFX)|VocalType\s*:\s*(?:Instrumental|Vocals?))\b\s*[,;.]?\s*",
    re.IGNORECASE,
)


def validate_instrumental_options(data: Mapping[str, Any]) -> tuple[int | None, str | None, str | None]:
    """Reject contradictory singer/planner settings rather than ignoring them."""
    if data.get("lyrics", LYRICS) != LYRICS:
        raise ValueError("ACE-Step の歌詞は [Instrumental] 固定です。歌詞やボーカルは指定できません。")
    if data.get("vocal_language", VOCAL_LANGUAGE) != VOCAL_LANGUAGE:
        raise ValueError("ACE-Step のインスト生成では vocal_language は unknown 固定です。")
    if data.get("task_type", "text2music") != "text2music":
        raise ValueError("ACE-Step の比較生成では task_type は text2music 固定です。")
    if data.get("audio_codes") is not None:
        raise ValueError("audio_codes は専用plannerで作成します。直接指定できません。")
    for key in ("planner", "use_planner", "use_lm", "thinking", "use_cot_metas", "use_cot_caption"):
        if key in data and data[key] is not False and data[key] is not None:
            raise ValueError(f"ACE-Step planner は無効です ({key})。")
    for key in ("planner_model", "lm_model_path"):
        if data.get(key):
            raise ValueError(f"ACE-Step planner モデルは使用しません ({key})。")
    bpm = data.get("bpm")
    if bpm is None or bpm == "" or (type(bpm) is int and bpm == 0):
        bpm = None
    else:
        bpm = _integer(bpm, "BPM")
        if not 30 <= bpm <= 300:
            raise ValueError("ACE-Step の BPM は 30〜300 です。")
    keyscale = data.get("keyscale")
    if keyscale is not None:
        if not isinstance(keyscale, str):
            raise ValueError("ACE-Step の調は C major / A minor などの文字列で指定してください。")
        keyscale = keyscale.strip()
        if not keyscale:
            keyscale = None
        elif not re.fullmatch(r"[A-G](?:#|b)? (?:major|minor)", keyscale):
            raise ValueError("ACE-Step の調は C major / A minor などで指定してください。")
    timesignature = data.get("timesignature")
    if timesignature is not None:
        if not isinstance(timesignature, str):
            raise ValueError("ACE-Step の拍子は 2 / 3 / 4 / 6 の文字列で指定してください。")
        timesignature = timesignature.strip() or None
        if timesignature is not None and timesignature not in {"2", "3", "4", "6"}:
            raise ValueError("ACE-Step の拍子は 2 / 3 / 4 / 6 の文字列で指定してください。")
    return bpm, keyscale, timesignature


def instrumental_caption(prompt: str, *, continuous: bool = False) -> str:
    # A comparison may reuse a Stable Audio caption; those two model-specific
    # tags are not ACE conditioning. Preserve genre, instruments, and prose.
    caption = _STABLE_TAGS.sub("", prompt).strip()
    # Keep the scene's musical identity and melody ahead of the silence control.
    # Do not invent a melody for a manually specified nonmelodic texture.
    if continuous:
        separator = " " if caption.endswith((".", "!", "?")) else ". "
        caption += separator + CONTINUITY
    return "Instrumental background music. No vocals, singing, speech, humming, or choir. " + caption


def generation_contract(
    request: GenerateRequest, *, planner_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "pipeline": "AceStepPipeline",
        "variant": MODEL_SPECS[MODEL_KEY]["dit_config"],
        "caption": instrumental_caption(request.prompt, continuous=request.ace_continuous),
        "continuous_accompaniment": request.ace_continuous,
        "lyrics": LYRICS,
        "vocal_language": VOCAL_LANGUAGE,
        "task_type": "cover" if request.ace_planner else "text2music",
        "planner_enabled": request.ace_planner,
        "audio_codes": planner_result["audio_codes"] if planner_result else None,
        "semantic_codec_loaded": request.ace_planner,
        "planner": planner_result if planner_result else None,
        "guidance_scale": 1.0,
        "shift": SHIFT,
        "bpm": request.bpm,
        "keyscale": request.keyscale,
        "timesignature": request.timesignature,
        "pipeline_peak_normalization_dbfs": -1.0,
        "vae_tiling": {"enabled": True, "latent_chunk_size": 512, "latent_overlap": 64},
    }


def validate_directory(path: Path, index: Any) -> Path:
    if not isinstance(index, dict) or index.get("_class_name") != "AceStepPipeline":
        raise AudioGenerationError("指定したモデルは ACE-Step 1.5 の AceStepPipeline 用ではありません。")
    for component in ("transformer", "condition_encoder", "vae", "text_encoder", "tokenizer", "scheduler"):
        if not (path / component).is_dir():
            raise AudioGenerationError(
                f"ACE-Step の変換済みモデルが不完全です ({component} がありません)。モデル準備を再実行してください。"
            )
    return path


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AudioGenerationError(f"{description}を読めません。モデル準備を再実行してください。") from exc
    if not isinstance(value, dict):
        raise AudioGenerationError(f"{description}は JSON オブジェクトで保存してください。")
    return value


def validate_identity(path: Path) -> dict[str, Any]:
    """Require the standard 2B turbo architecture, even when provenance is present."""
    config = _read_object(path / "transformer" / "config.json", "ACE-Step の transformer/config.json")
    signature = tuple(config.get(key) for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "in_channels", "audio_acoustic_hidden_dim",
    ))
    if signature != (2048, 6144, 24, 16, 8, 128, 192, 64) or not (
        config.get("is_turbo") is True or config.get("model_version") == "turbo"
    ):
        raise AudioGenerationError("指定したモデルは標準 ACE-Step 1.5 Turbo (2B) ではありません。")
    manifest_path = path / "ace_preparation.json"
    if not manifest_path.is_file():
        return {"model": MODEL_KEY, "repo_id": None, "verified_by": "transformer/config.json"}
    manifest = _read_object(manifest_path, "ACE-Step の準備記録")
    if manifest.get("model") != MODEL_KEY or manifest.get("repo_id") != MODEL_SPECS[MODEL_KEY]["repo_id"]:
        raise AudioGenerationError("ACE-Step の準備記録と選択モデルの公式 ID が一致しません。")
    if manifest.get("dit_config") != MODEL_SPECS[MODEL_KEY]["dit_config"]:
        raise AudioGenerationError("ACE-Step の準備記録が標準 1.5 Turbo 用ではありません。")
    provenance = {key: manifest[key] for key in (
        "repo_revision", "converter_url", "converter_sha256", "diffusers_version", "dtype", "dit_config",
    ) if key in manifest}
    return {"model": MODEL_KEY, "repo_id": manifest["repo_id"],
            "verified_by": "ace_preparation.json", **provenance}


def load_pipeline(
    request: GenerateRequest, *, cancelled: Callable[[], bool] | None = None,
) -> LoadedPipeline:
    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise GenerationCancelled("生成を中止しました。")

    check_cancelled()
    model_path = validate_model_directory(request.model_path, MODEL_KEY)
    identity = validate_identity(model_path)
    if request.ace_planner:
        for component in ("audio_tokenizer", "audio_token_detokenizer"):
            if not (model_path / component / "config.json").is_file():
                raise AudioGenerationError(
                    f"planner の音声化に必要な {component} がありません。ACEモデル準備を再実行してください。"
                )
    try:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise AudioGenerationError("音楽生成ランタイムが未準備です。環境セットアップを実行してください。") from exc
    version = str(getattr(diffusers, "__version__", "0"))
    parsed = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    version_tuple = tuple(int(item or 0) for item in parsed.groups()) if parsed else (0,)
    if version_tuple < MIN_DIFFUSERS_VERSION or not hasattr(diffusers, "AceStepPipeline"):
        raise AudioGenerationError(
            f"AceStepPipeline には Diffusers 0.40.0 以降が必要です (現在 {version})。"
            "環境セットアップを再実行してください。"
        )
    device, dtype_name, dtype = resolve_device_dtype(torch, request)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    codec_options = {} if request.ace_planner else {"audio_tokenizer": None, "audio_token_detokenizer": None}
    pipeline = diffusers.AceStepPipeline.from_pretrained(
        str(model_path), torch_dtype=dtype, local_files_only=True, use_safetensors=True,
        **codec_options,
    )
    if request.ace_planner and any(getattr(pipeline, name, None) is None for name in (
        "audio_tokenizer", "audio_token_detokenizer",
    )):
        raise AudioGenerationError("専用planner用のsemantic codecを読み込めませんでした。")
    if int(pipeline.sample_rate) != 48000:
        raise AudioGenerationError("ACE-Step のモデルは 48 kHz 音声を出力する必要があります。")
    # AutoencoderOobleck exposes these public attributes in Diffusers 0.40.
    # Long clips otherwise decode in one large allocation despite pipeline comments.
    pipeline.vae.use_tiling = True
    pipeline.vae.tile_latent_min_length = 512
    pipeline.vae.tile_latent_overlap = 64
    pipeline.set_progress_bar_config(disable=True)
    if request.cpu_offload:
        pipeline.enable_model_cpu_offload(gpu_id=0)
    else:
        pipeline.to(device)
    check_cancelled()
    return LoadedPipeline(
        pipeline=pipeline, torch=torch, np=np, device=device, dtype_name=dtype_name,
        model_path=model_path, model_identity=identity, load_seconds=time.perf_counter() - started,
        diffusers_version=version,
    )


def infer_audio(
    loaded: LoadedPipeline,
    request: GenerateRequest,
    *,
    generator: Any,
    on_step_end: Callable[..., dict],
    check_cancelled: Callable[[], None],
    report: Callable[..., None],
    audio_codes: str | None = None,
) -> Any:
    pipeline = loaded.pipeline

    def callback(pipe: Any, step: int, timestep: Any, values: dict) -> dict:
        values = on_step_end(pipe, step, timestep, values)
        if step + 1 == request.steps:
            report("decoding", "ACE-Step の音声を復元しています。", step=request.steps)
        return values

    try:
        if request.ace_planner and not audio_codes:
            raise AudioGenerationError("専用plannerの音楽コードがありません。音声生成を開始できません。")
        if not request.ace_planner and audio_codes is not None:
            raise AudioGenerationError("plannerを無効にした生成に音楽コードは渡せません。")
        output = pipeline(
            prompt=instrumental_caption(request.prompt, continuous=request.ace_continuous), lyrics=LYRICS,
            vocal_language=VOCAL_LANGUAGE, task_type="text2music", audio_codes=audio_codes,
            audio_duration=request.duration, num_inference_steps=request.steps,
            guidance_scale=1.0, shift=SHIFT, bpm=request.bpm, keyscale=request.keyscale,
            timesignature=request.timesignature, generator=generator,
            callback_on_step_end=callback, output_type="pt",
        )
        check_cancelled()
        audio = output.audios
        expected = round(request.duration * pipeline.sample_rate)
        if len(audio.shape) != 3 or audio.shape[0] != 1 or audio.shape[1] != 2:
            raise AudioGenerationError("ACE-Step の出力が stereo の [1, 2, samples] 形式ではありません。")
        if audio.shape[2] < expected:
            raise AudioGenerationError("ACE-Step の出力が指定した生成秒数より短くなっています。")
        return audio[:, :, :expected]
    finally:
        pipeline.maybe_free_model_hooks()
