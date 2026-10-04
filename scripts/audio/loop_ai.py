"""Bounded Stable Audio 3 inpainting for one circular PCM boundary."""

from __future__ import annotations

import importlib

from .engine import AudioGenerationError, load_pipeline, validate_request
from .loops import check_cancelled, fade_weights, validate_pcm


def bridge_reference(audio, sample_rate: int, context_seconds: float, repair_seconds: float, np):
    context = min(round(context_seconds * sample_rate), len(audio))
    if context < 10 * sample_rate:
        raise AudioGenerationError("AI継ぎ目修復には10秒以上の元音声が必要です。")
    half = min(round(repair_seconds * sample_rate / 2), context // 3, (len(audio) - 4) // 4)
    reference = np.concatenate((audio[-context:], audio[:context])).astype(np.float32)
    return reference, context, context - half, context + half


def apply_bridge(audio, reference, generated, center: int, start: int, end: int,
                 shoulder: int, np):
    """Copy generated audio only across the hole and blended shoulders."""
    if generated.shape != reference.shape or not np.isfinite(generated).all():
        raise AudioGenerationError("AI修復出力の長さ・形式または数値が不正です。")
    shoulder = min(shoulder, start, len(reference) - end, (len(audio) - (end - start)) // 2)
    if shoulder < 2:
        raise AudioGenerationError("AI修復の両肩をブレンドする余白がありません。")
    left, right = start - shoulder, end + shoulder
    weights = np.ones((right - left, 1), dtype=np.float32)
    weights[:shoulder] = fade_weights(shoulder, np)
    weights[-shoulder:] = 1 - fade_weights(shoulder, np)
    repaired = reference[left:right] * (1 - weights) + generated[left:right] * weights
    # The reference is tail+head. A modulo mapping preserves the original period
    # and leaves every PCM sample outside this bounded area byte-for-byte intact.
    indices = (np.arange(left, right) - center) % len(audio)
    output = audio.copy()
    output[indices] = repaired
    return output, {
        "repair_start_sample": int(start), "repair_end_sample": int(end),
        "repair_coordinate_system": "tail_plus_head_reference",
        "edited_sample_count": int(right - left), "shoulder_samples": int(shoulder),
        "outside_pcm_preserved": True, "preservation_scope": "decoded_source_pcm_before_encoding",
        "preservation_note": "修復域と両肩以外は入力PCMを保持します。冒頭は一度再生する別区間として元PCMを保持します。MP3圧縮後のサンプル一致は保証しません。",
    }


def repair_bridge(audio, request, sample_rate: int, *, seed: int, progress=None, cancelled=None):
    np = importlib.import_module("numpy")
    audio = validate_pcm(audio, np)
    reference, center, start, end = bridge_reference(
        audio, sample_rate, request.bridge_context_seconds, request.bridge_seconds, np)
    prompt = (request.source_generation.get("settings") or {}).get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise AudioGenerationError("AI継ぎ目修復には元音声の保存済み英語プロンプトが必要です。")
    settings = validate_request({
        "model": request.model, "model_path": request.model_path, "prompt": prompt,
        "duration": len(reference) / sample_rate, "steps": request.steps, "seed": seed,
        "device": request.device, "dtype": request.dtype, "cpu_offload": request.cpu_offload,
    })
    check_cancelled(cancelled)
    if progress:
        progress({"phase": "loading", "message": "AI継ぎ目修復用モデルを読み込んでいます。"})
    loaded = load_pipeline(settings, cancelled=cancelled)
    torch = loaded.torch
    components = loaded.pipeline.components
    transformer = components["transformer"]
    latent_dim = int(components["vae"].config.latent_dim)
    config = transformer.config
    if int(getattr(config, "local_add_cond_dim", 0)) != latent_dim + 1 or int(getattr(config, "patch_size", 1)) != 1:
        raise AudioGenerationError("このモデルの条件付けはStable Audio 3 Inpaintに対応していません。")
    diffusers = importlib.import_module("diffusers")
    if not hasattr(diffusers, "StableAudio3InpaintPipeline"):
        raise AudioGenerationError("専用DiffusersにStableAudio3InpaintPipelineがありません。")
    pipeline = diffusers.StableAudio3InpaintPipeline(**components)
    pipeline.set_progress_bar_config(disable=True)
    if request.cpu_offload:
        pipeline.enable_model_cpu_offload(gpu_id=0)

    def on_step_end(pipe, step, timestep, callback_kwargs):
        check_cancelled(cancelled)
        if progress:
            progress({"phase": "generating", "message": f"継ぎ目を修復しています ({step + 1}/{request.steps})。",
                      "step": step + 1, "steps": request.steps})
        return callback_kwargs

    source_tensor = torch.from_numpy(reference.T.copy()[None])
    generator = torch.Generator(device=loaded.device).manual_seed(seed)
    try:
        with torch.random.fork_rng(devices=[0] if loaded.device == "cuda" else []), torch.inference_mode():
            torch.manual_seed(seed)
            result = pipeline(
                prompt=prompt, audio=source_tensor, duration=len(reference) / sample_rate,
                mask_start_seconds=start / sample_rate, mask_end_seconds=end / sample_rate,
                num_inference_steps=request.steps, silence_padding_duration=0.0,
                generator=generator, callback_on_step_end=on_step_end, output_type="latent",
            )
            check_cancelled(cancelled)
            generated = pipeline.vae.decode(result.audios).sample[:, :, :len(reference)]
            generated = generated.detach().cpu().float().numpy()[0].T
    finally:
        pipeline.maybe_free_model_hooks()
        loaded.pipeline.maybe_free_model_hooks()
    check_cancelled(cancelled)
    shoulder = max(2, round(min(request.crossfade_seconds, 2.0) * sample_rate))
    output, info = apply_bridge(audio, reference, generated, center, start, end, shoulder, np)
    info.update({
        "source_start_seconds": 0.0, "source_end_seconds": len(audio) / sample_rate,
        "repair_start_seconds": start / sample_rate, "repair_end_seconds": end / sample_rate,
        "reference_duration_seconds": len(reference) / sample_rate,
        "context_each_side_seconds": center / sample_rate,
        "repair_seconds": (end - start) / sample_rate, "crossfade_seconds": info["shoulder_samples"] / sample_rate,
        "overlap_shortening_seconds": 0.0, "period_preserved": True,
        "model_identity": loaded.model_identity, "load_seconds": loaded.load_seconds,
        "device": loaded.device, "dtype": loaded.dtype_name, "seed": seed,
        "saved_prompt_reused": True, "confidence": "unrated",
        "quality_flags": ["experimental_ai_bridge", "musical_loop_unrated"],
    })
    return output, info
