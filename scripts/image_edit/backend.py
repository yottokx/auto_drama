"""Shared Qwen inference operations. GPU exclusion belongs to the caller.

Importing this module does not import an inference runtime. The experiment
adapter and the worker's owned child use the same pipeline and parameters.
"""
from __future__ import annotations

import json
from pathlib import Path


def _offload_with_layered_text_encoder(pipe, torch):
    """Keep the peak below the 17 GB text encoder, which runs once per image.

    Its layers visit the GPU one at a time. The transformer and VAE still swap as whole
    components: decoding alone needs several GB, so the transformer has to leave first.
    The pipeline's own offload cannot be combined with layered components.
    """
    from accelerate import cpu_offload_with_hook
    from diffusers.hooks import apply_group_offloading

    device = torch.device("cuda")
    apply_group_offloading(pipe.text_encoder, onload_device=device,
                           offload_device=torch.device("cpu"), offload_type="leaf_level")
    hook = None
    for model in (pipe.transformer, pipe.vae):
        _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)


def load_pipeline(request: dict, torch):
    from diffusers import QwenImage21Pipeline

    with torch.inference_mode():
        pipe = QwenImage21Pipeline.from_pretrained(
            request["model_path"], dtype=getattr(torch, request["dtype"]),
            local_files_only=True, trust_remote_code=False)
        if request.get("transformer_storage") == "fp8":
            # Halve the transformer's 14 GB: weights rest as 8-bit floats and each layer is
            # widened to the compute dtype only while it runs. No extra package or model file.
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=getattr(torch, request["dtype"]))
        if request.get("vae_tiling"):
            # Decoding 1536x1024 in one piece needs about 9 GB beyond the VAE itself, which
            # becomes the peak once the transformer is small. Overlapping tiles are blended.
            pipe.vae.enable_tiling(tile_sample_min_height=768, tile_sample_min_width=768,
                                   tile_sample_stride_height=640, tile_sample_stride_width=640)
        if request["cpu_offload"] and request.get("text_encoder_offload") == "layers":
            _offload_with_layered_text_encoder(pipe, torch)
        elif request["cpu_offload"]:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
    pipe.sample_sigmas = sample_sigmas(request["model_path"])
    return pipe


def sample_sigmas(model_path) -> list[float] | None:
    """The fixed step schedule a few-step checkpoint ships in model_index.json, if it has one.

    The pinned Diffusers predates the pipeline reading it by itself, so it is passed in.
    """
    index = Path(model_path) / "model_index.json"
    value = json.loads(index.read_text(encoding="utf-8")).get("sample_sigmas") if index.is_file() else None
    if value is None:
        return None
    if (not isinstance(value, list) or not value
            or any(type(item) not in (int, float) or not 0 < item <= 1 for item in value)):
        raise ValueError("The checkpoint's step schedule is not a list of sigmas.")
    return [float(item) for item in value]


def render_image(pipe, torch, request: dict, images: list, seed: int, callback):
    sigmas = getattr(pipe, "sample_sigmas", None)
    if sigmas is not None and request["steps"] != len(sigmas):
        raise ValueError(f"This checkpoint runs its own {len(sigmas)}-step schedule.")
    with torch.inference_mode():
        return pipe(
            **({"sigmas": sigmas} if sigmas is not None else {}),
            prompt=request["prompt"], image=images, width=request["width"],
            height=request["height"], num_inference_steps=request["steps"],
            true_cfg_scale=1.0, output_resolution=request["reference_resolution"],
            use_kv_cache=request["use_kv_cache"],
            generator=torch.Generator("cpu").manual_seed(seed), output_type="pil",
            callback_on_step_end=callback, callback_on_step_end_tensor_inputs=[]).images[0]
