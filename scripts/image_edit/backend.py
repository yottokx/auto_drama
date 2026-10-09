"""Shared Qwen inference operations. GPU exclusion belongs to the caller.

Importing this module does not import an inference runtime. The experiment
adapter and the worker's owned child use the same pipeline and parameters.
"""
from __future__ import annotations


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
    return pipe


def render_image(pipe, torch, request: dict, images: list, seed: int, callback):
    with torch.inference_mode():
        return pipe(
            prompt=request["prompt"], image=images, width=request["width"],
            height=request["height"], num_inference_steps=request["steps"],
            true_cfg_scale=1.0, output_resolution=request["reference_resolution"],
            use_kv_cache=request["use_kv_cache"],
            generator=torch.Generator("cpu").manual_seed(seed), output_type="pil",
            callback_on_step_end=callback, callback_on_step_end_tensor_inputs=[]).images[0]
