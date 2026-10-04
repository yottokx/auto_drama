"""Shared Qwen inference operations. GPU exclusion belongs to the caller.

Importing this module does not import an inference runtime. The experiment
adapter and the worker's owned child use the same pipeline and parameters.
"""
from __future__ import annotations


def load_pipeline(request: dict, torch):
    from diffusers import QwenImage21Pipeline

    with torch.inference_mode():
        pipe = QwenImage21Pipeline.from_pretrained(
            request["model_path"], dtype=getattr(torch, request["dtype"]),
            local_files_only=True, trust_remote_code=False)
        if request["cpu_offload"]:
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
