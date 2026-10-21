"""The two models of the FLUX 3 Image Klein experiments, one after the other.

The upsampler writes every caption first and leaves the GPU; only then is Klein loaded, and it
draws them all. References are given to both as RGB over white, as the upsampler was trained.
"""

from __future__ import annotations

import gc
import re
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "services/worker/runtimes/flux3_klein/models"
ORDINALS = {"1": "first", "2": "second", "3": "third", "4": "fourth"}


def on_white(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", image.size, (255, 255, 255, 255)), image).convert("RGB")


def from_qwen_prompt(prompt: str, *, white: bool = True) -> str:
    """A production Qwen prompt as a Klein request: images by ordinal, and white for transparency."""
    lines = [line for line in prompt.splitlines() if "RGBA" not in line and "alpha channel" not in line]
    text = "\n".join(lines + (["The background is plain white."] if white else []))
    return re.sub(r"([Rr])eference image (\d)",
                  lambda match: f"{'tT'[match.group(1) == 'R']}he {ORDINALS[match.group(2)]} image", text)


def write_captions(cases: list[dict]) -> None:
    """Fill each case's ``caption`` from its ``request`` and ``references``."""
    import torch

    from diffusers.modular_pipelines.flux3.prompt_upsampler import prepare_references, upsample_prompt
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    path = MODELS / "flux-3-image-klein-pup"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(path)
    for case in cases:
        started = time.perf_counter()
        case["caption"] = upsample_prompt(model, processor, case["request"],
                                          images=prepare_references(case["references"]))
        case["upsample_seconds"] = round(time.perf_counter() - started, 2)
        print(f"caption {case['name']}: {case['caption'][:100]}...", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_pipeline():
    import torch

    from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel
    from diffusers.modular_pipelines.flux3 import Flux3DistilledAutoBlocks
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    path = MODELS / "flux-3-image-klein"
    blocks = Flux3DistilledAutoBlocks()
    blocks.sub_blocks.pop("prompt_upsampler")  # the captions are already written
    pipe = blocks.init_pipeline()
    pipe.update_components(
        transformer=Flux3Transformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=torch.bfloat16),
        vae=AutoencoderKLFlux2.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(path / "text_encoder", dtype=torch.bfloat16),
        tokenizer=AutoProcessor.from_pretrained(path / "text_encoder"),
    )
    pipe.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    return pipe


def draw(pipe, case: dict, width: int, height: int) -> Image.Image:
    import torch

    started = time.perf_counter()
    (image,) = pipe(prompt=case["caption"], image=case["references"], width=width, height=height,
                    generator=torch.Generator("cpu").manual_seed(case["seed"]), output="images")
    case["draw_seconds"] = round(time.perf_counter() - started, 2)
    print(f"drew {case['name']} ({case['draw_seconds']} s)", flush=True)
    return image.convert("RGB")


def peak_gib() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**30, 2)
