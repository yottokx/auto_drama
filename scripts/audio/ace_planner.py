"""Isolated native ACE 5Hz planning followed by explicit GPU release.

Gemma's scene caption and musical metadata remain authoritative. The native LM
generates only audio semantic codes using the official open-assistant format.
This child deliberately does not acquire another GPU lease: its parent owns it.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.audio.ace_planner_prepare import (
    DEFAULT_MODEL_PATH,
    MODEL_KEY,
    MODEL_SUBFOLDER,
    REPO_ID,
    validate_model,
)
from scripts.audio.ace_prompts import normalize_ace_metadata
from scripts.audio.json_io import write_json

LYRICS = "[Instrumental]"
LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"
AUDIO_CODE_MAX = 63999
TEMPERATURE = 0.85
TOP_P = 0.9
CFG_SCALE = 2.0
NEGATIVE_PROMPT = "NO USER INPUT"


class PlannerCancelled(Exception):
    pass


def validate_request(data: Any) -> dict:
    if not isinstance(data, dict):
        raise TypeError("planner の要求は JSON オブジェクトで指定してください。")
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 10000:
        raise ValueError("ACE planner には音楽の説明が必要です (10,000 文字以内)。")
    duration = data.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
        raise ValueError("ACE planner の生成秒数は有限の数値で指定してください。")
    if not 10 <= duration <= 600:
        raise ValueError("ACE planner の生成秒数は 10〜600 秒です。")
    seed = data.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("ACE planner の Seed は 0〜4294967295 の整数です。")
    device = data.get("device", "auto")
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("ACE planner の device は auto / cuda / cpu です。")
    supplied = {key: data.get(key) for key in ("bpm", "keyscale", "timesignature")}
    normalized = normalize_ace_metadata(supplied, allow_unspecified=True)
    metadata = {
        "bpm": normalized["bpm"] or None,
        "keyscale": normalized["keyscale"] or None,
        "timesignature": normalized["timesignature"] or None,
        "duration": float(duration), "language": "unknown",
    }
    if data.get("lyrics", LYRICS) != LYRICS or data.get("vocal_language", "unknown") != "unknown":
        raise ValueError("ACE planner の歌詞は [Instrumental]、言語は unknown 固定です。")
    model_path = data.get("planner_model_path", str(DEFAULT_MODEL_PATH))
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("ACE planner のモデルフォルダーを指定してください。")
    return {"prompt": prompt.strip(), "duration": float(duration), "seed": seed,
            "device": device, "planner_model_path": str(Path(model_path).resolve()),
            "metadata": metadata, "code_count": math.ceil(float(duration) * 5)}


def build_prompt(tokenizer: Any, caption: str, metadata: dict) -> str:
    # The official LM uses sorted YAML, not JSON or a closed assistant turn.
    # Every interpolated field is a validated scalar; quote the free-form caption
    # with JSON syntax, which is also a valid YAML double-quoted scalar.
    fields = {key: value for key, value in metadata.items() if value is not None}
    fields["caption"] = caption
    lines = []
    for key in sorted(fields):
        value = fields[key]
        if key in {"duration", "bpm"}:
            numeric = float(value)
            scalar = str(int(numeric)) if numeric.is_integer() else str(numeric)
        elif key == "timesignature":
            scalar = str(int(value))
        else:
            scalar = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {scalar}")
    messages = [
        {"role": "system", "content": f"# Instruction\n{LM_INSTRUCTION}\n\n"},
        {"role": "user", "content": f"# Caption\n{caption}\n\n# Lyric\n{LYRICS}\n"},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return formatted + "<think>\n" + "\n".join(lines) + "\n</think>\n\n"


def audio_token_map(vocabulary: dict[str, int]) -> dict[int, int]:
    pattern = re.compile(r"<\|audio_code_(\d+)\|>")
    result = {}
    for token, token_id in vocabulary.items():
        match = pattern.fullmatch(token)
        if match and 0 <= int(match[1]) <= AUDIO_CODE_MAX:
            result[token_id] = int(match[1])
    if len(result) != AUDIO_CODE_MAX + 1 or set(result.values()) != set(range(AUDIO_CODE_MAX + 1)):
        raise ValueError("ACE planner の音楽トークン 0〜63999 が完全に揃っていません。")
    if any(type(key) is not int or key < 0 for key in result):
        raise ValueError("ACE planner の音楽トークン ID が不正です。")
    return result


def build_unconditional_prompt(tokenizer: Any) -> str:
    # Match the official training CFG-dropout branch. It deliberately carries
    # neither the conditional caption nor its lyrics/metadata wrapper.
    messages = [
        {"role": "system", "content": f"# Instruction\n{LM_INSTRUCTION}\n\n"},
        {"role": "user", "content": NEGATIVE_PROMPT},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    ) + "<think>\n\n</think>\n\n"


def make_cfg_processor(torch: Any, transformers: Any, model: Any, negative_inputs: Any) -> Any:
    class NativeCFG(transformers.UnbatchedClassifierFreeGuidanceLogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            # HF's built-in log-probability formula differs from native raw
            # logits by only a per-row additive constant. Upcasting both raw
            # branches first reproduces ACE's fp32 formula without BF16
            # log_softmax rounding; HF still manages the separate KV cache.
            unconditional = self.get_unconditional_logits(input_ids)[:, -1, :].float()
            conditional = scores.float()
            guided = unconditional + self.guidance_scale * (conditional - unconditional)
            return torch.nan_to_num(guided, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))

    return NativeCFG(
        CFG_SCALE, model, unconditional_ids=negative_inputs["input_ids"],
        unconditional_attention_mask=negative_inputs.get("attention_mask"), use_cache=True,
    )


def generate_codes(request: dict, output_dir: Path) -> dict:
    import torch
    import transformers

    started = time.perf_counter()

    def cancelled() -> None:
        if (output_dir / "stop.request").exists():
            raise PlannerCancelled("ACE planner の生成を中止しました。")

    def status(message: str, **details: Any) -> None:
        write_json(output_dir / "status.json", {"phase": "planning", "message": message, **details})
        print(message, flush=True)

    cancelled()
    manifest = validate_model(Path(request["planner_model_path"]))
    device = request["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("ACE planner 用の CUDA GPU が見つかりません。")
    dtype_name = "bfloat16" if device == "cuda" else "float32"
    dtype = getattr(torch, dtype_name)
    model = tokenizer = inputs = negative_inputs = outputs = processor = guidance = stopping = None
    result = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        status("ACE planner 1.7B を読み込んでいます。")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            request["planner_model_path"], local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        code_tokens = audio_token_map(tokenizer.get_vocab())
        cancelled()
        model = transformers.AutoModelForCausalLM.from_pretrained(
            request["planner_model_path"], local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=dtype, attn_implementation="sdpa",
        ).to(device).eval()
        cancelled()
        caption = request["prompt"]
        formatted = build_prompt(tokenizer, caption, request["metadata"])
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False, truncation=False).to(device)
        negative_inputs = tokenizer(
            build_unconditional_prompt(tokenizer), return_tensors="pt", add_special_tokens=False, truncation=False,
        ).to(device)
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length + request["code_count"] > int(model.config.max_position_embeddings):
            raise ValueError("ACE planner のプロンプトと生成秒数がモデルのコンテキスト長を超えています。")

        class OnlyAudioCodes(transformers.LogitsProcessor):
            def __init__(self) -> None:
                self.mask = torch.full((int(model.config.vocab_size),), float("-inf"), device=device)
                self.mask[list(code_tokens)] = 0

            def __call__(self, _ids: Any, scores: Any) -> Any:
                return scores + self.mask.to(dtype=scores.dtype)

        class CancelAndProgress(transformers.StoppingCriteria):
            last_report = -1

            def __call__(self, input_ids: Any, _scores: Any, **_kwargs: Any) -> bool:
                cancelled()
                count = input_ids.shape[1] - prompt_length
                if count == request["code_count"] or count >= self.last_report + 25:
                    self.last_report = count
                    status("ACE planner が曲の構成を作成しています。", step=count, steps=request["code_count"])
                return False

        processor = OnlyAudioCodes()
        guidance = make_cfg_processor(torch, transformers, model, negative_inputs)
        stopping = CancelAndProgress()
        torch.manual_seed(request["seed"])
        if device == "cuda":
            torch.cuda.manual_seed_all(request["seed"])
        load_seconds = time.perf_counter() - started
        status("ACE planner が曲の構成を作成しています。", step=0, steps=request["code_count"])
        infer_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, max_new_tokens=request["code_count"], min_new_tokens=request["code_count"],
                do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, top_k=0,
                repetition_penalty=1.0, use_cache=True, guidance_scale=1.0,
                logits_processor=transformers.LogitsProcessorList([guidance, processor]),
                stopping_criteria=transformers.StoppingCriteriaList([stopping]),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        cancelled()
        token_ids = outputs[0, prompt_length:].detach().cpu().tolist()
        if len(token_ids) != request["code_count"] or any(token not in code_tokens for token in token_ids):
            raise RuntimeError("ACE planner が指定秒数分の正しい音楽コードを返しませんでした。")
        audio_codes = "".join(f"<|audio_code_{code_tokens[token]}|>" for token in token_ids)
        peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        result = {
            "ok": True, "audio_codes": audio_codes, "code_count": len(token_ids), "actual_device": device,
            "metadata": request["metadata"],
            "planner_metadata": {
                "model": MODEL_KEY, "model_subfolder": MODEL_SUBFOLDER,
                "repo_id": REPO_ID, "repo_revision": manifest["repo_revision"],
                "seed": request["seed"], "device": device, "actual_device": device, "dtype": dtype_name,
                "transformers_version": transformers.__version__,
                "caption": caption, "lyrics": LYRICS, "vocal_language": "unknown",
                "generation_phase": "codes", "cot_caption_rewrite": False,
                "metadata_source": "provided_scene_brief",
                "unspecified_metadata": [key for key in ("bpm", "keyscale", "timesignature")
                                         if request["metadata"][key] is None],
                "audio_code_range": [0, AUDIO_CODE_MAX], "audio_code_rate_hz": 5,
                "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": 0,
                "repetition_penalty": 1.0, "cfg_scale": CFG_SCALE,
                "cfg_implementation": "native_raw_logits_float32_with_transformers_kv_cache",
                "cfg_negative_prompt": NEGATIVE_PROMPT, "cfg_unconditional_metadata": False,
                "load_seconds": round(load_seconds, 3),
                "inference_seconds": round(time.perf_counter() - infer_started, 3),
                "cuda_peak_allocated_bytes": peak, "remote_code": False,
            },
        }
    finally:
        # Drop every GPU-bearing reference before publishing successful results.
        model = tokenizer = inputs = negative_inputs = outputs = processor = guidance = stopping = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    if result is None:
        raise RuntimeError("ACE planner の結果がありません。")
    cancelled()
    result["planner_released"] = True
    result["planner_metadata"]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result["planner_metadata"]["cuda_allocated_after_release_bytes"] = (
        int(torch.cuda.memory_allocated()) if device == "cuda" else 0
    )
    return result


def run_request(request_path: Path, output_dir: Path) -> int:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if result_path.exists():
        print("Existing planner result will not be overwritten.", file=sys.stderr)
        return 2
    try:
        request = validate_request(json.loads(request_path.read_text(encoding="utf-8-sig")))
        result = generate_codes(request, output_dir)
        write_json(result_path, result)
        write_json(output_dir / "status.json", {"phase": "done", "message": "ACE planner を解放しました。音声生成へ進めます。"})
        return 0
    except Exception as exc:  # noqa: BLE001 - subprocess must publish any failure for the GUI.
        cancelled = isinstance(exc, PlannerCancelled) or (output_dir / "stop.request").exists()
        write_json(result_path, {"ok": False, "cancelled": cancelled, "error": str(exc),
                                 "error_type": type(exc).__name__})
        write_json(output_dir / "status.json", {"phase": "cancelled" if cancelled else "error", "message": str(exc)})
        if not cancelled:
            traceback.print_exc()
        return 130 if cancelled else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate instrumental ACE-Step 5Hz semantic codes.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_request(args.request, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
