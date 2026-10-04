"""Sample-accurate playback plans for saved and legacy BGM loop experiments.

This module uses only the standard library. Audio decoding and playback remain
in the dedicated audio process; the GUI can inspect points without loading it.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 44100


@dataclass(frozen=True)
class LoopPlaybackPlan:
    start_sample: int
    end_sample: int
    cli_options: list[str]
    description: str
    warning: str = ""
    restored_intro: bool = False


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} は有限の数値で指定してください。")  # noqa: TRY004 - GUI contract
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} は有限の数値で指定してください。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} は有限の数値で指定してください。")
    return number


def _sample(value: Any, label: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"{label} は0以上の整数サンプル位置で指定してください。")
        return value
    number = _number(value, label)
    if number < 0 or not number.is_integer():
        raise ValueError(f"{label} は0以上の整数サンプル位置で指定してください。")
    return int(number)


def _seconds_to_samples(seconds: float, label: str) -> int:
    samples = seconds * SAMPLE_RATE
    if not math.isfinite(samples):
        raise ValueError(f"{label} のサンプル位置が大きすぎます。")
    return round(samples)


def _stored_length(metadata: Mapping[str, Any]) -> int:
    if "sample_count" in metadata:
        length = _sample(metadata["sample_count"], "音源のサンプル数")
    else:
        duration = _number(metadata.get("duration_seconds"), "音源の長さ")
        if duration <= 0:
            raise ValueError("音源の長さは0より大きくしてください。")
        length = _seconds_to_samples(duration, "音源の長さ")
    if length <= 0:
        raise ValueError("音源のサンプル数は0より大きくしてください。")
    if "sample_rate" in metadata and _number(metadata["sample_rate"], "サンプリング周波数") != SAMPLE_RATE:
        raise ValueError("ループ地点のサンプリング周波数は44100 Hzである必要があります。")
    return length


def _bounds(start: int, end: int, length: int) -> list[str]:
    if not 0 <= start < end <= length:
        raise ValueError("ループ地点は 0 <= A < B <= 音源のサンプル数 にしてください。")
    return ["--loop-start-sample", str(start), "--loop-end-sample", str(end)]


def _description(start: int, end: int) -> str:
    return (
        f"初回 0.00 → {end / SAMPLE_RATE:.2f}秒 / "
        f"繰り返し {start / SAMPLE_RATE:.2f} → {end / SAMPLE_RATE:.2f}秒"
        f"（{(end - start) / SAMPLE_RATE:.2f}秒）"
    )


def _basename(value: str) -> str:
    # Accept Windows provenance paths even when tests inspect a moved export on
    # another platform. Matching a basename is deliberate for moved directories.
    return value.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _raw_source(source: Path, metadata: Mapping[str, Any]) -> bool:
    if source.suffix.casefold() != ".wav":
        return False
    generation = metadata.get("source_generation") or {}
    if isinstance(generation, Mapping) and isinstance(generation.get("metadata"), Mapping):
        generation = generation["metadata"]
    original = generation.get("float_audio_path") if isinstance(generation, Mapping) else None
    if isinstance(original, str) and original:
        if _basename(original) == source.name.casefold():
            return True
        if os.path.normcase(str(Path(original).expanduser().resolve())) == os.path.normcase(str(source)):
            return True
    return source.stem.casefold() == "output-float"


def loop_playback_plan(metadata: Mapping[str, Any]) -> LoopPlaybackPlan | None:
    """Plan intro-once playback, adapting old cropped/rotated loop exports.

    New ``loop.start_sample/end_sample`` describe the saved file. The similarly
    named legacy ``loop.start/end`` describe the original source and are never
    interpreted as points in a cropped file.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("音声の生成記録はJSONオブジェクトで指定してください。")  # noqa: TRY004 - GUI contract
    loop = metadata.get("loop")
    if not loop:
        return None
    if not isinstance(loop, Mapping):
        raise ValueError("ループ情報はJSONオブジェクトで指定してください。")  # noqa: TRY004 - GUI contract
    length = _stored_length(metadata)
    if "start_sample" in loop or "end_sample" in loop:
        start = _sample(loop.get("start_sample"), "ループ地点A")
        end = _sample(loop.get("end_sample"), "ループ地点B")
        options = _bounds(start, end, length)
        return LoopPlaybackPlan(start, end, options, _description(start, end))

    method = loop.get("method")
    offset = 0.0
    if method == "smart_region":
        offset = _number(loop.get("source_start_seconds", 0), "元音声のループ開始秒数")
    elif method == "whole_crossfade":
        # Old whole-file exports started immediately after the overlapped head.
        offset = _number(
            loop.get("overlap_shortening_seconds", loop.get("crossfade_seconds", 0)),
            "旧形式のクロスフェード秒数",
        )
    if offset < 0:
        raise ValueError("元音声のループ開始秒数は0以上にしてください。")
    prefix = _seconds_to_samples(offset, "元音声のループ開始秒数")
    # AI exports already start at original zero; their repaired head belongs to
    # the saved signal and must not be replaced by an unmodified source intro.
    if prefix == 0:
        options = _bounds(0, length, length)
        return LoopPlaybackPlan(0, length, options, _description(0, length))

    source_value = metadata.get("source_audio")
    source = Path(source_value).expanduser().resolve() if isinstance(source_value, str) and source_value.strip() else None
    if source is not None and source.is_file():
        gain = _number(metadata.get("preview_gain", 1.0), "イントロの音量倍率") if _raw_source(source, metadata) else 1.0
        if not 0 < gain <= 1:
            raise ValueError("イントロの音量倍率は0より大きく1以下にしてください。")
        end = prefix + length
        options = [
            "--intro-input", str(source), "--intro-samples", str(prefix),
            "--intro-gain", format(gain, ".17g"), *_bounds(prefix, end, end),
        ]
        return LoopPlaybackPlan(prefix, end, options, _description(prefix, end), restored_intro=True)

    warning = (
        f"旧形式の音源は元の曲の{prefix / SAMPLE_RATE:.2f}秒から始まります。"
        "元音源が見つからないためイントロを復元できません。"
        "保存区間の冒頭から試聴します。イントロを含めるにはループ音源を再作成してください。"
    )
    options = _bounds(0, length, length)
    return LoopPlaybackPlan(0, length, options, _description(0, length), warning=warning)
