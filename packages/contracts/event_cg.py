"""Bounded event CG planning and generation, independent of story writing."""
from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from typing import Literal

from pydantic import Field, model_validator

from .script import Contract, Identifier

MODEL_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
BUDGET_ALLOCATION_VERSION = 2
STAGING_RULES_VERSION = 2
# These values belong to planning version 2. Change the version for new rules.
MIN_IMAGE_UTTERANCES = 5
TARGET_CG_UTTERANCES = (12, 30)
CG_KINDS = frozenset({"m3_event_cg_budget", "m3_event_cg_plan", "m3_event_cg"})
CG_LLM_KINDS = frozenset({"m3_event_cg_budget", "m3_event_cg_plan"})
CG_SIZES = ((960, 640), (1536, 1024))
# Selectable memory configurations. Peak VRAM is PyTorch's allocation with three references
# at 1536x1024 (docs/setup/qwen-image-edit.md); the driver and desktop need 3-4 GiB beyond
# it. time_ratio is inference time relative to the standard configuration: seconds depend
# on the hardware and are never shown.
CG_CONFIGURATION_FIELDS = ("reference_resolution", "use_kv_cache", "transformer_storage", "vae_tiling")
CG_CONFIGURATIONS = (
    {"id": "vram32", "label": "32GB：標準", "vram_gb": 32, "reference_resolution": 1024,
     "use_kv_cache": True, "transformer_storage": "native", "vae_tiling": False,
     "peak_vram_gib": 22.8, "time_ratio": 1.0,
     "quality": "基準。量子化なしで、参照画像を最も細かく読み取ります。"},
    {"id": "vram24_fast", "label": "24GB：速度優先", "vram_gb": 24, "reference_resolution": 768,
     "use_kv_cache": True, "transformer_storage": "native", "vae_tiling": False,
     "peak_vram_gib": 19.4, "time_ratio": 0.7,
     "quality": "量子化なし。参照画像を粗く読むため、服の柄やワッペンなどの細部が簡略になることがあります。"},
    {"id": "vram24_detail", "label": "24GB：細部優先", "vram_gb": 24, "reference_resolution": 1024,
     "use_kv_cache": False, "transformer_storage": "native", "vae_tiling": False,
     "peak_vram_gib": 16.6, "time_ratio": 1.8,
     "quality": "基準と同等。参照画像を毎ステップ読み直すため、時間がかかります。"},
    {"id": "vram24_8bit", "label": "24GB：8ビット", "vram_gb": 24, "reference_resolution": 1024,
     "use_kv_cache": True, "transformer_storage": "fp8", "vae_tiling": True,
     "peak_vram_gib": 16.2, "time_ratio": 0.9,
     "quality": "8ビット。構図と人物は基準とほぼ同じで、口元などの細部が変わることがあります。"},
    {"id": "vram16_8bit", "label": "16GB：8ビット・速度優先", "vram_gb": 16, "reference_resolution": 768,
     "use_kv_cache": True, "transformer_storage": "fp8", "vae_tiling": True,
     "peak_vram_gib": 12.8, "time_ratio": 0.8,
     "quality": "8ビット。参照画像も粗く読むため、小物や柄の再現が基準より落ちることがあります。"},
    {"id": "vram16_8bit_detail", "label": "16GB：8ビット・細部優先", "vram_gb": 16,
     "reference_resolution": 1024, "use_kv_cache": False, "transformer_storage": "fp8",
     "vae_tiling": True, "peak_vram_gib": 10.0, "time_ratio": 1.9,
     "quality": "8ビット。参照画像は細かく読みます。時間がかかります。"},
    {"id": "vram12_8bit", "label": "12GB：8ビット", "vram_gb": 12, "reference_resolution": 768,
     "use_kv_cache": False, "transformer_storage": "fp8", "vae_tiling": True,
     "peak_vram_gib": 9.2, "time_ratio": 1.2,
     "quality": "8ビット。参照画像も粗く読みます。最も軽い構成です。"},
)


class EventCgProfile(Contract):
    backend: Literal["qwen_image21"] = "qwen_image21"
    model_revision: Literal["d26bb61231c349cf6b7896fa83353113880e1ba3"] = MODEL_REVISION
    dtype: Literal["bfloat16"] = "bfloat16"
    cpu_offload: bool = Field(default=True, strict=True)
    use_kv_cache: bool = Field(default=True, strict=True)
    steps: int = Field(default=40, ge=1, le=100, strict=True)
    width: Literal[960, 1536] = 960
    height: Literal[640, 1024] = 640
    # Defaults reproduce productions frozen before these options existed.
    reference_resolution: Literal[768, 1024] = 1024
    text_encoder_offload: Literal["model", "layers"] = "model"
    transformer_storage: Literal["native", "fp8"] = "native"
    vae_tiling: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def supported_size(self):
        if (self.width, self.height) not in CG_SIZES:
            raise ValueError("CG size must be 960x640 or 1536x1024.")
        return self


def configuration_id(profile: dict | EventCgProfile) -> str | None:
    """The listed configuration a profile corresponds to, if any."""
    value = EventCgProfile.model_validate(profile).model_dump(mode="json")
    return next((row["id"] for row in CG_CONFIGURATIONS
                 if all(value[key] == row[key] for key in CG_CONFIGURATION_FIELDS)), None)


class EventCgPolicy(Contract):
    planning_version: Literal[1, 2] = 1
    max_cgs: int = Field(default=0, ge=0, le=100, strict=True)
    max_variants_per_cg: int = Field(default=0, ge=0, le=10, strict=True)
    generation_profile: EventCgProfile = Field(default_factory=EventCgProfile)


class EventCgChapterBudget(Contract):
    chapter_number: int = Field(ge=1, le=100, strict=True)
    limit: int = Field(ge=0, le=100, strict=True)
    reason: str = Field(min_length=1, max_length=2000)


class EventCgBudget(Contract):
    schema_version: Literal[1] = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chapters: list[EventCgChapterBudget] = Field(max_length=100)
    omission_reason: str = Field(default="", max_length=4000)


class EventCgStaging(Contract):
    visual_state: str = Field(min_length=1, max_length=2000)
    evidence_utterance_ids: list[Identifier] = Field(min_length=1, max_length=20)
    safe_end_utterance_id: Identifier | None
    reason: str = Field(min_length=1, max_length=2000)
    change: str = Field(default="", max_length=2000)


class EventCgPlannedVariant(Contract):
    id: Identifier
    start_utterance_id: Identifier
    interpretation: str = Field(min_length=1, max_length=2000)
    prompt: str = Field(min_length=1, max_length=10000)
    staging: EventCgStaging | None = None


class EventCgPlannedImage(Contract):
    id: Identifier
    scene_id: Identifier
    start_utterance_id: Identifier
    end_utterance_id: Identifier | None
    character_ids: list[str] = Field(min_length=1, max_length=9)
    interpretation: str = Field(min_length=1, max_length=2000)
    prompt: str = Field(min_length=1, max_length=10000)
    variants: list[EventCgPlannedVariant] = Field(default_factory=list, max_length=10)
    staging: EventCgStaging | None = None
    composition: str = Field(default="", max_length=2000)
    end_reason: str = Field(default="", max_length=2000)
    end_evidence_utterance_id: Identifier | None = None


class EventCgPromptOmission(Contract):
    cg_id: Identifier
    reason: str = Field(min_length=1, max_length=4000)


class EventCgPlanningNote(Contract):
    cg_id: Identifier
    variant_id: Identifier | None = None
    reason: str = Field(min_length=1, max_length=4000)


class EventCgPlan(Contract):
    schema_version: Literal[1] = 1
    planning_version: Literal[1, 2] = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cgs: list[EventCgPlannedImage] = Field(default_factory=list, max_length=100)
    omission_reason: str = Field(default="", max_length=4000)
    prompt_omissions: list[EventCgPromptOmission] = Field(default_factory=list, max_length=100)
    planning_notes: list[EventCgPlanningNote] = Field(default_factory=list, max_length=1100)


class EventCgResult(Contract):
    schema_version: Literal[1] = 1
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cg_id: Identifier
    variant_id: Identifier | None = None
    status: Literal["complete", "omitted"]
    reason: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def explicit_omission(self):
        if self.status == "omitted" and not self.reason.strip():
            raise ValueError("CG omission requires a reason.")
        if self.status == "complete" and self.reason:
            raise ValueError("Completed CG cannot contain an omission reason.")
        return self


def image_input_sha256(payload: dict) -> str:
    """The coordinator request, before worker-local reference paths are attached."""
    data = {key: value for key, value in payload.items()
            if key not in {"input_sha256", "reference_paths"}}
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_budget(value: dict | EventCgBudget, payload: dict) -> EventCgBudget:
    result = EventCgBudget.model_validate(value)
    context = payload["context"]
    numbers = [row["number"] for row in context["overall_plot"]["chapters"]]
    policy = EventCgPolicy.model_validate(payload["policy"])
    if (result.source_sha256 != context["source_sha256"]
            or [row.chapter_number for row in result.chapters] != numbers
            or sum(row.limit for row in result.chapters) > policy.max_cgs):
        raise ValueError("CG budget differs from the approved plot or total allowance.")
    if result.omission_reason and any(row.limit for row in result.chapters):
        raise ValueError("Omitted CG budget must allocate zero images.")
    if (payload.get("budget_allocation_version", 1) >= BUDGET_ALLOCATION_VERSION
            and not result.omission_reason
            and (allocated := sum(row.limit for row in result.chapters)) != policy.max_cgs):
        raise ValueError(
            f"CG chapter allowances total {allocated}; allocate all {policy.max_cgs} available slots "
            "across the supplied chapters. Actual image selection may use fewer slots.")
    return result


def validate_plan(value: dict | EventCgPlan, payload: dict) -> EventCgPlan:
    result = EventCgPlan.model_validate(value)
    policy = EventCgPolicy.model_validate(payload["policy"])
    if result.planning_version != policy.planning_version:
        raise ValueError("CG planning version differs from the frozen production policy.")
    context = payload["context"]
    scenes = context["narrative"]["scenes"]
    scenes_by_id = {scene["id"]: scene for scene in scenes}
    utterances = [(scene["id"], row["id"]) for scene in scenes for row in scene["utterances"]]
    order = {uid: index for index, (_, uid) in enumerate(utterances)}
    known = {row["character_id"] for row in payload["references"]}
    if (result.source_sha256 != context["source_sha256"]
            or len(result.cgs) + len(result.prompt_omissions) > min(policy.max_cgs, payload["chapter_budget"])):
        raise ValueError("CG plan differs from the frozen narrative or chapter allowance.")
    if result.omission_reason and result.cgs:
        raise ValueError("Omitted CG plan cannot contain images.")
    omitted_ids = [row.cg_id for row in result.prompt_omissions]
    if len(set(omitted_ids)) != len(omitted_ids):
        raise ValueError("Omitted CG IDs must be unique.")
    identifiers, intervals = set(omitted_ids), []
    for cg in result.cgs:
        scene = scenes_by_id.get(cg.scene_id)
        start = order.get(cg.start_utterance_id, -1)
        end = len(utterances) if cg.end_utterance_id is None else order.get(cg.end_utterance_id, -1)
        if (scene is None or not 0 <= start < end <= len(utterances)
                or any(sid != cg.scene_id for sid, _ in utterances[start:end])):
            raise ValueError("CG range must identify a nonempty interval inside one frozen scene.")
        if (len(set(cg.character_ids)) != len(cg.character_ids)
                or not set(cg.character_ids) <= known
                or not set(cg.character_ids) <= set(scene["plan"]["character_ids"])
                or len(cg.variants) > policy.max_variants_per_cg):
            raise ValueError("CG cast or variations differ from the permitted references.")
        ids = [cg.id] + [variant.id for variant in cg.variants]
        if len(set(ids)) != len(ids) or identifiers.intersection(ids):
            raise ValueError("CG and variant IDs must be unique within a chapter.")
        identifiers.update(ids)
        anchors = [order.get(variant.start_utterance_id, -1) for variant in cg.variants]
        if anchors != sorted(set(anchors)) or any(not start < anchor < end for anchor in anchors):
            raise ValueError("CG variants require distinct, ordered anchors strictly inside the interval.")
        if policy.planning_version >= STAGING_RULES_VERSION:
            _validate_staging(cg, utterances, order, start, end, anchors)
        intervals.append((start, end))
    intervals.sort()
    if any(left[1] > right[0] for left, right in pairwise(intervals)):
        raise ValueError("CG intervals cannot overlap.")
    return result


def _validate_staging(cg, utterances, order, start, end, anchors):
    if not cg.composition.strip() or not cg.end_reason.strip():
        raise ValueError(f"{cg.id}: provide the shared composition and a reason for ending the CG.")
    ending_evidence = order.get(cg.end_evidence_utterance_id, -1)
    if (ending_evidence < end - 1 or
            (utterances[ending_evidence][0] != cg.scene_id and ending_evidence != end)):
        raise ValueError(f"{cg.id}: ending evidence must identify the last displayed or a later "
                         "utterance in this scene, or the exact next-scene boundary.")
    images = [cg, *cg.variants]
    boundaries = [start, *anchors, end]
    short = [f"{image.id}: {stop - begin} utterances" for image, begin, stop in
             zip(images, boundaries[:-1], boundaries[1:], strict=True) if stop - begin < MIN_IMAGE_UTTERANCES]
    if short:
        raise ValueError(f"Each CG image needs at least {MIN_IMAGE_UTTERANCES} utterances; " +
                         "; ".join(short) + ". Reduce variants or reselect a coherent span; do not pad the story.")
    for index, image in enumerate(images):
        staging = image.staging
        if staging is None or not staging.visual_state.strip() or not staging.reason.strip():
            raise ValueError(f"{image.id}: provide a visible state, evidence and selection reason.")
        if index and not staging.change.strip():
            raise ValueError(f"{image.id}: a variant needs a significant, sustained visual change.")
        evidence = staging.evidence_utterance_ids
        if len(set(evidence)) != len(evidence) or any(
                uid not in order or utterances[order[uid]][0] != cg.scene_id or
                order[uid] > boundaries[index] for uid in evidence):
            raise ValueError(f"{image.id}: state evidence must be unique utterances in this scene "
                             "at or before image start; never reveal a later event early.")
        safe_end = (len(utterances) if staging.safe_end_utterance_id is None else
                    order.get(staging.safe_end_utterance_id, -1))
        if not boundaries[index + 1] <= safe_end <= end:
            raise ValueError(f"{image.id}: safe end must cover its assigned span and stay inside the CG.")


def image_spans(cg: dict | EventCgPlannedImage, narrative: dict) -> list[dict]:
    """Compute display pages and characters from trusted immutable utterances."""
    cg = EventCgPlannedImage.model_validate(cg)
    rows = [row for scene in narrative["scenes"] for row in scene["utterances"]]
    order = {row["id"]: index for index, row in enumerate(rows)}
    images = [cg, *cg.variants]
    try:
        positions = [order[image.start_utterance_id] for image in images]
        positions.append(len(rows) if cg.end_utterance_id is None else order[cg.end_utterance_id])
    except KeyError as exc:
        raise ValueError("CG metrics require known utterance anchors.") from exc
    if any(start >= end for start, end in pairwise(positions)):
        raise ValueError("CG metrics require increasing image anchors.")
    return [{"id": image.id, "variant_id": image.id if index else None,
             "start_utterance_id": image.start_utterance_id,
             "end_utterance_id": rows[end]["id"] if end < len(rows) else None,
             "utterance_count": end - start,
             "character_count": sum(len(row.get("display_text", "")) for row in rows[start:end])}
            for index, (image, start, end) in enumerate(zip(images, positions[:-1], positions[1:], strict=True))]
