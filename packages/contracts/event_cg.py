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
CG_KINDS = frozenset({"m3_event_cg_budget", "m3_event_cg_plan", "m3_event_cg"})
CG_LLM_KINDS = frozenset({"m3_event_cg_budget", "m3_event_cg_plan"})


class EventCgProfile(Contract):
    backend: Literal["qwen_image21"] = "qwen_image21"
    model_revision: Literal["d26bb61231c349cf6b7896fa83353113880e1ba3"] = MODEL_REVISION
    dtype: Literal["bfloat16"] = "bfloat16"
    cpu_offload: bool = Field(default=True, strict=True)
    use_kv_cache: bool = Field(default=True, strict=True)
    steps: int = Field(default=40, ge=1, le=100, strict=True)
    width: Literal[960] = 960
    height: Literal[640] = 640


class EventCgPolicy(Contract):
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


class EventCgPlannedVariant(Contract):
    id: Identifier
    start_utterance_id: Identifier
    interpretation: str = Field(min_length=1, max_length=2000)
    prompt: str = Field(min_length=1, max_length=10000)


class EventCgPlannedImage(Contract):
    id: Identifier
    scene_id: Identifier
    start_utterance_id: Identifier
    end_utterance_id: Identifier | None
    character_ids: list[str] = Field(min_length=1, max_length=9)
    interpretation: str = Field(min_length=1, max_length=2000)
    prompt: str = Field(min_length=1, max_length=10000)
    variants: list[EventCgPlannedVariant] = Field(default_factory=list, max_length=10)


class EventCgPromptOmission(Contract):
    cg_id: Identifier
    reason: str = Field(min_length=1, max_length=4000)


class EventCgPlan(Contract):
    schema_version: Literal[1] = 1
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cgs: list[EventCgPlannedImage] = Field(default_factory=list, max_length=100)
    omission_reason: str = Field(default="", max_length=4000)
    prompt_omissions: list[EventCgPromptOmission] = Field(default_factory=list, max_length=100)


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
        intervals.append((start, end))
    intervals.sort()
    if any(left[1] > right[0] for left, right in pairwise(intervals)):
        raise ValueError("CG intervals cannot overlap.")
    return result
