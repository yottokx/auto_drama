"""Semantic STEP4 common plan, shared by the coordinator and local worker.

Engine record hashes, request journals and chapter checkpoints are deliberately
outside this content contract. Approval binds this content to its M2 snapshot.
"""
from __future__ import annotations

import copy

from pydantic import Field

from packages.contracts.outfits import snapshot_outfits
from packages.contracts.script import Contract
from services.worker.generation.script_cast import CastPlan, check_cast
from services.worker.generation.script_chapter_count import ChapterCountJudgment
from services.worker.generation.script_plot import (
    CausalRoute,
    DetailedPlot,
    check_chapters,
    check_core,
    check_outfit_appearances,
)


def planning_protocol() -> dict:
    return {"workflow": 2, "prompt": 1, "implementation": 1,
            "contract_revision": 1, "policy": "common_plan_v1"}


class CommonPlanContent(Contract):
    cast_plan: CastPlan
    # A long-form plan first holds only its cast and how many chapters they were judged to need;
    # the plot comes once the user chose the count. Earlier plans hold neither field.
    plot: DetailedPlot | None = Field(default=None, exclude_if=lambda value: value is None)
    chapter_count: int | None = Field(default=None, ge=1, le=100, strict=True,
                                      exclude_if=lambda value: value is None)
    chapter_judgment: ChapterCountJudgment | None = Field(default=None, exclude_if=lambda value: value is None)


def plan_snapshot(snapshot: dict, content: dict | None) -> dict:
    """The approval snapshot as a plan's chapters see it: a chosen count replaces the approval's."""
    count = (content or {}).get("chapter_count")
    if count is None:
        return snapshot
    value = copy.deepcopy(snapshot)
    value["world"]["result"]["chapterCount"] = count
    return value


def validate_plan_content(content: dict | CommonPlanContent, snapshot: dict) -> CommonPlanContent:
    value = CommonPlanContent.model_validate(content)
    main = {row["result"]["id"] for row in snapshot["characters"]}
    if value.plot is None:
        if value.chapter_judgment is None or value.chapter_count is not None:
            raise ValueError("A plan without its plot holds its cast and their chapter count's judgment.")
        check_cast(value.cast_plan, main)
        return value
    chapters = []
    for chapter in value.plot.chapters:
        if chapter.events:
            routes = [event.as_route() for event in chapter.events]
            route = CausalRoute(start_condition=routes[0].start_condition, next_state=routes[-1].next_state,
                                **{key: " → ".join(getattr(row, key) for row in routes)
                                   for key in ("attempt", "consequence", "choice", "core_progress")})
            chapter = chapter.model_copy(update={"route": route})
        chapters.append(chapter)
    value = value.model_copy(update={"plot": value.plot.model_copy(update={"chapters": chapters})})
    count = value.chapter_count or snapshot["world"]["result"]["chapterCount"]
    check_cast(value.cast_plan, main)
    check_core(value.plot.core, main, count)
    check_chapters(value.plot.chapters, list(range(1, count + 1)))
    outfits = snapshot_outfits(snapshot)
    if outfits:
        check_outfit_appearances(value.plot.chapters, outfits)
    elif any(chapter.outfit_appearances for chapter in value.plot.chapters):
        raise ValueError("Only an approved alternate look can be planned for a chapter.")
    available = main | {row.id for row in value.cast_plan.supporting_characters}
    for chapter in value.plot.chapters:
        for topic in chapter.conversation_topics:
            if (len(set(topic.character_ids)) != len(topic.character_ids)
                    or not set(topic.character_ids) <= available):
                raise ValueError("Conversation topics need distinct registered participants.")
        for event in chapter.events:
            if any(step.character_id not in available for step in event.steps):
                raise ValueError("Plot events must identify registered characters.")
    return value
