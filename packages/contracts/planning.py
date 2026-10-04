"""Semantic STEP4 common plan, shared by the coordinator and local worker.

Engine record hashes, request journals and chapter checkpoints are deliberately
outside this content contract. Approval binds this content to its M2 snapshot.
"""
from __future__ import annotations

from packages.contracts.script import Contract
from services.worker.generation.script_cast import CastPlan, check_cast
from services.worker.generation.script_plot import (
    CausalRoute,
    DetailedPlot,
    check_chapters,
    check_core,
)


def planning_protocol() -> dict:
    return {"workflow": 2, "prompt": 1, "implementation": 1,
            "contract_revision": 1, "policy": "common_plan_v1"}


class CommonPlanContent(Contract):
    cast_plan: CastPlan
    plot: DetailedPlot


def validate_plan_content(content: dict | CommonPlanContent, snapshot: dict) -> CommonPlanContent:
    value = CommonPlanContent.model_validate(content)
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
    main = {row["result"]["id"] for row in snapshot["characters"]}
    count = snapshot["world"]["result"]["chapterCount"]
    check_cast(value.cast_plan, main)
    check_core(value.plot.core, main, count)
    check_chapters(value.plot.chapters, list(range(1, count + 1)))
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
