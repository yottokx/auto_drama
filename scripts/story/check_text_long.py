"""Run check_text.py's script experiment with a chapter count other than three.

The product plans a work as cast plan -> chain of events -> chapter allocation only for three
chapters; any other count falls back to the old flow without a cast plan or a chain. This
script swaps in chapter-count-aware versions of the three-chapter parts for this process only:
the chain's event range and wording, the allocation's shape and wording, the fixed allocation
for as many events as chapters, and the cast plan's three-chapter gate. Every later step (the
chapter scene plans, scenes and checks) already follows the approved chapter count.

With --chapters 3 nothing is swapped, so the same input runs the product flow as is.
--expand adds a step after the allocation: each chapter in turn, seeing the whole skeleton and the
chapters already expanded, gets its own added steps (the supporting characters' errands, tries and
answers) around its skeleton steps, which stay as they are. Nothing in the product changes.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

from pydantic import Field, create_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.contracts.m3 import Foreshadowing
from packages.contracts.script import Contract
from services.worker.generation import pipeline, script_continuation, script_plot
from services.worker.generation.script_plot import (
    ChainEvent,
    ChainEventDraft,
    ChapterBoundary,
    ChapterPresentation,
    DetailedPlot,
    StoryChain,
    StoryChainDraft,
)
from services.worker.generation.text_debug import run_text_debug
from services.worker.model_config import select_config

EVENTS_PER_CHAPTER = 3  # the product's 3-9 events for three chapters


def replaced(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        if text.count(old) != 1:
            raise ValueError(f"The product wording differs from the expected text: {old}")
        text = text.replace(old, new)
    return text


def chapter_models(count: int):
    """The chain, allocation and fixed-allocation models for ``count`` chapters."""
    most = count * EVENTS_PER_CHAPTER

    class LongStoryChain(StoryChain):
        events: list[ChainEvent] = Field(min_length=count, max_length=most,
            description=f"{count}章に収まる主要な出来事の連鎖。章境界はまだ決めない。各出来事は場面とは別。")

    class LongStoryChainDraft(StoryChainDraft):
        events: list[ChainEventDraft] = Field(min_length=count, max_length=most,
            description=f"{count}章に収まる主要な出来事の連鎖。全て予定であり、章境界はまだ決めない。")

        def as_chain(self) -> LongStoryChain:
            # StoryChainDraft.as_chain, building the longer chain.
            events, condition = [], self.opening_condition
            for event in self.events:
                events.append(ChainEvent(start_condition=condition, steps=event.steps))
                condition = event.steps[-1].result
            return LongStoryChain(core=self.core, events=events)

    class LongChapterBoundary(ChapterBoundary):
        number: int = Field(ge=1, le=count, strict=True)
        last_event: int = Field(ge=1, le=most, strict=True,
                                description=ChapterBoundary.model_fields["last_event"].description)

    class LongChapterAllocation(script_plot.ChapterAllocation):
        chapters: list[LongChapterBoundary] = Field(min_length=count, max_length=count)
        foreshadowing: list[Foreshadowing] = Field(max_length=30)

    class FixedAllocation(Contract):
        def as_allocation(self) -> LongChapterAllocation:
            # As many events as chapters: one event each, no boundary choice.
            return LongChapterAllocation(chapters=[LongChapterBoundary(
                number=n, last_event=n, **getattr(self, f"chapter_{n}").model_dump())
                for n in range(1, count + 1)], foreshadowing=self.foreshadowing)

    fixed = create_model("LongFixedChapterAllocation", __base__=FixedAllocation,
                         **{f"chapter_{n}": (ChapterPresentation, ...) for n in range(1, count + 1)},
                         foreshadowing=(list[Foreshadowing], Field(max_length=30)))
    return LongStoryChain, LongStoryChainDraft, LongChapterAllocation, fixed


MOST_OPTIONS = 30


def balanced_options(sizes: list[int], count: int) -> list[dict]:
    """Ways to cut the chain into ``count`` chapters at event ends, the most even first."""
    options = []
    for cuts in itertools.combinations(range(1, len(sizes)), count - 1):
        ends = [*cuts, len(sizes)]
        steps = [sum(sizes[start:end]) for start, end in zip([0, *ends[:-1]], ends)]
        options.append((max(steps) - min(steps), sum((s - sum(steps) / count) ** 2 for s in steps), ends, steps))
    options.sort(key=lambda row: row[:2])
    return [{"last_events": ends, "steps_per_chapter": steps} for _, _, ends, steps in options[:MOST_OPTIONS]]


def use_chapter_count(count: int, balanced: bool = False) -> None:
    """Make the cast plan, chain and allocation steps follow ``count`` chapters in this process.

    With ``balanced``, the allocation picks one of the most even ways to cut the chain, as the
    product's sized_allocation does for three chapters, with its instruction."""
    sc = script_continuation
    chain_model, draft_model, allocation_model, fixed_model = chapter_models(count)
    most = count * EVENTS_PER_CHAPTER
    chain_instruction = replaced(script_plot.CHAIN_INSTRUCTION, [
        ("全3章の分量", f"全{count}章の分量"),
        ("3〜9個の出来事", f"{count}〜{most}個の出来事"),
        ("3章の各区間で", f"{count}章の各区間で"),
    ])
    allocation_instruction = replaced(script_plot.ALLOCATION_INSTRUCTION, [
        ("全3章へ配分", f"全{count}章へ配分"),
        ("均等に三分割", f"均等に{count}分割"),
    ])
    sized_instruction = replaced(script_plot.SIZED_ALLOCATION_INSTRUCTION, [("全3章へ配分", f"全{count}章へ配分")])

    check_core = script_plot.check_core
    # The chain's plot is checked against the approved count, not three.
    script_plot.check_core = lambda core, characters, _count: check_core(core, characters, count)

    def allocate(chain, allocation):
        if [row.number for row in allocation.chapters] != list(range(1, count + 1)):
            raise ValueError(f"Chapter allocation must preserve the {count} approved chapters.")
        ends = [row.last_event for row in allocation.chapters]
        if ends != sorted(set(ends)) or ends[-1] != len(chain.events):
            raise ValueError("Chapter boundaries must partition every event exactly once in order.")
        starts = [0, *ends[:-1]]
        return script_plot._plot_from_chapters(
            chain, allocation.chapters,
            [chain.events[start:end] for start, end in zip(starts, ends, strict=True)],
            allocation.foreshadowing)

    cast_plan = sc.ScriptRun.cast_plan

    def long_cast_plan(self):
        # The product skips the cast plan for counts other than three unless the run comes
        # from a plan approval; mark the run as one for this call only.
        added = "planning_protocol" not in self.manifest
        if added:
            self.manifest["planning_protocol"] = "long-form-experiment"
        try:
            return cast_plan(self)
        finally:
            if added:
                del self.manifest["planning_protocol"]

    def outline(self):
        # script_continuation.ScriptRun.outline's three-chapter branch, for ``count`` chapters.
        cast = self.cast_plan()
        setting = self.setting(planned=True)
        characters = set(setting["main_character_ids"])
        available = characters | {row.id for row in cast.supporting_characters}
        outfits = self.outfits
        events_note = sc.OUTFIT_EVENTS if outfits else ""
        allocation_note = sc.OUTFIT_ALLOCATION if outfits else ""

        def check(value):
            sc.check_core(value.core, characters, count)
            sc.check_chapters(value.chapters, list(range(1, count + 1)))
            if outfits:
                sc.check_outfit_appearances(value.chapters, outfits)
            for chapter in value.chapters:
                for topic in chapter.conversation_topics:
                    if len(set(topic.character_ids)) != len(topic.character_ids) or not set(topic.character_ids) <= available:
                        raise ValueError("Conversation topics need distinct registered participants.")

        if "plot" in self.state:
            plot = DetailedPlot.model_validate(self.state["plot"])
            check(plot)
            chain = chain_model.model_validate(self.state["story_chain"])
            allocation = allocation_model.model_validate(self.state["chapter_allocation"])
            sc.check_chain(chain, characters, available)
            if (sc.digest(self.state["story_chain"]) != self.state["story_chain_sha256"]
                    or allocate(chain, allocation) != plot):
                raise ValueError("Saved plot differs from its story chain or chapter allocation.")
            if sc.digest(self.state["plot"]) != self.state["plot_sha256"]:
                raise ValueError("Saved detailed plot was changed.")
            if plot.as_outline().model_dump(mode="json") != self.state["outline"]:
                raise ValueError("Saved outline differs from its detailed plot.")
            return plot.as_outline()

        context = {"setting": setting, "outline": "", "brief": self.options.for_stage("chain"),
                   "chapters": [], "notes": []}
        schema = draft_model.model_json_schema()
        sc.bind_core_cast(schema, characters, core="ChainCore", character="ChainCharacter")
        schema["$defs"]["ChainStep"]["properties"]["character_id"]["enum"] = sorted(available)
        chain = self.structured_core("story-chain", "script-outline", 1, chain_instruction + events_note,
            context, draft_model,
            lambda value: sc.check_chain(value.as_chain(), characters, available), schema).as_chain()
        chain_data = chain.model_dump(mode="json")
        self.state.update(story_chain=chain_data, story_chain_sha256=sc.digest(chain_data))
        self.persist()
        sc.write_json(self.output / "story-chain.json", chain_data)
        events = len(chain.events)
        material = {"core": chain.core.model_dump(mode="json"), "event_count": events, "events": [
            {"event_number": n, **event.model_dump(mode="json")} for n, event in enumerate(chain.events, 1)]}
        sized = balanced and events > count
        if sized:
            material.update(total_steps=sum(len(event.steps) for event in chain.events),
                            boundary_options=balanced_options([len(event.steps) for event in chain.events], count))
        allowed = [row["last_events"] for row in material.get("boundary_options", [])]
        allocation_context = {**context, "brief": self.options.for_stage("allocation")
                              + f"\n全{count}章へ配分する保存済みプロット（全て未実施）:\n"
                              + json.dumps(material, ensure_ascii=False)}

        def check_allocation(value):
            check(allocate(chain, value))
            if sized and [row.last_event for row in value.chapters] not in allowed:
                raise ValueError("Chapter boundaries must be one of boundary_options' last_events: "
                                 + json.dumps(allowed))

        if events == count:
            schema = fixed_model.model_json_schema()
            schema["$defs"]["ConversationTopic"]["properties"]["character_ids"]["items"]["enum"] = sorted(available)
            if outfits:
                sc.add_outfit_appearances(schema, outfits)
            each = "、".join(f"chapter_{n}は出来事{n}だけ" for n in range(1, count + 1))
            allocation = self.structured("chapter-allocation", "script-allocation", 1,
                sc.PRESENTATION_INSTRUCTION + allocation_note
                + f"出来事は{count}個、章は{count}章なので配分はコードで確定しています。{each}を担当します。"
                "境界番号は出力せず、担当する出来事を変えずに各章のtitle・role・conversation_topicsを設計します。",
                allocation_context, fixed_model,
                lambda value: check_allocation(value.as_allocation()), schema).as_allocation()
        else:
            schema = allocation_model.model_json_schema()
            schema["$defs"]["LongChapterBoundary"]["properties"]["last_event"]["maximum"] = events
            schema["$defs"]["ConversationTopic"]["properties"]["character_ids"]["items"]["enum"] = sorted(available)
            if outfits:
                sc.add_outfit_appearances(schema, outfits)
            allocation = self.structured("chapter-allocation", "script-allocation", 1,
                (sized_instruction if sized else allocation_instruction)
                + allocation_note + f"出来事は全{events}個です。last_eventはevent_numberを使い、"
                f"{count}章の値を狭義昇順にし、第{count}章の値は必ず{events}にします。",
                allocation_context, allocation_model, check_allocation, schema)
        plot = allocate(chain, allocation)
        self.state["chapter_allocation"] = allocation.model_dump(mode="json")
        sc.write_json(self.output / "chapter-allocation.json", self.state["chapter_allocation"])
        check(plot)
        result = plot.as_outline()
        self.state.update(plot=plot.model_dump(mode="json"), outline=result.model_dump(mode="json"))
        self.state["plot_sha256"] = sc.digest(self.state["plot"])
        self.persist()
        sc.write_json(self.output / "plot.json", self.state["plot"])
        sc.write_json(self.output / "outline.json", self.state["outline"])
        return result

    sc.ScriptRun.cast_plan = long_cast_plan
    sc.ScriptRun.outline = outline


class AddedStep(script_plot.ChainStep):
    after_step: int = Field(ge=0, le=60, strict=True,
        description="当章の骨組みの手順番号のどれの後に置くか。0は最初の手順の前。")


class ChapterExpansion(Contract):
    added_steps: list[AddedStep] = Field(max_length=8,
        description="当章に加える手順。骨組みの手順は書き直さず、ここに含めない。")


EXPANSION_INSTRUCTION = (
    "会話中心のビジュアルノベルの全体プロットを、章ごとに肉付けしています。今回は当章だけです。"
    "skeletonは全章の骨組み（各章の題名・役割・予定の手順）、expanded_chaptersは肉付け済みの前の章、"
    "current_chapter_stepsは当章の骨組みの手順（番号付き）です。すべて未実施の予定です。"
    "当章が1話（約10分）の会話と出来事を支えられるよう、added_stepsに当章で新しく起きる手順を加えます。"
    "骨組みの手順は自動で残るので、書き直さず、added_stepsにも含めません。"
    "加える手順は、骨組みの手順へ至る試行・応答・発見や、サブや脇の人物自身の用事・関心・対立による行動、"
    "メインとサブの具体的なやり取りです。少人数の密な話なら、主要な人物同士の具体的な試行と応答で構いません。"
    "after_stepは、その手順を置く位置（当章の骨組みの手順番号の後。0は最初の前）です。"
    "骨組みの手順の結果と矛盾させず、当章の最後の到達状態を変えません。"
    "他の章の骨組みにある出来事を先取りせず、前の章で済んだ出来事や同じ主張の確認を繰り返しません。"
    "目安は、骨組みと合わせて当章が4〜6手順になる量です。骨組みだけで十分なら空配列も可能です。"
    "stepsは登録人物のactionと具体的なresultで、各1〜2文。全台詞や細かな演出は書きません。"
)


# --distinct-steps: added steps must be new actions, not a restatement or lead-in of a skeleton step.
DISTINCT_EXPANSION_INSTRUCTION = EXPANSION_INSTRUCTION.replace(
    "加える手順は、骨組みの手順へ至る試行・応答・発見や、サブや脇の人物自身の用事・関心・対立による行動、"
    "メインとサブの具体的なやり取りです。",
    "加える手順は、骨組みの手順の前後や間で起きる、骨組みに無い行動・発見・応答です。"
    "サブや脇の人物自身の用事・関心・対立による行動や、メインとサブの具体的なやり取りを中心にします。"
    "骨組みの手順と同じ行動・同じ提案・同じ決定を、言い換えや前振りとして足しません。"
    "骨組みの手順の準備を足すなら、その手順とは別の人物か別の用事にします。")
assert DISTINCT_EXPANSION_INSTRUCTION != EXPANSION_INSTRUCTION
# --fresh-roles: a person's added step may not repeat the role they had in an earlier chapter.
FRESH_ROLES_RULE = (
    "expanded_chaptersで、その人物が前の章で担った役回り（同じ働き、同じ種類の転機）を繰り返しません。"
    "その人物の新しい面や、前の章の結果を受けた次の段階を描くか、"
    "当章では手順を足さずに、別の人物の手順にします。全員を毎章登場させる必要はありません。"
    "場面の外で起きることや、本人が知覚できないことは手順にしません。"
)


def expanded_events(events: list[ChainEvent], added: list[AddedStep]) -> list[ChainEvent]:
    """The chapter's skeleton steps with the added ones in place, as events of up to six steps."""
    skeleton = [step for event in events for step in event.steps]
    if any(row.after_step > len(skeleton) for row in added):
        raise ValueError(f"after_step must be between 0 and {len(skeleton)}.")
    steps = [script_plot.ChainStep(**row.model_dump(exclude={"after_step"}))
             for row in added if row.after_step == 0]
    for number, step in enumerate(skeleton, 1):
        steps.append(step)
        steps.extend(script_plot.ChainStep(**row.model_dump(exclude={"after_step"}))
                     for row in added if row.after_step == number)
    result, condition = [], events[0].start_condition
    for start in range(0, len(steps), 6):
        chunk = steps[start:start + 6]
        result.append(ChainEvent(start_condition=condition, steps=chunk))
        condition = chunk[-1].result
    return result


# --final-closure: the last chapter keeps its last skeleton step at the end (no step is added
# after it), and its scene plan is told to end on the core's resolution.
FINAL_EXPANSION_RULE = "最終章です。骨組みの最後の手順は作品の締めなので、その後には手順を足しません。"
FINAL_PLAN_RULE = (
    "最終章です。作品全体の締めとして、最後の場面の終了状態を、全体プロットの核心にある外的な決着と"
    "人物・関係の着地にします。当章の予定の最後の手順は、その最後の場面で描きます。"
    "核心の外的な決着: {external}\n核心の人物・関係の着地: {relationship}\n"
)


# --closure-end: no core text; the last scene ends where the chapter's own schedule ends.
FINAL_END_RULE = "最終章です。最後の場面の終了状態を、当章の予定の最後の手順の結果にします。\n"


def use_final_closure(core_text: bool = True) -> None:
    """Tell the last chapter's scene plan to end on the core's resolution, or on its own last step."""
    sc = script_continuation
    structured = sc.ScriptRun.structured

    def closing(self, key, purpose, number, prompt, context, model, validate=None, schema=None):
        if key == f"plan-{self.manifest['approved_chapter_count']:03d}" and not core_text:
            prompt = FINAL_END_RULE + prompt
        elif key == f"plan-{self.manifest['approved_chapter_count']:03d}":
            core = self.state["plot"]["core"]
            prompt = FINAL_PLAN_RULE.format(external=core["external_resolution"],
                                            relationship=core["relationship_resolution"]) + prompt
        return structured(self, key, purpose, number, prompt, context, model, validate, schema)

    sc.ScriptRun.structured = closing


def use_expansion(most_steps: int | None = None, distinct: bool = False, fresh_roles: bool = False,
                  final_closure: bool = False) -> None:
    """After the chain and allocation, add each chapter's own steps to its skeleton, in order.

    With ``most_steps``, the answer format lets a chapter reach at most that many steps."""
    sc = script_continuation
    skeleton_outline = sc.ScriptRun.outline

    def outline(self):
        if "skeleton_plot" in self.state:
            if (sc.digest(self.state["skeleton_plot"]) != self.state["skeleton_plot_sha256"]
                    or sc.digest(self.state["plot"]) != self.state["plot_sha256"]):
                raise ValueError("Saved skeleton or expanded plot was changed.")
            return DetailedPlot.model_validate(self.state["plot"]).as_outline()
        skeleton_outline(self)
        skeleton = DetailedPlot.model_validate(self.state["plot"])
        setting = self.setting(planned=True)
        available = set(setting["main_character_ids"]) | {
            row["id"] for row in self.state.get("cast_plan", {}).get("plan", {}).get("supporting_characters", [])}
        sketch = [{"number": row.number, "title": row.title, "role": row.role,
                   "steps": [step.model_dump(mode="json") for event in row.events for step in event.steps]}
                  for row in skeleton.chapters]
        chapters = []
        for row in skeleton.chapters:
            current = [{"step_number": n, **step.model_dump(mode="json")}
                       for n, step in enumerate((step for event in row.events for step in event.steps), 1)]
            brief = json.dumps({"total_chapters": len(skeleton.chapters), "current_chapter": row.number,
                                "skeleton": sketch,
                                "expanded_chapters": [{"number": done.number, "title": done.title, "steps": [
                                    step.model_dump(mode="json") for event in done.events for step in event.steps]}
                                    for done in chapters],
                                "current_chapter_steps": current}, ensure_ascii=False)
            context = {"setting": setting, "outline": "", "brief": self.options.for_stage("chain") + "\n" + brief,
                       "chapters": [], "notes": []}
            schema = ChapterExpansion.model_json_schema()
            step_schema = schema["$defs"]["AddedStep"]
            step_schema["properties"]["character_id"]["enum"] = sorted(available)
            last = row.number == len(skeleton.chapters)
            # The last chapter's closing step stays last: nothing is placed after it.
            step_schema["properties"]["after_step"]["maximum"] = (
                len(current) - 1 if final_closure and last else len(current))
            room = None if most_steps is None else max(0, most_steps - len(current))
            if room is not None:
                schema["properties"]["added_steps"]["maxItems"] = room
            # The position first, as the instruction reads it.
            step_schema["properties"] = {"after_step": step_schema["properties"].pop("after_step"),
                                         **step_schema["properties"]}

            def check(value, events=row.events, room=room):
                if room is not None and len(value.added_steps) > room:
                    raise ValueError(f"Add at most {room} steps to this chapter.")
                expanded_events(events, value.added_steps)

            expansion = self.structured(f"expand-{row.number:03d}", "script-outline", row.number,
                                        f"全{len(skeleton.chapters)}章の第{row.number}章を肉付けします。"
                                        + (DISTINCT_EXPANSION_INSTRUCTION if distinct else EXPANSION_INSTRUCTION)
                                        + (FRESH_ROLES_RULE if fresh_roles else "")
                                        + (FINAL_EXPANSION_RULE if final_closure and last else ""),
                                        context, ChapterExpansion, check, schema)
            chapters.append(row.model_copy(update={"events": expanded_events(row.events, expansion.added_steps)}))
        plot = DetailedPlot(core=skeleton.core, chapters=chapters)
        result = plot.as_outline()
        self.state.update(skeleton_plot=self.state["plot"], skeleton_plot_sha256=self.state["plot_sha256"],
                          plot=plot.model_dump(mode="json"), outline=result.model_dump(mode="json"))
        self.state["plot_sha256"] = sc.digest(self.state["plot"])
        self.persist()
        sc.write_json(self.output / "skeleton-plot.json", self.state["skeleton_plot"])
        sc.write_json(self.output / "plot.json", self.state["plot"])
        sc.write_json(self.output / "outline.json", self.state["outline"])
        return result

    sc.ScriptRun.outline = outline


PLAN_KEYS = ("cast-plan", "story-chain", "chapter-allocation", "expand-")


def reuse_plan(source: Path) -> None:
    """Answer the plan steps (cast, chain, allocation, expansions) with another run's accepted
    replies, read and checked as structured() does, so only the chapters are written anew."""
    sc = script_continuation
    steps = json.loads((source / "draft-state.json").read_text(encoding="utf-8"))["steps"]
    structured = sc.ScriptRun.structured

    def saved(self, key, purpose, number, prompt, context, model, validate=None, schema=None):
        if not key.startswith(PLAN_KEYS):
            return structured(self, key, purpose, number, prompt, context, model, validate, schema)
        rows = [row for row in steps.get(key, {}).get("attempts", [])
                if row["status"] == "completed" and "validation_error" not in row]
        if not rows:
            raise ValueError(f"{source} has no accepted reply for {key}.")
        value = model.model_validate_json(rows[-1]["reply"]["content"])
        prepared = validate(value) if validate else None
        return prepared if prepared is not None else value

    sc.ScriptRun.structured = saved


def reuse_cast(source: Path) -> None:
    """Answer the cast plan with another run's saved plan (e.g. the cast a chapter count was
    judged on), checked as the cast plan step checks its own answer."""
    from services.worker.generation.script_cast import CastPlan

    sc = script_continuation
    plan = CastPlan.model_validate(json.loads((source / "cast-plan.json").read_text(encoding="utf-8"))["plan"])
    structured = sc.ScriptRun.structured

    def saved(self, key, purpose, number, prompt, context, model, validate=None, schema=None):
        if key != "cast-plan":
            return structured(self, key, purpose, number, prompt, context, model, validate, schema)
        prepared = validate(plan) if validate else None
        return prepared if prepared is not None else plan

    sc.ScriptRun.structured = saved


def experiment_input(value: dict, chapters: int, model_id: str | None, skip_staging: bool,
                     cast_limit: int | None = None) -> dict:
    value = copy.deepcopy(value)
    value["approval_snapshot"]["world"]["result"]["chapterCount"] = chapters
    if model_id:
        value.setdefault("profile", {})["model_id"] = model_id
    if skip_staging:
        value.setdefault("script_options", {})["skip_staging"] = True
    if cast_limit is not None:
        value.setdefault("script_options", {})["cast_limit"] = cast_limit
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Approval snapshot with script_options and profile, as check_text_with_cast.py takes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chapters", type=int, default=5)
    parser.add_argument("--model-id", help="Override the profile's model")
    parser.add_argument("--skip-staging", action="store_true", help="No staging or outfit-moment calls")
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--cast-limit", type=int, help="Override the input's script_options cast_limit")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--chapter-limit", type=int)
    parser.add_argument("--expand", action="store_true", help="Add each chapter's own steps to the skeleton")
    parser.add_argument("--balanced", action="store_true",
                        help="Pick one of the most even ways to cut the chain (product sized_allocation wording)")
    parser.add_argument("--expansion-steps", type=int, help="With --expand: the most steps a chapter may reach")
    parser.add_argument("--distinct-steps", action="store_true",
                        help="With --expand: added steps may not restate or lead into a skeleton step")
    parser.add_argument("--fresh-roles", action="store_true",
                        help="With --expand: a person's added step may not repeat their role from an earlier chapter")
    parser.add_argument("--plot-from", type=Path,
                        help="A run whose accepted plan replies this run reuses (same chapters and options)")
    parser.add_argument("--cast-from", type=Path, help="A run whose cast-plan.json this run uses as its cast")
    parser.add_argument("--final-closure", action="store_true",
                        help="Keep the last chapter's closing step last and end its scene plan on the core's resolution")
    parser.add_argument("--closure-plan", action="store_true",
                        help="Only the scene plan half of --final-closure: no change to the plot")
    parser.add_argument("--closure-end", action="store_true",
                        help="The last chapter's scene plan ends on the chapter's last step's result, without the core")
    parser.add_argument("--closing-step-last", action="store_true",
                        help="Only the plot half of --final-closure: the last chapter's expansion adds nothing "
                             "after its last skeleton step")
    parser.add_argument("--product", action="store_true",
                        help="The product's long-form plan (script_options.long_form and chapter_scope_guard) "
                             "instead of this script's replacements: the same as --balanced --expand "
                             "--expansion-steps 6 --distinct-steps --fresh-roles --closing-step-last --closure-end, "
                             "but a cast it plans names no count, and a chapter plan sees later chapters in brief")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.chapters < 1:
        parser.error("--chapters must be at least 1.")
    replacements = ("balanced", "expand", "expansion_steps", "distinct_steps", "fresh_roles", "final_closure",
                    "closure_plan", "closure_end", "closing_step_last", "plot_from")
    if args.product and any(getattr(args, name) for name in replacements):
        parser.error("--product runs the product's plan; it takes no plan replacement but --cast-from.")
    value = experiment_input(json.loads(args.input.read_text(encoding="utf-8-sig")),
                             args.chapters, args.model_id, args.skip_staging, args.cast_limit)
    if args.product:
        value.setdefault("script_options", {}).update(long_form=True, chapter_scope_guard=True)
    elif args.chapters != 3:
        use_chapter_count(args.chapters, args.balanced)
    if args.expand:
        use_expansion(args.expansion_steps, args.distinct_steps, args.fresh_roles,
                      args.final_closure or args.closing_step_last)
    if args.final_closure or args.closure_plan:
        use_final_closure()
    elif args.closure_end:
        use_final_closure(core_text=False)
    if args.plot_from:
        reuse_plan(args.plot_from)
    if args.cast_from:
        reuse_cast(args.cast_from)
    profile = value.get("profile", {})
    if args.context_size:
        # The model route is chosen from the profile, so the context belongs there too.
        profile["context_size"] = args.context_size
    if profile.get("common_settings_version"):
        # As a production with the common settings does: the chosen model and context for every step.
        load_config = pipeline.load_config
        pipeline.load_config = lambda: select_config(load_config(), {"profile": profile}, pipeline.ROOT)
    report = run_text_debug(value, args.output_dir, workflow="causal", context_size=args.context_size,
                            chapter_limit=args.chapter_limit, resume=args.resume, seed=args.seed,
                            policy="script_continuation_v1", plot_only=args.plot_only)
    print(json.dumps({"status": report["status"], "chapters": report.get("saved_chapter_count"),
                      "story": str(args.output_dir.resolve() / "story.html")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
