"""STEP4 with a chapter count that comes from the plot, not from the approval.

The product plans cast plan -> chain of events -> allocation to the approved three chapters.
Here the chain is planned without a chapter count or an event range, and the allocation cuts it
into as many chapters of about ten minutes as its steps need. Only the plan is made (cast plan,
chain, allocation); no chapter is written. Nothing in the product changes.

  open     the chain's wording loses the chapter count and the 3-9 event range; the event count
           follows the work (few for a small, close story, more when many people's errands
           and conflicts are involved)
  threads  open, and the chain first lists the threads the work tells (the main line and the
           supporting characters' own errands, aims and conflicts it needs) before its events

--cast-from reuses another run's cast plan, so two arms differ only in the chain; --chain-from
reuses a run's chain, so only the allocation is made again. --most-steps lifts the product's limit
of six steps an event.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.contracts.m3 import Foreshadowing
from packages.contracts.script import Contract, Identifier
from services.worker.generation import pipeline, script_continuation, script_plot
from services.worker.generation.script_cast import CastPlan, cast_limit_guide
from services.worker.generation.script_plot import (
    ChainEvent,
    ChainEventDraft,
    ChapterBoundary,
    DetailedPlot,
    StoryChain,
    StoryChainDraft,
)
from services.worker.generation.text_debug import run_text_debug
from services.worker.model_config import select_config

ARMS = ("open", "threads")
# Only bounds for the answer format; neither is shown to the model as a target.
MOST_EVENTS = 40
MOST_CHAPTERS = 30
OUTPUT_TOKENS = 8192
# A chapter of about ten minutes: with 2 steps a chapter repeated itself, with about 4 it did
# not (docs/experiments/long-form-20261019.md). The allocation picks one of the ways to cut
# the chain at event ends into chapters of MIN_STEPS-MAX_STEPS steps (or one larger event).
MIN_STEPS, MAX_STEPS = 3, 6
MOST_OPTIONS = 60


def replaced(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        if text.count(old) != 1:
            raise ValueError(f"The product wording differs from the expected text: {old}")
        text = text.replace(old, new)
    return text


CHAIN_INSTRUCTION = replaced(script_plot.CHAIN_INSTRUCTION, [
    ("全3章の分量を支える試行・応答と、日常の掛け合いを用意します。",
     "章数はまだ決めていません。この作品に要る出来事の量から、後で章数を決めます。"
     "試行・応答と、日常の掛け合いを用意します。"),
    ("全体をまだ章ごとに分けず、3〜9個の出来事へまとめます。この個数は場面数や均等配分を指定するものではありません。",
     "全体をまだ章ごとに分けません。出来事の数に目安はありません。少人数の密な話なら少なく、"
     "多くの人物の用事や対立が絡む話なら、それに見合う数にします。"),
    ("3章の各区間で新しく経験することと到達状態が変わる連鎖にします。同じ状態の再確認を重ねて章数を埋めません。",
     "各出来事で新しく経験することと到達状態が変わる連鎖にします。同じ状態の再確認を重ねて分量を埋めません。"),
])
THREADS_ORDER = ("出力はcore、resolution_basis、opening_condition、eventsの順です。",
                 "出力はcore、threads、resolution_basis、opening_condition、eventsの順です。"
                 "threadsには、この作品で描く筋を挙げます。主筋と、サブや脇の人物自身の用事・目的・対立のうち、"
                 "この物語に要るものです。各筋は担う人物のIDと、誰が何を求め、主筋とどう関わるかを1文で記します。"
                 "少人数の密な話なら主筋だけでも構いません。eventsでは、挙げた筋をそれぞれの人物の行動として進め、決着まで描きます。")
THREADED_CHAIN_INSTRUCTION = replaced(CHAIN_INSTRUCTION, [THREADS_ORDER])

ALLOCATION_INSTRUCTION = replaced(script_plot.ALLOCATION_INSTRUCTION, [
    ("保存済みの出来事の連鎖を、順番と内容を保って全3章へ配分します。",
     "保存済みの出来事の連鎖を、順番と内容を保って章に分けます。章数はまだ決まっていません。"),
    (script_plot.EQUAL_SHARE_NOT_NEEDED,
     "1章は約10分の1話です。手順の数と内容から、1話で会話と応答を十分に描ける量ごとに区切り、その数を章数にします。"
     "1章あたり手順3〜5個を目安にし、重い決着や大きな転機を含む章は少なめ、短いやり取りが続く章は多めにします。"
     "手順が少なすぎる章は同じやり取りの繰り返しで埋まり、多すぎる章は1話に収まりません。"),
])


class StoryThread(Contract):
    character_ids: list[Identifier] = Field(min_length=1, max_length=4)
    thread: str = Field(min_length=1, max_length=160,
        description="誰が何を求め・抱え、主筋とどう関わるか。1文。")


class OpenStoryChain(StoryChain):
    events: list[ChainEvent] = Field(min_length=1, max_length=MOST_EVENTS,
        description="作品に要る主要な出来事の連鎖。章境界はまだ決めない。各出来事は場面とは別。")


class OpenStoryChainDraft(StoryChainDraft):
    events: list[ChainEventDraft] = Field(min_length=1, max_length=MOST_EVENTS,
        description="作品に要る主要な出来事の連鎖。全て予定であり、章境界はまだ決めない。")

    def as_chain(self) -> OpenStoryChain:
        # StoryChainDraft.as_chain, building the open-length chain.
        events, condition = [], self.opening_condition
        for event in self.events:
            events.append(ChainEvent(start_condition=condition, steps=event.steps))
            condition = event.steps[-1].result
        return OpenStoryChain(core=self.core, events=events)


class ThreadedStoryChainDraft(OpenStoryChainDraft):
    threads: list[StoryThread] = Field(min_length=1, max_length=12,
        description="この作品で描く筋。主筋と、物語に要るサブ・脇の人物自身の用事・目的・対立。")


class OpenChapterBoundary(ChapterBoundary):
    number: int = Field(ge=1, le=MOST_CHAPTERS, strict=True)
    last_event: int = Field(ge=1, le=MOST_EVENTS, strict=True,
                            description=ChapterBoundary.model_fields["last_event"].description)


class OpenChapterAllocation(script_plot.ChapterAllocation):
    chapters: list[OpenChapterBoundary] = Field(min_length=1, max_length=MOST_CHAPTERS)
    foreshadowing: list[Foreshadowing] = Field(max_length=30)


def chain_schema(model) -> dict:
    schema = model.model_json_schema()
    if "threads" in schema["properties"]:
        # The threads come right after the core, before the events they are told through.
        properties = schema["properties"]
        order = ["core", "threads", *(key for key in properties if key not in {"core", "threads"})]
        schema["properties"] = {key: properties[key] for key in order}
    return schema


def allocate(chain: OpenStoryChain, allocation: OpenChapterAllocation) -> DetailedPlot:
    count = len(allocation.chapters)
    if [row.number for row in allocation.chapters] != list(range(1, count + 1)):
        raise ValueError("Chapter numbers must run from 1 without gaps.")
    ends = [row.last_event for row in allocation.chapters]
    if ends != sorted(set(ends)) or ends[-1] != len(chain.events):
        raise ValueError("Chapter boundaries must partition every event exactly once in order.")
    starts = [0, *ends[:-1]]
    check_core = script_plot.check_core
    # _plot_from_chapters checks foreshadowing against three chapters.
    script_plot.check_core = lambda core, characters, _count: check_core(core, characters, count)
    try:
        return script_plot._plot_from_chapters(
            chain, allocation.chapters,
            [chain.events[start:end] for start, end in zip(starts, ends, strict=True)],
            allocation.foreshadowing)
    finally:
        script_plot.check_core = check_core


def boundary_options(sizes: list[int]) -> list[dict]:
    """Every cut at event ends whose chapters hold MIN_STEPS-MAX_STEPS steps (a larger single
    event stays one chapter); with none, the whole chain is one chapter. When there are more
    than MOST_OPTIONS, the ones whose chapters are the most even are kept."""
    found = []

    def walk(start, ends, steps):
        if start == len(sizes):
            found.append((ends, steps))
            return
        total = 0
        for end in range(start + 1, len(sizes) + 1):
            total += sizes[end - 1]
            if total > MAX_STEPS and end - start > 1:
                break
            if total >= MIN_STEPS or end == len(sizes) and not ends:
                walk(end, [*ends, end], [*steps, total])
        if len(found) > 5000:
            raise ValueError("Too many ways to cut the chain.")

    walk(0, [], [])
    if not found:
        found = [([len(sizes)], [sum(sizes)])]
    found.sort(key=lambda row: (max(row[1]) - min(row[1]), len(row[0])))
    return [{"last_events": ends, "steps_per_chapter": steps} for ends, steps in found[:MOST_OPTIONS]]


def allow_steps(most: int) -> None:
    """Let an event hold up to ``most`` steps, in the chain and in every model holding it."""
    from annotated_types import MaxLen
    from pydantic import BaseModel

    for cls in (ChainEvent, ChainEventDraft):
        field = copy.copy(cls.model_fields["steps"])
        field.metadata = [MaxLen(most) if isinstance(item, MaxLen) else item for item in field.metadata]
        cls.model_fields["steps"] = field
        cls.model_rebuild(force=True)
    # Models holding an event keep the old limit until rebuilt.
    for module in (script_plot, sys.modules[__name__]):
        for value in list(vars(module).values()):
            if isinstance(value, type) and issubclass(value, BaseModel) and value.__module__ == module.__name__:
                value.model_rebuild(force=True)


def use_variable_chapters(arm: str, cast: CastPlan | None, chain_from: Path | None = None) -> None:
    sc = script_continuation
    draft_model = ThreadedStoryChainDraft if arm == "threads" else OpenStoryChainDraft
    chain_instruction = THREADED_CHAIN_INSTRUCTION if arm == "threads" else CHAIN_INSTRUCTION

    setting = sc.ScriptRun.setting

    def setting_without_count(self, *args, **kwargs):
        # The approval's chapter count is only this run's bound; the plan must not see it.
        result = setting(self, *args, **kwargs)
        result["world"] = {key: value for key, value in result["world"].items() if key != "chapterCount"}
        return result

    cast_instruction = sc.cast_instruction

    def cast_instruction_without_count(chapters, room):
        text = cast_instruction(chapters, room)
        return replaced(text, [(f"全{chapters}章の物語で", "物語で")]) if room else text

    structured = sc.ScriptRun.structured

    def reused_cast(self, key, purpose, number, prompt, context, model, validate=None, schema=None):
        if key == "cast-plan" and cast is not None:
            return validate(cast) if validate else cast
        if key.startswith("story-chain") and chain_from is not None:
            # The saved chain as a draft: its opening, events and threads (resolution_basis is
            # a scaffold the chain does not keep).
            saved = json.loads((chain_from / "story-chain.json").read_text(encoding="utf-8"))
            threads = chain_from / "story-threads.json"
            draft = model.model_validate({
                "core": saved["core"], "resolution_basis": [],
                "opening_condition": saved["events"][0]["start_condition"],
                "events": [{"steps": event["steps"]} for event in saved["events"]],
                **({"threads": json.loads(threads.read_text(encoding="utf-8"))} if "threads" in model.model_fields else {})})
            if validate:
                validate(draft)
            return draft
        return structured(self, key, purpose, number, prompt, context, model, validate, schema)

    cast_plan = sc.ScriptRun.cast_plan

    def any_count_cast_plan(self):
        # The product skips the cast plan for counts other than three unless the run comes
        # from a plan approval; mark the run as one for this call only.
        added = "planning_protocol" not in self.manifest
        if added:
            self.manifest["planning_protocol"] = "variable-chapters-experiment"
        try:
            return cast_plan(self)
        finally:
            if added:
                del self.manifest["planning_protocol"]

    def outline(self):
        cast_value = self.cast_plan()
        setting = self.setting(planned=True)
        characters = set(setting["main_character_ids"])
        available = characters | {row.id for row in cast_value.supporting_characters}
        outfits = self.outfits
        events_note = sc.OUTFIT_EVENTS if outfits else ""
        allocation_note = sc.OUTFIT_ALLOCATION if outfits else ""

        def check(plot):
            count = len(plot.chapters)
            sc.check_core(plot.core, characters, count)
            sc.check_chapters(plot.chapters, list(range(1, count + 1)))
            if outfits:
                sc.check_outfit_appearances(plot.chapters, outfits)
            for chapter in plot.chapters:
                for topic in chapter.conversation_topics:
                    if len(set(topic.character_ids)) != len(topic.character_ids) or not set(topic.character_ids) <= available:
                        raise ValueError("Conversation topics need distinct registered participants.")

        if "plot" in self.state:
            plot = DetailedPlot.model_validate(self.state["plot"])
            check(plot)
            chain = OpenStoryChain.model_validate(self.state["story_chain"])
            allocation = OpenChapterAllocation.model_validate(self.state["chapter_allocation"])
            if (sc.digest(self.state["story_chain"]) != self.state["story_chain_sha256"]
                    or allocate(chain, allocation) != plot
                    or sc.digest(self.state["plot"]) != self.state["plot_sha256"]):
                raise ValueError("Saved plot differs from its story chain or chapter allocation.")
            return plot.as_outline()

        context = {"setting": setting, "outline": "", "brief": self.options.for_stage("chain"),
                   "chapters": [], "notes": []}
        schema = chain_schema(draft_model)
        sc.bind_core_cast(schema, characters, core="ChainCore", character="ChainCharacter")
        schema["$defs"]["ChainStep"]["properties"]["character_id"]["enum"] = sorted(available)
        if "StoryThread" in schema["$defs"]:
            schema["$defs"]["StoryThread"]["properties"]["character_ids"]["items"]["enum"] = sorted(available)

        def check_draft(value):
            sc.check_chain(value.as_chain(), characters, available)
            for row in getattr(value, "threads", []):
                if not set(row.character_ids) <= available:
                    raise ValueError("Threads must name registered characters.")

        draft = self.structured_core("story-chain", "script-outline", 1, chain_instruction + events_note,
                                     context, draft_model, check_draft, schema)
        chain = draft.as_chain()
        chain_data = chain.model_dump(mode="json")
        self.state.update(story_chain=chain_data, story_chain_sha256=sc.digest(chain_data))
        if isinstance(draft, ThreadedStoryChainDraft):
            # A scaffold like resolution_basis: kept as a record, never passed downstream.
            self.state["story_threads"] = [row.model_dump(mode="json") for row in draft.threads]
            sc.write_json(self.output / "story-threads.json", self.state["story_threads"])
        self.persist()
        sc.write_json(self.output / "story-chain.json", chain_data)
        events = len(chain.events)
        steps = sum(len(event.steps) for event in chain.events)
        material = {"core": chain.core.model_dump(mode="json"), "event_count": events, "total_steps": steps,
                    "events": [{"event_number": n, "step_count": len(event.steps), **event.model_dump(mode="json")}
                               for n, event in enumerate(chain.events, 1)],
                    "boundary_options": boundary_options([len(event.steps) for event in chain.events])}
        allocation_context = {**context, "brief": self.options.for_stage("allocation")
                              + "\n章に分ける保存済みプロット（全て未実施）:\n" + json.dumps(material, ensure_ascii=False)}
        allowed = [row["last_events"] for row in material["boundary_options"]]
        schema = OpenChapterAllocation.model_json_schema()
        schema["properties"]["chapters"]["maxItems"] = min(events, MOST_CHAPTERS)
        schema["$defs"]["OpenChapterBoundary"]["properties"]["last_event"]["maximum"] = events
        schema["$defs"]["OpenChapterBoundary"]["properties"]["number"]["maximum"] = min(events, MOST_CHAPTERS)
        schema["$defs"]["ConversationTopic"]["properties"]["character_ids"]["items"]["enum"] = sorted(available)
        if outfits:
            sc.add_outfit_appearances(schema, outfits)
        def check_allocation(plot_value, value):
            check(plot_value)
            if [row.last_event for row in value.chapters] not in allowed:
                raise ValueError("Chapter boundaries must be one of boundary_options' last_events: "
                                 + json.dumps(allowed))

        allocation = self.structured("chapter-allocation", "script-allocation", 1,
            ALLOCATION_INSTRUCTION + allocation_note + f"出来事は全{events}個、手順は全{steps}個です。"
            "boundary_optionsは、各章の手順が1話に収まる区切り方の一覧です（last_eventsと各章の手順数steps_per_chapter）。"
            "出来事の結果や選択が一区切りになり、各章の内容が1話にまとまるものを一つ選び、"
            "そのlast_eventsの通りに章を作ります。章数は選んだ区切り方の長さです。numberは1から順に振ります。"
            f"last_eventはevent_numberを使い、章の順に狭義昇順にし、最終章の値は必ず{events}にします。",
            allocation_context, OpenChapterAllocation, lambda value: check_allocation(allocate(chain, value), value), schema)
        plot = allocate(chain, allocation)
        self.state["chapter_allocation"] = allocation.model_dump(mode="json")
        sc.write_json(self.output / "chapter-allocation.json", self.state["chapter_allocation"])
        result = plot.as_outline()
        self.state.update(plot=plot.model_dump(mode="json"), outline=result.model_dump(mode="json"))
        self.state["plot_sha256"] = sc.digest(self.state["plot"])
        self.persist()
        sc.write_json(self.output / "plot.json", self.state["plot"])
        sc.write_json(self.output / "outline.json", self.state["outline"])
        return result

    sc.ScriptRun.setting = setting_without_count
    sc.cast_instruction = cast_instruction_without_count
    sc.ScriptRun.structured = reused_cast
    sc.ScriptRun.cast_plan = any_count_cast_plan
    sc.ScriptRun.outline = outline


def experiment_input(value: dict, profile: dict) -> dict:
    """The work's approval with the product's current plan options and the chosen model."""
    snapshot = copy.deepcopy(value.get("approval_snapshot", value))
    # Only the run's bound; setting_without_count hides it from every request.
    snapshot["world"]["result"]["chapterCount"] = MOST_CHAPTERS
    return {"approval_snapshot": snapshot, "profile": profile,
            "script_options": {"planned_cast_appears": True, "open_scene_cast": True,
                               "cast_limit": cast_limit_guide(profile)},
            # Many chapters' titles, roles and topics need more than the 2,048-token allocation answer.
            "profiles": {"script-allocation": {"max_tokens": OUTPUT_TOKENS}}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="An approval snapshot, or a JSON holding one under approval_snapshot")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--model-id", default="gemma-4-31B-it-UD-Q4_K_XL")
    parser.add_argument("--reasoning", default="none")
    parser.add_argument("--context-size", type=int, default=32768)
    parser.add_argument("--cast-from", type=Path, help="A run whose cast-plan.json this run reuses")
    parser.add_argument("--chain-from", type=Path, help="A run whose story chain this run reuses (same arm)")
    parser.add_argument("--most-steps", type=int, help="Steps an event may hold (product: 6)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    profile = {"provider": "local", "model_id": args.model_id, "context_size": args.context_size,
               "temperature": 0.65, "top_p": 0.95, "reasoning_level": args.reasoning,
               "common_settings_version": 1}
    value = experiment_input(json.loads(args.input.read_text(encoding="utf-8-sig")), profile)
    cast = (CastPlan.model_validate(json.loads((args.cast_from / "cast-plan.json").read_text(encoding="utf-8"))["plan"])
            if args.cast_from else None)
    if args.most_steps:
        allow_steps(args.most_steps)
    use_variable_chapters(args.arm, cast, args.chain_from)
    # As a production with the common settings does: the chosen model and context for every step.
    load_config = pipeline.load_config
    pipeline.load_config = lambda: select_config(load_config(), {"profile": profile}, pipeline.ROOT)
    report = run_text_debug(value, args.output_dir, workflow="causal", resume=args.resume, seed=args.seed,
                            policy="script_continuation_v1", plot_only=True)
    plot = json.loads((args.output_dir / "plot.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": report["status"], "chapters": len(plot["chapters"]),
                      "plot": str(args.output_dir.resolve() / "plot.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
