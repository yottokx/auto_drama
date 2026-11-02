"""Chapter allocation fixes, run on story chains saved by probe_plot_drama.py.

Some long-form chains (ten one-step events for six chapters) failed at chapter allocation: the
answer writes each chapter's title and role before its last_event, the model gives one event to
each chapter in turn, puts the rest in the last chapter and picks a cut outside boundary_options.
This probe reuses a saved chain and cast and runs the allocation call, then stops, or with --full
goes on to the chapter expansions and writes the plan like probe_plot_drama.py:

  now    the product's allocation
  pick   the answer starts with boundary_option, the number of one of boundary_options;
         the chapters carry no last_event and get theirs from the chosen option
  code   the code takes the most even option; the model writes only the chapters' presentation
  order  the product's allocation with number and last_event before each chapter's title

Nothing in the product changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_plot_drama as drama
from probe_plot_drama import pipeline, replaced, script_continuation as sc, script_plot

from packages.contracts.script import Contract
from services.worker.generation.script_cast import CastPlan
from services.worker.generation.text_debug import run_text_debug
from services.worker.model_config import select_config

FIXES = ("now", "pick", "code", "order")
SPLIT_RULE = "各出来事の結果や選択が一区切りになる箇所で章を区切ります。"
SIZED_RULES = script_plot.EVENT_RULE + script_plot.SIZED_ALLOCATION_RULE
PICK_RULE = (
    "各章は連続した一つ以上の出来事を担当します。boundary_optionsは、区切り方ごとの番号option、"
    "各章の最後の出来事last_events、各章の手順数steps_per_chapterです。結果や選択が一区切りになる区切りのうち、"
    "各章の手順数が近いものを選びます。" + script_plot.BALANCE_RULE
)
PICK_TASK = ("回答の最初のboundary_optionに、選んだ区切り方のoptionを書きます。"
             "chaptersの各章は、選んだ区切り方でその章が担当する出来事から題名・役割・会話の材料を付けます。")


class AllocationDone(Exception):
    pass


class PickChapter(script_plot.ChapterPresentation):
    number: int = Field(ge=1, strict=True)


class PickAllocation(Contract):
    boundary_option: int = Field(ge=1, strict=True)
    chapters: list[PickChapter]
    foreshadowing: list[script_plot.Foreshadowing] = Field(max_length=30)


class CodeAllocation(Contract):
    chapters: list[PickChapter]
    foreshadowing: list[script_plot.Foreshadowing] = Field(max_length=30)


def ranges(ends: list[int]) -> str:
    return "、".join(f"第{n}章は出来事{start + 1}" + (f"〜{end}" if end > start + 1 else "")
                    for n, (start, end) in enumerate(zip([0, *ends[:-1]], ends), 1))


def tail(count: int, events: int) -> str:
    return (f"出来事は全{events}個です。last_eventはevent_numberを使い、"
            f"{count}章の値を狭義昇順にし、第{count}章の値は必ず{events}にします。")


def presentation_schema(schema: dict) -> dict:
    """The allocation schema without the chapters' last_event."""
    schema = copy.deepcopy(schema)
    boundary = schema["$defs"]["LongChapterBoundary"]
    boundary["properties"].pop("last_event")
    boundary["required"] = [key for key in boundary["required"] if key != "last_event"]
    return schema


def use_fix(fix: str, chain_from: Path, full: bool) -> None:
    saved_chain = json.loads((chain_from / "story-chain.json").read_text(encoding="utf-8"))
    forces = chain_from / "story-forces.json"
    structured_core = sc.ScriptRun.structured_core

    def saved_core(self, key, purpose, number, prompt, context, model, validate, schema):
        if key != "story-chain":
            return structured_core(self, key, purpose, number, prompt, context, model, validate, schema)
        if forces.exists():
            self.state["story_forces"] = json.loads(forces.read_text(encoding="utf-8"))
        chain_model = sc.chapter_models(self.manifest["approved_chapter_count"])[0]
        return SimpleNamespace(as_chain=lambda: chain_model.model_validate(saved_chain))

    sc.ScriptRun.structured_core = saved_core

    def stop(self, *args, **kwargs):
        raise AllocationDone

    if not full:
        sc.ScriptRun.expand_plot = stop
    if fix == "now":
        return
    options = sc.balanced_boundary_options
    if fix == "pick":
        sc.balanced_boundary_options = lambda sizes, count: [
            {"option": n, **row} for n, row in enumerate(options(sizes, count), 1)]
    elif fix == "code":
        sc.balanced_boundary_options = lambda sizes, count: options(sizes, count)[:1]
    structured = sc.ScriptRun.structured

    def allocation(self, key, purpose, number, prompt, context, model, validate=None, schema=None):
        if key != "chapter-allocation" or model.__name__ != "LongChapterAllocation":
            return structured(self, key, purpose, number, prompt, context, model, validate, schema)
        count = self.manifest["approved_chapter_count"]
        chain = self.state["story_chain"]
        events = len(chain["events"])
        rows = sc.balanced_boundary_options([len(event["steps"]) for event in chain["events"]], count)
        if fix == "order":
            boundary = schema["$defs"]["LongChapterBoundary"]
            drama.ordered(boundary, ["number", "last_event",
                                     *(key for key in boundary["properties"] if key not in ("number", "last_event"))])
            return structured(self, key, purpose, number, prompt, context, model, validate, schema)

        def with_ends(value, ends):
            return model.model_validate({
                "chapters": [{**row.model_dump(mode="json"), "last_event": end}
                             for row, end in zip(value.chapters, ends, strict=True)],
                "foreshadowing": [row.model_dump(mode="json") for row in value.foreshadowing]})

        reduced = presentation_schema(schema)
        if fix == "pick":
            reduced["properties"] = {"boundary_option": {"type": "integer", "enum": list(range(1, len(rows) + 1))},
                                     **reduced["properties"]}
            reduced["required"] = ["boundary_option", *reduced["required"]]
            prompt = replaced(prompt, [(SIZED_RULES, PICK_RULE), (tail(count, events), PICK_TASK)])
            ends_of = lambda value: rows[value.boundary_option - 1]["last_events"]
            answer_model = PickAllocation
        else:
            ends = rows[0]["last_events"]
            prompt = replaced(prompt, [
                (SPLIT_RULE + SIZED_RULES, "各章は連続した一つ以上の出来事を担当します。"),
                (tail(count, events), f"出来事は全{events}個、章は{count}章で、配分はコードで確定しています。"
                 f"{ranges(ends)}を担当します。境界番号は出力せず、担当する出来事を変えずに"
                 "各章のtitle・role・conversation_topicsを設計します。")])
            ends_of = lambda value: ends
            answer_model = CodeAllocation

        def check(value):
            if len(value.chapters) != count:
                raise ValueError(f"The allocation needs {count} chapters.")
            validate(with_ends(value, ends_of(value)))

        value = structured(self, key, purpose, number, prompt, context, answer_model, check, reduced)
        return with_ends(value, ends_of(value))

    sc.ScriptRun.structured = allocation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--chain-from", type=Path, required=True, help="A probe_plot_drama.py run")
    parser.add_argument("--arm", choices=drama.ARMS, required=True, help="The arm that made the chain")
    parser.add_argument("--fix", choices=FIXES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--full", action="store_true", help="Also expand the chapters into the whole plan")
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    plan = json.loads((args.chain_from / "cast-plan.json").read_text(encoding="utf-8"))["plan"]
    drama.use_arm(args.arm, CastPlan.model_validate(plan))
    use_fix(args.fix, args.chain_from, args.full)
    profile = value["profile"]
    load_config = pipeline.load_config
    pipeline.load_config = lambda: select_config(load_config(), {"profile": profile}, pipeline.ROOT)
    status = "failed"
    try:
        run_text_debug({key: value[key] for key in ("approval_snapshot", "profile", "script_options")},
                       args.output_dir, workflow="causal", seed=args.seed,
                       policy="script_continuation_v1", plot_only=True)
        status = "planned"
    except AllocationDone:
        status = "allocated"
    except Exception as exc:  # noqa: BLE001 - the run's own report keeps the details
        print(f"{type(exc).__name__}: {exc}"[:500], file=sys.stderr)
    if (args.output_dir / "plot.json").exists():
        markdown = (f"# {value['title']}（{args.arm}・{args.fix}）\n\n"
                    + drama.plan_markdown(args.output_dir, drama.names(value, plan)))
        (args.output_dir / "drama-plan.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": status, "output": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
