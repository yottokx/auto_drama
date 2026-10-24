"""Durable STEP4 common planning/revisions without chapter or asset generation."""
from __future__ import annotations

import copy
import time

from pydantic import Field

from packages.contracts.m2 import CharacterResult
from packages.contracts.planning import CommonPlanContent, planning_protocol, validate_plan_content
from packages.contracts.script import Contract

from .cancellation import GenerationCancelled, check_cancelled
from .causal_runtime import digest
from .draft_story import _read
from .execution_settings import enabled, rebind_manifest, semantic_identity
from .llm import write_json
from .planning_state import install_plan
from .script_cast import CastConnection, ScriptOptions
from .script_chapter_count import (
    CHAPTER_COUNT_EXAMPLES,
    CHAPTER_COUNT_SYSTEM,
    CHAPTER_COUNT_TASK,
    ChapterCountJudgment,
)
from .script_continuation import POLICY, ScriptRun
from .script_examples import example_catalog_identity
from .script_plot import (
    OUTFIT_ALLOCATION,
    DetailedPlot,
    PlotChapter,
    add_outfit_appearances,
    bind_core_cast,
)
from .script_production import ProductionScriptRun, _generation_identity_config
from .workflow_version import generator_protocol


class ConnectionsRevision(Contract):
    connections: list[CastConnection] = Field(max_length=30)


class CommonPlanRun(ScriptRun):
    retry_failed_steps = ProductionScriptRun.retry_failed_steps

    def __init__(self, output, config, payload):
        count = payload["approval_snapshot"]["world"]["result"]["chapterCount"]
        # A long-form plan's chosen count; the coordinator sends it as the snapshot's count too.
        self.chosen = payload.get("chapter_count")
        if self.chosen is not None and (self.chosen != count or payload.get("plan_content") is None):
            raise ValueError("A chosen chapter count plans the plot of a cast, as its snapshot's count.")
        identity = {"schema_version": 1, "execution_mode": "planning", "workflow_policy": POLICY,
                    "generator_protocol": generator_protocol("causal", POLICY),
                    "planning_protocol": planning_protocol(),
                    "approval_snapshot": payload["approval_snapshot"], "approved_chapter_count": count,
                    "seed": payload["seed"], "profile": payload.get("profile", {}),
                    "profiles": payload.get("profiles", {}), "workflow_limits": payload.get("workflow_limits", {}),
                    "generation_config": _generation_identity_config(config, payload),
                    "script_options": ScriptOptions.model_validate(payload.get("script_options", {})).model_dump(),
                    "script_examples": example_catalog_identity(),
                    "planning_id": payload.get("planning_id"), "planning_revision": payload.get("planning_revision", 0)}
        self.source = None
        if payload.get("plan_content") is not None:
            self.source = validate_plan_content(payload["plan_content"], payload["approval_snapshot"]).model_dump(mode="json")
            if payload.get("judge_only"):
                # The cast was edited by hand before its count was chosen: judge the count again.
                if "plot" in self.source or self.chosen is not None:
                    raise ValueError("Only a long-form plan's cast is judged again.")
                identity["judge_only"] = True
            elif self.chosen is None:
                self._validate_revision_request(payload, self.source)
                identity.update(revision={key: payload.get(key)
                                for key in ("target", "character_id", "chapter_number", "instruction")})
            else:
                if "chapter_judgment" not in self.source:
                    raise ValueError("Only a long-form plan's cast is planned at a chosen chapter count.")
                identity["chapter_count"] = self.chosen
            identity["source_plan_sha256"] = digest(self.source)
        if (ScriptOptions.model_validate(payload.get("script_options", {})).long_form and self.chosen is None
                and (self.source is None or "plot" not in self.source)):
            # The cast and the judgment of how many chapters it needs see no chapter count.
            identity["chapter_count_chosen"] = False
        if enabled(payload):
            identity.update(execution_settings_version=1,
                            execution_settings_revision=payload.get("execution_settings_revision", 0))
        manifest = {**identity, "input_sha256": digest(
            semantic_identity(identity) if enabled(payload) else identity)}
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "experiment.json"
        if manifest_path.exists() and _read(manifest_path) != manifest:
            if not enabled(payload):
                raise ValueError("Common planning input or model configuration changed.")
            manifest = rebind_manifest(output, _read(manifest_path), manifest)
        write_json(manifest_path, manifest)
        journal = output / "draft-state.json"
        if self.source is not None and not journal.exists():
            if any(p.is_file() and p != manifest_path for p in output.rglob("*")):
                raise ValueError("Planning journal is missing beside generated material.")
            state = {"input_sha256": manifest["input_sha256"], "steps": {}, "chapters": [],
                     "notes": [], "fallbacks": [], "sessions": [], "request_ordinal": 0, "partial": None}
            # At a chosen count only the cast is kept; its plot is planned anew.
            installed = self.source if self.chosen is None else {
                "cast_plan": self.source["cast_plan"], "chapter_judgment": self.source["chapter_judgment"]}
            install_plan(state, output, manifest, installed,
                         {"sha256": digest(self.source), "planning_protocol": planning_protocol()})
            write_json(journal, state)
        super().__init__(output, manifest, config, payload, 1)

    @staticmethod
    def _validate_revision_request(payload, source):
        target = payload.get("target", "all")
        if target not in {"all", "plot", "character", "relationships"}:
            raise ValueError("Unknown common plan revision target.")
        if not isinstance(payload.get("instruction"), str) or not payload["instruction"].strip():
            raise ValueError("Common plan revision requires an instruction.")
        if target == "character":
            if payload.get("character_id") not in {row["id"] for row in source["cast_plan"]["supporting_characters"]}:
                raise ValueError("Character revision must identify a planned supporting character.")
        elif payload.get("character_id") is not None:
            raise ValueError("character_id applies only to character revision.")
        if "plot" not in source and (target not in {"character", "relationships"}
                                     or payload.get("chapter_number") is not None):
            raise ValueError("A plan without its plot revises only its supporting characters.")
        if payload.get("chapter_number") is not None:
            number = payload["chapter_number"]
            if (target != "plot" or type(number) is not int
                    or not 1 <= number <= len(source["plot"]["chapters"])):
                raise ValueError("chapter_number applies only to a planned plot chapter.")

    def _verify_narratives(self):
        if self.state["chapters"] or self.state.get("partial"):
            raise ValueError("Common planning cannot contain chapter bodies.")
        if (self.source is not None
                and (self.state.get("cast_plan", {}).get("plan") != self.source["cast_plan"]
                     or (self.chosen is None and self.state.get("plot") != self.source.get("plot")))):
            raise ValueError("Common plan revision source was changed.")
        saved = self.state.get("common_plan_result")
        if saved:
            value = validate_plan_content(saved["content"], self.payload["approval_snapshot"])
            if digest(value.model_dump(mode="json")) != saved["sha256"]:
                raise ValueError("Saved common plan result was changed.")

    def revise(self):
        source = copy.deepcopy(self.source)
        target = self.payload.get("target", "all")
        number = self.payload.get("chapter_number")
        cid = self.payload.get("character_id")
        model = {"all": CommonPlanContent, "plot": PlotChapter if number else DetailedPlot,
                 "character": CharacterResult, "relationships": ConnectionsRevision}[target]
        schema = model.model_json_schema()
        if target == "all":
            # The whole plan is its cast and its plot; a long-form plan's count stays as chosen.
            schema["properties"] = {"cast_plan": schema["properties"]["cast_plan"],
                                    "plot": {"$ref": "#/$defs/DetailedPlot"}}
            schema["required"] = ["cast_plan", "plot"]
            schema["$defs"].pop("ChapterCountJudgment")
        main = {row["result"]["id"] for row in self.payload["approval_snapshot"]["characters"]}
        if target in {"all", "plot"} and not number:
            bind_core_cast(schema, main, core="PlotCore", character="PlotCharacter")
        outfits = self.outfits if target in {"all", "plot"} else {}
        if outfits:
            add_outfit_appearances(schema, outfits)
        context = {"setting": {"approved_setting": self.setting(planned=True), "current_common_plan": source},
                   "outline": "", "brief": self.payload["instruction"], "chapters": [], "notes": [],
                   "optional_example": None}
        instruction = ("ユーザーの指示に従い、未執筆の全体計画を修正します。承認済み世界・メインキャラ・"
                       "メイン同士の関係は維持し、登録人物IDと全章の数・順序を維持します。本文や素材は生成しません。")
        scope = {"all": "全体プロットと予定サブキャラ・関係を返します。",
                 "plot": f"第{number}章だけを返します。章番号は維持します。" if number else "全体プロットだけを返します。",
                 "character": f"サブキャラ{cid}の完全な設定だけを返します。IDは維持します。",
                 "relationships": "サブキャラを含む関係connectionsだけを返します。"}[target]

        def combine(value):
            content = copy.deepcopy(source)
            data = value.model_dump(mode="json")
            if target == "all":
                content = {**data, **{key: source[key] for key in ("chapter_count", "chapter_judgment")
                                      if key in source}}
            elif target == "plot":
                if number:
                    if value.number != number:
                        raise ValueError("A targeted plot revision must preserve its chapter number.")
                    content["plot"]["chapters"][number - 1] = data
                else:
                    content["plot"] = data
            elif target == "character":
                if value.id != cid:
                    raise ValueError("A character revision must preserve the selected character ID.")
                content["cast_plan"]["supporting_characters"] = [data if row["id"] == cid else row
                                                               for row in content["cast_plan"]["supporting_characters"]]
            else:
                content["cast_plan"]["connections"] = data["connections"]
            validate_plan_content(content, self.payload["approval_snapshot"])
            return content

        def check_revision(row):
            combine(row)

        if self.options.planned_cast_appears and target in {"all", "plot"} and not number:
            # Supporting characters added or replaced by a revision would otherwise stay out of the old events.
            scope += ("予定サブキャラは全員、いずれかの章のeventsのstepsで実際に行動させます。"
                      "一度も登場しない人物を残さず、追加・変更した人物の出番もプロットへ入れます。")
        if outfits:
            scope += OUTFIT_ALLOCATION + "current_common_planのoutfit_appearancesは、指示で変えない限り維持します。"
        value = self.structured("common-plan-revision", "script-outline", 1, instruction + scope,
                                context, model, check_revision, schema)
        return combine(value)

    def judge_chapter_count(self, cast_plan: dict) -> dict:
        """How many chapters of about ten minutes the cast and setting need (script_chapter_count)."""
        context = {"setting": self.setting(planned=True, cast_plan=cast_plan), "outline": "",
                   "brief": CHAPTER_COUNT_EXAMPLES + CHAPTER_COUNT_TASK, "chapters": [], "notes": [],
                   "optional_example": None}
        extra = {"response_format": {"type": "json_schema", "json_schema": {
            "name": "chapter-count", "strict": True, "schema": ChapterCountJudgment.model_json_schema()}}}
        reply = self.call("chapter-count", "script-cast", 1, CHAPTER_COUNT_SYSTEM, context, extra=extra)
        value = ChapterCountJudgment.model_validate_json(reply["content"])
        self.progress_step("chapter-count", "script-cast", 1, "completed")
        return value.model_dump(mode="json")

    def execute(self, llm):
        self.llm, self.started = llm, time.monotonic()
        llm.requests = self.state["request_ordinal"]
        llm._load_number = max((int(path.name.split("-", 1)[0]) for path in (llm.output / "runtimes").glob("*")
                               if path.name.split("-", 1)[0].isdigit()), default=0)
        self.session = {"status": "running", "elapsed_seconds": 0, "chapter_limit": 0,
                        "requested_phase": "plot"}
        self.state["sessions"].append(self.session)
        status, error = "running", None
        try:
            self.save(status)
            self.restore_progress()
            check_cancelled()
            self.check_budget(1)
            saved = self.state.get("common_plan_result")
            if saved is not None:
                content = saved["content"]
            elif self.chosen is not None:
                # A long-form plan's cast at the chosen count: its plot is planned now.
                self.outline()
                content = {"cast_plan": self.state["cast_plan"]["plan"], "plot": self.state["plot"],
                           "chapter_count": self.chosen, "chapter_judgment": self.source["chapter_judgment"]}
            elif self.payload.get("judge_only"):
                content = {**self.source, "chapter_judgment": self.judge_chapter_count(self.source["cast_plan"])}
            elif self.source is not None:
                content = self.revise()
                if "plot" not in content:
                    # The cast changed before its count was chosen: judge the count again.
                    content["chapter_judgment"] = self.judge_chapter_count(content["cast_plan"])
            elif self.options.long_form:
                self.cast_plan()
                cast = self.state["cast_plan"]["plan"]
                content = {"cast_plan": cast, "chapter_judgment": self.judge_chapter_count(cast)}
            else:
                self.outline()
                content = {"cast_plan": self.state["cast_plan"]["plan"], "plot": self.state["plot"]}
            self.progress_step("plan-validation", "", 1, "running", stage="validation")
            content = validate_plan_content(content, self.payload["approval_snapshot"]).model_dump(mode="json")
            self.state["common_plan_result"] = {"content": content, "sha256": digest(content)}
            self.persist()
            self.progress_step("plan-validation", "", 1, "completed", stage="validation")
            status = "plot_complete"
            check_cancelled()
            return content
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, GenerationCancelled)) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, (KeyboardInterrupt, GenerationCancelled)):
                self.fail_progress()
            raise
        finally:
            self.session["status"] = status
            self.save(status, error)

    def save(self, status, error=None):
        report = super().save(status, error)
        report.update(execution_mode="planning", requested_phase="revision" if self.source else "planning",
                      exported_chapter_count=0, planning_protocol=planning_protocol())
        write_json(self.output / "report.json", report)
        return report
