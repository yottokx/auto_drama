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
from .script_continuation import POLICY, ScriptRun
from .script_examples import example_catalog_identity
from .script_plot import DetailedPlot, PlotChapter, bind_core_cast
from .script_production import ProductionScriptRun, _generation_identity_config
from .workflow_version import generator_protocol


class ConnectionsRevision(Contract):
    connections: list[CastConnection] = Field(max_length=30)


class CommonPlanRun(ScriptRun):
    retry_failed_steps = ProductionScriptRun.retry_failed_steps

    def __init__(self, output, config, payload):
        count = payload["approval_snapshot"]["world"]["result"]["chapterCount"]
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
            self._validate_revision_request(payload, self.source)
            identity.update(source_plan_sha256=digest(self.source), revision={key: payload.get(key)
                            for key in ("target", "character_id", "chapter_number", "instruction")})
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
            install_plan(state, output, manifest, self.source,
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
                     or self.state.get("plot") != self.source["plot"])):
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
        main = {row["result"]["id"] for row in self.payload["approval_snapshot"]["characters"]}
        if target in {"all", "plot"} and not number:
            bind_core_cast(schema, main, core="PlotCore", character="PlotCharacter")
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
                content = data
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

        value = self.structured("common-plan-revision", "script-outline", 1, instruction + scope,
                                context, model, check_revision, schema)
        return combine(value)

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
            elif self.source is not None:
                content = self.revise()
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
