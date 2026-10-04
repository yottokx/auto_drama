"""Six single-shot craft comparisons using an unchanged completed r12 baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.narrative import parse_scene_text
from services.worker.generation import narrative, pipeline
from services.worker.generation.causal_runtime import digest
from services.worker.generation.llm import write_json
from services.worker.generation.model_routing import RoutedLLM
from services.worker.generation.processes import gpu_lock
from services.worker.generation.script_budget import scene_output_budget
from services.worker.generation.script_cast import CastPlan, ScriptOptions, script_metrics
from services.worker.generation.script_chapter_plan import FirstChapterPlan, project_plan
from services.worker.generation.script_context import SOURCE_RULES, fit_context
from services.worker.generation.script_continuation import (
    PROGRESSION,
    SCENE_INSTRUCTION,
    SCRIPT_SYSTEM,
    ScriptRun,
)
from services.worker.generation.script_examples import example_catalog_identity, example_for
from services.worker.generation.script_inputs import scene_material
from services.worker.generation.script_plot import CHAIN_INSTRUCTION, StoryChainDraft, check_chain
from services.worker.generation.text_debug import extract_input
from services.worker.generation.workflow_version import generator_protocol


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def successful_attempt(state, key):
    return next(row for row in reversed(state["steps"][key]["attempts"])
                if row["status"] == "completed" and not row.get("validation_error"))


def response_format(purpose, schema):
    return {"response_format": {"type": "json_schema", "json_schema": {
        "name": purpose, "strict": True, "schema": schema}}}


def without_basis(instruction, schema):
    """Ablate only the internal scaffold and its corresponding prompt references."""
    schema = copy.deepcopy(schema)
    del schema["properties"]["resolution_basis"]
    schema["required"].remove("resolution_basis")
    del schema["$defs"]["ResolutionBasis"]
    instruction = instruction.replace("core、resolution_basis、opening_condition、events", "core、opening_condition、events")
    start = instruction.index("次にresolution_basisで")
    end = instruction.index("その下書きを受けてeventsを作ります。")
    # Keep the setting/knowledge guard that occurs within the scaffold guidance.
    guard = instruction[instruction.index("承認設定と矛盾する力や条件を", start):end]
    instruction = instruction[:start] + guard + instruction[end:]
    instruction = instruction.replace("その下書きを受けてeventsを作ります。", "coreを受けてeventsを作ります。")
    instruction = instruction.replace("下書きに書いたこと自体は出来事の実施を意味しません。", "")
    return instruction, schema


@dataclass
class Condition:
    key: str
    purpose: str
    system: str
    context: dict
    extra: dict
    validate: Callable
    comparison: str
    output_estimate: dict | None = None

    def material(self):
        return {"key": self.key, "purpose": self.purpose, "system": self.system,
                "context": self.context, "extra": self.extra, "comparison": self.comparison,
                "output_estimate": self.output_estimate}


def prepare_conditions(source, manifest, state, payload, config):
    """Build all six independent inputs without a model or baseline writes."""
    reader = object.__new__(ScriptRun)
    reader.payload, reader.manifest, reader.config = payload, manifest, config
    ScriptOptions.model_validate(manifest["script_options"])
    reader.state = copy.deepcopy(state)
    reader.state.update(chapters=[], notes=[], locations={})
    setting = reader.setting(planned=True)
    main = set(setting["main_character_ids"])
    available = {row["id"] for row in setting["characters"]}
    schema = StoryChainDraft.model_json_schema()
    schema["$defs"]["ChainCharacter"]["properties"]["character_id"]["enum"] = sorted(main)
    schema["$defs"]["ChainStep"]["properties"]["character_id"]["enum"] = sorted(available)
    chain_context = {"setting": setting, "outline": "", "brief": reader.options.for_stage("chain"),
                     "chapters": [], "notes": [], "optional_example": example_for("script-outline")}
    chain_seed = read(source / "requests/story-chain-1.json")["request"]["seed"]
    conditions = []
    for basis in (False, True):
        instruction, current_schema = (CHAIN_INSTRUCTION, schema) if basis else without_basis(CHAIN_INSTRUCTION, schema)

        def validate_chain(text, basis=basis):
            data = json.loads(text)
            if not basis:
                if "resolution_basis" in data:
                    raise ValueError("No-basis condition returned the excluded field.")
                data = {**data, "resolution_basis": []}
            draft = StoryChainDraft.model_validate(data)
            check_chain(draft.as_chain(), main, available)
            return {"draft": draft.model_dump(), "chain": draft.as_chain().model_dump()}

        conditions.append(Condition("plot-basis-" + ("on" if basis else "off"), "script-outline",
            SCRIPT_SYSTEM + PROGRESSION + instruction, copy.deepcopy(chain_context),
            {**response_format("script-outline", current_schema), "seed": chain_seed}, validate_chain,
            "Same new instruction/example, except basis field and scaffold-specific instruction references; not a prompt-identical comparison."))

    captured = {}
    saved_plan = FirstChapterPlan.model_validate_json(successful_attempt(state, "plan-001")["reply"]["content"])

    def capture(key, purpose, number, prompt, context, model, validate=None, schema=None):
        captured.update(system=SCRIPT_SYSTEM + PROGRESSION + prompt, context=copy.deepcopy(context),
                        schema=schema or model.model_json_schema(), validate=validate)
        return saved_plan

    reader.structured = capture
    # This reader captures request material without running a production job or
    # owning its progress journal. Keep the diagnostic's baseline read-only.
    reader.progress_step = lambda *args, **kwargs: None
    reader.chapter_plan(1, reader.material(setting, None, 1), [])
    saved_request = read(source / "requests/plan-001-1.json")["request"]
    old_system = saved_request["messages"][0]["content"]
    if not old_system.endswith(SOURCE_RULES):
        raise ValueError("Saved plan system does not end in the expected source rules.")
    old_system = old_system[:-len(SOURCE_RULES)]
    captured["context"]["optional_example"] = example_for("script-plan")

    def validate_plan(text):
        draft = FirstChapterPlan.model_validate_json(text)
        captured["validate"](draft)
        return {"draft": draft.model_dump(), "plan": project_plan(draft, reader.options).model_dump()}

    for label, system in (("old", old_system), ("new", captured["system"])):
        conditions.append(Condition("plan-instruction-" + label, "script-plan", system,
            copy.deepcopy(captured["context"]),
            {**response_format("script-plan", copy.deepcopy(captured["schema"])), "seed": saved_request["seed"]},
            validate_plan, "Old/new instruction-bundle comparison, not an isolated exchange-field effect. Same canonical legacy plot, first chapter, schema, cast, example and no executed history."))

    plan = project_plan(saved_plan, reader.options)
    cast = CastPlan.model_validate(state["cast_plan"]["plan"])
    used = {cid for scene in plan.scenes for cid in scene.character_ids}
    supporting = [row for row in cast.supporting_characters if row.id in used] + plan.new_characters
    context = reader.material(reader.setting(supporting, planned=True), None, 1, stage="writer")
    context["setting"]["introduced_character_ids"] = setting["introduced_character_ids"]
    context["setting"]["locations"] = [row.model_dump() for row in plan.locations]
    context["planned_connection"] = plan.continuation
    scene = plan.scenes[0]
    context = scene_material(1, scene, context, plan.scenes[1:])
    scene_request = read(source / f"requests/c001-{scene.id}-text-1.json")["request"]
    # A first-scene request has no earlier completed writer samples.
    estimate = scene_output_budget(plan.scene_sizes[scene.id], scene.character_ids,
        reader.options.scene_tokens, scene_request["max_tokens"], [])

    def validate_scene(text):
        parse_scene_text(text, scene.id, set(scene.character_ids))
        return {"metrics": script_metrics(text)}

    for enabled in (False, True):
        condition_context = {**copy.deepcopy(context), "optional_example": example_for("script-scene") if enabled else None}
        conditions.append(Condition("scene-example-" + ("on" if enabled else "off"), "script-scene",
            SCENE_INSTRUCTION, condition_context,
            {"grammar": narrative._source_grammar(scene), "max_tokens": estimate["max_tokens"], "seed": scene_request["seed"]},
            validate_scene, "Same saved first scene plan, future scope, no executed history, body/dialogue targets and new writer instruction. Only optional example differs.", estimate))
    return conditions


def run_condition(llm, output, condition):
    row = {"key": condition.key, "purpose": condition.purpose, "status": "pending", "dispatched": False,
           "input_sha256": digest(condition.material()), "comparison": condition.comparison,
           "output_estimate": condition.output_estimate, "attempts": 1}
    start, trace_start = time.monotonic(), len(llm.trace)
    try:
        llm.select_purpose(condition.purpose)
        llm._ensure_runtime()
        if (llm.server_context_size != 16384 or llm.profile["context_size"] != 16384
                or llm.profile["reasoning_level"] != "none" or llm.selected_route != "generation"):
            raise ValueError("This probe requires the actual 16k Gemma generation runtime, reasoning none.")
        messages, selection = fit_context(llm, condition.purpose, condition.system,
                                         **condition.context, extra=condition.extra)
        request, fingerprint = llm._chat_request(llm.requests + 1, messages, condition.extra)
        row.update(selection=selection, budget=selection["budget"], request_sha256=fingerprint,
                   request_ordinal=llm.requests + 1, runtime=llm.runtime_identity())
        write_json(output / "requests" / f"{condition.key}.json", {"request": request,
            "request_sha256": fingerprint, "selection": selection, "input_sha256": row["input_sha256"]})
        if condition.context.get("optional_example") and not selection.get("example", {}).get("included"):
            raise ValueError("Requested example was omitted for capacity; this comparison condition is unavailable. No generation dispatched.")
        row["dispatched"] = True
        reply = llm.chat(condition.purpose, messages, allow_truncated=True, **condition.extra)
        row["reply"] = reply
        text = reply.get("content") or ""
        (output / "responses" / f"{condition.key}.txt").write_text(text, encoding="utf-8")
        if reply.get("_finish_reason") != "stop" or reply.get("tool_calls") or not text.strip():
            raise ValueError("Incomplete, empty or tool-call response; no retry or continuation is allowed.")
        row["validated"] = condition.validate(text)
        row["status"] = "completed"
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if hasattr(exc, "selection"):
            row["selection"] = exc.selection
    finally:
        row["elapsed_seconds"] = time.monotonic() - start
        row["trace"] = llm.trace[trace_start:]
        metrics = [item for item in row["trace"] if item.get("type") == "llm_generation"]
        row["usage"] = metrics[-1].get("usage", {}) if metrics else {}
        write_json(output / "responses" / f"{condition.key}.json", row)
    return row


def save_report(output, rows, source_hashes, llm=None):
    unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in source_hashes.items())
    status = "failed" if any(row["status"] == "failed" for row in rows) else "complete" if len(rows) == 6 else "incomplete"
    report = {"status": status, "planned_conditions": 6, "results": rows,
              "source_files_unchanged": unchanged, "source_files_sha256": source_hashes,
              "generation_calls": llm.requests if llm else 0, "model_loads": llm._load_number if llm else 0,
              "limitation": "Single output per condition; no quality retries, continuation, semantic gates, allocation, staging or full story generation. Not a full-run success or general improvement-rate estimate."}
    write_json(output / "report.json", report)
    lines = ["# 作劇の小規模比較", "", "各条件1回、最大6要求。途中失敗で停止し、再抽選・続筆はしません。旧作品を変更しません。",
             "本文は同一の保存第1場面を使う診断であり、新プロットからの通し生成ではありません。", "",
             "|条件|状態|入力tokens|回答枠|実出力tokens|秒|", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        budget = row.get("budget", {})
        lines.append(f"|{row['key']}|{row['status']}|{budget.get('prompt_tokens')}|{budget.get('output_tokens')}|{row['usage'].get('completion_tokens')}|{row['elapsed_seconds']:.1f}|")
    for row in rows:
        lines += ["", "## " + row["key"], "", row["comparison"], "",
                  "作例採否: " + json.dumps(row.get("selection", {}).get("example"), ensure_ascii=False), ""]
        if row.get("error"):
            lines += ["停止理由: " + row["error"], ""]
        lines += ["```text", row.get("reply", {}).get("content") or "（本文なし）", "```"]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.baseline.resolve(), args.output_dir.resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Use a new, separate output directory; the baseline stays read-only.")
    paths = [args.input.resolve(), source / "experiment.json", source / "draft-state.json",
             source / "requests/story-chain-1.json", source / "requests/plan-001-1.json"]
    manifest, state = read(paths[1]), read(paths[2])
    snapshot, profile, seed = extract_input(read(paths[0]))
    config = pipeline.load_config()
    if (snapshot != manifest["approval_snapshot"] or seed != manifest["seed"]
            or profile != manifest["profile"] or config != manifest["generation_config"]):
        raise ValueError("Approval, seed, profile and generation configuration must match the saved baseline.")
    payload = {"approval_snapshot": snapshot, "seed": seed, "profile": profile,
               "profiles": manifest.get("profiles", {}), "story_workflow_version": 2,
               "workflow_policy": "script_continuation_v1", "script_options": manifest["script_options"]}
    conditions = prepare_conditions(source, manifest, state, payload, config)
    scene_id = FirstChapterPlan.model_validate_json(successful_attempt(state, "plan-001")["reply"]["content"]).scenes[0].id
    paths.append(source / f"requests/c001-{scene_id}-text-1.json")
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output.mkdir(parents=True)
    for name in ("requests", "responses"):
        (output / name).mkdir()
    write_json(output / "probe-input.json", {"baseline": str(source), "payload": payload,
        "generation_config": config, "generator_protocol": generator_protocol("causal", "script_continuation_v1"),
        "examples": example_catalog_identity(), "source_files_sha256": source_hashes,
        "conditions": [condition.material() for condition in conditions]})
    rows = []
    with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
          RoutedLLM(pipeline.ROOT, config, payload, output / "llm") as llm):
        for condition in conditions:
            row = run_condition(llm, output, condition)
            rows.append(row)
            save_report(output, rows, source_hashes, llm)
            print(json.dumps({"condition": row["key"], "status": row["status"],
                              "error": row.get("error")}, ensure_ascii=False), flush=True)
            if row["status"] != "completed":
                break
    report = save_report(output, rows, source_hashes, llm)
    print(json.dumps({"report": str(output / "report.md"), "status": report["status"],
                      "generation_calls": report["generation_calls"]}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "complete" and report["source_files_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
