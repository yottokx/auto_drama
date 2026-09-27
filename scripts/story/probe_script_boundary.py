"""Generate chapter 2's plan and opening scene once against a frozen chapter 1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.contracts.m2 import CharacterResult
from packages.narrative import parse_scene_text
from services.worker.generation import narrative, pipeline
from services.worker.generation.causal_runtime import digest
from services.worker.generation.llm import write_json
from services.worker.generation.model_routing import (
    RoutedLLM,
    model_configuration_identity,
    resolve_purpose_profile,
)
from services.worker.generation.processes import gpu_lock
from services.worker.generation.script_budget import scene_output_budget
from services.worker.generation.script_cast import CastPlan, script_metrics
from services.worker.generation.script_chapter_plan import ChapterScriptPlan, project_plan
from services.worker.generation.script_context import fit_context
from services.worker.generation.script_continuation import (
    PROGRESSION,
    SCENE_INSTRUCTION,
    SCRIPT_SYSTEM,
    ScriptRun,
)
from services.worker.generation.script_examples import example_catalog_identity, example_for
from services.worker.generation.script_inputs import scene_material
from services.worker.generation.script_overlap import trim_chapter_overlap
from services.worker.generation.text_debug import extract_input
from services.worker.generation.workflow_version import generator_protocol

LIMITATION = (
    "保存済み第1章・旧プロット・キャストを固定した、第2章への接続だけの局所診断です。"
    "章計画と冒頭1場面を各1回生成します。新プロットからの通し生成や、一般的な改善率の評価ではありません。"
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def successful_attempt(state, key):
    return next(row for row in reversed(state["steps"][key]["attempts"])
                if row["status"] == "completed" and not row.get("validation_error"))


@dataclass
class Stage:
    key: str
    purpose: str
    system: str
    context: dict
    extra: dict
    validate: object
    expected_profile: dict
    output_estimate: dict | None = None
    previous_text: str | None = None

    def material(self):
        return {"key": self.key, "purpose": self.purpose, "system": self.system,
                "context": self.context, "extra": self.extra, "expected_profile": self.expected_profile,
                "output_estimate": self.output_estimate,
                "previous_source_sha256": digest(self.previous_text) if self.previous_text is not None else None}


class _PlanCaptured(Exception):
    pass


def prepare_reader(manifest, state, payload, config):
    """Keep only chapter 1's executed material; plot/cast remain planned material."""
    reader = object.__new__(ScriptRun)
    reader.payload, reader.manifest, reader.config = payload, manifest, config
    reader.state = copy.deepcopy(state)
    first = next(row for row in reader.state["chapters"] if row["number"] == 1)
    if digest(first["text"]) != first["sha256"]:
        raise ValueError("Saved chapter 1 source hash does not match its text.")
    reader.state["chapters"] = [first]
    reader.state["notes"] = [row for row in reader.state.get("notes", []) if row["number"] == 1]
    reader.state["locations"] = {row["id"]: row for row in first["narrative"]["locations"]}
    reader.state["steps"] = {key: row for key, row in reader.state["steps"].items()
                             if key.startswith("c001-") or key == "plan-001"}
    reader.state["chapter_plans"] = {key: row for key, row in reader.state.get("chapter_plans", {}).items()
                                     if key == "1"}
    for name in ("scene_commits", "scene_budgets", "chapter_boundary_corrections"):
        reader.state[name] = {key: row for key, row in reader.state.get(name, {}).items() if key.startswith("c001-")}
    reader.state.update(partial=None, sessions=[], fallbacks=[])
    supporting = [CharacterResult.model_validate(row) for row in first["narrative"]["supporting_characters"]]
    return reader, supporting


def prepare_plan(reader, supporting, seed, expected_profile):
    captured = {}

    def capture(key, purpose, number, prompt, context, model, validate=None, schema=None):
        captured.update(system=SCRIPT_SYSTEM + PROGRESSION + prompt, context=copy.deepcopy(context),
                        model=model, schema=schema or model.model_json_schema(), validate=validate)
        raise _PlanCaptured

    reader.structured = capture
    try:
        reader.chapter_plan(2, reader.material(reader.setting(supporting, planned=True), None, 2), supporting)
    except _PlanCaptured:
        pass
    if not captured:
        raise ValueError("Production chapter plan did not provide a request.")
    captured["context"]["optional_example"] = example_for("script-plan")

    def validate(text):
        draft = captured["model"].model_validate_json(text)
        if captured["validate"]:
            captured["validate"](draft)
        return {"draft": draft.model_dump(mode="json"),
                "plan": project_plan(draft, reader.options).model_dump(mode="json")}

    extra = {"seed": seed, "response_format": {"type": "json_schema", "json_schema": {
        "name": "script-plan", "strict": True, "schema": captured["schema"]}}}
    return Stage("chapter-002-plan", "script-plan", captured["system"], captured["context"], extra,
                 validate, copy.deepcopy(expected_profile))


def writer_samples(reader, profile):
    samples = []
    for key, step in reader.state["steps"].items():
        if not key.startswith("c001-") or not key.endswith("-text"):
            continue
        for row in reversed(step["attempts"]):
            if (row["status"] != "completed" or row.get("validation_error")
                    or row["purpose"] != "script-scene" or row["profile"]["model_id"] != profile["model_id"]
                    or row.get("reply", {}).get("_finish_reason") != "stop"):
                continue
            tokens = (row.get("usage") or {}).get("completion_tokens")
            body = script_metrics(row["reply"]["content"])["body_characters"]
            if type(tokens) is int and tokens > 0 and body > 0:
                samples.append({"step": key, "source_sha256": digest(row["reply"]["content"]),
                                "completion_tokens": tokens, "body_characters": body})
            break
    return samples


def prepare_scene(reader, supporting, plan_data, seed, expected_profile, output_limit):
    plan = ChapterScriptPlan.model_validate(plan_data)
    supporting = list(supporting)
    introduced = reader.setting(supporting, planned=True)["introduced_character_ids"]
    registered = {row.id for row in supporting}
    used = {cid for scene in plan.scenes for cid in scene.character_ids}
    cast = CastPlan.model_validate(reader.state["cast_plan"]["plan"])
    supporting.extend(row for row in cast.supporting_characters if row.id in used and row.id not in registered)
    supporting.extend(plan.new_characters)
    context = reader.material(reader.setting(supporting, planned=True), None, 2, stage="writer")
    context["setting"].update(introduced_character_ids=introduced,
                              locations=[row.model_dump(mode="json") for row in plan.locations])
    context["planned_connection"] = ""
    scene = plan.scenes[0]
    context = scene_material(2, scene, context, plan.scenes[1:])
    context["optional_example"] = example_for("script-scene")
    estimate = scene_output_budget(plan.scene_sizes[scene.id], scene.character_ids,
        reader.options.scene_tokens, min(expected_profile["max_tokens"], output_limit),
        writer_samples(reader, expected_profile))

    def validate(text):
        parse_scene_text(text, scene.id, set(scene.character_ids))
        return {"scene_id": scene.id, "metrics": script_metrics(text)}

    return Stage("chapter-002-opening", "script-scene", SCENE_INSTRUCTION, context,
        {"seed": seed, "grammar": narrative._source_grammar(scene), "max_tokens": estimate["max_tokens"]},
        validate, copy.deepcopy(expected_profile), estimate, reader.state["chapters"][0]["text"])


def run_stage(llm, output, stage):
    row = {"key": stage.key, "purpose": stage.purpose, "status": "pending", "dispatched": False,
           "attempts": 1, "input_sha256": digest(stage.material()), "output_estimate": stage.output_estimate}
    start, trace_start = time.monotonic(), len(llm.trace)
    try:
        llm.select_purpose(stage.purpose)
        if llm.profile != stage.expected_profile:
            raise ValueError("Resolved profile differs from the baseline's saved stage profile.")
        llm._ensure_runtime()
        if (llm.server_context_size != 16384 or llm.profile["context_size"] != 16384
                or llm.profile["reasoning_level"] != "none" or llm.selected_route != "generation"):
            raise ValueError("This probe requires the baseline Gemma runtime at 16k with reasoning none.")
        messages, selection = fit_context(llm, stage.purpose, stage.system, **stage.context, extra=stage.extra)
        request, fingerprint = llm._chat_request(llm.requests + 1, messages, stage.extra)
        row.update(selection=selection, budget=selection["budget"], request_sha256=fingerprint,
                   request_ordinal=llm.requests + 1, runtime=llm.runtime_identity())
        write_json(output / "requests" / f"{stage.key}.json", {"request": request, "selection": selection,
            "request_sha256": fingerprint, "input_sha256": row["input_sha256"], "material": stage.material()})
        row["dispatched"] = True
        reply = llm.chat(stage.purpose, messages, allow_truncated=True, **stage.extra)
        row["reply"] = reply
        text = reply.get("content") or ""
        (output / "responses" / f"{stage.key}.raw.txt").write_text(text, encoding="utf-8")
        if reply.get("_finish_reason") != "stop" or reply.get("tool_calls") or not text.strip():
            raise ValueError("Incomplete, empty or tool-call response; no retry or continuation is allowed.")
        if stage.previous_text is not None:
            effective, correction = trim_chapter_overlap(stage.previous_text, text)
            row.update(overlap=correction, raw_metrics=script_metrics(text), effective_metrics=script_metrics(effective))
            write_json(output / "responses" / f"{stage.key}.overlap.json", correction)
            (output / "responses" / f"{stage.key}.effective.txt").write_text(effective, encoding="utf-8")
            if not effective.strip():
                raise ValueError("No new script after removing repeated chapter boundary; original source retained.")
            text = effective
        row["validated"] = stage.validate(text)
        row["status"] = "completed"
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if hasattr(exc, "selection"):
            row["selection"] = exc.selection
        if hasattr(exc, "budget"):
            row["budget"] = exc.budget
    finally:
        row["elapsed_seconds"] = time.monotonic() - start
        row["trace"] = llm.trace[trace_start:]
        metrics = [item for item in row["trace"] if item.get("type") == "llm_generation"]
        row["usage"] = metrics[-1].get("usage", {}) if metrics else {}
        write_json(output / "responses" / f"{stage.key}.json", row)
    return row


def save_report(output, rows, hashes, llm=None, error=None):
    unchanged = all(Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
                    for path, sha in hashes.items())
    failed = error or any(row["status"] == "failed" for row in rows) or not unchanged
    report = {"status": "failed" if failed else "complete" if len(rows) == 2 else "incomplete",
              "planned_stages": 2, "results": rows, "error": error,
              "source_files_unchanged": unchanged, "source_files_sha256": hashes,
              "generation_calls": llm.requests if llm else 0, "model_loads": llm._load_number if llm else 0,
              "limitation": LIMITATION}
    write_json(output / "report.json", report)
    lines = ["# 第1章から第2章への接続診断", "", LIMITATION, "",
             "途中失敗で停止。品質による再生成、続筆、演出・音声・画像・エクスポートは行いません。", "",
             "状態: " + report["status"], "原資料変更なし: " + str(unchanged), "",
             "|工程|状態|入力tokens|回答枠|実出力tokens|秒|", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        budget = row.get("budget", {})
        lines.append(f"|{row['key']}|{row['status']}|{budget.get('prompt_tokens')}|{budget.get('output_tokens')}|{row['usage'].get('completion_tokens')}|{row['elapsed_seconds']:.1f}|")
    for row in rows:
        lines += ["", "## " + row["key"], "",
                  "作例採否: " + json.dumps(row.get("selection", {}).get("example"), ensure_ascii=False)]
        if row.get("error"):
            lines += ["", "停止理由: " + row["error"]]
        if row.get("overlap") is not None:
            lines += ["", "生本文の分量: " + json.dumps(row["raw_metrics"], ensure_ascii=False),
                      "採用本文の分量: " + json.dumps(row["effective_metrics"], ensure_ascii=False),
                      "再演除去: " + json.dumps(row["overlap"], ensure_ascii=False), "",
                      "```text", (output / "responses" / f"{row['key']}.effective.txt").read_text(encoding="utf-8"), "```"]
        else:
            lines += ["", "```text", row.get("reply", {}).get("content") or "（本文なし）", "```"]
    if error:
        lines += ["", "停止理由: " + error]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def execute(llm, output, reader, supporting, plan_stage, writer_seed, writer_profile, source_hashes):
    """At most two calls; a failed plan can never dispatch a writer."""
    rows = []
    row = run_stage(llm, output, plan_stage)
    rows.append(row)
    save_report(output, rows, source_hashes, llm)
    print(json.dumps({"stage": row["key"], "status": row["status"], "error": row.get("error")}, ensure_ascii=False), flush=True)
    if row["status"] != "completed":
        return rows
    write_json(output / "chapter-002.plan.json", row["validated"]["plan"])
    scene = prepare_scene(reader, supporting, row["validated"]["plan"], writer_seed, writer_profile,
                          llm._output_limit(writer_profile))
    write_json(output / "writer-input.json", scene.material())
    row = run_stage(llm, output, scene)
    rows.append(row)
    save_report(output, rows, source_hashes, llm)
    print(json.dumps({"stage": row["key"], "status": row["status"], "error": row.get("error")}, ensure_ascii=False), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.baseline.resolve(), args.output_dir.resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Use a new output directory outside the read-only baseline.")
    paths = [args.input.resolve(), source / "experiment.json", source / "draft-state.json"]
    manifest, state = read(paths[1]), read(paths[2])
    snapshot, profile, seed = extract_input(read(paths[0]))
    config = pipeline.load_config()
    if (snapshot != manifest["approval_snapshot"] or seed != manifest["seed"]
            or profile != manifest["profile"] or config != manifest["generation_config"]
            or model_configuration_identity(pipeline.ROOT, config) != manifest["model_configuration"]):
        raise ValueError("Approval, seed, profile and model configuration must match the saved baseline.")
    payload = {"approval_snapshot": snapshot, "seed": seed, "profile": profile,
               "profiles": manifest.get("profiles", {}), "story_workflow_version": 2,
               "workflow_policy": "script_continuation_v1", "script_options": manifest["script_options"]}
    first_scene_id = state["chapter_plans"]["2"]["plan"]["scenes"][0]["id"]
    plan_attempt = successful_attempt(state, "plan-002")
    scene_attempt = successful_attempt(state, f"c002-{first_scene_id}-text")
    paths += [source / "requests" / f"plan-002-{plan_attempt['attempt']}.json",
              source / "requests" / f"c002-{first_scene_id}-text-{scene_attempt['attempt']}.json"]
    plan_seed, writer_seed = (read(path)["request"]["seed"] for path in paths[-2:])
    for purpose, attempt in (("script-plan", plan_attempt), ("script-scene", scene_attempt)):
        if resolve_purpose_profile(config, payload, purpose)[1] != attempt["profile"]:
            raise ValueError("Saved stage profile no longer matches the current baseline configuration.")
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    reader, supporting = prepare_reader(manifest, state, payload, config)
    plan_stage = prepare_plan(reader, supporting, plan_seed, plan_attempt["profile"])
    output.mkdir(parents=True)
    for name in ("requests", "responses"):
        (output / name).mkdir()
    write_json(output / "probe-input.json", {"baseline": str(source), "payload": payload,
        "generation_config": config, "generator_protocol": generator_protocol("causal", "script_continuation_v1"),
        "examples": example_catalog_identity(), "source_files_sha256": source_hashes,
        "frozen_chapter_numbers": [1], "plan": plan_stage.material(), "writer_seed": writer_seed,
        "writer_profile": scene_attempt["profile"], "limitation": LIMITATION})
    rows, llm, error = [], None, None
    try:
        with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
              RoutedLLM(pipeline.ROOT, config, payload, output / "llm") as llm):
            rows = execute(llm, output, reader, supporting, plan_stage, writer_seed,
                           scene_attempt["profile"], source_hashes)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        if (output / "report.json").exists():
            rows = read(output / "report.json")["results"]
    if llm is not None:
        write_json(output / "provenance.json", llm.provenance())
    report = save_report(output, rows, source_hashes, llm, error)
    print(json.dumps({"report": str(output / "report.md"), "status": report["status"],
                      "generation_calls": report["generation_calls"]}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
