"""Measure revised requests against saved scripts, using only the model tokenizer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.contracts.m3 import NarrativeResult
from services.worker.generation import narrative, pipeline
from services.worker.generation.causal_runtime import digest
from services.worker.generation.llm import ContextBudgetError, write_json
from services.worker.generation.model_routing import RoutedLLM
from services.worker.generation.processes import gpu_lock
from services.worker.generation.script_budget import scene_output_budget
from services.worker.generation.script_cast import CastPlan, ScriptOptions, script_metrics
from services.worker.generation.script_chapter_plan import (
    ChapterScriptPlan,
    FirstChapterPlan,
    project_plan,
)
from services.worker.generation.script_context import fit_context
from services.worker.generation.script_continuation import (
    PROGRESSION,
    SCENE_INSTRUCTION,
    SCRIPT_SYSTEM,
    ScriptRun,
)
from services.worker.generation.script_examples import example_catalog_identity, example_for
from services.worker.generation.script_inputs import scene_material
from services.worker.generation.text_debug import extract_input
from services.worker.generation.workflow_version import generator_protocol


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


class TokenizerOnlyLLM(RoutedLLM):
    def chat(self, *args, **kwargs):
        raise RuntimeError("Generation is forbidden in a tokenizer-only probe.")

    def request(self, path, value=None, timeout=None):
        if path not in {"/apply-template", "/tokenize", "/health", "/props"}:
            raise RuntimeError(f"Non-tokenizer request forbidden: {path}")
        return super().request(path, value, timeout)


def saved_attempt(state, key):
    """Include a failed preflight, but never invent a later unsaved scene."""
    step = state["steps"].get(key, {})
    completed = [row for row in step.get("attempts", [])
                 if row["status"] == "completed" and not row.get("validation_error")]
    if completed:
        return completed[-1]
    return next(iter(reversed(step.get("preflight_failures", []))), None) or next(
        iter(reversed(step.get("attempts", []))), None)


def saved_chapter_plan(state, number, options):
    """Read the accepted baseline plan without applying today's opening rules."""
    record = state.get("chapter_plans", {}).get(str(number))
    if record is not None:
        if record["number"] != number or digest(record["plan"]) != record["sha256"]:
            raise ValueError(f"Saved chapter plan hash or number mismatch: {number}")
        return ChapterScriptPlan.model_validate(copy.deepcopy(record["plan"]))
    attempt = saved_attempt(state, f"plan-{number:03d}")
    if (attempt is None or attempt["status"] != "completed"
            or attempt.get("validation_error")):
        return None
    legacy = json.loads(attempt["reply"]["content"])
    continuation = legacy.pop("continuation", "")
    # Old replies predate the opening-action meaning and its 240-character cap.
    # FirstChapterPlan projects the same scenes/volume without injecting that
    # historical summary into the first event; retain it only as saved metadata.
    projected = project_plan(FirstChapterPlan.model_validate(legacy), options)
    return ChapterScriptPlan.model_validate({**projected.model_dump(), "continuation": continuation})


def prior_history(reader, saved_chapters, notes, number):
    """Expose only chapters that existed when this baseline stage was run."""
    reader.state["chapters"] = [copy.deepcopy(saved_chapters[n]) for n in sorted(saved_chapters) if n < number]
    reader.state["notes"] = [copy.deepcopy(row) for row in notes if row["number"] < number]
    reader.state["locations"] = {location["id"]: location for row in reader.state["chapters"]
                                  for location in row["narrative"]["locations"]}


class _PlanMeasured(Exception):
    def __init__(self, result):
        self.result = result


def measure_chapter_plan(reader, number, supporting, llm, output, source_state):
    """Capture the current request, then stop before validating any old reply."""
    def capture(key, purpose, chapter, prompt, context, model, validate=None, schema=None):
        extra = {"response_format": {"type": "json_schema", "json_schema": {
            "name": purpose, "strict": True, "schema": schema or model.model_json_schema()}}}
        result = measure(llm, output, key, purpose, SCRIPT_SYSTEM + PROGRESSION + prompt,
                         context, extra, saved_attempt(source_state, key) or {})
        raise _PlanMeasured(result)

    missing = object()
    original = reader.__dict__.get("structured", missing)
    reader.structured = capture
    try:
        reader.chapter_plan(number, reader.material(reader.setting(supporting, planned=True), None, number), supporting)
    except _PlanMeasured as measured:
        return measured.result
    finally:
        if original is missing:
            del reader.structured
        else:
            reader.structured = original
    raise RuntimeError("Chapter planning did not reach its request capture.")


def probe_output_budget(state, scope, size, character_ids, options, llm, samples):
    """Reuse a frozen reservation; only older baselines need a fresh estimate."""
    output_limit = min(llm.profile["max_tokens"], llm._output_limit(llm.profile))
    saved = state.get("scene_budgets", {}).get(scope)
    if saved is not None:
        budget = saved["budget"]
        if digest(budget) != saved["sha256"] or budget["scene_size"] != size.model_dump():
            raise ValueError(f"Saved output budget hash or scene size mismatch: {scope}")
        if (type(budget["max_tokens"]) is not int
                or not 0 < budget["max_tokens"] <= output_limit):
            raise ValueError(f"Saved output reservation exceeds the current model allowance: {scope}")
        return copy.deepcopy(budget), "saved_scene_budgets"
    return scene_output_budget(size, character_ids, options.scene_tokens,
                               output_limit, samples), "estimated_from_prior_saved_outputs"


def history_summary(selection):
    notes = {row["number"] for row in selection.get("notes", []) if row["included"]}
    values = []
    for row in selection.get("chapters", []):
        included = sum(end - start for start, end in row["included_ranges"])
        values.append(f"ch{row['number']}:{included}/{row['characters']} chars"
                      + ("+memo" if row["number"] in notes else ""))
    return "; ".join(values) or "none"


def measure(llm, output, key, purpose, system, context, extra, baseline, output_budget=None,
            output_budget_source="profile_default"):
    llm.select_purpose(purpose)
    llm._ensure_runtime()
    if (llm.profile["context_size"] != 16384 or llm.server_context_size != 16384
            or llm.profile["reasoning_level"] != "none"):
        raise ValueError("This comparison requires the actual 16k, reasoning-none runtime.")
    initial_trace = len(llm.trace)
    context = {**context, "optional_example": example_for(purpose)}
    try:
        messages, selection = fit_context(llm, purpose, system, **context, extra=extra)
        request, fingerprint = llm._chat_request(llm.requests + 1, messages, extra)
        artifact = {"request": request, "request_sha256": fingerprint, "selection": selection}
        status, error = "fits", None
    except ContextBudgetError as exc:
        selection = exc.selection
        artifact = {"system": system, "context": context, "extra": extra, "selection": selection}
        status, error = "overflow", str(exc)
    write_json(output / "requests" / f"{key}.json", artifact)
    row = {"step": key, "purpose": purpose, "status": status, "error": error,
           "budget": selection["budget"], "selection": selection,
           "history": history_summary(selection), "output_budget": output_budget,
           "output_budget_source": output_budget_source,
           "output_estimate": output_budget
               if output_budget_source == "estimated_from_prior_saved_outputs" else None,
           "baseline": {"budget": baseline.get("selection", {}).get("budget"),
                        "selection": baseline.get("selection"),
                        "history": history_summary(baseline.get("selection", {})),
                        "actual_completion_tokens": baseline.get("usage", {}).get("completion_tokens")},
           "budget_attempts": [item for item in llm.trace[initial_trace:]
                               if item.get("type") == "context_budget"]}
    print(json.dumps({"step": key, "status": status, "budget": row["budget"],
                      "history": row["history"], "example": selection["example"],
                      "output_budget_source": output_budget_source}, ensure_ascii=False), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output, source = args.output_dir.resolve(), args.baseline.resolve()
    if output == source or source in output.parents or output.exists():
        raise ValueError("Use a new, separate output directory; saved experiments stay read-only.")
    state_path, manifest_path = source / "draft-state.json", source / "experiment.json"
    originals = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (state_path, manifest_path, args.input.resolve())}
    source_state, manifest = read(state_path), read(manifest_path)
    snapshot, profile, seed = extract_input(read(args.input))
    if snapshot != manifest["approval_snapshot"] or seed != manifest["seed"]:
        raise ValueError("Probe input must match the saved approved setting and seed.")
    config = pipeline.load_config()
    if config != manifest["generation_config"]:
        raise ValueError("Baseline and current model configurations differ; use a separate comparison.")
    options = ScriptOptions.model_validate(manifest["script_options"])
    payload = {"approval_snapshot": snapshot, "seed": seed, "profile": profile,
               "profiles": manifest.get("profiles", {}), "story_workflow_version": 2,
               "workflow_policy": "script_continuation_v1", "script_options": options.model_dump()}
    plans = {}
    for number in range(1, manifest["approved_chapter_count"] + 1):
        plan = saved_chapter_plan(source_state, number, options)
        if plan is None:
            break
        plans[number] = plan
    if not plans:
        raise ValueError("Baseline needs at least one completed chapter plan.")
    saved_chapters = {row["number"]: row for row in source_state["chapters"]}
    for row in saved_chapters.values():
        if digest(row["text"]) != row["sha256"]:
            raise ValueError("Saved source hash mismatch.")
    notes = copy.deepcopy(source_state["notes"])
    for row in notes:
        row.setdefault("available", True)
        row.setdefault("source_sha256", saved_chapters[row["number"]]["sha256"])
    baseline_summary = {"directory": str(source),
        "generator_protocol": manifest.get("generator_protocol"),
        "last_session_status": source_state.get("sessions", [{}])[-1].get("status")
            if source_state.get("sessions") else None,
        "saved_chapters": sorted(saved_chapters), "planned_chapters": sorted(plans),
        "saved_scene_count": sum(len(row["narrative"]["scenes"]) for row in saved_chapters.values())}
    output.mkdir(parents=True)
    (output / "requests").mkdir()
    write_json(output / "probe-input.json", {"input_sources_sha256": originals,
        "baseline": str(source), "generator_protocol": generator_protocol("causal", "script_continuation_v1"),
        "baseline_summary": baseline_summary, "example_catalog": example_catalog_identity(),
        "payload": payload, "generation_config": config,
        "reconstructed_plans": {str(n): value.model_dump() for n, value in plans.items()},
        "notes": notes, "note_provenance": "Saved baseline handoff notes, linked to their saved source hashes."})
    results, samples, skipped = [], [], []
    # A lightweight reader uses the production material and setting functions;
    # it never initializes, resumes or persists the baseline experiment.
    reader = object.__new__(ScriptRun)
    reader.payload, reader.manifest, reader.config = payload, manifest, config
    reader.state = copy.deepcopy(source_state)
    reader.state["notes"] = notes
    with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
          TokenizerOnlyLLM(pipeline.ROOT, config, payload, output / "llm") as llm):
        for number, plan in plans.items():
            previous = saved_chapters.get(number - 1)
            supporting = (list(NarrativeResult.model_validate(previous["narrative"]).supporting_characters)
                          if previous else [])
            prior_history(reader, saved_chapters, notes, number)
            if number > 1:
                results.append(measure_chapter_plan(reader, number, supporting, llm, output, source_state))
            introduced = reader.setting(supporting, planned=True)["introduced_character_ids"]
            used = {cid for scene in plan.scenes for cid in scene.character_ids}
            registered = {row.id for row in supporting}
            saved_cast = CastPlan.model_validate(source_state["cast_plan"]["plan"])
            supporting.extend(row for row in saved_cast.supporting_characters
                              if row.id in used and row.id not in registered)
            supporting.extend(plan.new_characters)
            context = reader.material(reader.setting(supporting, planned=True), None, number, stage="writer")
            context["setting"]["introduced_character_ids"] = introduced
            context["setting"]["locations"] = [row.model_dump() for row in plan.locations]
            context["planned_connection"] = ""
            saved_scenes = (NarrativeResult.model_validate(saved_chapters[number]["narrative"]).scenes
                            if number in saved_chapters else [])
            for index, scene in enumerate(plan.scenes):
                key = f"c{number:03d}-{scene.id}-text"
                baseline = saved_attempt(source_state, key)
                if baseline is None or len(saved_scenes) < index:
                    skipped.append({"step": key, "reason": "no_saved_attempt" if baseline is None
                                    else "preceding_scene_source_unavailable"})
                    continue
                material = scene_material(number, scene, context, plan.scenes[index + 1:])
                if index:
                    material["chapters"] = [*context["chapters"], {"number": number,
                        "text": reader.script_text(saved_scenes[:index]),
                        "scene_starts": reader.scene_offsets(saved_scenes[:index])}]
                llm.select_purpose("script-scene")
                reservation, reservation_source = probe_output_budget(source_state,
                    f"c{number:03d}-{scene.id}", plan.scene_sizes[scene.id], scene.character_ids,
                    options, llm, samples)
                results.append(measure(llm, output, key, "script-scene", SCENE_INSTRUCTION, material,
                    {"grammar": narrative._source_grammar(scene), "max_tokens": reservation["max_tokens"]},
                    baseline, reservation, reservation_source))
                tokens = baseline.get("usage", {}).get("completion_tokens")
                if (baseline["status"] == "completed" and not baseline.get("validation_error")
                        and baseline.get("reply", {}).get("_finish_reason") == "stop"
                        and baseline.get("profile", {}).get("model_id") == llm.profile["model_id"]
                        and type(tokens) is int and tokens > 0):
                    raw = baseline["reply"]["content"]
                    body_characters = script_metrics(raw)["body_characters"]
                    if body_characters > 0:
                        samples.append({"step": key, "source_sha256": digest(raw),
                            "completion_tokens": tokens, "body_characters": body_characters})
                write_json(output / "measurements.json", results)
        runtime = llm.runtime_identity()
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path, sha in originals.items()):
        raise ValueError("A source changed while the probe ran.")
    counts = [row["completion_tokens"] for row in samples]
    report = {"status": "measured", "generation_calls": 0, "model_loads": llm._load_number,
        "runtime": runtime, "source_files_unchanged": True, "measurements": results,
        "baseline": baseline_summary, "skipped_scenes": skipped,
        "actual_baseline_outputs": {"count": len(counts), "minimum_tokens": min(counts, default=None),
                                    "maximum_tokens": max(counts, default=None), "total_tokens": sum(counts)},
        "sample_policy": "Frozen baseline scene reservations are reused when available. Otherwise estimates use chronological saved baseline outputs only, same model; no future-scene samples.",
        "limitation": "Tokenizer-only capacity replay of the baseline plot, chapter plans and saved scripts with current prompts and optional examples. Baseline story content is unchanged. No new prose or plan was generated. This is not a quality or full-run success evaluation."}
    write_json(output / "report.json", report)
    lines = ["# 保存台本によるコンテキスト容量再計測", "",
        "実生成なし。Gemma reasoning none / 実16,384 tokens。旧プロットと実台本を現行入力へ再構成した容量診断であり、物語品質や通し生成の成功を証明しません。",
        f"基準実験: {source.name}。保存済み{len(saved_chapters)}章・{baseline_summary['saved_scene_count']}場面、最終状態: {baseline_summary['last_session_status']}。",
        "保存済みの筋・計画・本文を使い、原文・履歴メモの出典と採用範囲をreport.jsonに記録しています。", "",
        "|工程|旧入力|旧出力枠|新入力|新出力枠|新合計（余白込み）|作例|出力枠の出典|判定|",
        "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for row in results:
        old, new = row["baseline"]["budget"] or {}, row["budget"] or {}
        example = row["selection"].get("example")
        example_state = "採用" if example and example["included"] else "容量で省略" if example else "対象外"
        lines.append(f"|{row['step']}|{old.get('prompt_tokens')}|{old.get('output_tokens')}|{new.get('prompt_tokens')}|{new.get('output_tokens')}|{new.get('required_tokens')}|{example_state}|{row['output_budget_source']}|{row['status']}|")
    lines.extend(["", "## 履歴の保持", ""])
    for row in results:
        lines.extend([f"- {row['step']}: {row['history']}", f"  - 旧入力: {row['baseline']['history']}"])
    lines.extend(["", f"基準実験の採用した{len(counts)}場面の実出力: 最小{min(counts, default=None)} / 最大{max(counts, default=None)} / 合計{sum(counts)} tokens。",
        "出力枠は保存済みscene_budgetsの固定値を優先。記録がない旧実験のみ目標字数・話者ID・安全余裕と、時系列で既出の保存実測から推定しています。後続場面の実測は先に使いません。",
        f"モデル起動{llm._load_number}回、生成0回。旧実験・初期入力は変更なし。"])
    if skipped:
        lines.extend(["", "## 再構成しなかった場面", ""])
        lines.extend(f"- {row['step']}: {row['reason']}" for row in skipped)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(output / "report.md"), "fits": sum(r["status"] == "fits" for r in results),
                      "measured": len(results), "generation_calls": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
