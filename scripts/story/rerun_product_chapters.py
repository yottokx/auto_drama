"""Write a product work's later chapters again with other script options, through the product's own chapter jobs.

Reads the saved requests of the product's chapter jobs (``services/worker/cache/m2/jobs/<namespace>/<job>/
job-request.json``), turns on the given script options (e.g. chapter_scope_guard, skip_staging) and runs the
same m3_narrative job code in a new work directory for each chapter. The first chapter starts from the
product's own checkpoint, so the chapters before it are the ones the product wrote; each later chapter
continues from this run's previous chapter. The approval, the approved plan and the seed stay the product's.

Options are part of a job's identity, so the inherited checkpoint is bound again to the new identity (its
hashes recomputed, its history unchanged). The request counter carries on from the product's, so the first
calls (e.g. the note on the previous chapter) are asked with the product's seeds; chapter_scope_guard's
check comes after that note, so the scene plan's seed is one later. --shift moves the counter, every call's
seed with it. Run it while the product's worker is stopped; it loads the same model on the same GPU.

--chain-starts gives a long-form plan made before the fix of docs/experiments/chapter-starts-20261024.md
its chained starts: each chapter after the first starts from the previous chapter's last planned step, where
those plans start it from the skeleton's chain, before the steps added after it. The approved plan's content
and hash change with it, and the inherited checkpoint carries the same plan.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from packages.contracts.planning import validate_plan_content
from services.worker.generation import m3_pipeline
from services.worker.generation.causal_runtime import digest
from services.worker.generation.execution_settings import execution_revision
from services.worker.generation.script_cast import ScriptOptions
from services.worker.generation.script_expansion import chained_starts
from services.worker.generation.script_production import ProductionScriptRun


def chain_starts(payload: dict) -> None:
    """Give the request's approved plan, and the checkpoint it inherits, the chained starts."""
    seed = payload["approved_plan"]
    content = validate_plan_content({**seed["content"], "plot": chained_starts(seed["content"]["plot"])},
                                    payload["approval_snapshot"]).model_dump(mode="json")
    seed.update(content=content, sha256=digest(content))
    checkpoint = payload.get("script_checkpoint")
    if isinstance(checkpoint, dict):
        state = checkpoint["state"]
        state.update(plot=copy.deepcopy(content["plot"]), plot_sha256=digest(content["plot"]))
        state["accepted_plan"]["sha256"] = seed["sha256"]
        checkpoint["plan_sha256"] = seed["sha256"]


def rebound(checkpoint: dict, identity: str, shift: int) -> dict:
    """The checkpoint bound to another generation identity, its request counter moved by ``shift``."""
    result = copy.deepcopy(checkpoint)
    old = result.get("generation_identity")
    if old is not None:
        result["generation_identity"] = identity
    cast = result["state"].get("cast_plan")
    if cast is not None and old is not None and cast.get("input_sha256") == old:
        cast["input_sha256"] = identity
    result["state"]["request_ordinal"] += shift
    result["sha256"] = digest({key: value for key, value in result.items() if key != "sha256"})
    return result


def rebind_first_checkpoint(first: int, shift: int) -> None:
    """Bind the product's checkpoint to this run's identity, for the first chapter only; later
    chapters inherit this run's own checkpoints, already bound to it."""
    original = ProductionScriptRun._history

    def history(payload, manifest, number):
        checkpoint = payload.get("script_checkpoint")
        if number != first or not isinstance(checkpoint, dict):
            return original(payload, manifest, number)
        return original({**payload, "script_checkpoint": rebound(checkpoint, manifest["input_sha256"], shift)},
                        manifest, number)

    ProductionScriptRun._history = staticmethod(history)


def chapter_payload(request: dict, options: list[str], previous: dict | None) -> dict:
    if request.get("kind") != "m3_narrative":
        raise ValueError("Only the product's chapter (m3_narrative) jobs can be written again.")
    payload = copy.deepcopy(request["payload"])
    if payload.get("workflow_policy") != "script_continuation_v1":
        raise ValueError("The job is not a script_continuation_v1 chapter.")
    payload["script_options"] = {**payload.get("script_options", {}), **{name: True for name in options}}
    ScriptOptions.model_validate(payload["script_options"])
    if previous is not None:
        # This run's previous chapter replaces the one the product wrote.
        payload.update(script_checkpoint=previous["checkpoint"], previous_narrative=previous["narrative"],
                       previous_narrative_artifact_id="rerun-" + digest(previous["narrative"]))
    return payload


def summary(work: Path, number: int) -> dict:
    script = work / "script"
    plan = json.loads((script / "chapters" / f"chapter-{number:03d}.plan.json").read_text(encoding="utf-8"))
    text = (script / "chapters" / f"chapter-{number:03d}.md").read_text(encoding="utf-8")
    trace = json.loads((work / "generation-trace.json").read_text(encoding="utf-8"))
    done = [row for row in trace if row.get("type") == "route_progress" and row.get("chapter") == number]
    checked = [row for row in trace if row.get("type") == "opening_check" and row.get("chapter") == number]
    return {"chapter": number, "characters": len(text), "route_progress": done, "opening_check": checked,
            "continuation": plan.get("continuation", ""),
            "scenes": [{"id": scene["id"], "location_id": scene["location_id"],
                        "objective": scene["objectives"].split("\n")[0], "end_state": scene["end_state"],
                        "events": len(scene["required_events"])} for scene in plan["scenes"]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path, nargs="+",
                        help="The product's job directories of the chapters to write again, in chapter order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--option", action="append", default=[],
                        help="A script option to turn on (repeatable), e.g. chapter_scope_guard")
    parser.add_argument("--shift", type=int, default=0,
                        help="Move the first chapter's inherited request counter, so every call gets another seed")
    parser.add_argument("--chain-starts", action="store_true",
                        help="Start each chapter from the previous chapter's last planned step (the fix tried)")
    args = parser.parse_args()
    for name in args.option:
        field = ScriptOptions.model_fields.get(name)
        if field is None or field.annotation is not bool:
            parser.error(f"{name} is not an on/off script option.")
    requests = [json.loads((job / "job-request.json").read_text(encoding="utf-8")) for job in args.jobs]
    numbers = [row["payload"].get("chapter_number") for row in requests]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))) or numbers[0] < 2:
        parser.error(f"The jobs must be consecutive chapters from chapter 2 on, in order (got {numbers}).")
    if args.chain_starts:
        for request in requests:
            chain_starts(request["payload"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rebind_first_checkpoint(numbers[0], args.shift)
    previous = None
    for request, number in zip(requests, numbers):
        payload = chapter_payload(request, args.option, previous)
        work = args.output_dir / f"chapter-{number:03d}"
        job = {"id": f"rerun-{number:03d}", "kind": "m3_narrative", "payload": payload,
               "retry_generation": execution_revision(payload)}
        m3_pipeline.generate_job(job, work)
        envelope = json.loads((work / "result.json").read_text(encoding="utf-8"))
        previous = {"narrative": envelope["result"], "checkpoint": envelope["provenance"]["script_checkpoint"]}
        for name in (f"chapter-{number:03d}.md", f"chapter-{number:03d}.plan.json"):
            shutil.copyfile(work / "script" / "chapters" / name, args.output_dir / name)
        row = summary(work, number)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
