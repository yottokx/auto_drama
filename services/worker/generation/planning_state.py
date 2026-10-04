"""Bind approved semantic plans to the current worker's durable record identity."""
from __future__ import annotations

import copy

from packages.contracts.planning import planning_protocol, validate_plan_content

from .causal_runtime import digest
from .llm import write_json


def approved_plan_identity(payload: dict) -> tuple[dict, dict] | None:
    seed = payload.get("approved_plan")
    if seed is None:
        return None
    if not isinstance(seed, dict) or not isinstance(seed.get("approval_id"), str) or not seed["approval_id"]:
        raise ValueError("An approved common plan requires its approval identity.")
    content = validate_plan_content(seed.get("content"), payload["approval_snapshot"]).model_dump(mode="json")
    sha256 = digest(content)
    if seed.get("sha256") != sha256:
        raise ValueError("Approved common plan content hash was changed.")
    if seed.get("planning_protocol", planning_protocol()) != planning_protocol():
        raise ValueError("Approved common plan protocol changed.")
    return content, {"approval_id": seed["approval_id"], "sha256": sha256,
                     "planning_protocol": planning_protocol()}


def install_plan(state: dict, output, manifest: dict, content: dict, identity: dict) -> None:
    cast = copy.deepcopy(content["cast_plan"])
    record = {"plan": cast, "sha256": digest(cast), "input_sha256": manifest["input_sha256"],
              "protocol": manifest["generator_protocol"]}
    plot = copy.deepcopy(content["plot"])
    value = validate_plan_content(content, manifest["approval_snapshot"])
    state.update(cast_plan=record, plot=plot, plot_sha256=digest(plot),
                 outline=value.plot.as_outline().model_dump(mode="json"), accepted_plan=copy.deepcopy(identity))
    # The accepted detailed plot is the single canonical version. Generation
    # journals from STEP4 never become chapter execution/checkpoint material.
    for key in ("story_chain", "story_chain_sha256", "chapter_allocation"):
        state.pop(key, None)
    for key, filename in (("cast_plan", "cast-plan.json"), ("plot", "plot.json"),
                          ("outline", "outline.json")):
        write_json(output / filename, state[key])
