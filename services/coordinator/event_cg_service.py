"""Immutable CG budgets, generation dependencies, adoption and publication."""
from __future__ import annotations

import hashlib
import io
import json

from PIL import Image

from packages.contracts import Script
from packages.contracts.event_cg import (
    BUDGET_ALLOCATION_VERSION,
    EventCgBudget,
    EventCgPlan,
    EventCgPolicy,
    EventCgResult,
    image_input_sha256,
    image_spans,
    validate_budget,
    validate_plan,
)

from .m2_bundle import validate_png
from .service import encode_json, required


def policy(connection, production):
    row = connection.execute("SELECT policy FROM event_cg_production WHERE production_id=?",
                             (production["storyline_id"],)).fetchone()
    return EventCgPolicy.model_validate_json(row["policy"]) if row else EventCgPolicy()


def image_target(cg_id, variant_id=None):
    return "cg-" + hashlib.sha256(json.dumps([cg_id, variant_id]).encode()).hexdigest()[:24]


def _requirement(connection, production_id, kind, target):
    row = connection.execute("SELECT * FROM m3_requirement WHERE production_id=? AND kind=? AND target_id=?",
                             (production_id, kind, target)).fetchone()
    return dict(row) if row else None


def _read(m3, connection, row, kind, production_id):
    record = required(connection, "artifact", row["artifact_id"])
    owner = required(connection, "m3_production", production_id)
    if (record["kind"] != kind or record["project_id"] != owner["project_id"]
            or json.loads(record["provenance"]).get("production_id") != production_id):
        raise ValueError("CG artifact has an invalid owner or kind.")
    return json.loads(m3.store.read(record))


def ensure_budget(m3, connection, production):
    """Allocate from the approved whole-story plot before any chapter writing."""
    settings = policy(connection, production)
    if settings.max_cgs == 0:
        return None
    root = m3._root(connection, production)
    budget_row = _requirement(connection, root["id"], "m3_event_cg_budget", "event-cg-budget")
    if budget_row is None:
        approved = m3._approved_plan(connection, root)
        if not approved and not root["narrative_artifact_id"]:
            # Legacy productions have no separately approved plot. They must
            # finish the first narrative to obtain their original outline.
            return None
        plot = (approved["content"]["plot"] if approved else
                m3._load_narrative(connection, root).outline.model_dump(mode="json"))
        source = approved["sha256"] if approved else hashlib.sha256(encode_json(plot)).hexdigest()
        m3._requirement(connection, root, "m3_event_cg_budget", "event-cg-budget", {
            "budget_allocation_version": BUDGET_ALLOCATION_VERSION,
            "policy": settings.model_dump(mode="json"),
            "context": {"overall_plot": plot, "source_sha256": source,
                        "approval_snapshot": m3._snapshot(connection, root)},
        })
        budget_row = _requirement(connection, root["id"], "m3_event_cg_budget", "event-cg-budget")
    if not budget_row["artifact_id"] and not budget_row["job_id"] and root["control_state"] == "running":
        descriptor = json.loads(budget_row["descriptor"])
        job_id = m3._enqueue(connection, root, "m3_event_cg_budget",
                             {**descriptor, "requirement_id": budget_row["id"]})
        connection.execute("UPDATE m3_requirement SET job_id=? WHERE id=?", (job_id, budget_row["id"]))
        budget_row["job_id"] = job_id
    return budget_row


def ensure_plan(m3, connection, production):
    settings = policy(connection, production)
    if settings.max_cgs == 0:
        return
    root = m3._root(connection, production)
    budget_row = ensure_budget(m3, connection, production)
    if not budget_row or not budget_row["artifact_id"]:
        return
    if _requirement(connection, production["id"], "m3_event_cg_plan", "event-cg-plan"):
        return
    budget = EventCgBudget.model_validate(_read(m3, connection, budget_row, "event_cg_budget", root["id"]))
    limit = next(row.limit for row in budget.chapters if row.chapter_number == production["chapter_number"])
    if limit == 0:
        return
    from .music_service import context

    narrative = m3._load_narrative(connection, production)
    record = required(connection, "artifact", production["narrative_artifact_id"])
    source = context(m3, connection, production, narrative)
    source.update(source_sha256=record["sha256"], narrative_artifact_id=record["id"])
    cast = {row["id"]: row["result"] for row in source["approval_snapshot"]["characters"]}
    cast.update({row.id: row.model_dump(mode="json") for row in narrative.supporting_characters})
    used = {cid for scene in narrative.scenes for cid in scene.plan.character_ids}
    references = [{"character_id": cid, "name": cast[cid]["name"], "outfit_id": "default"}
                  for cid in sorted(used)]
    m3._requirement(connection, production, "m3_event_cg_plan", "event-cg-plan", {
        "policy": settings.model_dump(mode="json"), "chapter_budget": limit,
        "context": source, "references": references,
    })


def _omit(m3, connection, production, requirement, descriptor, reason):
    result = EventCgResult(cg_id=descriptor["cg_id"], variant_id=descriptor.get("variant_id"),
                           input_sha256=image_input_sha256(descriptor), status="omitted", reason=reason)
    record = m3._artifact(connection, production, "media-" + requirement["id"],
                          "event_cg_omission", "event-cg-omission.json", encode_json(result.model_dump(mode="json")),
                          provenance={"production_id": production["id"], **result.model_dump(mode="json")})
    connection.execute("UPDATE m3_requirement SET artifact_id=? WHERE id=?", (record["id"], requirement["id"]))
    requirement["artifact_id"] = record["id"]


def prepare_image(m3, connection, production, requirement, descriptor):
    """Bind actual artifacts only after all required portraits/base CG are adopted."""
    references = []
    if descriptor.get("variant_id"):
        base = _requirement(connection, production["id"], "m3_event_cg", image_target(descriptor["cg_id"]))
        if not base or not base["artifact_id"]:
            return None
        record = required(connection, "artifact", base["artifact_id"])
        if record["kind"] == "event_cg_omission":
            _omit(m3, connection, production, requirement, descriptor, "基本CGを生成できなかったため差分を省略しました。")
            return None
        if record["kind"] != "event_cg" or record["project_id"] != production["project_id"]:
            raise ValueError("CG variation requires this production's base image.")
        references.append({"role": "base_cg", "artifact_id": record["id"], "sha256": record["sha256"]})
    for character in descriptor["characters"]:
        row = _requirement(connection, production["id"], "m3_image", character["character_id"])
        if not row or not row["artifact_id"]:
            return None
        record = required(connection, "artifact", row["artifact_id"])
        if record["kind"] == "portrait_omission":
            _omit(m3, connection, production, requirement, descriptor,
                  f"{character['name']}の参照立ち絵を用意できなかったためCGを省略しました。")
            return None
        if record["kind"] != "character" or record["project_id"] != production["project_id"]:
            raise ValueError("CG reference must be an adopted portrait in this project.")
        references.append({"role": "character", **character,
                           "artifact_id": record["id"], "sha256": record["sha256"]})
    for index, reference in enumerate(references, 1):
        reference["reference_index"] = index
    return {**descriptor, "references": references}


def validate_result(kind, result, payload, files):
    if kind == "m3_event_cg_budget":
        return validate_budget(result, payload)
    if kind == "m3_event_cg_plan":
        return validate_plan(result, payload)
    value = EventCgResult.model_validate(result)
    if (value.cg_id != payload["cg_id"] or value.variant_id != payload.get("variant_id")
            or value.input_sha256 != payload["input_sha256"]
            or value.input_sha256 != image_input_sha256(payload)):
        raise ValueError("CG result differs from its frozen request.")
    if value.status == "complete":
        profile = payload["cg_profile"]
        for name in ("image.png", "original.png"):
            validate_png(files[name], require_transparency=False)
        with Image.open(io.BytesIO(files["image.png"])) as display, Image.open(io.BytesIO(files["original.png"])) as original:
            if display.size != (profile["width"], profile["height"]) or original.size != display.size:
                raise ValueError("CG dimensions differ from the frozen profile.")
            if display.convert("RGBA").getchannel("A").getextrema() != (255, 255):
                raise ValueError("Published CG must be opaque.")
            histogram = original.convert("RGBA").getchannel("A").histogram()
            if sum(histogram[:245]) / (original.width * original.height) > 0.001:
                raise ValueError("CG original contains an unsupported transparent region.")
            if original.convert("RGB").tobytes() != display.convert("RGB").tobytes():
                raise ValueError("Display CG must preserve the original RGB pixels.")
    return value


def adopt(m3, connection, production, requirement, job, attempt, payload, result, files, provenance):
    if job["kind"] != "m3_event_cg_budget":
        source = required(connection, "artifact", production["narrative_artifact_id"])
        if (payload["context"]["source_sha256"] != source["sha256"]
                or payload["context"]["narrative_artifact_id"] != source["id"]):
            raise ValueError("CG result refers to another narrative revision.")
    metadata = {**provenance, "production_id": production["id"],
                "storyline_id": production["storyline_id"], **result.model_dump(mode="json")}
    if job["kind"] in {"m3_event_cg_budget", "m3_event_cg_plan"}:
        kind = job["kind"][3:]
        record = m3._artifact(connection, production, "media-" + requirement["id"], kind,
                              kind + ".json", encode_json(result.model_dump(mode="json")),
                              provenance=metadata, job=job, attempt=attempt)
        if job["kind"] == "m3_event_cg_plan":
            characters = {row["character_id"]: row for row in payload["references"]}
            for cg in result.cgs:
                for variant in [None, *cg.variants]:
                    variant_id = variant.id if variant else None
                    m3._requirement(connection, production, "m3_event_cg", image_target(cg.id, variant_id), {
                        "cg_id": cg.id, "variant_id": variant_id,
                        "prompt": variant.prompt if variant else cg.prompt,
                        "cg_profile": payload["policy"]["generation_profile"],
                        "characters": [characters[cid] for cid in cg.character_ids],
                        "context": {key: payload["context"][key]
                                    for key in ("source_sha256", "narrative_artifact_id")},
                    })
        return record
    if result.status == "omitted":
        return m3._artifact(connection, production, "media-" + requirement["id"], "event_cg_omission",
                            "event-cg-omission.json", encode_json(result.model_dump(mode="json")),
                            provenance=metadata, job=job, attempt=attempt)
    metadata.update(references=payload["references"], prompt=payload["prompt"],
                    cg_profile=payload["cg_profile"], seed=payload["seed"],
                    source_sha256=payload["context"]["source_sha256"])
    original = m3._artifact(connection, production, "cg-source-" + requirement["id"], "event_cg_source",
                            "original.png", files["original.png"], provenance=metadata, job=job, attempt=attempt)
    metadata["original_artifact_id"] = original["id"]
    return m3._artifact(connection, production, "media-" + requirement["id"], "event_cg", "image.png",
                        files["image.png"], provenance=metadata, job=job, attempt=attempt)


def add_to_script(m3, connection, production, script, content, requirements):
    plans = [row for row in requirements if row["kind"] == "m3_event_cg_plan"]
    if not plans:
        return script
    plan = EventCgPlan.model_validate(_read(m3, connection, plans[0], "event_cg_plan", production["id"]))
    narrative = required(connection, "artifact", production["narrative_artifact_id"])
    if plan.source_sha256 != narrative["sha256"]:
        raise ValueError("Published CG plan differs from the narrative.")
    images = {row["target_id"]: row for row in requirements if row["kind"] == "m3_event_cg"}
    assets, segments = list(script.assets), []

    def resolve(cg_id, variant_id=None):
        row = images[image_target(cg_id, variant_id)]
        record = required(connection, "artifact", row["artifact_id"])
        metadata = json.loads(record["provenance"])
        if (record["project_id"] != production["project_id"]
                or metadata.get("production_id") != production["id"]
                or metadata.get("cg_id") != cg_id or metadata.get("variant_id") != variant_id):
            raise ValueError("CG publication references the wrong image.")
        data = m3.store.read(record)
        if record["kind"] == "event_cg_omission":
            EventCgResult.model_validate_json(data)
            return None
        if record["kind"] != "event_cg" or metadata.get("source_sha256") != narrative["sha256"]:
            raise ValueError("CG publication has an invalid image kind or narrative.")
        validate_png(data, require_transparency=False)
        identifier = "event-" + row["target_id"]
        assets.append({"id": identifier, "kind": "event_cg", "artifact_id": record["id"],
                       "filename": identifier + ".png", "sha256": record["sha256"]})
        content[identifier] = data
        return identifier

    order = {line.id: index for index, line in enumerate(script.utterances)}

    def boundary(identifier):
        return len(order) if identifier is None else order[identifier]

    for cg in plan.cgs:
        base = resolve(cg.id)
        if base is None:
            continue
        variants, end = [], cg.end_utterance_id
        current = cg
        for variant in cg.variants:
            if plan.planning_version >= 2:
                if current.staging is None:
                    raise ValueError("CG staging is required for safe image fallback.")
                if boundary(variant.start_utterance_id) > boundary(current.staging.safe_end_utterance_id):
                    # A missing image may be bridged only while the last image
                    # still matches the story. Do not reopen this CG later.
                    break
            image = resolve(cg.id, variant.id)
            if image:
                variants.append({"id": variant.id, "utterance_id": variant.start_utterance_id, "asset_id": image})
                current = variant
        if plan.planning_version >= 2:
            if current.staging is None:
                raise ValueError("CG staging is required for safe image fallback.")
            if boundary(current.staging.safe_end_utterance_id) < boundary(end):
                end = current.staging.safe_end_utterance_id
        segments.append({"id": cg.id, "start_utterance_id": cg.start_utterance_id,
                         "end_utterance_id": end, "base_asset_id": base, "variants": variants})
    return Script.model_validate(script.model_dump(mode="json") | {"assets": assets, "event_cg_segments": segments})


def summary(m3, connection, production, requirements, *, chapter_number=None, budget_requirements=None):
    settings = policy(connection, production)
    if not settings.max_cgs:
        return None
    budget_row = (_requirement(connection, production["storyline_id"], "m3_event_cg_budget", "event-cg-budget")
                  if budget_requirements is None else next(
                      (row for row in budget_requirements if row["kind"] == "m3_event_cg_budget"), None))
    limit, chapter_budgets, budget_omission_reason = None, [], ""
    budget_completed = bool(budget_row and budget_row["artifact_id"])
    if budget_completed:
        budget = EventCgBudget.model_validate(_read(m3, connection, budget_row, "event_cg_budget", production["storyline_id"]))
        budget_omission_reason = budget.omission_reason
        chapter_budgets = [{"chapter_number": row.chapter_number, "limit": row.limit} for row in budget.chapters]
        if chapter_number:
            limit = next((row.limit for row in budget.chapters if row.chapter_number == chapter_number), 0)
    images = [row for row in requirements if row["kind"] == "m3_event_cg"]
    generated, omissions = 0, []
    for row in images:
        if not row["artifact_id"]:
            continue
        record = required(connection, "artifact", row["artifact_id"])
        if record["kind"] == "event_cg":
            generated += 1
        elif record["kind"] == "event_cg_omission":
            value = EventCgResult.model_validate_json(m3.store.read(record))
            omissions.append({"cg_id": value.cg_id, "variant_id": value.variant_id, "reason": value.reason})
    plans = [row for row in requirements if row["kind"] == "m3_event_cg_plan"]
    omitted_count = len(omissions)
    plan_details, planning_notes = [], []
    planning_omitted = 0
    for row in requirements:
        if row["kind"] in {"m3_event_cg_budget", "m3_event_cg_plan"} and row["artifact_id"]:
            record = required(connection, "artifact", row["artifact_id"])
            value = json.loads(m3.store.read(record))
            if row["kind"] == "m3_event_cg_plan":
                plan = EventCgPlan.model_validate(value)
                owner_id = json.loads(record["provenance"])["production_id"]
                owner = required(connection, "m3_production", owner_id)
                # The frozen requirement owns the narrative used by this plan;
                # do not calculate its lengths against a later selected edition.
                frozen = json.loads(row["descriptor"])["context"]["narrative"]
                for cg in plan.cgs:
                    planned_images = {None: cg, **{variant.id: variant for variant in cg.variants}}
                    spans = []
                    for span in image_spans(cg, frozen):
                        image = planned_images[span["variant_id"]]
                        staging = image.staging
                        spans.append({**span, "reason": staging.reason if staging else image.interpretation,
                                      "visual_change": staging.change if staging else ""})
                    plan_details.append({"chapter_number": owner["chapter_number"], "cg_id": cg.id,
                                         "planning_version": plan.planning_version,
                                         "start_reason": cg.staging.reason if cg.staging else cg.interpretation,
                                         "end_reason": cg.end_reason, "composition": cg.composition,
                                         "images": spans})
                planning_notes.extend({"chapter_number": owner["chapter_number"], **note.model_dump(mode="json")}
                                      for note in plan.planning_notes)
            partial = value.get("prompt_omissions", [])
            planning_omitted += len(partial)
            omissions.extend({"cg_id": item["cg_id"], "variant_id": None, "reason": item["reason"]} for item in partial)
            reason = value.get("omission_reason")
            if reason and not partial:
                omissions.append({"cg_id": None, "variant_id": None, "reason": reason})
    return {"max_cgs": settings.max_cgs, "max_variants_per_cg": settings.max_variants_per_cg,
            "planned": len(images), "generated": generated, "omitted": omitted_count, "omissions": omissions,
            "planning_omitted": planning_omitted,
            "plans": plan_details, "planning_notes": planning_notes,
            "budget_omission_reason": budget_omission_reason,
            "budget_completed": budget_completed, "chapter_budgets": chapter_budgets,
            "chapter_budget": limit, "plan_completed": limit == 0 or bool(plans and all(row["artifact_id"] for row in plans))}
