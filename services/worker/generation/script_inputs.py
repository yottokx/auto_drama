"""Stage-specific views of complete saved story data; source registries stay intact."""

import json
from copy import deepcopy


def scene_material(number, plan, context, following_plans):
    result = deepcopy(context)
    result["current_chapter"] = number
    setting = result["setting"]
    active = set(plan.character_ids)
    mentioned = plan.model_dump_json() + result.get("planned_connection", "")
    relevant = active | {row["id"] for row in setting["characters"]
                         if row["id"] in mentioned or row["name"] in mentioned}
    result["material_selection"] = {
        "full_character_ids": sorted(relevant),
        "identity_only_character_ids": [row["id"] for row in setting["characters"] if row["id"] not in relevant],
        "included_location_ids": [plan.location_id],
        "excluded_location_ids": [row["id"] for row in setting["locations"] if row["id"] != plan.location_id],
        "reason": "current_scene_and_named_references; source_registries_unchanged",
    }
    setting["characters"] = [row if row["id"] in relevant else
        {key: row[key] for key in ("id", "name", "role") if key in row}
        for row in setting["characters"]]
    setting["locations"] = [row for row in setting["locations"] if row["id"] == plan.location_id]
    relationships = setting.get("relationships", {})
    if isinstance(relationships, dict) and "pairs" in relationships:
        result["material_selection"]["excluded_relationship_pairs"] = [row["characterIds"]
            for row in relationships["pairs"] if not relevant.intersection(row["characterIds"])]
        relationships["pairs"] = [row for row in relationships["pairs"]
                                    if relevant.intersection(row["characterIds"])]
    for key, ids_key in (("planned_connections_not_events", "character_ids"),
                         ("initial_cast_context_not_events", "character_id")):
        result["material_selection"]["excluded_" + key] = [row[ids_key] for row in setting.get(key, [])
            if not relevant.intersection(row[ids_key] if isinstance(row[ids_key], list) else [row[ids_key]])]
        setting[key] = [row for row in setting.get(key, []) if relevant.intersection(
            row[ids_key] if isinstance(row[ids_key], list) else [row[ids_key]])]
    # Preserve all public scene boundary fields but avoid repeating this scene's
    # location definition and endpoint; both already have a single source here.
    scope = {"following_scenes_not_yet_executed": [
        row.model_dump(mode="json", exclude={"objectives", "atmosphere"}) for row in following_plans]}
    result["brief"] += ("\n今回の場面計画（未実施の予定）:\n" + plan.model_dump_json()
                        + "\n執筆範囲と後続場面の担当（実績ではない）:\n"
                        + json.dumps(scope, ensure_ascii=False))
    if result.get("future"):
        for row in result["future"]["routes"]:
            if row["number"] == number:
                for key in ("events", "route", "conversation_topics"):
                    row.pop(key, None)
    return result
