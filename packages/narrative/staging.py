"""Pure, auditable limits for repeated stage directions."""

from __future__ import annotations

from copy import deepcopy

MAX_PAUSE_MS_PER_UTTERANCE = 1500
DIRECTION_NORMALIZATION_POLICY = "direction-normalization"
DIRECTION_NORMALIZATION_VERSION = 1
_TIMING_ORDER = {"before": 0, "start": 1, "after": 2}


def normalize_directions(directions: list[dict]) -> tuple[list[dict], dict]:
    """Normalize validated directions without changing input or surviving source order.

    Consecutive means consecutive in the compiler's (utterance, timing) group.
    IDs do not affect semantic equality. Pause runs retain the first ID and the
    longest duration, then each utterance shares a 1500 ms pause budget in actual
    execution order (before, start, after). Other direction durations are uncapped.
    Report indices always refer to the original input.
    """
    original = deepcopy(directions)
    rows = list(enumerate(deepcopy(directions)))
    changes: list[dict] = []

    def semantic(direction, *, ignore_duration=False):
        ignored = {"id", "duration_ms"} if ignore_duration else {"id"}
        return {key: value for key, value in direction.items() if key not in ignored}

    def change(action, index, direction, **details):
        changes.append({
            "action": action, "source_index": index,
            "direction_id": direction.get("id"),
            "utterance_id": direction["utterance_id"], "timing": direction["timing"],
            "kind": direction["kind"], **details,
        })

    def coalesce(values):
        previous = {}
        result = []
        for index, direction in values:
            anchor = (direction["utterance_id"], direction["timing"])
            retained = previous.get(anchor)
            if retained is not None:
                kept_index, kept = retained
                details = {"retained_source_index": kept_index,
                           "retained_direction_id": kept.get("id")}
                if semantic(kept) == semantic(direction):
                    change("duplicate", index, direction, **details)
                    continue
                if (kept["kind"] == direction["kind"] == "pause"
                        and semantic(kept, ignore_duration=True)
                        == semantic(direction, ignore_duration=True)):
                    duration = max(kept["duration_ms"], direction["duration_ms"])
                    change("pause_merge", index, direction, **details,
                           previous_duration_ms=kept["duration_ms"],
                           original_duration_ms=direction["duration_ms"],
                           normalized_duration_ms=duration)
                    kept["duration_ms"] = duration
                    continue
            previous[anchor] = (index, direction)
            result.append((index, direction))
        return result

    rows = coalesce(rows)
    used: dict[str, int] = {}
    dropped = set()
    for index, direction in sorted(rows, key=lambda row: _TIMING_ORDER[row[1]["timing"]]):
        if direction["kind"] != "pause":
            continue
        utterance = direction["utterance_id"]
        available = MAX_PAUSE_MS_PER_UTTERANCE - used.get(utterance, 0)
        duration = direction["duration_ms"]
        if duration > available:
            change("pause_clip" if available else "pause_drop", index, direction,
                   original_duration_ms=duration, normalized_duration_ms=available)
            if not available:
                dropped.add(index)
            else:
                direction["duration_ms"] = available
        used[utterance] = used.get(utterance, 0) + min(duration, available)
    # Dropping an over-budget pause can make two identical actions consecutive.
    # Coalesce once more so applying the policy again never changes the result.
    rows = coalesce([row for row in rows if row[0] not in dropped])
    result = [direction for _, direction in rows]
    report = {
        "policy": DIRECTION_NORMALIZATION_POLICY,
        "version": DIRECTION_NORMALIZATION_VERSION,
        "max_pause_ms_per_utterance": MAX_PAUSE_MS_PER_UTTERANCE,
        "input_count": len(directions), "output_count": len(result),
        "removed_count": len(directions) - len(result),
        "duplicate_count": sum(value["action"] == "duplicate" for value in changes),
        "merge_count": sum(value["action"] == "pause_merge" for value in changes),
        "clipped_count": sum(value["action"] in {"pause_clip", "pause_drop"}
                             for value in changes),
        "changes": changes,
    }
    if changes:
        report["original_directions"] = original
    return result, report
