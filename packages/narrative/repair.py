"""Remap reviewed scene metadata after source-preserving speech separation."""

from __future__ import annotations

from packages.contracts.m3 import NarrativeScene

from .speech import SpeechSeparation
from .validation import parse_scene_text


def remap_scene(
    scene: NarrativeScene,
    separation: SpeechSeparation,
    delivery_hints: dict[str, str] | None = None,
) -> NarrativeScene:
    """Keep dialogue annotations and move direction/review anchors to the new source.

    This does not repeat semantic review. The caller must validate the complete
    narrative, including its event evidence, before adopting the repaired scene.
    """
    original = {utterance.id: utterance for utterance in scene.utterances}
    parsed = parse_scene_text(separation.raw_text, scene.id, set(scene.plan.character_ids))
    normalized = {utterance.id: utterance for utterance in parsed}
    source_ids = {}
    for segment in separation.segments:
        if segment.kind == "delivery" or segment.utterance_id is None:
            continue
        if (segment.utterance_id in source_ids
                and source_ids[segment.utterance_id] != segment.original_utterance_id):
            raise ValueError("A repaired utterance must have exactly one original source.")
        if segment.original_utterance_id not in original or segment.utterance_id not in normalized:
            raise ValueError("Speech separation references an unknown utterance.")
        source_ids[segment.utterance_id] = segment.original_utterance_id
    if set(source_ids) != set(normalized):
        raise ValueError("Speech separation must account for every repaired utterance.")
    if set(separation.utterance_map) != set(original):
        raise ValueError("Speech separation must account for every original utterance.")
    for original_id, mapped_ids in separation.utterance_map.items():
        expected = [uid for uid in normalized if source_ids[uid] == original_id]
        if list(mapped_ids) != expected:
            raise ValueError("Speech separation has an inconsistent utterance map.")

    hints = delivery_hints or {}
    if any(uid not in normalized or normalized[uid].speaker_id is None for uid in hints):
        raise ValueError("Delivery hints must reference repaired dialogue.")
    utterances = []
    for utterance in parsed:
        values = utterance.model_dump()
        if utterance.speaker_id is not None:
            previous = original[source_ids[utterance.id]]
            values.update(
                inner_emotion=previous.inner_emotion,
                voice_emotion=previous.voice_emotion,
                delivery=hints.get(utterance.id, previous.delivery),
            )
        # Revalidation enforces delivery length, including supplied overrides.
        utterances.append(type(utterance).model_validate(values))

    directions = []
    for direction in scene.directions:
        mapped_ids = separation.utterance_map.get(direction.utterance_id)
        if mapped_ids is None:
            raise ValueError("Direction references an unknown original utterance.")
        if direction.kind == "focus":
            target = next((uid for uid in mapped_ids
                           if normalized[uid].speaker_id is not None
                           and normalized[uid].speaker_id == direction.character_id), None)
            if target is None:
                continue
        else:
            if not mapped_ids:
                raise ValueError("A stage direction has no repaired utterance to anchor to.")
            target = mapped_ids[-1] if direction.timing == "after" else mapped_ids[0]
        directions.append(direction.model_copy(update={"utterance_id": target}))

    events = []
    for event in scene.review.events:
        evidence = []
        for original_id in event.evidence_utterance_ids:
            if original_id not in separation.utterance_map:
                raise ValueError("Review evidence references an unknown original utterance.")
            for utterance_id in separation.utterance_map[original_id]:
                if utterance_id not in evidence:
                    evidence.append(utterance_id)
        events.append({**event.model_dump(), "evidence_utterance_ids": evidence})
    values = scene.model_dump()
    values.update(
        raw_text=separation.raw_text,
        utterances=utterances,
        directions=directions,
        review={**scene.review.model_dump(), "events": events},
    )
    return NarrativeScene.model_validate(values)


def voice_reuse_map(
    old_scene: NarrativeScene,
    new_scene: NarrativeScene,
    separation: SpeechSeparation,
) -> dict[str, str]:
    """Find exact audio-input matches; the caller must also check voice references."""
    if old_scene.id != new_scene.id or new_scene.raw_text != separation.raw_text:
        raise ValueError("Voice reuse must compare the same separated scene.")
    old = {utterance.id: utterance for utterance in old_scene.utterances}
    new = {utterance.id: utterance for utterance in new_scene.utterances}
    fields = ("speaker_id", "spoken_text", "voice_emotion", "delivery")
    reusable = {}
    for original_id, mapped_ids in separation.utterance_map.items():
        if original_id not in old:
            raise ValueError("Voice reuse references an unknown original utterance.")
        previous = old[original_id]
        for utterance_id in mapped_ids:
            if utterance_id not in new:
                raise ValueError("Voice reuse references an unknown repaired utterance.")
            current = new[utterance_id]
            if current.speaker_id is not None and all(
                getattr(previous, field) == getattr(current, field) for field in fields
            ):
                reusable[utterance_id] = original_id
    return reusable
