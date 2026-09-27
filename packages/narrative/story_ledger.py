"""Deterministic provenance, state application and retrieval for narrative memory.

These checks establish lineage and referential integrity, not semantic truth.
Whether a passage really supports an observation is a separate review gate.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from packages.contracts.m4 import CharacterStoryState, StoryState
from packages.contracts.story_workflow import (
    AuthorFact,
    ChapterExtraction,
    ChapterMemory,
    ContextPacket,
    KnowledgeRecord,
    OpenThread,
    RepeatedPassage,
    ReviewReport,
    SourceExcerpt,
    SourceRef,
    StateEntry,
)


def content_hash(value: Any) -> str:
    """Hash a JSON value without changing ordering inside meaningful lists."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def memory_hash(value: dict | ChapterMemory) -> str:
    return content_hash(ChapterMemory.model_validate(value))


def project_story_state(memory: dict | ChapterMemory, character_ids: Iterable[str]) -> StoryState:
    """Build the legacy display projection; continuation uses ChapterMemory itself.

    The old schema cannot faithfully express qualified evidence or beliefs, so
    it never invents legacy fact IDs from those richer records.
    """
    memory = ChapterMemory.model_validate(memory)
    characters = []
    for character_id in sorted(set(character_ids)):
        uncertain = {(finding.scope, finding.entity_id, finding.key)
                     for finding in memory.pending_findings if not finding.resolved_by}
        state = {entry.key: entry.value for entry in memory.state
                 if entry.scope == "character" and entry.entity_id == character_id
                 and (entry.scope, entry.entity_id, entry.key) not in uncertain}

        def values(*prefixes, entries=state):
            return [value[:4000] for key, value in entries.items() if any(
                key == prefix or key.startswith(prefix + "_") for prefix in prefixes)]

        characters.append(CharacterStoryState(character_id=character_id,
            location=state.get("location", "未提示")[:4000],
            relationships=values("relationship")[:20], possessions=values("possession")[:30],
            injuries=values("physical_condition", "injury", "condition")[:20],
            promises=[thread.question[:4000] for thread in memory.threads
                      if thread.status == "open" and character_id in thread.character_ids][:30],
            knowledge=[], inner_emotion=state.get("emotion", "未提示")[:4000]))
    return StoryState(schema_version=1, chapter_number=memory.chapter_number,
        summary=memory.summary[:4000], characters=characters, facts=[], foreshadowing=[])


def _get(value: Any, key: str, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _scenes(value: Any) -> list:
    if isinstance(value, Mapping) or hasattr(value, "scenes"):
        value = _get(value, "scenes")
    return list(value)


def _unique(values: Iterable, label: str) -> None:
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}.")


def _ref_key(ref: SourceRef) -> tuple:
    return (ref.storyline_id, ref.chapter_number, ref.scene_id, ref.scene_revision,
            ref.utterance_id, ref.source_start, ref.source_end, ref.text_hash)


def source_catalog(scenes: Any, *, storyline_id: str, chapter_number: int) -> list[SourceRef]:
    """Produce exact source references for mapped utterances in NarrativeResult.scenes.

    Identical utterance IDs in other chapters or later scene revisions cannot
    alias these references. LLM short IDs should be expanded through this list.
    """
    return [item.ref for item in source_excerpts(
        scenes, storyline_id=storyline_id, chapter_number=chapter_number)]


def source_excerpts(scenes: Any, *, storyline_id: str,
                    chapter_number: int) -> list[SourceExcerpt]:
    result = []
    scenes = _scenes(scenes)
    _unique((_get(scene, "id") for scene in scenes), "scene IDs in source")
    for scene in scenes:
        scene_id = _get(scene, "id")
        raw = _get(scene, "raw_text")
        if not isinstance(raw, str):
            raise TypeError("A source scene must contain its original raw_text.")
        revision = text_hash(raw)
        utterances = _get(scene, "utterances", [])
        _unique((_get(u, "id") for u in utterances), "utterance IDs in scene")
        for utterance in utterances:
            start, end = _get(utterance, "source_start"), _get(utterance, "source_end")
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(raw)):
                raise ValueError("Utterance source span is outside the original scene.")
            original = raw[start:end]
            if original != _get(utterance, "display_text"):
                raise ValueError("Utterance text does not match its original source span.")
            ref = SourceRef(storyline_id=storyline_id, chapter_number=chapter_number,
                scene_id=scene_id, scene_revision=revision, utterance_id=_get(utterance, "id"),
                source_start=start, source_end=end, text_hash=text_hash(original))
            result.append(SourceExcerpt(ref=ref, text=original,
                                        speaker_id=_get(utterance, "speaker_id")))
    return result


def _resolve_from_scenes(ref: SourceRef, scenes: Any, *, storyline_id: str,
                         chapter_number: int) -> SourceExcerpt:
    if (ref.storyline_id, ref.chapter_number) != (storyline_id, chapter_number):
        raise ValueError("Evidence refers to a different storyline or chapter.")
    matching = [scene for scene in _scenes(scenes) if _get(scene, "id") == ref.scene_id]
    if len(matching) != 1:
        raise ValueError("Evidence refers to an unknown or ambiguous scene.")
    scene = matching[0]
    raw = _get(scene, "raw_text")
    if ref.scene_revision != text_hash(raw):
        raise ValueError("Evidence refers to a stale scene revision.")
    if ref.source_end > len(raw):
        raise ValueError("Evidence source span is outside the scene.")
    original = raw[ref.source_start:ref.source_end]
    if text_hash(original) != ref.text_hash:
        raise ValueError("Evidence text hash does not match the original source.")
    speaker = None
    if ref.utterance_id is not None:
        utterances = [u for u in _get(scene, "utterances")
                      if _get(u, "id") == ref.utterance_id]
        if len(utterances) != 1:
            raise ValueError("Evidence refers to an unknown or ambiguous utterance.")
        utterance = utterances[0]
        if (ref.source_start, ref.source_end) != (
                _get(utterance, "source_start"), _get(utterance, "source_end")):
            raise ValueError("Utterance evidence must name that utterance's exact span.")
        if original != _get(utterance, "display_text"):
            raise ValueError("Evidence utterance disagrees with its original source.")
        speaker = _get(utterance, "speaker_id")
    return SourceExcerpt(ref=ref, text=original, speaker_id=speaker)


def validate_evidence(refs: Iterable[dict | SourceRef], *,
                      narratives: Iterable[Any] = (),
                      sources: Iterable[dict | SourceExcerpt] = ()) -> list[SourceExcerpt]:
    """Resolve evidence against original narratives or previously verified excerpts.

    A narrative must have storyline_id, chapter_number and scenes. A reference
    with utterance_id=None can instead identify an exact span in raw_text.
    """
    refs = [SourceRef.model_validate(ref) for ref in refs]
    _unique((_ref_key(ref) for ref in refs), "evidence references")
    narratives = list(narratives)
    source_map = {}
    for item in sources:
        source = SourceExcerpt.model_validate(item)
        if text_hash(source.text) != source.ref.text_hash:
            raise ValueError("Stored source excerpt hash is corrupt.")
        key = _ref_key(source.ref)
        if key in source_map and source_map[key] != source:
            raise ValueError("Conflicting source excerpts share the same reference.")
        source_map[key] = source
    resolved = []
    for ref in refs:
        matching = [n for n in narratives if (_get(n, "storyline_id"),
                    _get(n, "chapter_number")) == (ref.storyline_id, ref.chapter_number)]
        if len(matching) > 1:
            raise ValueError("Ambiguous original chapter for evidence validation.")
        if matching:
            excerpt = _resolve_from_scenes(ref, _get(matching[0], "scenes"),
                storyline_id=ref.storyline_id, chapter_number=ref.chapter_number)
        else:
            excerpt = source_map.get(_ref_key(ref))
            if excerpt is None:
                raise ValueError("Evidence does not resolve to an available adopted source.")
        resolved.append(excerpt)
    return resolved


def empty_memory(storyline_id: str, *, state: Iterable[dict | StateEntry] = (),
                 author_facts: Iterable[dict | AuthorFact] = ()) -> ChapterMemory:
    """Create initial canon without treating planned developments as observations."""
    entries = [StateEntry.model_validate(entry) for entry in state]
    _unique(((e.scope, e.entity_id, e.key) for e in entries), "initial state keys")
    if any(e.changed_chapter != 0 or e.evidence for e in entries):
        raise ValueError("Initial state cannot cite prose that has not happened yet.")
    facts = [AuthorFact.model_validate(fact) for fact in author_facts]
    _unique((f.id for f in facts), "author fact IDs")
    if any(f.origin != "initial_canon" or f.cause_event_ids for f in facts):
        raise ValueError("Initial author facts must come from initial canon.")
    return ChapterMemory(storyline_id=storyline_id, chapter_number=0,
        summary="本編開始前。実績はまだありません。", state=entries, author_facts=facts)


def _all_evidence(extraction: ChapterExtraction) -> list[SourceRef]:
    return [ref for group in (extraction.events, extraction.state_deltas,
        extraction.knowledge_updates, extraction.thread_updates, extraction.introductions,
        extraction.pending_findings)
        for item in group for ref in item.evidence]


def _validate_memory(memory: ChapterMemory, *, allow_missing_sources: bool = False) -> None:
    """Reject duplicate identities or corrupted persisted evidence before replay."""
    for label, values in (("event IDs", memory.events), ("state delta IDs", memory.state_history),
                          ("knowledge IDs", memory.knowledge), ("thread IDs", memory.threads),
                          ("introduction IDs", memory.introductions),
                          ("author fact IDs", memory.author_facts),
                          ("pending finding IDs", memory.pending_findings)):
        _unique((v.id for v in values), label)
    _unique(((v.scope, v.entity_id, v.key) for v in memory.state), "state keys")
    _unique((_ref_key(source.ref) for source in memory.sources), "stored source references")
    evidence = {}
    for group in (memory.events, memory.state, memory.state_history, memory.knowledge,
                  memory.threads, memory.thread_history, memory.introductions,
                  memory.pending_findings):
        for item in group:
            for ref in item.evidence:
                if ref.storyline_id != memory.storyline_id or ref.chapter_number > memory.chapter_number:
                    raise ValueError("Stored evidence has invalid storyline or future lineage.")
                evidence[_ref_key(ref)] = ref
            if hasattr(item, "resolved_by"):
                for ref in item.resolved_by:
                    if ref.storyline_id != memory.storyline_id or ref.chapter_number > memory.chapter_number:
                        raise ValueError("Pending resolution has invalid lineage.")
                    evidence[_ref_key(ref)] = ref
    if allow_missing_sources:
        available = {_ref_key(source.ref) for source in memory.sources}
        evidence = {key: ref for key, ref in evidence.items() if key in available}
    validate_evidence(evidence.values(), sources=memory.sources)


def apply_chapter_memory(previous: dict | ChapterMemory, extraction: dict | ChapterExtraction, *,
                         scenes: Any, storyline_id: str, chapter_number: int,
                         historical_sources: Iterable[dict | SourceExcerpt] = (),
                         immutable_keys: Iterable[tuple[str, str, str]] = (),
                         allow_same_chapter: bool = False) -> ChapterMemory:
    """Validate an extraction and apply only its explicit state changes.

    This returns a new candidate and does not mutate/persist the predecessor.
    Its caller commits it only after the chapter's semantic reviews pass.
    Use apply_scene_memory for successive provisional scenes of one chapter.
    """
    previous = ChapterMemory.model_validate(previous)
    extraction = ChapterExtraction.model_validate(extraction)
    _validate_memory(previous)
    allowed = {previous.chapter_number + 1}
    if allow_same_chapter and previous.chapter_number > 0:
        allowed.add(previous.chapter_number)
    if (previous.storyline_id != storyline_id or extraction.chapter_number != chapter_number
            or chapter_number not in allowed):
        raise ValueError("Memory extraction must continue the exact preceding chapter/storyline.")
    scenes = _scenes(scenes)
    # Validate source mappings even when an extraction forgets to cite an utterance.
    current_sources = source_excerpts(scenes, storyline_id=storyline_id,
                                      chapter_number=chapter_number)
    current_scene_ids = {_get(scene, "id") for scene in scenes}
    current_revisions = {_get(scene, "id"): text_hash(_get(scene, "raw_text")) for scene in scenes}
    for stored in previous.sources:
        ref = stored.ref
        if (ref.chapter_number == chapter_number and ref.scene_id in current_revisions
                and ref.scene_revision != current_revisions[ref.scene_id]):
            raise ValueError("An adopted scene revision cannot be replaced by an append operation.")
    available_sources = [*previous.sources, *historical_sources, *current_sources]
    # Span-only references also resolve directly against the supplied raw text.
    unique_refs = {_ref_key(ref): ref for ref in _all_evidence(extraction)}
    resolved = {}
    for key, ref in unique_refs.items():
        if (ref.storyline_id == storyline_id and ref.chapter_number == chapter_number
                and ref.scene_id in current_scene_ids):
            resolved[key] = _resolve_from_scenes(ref, scenes, storyline_id=storyline_id,
                                                chapter_number=chapter_number)
        else:
            resolved[key] = validate_evidence([ref], sources=available_sources)[0]
    for group in (extraction.events, extraction.state_deltas, extraction.knowledge_updates,
                  extraction.thread_updates, extraction.introductions, extraction.pending_findings):
        for item in group:
            _unique((_ref_key(ref) for ref in item.evidence), "evidence references")
            if any(ref.storyline_id != storyline_id or ref.chapter_number > chapter_number
                   for ref in item.evidence):
                raise ValueError("New observations cannot cite another storyline or future prose.")
            if not any(ref.chapter_number == chapter_number and ref.scene_id in current_scene_ids
                       for ref in item.evidence):
                raise ValueError("Every new observation needs evidence in the supplied scene text.")

    old_events = {item.id: item for item in previous.events}
    new_events = {item.id: item for item in extraction.events}
    _unique((item.id for item in extraction.events), "new event IDs")
    if old_events.keys() & new_events.keys():
        raise ValueError("An event ID already belongs to adopted history.")
    events_by_id = {**old_events, **new_events}
    pending = set(new_events)
    resolved_ids = set(old_events)
    while pending:
        ready = {event_id for event_id in pending
                 if set(new_events[event_id].causes).issubset(resolved_ids)}
        if not ready:
            raise ValueError("Event causes contain an unknown ID or a causal cycle.")
        pending.difference_update(ready)
        resolved_ids.update(ready)

    state = {(item.scope, item.entity_id, item.key): item for item in previous.state}
    pending_findings = list(previous.pending_findings)
    immutable = set(immutable_keys)
    _unique((item.id for item in extraction.state_deltas), "new state delta IDs")
    if {d.id for d in previous.state_history} & {d.id for d in extraction.state_deltas}:
        raise ValueError("A state delta ID already belongs to adopted history.")
    current_deltas = [d for d in extraction.state_deltas if d.time_scope == "current"]
    _unique(((d.scope, d.entity_id, d.key) for d in current_deltas), "concurrent state updates")
    for delta in extraction.state_deltas:
        event = new_events.get(delta.event_id)
        if event is None:
            raise ValueError("A state change must cite an event extracted from the same scene batch.")
        if delta.time_scope == "historical":
            # History is recorded separately and cannot injure/heal the present.
            continue
        if event.presentation == "flashback":
            raise ValueError("A flashback event cannot directly change present state.")
        if event.assertion != "observed":
            raise ValueError("A current state change requires an observed event, not a report or belief.")
        key = (delta.scope, delta.entity_id, delta.key)
        old = state.get(key)
        uncertain = any(not finding.resolved_by and
                        (finding.scope, finding.entity_id, finding.key) == key
                        for finding in pending_findings)
        expected_before = None if uncertain else old.value if old else None
        if expected_before != delta.before:
            raise ValueError("State delta's before value does not match adopted state.")
        if key in immutable:
            raise ValueError("State delta changes an explicitly immutable canon condition.")
        if delta.after is None:
            state.pop(key)
        else:
            state[key] = StateEntry(scope=delta.scope, entity_id=delta.entity_id,
                key=delta.key, value=delta.after, changed_chapter=chapter_number,
                evidence=delta.evidence)
        pending_findings = [finding.model_copy(update={"resolved_by": delta.evidence})
                            if not finding.resolved_by and (finding.scope, finding.entity_id,
                            finding.key) == key else finding for finding in pending_findings]

    _unique((finding.id for finding in extraction.pending_findings), "new pending finding IDs")
    if {finding.id for finding in pending_findings} & {finding.id for finding in extraction.pending_findings}:
        raise ValueError("Pending finding ID already belongs to history.")
    for finding in extraction.pending_findings:
        key = (finding.scope, finding.entity_id, finding.key)
        if any(delta.time_scope == "current" and
               (delta.scope, delta.entity_id, delta.key) == key
               for delta in extraction.state_deltas):
            raise ValueError("A state key cannot be both confirmed and pending in one scene.")
        if any(not old.resolved_by and (old.scope, old.entity_id, old.key) == key
               for old in pending_findings):
            raise ValueError("Duplicate unresolved finding for the same state key.")
        if finding.last_known_value != (state[key].value if key in state else None):
            raise ValueError("Pending finding's last known value must match the recorded state.")
        if finding.resolved_by:
            raise ValueError("A new pending finding cannot already be resolved.")
        pending_findings.append(finding)

    knowledge = list(previous.knowledge)
    latest_knowledge = {(k.character_id, k.fact_id): k for k in knowledge}
    _unique((k.id for k in extraction.knowledge_updates), "new knowledge IDs")
    if {k.id for k in knowledge} & {k.id for k in extraction.knowledge_updates}:
        raise ValueError("A knowledge ID already belongs to adopted history.")
    _unique(((k.character_id, k.fact_id) for k in extraction.knowledge_updates),
            "concurrent knowledge updates")
    for update in extraction.knowledge_updates:
        if not set(update.event_ids).issubset(events_by_id):
            raise ValueError("Knowledge references an unknown event.")
        old = latest_knowledge.get((update.character_id, update.fact_id))
        if update.supersedes_id != (old.id if old else None):
            raise ValueError("A knowledge revision must explicitly supersede its prior record.")
        if old is None and update.kind == "retracted":
            raise ValueError("Knowledge cannot retract a belief that was never recorded.")
        record = KnowledgeRecord(**update.model_dump(), disclosed_chapter=chapter_number)
        knowledge.append(record)
        latest_knowledge[(record.character_id, record.fact_id)] = record

    threads = {thread.id: thread for thread in previous.threads}
    thread_history = list(previous.thread_history)
    _unique((item.id for item in extraction.thread_updates), "concurrent thread updates")
    for update in extraction.thread_updates:
        old = threads.get(update.id)
        if update.expected_status != (old.status if old else None):
            raise ValueError("A thread update must name its adopted previous status.")
        if old is None and update.status != "open":
            raise ValueError("A new unresolved thread cannot be automatically marked resolved.")
        if not set(update.event_ids).issubset(events_by_id):
            raise ValueError("A thread references an unknown event.")
        record = OpenThread(**update.model_dump(),
            introduced_chapter=old.introduced_chapter if old else chapter_number,
            changed_chapter=chapter_number)
        threads[record.id] = record
        thread_history.append(record)

    _unique((i.id for i in extraction.introductions), "new introduction IDs")
    if {i.id for i in previous.introductions} & {i.id for i in extraction.introductions}:
        raise ValueError("An introduction ID already belongs to adopted history.")
    sources = {_ref_key(item.ref): item for item in previous.sources}
    sources.update(resolved)
    # Only cited excerpts enter memory. Complete original prose remains a separate artifact.
    return ChapterMemory(storyline_id=storyline_id, chapter_number=chapter_number,
        revision=previous.revision + 1,
        previous_memory_hash=(previous.previous_memory_hash
            if previous.chapter_number == chapter_number else memory_hash(previous)),
        summary=extraction.summary, events=[*previous.events, *extraction.events],
        state=[state[key] for key in sorted(state)],
        state_history=[*previous.state_history, *extraction.state_deltas], knowledge=knowledge,
        threads=[threads[key] for key in sorted(threads)], thread_history=thread_history,
        introductions=[*previous.introductions, *extraction.introductions],
        sources=[sources[key] for key in sorted(sources, key=repr)],
        author_facts=previous.author_facts, pending_findings=pending_findings)


def apply_scene_memory(previous: dict | ChapterMemory, extraction: dict | ChapterExtraction, **kwargs
                       ) -> ChapterMemory:
    """Apply another scene to a provisional chapter without changing its predecessor hash."""
    kwargs["allow_same_chapter"] = True
    return apply_chapter_memory(previous, extraction, **kwargs)


def adopt_author_facts(previous: dict | ChapterMemory, facts: Iterable[dict | AuthorFact], *,
                       review: dict | ReviewReport,
                       authority_hashes: Mapping[str, str]) -> ChapterMemory:
    """Adopt separately reviewed author-only truth; never invent visible prose evidence.

    The caller supplies hashes of actual canon/author-event artifacts, not a plan
    summary. Event facts need an existing causal anchor and a matching review.
    """
    previous = ChapterMemory.model_validate(previous)
    _validate_memory(previous)
    facts = [AuthorFact.model_validate(fact) for fact in facts]
    report = ReviewReport.model_validate(review)
    if report.verdict != "pass" or report.subject_hash != content_hash(
            [fact.model_dump(mode="json") for fact in facts]):
        raise ValueError("Author facts need a passing review of this exact candidate.")
    _unique((fact.id for fact in facts), "new author fact IDs")
    if {f.id for f in facts} & {f.id for f in previous.author_facts}:
        raise ValueError("Author facts cannot overwrite adopted truth.")
    event_ids = {event.id for event in previous.events}
    for fact in facts:
        if authority_hashes.get(fact.authority_id) != fact.authority_hash:
            raise ValueError("Author fact authority is absent or has changed.")
        if fact.origin == "initial_canon" and previous.chapter_number != 0:
            raise ValueError("Initial secrets cannot be invented after story events begin.")
        if fact.origin == "author_event" and (
                not fact.cause_event_ids or not set(fact.cause_event_ids).issubset(event_ids)):
            raise ValueError("Off-screen author events need adopted causal evidence.")
    return previous.model_copy(update={"author_facts": [*previous.author_facts, *facts],
                                       "revision": previous.revision + 1})


def select_context(memory: dict | ChapterMemory, *, character_ids: Iterable[str] = (),
                   location_ids: Iterable[str] = (), event_ids: Iterable[str] = (),
                   max_events: int | None = None, viewpoint_character_id: str | None = None,
                   include_author_facts: bool = False) -> ContextPacket:
    """Retrieve full-history evidence by identity, unresolved duties and causal links.

    max_events is a capacity check, never permission to discard required facts.
    An oversized required set returns insufficient_evidence for scope splitting.
    Token measurement remains the generation layer's responsibility.
    """
    memory = ChapterMemory.model_validate(memory)
    _validate_memory(memory, allow_missing_sources=True)
    if max_events is not None and (type(max_events) is not int or max_events < 1):
        raise ValueError("max_events must be a positive integer or None.")
    characters, locations = set(character_ids), set(location_ids)
    requested = set(event_ids)
    all_scope = not (characters or locations or requested)
    entities = characters | locations
    events_by_id = {event.id: event for event in memory.events}
    missing = [f"Unknown required event: {event_id}" for event_id in sorted(requested - events_by_id.keys())]
    selected_ids = requested & events_by_id.keys()
    selected_ids |= {event.id for event in memory.events if all_scope
        or characters.intersection(event.character_ids) or locations.intersection(event.location_ids)}
    state = [entry for entry in memory.state if all_scope or entry.scope == "world"
        or (entry.scope == "character" and entry.entity_id in characters)
        or (entry.scope == "location" and entry.entity_id in locations)]
    latest_knowledge = {(k.character_id, k.fact_id): k for k in memory.knowledge}
    knowledge = [k for k in latest_knowledge.values()
        if (k.character_id == viewpoint_character_id if viewpoint_character_id is not None
            else all_scope or k.character_id in characters)]
    # Keep unresolved obligations even when their owner is temporarily off stage.
    threads = [thread for thread in memory.threads if thread.status == "open"]
    introductions = [item for item in memory.introductions
        if all_scope or entities.intersection(item.entity_ids)
        or characters.intersection(item.audience_character_ids)]
    for item in [*knowledge, *threads]:
        selected_ids.update(item.event_ids)
    # Include events behind evidence-bearing state/introduction records and
    # recursively retain their causes, including causes many chapters earlier.
    evidence_keys = {_ref_key(ref) for item in [*state, *knowledge, *threads, *introductions]
                     for ref in item.evidence}
    selected_ids.update(event.id for event in memory.events
                        if any(_ref_key(ref) in evidence_keys for ref in event.evidence))
    pending = list(selected_ids)
    while pending:
        event = events_by_id.get(pending.pop())
        if event is None:
            continue
        for cause in event.causes:
            if cause not in selected_ids:
                selected_ids.add(cause)
                pending.append(cause)
    unknown = selected_ids - events_by_id.keys()
    missing.extend(f"Unavailable causal event: {event_id}" for event_id in sorted(unknown))
    events = [event for event in memory.events if event.id in selected_ids]
    if max_events is not None and len(events) > max_events:
        missing.append(f"Required {len(events)} events exceed capacity {max_events}; split the scope.")
    evidence = {_ref_key(ref): ref for item in [*events, *state, *knowledge, *threads, *introductions]
                for ref in item.evidence}
    source_map = {_ref_key(item.ref): item for item in memory.sources}
    sources = []
    for key in sorted(evidence, key=repr):
        if key in source_map:
            sources.append(source_map[key])
        else:
            ref = evidence[key]
            missing.append(f"Missing source: chapter {ref.chapter_number} / {ref.scene_id}.")
    return ContextPacket(memory_hash=memory_hash(memory), chapter_number=memory.chapter_number,
        character_ids=sorted(characters), location_ids=sorted(locations), events=events,
        state=state, knowledge=knowledge, threads=threads, introductions=introductions,
        sources=sources, author_facts=memory.author_facts if include_author_facts else [],
        required_event_ids=sorted(selected_ids | requested), missing_information=missing,
        status="insufficient_evidence" if missing else "ready")


def find_repeated_passages(previous_scenes: Any, current_scenes: Any, *,
                            min_utterances: int = 4, min_characters: int = 80
                            ) -> list[RepeatedPassage]:
    """Find exact long runs as review candidates, preserving speakers and wording.

    It does not reject legitimate refrains or flashbacks. It reports where a
    reviewer must explain the repetition and its changed meaning, if intended.
    """
    if min_utterances < 1 or min_characters < 1:
        raise ValueError("Repetition thresholds must be positive.")
    previous_scenes, current_scenes = _scenes(previous_scenes), _scenes(current_scenes)
    result = []
    for previous in previous_scenes:
        old = _get(previous, "utterances")
        old_keys = [(_get(u, "speaker_id"), _get(u, "display_text")) for u in old]
        positions = defaultdict(list)
        for index, key in enumerate(old_keys):
            positions[key].append(index)
        for current in current_scenes:
            new = _get(current, "utterances")
            new_keys = [(_get(u, "speaker_id"), _get(u, "display_text")) for u in new]
            for new_start, key in enumerate(new_keys):
                for old_start in positions.get(key, ()):
                    # Each maximal diagonal match is emitted once.
                    if (old_start > 0 and new_start > 0
                            and old_keys[old_start - 1] == new_keys[new_start - 1]):
                        continue
                    length = 0
                    while (old_start + length < len(old_keys)
                           and new_start + length < len(new_keys)
                           and old_keys[old_start + length] == new_keys[new_start + length]):
                        length += 1
                    character_count = sum(len(text) for _, text
                                          in new_keys[new_start:new_start + length])
                    if length >= min_utterances and character_count >= min_characters:
                        result.append(RepeatedPassage(previous_scene_id=_get(previous, "id"),
                            current_scene_id=_get(current, "id"),
                            previous_utterance_ids=[_get(u, "id") for u
                                in old[old_start:old_start + length]],
                            current_utterance_ids=[_get(u, "id") for u
                                in new[new_start:new_start + length]],
                            character_count=character_count))
    return result
