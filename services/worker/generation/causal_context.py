"""Purpose-specific views of immutable memory, with reversible evidence lookup.

Index excerpts are discovery hints, never evidence. Source quotations and state
values selected for a task are kept verbatim; token limits belong to the caller.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable

from packages.contracts.m3 import NarrativeScene
from packages.contracts.story_workflow import ChapterMemory, ReviewReport
from packages.narrative.story_ledger import memory_hash as hash_memory
from packages.narrative.story_ledger import source_catalog, text_hash, validate_evidence
from packages.narrative.validation import parse_scene_text

INDEX_DESCRIPTION_CHARACTERS = 120
INDEX_INSTRUCTION = (
    "event_index/thread_indexは過去記録を探すための索引です。excerptは全文ではなく、"
    "索引だけを原文の照合済み根拠としてpassにしません。必要な記録はrequested_event_idsに"
    "eventのid、必要な原文はrequested_source_idsにevidenceのq IDを指定して取得します。"
    "取得された全文・原文を照合してから判断し、取得できなければinsufficient_evidence。"
    "state/knowledge等は採用済み台帳の現在値であり、この呼出で再検証した事実ではありません。"
    "これらのevidenceの原文が未添付なら同じq IDで取得できます。"
    "q IDはこのmemory_hashだけで有効です。今回の資料にない事実を存在しないと解釈しません。"
)
PRECEDING_SCENE_INSTRUCTION = (
    "direct_preceding_sceneは直前に採用済みの場面の完全な原文であり、今回の場面の文例ではありません。"
    "既に起きた行動・会話・到達点を引き継ぎ、同じ導入や発見を新たな出来事として再演しません。"
    "source_references_to_preceding_sceneのq IDは、この原文の発話・文字範囲を指す完全な証拠です。"
    "原文を添付しただけで意味検査をpassにしません。採用済み台帳と原文が矛盾する場合は"
    "extraction/contextの修復として報告し、採用済み原文の改変や都合のよい推測で埋めません。"
)


def _encoded(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _records(memory):
    return [*memory.events, *memory.state, *memory.state_history, *memory.knowledge,
            *memory.threads, *memory.thread_history, *memory.introductions]


class _Evidence:
    def __init__(self, memory):
        self.memory = memory
        refs = {_encoded(ref): ref for record in _records(memory) for ref in record.evidence}
        sources = {}
        for source in memory.sources:
            key = _encoded(source.ref)
            if key in sources:
                raise ValueError("Duplicate source reference in causal memory.")
            sources[key] = source
            refs[key] = source.ref
        self.aliases = {key: f"q{index}" for index, key in enumerate(sorted(refs), 1)}
        self.references = {self.aliases[key]: ref for key, ref in refs.items()}
        self.sources = {self.aliases[key]: value for key, value in sources.items()}

    def small(self, record):
        value = record.model_dump(mode="json")
        if "evidence" in value:
            value["evidence"] = [self.aliases[_encoded(ref)] for ref in record.evidence]
        return value

    def source_ids(self, records):
        return {self.aliases[_encoded(ref)] for record in records for ref in record.evidence}

    def quotes(self, source_ids):
        quotes, missing = [], []
        for alias in sorted(set(source_ids), key=lambda key: (len(key), key)):
            source = self.sources.get(alias)
            if source is None:
                missing.append(f"Unavailable source ID: {alias}")
                continue
            ref = source.ref
            if (ref.storyline_id != self.memory.storyline_id
                    or ref.chapter_number > self.memory.chapter_number
                    or text_hash(source.text) != ref.text_hash):
                missing.append(f"Invalid source identity or text hash: {alias}")
                continue
            quotes.append({"id": alias, "chapter": ref.chapter_number, "scene_id": ref.scene_id,
                           "utterance_id": ref.utterance_id, "speaker": source.speaker_id,
                           "text": source.text})
        return quotes, missing

    def index(self, record, description):
        excerpt = description[:INDEX_DESCRIPTION_CHARACTERS]
        return {"id": record.id, "excerpt": excerpt,
                "description_complete": len(excerpt) == len(description),
                "evidence_status": "index_only", "evidence": sorted(self.source_ids([record]))}


def _event_map(memory):
    result = {event.id: event for event in memory.events}
    if len(result) != len(memory.events):
        raise ValueError("Duplicate event ID in causal memory.")
    return result


def _chapter(record):
    return max((ref.chapter_number for ref in record.evidence), default=0)


def _scene_key(ref):
    return ref.chapter_number, ref.scene_id, ref.scene_revision


def _preceding_scene(events):
    """Use adopted event order, never scene-name sorting or historical citations.

    Every extracted event cites its own scene. If its evidence also refers to
    history, the last scene first encountered in adopted event order is current.
    An ambiguous imported ledger must request clarification rather than replay
    every candidate scene as if all of them were the immediate predecessor.
    """
    if not events:
        return None, []
    latest_chapter = _chapter(events[-1])
    candidates = {_scene_key(ref) for ref in events[-1].evidence
                  if ref.chapter_number == latest_chapter}
    first_seen = {}
    for index, event in enumerate(events):
        for ref in event.evidence:
            first_seen.setdefault(_scene_key(ref), index)
    newest = max(first_seen[key] for key in candidates)
    candidates = {key for key in candidates if first_seen[key] == newest}
    if len(candidates) != 1:
        return None, ["Cannot identify the immediately preceding adopted scene from event evidence."]
    return candidates.pop(), []


def _related(record, characters, locations):
    return bool(characters.intersection(record.character_ids)
                or locations.intersection(record.location_ids))


def _review_coverage(reviews, expected):
    reports = [ReviewReport.model_validate(review) for review in reviews]
    continuity, extraction = set(), set()
    for report in reports:
        if report.verdict != "pass":
            continue
        if report.scope.startswith("scene-continuity-"):
            continuity.update(report.checked_scene_ids)
        if (report.scope.startswith("extraction-review-")
                and {"support", "completeness"}.issubset(report.checked_categories)):
            extraction.update(report.checked_scene_ids)
    missing = [f"Missing accepted scene continuity review: {scene}" for scene in sorted(expected - continuity)]
    missing += [f"Missing accepted extraction support/completeness review: {scene}"
                for scene in sorted(expected - extraction)]
    return {"expected_scene_ids": sorted(expected), "continuity_scene_ids": sorted(continuity),
            "extraction_scene_ids": sorted(extraction), "prior_reviews": [
                {"scope": r.scope, "subject_hash": r.subject_hash, "input_hash": r.input_hash,
                 "verdict": r.verdict, "checked_scene_ids": r.checked_scene_ids,
                 "checked_categories": r.checked_categories} for r in reports]}, missing


def build_context(memory, *, scope: str, character_ids: Iterable[str] = (),
                  location_ids: Iterable[str] = (), event_ids: Iterable[str] = (),
                  chapter_number: int | None = None, include_author_facts: bool = False,
                  scene_reviews=(), expected_scene_ids: Iterable[str] = ()) -> dict:
    """Build a bounded-by-purpose view, never silently truncate required sources.

The caller must measure the exact template/schema and handle an oversized scope;
automatic partitioning is not yet implemented. ``ready`` means requested records
are available, not that a review passed.
"""
    memory = ChapterMemory.model_validate(memory)
    if scope not in {"plan", "scene", "extraction", "progress"}:
        raise ValueError("Unknown causal context scope.")
    chapter = memory.chapter_number if chapter_number is None else chapter_number
    if type(chapter) is not int or not 0 <= chapter <= memory.chapter_number:
        raise ValueError("Context chapter must be an available observed chapter.")
    characters, locations, required = set(character_ids), set(location_ids), set(event_ids)
    by_id = _event_map(memory)
    unknown = required - by_id.keys()
    if unknown:
        raise ValueError("Unknown required event IDs: " + ", ".join(sorted(unknown)))
    evidence = _Evidence(memory)
    all_entities = scope in {"plan", "progress"} or not (characters or locations)
    open_findings = [finding for finding in memory.pending_findings if not finding.resolved_by]
    uncertain_keys = {(finding.scope, finding.entity_id, finding.key) for finding in open_findings}
    state = [s for s in memory.state if (s.scope, s.entity_id, s.key) not in uncertain_keys
             and (all_entities or s.scope == "world"
             or (s.scope == "character" and s.entity_id in characters)
             or (s.scope == "location" and s.entity_id in locations))]
    relevant_findings = [finding for finding in open_findings if all_entities
                         or finding.scope == "world"
                         or (finding.scope == "character" and finding.entity_id in characters)
                         or (finding.scope == "location" and finding.entity_id in locations)]
    latest = {(k.character_id, k.fact_id): k for k in memory.knowledge}
    knowledge = [k for k in latest.values() if all_entities or k.character_id in characters]
    open_threads = [t for t in memory.threads if t.status == "open"]
    threads = [t for t in open_threads if all_entities or _related(t, characters, locations)
               or not (t.character_ids or t.location_ids) or required.intersection(t.event_ids)]
    introductions = [i for i in memory.introductions if all_entities
                     or (characters | locations).intersection(i.entity_ids)
                     or characters.intersection(i.audience_character_ids)]
    selected = set(required)
    linked, scene_source_ids, missing = set(), set(), []
    if scope in {"plan", "progress"}:
        selected.update(e.id for e in memory.events if _chapter(e) == chapter)
    elif scope == "scene":
        # Adopted values remain complete below. Their older proof is addressable
        # through q aliases, not copied into every later scene with the same cast.
        linked = {event_id for item in [*knowledge, *threads] for event_id in item.event_ids}
        previous_scene, scene_missing = _preceding_scene(memory.events)
        missing.extend(scene_missing)
        if previous_scene is not None:
            selected.update(e.id for e in memory.events if any(
                _scene_key(ref) == previous_scene for ref in e.evidence))
            # Include adopted excerpts used only by knowledge/introduction/state
            # records in that scene, and report missing required excerpts too.
            scene_source_ids = {alias for alias, ref in evidence.references.items()
                                if _scene_key(ref) == previous_scene}
    missing.extend(f"Unavailable linked event: {event_id}"
                   for event_id in sorted((selected | linked) - by_id.keys()))
    selected &= by_id.keys()
    events = [event for event in memory.events if event.id in selected] if scope != "extraction" else []
    deltas = [delta for delta in memory.state_history if _chapter(delta) == chapter] if scope == "progress" else []
    quoted_records = ([*events, *deltas, *threads] if scope == "progress" else
                      [by_id[event_id] for event_id in required] if scope == "scene" else
                      events if scope == "plan" else [])
    quotes, source_missing = evidence.quotes(evidence.source_ids(quoted_records) | scene_source_ids)
    missing.extend(source_missing)
    # Preserve every causal edge in a compact lookup index. Indexed ancestors
    # are available for follow-up checks without inserting their original prose.
    indexed_events = memory.events if scope in {"plan", "extraction"} else [
        e for e in memory.events if e.id in (selected | linked) or _related(e, characters, locations)]
    causal_roots = [*events, *(by_id[event_id] for event_id in linked if event_id in by_id)]
    ancestors = {cause for event in causal_roots for cause in event.causes}
    pending = list(ancestors)
    while pending:
        cause_id = pending.pop()
        cause = by_id.get(cause_id)
        if cause is None:
            missing.append(f"Unavailable causal event: {cause_id}")
            continue
        for parent in cause.causes:
            if parent not in ancestors:
                ancestors.add(parent)
                pending.append(parent)
    indexed_ids = {e.id for e in indexed_events} | ancestors
    index = [{**evidence.index(e, e.description), "character_ids": e.character_ids,
              "location_ids": e.location_ids, "causes": e.causes}
             for e in memory.events if e.id in indexed_ids]
    coverage, coverage_missing = _review_coverage(scene_reviews, set(expected_scene_ids))
    if scope == "progress":
        missing.extend(coverage_missing)
    coverage.update({"supplied_event_ids": [e.id for e in events],
                     "index_only_event_ids": sorted(indexed_ids - {e.id for e in events}),
                     "supplied_source_ids": [q["id"] for q in quotes],
                     "new_semantic_checks_performed": False})
    return {"scope": scope, "memory_hash": hash_memory(memory), "chapter_number": chapter,
            "status": "insufficient_evidence" if missing else "ready",
            "missing_information": list(dict.fromkeys(missing)), "retrieval_instructions": INDEX_INSTRUCTION,
            "events": [evidence.small(e) for e in events], "event_index": index,
            "state": [evidence.small(s) for s in state], "state_deltas": [evidence.small(d) for d in deltas],
            **({"pending_findings": [finding.model_dump(mode="json")
                                     for finding in relevant_findings]} if relevant_findings else {}),
            "knowledge": [evidence.small(k) for k in knowledge],
            "unresolved": [evidence.small(t) for t in threads],
            "thread_index": [{**evidence.index(t, t.question), "status": t.status,
                              "event_ids": t.event_ids} for t in memory.threads],
            "already_introduced": [evidence.small(i) for i in introductions],
            "author_secrets_not_shared_knowledge": [fact.model_dump(mode="json")
                for fact in memory.author_facts] if include_author_facts else [],
            "source_quotes": quotes, "coverage": coverage}


def retrieve_context_evidence(memory, *, memory_hash: str, record_ids: Iterable[str] = (),
                              source_ids: Iterable[str] = ()) -> dict:
    """Resolve full event records and q aliases against the exact indexed snapshot."""
    memory = ChapterMemory.model_validate(memory)
    if hash_memory(memory) != memory_hash:
        raise ValueError("Evidence lookup memory_hash is stale or belongs to another snapshot.")
    evidence = _Evidence(memory)
    records = _event_map(memory)
    requested = set(record_ids)
    missing = [f"Unknown requested event ID: {record_id}" for record_id in sorted(requested - records.keys())]
    selected = [record for record in memory.events if record.id in requested]
    quotes, source_missing = evidence.quotes(set(source_ids) | evidence.source_ids(selected))
    missing.extend(source_missing)
    return {"memory_hash": memory_hash, "events": [evidence.small(record) for record in selected],
            "source_quotes": quotes, "status": "insufficient_evidence" if missing else "ready",
            "missing_information": missing, "new_semantic_checks_performed": False}


def attach_preceding_scene(packet: dict, memory: ChapterMemory, preceding: dict) -> dict:
    """Attach exact adopted prose once, keeping q evidence references reversible.

    This validates source identity and mappings, not semantic consistency. It
    never shortens prose or upgrades a packet/review verdict to accommodate an
    input budget; the caller must measure the complete returned packet.
    """
    memory = ChapterMemory.model_validate(memory)
    if packet.get("memory_hash") != hash_memory(memory):
        raise ValueError("Preceding scene packet memory_hash does not match its adopted memory.")
    chapter = preceding.get("chapter_number")
    if type(chapter) is not int or not 1 <= chapter <= memory.chapter_number:
        raise ValueError("Preceding scene chapter must be an available adopted chapter.")
    scene = NarrativeScene.model_validate(preceding.get("scene"))
    revision = text_hash(scene.raw_text)
    expected, missing = _preceding_scene(memory.events)
    if missing or expected != (chapter, scene.id, revision):
        raise ValueError("Preceding scene does not match the immediately preceding adopted scene/revision.")
    if scene.id != scene.plan.id:
        raise ValueError("Preceding scene and plan IDs differ.")
    # Validate the entire source, including utterances not selected by extraction.
    source_catalog([scene], storyline_id=memory.storyline_id, chapter_number=chapter)
    parsed = parse_scene_text(scene.raw_text, scene.id, set(scene.plan.character_ids))
    fields = ("id", "speaker_id", "display_text", "spoken_text", "source_start", "source_end")
    if len(parsed) != len(scene.utterances) or any(
            any(getattr(actual, field) != getattr(original, field) for field in fields)
            for actual, original in zip(scene.utterances, parsed, strict=False)):
        raise ValueError("Preceding source mapping changed text, speaker, order, or source span.")
    evidence = _Evidence(memory)
    matching = {alias: ref for alias, ref in evidence.references.items() if _scene_key(ref) == expected}
    resolved = validate_evidence(matching.values(), narratives=[{
        "storyline_id": memory.storyline_id, "chapter_number": chapter, "scenes": [scene]}])
    for alias, original in zip(matching, resolved, strict=True):
        stored = evidence.sources.get(alias)
        if stored is not None and stored != original:
            raise ValueError(f"Stored preceding source differs from adopted full prose: {alias}")
    direct = {"chapter_number": chapter, "scene_id": scene.id, "scene_revision": revision,
              "raw_text": scene.raw_text, "utterances": [
                  {"id": u.id, "speaker": u.speaker_id,
                   "source_start": u.source_start, "source_end": u.source_end}
                  for u in scene.utterances]}
    if packet.get("direct_preceding_scene", direct) != direct:
        raise ValueError("Packet already contains a different preceding scene.")

    def pointer(alias):
        ref = matching.get(alias)
        if ref is None:
            raise ValueError(f"Source ID does not belong to the preceding scene: {alias}")
        return {"utterance_id": ref.utterance_id, "source_start": ref.source_start,
                "source_end": ref.source_end, "text_hash": ref.text_hash}

    result = copy.deepcopy(packet)
    references = result.get("source_references_to_preceding_scene", {})
    for alias, existing in references.items():
        if existing != pointer(alias):
            raise ValueError(f"Preceding source pointer differs from adopted evidence: {alias}")
    remaining = []
    for quote in result.get("source_quotes", []):
        alias = quote.get("id")
        ref = evidence.references.get(alias)
        if ref is None:
            raise ValueError(f"Unknown source ID in context packet: {alias}")
        if _scene_key(ref) != expected:
            remaining.append(quote)
            continue
        originals, quote_missing = evidence.quotes([alias])
        if quote_missing or any(quote.get(key) != value for key, value in originals[0].items()):
            raise ValueError(f"Context quotation differs from adopted evidence: {alias}")
        references[alias] = pointer(alias)
    result["source_quotes"] = remaining
    result["source_references_to_preceding_scene"] = references
    result["direct_preceding_scene"] = direct
    result["preceding_scene_instructions"] = PRECEDING_SCENE_INSTRUCTION
    # supplied_source_ids is unchanged: deduplicated evidence is still supplied,
    # now as exact spans of direct_preceding_scene rather than a second text copy.
    coverage = result.setdefault("coverage", {})
    coverage["direct_preceding_scene"] = {
        "chapter_number": chapter, "scene_id": scene.id, "scene_revision": revision,
        "complete_raw_text_supplied": True, "deduplicated_source_ids": sorted(references),
        "new_semantic_checks_performed": False}
    return result
