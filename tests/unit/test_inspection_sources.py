"""Source aliases preserve occurrences and never recover references from text alone."""

import pytest

from packages.narrative.story_ledger import source_excerpts, text_hash
from services.worker.generation.causal_runtime import digest
from services.worker.generation.inspection_sources import (
    build_inspection_catalog,
    resolve_inspection_quote,
    validate_catalog,
)
from tests.unit.test_causal_fact_comparison import scene


def excerpt(text="同じ台詞。", *, chapter=1, trailing=""):
    value = scene("Hero: " + text + trailing)
    return source_excerpts([value], storyline_id="story", chapter_number=chapter)[0]


def test_same_utterance_and_identical_text_in_another_chapter_or_revision_keep_separate_ids():
    first = excerpt()
    revised = excerpt(trailing="\nkeeper: 別版の発言。")
    current = excerpt(chapter=2)
    assert first.ref.utterance_id == revised.ref.utterance_id == current.ref.utterance_id
    assert first.text == revised.text == current.text
    catalog = build_inspection_catalog(earlier=[first, first, revised], current=[current])
    assert [s.source_id for s in catalog.sources] == ["src001", "src002", "src003"]
    refs = [resolve_inspection_quote(catalog, source_id=s.source_id, quote=s.text,
        phase=s.phase, allowed_source_ids=[s.source_id]).ref for s in catalog.sources]
    assert refs == [first.ref, revised.ref, current.ref]
    assert len({digest(ref) for ref in refs}) == 3


def test_subquote_preserves_complete_ref_and_resolves_unique_absolute_offsets():
    original = excerpt("一枚は貼った。二枚目は未着手。")
    catalog = build_inspection_catalog(earlier=[original], current=[])
    quote = "二枚目は未着手。"
    resolved = resolve_inspection_quote(catalog, source_id="src001", quote=quote,
        phase="earlier", allowed_source_ids=["src001"])
    assert resolved.ref == original.ref
    assert resolved.quote_start == original.ref.source_start + original.text.index(quote)
    assert resolved.quote_end == original.ref.source_end
    assert resolved.quote_hash == text_hash(quote)


def test_repeated_substring_requires_wider_unique_quote_not_first_occurrence():
    catalog = build_inspection_catalog(earlier=[excerpt("待って。まだだ。待って。今だ。")], current=[])
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_inspection_quote(catalog, source_id="src001", quote="待って。",
            phase="earlier", allowed_source_ids=["src001"])
    resolved = resolve_inspection_quote(catalog, source_id="src001", quote="待って。今だ。",
        phase="earlier", allowed_source_ids=["src001"])
    assert resolved.quote_start > resolved.ref.source_start


@pytest.mark.parametrize("key,phase,allowed", [
    ("s1-u1", "earlier", ["src001"]), ("q1", "earlier", ["src001"]),
    ("src099", "earlier", ["src001"]), ("src001", "current", ["src001"]),
    ("src001", "earlier", ["src002"]),
])
def test_no_legacy_unpresented_wrong_role_or_wrong_fact_fallback(key, phase, allowed):
    catalog = build_inspection_catalog(earlier=[excerpt()], current=[excerpt(chapter=2)])
    with pytest.raises(ValueError, match="Invalid source_id"):
        resolve_inspection_quote(catalog, source_id=key, quote="同じ台詞。",
            phase=phase, allowed_source_ids=allowed)


@pytest.mark.parametrize("change", ["mapping", "text", "span", "role", "revision"])
def test_catalog_rejects_mutated_source_snapshot(change):
    catalog = build_inspection_catalog(earlier=[excerpt()], current=[excerpt(chapter=2)])
    value = catalog.model_dump(mode="json")
    source = value["sources"][0]
    if change == "mapping":
        source["source_id"] = "src002"
    elif change == "text":
        source["text"] = "違う本文。"
    elif change == "span":
        source["ref"]["source_end"] += 1
    elif change == "role":
        source["phase"] = "current"
    else:
        source["ref"]["scene_revision"] = "f" * 64
    with pytest.raises(ValueError):
        validate_catalog(value)


def test_same_complete_reference_cannot_be_both_before_and_after():
    original = excerpt()
    with pytest.raises(ValueError, match="Conflicting source content or roles"):
        build_inspection_catalog(earlier=[original], current=[original])
