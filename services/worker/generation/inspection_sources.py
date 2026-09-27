"""One model-facing source namespace backed by immutable, complete references.

Aliases identify occurrences, never text alone. No legacy-ID fallback or search
outside the supplied catalog is performed when resolving model output.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from packages.contracts.m2 import CharacterId
from packages.contracts.script import Contract
from packages.contracts.story_workflow import Digest, SourceExcerpt, SourceRef
from packages.narrative.story_ledger import text_hash

from .causal_runtime import digest

SourceId = Annotated[str, StringConstraints(pattern=r"^src[0-9]{3,}$", max_length=64)]
SourcePhase = Literal["earlier", "current"]


class InspectionSource(Contract):
    source_id: SourceId
    phase: SourcePhase
    ref: SourceRef
    text: str = Field(min_length=1, max_length=100_000)
    speaker_id: CharacterId | None

    @model_validator(mode="after")
    def exact_excerpt(self):
        if (text_hash(self.text) != self.ref.text_hash
                or len(self.text) != self.ref.source_end - self.ref.source_start):
            raise ValueError("Inspection source text/hash/span does not match its complete reference.")
        return self

    def presented(self):
        return {"source_id": self.source_id, "phase": self.phase,
                "chapter": self.ref.chapter_number, "scene": self.ref.scene_id,
                "speaker": self.speaker_id, "text": self.text}


class InspectionCatalog(Contract):
    schema_version: Literal[1] = 1
    sources: list[InspectionSource]
    catalog_hash: Digest

    @model_validator(mode="after")
    def fixed_mapping(self):
        ids = [source.source_id for source in self.sources]
        if ids != [f"src{index:03d}" for index in range(1, len(ids) + 1)]:
            raise ValueError("Inspection source IDs must be unique deterministic catalog positions.")
        refs = [digest(source.ref) for source in self.sources]
        if len(refs) != len(set(refs)):
            raise ValueError("One complete source reference cannot occur twice or in both source roles.")
        if self.catalog_hash != digest([source.model_dump(mode="json") for source in self.sources]):
            raise ValueError("Inspection catalog snapshot hash does not match its mapping.")
        return self


class ResolvedQuote(Contract):
    source_id: SourceId
    quote: str = Field(min_length=1, max_length=1200)
    ref: SourceRef
    quote_start: int = Field(ge=0, strict=True)
    quote_end: int = Field(ge=1, strict=True)
    quote_hash: Digest


def build_inspection_catalog(*, earlier, current) -> InspectionCatalog:
    """Preserve supplied chronological order; deduplicate only exact identities."""
    sources, seen = [], {}
    for phase, excerpts in (("earlier", earlier), ("current", current)):
        for excerpt in excerpts:
            excerpt = SourceExcerpt.model_validate(excerpt)
            key = digest(excerpt.ref)
            if key in seen:
                if seen[key] != (phase, excerpt):
                    raise ValueError("Conflicting source content or roles for one complete reference.")
                continue
            seen[key] = phase, excerpt
            sources.append(InspectionSource(source_id=f"src{len(sources) + 1:03d}", phase=phase,
                ref=excerpt.ref, text=excerpt.text, speaker_id=excerpt.speaker_id))
    return InspectionCatalog(sources=sources,
        catalog_hash=digest([source.model_dump(mode="json") for source in sources]))


def validate_catalog(catalog) -> InspectionCatalog:
    # Reparse mutable nested lists even when a previously validated model is supplied.
    value = catalog.model_dump(mode="json") if hasattr(catalog, "model_dump") else catalog
    return InspectionCatalog.model_validate(value)


def resolve_inspection_quote(catalog, *, source_id: str, quote: str,
                             phase: SourcePhase, allowed_source_ids) -> ResolvedQuote:
    catalog = validate_catalog(catalog)
    allowed = set(allowed_source_ids)
    sources = {source.source_id: source for source in catalog.sources}
    source = sources.get(source_id)
    if source is None or source_id not in allowed or source.phase != phase:
        raise ValueError(f"Invalid source_id {source_id!r} for {phase} evidence; "
                         f"select only from {sorted(allowed)} in this fixed snapshot.")
    if not quote or not quote.strip():
        raise ValueError("Evidence quote must contain original source text.")
    start = source.text.find(quote)
    if start < 0:
        raise ValueError(f"Quote must match exact original source {source_id}; do not paraphrase it.")
    if source.text.find(quote, start + 1) >= 0:
        raise ValueError(f"Quote is ambiguous within {source_id}; select a wider unique original span.")
    return ResolvedQuote(source_id=source_id, quote=quote, ref=source.ref,
        quote_start=source.ref.source_start + start,
        quote_end=source.ref.source_start + start + len(quote), quote_hash=text_hash(quote))
