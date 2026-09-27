"""Shared generator identity for provenance and experimental replay boundaries."""

CAUSAL_PROTOCOL = {"workflow": 2, "prompt": 8, "implementation": 15, "contract_revision": 3}
CHAPTER_EDITOR_PROTOCOL = {"workflow": 2, "prompt": 9, "implementation": 16,
                           "contract_revision": 4, "policy": "chapter_editor_v1"}
STORY_DRAFT_PROTOCOL = {"workflow": 2, "prompt": 1, "implementation": 1,
                        "contract_revision": 1, "policy": "story_draft_v1"}
SCRIPT_CONTINUATION_PROTOCOL = {"workflow": 2, "prompt": 14, "implementation": 15,
                                "contract_revision": 13, "policy": "script_continuation_v1"}
LEGACY_PROTOCOL = {"workflow": 1, "prompt": 6, "implementation": 1, "contract_revision": 1}


def generator_protocol(workflow: str, policy: str | None = None) -> dict:
    """Return an independent value; unrelated source edits do not invalidate runs."""
    if workflow not in {"causal", "legacy"}:
        raise ValueError("Unknown story generator workflow.")
    if policy == "chapter_editor_v1" and workflow == "causal":
        return dict(CHAPTER_EDITOR_PROTOCOL)
    if policy == "story_draft_v1" and workflow == "causal":
        return dict(STORY_DRAFT_PROTOCOL)
    if policy == "script_continuation_v1" and workflow == "causal":
        return dict(SCRIPT_CONTINUATION_PROTOCOL)
    if policy is not None:
        raise ValueError("Unknown or unsupported story generator policy.")
    return dict(CAUSAL_PROTOCOL if workflow == "causal" else LEGACY_PROTOCOL)
