# M2 cast extension contract

The existing wizard and immutable artifacts remain compatible. The new limit is three main characters (both API and UI). Do not delete pre-existing drafts/artifacts or manufacture introductions for legacy results.

## Character text

`CharacterBrief` gains optional/defaulted `selfIntroduction: string = ''` and `sampleLines: string[] = []`. New generated results require a nonempty self-introduction (roughly 60–120 Japanese characters, actual first-person spoken dialogue), and three representative spoken lines (not explanations, each reasonably short). These are rendered as prose/quotes in the result-first review.

Both fields belong to the settings scope. `selfIntroduction` also depends on the voice lock: preserving a fixed reference recording preserves its self-introduction. Reject direct changes to selfIntroduction while voice is locked; whole/partial generation merges preserve it when either settings or voice is locked. Changes to selfIntroduction invalidate/rebuild the unlocked reference voice even if the voice description is unchanged. Representative lines alone do not invalidate voice.

New jobs carry `character_contract_version: 2`. Preserve legacy completed results verbatim for viewing and approval artifact integrity. Old results need not acquire fabricated speech fields on read. Support old results in direct edits/lock toggles and old queued jobs; new full generation upgrades their text. UI displays absent legacy speech as not generated and offers the ordinary regeneration workflow. New character generation requires the extended complete result.

Individual settings describe only the character's own identity, personality, background, motivations and voice. Do not embed relationships to the protagonist/other named main characters. Role can still be 'protagonist'. All inter-character relationship facts go in the independent relationship result.

## Relationships

Draft fields:

```
relationshipInputs: [{ characterIds: [string,string], instruction: string }]
relationships: {
  result: { pairs: [{ characterIds:[string,string], summary:string,
                     firstToSecond:string, secondToFirst:string }] } | null,
  artifactId: string|null, version:number|null, pendingChanges:boolean
}
```

Each pair uses distinct valid main-character IDs sorted lexicographically, with exactly one pair per unordered combination. Three characters have three pairs; two have one; one needs none. Input instructions are optional. The API normalizes pair order or rejects invalid pairs/duplicates/self-pairs. The UI labels the two names explicitly, including directions in the result.

The final LLM response uses required object keys `pair_1`, `pair_2`, and (for three characters) `pair_3`, with only the summary and two directional descriptions. The worker assigns each key to a canonical pair of real character IDs and explicitly names its first/second character in the prompt. Opaque IDs are added by the worker when building the public `pairs` artifact; the LLM does not copy or sort them. Worker validation can normalize a reversed pair only by swapping both IDs and the directional descriptions, and still rejects missing, duplicate, unknown or self pairs. Changing the final prompt/schema changes its request-cache fingerprint, so pre-fix failed responses are not reused.

`save-characters` additionally accepts optional `relationshipInputs` for atomic input saving. Omission preserves existing matching pairs. Add `save-relationships` with `relationshipInputs` and `generate-relationships` with optional `instruction` (for relation-only revision). Input changes invalidate the relationship result's current status and final approval, retaining old result for review. Results have their own versioned artifact.

After `generate-characters`, enqueue `m2_relationships` after all characters complete, whenever count >= 2, even without relationship instructions. Character identity/settings changes invalidate and automatically enqueue relation generation after applicable character/media jobs; voice/appearance-only changes and media retakes do not. Direct settings edits may leave relationship regeneration pending and expose a clear action to regenerate. Membership and world changes invalidate relations. Approval of multiple characters requires the exact full mesh, current complete character results and a nonpending relationship artifact; persist the relation artifact/version/hash in approval.

Generation payload additions for character/relationship jobs:

```
cast_inputs: CharacterBrief[]
cast_results: CharacterResult[]  // completed available results
relationship_inputs: [{characterIds:[id,id],instruction}]
relationships_result: {pairs:[...]}|null
```

`m2_relationships` result envelope has `result: {pairs:[...]}`, no media file, character_id null, scope `relationships`. The coordinator independently validates exact pair coverage, directions, IDs and nonempty fields. The LLM sees all personal settings, world, optional pair instructions, and existing relation result for revisions. Character generation can use pair input constraints for consistent identity without writing the relations in personal settings.

## Reference speech and voice-clone trials

Generated, visible voice descriptions include exactly one of `男声話者` / `女声話者` and describe perceived voice age, for example `若い女声話者。柔らかく明瞭な声で、自然な速さで話す。`. Age wording such as `幼い`, `若い`, and `年老いた` describes how the voice sounds; it is not a schema enum. These are synthesis characteristics, independent of the character's gender identity, species, or chronological age. Nonbinary, genderless, and nonhuman characters still receive a voice-design choice. A thousand-year-old goddess can have a young voice; chronological age must not automatically become perceived voice age. Preserve any explicit user preference and existing speech limitations.

M2 character generation/revision and M3 supporting-character generation share this instruction for the saved `voice` text that users review. The reference-voice path passes this visible description directly to TTS, retaining the existing explicit retake-instruction handling. It adds no hidden caption conversion, normalization, age inference, or LLM call. Existing saved character descriptions, successful job caches, and adopted recordings remain intact; the improved descriptions apply when voice settings are generated or revised upstream. Name-only revisions and locked voice settings preserve the existing description.

`m2_voice` uses `character_result.selfIntroduction` as its actual text (new results), not the old global generic greeting. Legacy retakes can use the configured old reference text until a full character regeneration; report the actual spoken text. Store `provenance.voice.reference_text` with the reference artifact. The API exposes `voiceReferenceText` on each character entry, obtained from that artifact's provenance.

A failed non-clone generation blocks new clone requests with HTTP 409 so its active job and remaining queue stay retryable. A failed clone can be replaced by a new trial. Legacy drafts containing more than three main characters must reduce the cast before any operation that generates relationships.

Add action `clone-voice` with `character_id, text` (trimmed nonempty, max 1000 characters; runtime validates checkpoint token limit). It creates `m2_voice_clone`, taking the currently adopted reference WAV and that artifact's original spoken text. The API checks reference belongs to the character/project and is current/complete, and includes:

```
reference_voice: {artifact_id:string, sha256:string, text:string}
dialogue_text: string
```

Root WorkerClient downloads only `/api/artifacts/<id>/content` from its own coordinator, checks SHA256 and PCM format, writes atomic `reference-voice.wav` in the job directory. The original payload/fingerprint is unchanged; generation reads that known file. No arbitrary URL or path from payload is used.

The pinned Irodori API uses `ref_wav`/`no_ref:false` for speaker cloning and has no reference transcript parameter. Clones retain an empty design caption and use the adopted recording's voice; do not inject a new speaker or perceived-age description. Store and validate the original transcript in provenance; do not invent an unsupported API argument. Preserve watermarking and PCM WAV validation. Bundle has `result.json` (`result:{}`) and `voice.wav`, using kind `m2_voice_clone`.

Each character entry gains `voiceTests: [{id:string,text:string,artifactId:string,url:string,sourceVoiceArtifactId:string,createdAt:string}]` in chronological order. Trial outputs never replace the reference voice or change approval settings. The UI retains a trial's text, audio, and the fact it used an older reference if the voice has since changed. Clone creation/failure/success does not invalidate an existing approval. Current in-flight work can still block other mutations via the existing lease scheme. Keep records durable without silently deleting history.

## Review interaction

During generation, keep character selection, image expansion, audio playback and reading enabled. Disable conflicting edits/new generation actions only. A user can switch to any character whose result is ready while other character jobs continue. Do not reset the selected character on polls.

Portrait opens in a labelled modal with a large uncropped image, close control, Escape and focus return. Existing small preview stays in place. Voice trials, self-introduction, sample lines and relationship results follow settings/results; input/edit controls remain deliberate or collapsed.

## Ownership

- Backend agent: Python contracts/schema/coordinator + integration tests.
- Generation agent: generation pipeline/schemas/runners/config + unit tests.
- Frontend agent: apps/web/src + build.
- Root: WorkerClient dispatch/reference retrieval, worker tests, docs, actual-model and browser checks.
