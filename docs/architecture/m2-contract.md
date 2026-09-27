# M2 implementation contract

The coordinator owns persistence and adoption. The worker runs local Gemma, Anima/rembg and Irodori in isolated subprocesses. Inputs and generated results remain separate. Approval records a durable snapshot and starts M3 production using that frozen snapshot.

The [cast extension contract](m2-cast-contract.md) adds the three-character limit, independent pairwise relationships, self-introductions, sample lines and reference-voice cloning. It takes precedence for these additions and legacy-result compatibility.

## UI API

- `POST /api/m2/projects` body `{world: WorldBrief, characters?: CharacterBrief[]}` creates a saved draft (no generation).
- `GET /api/m2/projects` returns `{projects: [{id,title,...}]}`.
- `GET /api/m2/projects/{id}` and successful mutations return `{project: {...}, draft: M2Draft, jobs: Job[]}`.
- `POST /api/m2/projects/{id}/actions` accepts `{expected_revision, action, ...}`; optimistic conflicts are HTTP 409. Actions listed below. Mutations during a pending/running generation are rejected, except viewing. One operation per project at a time; unrelated projects can retain queued work.
- `POST /api/jobs/{id}/retry` keeps the M1 contract. UI refreshes after retry.
- Existing `GET /api/artifacts/{id}` exposes the immutable artifact's recorded provenance. The character review's initially collapsed 「生成の詳細」 loads this endpoint for the displayed `imageArtifactId`; it shows the actual prompt, negative prompt, seed and recorded model/size/steps/CFG. It does not substitute current model settings for historical values. Missing legacy fields are labelled as unrecorded, while an explicitly empty prompt is shown as unspecified. Switching images resets the panel and aborts obsolete requests; viewing details remains available while another character is generating.
- Existing `/api/artifacts/{id}/content` serves adopted JSON/PNG/WAV.

`WorldBrief` and `CharacterBrief` use the existing frontend camelCase fields in wizardState.ts. `M2Draft`:

```
{
  revision: number, step: WizardStep, worldInput: WorldBrief,
  worldResult: WorldBrief | null, worldConfirmed: boolean, worldPendingChanges: boolean,
  worldArtifactId: string|null, worldVersion: number|null, worldConfirmationArtifactId: string|null,
  characters: [{ id: string, input: CharacterBrief, result: CharacterBrief | null,
    locked: {settings:boolean,appearance:boolean,voice:boolean},
    resultArtifactId: string|null, resultVersion: number|null, pendingChanges: boolean,
    imagePendingChanges: boolean, voicePendingChanges: boolean,
    imageArtifactId: string|null, voiceArtifactId: string|null,
    imageUrl: string|null, voiceUrl: string|null }],
  approved: boolean, approval: object|null, requests: RevisionRequest[],
  activeJobId: string|null, remainingJobCount: number
}
```

Actions:

- `save-brief` with `world`, `characters` and `relationshipInputs`: atomically saves the combined initial instructions. Changes invalidate world confirmation and downstream adoption; retained results/assets remain available. Identical input is a no-op. The three visible steps are `world-input`, `world-review`, `character-review`; legacy `character-input` opens `world-input` without discarding data.
- `save-world` with `world`: save input, invalidate world confirmation and final approval; preserve existing result for review.
- `generate-world` with optional `instruction` (empty = initial/re-generate, nonempty = revise existing result); creates `m2_world` job.
- `edit-world` with `world`: save directly edited result; original worldInput remains unchanged, invalidate confirmations.
- `confirm-world`: requires complete worldResult, records confirmation and starts the character/media/relationship queue, advancing to character-review. Repeating an unchanged confirmation does not enqueue duplicate generation.
- `save-characters` with `characters: CharacterBrief[]`: replace optional briefs, preserve results/assets for retained IDs, at least one character; respects locks.
- `generate-characters`: requires world confirmation; generates each character and its missing/unlocked image and voice via durable jobs. The coordinator queues the next job on adoption.
- `revise-character` with `character_id, scope: all|settings|appearance|voice, instruction`: creates `m2_character`; coordinator enforces locked fields even if worker returns changes; rebuild affected unlocked assets, preserve unaffected/fixed artifacts.
- `edit-character` with `character_id, patch` (result fields only): changes result; invalidates dependent unlocked assets if appearance/voice changes; rejects locked changes.
- `toggle-lock` with `character_id, scope: settings|appearance|voice`: toggle protection; protected assets are preserved.
- `retake` with `character_id, scope: image-retake|voice-retake, instruction`: queue only the matching asset job; preserve all text and other asset references.
- `approve`: requires confirmed world, complete character results and image/voice assets, no pending/failed generation or unapplied edits; persist immutable approval snapshot with versions, asset IDs and timestamp, then start M3 from that snapshot.
- `go-to` with `step`: navigate while preserving server draft, prevent character review before world confirmation. The combined initial input is always accessible when navigation is allowed.

All successful settings changes increment revision and invalidate final approval. Jobs carry `base_revision` and input snapshots; obsolete results cannot overwrite newer drafts. Confirmation and approval snapshots are immutable artifacts. Retry preserves the job input and seed; definitively rejected character or image-prompt generation can derive a new LLM retry seed. Intentional new generation gets a new job/seed.

Unapplied character inputs require whole-character generation/revision before direct edits, partial revisions or retakes. Pending flags remain until that result is adopted. Old media remains visible during a retake, but cannot be approved until its pending flag is cleared. Only the current failed operation blocks approval; historical failures remain visible without making a later successful result unusable. Lock changes also create a new character result version, keeping approval JSON identical to its referenced result artifact.

## Worker protocol

M1 claim/heartbeat/fail API is reused; worker advertises configured supported M2 job kinds.

`job.kind`: `m2_world`, `m2_character`, `m2_image`, `m2_voice`.
`job.payload`:
```
{
  schema_version: 1, seed: integer, base_revision: integer,
  world_input: WorldBrief, world_result: WorldBrief|null,
  character_id: string|null, character_input: CharacterBrief|null,
  character_result: CharacterBrief|null,
  scope: world|all|settings|appearance|voice|image-retake|voice-retake,
  instruction: string, locked: {settings,appearance,voice},
  profile: {provider:'local', model_id:string, temperature:number,
    max_tokens:number, reasoning_level:string, prompt_version:1}
}
```

`POST /api/m2/jobs/{id}/complete?worker_id=...&lease_id=...` accepts a ZIP (max 32 MiB) containing `result.json` and, for media, `image.png` or `voice.wav`. No executable or arbitrary paths. `result.json`:
```
{schema_version:1, kind:job.kind, result: WorldBrief|CharacterBrief|{},
 provenance:{...non-secret model/seed/runtime information...},
 trace:[...hierarchical candidate decisions and random tool calls...]}
```
Image jobs output a transparent PNG; voice jobs output a nonempty PCM WAV. The complete ZIP itself is the immutable job result artifact for idempotent retransmission; contained result and media receive separate versioned artifact records and are atomically adopted. Lease validation applies before and after storage. Text results must be complete; truncated/invalid output fails, never becomes a partial generated result.

### Character identity and explicit revisions

Character generation uses `character_id` to identify the sole target; its input and existing result must carry that ID. The target's full settings are separated from summaries of other characters, whose complete results are excluded as output templates. Relationship generation continues to receive the full cast's established settings. New-character output constrains `id` to the requested value with a schema enum, and the worker rejects a mismatched raw result instead of overwriting its ID. The coordinator independently rejects a submitted result whose ID differs from the job target.

An explicit instruction with an existing character result uses a dedicated revision request returning `{id, changes}` rather than selecting a new person through candidate generation. Only unlocked fields within the requested scope are allowed in `changes`; omitted fields retain their existing values. `scope=all` permits changes across all unlocked groups but does not request rewriting every field. For name-only instructions, the model is instructed to change the name and necessary name references while preserving the person's role, background, appearance and voice. The worker merges the patch into the existing result and validates the complete character before adoption. Existing result and media versions remain in artifact history.

Image prompt translation uses a dedicated single-character context: the target's completed ID, name, age, gender and appearance, plus the current retake instruction. The completed ID must match `character_id`. Other cast members, relationships, story text and old character input must not enter this translation request. The image trace and provenance retain the target ID and source appearance alongside the actual English prompt, configured quality prefix and negative prompt. Character/world writing still uses its own broader context; image translation does not reuse it.

### Image prompt conversion v5

Structured conversion returns exactly these string fields: `subject`, `body`, `skin`, `hair`, `eyes`, `clothing`, `accessories`, `other_features`, and `rendering`. The first eight fields describe visible facts; unspecified or inapplicable parts remain empty, including for nonhuman characters. Explicit absence, such as having no arms or legs or holding nothing, is retained as a visible constraint rather than treated as unspecified. Visible expressions, how objects are held and where accessories are worn are also preserved. `rendering` carries only the current retake's composition, pose or drawing adjustments. There is no minimum word count. Concise English tags/phrases describe attributes, while sentences can express relationships, conditions and how objects are held. Colors, material and transparency remain attached to the relevant body part or object. Quoted text and inscriptions on clothing or accessories are also translated into English; necessary Japanese proper names are romanized. Translation must preserve the object, its color and location rather than remove the feature to avoid Japanese text.

The conversion instructions distinguish literary comparisons from literal anatomy: 「透き通るような白い肌」 becomes fair skin, while explicitly translucent slime, blue skin or a ceramic body retains those properties. The converter must not invent human anatomy, species, skin color, ethnicity, clothing or props for unspecified features, or normalize an explicitly sickly or ghostly character into healthy human skin. There is no global skin replacement or blacklist of fantasy features. Schema validation rejects missing/extra keys and incorrect types; text checks reject an empty set of visual facts, non-English Japanese/CJK text and excessive output (over 1,200 characters per field or 5,000 in total). These structural checks do not certify semantic or image fidelity, which still requires generation comparisons and review.

A rejected structured conversion gets one correction request containing the validation error and, when available, the rejected JSON. If conversion still fails validation, the worker persists a retry checkpoint so the next job retry resamples that conversion with a derived LLM seed; the original job input and seed remain unchanged. Valid conversions remain cached after image-generation or transport failures. This checkpoint is local to image-prompt conversion and does not invalidate a completed image ZIP.

The worker joins nonempty visual fields in the fixed order above, adds the code-owned portrait composition (solo, entire body in frame, neutral pose, anime color illustration, bright frontal lighting and white background), then the optional rendering adjustment. The neutral-pose wording avoids imposing a humanoid standing posture on characters without limbs. The configured quality prefix is prepended immediately before inference; the negative prompt is passed separately. Image runtime output must report those exact effective prompts. The trace and adopted image's `provenance.image` additionally preserve `prompt_version: 5`, `visual_features` (the first eight fields), `rendering_instruction` and `composition_prompt`. Existing image artifacts are not rewritten. Quality prefix and negative prompt remain configurable through `config/m2-generation.json`, with per-model settings UI planned for the later model-management work.

## Ownership for parallel implementation

- Backend: contracts, migration, m2 service and routes, coordinator integration, backend tests.
- Generation: services/worker/generation/*, config/m2-generation.json, isolated media runners, unit tests. Expose `generate_job(job: dict, work_dir: Path) -> bytes` returning the ZIP above; do not edit WorkerClient.
- Frontend: API hook and WizardApp/WorldSteps/CharacterSteps integration, live state/errors/progress/library, preserve accepted result-first UI and explicit demo route.
- Root: WorkerClient dispatch/launcher, cross-layer integration, end-to-end actual-model verification, documentation.
