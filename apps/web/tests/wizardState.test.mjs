import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

// Use the app's installed transpiler so tests support Node 20 as well as Node 22+.
// A data URL needs an absolute URL for its only runtime dependency; type imports
// disappear during transpilation. No temporary files or extra loader are needed.
const source = await readFile(new URL('../src/wizardState.ts', import.meta.url), 'utf8')
const { code } = await transformWithOxc(
  source.replace("from 'react'", `from ${JSON.stringify(import.meta.resolve('react'))}`),
  'wizardState.ts',
  { target: 'es2022' },
)
const {
  addCharacterDraft, approveDraft, approvePlanDraft, confirmWorldDraft, createInitialWizardDraft, goToWizardStep, wizardStepLocked, wizardSetupBusy,
  readWizardDraft, removeCharacterDraft, requestDraftRevision, toggleCharacterLock, updateCharacterDraft, updateWorldDraft,
} = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)

test('the combined brief advances through world confirmation directly to character review', () => {
  const initial = createInitialWizardDraft()
  const brief = updateCharacterDraft(updateWorldDraft(initial, { prompt: '港町の物語' }), 'character-1', { freeform: '船を修理する案内役' })
  const reviewing = goToWizardStep(brief, 'world-review')
  assert.equal(reviewing.worldConfirmed, false)
  assert.equal(goToWizardStep(reviewing, 'character-review').step, 'world-review')
  const confirmed = confirmWorldDraft(reviewing)
  assert.equal(confirmed.step, 'character-review')
  assert.equal(confirmed.characters[0].freeform, '船を修理する案内役')
  assert.equal(confirmed.world.prompt, '港町の物語')
})

test('legacy character-input storage migrates without losing world, characters or relationship instructions', () => {
  const draft = addCharacterDraft(createInitialWizardDraft(), 'character-2')
  draft.world = { ...draft.world, prompt: '世界の希望', setting: '確定した世界観' }
  draft.worldConfirmed = true
  draft.characters[0] = { ...draft.characters[0], freeform: '人物の希望', appearance: '立ち絵の希望', voice: '声の希望' }
  draft.relationshipInputs = [{ characterIds: ['character-1', 'character-2'], instruction: '長年の友人' }]
  const loaded = readWizardDraft({ version: 1, draft: { ...draft, step: 'character-input' } })
  assert.ok(loaded)
  assert.equal(loaded.step, 'world-input')
  assert.deepEqual(loaded.world, draft.world)
  assert.deepEqual(loaded.characters, draft.characters)
  assert.deepEqual(loaded.relationshipInputs, draft.relationshipInputs)
  assert.equal(loaded.worldConfirmed, true)
})

test('returning to either earlier screen preserves approval, inputs and revision', () => {
  const approved = { ...confirmWorldDraft(createInitialWizardDraft()), approved: true }
  const worldReview = goToWizardStep(approved, 'world-review')
  const input = goToWizardStep(worldReview, 'world-input')
  assert.equal(input.approved, true)
  assert.equal(input.worldConfirmed, true)
  assert.equal(input.revision, approved.revision)
  assert.deepEqual(input.world, approved.world)
  assert.deepEqual(input.characters, approved.characters)
})

test('an actual character brief edit requires world confirmation again, a result edit does not', () => {
  const approved = { ...confirmWorldDraft(createInitialWizardDraft()), approved: true }
  const input = goToWizardStep(approved, 'world-input')
  assert.equal(updateCharacterDraft(input, 'character-1', { freeform: '' }), input)
  const changed = updateCharacterDraft(input, 'character-1', { freeform: '希望を変更' })
  assert.equal(changed.worldConfirmed, false)
  assert.equal(changed.approved, false)
  const resultChanged = updateCharacterDraft(approved, 'character-1', { settings: '細部を調整' })
  assert.equal(resultChanged.worldConfirmed, true)
  assert.equal(resultChanged.approved, false)
})

test('world instruction changes invalidate approval while identical values leave it untouched', () => {
  const approved = { ...confirmWorldDraft(createInitialWizardDraft()), approved: true }
  assert.equal(updateWorldDraft(approved, { prompt: approved.world.prompt }), approved)
  const changed = updateWorldDraft(approved, { prompt: '舞台を変更' })
  assert.equal(changed.worldConfirmed, false)
  assert.equal(changed.approved, false)
  assert.equal(changed.step, 'world-input')
})

test('removing a character drops only their relationship instructions and retains the other pair', () => {
  let draft = addCharacterDraft(addCharacterDraft(createInitialWizardDraft(), 'character-2'), 'character-3')
  draft.relationshipInputs = [
    { characterIds: ['character-1', 'character-2'], instruction: '友人' },
    { characterIds: ['character-1', 'character-3'], instruction: 'ライバル' },
    { characterIds: ['character-2', 'character-3'], instruction: '同僚' },
  ]
  draft = removeCharacterDraft(confirmWorldDraft(draft), 'character-1')
  assert.deepEqual(draft.relationshipInputs, [{ characterIds: ['character-2', 'character-3'], instruction: '同僚' }])
  assert.equal(draft.worldConfirmed, false)
  const loaded = readWizardDraft({ version: 1, draft })
  assert.deepEqual(loaded.relationshipInputs, draft.relationshipInputs)
})

test('production requires both main-cast and planning approval and remains reachable after viewing earlier steps', () => {
  const initial = createInitialWizardDraft()
  assert.equal(approveDraft(initial), null)
  assert.equal(goToWizardStep(initial, 'production').step, 'world-review')
  const confirmed = confirmWorldDraft(initial)
  assert.equal(goToWizardStep(confirmed, 'production').step, 'character-review')
  const approved = approveDraft(confirmed)
  assert.equal(approved.step, 'planning-review')
  assert.equal(goToWizardStep(approved, 'production').step, 'planning-review')
  assert.equal(approvePlanDraft(initial), null)
  assert.equal(approvePlanDraft(confirmed), null)
  const planned = approvePlanDraft(approved)
  assert.equal(planned.step, 'production')
  const reviewing = goToWizardStep(planned, 'world-input')
  assert.equal(reviewing.approved, true)
  assert.equal(goToWizardStep(reviewing, 'production').step, 'production')
  assert.equal(reviewing.revision, approved.revision)
})

test('stored production and unknown steps recover safely without discarding the draft', () => {
  const draft = approvePlanDraft(approveDraft(confirmWorldDraft(updateWorldDraft(createInitialWizardDraft(), { prompt: '保存した世界の指示' }))))
  const load = patch => readWizardDraft({ version: 1, draft: { ...draft, ...patch } })
  assert.equal(load({}).step, 'production')
  assert.equal(load({ planApproved: undefined }).step, 'planning-review')
  assert.equal(load({ planApproved: undefined }).planApproved, false)
  assert.equal(load({ approved: false }).step, 'character-review')
  assert.equal(load({ worldConfirmed: false }).step, 'world-review')
  assert.equal(load({ worldConfirmed: false }).approved, false)
  const recovered = load({ step: 'obsolete-step' })
  assert.equal(recovered.step, 'world-input')
  assert.deepEqual(recovered.world, draft.world)
  assert.deepEqual(recovered.characters, draft.characters)
  assert.equal(load({ step: 'character-review' }).step, 'character-review')
})

test('changes requiring new approval invalidate both approvals and leave production at the relevant review step', () => {
  const approved = approvePlanDraft(approveDraft(confirmWorldDraft(addCharacterDraft(createInitialWizardDraft(), 'character-2'))))
  const cases = [
    [updateWorldDraft(approved, { prompt: '新しい舞台' }), 'world-input'],
    [updateCharacterDraft(approved, 'character-1', { settings: '人物の設定を更新' }), 'character-review'],
    [addCharacterDraft(approved, 'character-3'), 'world-review'],
    [removeCharacterDraft(approved, 'character-2'), 'world-review'],
    [toggleCharacterLock(approved, 'character-1', 'appearance'), 'character-review'],
    [requestDraftRevision(approved, 'appearance', '衣装を変更', 'revision-1', '2026-09-23T00:00:00Z', 'character-1'), 'character-review'],
    [requestDraftRevision(approved, 'world', '世界を変更', 'revision-2', '2026-09-23T00:00:00Z'), 'world-review'],
  ]
  for (const [changed, expectedStep] of cases) {
    assert.equal(changed.approved, false)
    assert.equal(changed.planApproved, false)
    assert.equal(changed.step, expectedStep)
    assert.notEqual(goToWizardStep(changed, 'production').step, 'production')
  }
  assert.equal(updateCharacterDraft(approved, 'character-1', { settings: '' }), approved)
  assert.equal(updateWorldDraft(approved, { prompt: '' }), approved)
})

test('live step gates require new approval for a first production and preserve access to existing editions', () => {
  const draft = { worldConfirmed: true, approved: true, planningRequired: true, planApproved: false, hasProduction: false }
  assert.equal(wizardStepLocked('planning-review', draft), false)
  assert.equal(wizardStepLocked('production', draft), true)
  assert.equal(wizardStepLocked('production', { ...draft, hasProduction: true }), false)
  assert.equal(wizardStepLocked('production', { ...draft, planApproved: true }), false)
  assert.equal(wizardStepLocked('production', { ...draft, approved: false, planningRequired: false, hasProduction: true }), false)
  assert.equal(wizardStepLocked('production', { ...draft, planningRequired: false, hasProduction: false }), false)
  assert.equal(wizardStepLocked('production', { ...draft, approved: false, planningRequired: false, hasProduction: false }), true)
  assert.equal(wizardStepLocked('planning-review', { ...draft, approved: false }), true)
})

test('obsolete planning and frozen production jobs do not block setup editing', () => {
  const stale = [{ kind: 'm3_plan', status: 'pending' }, { kind: 'm3_narrative', status: 'pending' }, { kind: 'm3_image', status: 'running' }]
  assert.equal(wizardSetupBusy(stale), false)
  assert.equal(wizardSetupBusy([...stale, { kind: 'm2_image', status: 'completed' }]), false)
  assert.equal(wizardSetupBusy([...stale, { kind: 'm2_character', status: 'pending' }]), true)
  assert.equal(wizardSetupBusy([{ kind: 'm2_voice', status: 'running' }]), true)
})
