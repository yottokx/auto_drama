import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'
import { adjustmentFixture } from './adjustmentFixture.mjs'

const { code } = await transformWithOxc(await readFile(new URL('../src/adjustmentState.ts', import.meta.url), 'utf8'), 'adjustmentState.ts', { target: 'es2022' })
const state = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)
const start = () => state.receiveAdjustment(state.initialAdjustmentEditor, adjustmentFixture())

test('the scene supplies up to three unique known preview characters without changing the story cast', () => {
  const source = adjustmentFixture()
  assert.deepEqual(state.previewCast(source.scenes[0], source.cast.map(person => person.character_id)), { main: 'left', guide: 'right' })
  assert.deepEqual(state.previewCast(source.scenes[1], source.cast.map(person => person.character_id)), { later: 'center' })
  const scene = { ...source.scenes[0], character_ids: ['unknown', 'main', 'main', 'guide', 'later', 'extra'], positions: { main: 'right', guide: 'right' } }
  const before = structuredClone(scene)
  const selected = state.previewCast(scene, source.cast.map(person => person.character_id))
  assert.deepEqual(Object.keys(selected), ['main', 'guide', 'later'])
  assert.equal(new Set(Object.values(selected)).size, 3)
  assert.equal(selected.main, 'right')
  assert.deepEqual(scene, before)
})

test('preview selection and position swapping cannot add a fourth character or modify saved adjustments', () => {
  const saved = start()
  const before = structuredClone(saved)
  let selected = {}
  for (const id of ['main', 'later', 'guide', 'extra']) selected = state.togglePreviewCharacter(selected, id)
  assert.equal(Object.keys(selected).length, 3)
  assert.equal(selected.extra, undefined)
  const moved = state.movePreviewCharacter(selected, 'later', 'center')
  assert.equal(moved.later, 'center')
  assert.equal(moved.main, selected.later)
  assert.equal(Object.keys(state.togglePreviewCharacter(moved, 'later')).length, 2)
  assert.deepEqual(saved, before)
})

test('choosing a new image resets bounds and geometry while a voice selection preserves them and all settings', () => {
  const original = start()
  const edited = state.changeAdjustment(original, 'guide', { body_bounds: { left: .2, top: .1, right: .8, bottom: .9 }, offset_y: 38, scale: 1.25 })
  const image = { id: 'new-image', character_id: 'guide', kind: 'image', artifact_id: 'new', url: '/new.png' }
  const changed = state.selectAdjustmentCandidate(edited, image)
  assert.deepEqual(changed.characters.find(person => person.character_id === 'guide'), { ...original.characters[1], image_candidate_id: 'new-image' })
  const voice = state.selectAdjustmentCandidate(edited, { ...image, kind: 'voice', id: 'new-voice', url: '/new.wav' })
  assert.deepEqual(voice.characters[1], { ...edited.characters[1], voice_candidate_id: 'new-voice' })
  assert.deepEqual(voice.source.cast, original.source.cast, 'appearance, voice settings and accumulated instructions are read-only')
  assert.equal(state.selectAdjustmentCandidate(edited, { ...image, artifact_id: null, url: null }), edited)
})

test('job and candidate polling preserves a dirty draft; another revision conflicts until explicitly adopted', () => {
  let current = state.changeAdjustment(start(), 'main', { offset_y: -42 })
  const progress = adjustmentFixture()
  progress.jobs = [{ id: 'candidate-job', status: 'running' }]
  current = state.receiveAdjustment(current, progress)
  assert.equal(current.characters[0].offset_y, -42)
  assert.equal(current.source.jobs[0].status, 'running')
  assert.equal(current.conflict, false)
  const remote = structuredClone(progress)
  remote.draft.revision = 2
  remote.draft.characters[0].scale = 1.5
  current = state.receiveAdjustment(current, remote)
  assert.equal(current.conflict, true)
  assert.equal(current.base.revision, 1)
  assert.equal(current.characters[0].offset_y, -42)
  const reload = state.receiveAdjustment(current, remote, 'adopt')
  assert.equal(reload.conflict, false)
  assert.equal(reload.characters[0].scale, 1.5)
  assert.equal(state.adjustmentDirty(reload), false)
})

test('our candidate-generation response updates the revision without discarding unsaved layout or selecting its result', () => {
  const current = state.changeAdjustment(start(), 'main', { scale: 1.35 })
  const response = adjustmentFixture()
  response.draft.revision = 2
  response.candidates.push({ id: 'new-image', character_id: 'main', kind: 'image', artifact_id: 'new', url: '/new.png' })
  const next = state.receiveAdjustment(current, response, 'candidate')
  assert.equal(next.base.revision, 2)
  assert.equal(next.characters[0].scale, 1.35)
  assert.equal(next.characters[0].image_candidate_id, 'main-image')
  assert.equal(next.conflict, false)
  assert.equal(state.adjustmentDirty(next), true)
  assert.deepEqual(Object.keys(state.adjustmentPayload(next)).sort(), ['characters', 'expected_revision'])
})

test('upload checks reject empty, oversized and unsupported files without assuming that extension validates contents', () => {
  const image = { name: 'portrait.PNG', type: 'image/png', size: 1024 }
  assert.equal(state.adjustmentUploadError(image, 'image', 33554432), null)
  assert.equal(state.adjustmentUploadError({ ...image, name: 'voice.mp3', type: 'audio/mpeg' }, 'voice', 33554432), null)
  assert.match(state.adjustmentUploadError({ ...image, size: 0 }, 'image', 33554432), /空/)
  assert.match(state.adjustmentUploadError({ ...image, size: 33554433 }, 'image', 33554432), /32MiB/)
  assert.match(state.adjustmentUploadError({ ...image, name: 'page.html' }, 'image', 33554432), /PNG/)
  assert.match(state.adjustmentUploadError(image, 'voice', 33554432), /WAV/)
})

test('adjustment audio progress groups all lines and retains completed counts between claims', () => {
  const source = adjustmentFixture()
  source.jobs = Array.from({ length: 45 }, (_, index) => ({ id: `line-${index}`, kind: 'm3_voice_clone', status: index < 40 ? 'completed' : 'pending', attempt_count: index < 40 ? 1 : 0 }))
  assert.deepEqual(state.adjustmentJobProgress(source), [{ id: 'm3_voice_clone', label: '台詞・試聴の音声', completed: 40, current: 40, total: 45, status: 'running' }])
  source.jobs[40].status = 'failed'
  assert.equal(state.adjustmentJobProgress(source)[0].status, 'failed')
})
