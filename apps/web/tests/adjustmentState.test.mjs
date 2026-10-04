import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'
import { adjustmentFixture, adjustmentMusicFixture, adjustmentContinuityFixture } from './adjustmentFixture.mjs'

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

test('music assignment and volume are dirty draft edits isolated by chapter production and scene', () => {
  const source = adjustmentMusicFixture()
  const original = state.receiveAdjustment(state.initialAdjustmentEditor, source, 'adopt')
  const selected = state.selectAdjustmentMusicCandidate(original, source.scenes[0], source.music_candidates[0])
  assert.equal(state.adjustmentDirty(selected), true)
  assert.deepEqual(selected.scene_music[0], { production_id: 'chapter-1', scene_id: 'scene-1', candidate_id: 'music-1', action: 'play', volume: .35 })
  assert.deepEqual(selected.scene_music[1], original.scene_music[1])
  assert.equal(state.selectAdjustmentMusicCandidate(original, source.scenes[1], source.music_candidates[0]), original, 'same scene id in another chapter must not accept the candidate')
  const changed = state.changeAdjustmentMusic(selected, source.scenes[0], { volume: .12, production_id: 'wrong', scene_id: 'wrong' })
  assert.equal(changed.scene_music[0].volume, .12)
  assert.equal(changed.scene_music[0].production_id, 'chapter-1')
  assert.deepEqual(changed.characters, original.characters)
  assert.deepEqual(state.adjustmentPayload(changed).scene_music, changed.scene_music)
})

test('polling candidates preserves local BGM edits and conflicts on a remote revision change', () => {
  const source = adjustmentMusicFixture()
  let current = state.selectAdjustmentMusicCandidate(state.receiveAdjustment(state.initialAdjustmentEditor, source, 'adopt'), source.scenes[0], source.music_candidates[0])
  const updated = structuredClone(source)
  updated.music_candidates.push({ ...source.music_candidates[0], id: 'music-2' })
  current = state.receiveAdjustment(current, updated)
  assert.equal(current.scene_music[0].candidate_id, 'music-1')
  assert.equal(current.source.music_candidates.length, 2)
  assert.equal(current.conflict, false)
  updated.draft.revision = 2
  current = state.receiveAdjustment(current, updated)
  assert.equal(current.conflict, true)
  assert.equal(current.scene_music[0].candidate_id, 'music-1')
  current = state.receiveAdjustment(current, updated, 'adopt')
  assert.equal(current.conflict, false)
  assert.equal(current.scene_music[0].candidate_id, null)
  assert.equal(state.adjustmentDirty(current), false)
})

test('our music generation response advances revision without adopting a candidate or discarding edits', () => {
  const source = adjustmentMusicFixture()
  const current = state.changeAdjustmentMusic(state.receiveAdjustment(state.initialAdjustmentEditor, source, 'adopt'), source.scenes[1], { action: 'continue', volume: .4 })
  const response = structuredClone(source); response.draft.revision = 2
  const next = state.receiveAdjustment(current, response, 'candidate')
  assert.equal(next.base.revision, 2)
  assert.equal(next.scene_music[1].action, 'continue')
  assert.equal(next.scene_music[1].volume, .4)
  assert.equal(next.scene_music[0].candidate_id, null)
  assert.equal(state.adjustmentDirty(next), true)
  assert.equal(state.selectAdjustmentMusicCandidate(next, source.scenes[0], { ...source.music_candidates[0], artifact_id: null }), next)
})

test('music job progress remains separate from dialogue work', () => {
  const source = adjustmentMusicFixture()
  source.jobs = [{ id: 'music-1', kind: 'music', status: 'completed', attempt_count: 1 }, { id: 'music-2', kind: 'music', status: 'running', attempt_count: 1 }]
  assert.deepEqual(state.adjustmentJobProgress(source), [{ id: 'music', label: '場面のBGM候補', completed: 1, current: 1, total: 2, status: 'running' }])
})

test('normalized music jobs expose separate planning and generation progress', () => {
  const source = adjustmentMusicFixture()
  source.jobs = [
    { id: 'music-plan', kind: 'music', generation_kind: 'm3_music_plan', status: 'completed', attempt_count: 1 },
    { id: 'music-audio', kind: 'music', generation_kind: 'm3_music', status: 'running', attempt_count: 1 },
  ]
  assert.deepEqual(state.adjustmentJobProgress(source), [
    { id: 'm3_music_plan', label: 'BGM・場面転換の設計', completed: 1, current: 1, total: 1, status: 'completed' },
    { id: 'music', label: '場面のBGM候補', completed: 0, current: 0, total: 1, status: 'running' },
  ])
})

test('effective music ranges inherit the lead candidate and volume without mutating drafts or crossing chapters', () => {
  const source = adjustmentContinuityFixture()
  const editor = state.receiveAdjustment(state.initialAdjustmentEditor, source, 'adopt')
  const original = structuredClone(editor)
  const rows = state.adjustmentMusicContinuity(editor)
  assert.deepEqual(rows.slice(0, 3).map(row => row.candidate.id), ['music-1', 'music-1', 'music-1'])
  assert.equal(rows[0].range.end.scene_id, 'scene-3')
  assert.equal(rows[1].source_scene.scene_id, 'scene-1')
  assert.equal(rows[2].track_volume, .22)
  assert.equal(rows[3].candidate, undefined)
  assert.deepEqual(editor, original)
  const invalid = state.changeAdjustmentMusic(editor, source.scenes[5], { action: 'continue' })
  assert.match(state.adjustmentMusicContinuity(invalid)[5].error, /継続するBGMがありません/)
})

test('manual stop and play changes recompute the entire following continuation range immediately', () => {
  const source = adjustmentContinuityFixture()
  let editor = state.receiveAdjustment(state.initialAdjustmentEditor, source, 'adopt')
  editor = state.changeAdjustmentMusic(editor, source.scenes[1], { action: 'stop' })
  let rows = state.adjustmentMusicContinuity(editor)
  assert.equal(rows[0].range.end.scene_id, 'scene-1')
  assert.match(rows[2].error, /継続するBGMがありません/)
  editor = state.changeAdjustmentMusic(editor, source.scenes[1], { action: 'continue' })
  editor = state.changeAdjustmentMusic(editor, source.scenes[3], { action: 'play', candidate_id: 'music-4', volume: .5 })
  editor = state.changeAdjustmentMusic(editor, source.scenes[4], { action: 'continue' })
  rows = state.adjustmentMusicContinuity(editor)
  assert.equal(rows[3].range.end.scene_id, 'scene-5')
  assert.equal(rows[4].candidate.id, 'music-4')
  assert.equal(rows[4].track_volume, .5)
  assert.equal(rows[5].candidate, undefined)
})

test('automatic reset restores action, transition and reason while retaining the edited volume', () => {
  const source = adjustmentContinuityFixture()
  let editor = state.receiveAdjustment(state.initialAdjustmentEditor, source, 'adopt')
  editor = state.changeAdjustmentMusic(editor, source.scenes[1], { action: 'stop', volume: .13, reason: '手動でBGMなしに設定' })
  const reset = state.resetAutomaticSceneMusic(editor, source.scenes[1])
  assert.deepEqual(reset.scene_music[1], { ...source.draft.scene_music[1], volume: .13 })
  const missing = structuredClone(source.scenes[0]); missing.music_plan.candidate_id = 'unfinished'
  assert.equal(state.resetAutomaticSceneMusic(editor, missing), editor)
})

test('switching automatic continuation to another cue restores fades and preserves explicitly edited zero fades', () => {
  const source = adjustmentContinuityFixture()
  const editor = state.receiveAdjustment(state.initialAdjustmentEditor, source, 'adopt')
  const changed = state.changeAdjustmentMusic(editor, source.scenes[1], { action: 'play', candidate_id: 'new' })
  assert.deepEqual(changed.scene_music[1].transition, { ...source.draft.scene_music[1].transition, music_fade_out_ms: 1000, music_fade_in_ms: 1000 })
  const stopped = state.changeAdjustmentMusic(editor, source.scenes[1], { action: 'stop' })
  assert.equal(stopped.scene_music[1].transition.music_fade_out_ms, 1000)
  assert.equal(stopped.scene_music[1].transition.music_fade_in_ms, 0)
  const explicit = state.changeAdjustmentMusic(editor, source.scenes[1], { transition: { ...source.draft.scene_music[1].transition, visual: 'cut' }, reason: '画面遷移を手動調整' })
  const cut = state.changeAdjustmentMusic(explicit, source.scenes[1], { action: 'play' })
  assert.equal(cut.scene_music[1].transition.music_fade_out_ms, 0)
  assert.equal(cut.scene_music[1].transition.visual, 'cut')
})

test('chapter replans have independent progress without appearing as new audio generation', () => {
  const source = adjustmentContinuityFixture()
  source.jobs = [{ id: 'replan', kind: 'music', generation_kind: 'm3_music_plan', purpose: 'music_replan', status: 'running', attempt_count: 1 }]
  assert.deepEqual(state.adjustmentJobProgress(source), [{ id: 'music_replan', label: '章のBGM・場面転換の見直し', completed: 0, current: 0, total: 1, status: 'running' }])
})
