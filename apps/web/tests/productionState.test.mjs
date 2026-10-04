import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const source = await readFile(new URL('../src/productionState.ts', import.meta.url), 'utf8')
const { code } = await transformWithOxc(source, 'productionState.ts', { target: 'es2022' })
const { coordinatorNeedsRestart, productionChapters, productionView, chapterStatusNames, chapterAssetProgress, chapterGenerationProgress, llmProgressItems, productionCurrentProgress, progressItemLabel } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)

const chapter = (number, status, extra = {}) => ({
  chapter_number: number, production_id: `production-${number}`, status,
  narrative_artifact_id: status === 'waiting' || status === 'writing' ? null : `narrative-${number}`,
  jobs: [], error: null,
  build: status === 'published' ? { id: `build-${number}`, chapter_number: number, status: 'published' } : null,
  player_url: status === 'published' ? `/player/build-${number}` : null,
  export_url: status === 'published' ? `/export/build-${number}` : null,
  ...extra,
})
const production = extra => ({
  id: 'production-1', approval_id: 'approval-1', chapter_number: 1, chapter_count: 3,
  status: 'running', stage: 'assets', control_state: 'running', jobs: [],
  completed_jobs: 1, total_jobs: 2, error: null, narrative_artifact_id: 'narrative-1',
  build: { id: 'build-1', chapter_number: 1, status: 'published' },
  player_url: '/player/build-1', export_url: '/export/build-1',
  chapters: [chapter(1, 'published'), chapter(2, 'generating_assets'), chapter(3, 'writing')],
  ...extra,
})

test('a published first chapter is viewable while later chapters generate', () => {
  const view = productionView(production(), 3)
  assert.equal(view.complete, false)
  assert.equal(view.publishedCount, 1)
  assert.equal(view.chapters[0].player_url, '/player/build-1')
  assert.equal(view.chapters[0].export_url, '/export/build-1')
  assert.equal(view.canStop, true)
  assert.equal(view.canResume, false)
  assert.match(view.title, /完成した1章から鑑賞/)
  assert.equal(chapterStatusNames[view.chapters[1].status], '本文完成・素材を制作中')
  assert.equal(chapterStatusNames[view.chapters[2].status], '本文を制作中')
})

test('graceful stopping prevents another stop or resume and still allows interruption', () => {
  const view = productionView(production({ control_state: 'stopping' }), 3)
  assert.equal(view.canStop, false)
  assert.equal(view.canInterrupt, true)
  assert.equal(view.canResume, false)
  assert.equal(view.canRetry, false)
  assert.match(view.title, /保存して/)
  assert.equal(view.chapters[0].player_url, '/player/build-1')
})

for (const control_state of ['paused', 'interrupted']) {
  test(`${control_state} preserves all published links and offers resume instead of job retry`, () => {
    const current = production({ control_state, status: 'failed' })
    const before = structuredClone(current)
    const view = productionView(current, 3)
    assert.equal(view.canStop, false)
    assert.equal(view.canInterrupt, false)
    assert.equal(view.canRetry, false)
    assert.equal(view.canResume, true)
    assert.equal(view.chapters[0].player_url, '/player/build-1')
    assert.equal(view.chapters[0].export_url, '/export/build-1')
    assert.deepEqual(current, before)
  })
}

test('completed productions have chapter links and no generation controls', () => {
  const view = productionView(production({ chapters: [1, 2, 3].map(number => chapter(number, 'published')), status: 'published' }), 3)
  assert.equal(view.complete, true)
  assert.equal(view.publishedCount, 3)
  assert.equal(view.canStop, false)
  assert.equal(view.canInterrupt, false)
  assert.equal(view.canResume, false)
  assert.equal(view.canRetry, false)
  assert.deepEqual(view.chapters.map(item => item.player_url), ['/player/build-1', '/player/build-2', '/player/build-3'])
})

test('restored frozen work keeps published links without starting or retrying jobs', () => {
  const view = productionView(production({ history_frozen: true, control_state: 'paused' }), 3)
  assert.equal(view.historyFrozen, true)
  assert.equal(view.canStop, false)
  assert.equal(view.canResume, false)
  assert.equal(view.canRetry, false)
  assert.equal(view.chapters[0].player_url, '/player/build-1')
})

test('old one-chapter productions remain viewable and explicitly resume the remaining chapters', () => {
  const old = production({ chapters: undefined, chapter_count: undefined, status: 'published', control_state: 'paused' })
  const view = productionView(old, 3, coordinatorNeedsRestart({ status: 'ok', stage: 'm4', schema_version: 4 }))
  assert.equal(view.complete, false)
  assert.equal(view.canResume, true)
  assert.equal(view.chapters[0].build.id, 'build-1')
  assert.equal(view.chapters[0].player_url, '/player/build-1')
  assert.deepEqual(view.chapters.map(item => item.status), ['published', 'waiting', 'waiting'])
})

test('an outdated live server reports restart instead of offering unavailable production controls', () => {
  const old = production({ chapters: undefined, chapter_count: undefined, status: 'published', control_state: undefined })
  const needsRestart = coordinatorNeedsRestart({ status: 'ok', stage: 'm3', schema_version: 3 })
  assert.equal(needsRestart, true)
  const view = productionView(old, 3, needsRestart)
  assert.match(view.title, /制御サーバーの再起動/)
  assert.equal(view.canStop, false)
  assert.equal(view.canInterrupt, false)
  assert.equal(view.canResume, false)
  assert.equal(view.canRetry, false)
  assert.equal(view.chapters[0].player_url, '/player/build-1')
  assert.equal(view.chapters[0].export_url, '/export/build-1')
  assert.equal(productionView({ ...old, control_state: 'paused' }, 3, needsRestart).canResume, false)
})

test('server compatibility comes from health, including unstarted projects and later schema versions', () => {
  assert.equal(coordinatorNeedsRestart({ status: 'ok', stage: 'm3' }), true)
  assert.equal(coordinatorNeedsRestart({ status: 'ok', stage: 'm4', schema_version: 4 }), false)
  assert.equal(coordinatorNeedsRestart({ status: 'ok', stage: 'm5', schema_version: 5 }), false)
  assert.match(productionView(null, 3, true).title, /制御サーバーの再起動/)
})

test('missing later chapter records are shown in order without inventing a published build', () => {
  const chapters = productionChapters(production({ chapters: [chapter(2, 'writing'), chapter(1, 'published')] }), 3)
  assert.deepEqual(chapters.map(item => item.chapter_number), [1, 2, 3])
  assert.equal(chapters[2].status, 'waiting')
  assert.equal(chapters[2].build, null)
  assert.equal(chapters[2].player_url, null)
})

test('production chapter count is retained when the user edits draft settings', () => {
  const view = productionView(production(), 7)
  assert.equal(view.chapters.length, 3)
})

test('a failure in a later chapter allows retry while the published first chapter stays intact', () => {
  const view = productionView(production({ status: 'failed', chapters: [chapter(1, 'published'), chapter(2, 'failed')] }), 3)
  assert.equal(view.canRetry, true)
  assert.equal(view.chapters[0].build.id, 'build-1')
  assert.match(view.title, /再試行/)
})

test('an unstarted project does not display chapters or stop controls', () => {
  const view = productionView(null, 3)
  assert.deepEqual(view.chapters, [])
  assert.equal(view.complete, false)
  assert.equal(view.canResume, false)
  assert.equal(view.canStop, false)
})

const job = (id, kind, status, extra = {}) => ({ id, kind, status, attempt_count: 1, ...extra })
const requirement = (kind, target_id, artifact_id = null, job_id = null, descriptor = {}) => ({ kind, target_id, artifact_id, job_id, descriptor: JSON.stringify(descriptor) })
const supporting = { character_result: { name: '案内役' } }

test('music planning and scene tracks participate in progress and block building until all music is ready', () => {
  const current = chapter(1, 'generating_assets', {
    jobs: [job('music-plan', 'm3_music_plan', 'completed'), job('music-two', 'm3_music', 'running')],
    requirements: [
      requirement('m3_music_plan', 'chapter-1', 'music-prompts', 'music-plan'),
      requirement('m3_music', 'scene-1', 'music-one'),
      requirement('m3_music', 'scene-2', null, 'music-two'),
    ],
  })
  const groups = chapterAssetProgress(current)
  assert.deepEqual(groups.map(item => item.id), ['m3_image', 'm3_background', 'm3_voice', 'm3_voice_clone', 'm3_music_plan', 'm3_music'])
  assert.deepEqual(groups.slice(-2).map(item => [item.current, item.total, item.status]), [[1, 1, 'completed'], [1, 2, 'running']])
  assert.equal(chapterGenerationProgress(current).at(-1).status, 'pending')
  assert.match(productionCurrentProgress([current]), /場面のBGM（1\/2）/)
  current.requirements[2].artifact_id = 'music-two'
  current.jobs[1].status = 'completed'
  assert.equal(chapterGenerationProgress(current).at(-1).status, 'running')
})

test('enabled music stages are present from chapter start and design directly follows validation without a job counter', () => {
  const current = chapter(1, 'writing', { music_enabled: true, jobs: [job('narrative', 'm3_narrative', 'running')] })
  const saved = structuredClone(current)
  const rows = chapterGenerationProgress(current)
  assert.deepEqual(rows.map(item => item.id), ['narrative', 'scene-creation', 'chapter-validation', 'm3_music_plan', 'm3_background', 'm3_voice_clone', 'm3_music', 'build'])
  const design = rows.find(item => item.id === 'm3_music_plan'), tracks = rows.find(item => item.id === 'm3_music')
  assert.equal(design.status, 'pending')
  assert.equal(design.total, 1, 'the internal fixed total still prevents premature building')
  assert.equal(progressItemLabel(design), 'BGM・場面転換の設計')
  assert.equal(tracks.status, 'pending')
  assert.equal(tracks.total, null)
  assert.equal(progressItemLabel(tracks), '場面のBGM')
  assert.equal(rows.at(-1).status, 'pending')
  assert.deepEqual(current, saved)
})

test('track count stays unknown while the music plan is queued, running, failed or completed without adoption', () => {
  for (const status of ['pending', 'running', 'failed', 'completed']) {
    const current = chapter(1, 'generating_assets', { music_enabled: true,
      jobs: [job('music-plan', 'm3_music_plan', status, { attempt_count: status === 'pending' ? 0 : 1 })],
      requirements: [requirement('m3_music_plan', 'chapter-music', null, 'music-plan'), requirement('m3_background', 'port', 'background')],
    })
    const rows = chapterGenerationProgress(current)
    const tracks = rows.find(item => item.id === 'm3_music'), design = rows.find(item => item.id === 'm3_music_plan')
    assert.equal(tracks.total, null)
    assert.equal(tracks.status, 'pending')
    assert.equal(progressItemLabel(tracks), '場面のBGM')
    assert.equal(progressItemLabel(design), 'BGM・場面転換の設計')
    assert.notEqual(design.status, 'completed', 'job completion cannot replace plan artifact adoption')
    assert.equal(rows.at(-1).status, 'pending')
  }
})

test('an adopted silence-only plan fixes zero tracks and lets building proceed without artificial generation', () => {
  const current = chapter(1, 'generating_assets', { music_enabled: true,
    jobs: [job('music-plan', 'm3_music_plan', 'completed')],
    requirements: [requirement('m3_music_plan', 'chapter-music', 'music-plan', 'music-plan')],
  })
  const rows = chapterGenerationProgress(current)
  const design = rows.find(item => item.id === 'm3_music_plan'), tracks = rows.find(item => item.id === 'm3_music')
  assert.equal(design.status, 'completed')
  assert.equal(progressItemLabel(design), 'BGM・場面転換の設計')
  assert.deepEqual([tracks.status, tracks.total, tracks.completed], ['skipped', 0, 0])
  assert.equal(progressItemLabel(tracks), '場面のBGM（0曲）')
  assert.equal(rows.at(-1).status, 'running')
})

test('music design precedes later narrative memory steps and media while active counts appear only for accepted tracks', () => {
  const current = chapter(2, 'generating_assets', { music_enabled: true, jobs: [
    job('narrative', 'm3_narrative', 'completed', { progress: llmProgress({ active: false, steps: [
      { id: 'scene-plan', stage: 'chapter_scene_plan', status: 'completed', chapter_number: 2 },
      ...['script', 'speech_extraction', 'staging'].map(stage => ({ id: stage, stage, status: 'completed', chapter_number: 2, scene_number: 1 })),
      { id: 'validate', stage: 'validation', status: 'completed', chapter_number: 2 },
      { id: 'memory', stage: 'memory', status: 'completed', chapter_number: 2 },
    ] }) }), job('plan', 'm3_music_plan', 'completed'), job('track', 'm3_music', 'running'),
  ], requirements: [requirement('m3_music_plan', 'chapter-music', 'plan-artifact', 'plan'), requirement('m3_music', 'scene-1', null, 'track')] })
  const rows = chapterGenerationProgress(current), validation = rows.findIndex(item => item.id === 'chapter-validation')
  assert.deepEqual(rows.slice(validation, validation + 3).map(item => item.id), ['chapter-validation', 'm3_music_plan', 'memory'])
  assert.equal(rows.filter(item => item.id === 'm3_music_plan').length, 1)
  assert.equal(progressItemLabel(rows.find(item => item.id === 'm3_music')), '場面のBGM（0/1）')
  assert.equal(productionCurrentProgress([current]), '第2章・場面のBGM（0/1）を制作中')
})

test('disabled and pre-music legacy chapters retain their original phases while old music jobs remain visible', () => {
  for (const music_enabled of [undefined, false]) {
    const current = chapter(1, 'writing', { music_enabled })
    assert.equal(chapterGenerationProgress(current).some(item => item.id.startsWith('m3_music')), false)
    assert.equal(chapterAssetProgress(current).length, 4)
  }
  const legacy = chapter(1, 'generating_assets', { jobs: [job('plan', 'm3_music_plan', 'completed'), job('track', 'm3_music', 'running')] })
  assert.equal(progressItemLabel(chapterGenerationProgress(legacy).find(item => item.id === 'm3_music_plan')), 'BGM・場面転換の設計')
  assert.equal(progressItemLabel(chapterGenerationProgress(legacy).find(item => item.id === 'm3_music')), '場面のBGM（0/1）')
  const inherited = productionChapters(production({ music_enabled: true }), 3)
  assert.ok(inherited.every(item => item.music_enabled === true))
  const overridden = productionChapters(production({ music_enabled: true, chapters: [chapter(1, 'writing', { music_enabled: false })] }), 1)
  assert.equal(overridden[0].music_enabled, false)
})

test('media progress groups required assets in model order and includes reused and unqueued work without main cast', () => {
  const current = chapter(2, 'generating_assets', {
    jobs: [job('image-live', 'm3_image', 'running'), job('voice-failed', 'm3_voice', 'failed')],
    requirements: [
      requirement('m3_image', 'main', 'approved-main-image'),
      requirement('m3_voice', 'main', 'approved-main-voice', null, { reference_text: 'メインの自己紹介' }),
      requirement('m3_image', 'support-1', 'reused-image', null, supporting),
      requirement('m3_image', 'support-2', null, 'image-live', supporting),
      requirement('m3_image', 'support-3', null, null, supporting),
      requirement('m3_background', 'harbor'), requirement('m3_background', 'station'),
      requirement('m3_voice', 'support-1', null, 'voice-failed', supporting),
      ...Array.from({ length: 40 }, (_, index) => requirement('m3_voice_clone', `line-${index}`, index < 3 ? `audio-${index}` : null)),
    ],
  })
  const before = structuredClone(current)
  const groups = chapterAssetProgress(current)
  assert.deepEqual(groups.map(item => item.id), ['m3_image', 'm3_background', 'm3_voice', 'm3_voice_clone'])
  assert.deepEqual(groups.map(item => [item.current, item.total, item.completed, item.status]), [[1, 3, 1, 'running'], [0, 2, 0, 'pending'], [0, 1, 0, 'failed'], [3, 40, 3, 'pending']])
  assert.equal(progressItemLabel(groups[0]), 'サブキャラの立ち絵（1/3）')
  assert.equal(progressItemLabel(groups[3]), '台詞の音声（3/40）')
  assert.deepEqual(current, before)
})

test('adopted material stays complete despite an old failed job and empty categories are explicitly skipped', () => {
  const current = chapter(1, 'generating_assets', {
    jobs: [job('old-image', 'm3_image', 'failed')],
    requirements: [requirement('m3_image', 'support', 'recovered-image', 'old-image', supporting), requirement('m3_voice', 'main', 'main-voice', null, { reference_text: '自己紹介' })],
  })
  const groups = chapterAssetProgress(current)
  assert.deepEqual(groups.map(item => [item.current, item.total, item.status]), [[1, 1, 'completed'], [0, 0, 'skipped'], [0, 0, 'skipped'], [0, 0, 'skipped']])
  assert.equal(chapterGenerationProgress(current).at(-1).status, 'running')
  assert.equal(chapterGenerationProgress({ ...current, error: '組み立て失敗' }).at(-1).status, 'failed')
})

test('asset counts stay unknown before narrative and fall back to jobs for older snapshots', () => {
  const writing = chapter(1, 'writing')
  assert.ok(chapterAssetProgress(writing).every(item => item.total === null && item.status === 'pending'))
  assert.deepEqual(chapterGenerationProgress(chapter(2, 'waiting')), [])
  const legacy = chapter(1, 'generating_assets', { jobs: [job('a', 'm3_image', 'completed'), job('b', 'm3_image', 'running'), job('c', 'm3_dialogue', 'completed')] })
  const groups = chapterAssetProgress(legacy)
  assert.deepEqual(groups.map(item => [item.current, item.total]), [[1, 2], [0, 0], [0, 0], [1, 1]])
  assert.deepEqual(chapterAssetProgress({ ...legacy, requirements: [] }), groups, 'older frozen snapshots may expose missing requirements as an empty array')
})

test('dialogue phase stays active between sequential claims and counts only adopted completions', () => {
  const snapshot = (completed, active = null, terminalFailure = false) => chapter(1, 'generating_assets', {
    jobs: Array.from({ length: 45 }, (_, index) => job(`voice-${index}`, 'm3_voice_clone',
      index < completed ? 'completed' : index === active ? terminalFailure ? 'failed' : 'running' : 'pending',
      { attempt_count: index < completed || index === active ? 1 : 0 })),
    requirements: Array.from({ length: 45 }, (_, index) => requirement('m3_voice_clone', `line-${index}`, index < completed ? `audio-${index}` : null, `voice-${index}`)),
  })
  const snapshots = [snapshot(39, 39), snapshot(40), snapshot(40, 40)]
  assert.deepEqual(snapshots.map(current => {
    const row = chapterAssetProgress(current).find(item => item.id === 'm3_voice_clone')
    return [row.status, row.current, row.completed, progressItemLabel(row)]
  }), [
    ['running', 39, 39, '台詞の音声（39/45）'],
    ['running', 40, 40, '台詞の音声（40/45）'],
    ['running', 40, 40, '台詞の音声（40/45）'],
  ])
  const failed = chapterAssetProgress(snapshot(40, 40, true)).find(item => item.id === 'm3_voice_clone')
  assert.equal(failed.status, 'failed', 'an unfinished failed job overrides evidence that the phase started')
  const finished = chapterAssetProgress(snapshot(45)).find(item => item.id === 'm3_voice_clone')
  assert.equal(finished.status, 'completed')
  assert.equal(progressItemLabel(finished), '台詞の音声（45/45）')
})

test('reused assets and unclaimed jobs do not activate a phase but a retried pending job does', () => {
  const reused = requirement('m3_voice_clone', 'reused-line', 'reused-audio')
  const pending = requirement('m3_voice_clone', 'new-line', null, 'voice')
  for (const [attempt_count, expected] of [[0, 'pending'], [1, 'running']]) {
    const current = chapter(2, 'generating_assets', {
      requirements: [reused, pending],
      jobs: [job('voice', 'm3_voice_clone', 'pending', { attempt_count })],
    })
    const row = chapterAssetProgress(current).find(item => item.id === 'm3_voice_clone')
    assert.equal(row.status, expected)
    assert.equal(row.current, 1)
    assert.equal(row.completed, 1)
  }
  const reusedOnly = chapter(2, 'generating_assets', { requirements: [reused, requirement('m3_voice_clone', 'new-line')] })
  assert.equal(chapterAssetProgress(reusedOnly).find(item => item.id === 'm3_voice_clone').status, 'pending')
})

test('chapter progress shows scene and validation placeholders from the first job state without changing generic LLM progress', () => {
  for (const status of ['pending', 'running']) {
    const narrative = job('narrative', 'm3_narrative', status, { attempt_count: status === 'pending' ? 0 : 1 })
    const rows = chapterGenerationProgress(chapter(1, 'writing', { jobs: [narrative] }))
    assert.deepEqual(rows.slice(0, 3).map(row => [row.label, row.status]), [
      ['章とシーンの構成', status], ['シーンの作成', 'pending'], ['内容・整合性の確認', 'pending'],
    ])
    assert.ok(rows.slice(1, 3).every(row => row.current === 0 && row.completed === 0))
    assert.equal(llmProgressItems(narrative).length, 1, 'placeholders belong only to the chapter view')
    assert.equal(llmProgressItems(narrative)[0].label, 'シーン・台本の生成')
  }
  const narrative = job('narrative', 'm3_narrative', 'running', { progress: llmProgress({
    current_step: 'chapter-plan', steps: [{ id: 'chapter-plan', stage: 'chapter_scene_plan', status: 'running', chapter_number: 1 }],
  }) })
  const rows = chapterGenerationProgress(chapter(1, 'writing', { jobs: [narrative] }))
  assert.deepEqual(rows.slice(0, 3).map(row => row.label), ['章とシーンの構成', 'シーンの作成', '内容・整合性の確認'])
  assert.equal(rows.filter(row => row.label === 'シーンの作成').length, 1)
  assert.equal(rows.filter(row => row.label === '内容・整合性の確認').length, 1)
})

test('known scenes replace their placeholder and reported validation replaces its waiting row in place', () => {
  const progress = llmProgress()
  const narrative = job('narrative', 'm3_narrative', 'running', { progress })
  const initial = chapterGenerationProgress(chapter(2, 'writing', { jobs: [narrative] }))
  const validationIndex = initial.findIndex(row => row.label === '内容・整合性の確認')
  assert.deepEqual(initial.slice(0, validationIndex).map(row => row.label), ['章とシーンの構成', 'シーン1', 'シーン2', 'シーン3', 'シーン4', 'シーン5'])
  assert.equal(initial[validationIndex].status, 'pending')
  assert.equal(initial.some(row => row.label === 'シーンの作成'), false)
  const current = chapter(2, 'writing', { jobs: [{ ...narrative, progress: { ...progress, current_step: 'validation', steps: [
    ...progress.steps.map(step => ({ ...step, status: 'completed' })),
    { id: 'validation', stage: 'validation', status: 'running', chapter_number: 2 },
    { id: 'memory', stage: 'memory', status: 'completed', chapter_number: 2 },
  ] } }] })
  const actual = chapterGenerationProgress(current)
  assert.equal(actual.filter(row => row.label === '内容・整合性の確認').length, 1)
  assert.equal(actual[validationIndex].id, initial[validationIndex].id)
  assert.ok(actual[validationIndex].stepIds.includes('validation'))
  assert.equal(actual[validationIndex].status, 'running')
  assert.equal(actual[validationIndex + 1].label, '次章へ引き継ぐ情報の整理')
})

test('completed legacy narratives do not invent unfinished scene or validation placeholders', () => {
  for (const jobs of [[], [job('legacy-narrative', 'm3_narrative', 'completed')]]) {
    const rows = chapterGenerationProgress(chapter(1, 'published', { jobs }))
    assert.equal(rows[0].status, 'completed')
    assert.equal(rows.some(row => row.label === 'シーンの作成'), false)
    assert.equal(rows.some(row => row.label === '内容・整合性の確認' && row.status === 'pending'), false)
  }
})

const llmProgress = extra => ({ schema_version: 1, sequence: 6, phase: 'chapter', current_step: 'script-2', attempt: 1, active: true, updated_at: '2026-10-02T00:00:00Z', steps: [
  { id: 'chapter-plan', stage: 'chapter_scene_plan', status: 'completed', chapter_number: 2 },
  { id: 'script-1', stage: 'script', status: 'completed', chapter_number: 2, scene_number: 1, completed: 1, total: 5 },
  { id: 'speech-1', stage: 'speech_extraction', status: 'completed', chapter_number: 2, scene_number: 1, completed: 1, total: 5 },
  { id: 'staging-1', stage: 'staging', status: 'completed', chapter_number: 2, scene_number: 1, completed: 1, total: 5 },
  { id: 'script-2', stage: 'script', status: 'running', chapter_number: 2, scene_number: 2, completed: 1, total: 5 },
], ...extra })

test('completed and current scenes each occupy one row while chapter planning and media remain separate', () => {
  const narrative = job('narrative', 'm3_narrative', 'running', { progress: llmProgress() })
  const before = structuredClone(narrative)
  const current = chapter(2, 'writing', { jobs: [narrative] })
  const items = llmProgressItems(narrative)
  assert.deepEqual(items.map(item => item.label), ['章とシーンの構成', 'シーン1', 'シーン2', 'シーン3', 'シーン4', 'シーン5'])
  assert.deepEqual(items.slice(1).map(({ id, current, total, completed, status, counter }) => ({ id, current, total, completed, status, counter })), [
    { id: 'scene-2-1', current: 3, total: 3, completed: 3, status: 'completed', counter: 'step' },
    { id: 'scene-2-2', current: 1, total: 3, completed: 0, status: 'running', counter: 'step' },
    ...[3, 4, 5].map(number => ({ id: `scene-2-${number}`, current: 0, total: 3, completed: 0, status: 'pending', counter: 'step' })),
  ])
  assert.equal(progressItemLabel(items[1]), 'シーン1 STEP 3/3')
  assert.equal(progressItemLabel(items[2]), 'シーン2 STEP 1/3')
  assert.equal(items[1].statusText, undefined)
  assert.equal(items[2].statusText, '台本を生成中…')
  assert.equal(progressItemLabel(items[3]), 'シーン3 STEP 0/3', 'old progress can recover planned scenes from reported scene totals after plan completion')
  assert.equal(productionCurrentProgress([chapter(1, 'published'), current]), '第2章・シーン2 STEP 1/3 台本を生成中…')
  assert.equal(chapterGenerationProgress(current).filter(item => item.id === 'm3_image').length, 0, 'unknown supporting cast must not add an empty material row')
  assert.deepEqual(narrative, before)
})

test('an accepted chapter plan displays every scene before the first script starts', () => {
  const narrative = job('narrative', 'm3_narrative', 'running', { progress: llmProgress({
    chapter_plan: { chapter_number: 2, scene_count: 3, supporting_character_count: 0 }, current_step: null,
    steps: [{ id: 'chapter-plan', stage: 'chapter_scene_plan', status: 'completed', chapter_number: 2 }],
  }) })
  const before = structuredClone(narrative)
  const items = llmProgressItems(narrative)
  assert.equal(items[0].id, 'chapter-plan')
  assert.deepEqual(items.slice(1).map(({ id, current, total, completed, status, counter, statusText }) => ({ id, current, total, completed, status, counter, statusText })),
    [1, 2, 3].map(number => ({ id: `scene-2-${number}`, current: 0, total: 3, completed: 0, status: 'pending', counter: 'step', statusText: undefined })))
  assert.deepEqual(items.slice(1).map(progressItemLabel), ['シーン1 STEP 0/3', 'シーン2 STEP 0/3', 'シーン3 STEP 0/3'])
  assert.deepEqual(narrative, before)
})

test('a running chapter plan and STEP4 progress do not invent waiting scenes', () => {
  const narrative = job('narrative', 'm3_narrative', 'running', { progress: llmProgress({ current_step: 'chapter-plan', steps: [
    { id: 'chapter-plan', stage: 'chapter_scene_plan', status: 'running', chapter_number: 2 },
  ] }) })
  assert.deepEqual(llmProgressItems(narrative).map(item => item.id), ['chapter-plan'])
  const planning = job('planning', 'm3_plan', 'running', { progress: llmProgress({ phase: 'planning',
    chapter_plan: { chapter_number: 2, scene_count: 3, supporting_character_count: 1 },
    current_step: 'plot', steps: [{ id: 'plot', stage: 'plot_plan', status: 'running', completed: 1, total: 3 }],
  }) })
  assert.deepEqual(llmProgressItems(planning).map(progressItemLabel), ['物語の核心・全体プロット（1/3）'])
})

test('reported scene steps replace waiting rows under the same ids and remain in scene order', () => {
  const plan = { id: 'chapter-plan', stage: 'chapter_scene_plan', status: 'completed', chapter_number: 2 }
  const metadata = { chapter_number: 2, scene_count: 4, supporting_character_count: 1 }
  const before = llmProgressItems(job('narrative', 'm3_narrative', 'running', { progress: llmProgress({ chapter_plan: metadata, current_step: null, steps: [plan] }) }))
  const steps = [plan,
    { id: 'script-3', stage: 'script', status: 'running', scene_number: 3, completed: 2, total: 4 },
    ...['script', 'speech_extraction', 'staging'].map(stage => ({ id: `scene-1-${stage}`, stage, status: 'completed', chapter_number: 2, scene_number: 1, completed: 1, total: 4 })),
    { id: 'memory', stage: 'memory', status: 'completed', chapter_number: 2 },
  ]
  const after = llmProgressItems(job('narrative', 'm3_narrative', 'running', { progress: llmProgress({ chapter_plan: metadata, current_step: 'script-3', steps }) }))
  assert.deepEqual(after.map(item => item.id), [...before.map(item => item.id), 'memory'])
  assert.equal(new Set(after.map(item => item.id)).size, after.length)
  assert.deepEqual(after.slice(1, 5).map(item => [item.current, item.completed, item.status]), [[3, 3, 'completed'], [0, 0, 'pending'], [1, 0, 'running'], [0, 0, 'pending']])
  assert.equal(after[3].statusText, '台本を生成中…')
})

test('supporting material rows stay hidden before assets unless the accepted plan identifies supporting cast', () => {
  for (const supportingCount of [undefined, 0, 2]) {
    const narrative = job('narrative', 'm3_narrative', 'running', { progress: llmProgress({
      ...(supportingCount === undefined ? {} : { chapter_plan: { chapter_number: 2, scene_count: 3, supporting_character_count: supportingCount } }),
      current_step: null, steps: [{ id: 'chapter-plan', stage: 'chapter_scene_plan', status: 'completed', chapter_number: 2 }],
    }) })
    const current = chapter(2, 'writing', { jobs: [narrative] })
    assert.equal(chapterAssetProgress(current).length, 4, 'the raw asset summary retains all categories for readiness checks')
    const rows = chapterGenerationProgress(current)
    assert.deepEqual(rows.filter(item => item.id === 'm3_image' || item.id === 'm3_voice').map(item => [item.id, item.total, item.status]),
      supportingCount > 0 ? [['m3_image', null, 'pending'], ['m3_voice', null, 'pending']] : [])
    assert.deepEqual(rows.filter(item => item.id === 'm3_background' || item.id === 'm3_voice_clone').map(item => [item.id, item.total]),
      [['m3_background', null], ['m3_voice_clone', null]])
  }
})

test('main-only and historic requirements suppress empty supporting rows while actual legacy and reused supporting assets remain visible', () => {
  const mainOnly = chapter(1, 'published', { requirements: [
    requirement('m3_image', 'main', 'approved-image'),
    requirement('m3_voice', 'main', 'approved-voice', null, { reference_text: 'メインキャラです' }),
  ] })
  const supportingRows = current => chapterGenerationProgress(current).filter(item => item.id === 'm3_image' || item.id === 'm3_voice')
  assert.deepEqual(supportingRows(mainOnly), [])
  assert.deepEqual(supportingRows({ ...mainOnly, jobs: [job('narrative', 'm3_narrative', 'completed', { progress: llmProgress({
    chapter_plan: { chapter_number: 1, scene_count: 3, supporting_character_count: 2 },
  }) })] }), [], 'known empty material requirements take precedence over a prior cast count')
  assert.equal(chapterGenerationProgress(mainOnly).at(-1).status, 'completed')
  assert.deepEqual(supportingRows(chapter(1, 'generating_assets', { jobs: [job('old-image', 'm3_image', 'running'), job('old-voice', 'm3_voice', 'completed')] })).map(item => [item.id, item.total, item.status]),
    [['m3_image', 1, 'running'], ['m3_voice', 1, 'completed']], 'older jobs remain sufficient evidence without plan metadata')
  assert.deepEqual(supportingRows(chapter(2, 'generating_assets', { requirements: [
    requirement('m3_image', 'support', 'reused-image', null, supporting),
    requirement('m3_voice', 'support', 'reused-voice', null, supporting),
  ] })).map(item => [item.id, item.current, item.total, item.status]),
    [['m3_image', 1, 1, 'completed'], ['m3_voice', 1, 1, 'completed']], 'reused assets with no jobs still show completed supporting groups')
})

test('old attempt and paused progress cannot masquerade as a currently running LLM step', () => {
  for (const extra of [{ status: 'pending', progress: llmProgress({ active: false }) }, { attempt_count: 2, progress: llmProgress() }]) {
    const items = llmProgressItems(job('narrative', 'm3_narrative', 'running', extra))
    assert.equal(items[0].status, 'completed')
    assert.equal(items[2].status, 'pending')
    assert.equal(items[2].current, 1)
    assert.equal(items[2].statusText, undefined)
    assert.equal(items[2].completed, 0)
  }
  const failed = llmProgressItems(job('narrative', 'm3_narrative', 'failed', { progress: llmProgress({ active: false }) }))
  assert.equal(failed[2].status, 'failed')
  assert.equal(failed[0].status, 'completed')
  assert.equal(failed[2].statusText, undefined)
  const fallback = llmProgressItems(job('old', 'm3_narrative', 'running'))
  assert.equal(fallback.length, 1)
  assert.equal(fallback[0].label, 'シーン・台本の生成')
})

const sceneStages = ['script', 'speech_extraction', 'staging']
test('an active scene stays in progress between reported STEP completions and the next start', () => {
  const step = (stage, status) => ({ id: stage, stage, status, chapter_number: 1, scene_number: 1 })
  const snapshots = [
    [step('script', 'running')],
    [step('script', 'completed')],
    [step('script', 'completed'), step('speech_extraction', 'running')],
    [step('script', 'completed'), step('speech_extraction', 'completed')],
  ]
  const rows = snapshots.map(steps => llmProgressItems(job('narrative', 'm3_narrative', 'running', { progress: llmProgress({ steps, current_step: steps.at(-1).id }) }))[0])
  assert.deepEqual(rows.map(row => [row.status, row.current, row.completed, row.statusText]), [
    ['running', 1, 0, '台本を生成中…'],
    ['running', 2, 1, '次の工程を準備中…'],
    ['running', 2, 1, '台詞と話者を整理中…'],
    ['running', 3, 2, '次の工程を準備中…'],
  ])
  for (const extra of [{ status: 'pending' }, { attempt_count: 2 }, { progress: llmProgress({ active: false, steps: snapshots[1], current_step: 'script' }) }]) {
    const [row] = llmProgressItems(job('narrative', 'm3_narrative', 'running', { progress: llmProgress({ steps: snapshots[1], current_step: 'script' }), ...extra }))
    assert.equal(row.status, 'pending')
    assert.equal(row.current, 2)
    assert.equal(row.completed, 1)
    assert.equal(row.statusText, undefined)
  }
})

for (const [index, stage] of sceneStages.entries()) {
  test(`scene progress identifies active ${stage} as step ${index + 1} of three`, () => {
    const steps = sceneStages.slice(0, index + 1).map((value, i) => ({
      id: `scene-4-${value}`, stage: value, status: i === index ? 'running' : 'completed',
      chapter_number: 1, scene_number: 4, completed: i === index ? 3 : 4, total: 12,
    }))
    const narrative = job('narrative', 'm3_narrative', 'running', { progress: llmProgress({ current_step: steps.at(-1).id, steps }) })
    const rows = llmProgressItems(narrative)
    assert.equal(rows.length, 1)
    assert.equal(rows[0].id, 'scene-1-4')
    assert.equal(rows[0].current, index + 1)
    assert.equal(rows[0].total, 3, 'the displayed total counts stages, not chapter scenes')
    assert.equal(rows[0].completed, index)
    assert.equal(rows[0].status, 'running')
    assert.equal(rows[0].statusText, ['台本を生成中…', '台詞と話者を整理中…', '演出を設定中…'][index])
  })
}

test('a scene cannot become complete from a later stage or a completed job if required reports are missing', () => {
  for (const reported of [
    [{ id: 'only-script', stage: 'script', status: 'completed' }],
    [{ id: 'only-staging', stage: 'staging', status: 'completed' }],
    [{ id: 'script', stage: 'script', status: 'completed' }, { id: 'staging', stage: 'staging', status: 'completed' }],
  ]) {
    const steps = reported.map(row => ({ ...row, scene_number: 1 }))
    const rows = llmProgressItems(job('narrative', 'm3_narrative', 'completed', { progress: llmProgress({ active: false, current_step: null, steps }) }))
    assert.equal(rows.length, 1)
    assert.equal(rows[0].id, 'scene-0-1')
    assert.equal(rows[0].completed, reported.length)
    assert.notEqual(rows[0].status, 'completed')
    assert.notEqual(rows[0].status, 'running')
    assert.equal(rows[0].statusText, undefined)
  }
})

test('a failed second scene step retains completed script work and does not hide the failure', () => {
  const steps = [
    { id: 'script', stage: 'script', status: 'completed', chapter_number: 1, scene_number: 2 },
    { id: 'speech', stage: 'speech_extraction', status: 'failed', chapter_number: 1, scene_number: 2 },
  ]
  const rows = llmProgressItems(job('narrative', 'm3_narrative', 'failed', { progress: llmProgress({ active: false, current_step: 'speech', steps }) }))
  assert.equal(rows.length, 1)
  assert.equal(rows[0].current, 2)
  assert.equal(rows[0].completed, 1)
  assert.equal(rows[0].status, 'failed')
  assert.equal(rows[0].statusText, undefined)
  assert.equal(progressItemLabel(rows[0]), 'シーン2 STEP 2/3')
})

test('repeated reports of one stage cannot stand in for three distinct completed scene steps', () => {
  const steps = [1, 2, 3].map(number => ({ id: `script-try-${number}`, stage: 'script', status: 'completed', scene_number: 1 }))
  const rows = llmProgressItems(job('narrative', 'm3_narrative', 'running', { progress: llmProgress({ current_step: null, steps }) }))
  assert.equal(rows.length, 1)
  assert.equal(rows[0].completed, 1)
  assert.notEqual(rows[0].status, 'completed')
})

test('a completed job does not substitute for a missing scene-step completion report', () => {
  const steps = sceneStages.map((stage, index) => ({ id: stage, stage, status: index === 2 ? 'running' : 'completed', scene_number: 1 }))
  const rows = llmProgressItems(job('narrative', 'm3_narrative', 'completed', { progress: llmProgress({ active: false, current_step: 'staging', steps }) }))
  assert.equal(rows.length, 1)
  assert.equal(rows[0].completed, 2)
  assert.notEqual(rows[0].status, 'completed')
  assert.notEqual(rows[0].status, 'running')
  assert.equal(rows[0].statusText, undefined)
})

test('planning stages and scene-independent chapter work keep their existing labels and counters', () => {
  const planning = job('plan', 'm3_plan', 'running', { progress: llmProgress({ phase: 'planning', current_step: 'plot', steps: [
    { id: 'cast', stage: 'cast_plan', status: 'completed' },
    { id: 'plot', stage: 'plot_plan', status: 'running', completed: 2, total: 5 },
  ] }) })
  const rows = llmProgressItems(planning)
  assert.deepEqual(rows.map(progressItemLabel), ['サブキャラの設定・関係性', '物語の核心・全体プロット（2/5）'])
  assert.ok(rows.every(row => row.counter === undefined && row.statusText === undefined))
  const narrative = job('narrative', 'm3_narrative', 'running', { progress: llmProgress({ current_step: 'memory', steps: [
    { id: 'script', stage: 'script', status: 'completed', scene_number: 1 },
    { id: 'validation', stage: 'validation', status: 'completed', scene_number: 1 },
    { id: 'memory', stage: 'memory', status: 'running' },
  ] }) })
  const chapterRows = llmProgressItems(narrative)
  assert.deepEqual(chapterRows.map(row => row.label), ['シーン1', 'シーン1・内容・整合性の確認', '次章へ引き継ぐ情報の整理'])
  assert.equal(chapterRows[0].completed, 1, 'validation must not count as one of the three scene steps')
  assert.equal(chapterRows[2].counter, undefined)
})
