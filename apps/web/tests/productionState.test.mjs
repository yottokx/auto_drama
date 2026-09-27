import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const source = await readFile(new URL('../src/productionState.ts', import.meta.url), 'utf8')
const { code } = await transformWithOxc(source, 'productionState.ts', { target: 'es2022' })
const { coordinatorNeedsRestart, productionChapters, productionView, chapterStatusNames } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)

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
