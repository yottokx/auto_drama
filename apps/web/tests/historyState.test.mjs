import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const source = await readFile(new URL('../src/historyState.ts', import.meta.url), 'utf8')
const { code } = await transformWithOxc(source, 'historyState.ts', { target: 'es2022' })
const { acceptHistory, restoreBlockReason } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)

const earlier = { id: 'saved-1', number: 1, label: 'キャラクター設定を保存', created_at: '2026-09-24T01:00:00Z', restored_from_id: null, current: false }
const latest = { id: 'saved-2', number: 2, label: '立ち絵の表示を調整', created_at: '2026-09-24T01:10:00Z', restored_from_id: null, current: true }
const history = { project_id: 'story-1', version: 7, current_revision_id: latest.id, busy: false, entries: [earlier, latest], pending_operation: null }
const confirmation = { entry: earlier, version: 7, revision: 18 }

test('history ignores late results for another project or an older version', () => {
  assert.equal(acceptHistory(history, 'story-1', null), true)
  assert.equal(acceptHistory(history, 'story-2', null), false)
  assert.equal(acceptHistory({ ...history, version: 6 }, 'story-1', history), false)
  assert.equal(acceptHistory({ ...history, busy: true }, 'story-1', history), true)
})

test('the confirmed history and draft revisions must both stay current', () => {
  assert.equal(restoreBlockReason(history, confirmation, 18, false, false), null)
  assert.equal(restoreBlockReason({ ...history, version: 8 }, confirmation, 18, false, false), 'stale')
  assert.equal(restoreBlockReason(history, confirmation, 19, false, false), 'stale')
  assert.equal(restoreBlockReason(null, confirmation, 18, false, false), 'stale')
  assert.deepEqual(confirmation, { entry: earlier, version: 7, revision: 18 })
})

test('generation, another save, and local unsaved inputs all prevent restoring', () => {
  assert.equal(restoreBlockReason({ ...history, busy: true }, confirmation, 18, false, false), 'busy')
  assert.equal(restoreBlockReason(history, confirmation, 18, true, false), 'busy')
  assert.equal(restoreBlockReason(history, confirmation, 18, false, true), 'pending')
  assert.equal(restoreBlockReason({ ...history, pending_operation: { label: '音声生成', status: 'failed' } }, confirmation, 18, false, false), null)
})

test('the current or missing history entry cannot be restored from an old selection', () => {
  assert.equal(restoreBlockReason(history, { ...confirmation, entry: latest }, 18, false, false), 'current')
  assert.equal(restoreBlockReason({ ...history, current_revision_id: earlier.id }, confirmation, 18, false, false), 'current')
  assert.equal(restoreBlockReason({ ...history, entries: [latest] }, confirmation, 18, false, false), 'stale')
})

test('restoring adds a later entry while the later original state remains selectable', () => {
  const restored = { id: 'saved-3', number: 3, label: '過去の状態に復元', created_at: '2026-09-24T01:20:00Z', restored_from_id: earlier.id, current: true }
  const after = { ...history, version: 8, current_revision_id: restored.id, entries: [earlier, { ...latest, current: false }, restored] }
  assert.equal(restoreBlockReason(after, confirmation, 19, false, false), 'stale')
  const forward = { entry: after.entries[1], version: 8, revision: 19 }
  assert.equal(restoreBlockReason(after, forward, 19, false, false), null)
  assert.equal(after.entries.length, 3)
})
