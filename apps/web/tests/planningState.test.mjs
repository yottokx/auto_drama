import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const source = await readFile(new URL('../src/planningState.ts', import.meta.url), 'utf8')
const { code } = await transformWithOxc(source, 'planningState.ts', { target: 'es2022' })
const { initialPlanningEditor, planningDirty, receivePlanning, resetPlanning, editPlanningField, planningAction, planningActivity, planningDisplayStep, canApprovePlanning, planningTabDirty, cancelPlanningTab } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)

const content = {
  cast_plan: { supporting_characters: [{ id: 'guide', name: '案内役', settings: '町に暮らす案内役' }], everyday_context: [], connections: [] },
  plot: { core: { central_question: '手紙を届けられるか', characters: [], foreshadowing: [] }, chapters: [{ number: 1, title: '出会い', events: [{ start_condition: '町に着く', steps: [{ character_id: 'guide', action: '手紙を受け取る', result: '届け先を知る' }] }] }] },
  retained_metadata: { source: 'original-cast', version: 3 },
}
const plan = (revision = 1, extra = {}) => ({ id: 'plan-1', revision, status: 'planned', content: structuredClone(content), approval_id: null, active_job_id: null, jobs: [], error: null, ...extra })
const loaded = () => receivePlanning(initialPlanningEditor, plan())

test('initial generation and newer results appear without chapter production', () => {
  const generating = receivePlanning(initialPlanningEditor, plan(1, { status: 'generating', content: null }))
  assert.equal(generating.content, null)
  assert.equal(planningDirty(generating), false)
  const ready = receivePlanning(generating, plan(2))
  assert.equal(ready.content.plot.chapters[0].title, '出会い')
  assert.deepEqual(planningAction(initialPlanningEditor, 'generate'), { action: 'generate', expected_revision: 0 })
})

test('direct event edits preserve unrelated fields and metadata without mutating the source', () => {
  const before = loaded()
  const edited = editPlanningField(before.content, ['plot', 'chapters', 0, 'events', 0, 'steps', 0, 'action'], '宛先を尋ねる')
  assert.equal(before.content.plot.chapters[0].events[0].steps[0].action, '手紙を受け取る')
  assert.deepEqual(edited.retained_metadata, before.content.retained_metadata)
  assert.deepEqual(edited.cast_plan, before.content.cast_plan)
  const state = { ...before, content: edited }
  assert.equal(planningDirty(state), true)
  assert.deepEqual(planningAction(state, 'save'), { action: 'save', expected_revision: 1, content: edited })
})

test('cancelling one editing tab restores its baseline and retains the other tab and AI instruction', () => {
  const before = loaded()
  const plotEdit = { ...before, content: editPlanningField(before.content, ['plot', 'chapters', 0, 'title'], '新しい出会い') }
  const both = { ...plotEdit, content: editPlanningField(plotEdit.content, ['cast_plan', 'supporting_characters', 0, 'name'], '航'), instruction: '会話を自然に' }
  assert.equal(planningTabDirty(both, 'plot'), true)
  assert.equal(planningTabDirty(both, 'cast'), true)
  const cancelled = cancelPlanningTab(both, 'cast', plotEdit.content)
  assert.equal(cancelled.content.cast_plan.supporting_characters[0].name, '案内役')
  assert.equal(cancelled.content.plot.chapters[0].title, '新しい出会い')
  assert.equal(cancelled.instruction, '会話を自然に')
  assert.equal(planningTabDirty(cancelled, 'plot'), true)
  assert.equal(planningTabDirty(cancelled, 'cast'), false)
  assert.equal(canApprovePlanning(cancelled), false, 'hidden-tab edits and collapsed instructions still prevent approval')
  assert.deepEqual(planningAction(cancelled, 'save').content, cancelled.content)
  assert.equal(both.content.cast_plan.supporting_characters[0].name, '航', 'cancel does not mutate the original draft')
})

test('local cancellation preserves newer-server conflicts until the user reloads the plan', () => {
  const before = loaded()
  const edited = { ...before, content: editPlanningField(before.content, ['plot', 'chapters', 0, 'title'], '未保存'), instruction: '再考する' }
  const conflict = receivePlanning(edited, plan(2))
  const cancelled = cancelPlanningTab(conflict, 'plot', before.content)
  assert.equal(cancelled.base.revision, 1)
  assert.equal(cancelled.latest.revision, 2)
  assert.equal(cancelled.conflict, true)
  assert.equal(cancelled.instruction, '再考する')
  assert.throws(() => planningAction(cancelled, 'save'), /最新/)
})

test('polling a new revision preserves direct edits and instructions until explicit reload', () => {
  const before = loaded()
  const edited = { ...before, content: editPlanningField(before.content, ['cast_plan', 'supporting_characters', 0, 'name'], '航'), instruction: '親しい間柄に' }
  const next = receivePlanning(edited, plan(2))
  assert.equal(next.content.cast_plan.supporting_characters[0].name, '航')
  assert.equal(next.base.revision, 1)
  assert.equal(next.latest.revision, 2)
  assert.equal(next.instruction, '親しい間柄に')
  assert.equal(next.conflict, true)
  assert.throws(() => planningAction(next, 'save'), /最新/)
  const reset = resetPlanning(next)
  assert.equal(reset.base.revision, 2)
  assert.equal(reset.content.cast_plan.supporting_characters[0].name, '案内役')
  assert.equal(reset.instruction, '')
  assert.equal(reset.conflict, false)
})

test('instructions alone are protected and late older polls cannot replace a newer plan', () => {
  const before = { ...loaded(), instruction: '第1章に出会いを追加' }
  const updated = receivePlanning(before, plan(3))
  assert.equal(updated.conflict, true)
  assert.equal(receivePlanning(updated, plan(2)), updated)
  const restored = receivePlanning(before, plan(1, { id: 'restored-plan' }))
  assert.equal(restored.conflict, true)
  assert.equal(restored.base.id, 'plan-1')
})

test('same-revision job updates do not erase input', () => {
  const before = { ...loaded(), instruction: '声の話し方を調整' }
  const next = receivePlanning(before, plan(1, { status: 'failed', error: '接続エラー' }))
  assert.equal(next.instruction, before.instruction)
  assert.equal(next.latest.error, '接続エラー')
  assert.equal(next.conflict, false)
})

test('revision requests include only the selected target and protect unsaved edits', () => {
  const state = { ...loaded(), instruction: '  展開を自然に  ' }
  assert.deepEqual(planningAction(state, 'revise', { target: 'plot', chapterNumber: 1, characterId: 'guide' }), { action: 'revise', expected_revision: 1, target: 'plot', chapter_number: 1, instruction: '展開を自然に' })
  assert.deepEqual(planningAction(state, 'revise', { target: 'character', characterId: 'guide', chapterNumber: 1 }), { action: 'revise', expected_revision: 1, target: 'character', character_id: 'guide', instruction: '展開を自然に' })
  assert.deepEqual(planningAction(state, 'revise', { target: 'relationships', characterId: 'guide' }), { action: 'revise', expected_revision: 1, target: 'relationships', instruction: '展開を自然に' })
  assert.throws(() => planningAction(state, 'revise', { target: 'character', characterId: 'missing' }), /サブキャラ/)
  assert.throws(() => planningAction(state, 'revise', { target: 'plot', chapterNumber: 9 }), /章/)
  assert.throws(() => planningAction({ ...state, content: editPlanningField(state.content, ['plot', 'chapters', 0, 'title'], '変更') }, 'revise'), /保存/)
})

test('approval cannot bypass unsaved edits, pending instructions, or incomplete generation', () => {
  assert.throws(() => planningAction(initialPlanningEditor, 'approve'), /生成/)
  assert.throws(() => planningAction({ ...loaded(), instruction: '変更したい' }, 'approve'), /未保存/)
  const state = loaded()
  assert.throws(() => planningAction({ ...state, content: editPlanningField(state.content, ['plot', 'chapters', 0, 'title'], '変更') }, 'approve'), /未保存/)
  assert.deepEqual(planningAction(state, 'approve'), { action: 'approve', expected_revision: 1 })
})

test('only the selected planning job controls retry and busy status', () => {
  const jobs = [{ id: 'obsolete', status: 'running' }, { id: 'current', status: 'failed' }, { id: 'old-failure', status: 'failed' }]
  assert.deepEqual(planningActivity(plan(2, { jobs, active_job_id: 'current', status: 'failed' })), { generating: false, failed: [jobs[1]] })
  assert.deepEqual(planningActivity(plan(3, { jobs, active_job_id: null, status: 'ready' })), { generating: false, failed: [] })
  assert.equal(planningActivity(plan(3, { jobs: [{ id: 'current', status: 'pending' }], active_job_id: 'current' })).generating, true)
})

test('remote navigation and upstream edits retain a pending planning screen until save or explicit discard', () => {
  for (const remote of ['world-input', 'world-review', 'character-review', 'production']) {
    assert.equal(planningDisplayStep('planning-review', remote, true), 'planning-review')
    assert.equal(planningDisplayStep('planning-review', remote, false), remote)
  }
  assert.equal(planningDisplayStep('planning-review', 'planning-review', true), 'planning-review')
  assert.equal(planningDisplayStep('character-review', 'planning-review', false), 'planning-review')
  assert.equal(planningDisplayStep('production', 'world-review', true), 'production')
  assert.equal(planningDisplayStep('production', 'planning-review', true), 'production')
  assert.equal(planningDisplayStep('production', 'planning-review', false), 'planning-review')
})

test('a failed AI revision leaves its retained valid plan available for explicit approval', () => {
  const failed = receivePlanning(initialPlanningEditor, plan(4, {
    status: 'failed', active_job_id: 'revision-job',
    jobs: [{ id: 'revision-job', status: 'failed' }], error: '修正に失敗',
  }))
  assert.equal(planningDirty(failed), false)
  assert.equal(canApprovePlanning(failed), true)
  assert.deepEqual(planningAction(failed, 'approve'), { action: 'approve', expected_revision: 4 })
  assert.equal(canApprovePlanning(failed, true), false, 'upstream invalidation or a mutation still blocks approval')
  assert.equal(canApprovePlanning({ ...failed, instruction: '別の修正案' }), false)
  assert.equal(canApprovePlanning({ ...failed, conflict: true }), false)
  assert.equal(canApprovePlanning({ ...failed, content: editPlanningField(failed.content, ['plot', 'chapters', 0, 'title'], '未保存') }), false)
  assert.equal(canApprovePlanning(receivePlanning(initialPlanningEditor, plan(1, { status: 'failed', content: null }))), false, 'initial generation failure has no plan to approve')
  const retrying = receivePlanning(failed, plan(4, { status: 'generating', active_job_id: 'revision-job', jobs: [{ id: 'revision-job', status: 'pending' }] }))
  assert.equal(canApprovePlanning(retrying), false, 'an in-flight retry must complete before approval')
})
