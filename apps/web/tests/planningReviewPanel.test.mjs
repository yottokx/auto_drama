import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
const hooksUrl = dataUrl(`
let slots = [], cursor = 0;
export const effects = [];
export function reset() { slots = []; cursor = 0; effects.length = 0 }
export function beginRender() { cursor = 0; effects.length = 0 }
export function useState(initial) {
  const index = cursor++;
  if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial;
  return [slots[index], value => { slots[index] = typeof value === 'function' ? value(slots[index]) : value }];
}
export function useRef(initial) {
  const index = cursor++;
  if (!(index in slots)) slots[index] = { current: initial };
  return slots[index];
}
export function useEffect(callback, dependencies) { effects.push({ callback, dependencies }) }
export function useId() { return 'planning-test' }
`)
const apiUrl = dataUrl(`
export class ApiError extends Error { constructor(message, status) { super(message); this.status = status } }
export const errorMessage = reason => reason.message;
export const calls = [];
let handler;
export function respondWith(next) { handler = next; calls.length = 0 }
export async function request(...args) { calls.push(args); return handler(...args) }
`)
const childrenUrl = dataUrl(`
export function Icon() { return null }
export function Dialog({ children }) { return children }
export function PlanningContentEditor() { return null }
export function PlanningContentView() { return null }
export function GenerationProgress() { return null }
export function EventCgPolicyEditor() { return null }
`)
async function moduleUrl(file, replacements = {}) {
  let source = await readFile(new URL(`../src/${file}`, import.meta.url), 'utf8')
  for (const [name, url] of Object.entries(replacements)) source = source.replaceAll(`'${name}'`, JSON.stringify(url))
  source = source.replace("import './planning.css'", '')
  const { code } = await transformWithOxc(source, file, { target: 'es2022', jsx: { runtime: 'automatic' } })
  return dataUrl(code.replace('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime'))))
}
const stateUrl = await moduleUrl('planningState.ts')
const productionStateUrl = await moduleUrl('productionState.ts')
const state = await import(stateUrl)
const hooks = await import(hooksUrl)
const api = await import(apiUrl)
const { PlanningReviewPanel } = await import(await moduleUrl('PlanningReviewPanel.tsx', {
  react: hooksUrl, './api': apiUrl, './planningState': stateUrl,
  './Icons': childrenUrl, './PreviewComponents': childrenUrl,
  './PlanningContentEditor': childrenUrl, './PlanningContentView': childrenUrl,
  './GenerationProgress': childrenUrl, './productionState': productionStateUrl,
  './EventCgPolicyEditor': childrenUrl,
}))

const content = {
  cast_plan: { supporting_characters: [{ id: 'guide', name: '案内役' }], everyday_context: [], connections: [] },
  plot: { core: { central_question: '手紙は届くか' }, chapters: [{ number: 1, title: '出会い' }] },
}
const plan = () => ({ id: 'plan', revision: 1, status: 'ready', content: structuredClone(content), approval_id: null, active_job_id: null, jobs: [], error: null })
function walk(node, predicate) {
  if (!node || typeof node !== 'object') return null
  if (Array.isArray(node)) { for (const child of node) { const found = walk(child, predicate); if (found) return found } return null }
  return predicate(node) ? node : walk(node.props?.children, predicate)
}
function text(node) {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(text).join('')
  return node?.props ? text(node.props.children) : ''
}
function renderHarness(planOverrides = {}) {
  hooks.reset()
  const initial = { ...plan(), ...planOverrides }
  Object.assign(state.initialPlanningEditor, { latest: initial, base: initial, content: initial.content, instruction: '', conflict: false })
  const pending = []
  const props = { projectId: 'story', mainCharacters: [], mainApprovalId: 'main-1', setupApproved: true, serverStep: 'planning-review', onRefresh: async () => {}, onReloadSetup() {}, onPendingChange: value => pending.push(value), onApproved: async () => {}, onBack() {}, onProduction() {} }
  const render = () => { hooks.beginRender(); return PlanningReviewPanel(props) }
  const button = (tree, label) => {
    const match = walk(tree, node => node.type === 'button' && text(node).startsWith(label))
    assert.ok(match, `missing button: ${label}`)
    return match
  }
  const child = (tree, name) => walk(tree, node => node.type?.name === name)
  return { render, button, child, pending }
}

test('the planning panel starts in readable tab mode and exposes direct editing and AI instructions explicitly', () => {
  const h = renderHarness()
  let tree = h.render()
  assert.equal(h.child(tree, 'PlanningContentView').props.tab, 'plot')
  assert.equal(h.child(tree, 'PlanningContentEditor'), null)
  assert.equal(walk(tree, node => ['input', 'textarea', 'select', 'fieldset'].includes(node.type)), null)
  assert.equal(h.button(tree, 'AIに修正を指示').props['aria-expanded'], false)
  h.button(tree, 'サブキャラ').props.onClick()
  tree = h.render()
  assert.equal(h.child(tree, 'PlanningContentView').props.tab, 'cast')
  assert.equal(h.button(tree, 'サブキャラ').props['aria-selected'], true)
  let prevented = false
  h.button(tree, 'サブキャラ').props.onKeyDown({ key: 'Home', preventDefault() { prevented = true } })
  tree = h.render()
  assert.equal(prevented, true)
  assert.equal(h.button(tree, 'プロット').props['aria-selected'], true)
  h.button(tree, '直接編集').props.onClick()
  assert.equal(h.child(h.render(), 'PlanningContentEditor').props.tab, 'plot')
})

test('planning progress uses only the selected job and does not show obsolete generation stages', () => {
  const progress = { schema_version: 1, sequence: 2, phase: 'planning', current_step: 'current-plot', attempt: 1, active: true, updated_at: '', steps: [
    { id: 'current-cast', stage: 'cast_plan', status: 'completed' },
    { id: 'current-plot', stage: 'plot_plan', status: 'running' },
  ] }
  const h = renderHarness({ status: 'generating', active_job_id: 'selected', jobs: [
    { id: 'obsolete', kind: 'm3_plan', status: 'running', attempt_count: 1, progress: { ...progress, steps: [{ id: 'old', stage: 'revision', status: 'running' }] } },
    { id: 'selected', kind: 'm3_plan', status: 'running', attempt_count: 1, progress },
  ] })
  const display = h.child(h.render(), 'GenerationProgress')
  assert.deepEqual(display.props.items.map(item => item.id), ['current-cast', 'current-plot'])
  assert.deepEqual(display.props.items.map(item => item.label), ['サブキャラの設定・関係性', '物語の核心・全体プロット'])
})

test('only an explicit tab change scrolls to the nonsticky tab anchor after rendering', () => {
  const h = renderHarness()
  const scrolls = []
  const scrollEffect = () => hooks.effects.find(effect => effect.dependencies.length === 1 && ['plot', 'cast'].includes(effect.dependencies[0])).callback()
  let tree = h.render()
  const anchor = walk(tree, node => node.props?.className === 'planning-tabs-anchor')
  anchor.props.ref.current = { scrollIntoView: options => scrolls.push(options) }
  scrollEffect()
  assert.deepEqual(scrolls, [], 'loading a plan must not move the viewport')
  h.button(tree, 'プロット').props.onClick()
  tree = h.render()
  scrollEffect()
  assert.deepEqual(scrolls, [], 'selecting the already-active tab must not move the viewport')
  h.button(tree, 'サブキャラ').props.onClick()
  tree = h.render()
  assert.equal(scrolls.length, 0, 'scroll waits for the shorter panel to replace the long content')
  scrollEffect()
  assert.deepEqual(scrolls, [{ block: 'start', behavior: 'instant' }])
  tree = h.render()
  scrollEffect()
  assert.equal(scrolls.length, 1, 'unrelated rerenders must not scroll again')
  h.button(tree, 'サブキャラ').props.onKeyDown({ key: 'ArrowLeft', preventDefault() {} })
  h.render()
  scrollEffect()
  assert.equal(scrolls.length, 2, 'keyboard selection gets the same post-render positioning')
})

test('tab switching and scoped cancellation preserve other edits and collapsed AI input, then saving returns to reading', async () => {
  const h = renderHarness()
  let tree = h.render()
  h.button(tree, '直接編集').props.onClick()
  tree = h.render()
  let editor = h.child(tree, 'PlanningContentEditor')
  editor.props.onChange(state.editPlanningField(editor.props.content, ['plot', 'chapters', 0, 'title'], '新しい出会い'))
  tree = h.render()
  h.button(tree, 'サブキャラ').props.onClick()
  tree = h.render()
  assert.equal(h.child(tree, 'PlanningContentEditor'), null)
  assert.ok(text(h.button(tree, 'プロット')).includes('未保存'))
  assert.equal(h.button(tree, '全体計画を承認').props.disabled, true)
  h.button(tree, '直接編集').props.onClick()
  tree = h.render()
  editor = h.child(tree, 'PlanningContentEditor')
  editor.props.onChange(state.editPlanningField(editor.props.content, ['cast_plan', 'supporting_characters', 0, 'name'], '航'))
  tree = h.render()
  h.button(tree, 'AIに修正を指示').props.onClick()
  tree = h.render()
  walk(tree, node => node.type === 'textarea').props.onChange({ target: { value: '展開を自然に' } })
  tree = h.render()
  h.button(tree, '修正指示を閉じる').props.onClick()
  tree = h.render()
  assert.equal(walk(tree, node => node.type === 'textarea'), null)
  h.button(tree, '編集を取り消す').props.onClick()
  tree = h.render()
  let reader = h.child(tree, 'PlanningContentView')
  assert.equal(reader.props.content.cast_plan.supporting_characters[0].name, '案内役')
  assert.equal(reader.props.content.plot.chapters[0].title, '新しい出会い')
  h.button(tree, 'プロット').props.onClick()
  tree = h.render()
  editor = h.child(tree, 'PlanningContentEditor')
  assert.equal(editor.props.content.plot.chapters[0].title, '新しい出会い')
  api.respondWith((_url, _signal, body) => ({ project_id: 'story', planning: { ...plan(), revision: 2, content: body.content }, legacy_production: false }))
  h.button(tree, '変更を保存する').props.onClick()
  await new Promise(resolve => setImmediate(resolve))
  tree = h.render()
  reader = h.child(tree, 'PlanningContentView')
  assert.ok(reader)
  assert.equal(h.child(tree, 'PlanningContentEditor'), null)
  assert.equal(api.calls[0][2].expected_revision, 1)
  assert.equal(api.calls[0][2].content.plot.chapters[0].title, '新しい出会い')
  assert.equal(api.calls[0][2].content.cast_plan.supporting_characters[0].name, '案内役')
  h.button(tree, 'AIに修正を指示').props.onClick()
  tree = h.render()
  assert.equal(walk(tree, node => node.type === 'textarea').props.value, '展開を自然に')
  assert.equal(h.button(tree, '全体計画を承認').props.disabled, true)
  assert.equal(h.pending.at(-1), true)
})

test('planning approval waits for saved CG policy readiness and submits its revision', async () => {
  const h = renderHarness()
  let tree = h.render()
  assert.equal(h.button(tree, '全体計画を承認').props.disabled, true)
  const policy = { max_cgs: 2, max_variants_per_cg: 1, revision: 5 }
  h.child(tree, 'EventCgPolicyEditor').props.onStatusChange({ pending: true, canApprove: false, policy })
  tree = h.render()
  assert.equal(h.button(tree, '全体計画を承認').props.disabled, true)
  h.child(tree, 'EventCgPolicyEditor').props.onStatusChange({ pending: false, canApprove: true, policy })
  tree = h.render()
  assert.equal(h.button(tree, '全体計画を承認').props.disabled, false)
  h.button(tree, '全体計画を承認').props.onClick()
  tree = h.render()
  assert.match(text(tree), /イベントCGは作品全体で最大2件/)
  api.respondWith(() => ({ project_id: 'story', planning: { ...plan(), status: 'approved' }, legacy_production: false }))
  h.button(tree, '承認して第1章から制作する').props.onClick()
  await new Promise(resolve => setImmediate(resolve))
  assert.equal(api.calls[0][2].expected_event_cg_policy_revision, 5)
  assert.equal(api.calls[0][2].expected_revision, 1)
  assert.equal(api.calls[0][2].reconfirm, false)
})

test('an approved plan retains its confirmation button and CG controls and explicitly starts another production', async () => {
  const h = renderHarness({ status: 'approved', approval_id: 'approval-1', revision: 8 })
  let tree = h.render()
  assert.ok(h.button(tree, '本編の制作・鑑賞へ'))
  assert.equal(h.button(tree, '全体計画を承認').props.disabled, true, 'wait for CG settings to load')
  const policy = { max_cgs: 3, max_variants_per_cg: 1, revision: 6 }
  const cg = h.child(tree, 'EventCgPolicyEditor')
  assert.equal(cg.props.readonly, false)
  cg.props.onStatusChange({ pending: true, canApprove: false, policy })
  tree = h.render()
  assert.equal(h.button(tree, '全体計画を承認').props.disabled, true, 'unsaved CG changes block reconfirmation')
  h.child(tree, 'EventCgPolicyEditor').props.onStatusChange({ pending: false, canApprove: true, policy })
  tree = h.render()
  assert.equal(h.button(tree, '全体計画を承認').props.disabled, false)
  h.button(tree, '全体計画を承認').props.onClick()
  tree = h.render()
  const dialog = h.child(tree, 'Dialog')
  assert.match(text(dialog), /現在の本編制作を無効にして/)
  assert.match(text(dialog), /変更履歴から戻せます/)
  api.respondWith(() => ({ project_id: 'story', planning: { ...plan(), status: 'approved', revision: 9 }, legacy_production: false }))
  h.button(tree, '承認して第1章から制作する').props.onClick()
  await new Promise(resolve => setImmediate(resolve))
  assert.deepEqual(api.calls[0][2], { action: 'approve', expected_revision: 8, reconfirm: true, expected_event_cg_policy_revision: 6 })
})

test('a CG readiness or policy race reloads CG controls without invalidating the story plan', async () => {
  const h = renderHarness()
  let tree = h.render()
  const before = h.child(tree, 'EventCgPolicyEditor').key
  h.child(tree, 'EventCgPolicyEditor').props.onStatusChange({ pending: false, canApprove: true, policy: { max_cgs: 1, max_variants_per_cg: 0, revision: 2 } })
  tree = h.render()
  h.button(tree, '全体計画を承認').props.onClick()
  tree = h.render()
  api.respondWith(() => { throw new api.ApiError('イベントCG設定が更新されました。最新の設定を確認してください。', 409) })
  h.button(tree, '承認して第1章から制作する').props.onClick()
  await new Promise(resolve => setImmediate(resolve))
  tree = h.render()
  assert.notEqual(h.child(tree, 'EventCgPolicyEditor').key, before)
  assert.equal(h.child(tree, 'EventCgPolicyEditor').props.disabled, false)
  assert.equal(h.button(tree, '承認して第1章から制作する').props.disabled, true)
  assert.doesNotMatch(text(tree), /保存済みの全体計画が更新/)
})
