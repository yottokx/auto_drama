import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
const hooksUrl = dataUrl(`
let slots = [], cursor = 0, pending = [];
export function reset() { dispose(); slots = []; cursor = 0; pending = [] }
export function beginRender() { cursor = 0 }
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
export function useEffect(callback, deps) {
  const index = cursor++, old = slots[index];
  if (!old || deps.some((value, item) => !Object.is(value, old.deps[item]))) {
    slots[index] = { deps, cleanup: old?.cleanup };
    pending.push(() => { slots[index].cleanup?.(); slots[index].cleanup = callback() });
  }
}
export function flush() { const queued = pending; pending = []; queued.forEach(run => run()) }
export function dispose() { slots.forEach(slot => slot?.cleanup?.()) }
`)
const apiUrl = dataUrl(`
export class ApiError extends Error { constructor(message, status) { super(message); this.status = status } }
export const errorMessage = reason => reason.message;
export const calls = [];
let handler;
export function respondWith(next) { handler = next; calls.length = 0 }
export async function request(...args) { calls.push(args); return handler(...args) }
`)
async function moduleUrl(file, replacements = {}) {
  let source = await readFile(new URL(`../src/${file}`, import.meta.url), 'utf8')
  for (const [name, url] of Object.entries(replacements)) source = source.replaceAll(`'${name}'`, JSON.stringify(url))
  source = source.replaceAll(/import '\.\/.+\.css'/g, '')
  const { code } = await transformWithOxc(source, file, { target: 'es2022', jsx: { runtime: 'automatic' } })
  return dataUrl(code.replaceAll('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime'))))
}
const stateUrl = await moduleUrl('eventCgState.ts', { './api': apiUrl })
const state = await import(stateUrl)
const hooks = await import(hooksUrl)
const api = await import(apiUrl)
const { EventCgPolicyEditor } = await import(await moduleUrl('EventCgPolicyEditor.tsx', {
  react: hooksUrl, './api': apiUrl, './eventCgState': stateUrl,
}))
const { EventCgSettings } = await import(await moduleUrl('EventCgSettings.tsx', {
  react: hooksUrl, './api': apiUrl, './eventCgState': stateUrl,
}))
const productionStateUrl = await moduleUrl('productionState.ts')
const { EventCgStatus } = await import(await moduleUrl('ProductionPanel.tsx', {
  react: hooksUrl, './api': apiUrl, './eventCgState': stateUrl, './productionState': productionStateUrl,
  './Icons': dataUrl('export const Icon = () => null'), './PlotReview': dataUrl('export const PlotReview = () => null'),
  './GenerationProgress': dataUrl('export const GenerationProgress = () => null'),
}))

const policy = { max_cgs: 0, max_variants_per_cg: 0, revision: 3 }
const settings = {
  revision: 2, ready: true, workers: [{ id: 'gpu', name: '制作PC', ready: true }],
  profile: { backend: 'qwen_image21', model_revision: 'd26bb61231c349cf6b7896fa83353113880e1ba3', dtype: 'bfloat16', cpu_offload: true, use_kv_cache: true, steps: 40, width: 960, height: 640 },
}
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
function element(tree, type, label) {
  const found = walk(tree, node => node.type === type && text(node).startsWith(label))
  assert.ok(found, `missing ${type}: ${label}`)
  return found
}
function harness(component, props) {
  hooks.reset()
  const render = () => { hooks.beginRender(); return component(props) }
  const settle = async () => {
    let tree
    for (let i = 0; i < 4; i++) { tree = render(); hooks.flush(); await new Promise(resolve => setImmediate(resolve)) }
    return tree
  }
  return { render, settle, input(tree, label, value) { walk(element(tree, 'label', label), node => node.type === 'input').props.onChange({ target: { value } }) } }
}

test('CG limits count added variants and disabled policy does not require a Qwen environment', () => {
  assert.equal(state.eventCgMaximum({ max_cgs: 3, max_variants_per_cg: 2 }), 9)
  assert.equal(state.eventCgReadinessReason(0, null), null)
  assert.ok(state.eventCgReadinessReason(1, null))
  assert.ok(state.eventCgReadinessReason(1, { ...settings, workers: [] }))
  assert.ok(state.eventCgReadinessReason(1, { ...settings, ready: false }))
  assert.equal(state.eventCgReadinessReason(1, settings), null)
  for (const [n, d] of [[-1, 0], [101, 0], [1.5, 0], [NaN, 0], [1, 11], [1, -1]]) assert.ok(state.eventCgPolicyError(n, d))
})

test('production CG summaries distinguish unplanned images from a confirmed zero and show all chapter allocations', () => {
  const value = { max_cgs: 3, max_variants_per_cg: 2, budget_completed: false, plan_completed: false, planned: 0, generated: 0, omitted: 0, chapter_budget: null }
  let tree = EventCgStatus({ value })
  assert.match(text(tree), /第1章の本文生成前に、全章の基本CG枠を決定/)
  assert.match(text(tree), /枚数未確定/)
  assert.doesNotMatch(text(tree), /画像の計画 0枚/)
  tree = EventCgStatus({ value: { ...value, budget_completed: true, chapter_budgets: [{ chapter_number: 1, limit: 0 }, { chapter_number: 2, limit: 1 }, { chapter_number: 3, limit: 2 }] } })
  assert.match(text(tree), /作品全体のCG配分：確定/)
  assert.match(text(tree), /第1章：対象なし第2章：最大1件第3章：最大2件/)
  assert.match(text(tree), /各章の台本完成後に順次確定/)
  assert.doesNotMatch(text(tree), /画像の計画 0枚/)
  tree = EventCgStatus({ value: { ...value, budget_completed: true, chapter_budget: 0 }, chapter: true })
  assert.match(text(tree), /画像の計画 0枚（確定・差分を含む）/)
  assert.doesNotMatch(text(tree), /枚数未確定|配分待ち/)
  tree = EventCgStatus({ value: { ...value, budget_completed: true, planned: 2 } })
  assert.match(text(tree), /現時点で2枚（ほかの章は未確定/)
  tree = EventCgStatus({ value: { ...value, budget_completed: true, plan_completed: true, planned: 2 } })
  assert.match(text(tree), /画像の計画 2枚（確定・差分を含む）/)
  assert.equal(EventCgStatus({ value: { ...value, max_cgs: 0 }, chapter: true }), null)
  assert.equal(text(EventCgStatus({ value: { ...value, max_cgs: 0 } })), 'この制作版：イベントCGの自動生成なし')
})

test('failed budget generation shows the omission reason instead of a successfully confirmed zero', () => {
  const reason = '全体CG配分の生成に失敗しました：LLMの出力形式が不正です。'
  const value = {
    max_cgs: 3, max_variants_per_cg: 2, budget_completed: true, plan_completed: true,
    planned: 0, generated: 0, omitted: 0, chapter_budget: 0,
    chapter_budgets: [{ chapter_number: 1, limit: 0 }, { chapter_number: 2, limit: 0 }],
    budget_omission_reason: reason,
  }
  const tree = EventCgStatus({ value })
  assert.match(text(tree), /作品全体のCG配分：作成できず省略/)
  assert.equal(text(walk(tree, node => node.props?.role === 'status')), `理由：${reason}`)
  assert.match(text(tree), /画像の計画：CG配分を作成できなかったため省略/)
  assert.doesNotMatch(text(tree), /作品全体のCG配分：確定|対象なし|画像の計画 0枚|台本完成後/)
  const chapter = EventCgStatus({ value, chapter: true })
  assert.match(text(chapter), /この章の基本CG枠：CG配分を作成できず省略/)
  assert.match(text(chapter), /画像の計画：CG配分を作成できなかったため省略/)
  assert.equal(text(walk(chapter, node => node.props?.role === 'status')), `理由：${reason}`)
  assert.doesNotMatch(text(chapter), /最大0件|画像の計画 0枚/)
  assert.equal(state.eventCgBudgetCompleted(value), true, 'an adopted omitted budget still releases subsequent work')
  assert.equal(state.eventCgImagePlanText({ ...value, budget_omission_reason: undefined }, true), '画像の計画 0枚（確定・差分を含む）', 'legacy zero-budget summaries keep their existing behavior')
})

test('zero policy loads without probing settings and unsaved changes block approval', async () => {
  api.respondWith(async path => {
    if (path.endsWith('/policy')) return policy
    return { ...settings, ready: false, workers: [] }
  })
  let latest
  const h = harness(EventCgPolicyEditor, { projectId: 'story', disabled: false, readonly: false, onStatusChange: value => { latest = value } })
  try {
    let tree = await h.settle()
    assert.equal(api.calls.length, 1)
    assert.equal(latest.canApprove, true)
    h.input(tree, '作品全体の基本CG上限', '3')
    tree = await h.settle()
    assert.equal(latest.pending, true)
    assert.equal(latest.canApprove, false)
    assert.equal(api.calls.filter(([path]) => path === '/api/event-cg/settings').length, 1)
    assert.match(text(tree), /実行環境の準備が必要/)
    element(tree, 'button', '変更を取り消す').props.onClick()
    await h.settle()
    assert.equal(latest.canApprove, true)
    assert.equal(latest.pending, false)
  } finally { hooks.dispose() }
})

test('saving policy uses PUT and expected revision; conflicts preserve input and block adoption', async () => {
  api.respondWith(async path => path.endsWith('/policy') ? policy : settings)
  const originalFetch = globalThis.fetch
  const writes = []
  let latest
  globalThis.fetch = async (path, options) => {
    writes.push({ path, ...options, body: JSON.parse(options.body) })
    return { ok: false, status: 409, json: async () => ({ detail: '設定は別の画面で変更されました。' }) }
  }
  const h = harness(EventCgPolicyEditor, { projectId: 'story', disabled: false, readonly: false, onStatusChange: value => { latest = value } })
  try {
    let tree = await h.settle()
    h.input(tree, '作品全体の基本CG上限', '3')
    h.input(tree, 'CG1件あたりの追加差分上限', '2')
    tree = await h.settle()
    assert.match(text(tree), /最大9枚/)
    element(tree, 'button', 'CG設定を保存').props.onClick()
    tree = await h.settle()
    assert.equal(writes[0].method, 'PUT')
    assert.deepEqual(writes[0].body, { max_cgs: 3, max_variants_per_cg: 2, expected_revision: 3 })
    assert.equal(walk(element(tree, 'label', '作品全体の基本CG上限'), node => node.type === 'input').props.value, '3')
    assert.equal(latest.canApprove, false)
    assert.equal(element(tree, 'button', 'CG設定を保存').props.disabled, true)
    assert.match(text(tree), /入力は保持/)
  } finally { globalThis.fetch = originalFetch; hooks.dispose() }
})

test('image settings are lazy and keep unsaved profile when switching tabs', async () => {
  api.respondWith(async () => settings)
  const props = { visible: false, onBusyChange() {} }
  const h = harness(EventCgSettings, props)
  try {
    await h.settle()
    assert.equal(api.calls.length, 0)
    props.visible = true
    let tree = await h.settle()
    assert.equal(api.calls.length, 1)
    h.input(tree, '推論ステップ数', '25')
    props.visible = false
    await h.settle()
    props.visible = true
    tree = await h.settle()
    assert.equal(api.calls.length, 1)
    assert.equal(walk(element(tree, 'label', '推論ステップ数'), node => node.type === 'input').props.value, '25')
    assert.equal(element(tree, 'button', '画像設定を保存').props.disabled, false)
  } finally { hooks.dispose() }
})

test('image settings distinguish disconnected workers from connected workers without CG support', async () => {
  for (const value of [
    { ...settings, ready: false, workers: [] },
    { ...settings, ready: false, workers: [{ id: 'local', name: 'local', ready: false, reason: '画像生成環境を準備してください。' }] },
    settings,
  ]) {
    api.respondWith(async () => value)
    const h = harness(EventCgSettings, { visible: true, onBusyChange() {} })
    try {
      const tree = await h.settle()
      element(tree, 'h3', '接続中のワーカー')
      const list = walk(tree, node => node.type === 'ul' && node.props['aria-label'] === '接続中のワーカー')
      if (value.workers.length === 0) {
        assert.match(text(tree), /接続中のワーカーがありません。ワーカーを起動してから再読み込み/)
        assert.equal(list, null)
        assert.doesNotMatch(text(tree), /CG未準備/)
      } else if (!value.ready) {
        assert.match(text(tree), /ワーカーは接続していますが、イベントCGを生成する準備ができていません/)
        assert.match(text(list), /local画像生成環境を準備してください。CG未準備/)
        assert.doesNotMatch(text(tree), /接続中のワーカーがありません/)
      } else {
        assert.match(text(tree), /接続中のワーカーでイベントCGを生成できます/)
        assert.match(text(list), /制作PCCG生成可能/)
        assert.doesNotMatch(text(tree), /CG未準備/)
      }
    } finally { hooks.dispose() }
  }
})
