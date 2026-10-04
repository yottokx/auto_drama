import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
const hooksUrl = dataUrl(`
let slots = [], cursor = 0;
export function reset() { slots = []; cursor = 0 }
export function beginRender() { cursor = 0 }
export function useState(initial) { const i = cursor++; if (!(i in slots)) slots[i] = typeof initial === 'function' ? initial() : initial; return [slots[i], next => { slots[i] = typeof next === 'function' ? next(slots[i]) : next }]; }
export function useRef(initial) { const i = cursor++; if (!(i in slots)) slots[i] = { current: initial }; return slots[i]; }
export function useEffect() {}
`)
const apiUrl = dataUrl(`let api; export function setApi(value) { api = value }; export function useM2Api() { return api }; export function activeJob() { return false }`)
const childrenUrl = dataUrl(`
export function Icon() {}; export function Dialog() {}; export function WorldReview() {}; export function CharacterReview() {};
export function CombinedBriefInput() {}; export function RelationshipReview() {}; export function relationshipPairs() { return [] };
export function ProductionPanel() {}; export function AdjustmentScreen() {}; export function PlanningReviewPanel() {};
export function ChangeHistory() {}; export function SettingsDialog() {}; export function createCharacterBrief() {};
export function createInitialWizardDraft() {}; export function wizardSetupBusy() { return false }; export function wizardStepLocked() { return false };
export function planningDisplayStep() {};
`)
let source = await readFile(new URL('../src/LiveWizardApp.tsx', import.meta.url), 'utf8')
source = source.replace(/import '\.\/[^']+\.css'\r?\n/g, '')
source = source.replaceAll("from 'react'", `from '${hooksUrl}'`).replaceAll("from './m2Api'", `from '${apiUrl}'`)
source = source.replace(/from '\.\/[^']+'/g, `from '${childrenUrl}'`)
const { code } = await transformWithOxc(source, 'LiveWizardApp.tsx', { target: 'es2022', jsx: { runtime: 'automatic' } })
const { default: LiveWizardApp } = await import(dataUrl(code.replace('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime')))))
const hooks = await import(hooksUrl)
const apiModule = await import(apiUrl)
function walk(node, predicate) {
  if (!node || typeof node !== 'object') return []
  if (Array.isArray(node)) return node.flatMap(value => walk(value, predicate))
  return [...(predicate(node) ? [node] : []), ...walk(node.props?.children, predicate)]
}
function setup(t) {
  hooks.reset()
  const calls = []
  apiModule.setApi({
    selectedId: 'story', projects: [], loading: false, mutating: false,
    detail: { project: { id: 'story' }, jobs: [], draft: { step: 'production', worldInput: { title: '検証用の物語' } } },
    action: async (...args) => { calls.push(args); return true }, refresh: async () => {},
  })
  const oldWindow = globalThis.window
  globalThis.window = { scrollTo() {} }
  t.after(() => { globalThis.window = oldWindow })
  const render = () => { hooks.beginRender(); return LiveWizardApp() }
  const component = name => walk(render(), node => node.type?.name === name)[0]
  component('ProjectWorkspace').props.onOpenAdjustments()
  return { render, component, calls }
}

test('completed adjustments replace the entire project workspace and return to production', async t => {
  const h = setup(t)
  assert.ok(h.component('AdjustmentScreen'))
  assert.equal(h.component('ProjectWorkspace'), undefined)
  assert.equal(h.component('AdjustmentScreen').props.title, '検証用の物語')
  await h.component('AdjustmentScreen').props.onBack()
  assert.equal(h.component('AdjustmentScreen'), undefined)
  assert.ok(h.component('ProjectWorkspace'))
  assert.deepEqual(h.calls, [], 'returning to the saved production step should not change the story')
})

test('unsaved adjustment edits block both the back button and same-step sidebar navigation', async t => {
  const h = setup(t)
  h.component('AdjustmentScreen').props.onPendingChange(true)
  await h.component('AdjustmentScreen').props.onBack()
  assert.ok(h.component('AdjustmentScreen'))
  const nav = walk(h.render(), node => node.type === 'nav' && node.props['aria-label'] === '制作ステップ')[0]
  const productionButton = walk(nav, node => node.type === 'button').at(-1)
  await productionButton.props.onClick()
  assert.ok(h.component('AdjustmentScreen'))
  assert.deepEqual(h.calls, [])
  h.component('AdjustmentScreen').props.onPendingChange(false)
  await h.component('AdjustmentScreen').props.onBack()
  assert.ok(h.component('ProjectWorkspace'))
})
