import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
const hooksUrl = dataUrl(`
let slots = [], cursor = 0;
export function reset() { slots = []; cursor = 0 }
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
export function useEffect() {}
export function useCallback(callback) { return callback }
`)
const childrenUrl = dataUrl(`
export function Icon() { return null }
export function Dialog({ children }) { return children }
export function CharacterReview() { return null }
export function CombinedBriefInput() { return null }
export function ProductionPanel() { return null }
export function AdjustmentScreen() { return null }
export function PlanningReviewPanel() { return null }
export function ChangeHistory() { return null }
export function SettingsDialog() { return null }
export function useM2Api() { return null }
export const activeJob = job => ['pending', 'running'].includes(job.status);
`)
async function moduleUrl(file, replacements = {}, suffix = '') {
  let source = await readFile(new URL(`../src/${file}`, import.meta.url), 'utf8')
  for (const [name, url] of Object.entries(replacements)) source = source.replaceAll(`'${name}'`, JSON.stringify(url))
  source = source.replace(/^import '\.\/.*\.css'\r?$/gm, '')
  const { code } = await transformWithOxc(source + suffix, file, { target: 'es2022', jsx: { runtime: 'automatic' } })
  return dataUrl(code.replace('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime'))))
}
const hooks = await import(hooksUrl)
const wizardUrl = await moduleUrl('wizardState.ts', { react: hooksUrl })
const wizard = await import(wizardUrl)
const worldUrl = await moduleUrl('WorldSteps.tsx', { react: hooksUrl, './Icons': childrenUrl })
const { WorldReview } = await import(worldUrl)
const { ProjectWorkspace } = await import(await moduleUrl('LiveWizardApp.tsx', {
  react: hooksUrl, './Icons': childrenUrl, './PreviewComponents': childrenUrl,
  './WorldSteps': worldUrl, './CharacterSteps': childrenUrl, './CombinedBriefInput': childrenUrl,
  './m2Api': childrenUrl, './wizardState': wizardUrl,
  './planningState': await moduleUrl('planningState.ts'),
  './RelationshipSteps': await moduleUrl('RelationshipSteps.tsx', { './Icons': childrenUrl }),
  './ProductionPanel': childrenUrl, './AdjustmentScreen': childrenUrl,
  './PlanningReviewPanel': childrenUrl, './ChangeHistory': childrenUrl, './SettingsDialog': childrenUrl,
}, '\nexport { ProjectWorkspace }\n'))

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
function harness(step, approved = true) {
  hooks.reset()
  const initial = wizard.createInitialWizardDraft()
  const world = { ...initial.world, title: '旅のはじまり', setting: '風の町' }
  const character = { ...initial.characters[0], name: '主人公' }
  const detail = { project: { id: 'story' }, jobs: [], draft: {
    revision: 8, step, worldInput: world, worldResult: world,
    worldConfirmed: step === 'world-review' ? approved : true, approved,
    planApproved: approved, approval: approved ? { id: 'approval-1' } : null, requests: [],
    characters: [{ id: character.id, input: character, result: character, locked: character.locked, imageUrl: '/portrait.png', voiceUrl: '/voice.wav' }],
  } }
  const calls = []
  const api = { mutating: false, action: async (...args) => { calls.push(args); return detail } }
  const render = () => { hooks.beginRender(); return ProjectWorkspace({ detail, api, onPendingChange() {}, onNavigate: async () => {}, onNotice() {}, onOpenAdjustments() {} }) }
  return { render, detail, api, calls }
}

test('world confirmation stays actionable after later steps and sends explicit reconfirmation', async () => {
  const originalWindow = globalThis.window
  globalThis.window = { scrollTo() {} }
  try {
    for (const confirmed of [false, true]) {
      const h = harness('world-review', confirmed)
      let review = walk(h.render(), node => node.type === WorldReview)
      assert.equal(review.props.canConfirm, true)
      review.props.onConfirm()
      await new Promise(resolve => setImmediate(resolve))
      assert.deepEqual(h.calls, [['confirm-world', { reconfirm: confirmed }]])
      review.props.onPendingChange(true)
      review = walk(h.render(), node => node.type === WorldReview)
      assert.equal(review.props.canConfirm, false, 'unsaved result edits prevent confirmation')
      review.props.onPendingChange(false)
      h.detail.draft.worldPendingChanges = true
      assert.equal(walk(h.render(), node => node.type === WorldReview).props.canConfirm, false, 'unapplied brief changes prevent confirmation')
    }
  } finally { globalThis.window = originalWindow }
})

test('confirmed world keeps its original generation button and explains downstream invalidation', () => {
  const h = harness('world-review')
  const props = walk(h.render(), node => node.type === WorldReview).props
  hooks.reset(); hooks.beginRender()
  let tree = WorldReview(props)
  const confirm = walk(tree, node => node.type === 'button' && text(node).startsWith('世界観を確定してキャラクターを生成'))
  assert.ok(confirm)
  assert.equal(confirm.props.disabled, false)
  assert.match(text(tree), /キャラクター以降のSTEPを無効にして再生成/)
  assert.match(text(tree), /変更履歴から戻せます/)
  walk(tree, node => node.type === 'button' && text(node) === '世界観を修正').props.onClick()
  hooks.beginRender(); tree = WorldReview(props)
  assert.equal(walk(tree, node => node.type === 'button' && text(node).startsWith('世界観を確定してキャラクターを生成')).props.disabled, true)
})

test('main-cast confirmation sends explicit reconfirmation and describes the retained history', async () => {
  const originalWindow = globalThis.window
  globalThis.window = { scrollTo() {} }
  try {
    for (const approved of [false, true]) {
      const h = harness('character-review', approved)
      let tree = h.render()
      const review = walk(tree, node => node.type?.name === 'CharacterReview')
      assert.equal(review.props.canApprove, true)
      review.props.onApprove()
      tree = h.render()
      const dialog = walk(tree, node => node.type?.name === 'Dialog')
      if (approved) assert.match(text(dialog), /全体計画と本編制作を無効にして/)
      walk(dialog, node => node.type === 'button' && text(node) === '承認して全体計画へ').props.onClick()
      await new Promise(resolve => setImmediate(resolve))
      assert.deepEqual(h.calls, [['approve', { reconfirm: approved }]])
      h.api.mutating = true
      assert.equal(walk(h.render(), node => node.type?.name === 'CharacterReview').props.canApprove, false)
      h.api.mutating = false
      h.detail.draft.characters[0].imagePendingChanges = true
      assert.equal(walk(h.render(), node => node.type?.name === 'CharacterReview').props.canApprove, false)
    }
  } finally { globalThis.window = originalWindow }
})
