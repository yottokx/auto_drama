import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { transformWithOxc } from 'vite'
import { adjustmentFixture } from './adjustmentFixture.mjs'

const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
const hooksUrl = dataUrl(`
let slots = [], cursor = 0;
export const effects = [];
export function reset() { slots = []; cursor = 0; effects.length = 0 }
export function beginRender() { cursor = 0; effects.length = 0 }
export function useState(initial) { const i = cursor++; if (!(i in slots)) slots[i] = typeof initial === 'function' ? initial() : initial; return [slots[i], next => { slots[i] = typeof next === 'function' ? next(slots[i]) : next }]; }
export function useRef(initial) { const i = cursor++; if (!(i in slots)) slots[i] = { current: initial }; return slots[i]; }
export function useEffect(callback, dependencies) { effects.push({ callback, dependencies }) }
`)
const apiUrl = dataUrl(`export class ApiError extends Error { constructor(message, status) { super(message); this.status = status } }; export const errorMessage = error => error.message;`)
const childrenUrl = dataUrl(`export function Icon() { return null }; export function PortraitCanvas() { return null }; export function AdjustmentPreview() { return null }; export function GenerationProgress() { return null }; export function Dialog() { return null };`)
async function moduleUrl(file, replacements = {}) {
  let source = await readFile(new URL(`../src/${file}`, import.meta.url), 'utf8')
  for (const [name, url] of Object.entries(replacements)) source = source.replaceAll(`'${name}'`, JSON.stringify(url))
  source = source.replace("import './adjustment.css'", '')
  const { code } = await transformWithOxc(source, file, { target: 'es2022', jsx: { runtime: 'automatic' } })
  return dataUrl(code.replace('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime'))))
}
const stateUrl = await moduleUrl('adjustmentState.ts')
const state = await import(stateUrl)
const hooks = await import(hooksUrl)
const { AdjustmentEditor } = await import(await moduleUrl('AdjustmentEditor.tsx', { react: hooksUrl, './api': apiUrl, './Icons': childrenUrl, './PortraitEditor': childrenUrl, './AdjustmentPreview': childrenUrl, './GenerationProgress': childrenUrl, './PreviewComponents': childrenUrl, './adjustmentState': stateUrl }))
const { AdjustmentPreview } = await import(await moduleUrl('AdjustmentPreview.tsx'))
function walkAll(node, predicate) {
  if (!node || typeof node !== 'object') return []
  if (Array.isArray(node)) return node.flatMap(child => walkAll(child, predicate))
  return [...(predicate(node) ? [node] : []), ...walkAll(node.props?.children, predicate)]
}
function text(node) {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(text).join('')
  return node?.props ? text(node.props.children) : ''
}
const settle = () => new Promise(resolve => setImmediate(resolve))
function harness(t, source = adjustmentFixture(), props = {}) {
  hooks.reset()
  Object.assign(state.initialAdjustmentEditor, state.receiveAdjustment({ source: null, base: null, characters: [], conflict: false }, source, 'adopt'))
  const render = () => { hooks.beginRender(); return AdjustmentEditor({ projectId: 'story', ...props }) }
  let tree = render()
  const cleanup = hooks.effects.find(effect => effect.dependencies.length === 1 && effect.dependencies[0] === 'story').callback()
  t.after(cleanup)
  const button = (tree, label) => {
    const found = walkAll(tree, node => node.type === 'button' && text(node) === label)[0]
    assert.ok(found, `missing button: ${label}`)
    return found
  }
  const calls = []
  const originalFetch = globalThis.fetch
  let handler = async () => source
  globalThis.fetch = async (path, options) => {
    calls.push({ path, options })
    const value = await handler(path, options)
    return value?.ok === false ? value : { ok: true, json: async () => structuredClone(value) }
  }
  t.after(() => { globalThis.fetch = originalFetch })
  return { render, button, calls, handle(next) { handler = next }, source }
}

test('preview uses exact exported coordinates and 3:2 frame with background and message overlap', () => {
  const geometry = adjustmentFixture().draft.geometry
  const markup = renderToStaticMarkup(AdjustmentPreview({ geometry, backgroundUrl: '/background/harbor.png', characters: [{ id: 'main', name: '主人公', imageUrl: '/main.png', slot: 'right' }] }))
  assert.ok(markup.includes('viewBox="0 0 960 640"'))
  assert.match(markup, /href="\/background\/harbor.png"[^>]*width="960"[^>]*height="640"[^>]*preserveAspectRatio="xMidYMid slice"/)
  assert.match(markup, /href="\/main.png"[^>]*x="660"[^>]*y="60"[^>]*width="300"[^>]*height="700"/)
  assert.ok(markup.includes('x="20" y="440" width="920" height="180"'))
  assert.ok(markup.includes('padding:18px 24px 18px 24px'))
  assert.ok(markup.includes('font-size:24px'))
  assert.ok(markup.indexOf('/main.png') < markup.indexOf('<foreignObject'), 'message layer must cover the lower portrait')
})

test('the editor exposes all chapter cast and keeps preview-only selection out of draft changes', t => {
  const h = harness(t)
  let tree = h.render()
  const checkboxes = () => walkAll(h.render(), node => node.type === 'input' && node.props.type === 'checkbox')
  assert.equal(checkboxes().length, 4)
  checkboxes()[0].props.onChange(); checkboxes()[1].props.onChange(); checkboxes()[2].props.onChange()
  assert.equal(checkboxes()[3].props.disabled, true)
  tree = h.render()
  assert.equal(h.button(tree, '下書きを保存').props.disabled, true)
  assert.ok(text(tree).includes('後の章の人物'))
  assert.equal(h.calls.length, 0)
})

test('layout changes save all characters at the expected revision before applying without editing character settings', async t => {
  const h = harness(t)
  let tree = h.render()
  walkAll(tree, node => node.type?.name === 'AdjustmentRange' && node.props.name === 'offset-y')[0].props.onChange(-32)
  tree = h.render()
  assert.equal(h.button(tree, '調整版を反映').props.disabled, true)
  h.handle(async (path, options) => {
    const response = structuredClone(h.source)
    if (options.method === 'PUT') { response.draft.revision = 2; response.draft.characters = JSON.parse(options.body).characters }
    return response
  })
  h.button(tree, '下書きを保存').props.onClick()
  await settle()
  assert.equal(h.calls[0].path, '/api/m3/projects/story/adjustments')
  const body = JSON.parse(h.calls[0].options.body)
  assert.deepEqual(Object.keys(body).sort(), ['characters', 'expected_revision'])
  assert.equal(body.expected_revision, 1)
  assert.equal(body.characters.length, 4)
  assert.equal(body.characters[0].offset_y, -32)
  assert.ok(body.characters.every(person => !('result' in person) && !('appliedInstructions' in person)))
  tree = h.render()
  assert.equal(h.button(tree, '下書きを保存').props.disabled, true)
  assert.equal(h.button(tree, '調整版を反映').props.disabled, false)
  h.button(tree, '調整版を反映').props.onClick()
  await settle()
  assert.equal(h.calls[1].path, '/api/m3/projects/story/adjustments/apply')
  assert.deepEqual(JSON.parse(h.calls[1].options.body), { expected_revision: 2 })
})

test('a one-time generation sends only the new prompt and keeps unsaved geometry through its response', async t => {
  const h = harness(t)
  let tree = h.render()
  walkAll(tree, node => node.type?.name === 'AdjustmentRange' && node.props.name === 'scale')[0].props.onChange(135)
  h.button(tree, '立ち絵').props.onClick()
  walkAll(h.render(), node => node.type === 'textarea' && node.props['aria-label'] === '今回だけの変更指示')[0].props.onChange({ target: { value: '今回だけ柔らかい表情に' } })
  h.handle(async () => { const response = structuredClone(h.source); response.draft.revision = 2; return response })
  tree = h.render()
  h.button(tree, 'この設定で候補を生成').props.onClick()
  await settle()
  const body = JSON.parse(h.calls[0].options.body)
  assert.deepEqual(body, { expected_revision: 1, character_id: 'main', kind: 'image', source_prompt: '元の服装', instruction: '今回だけ柔らかい表情に' })
  tree = h.render()
  assert.equal(walkAll(tree, node => node.type === 'textarea' && node.props['aria-label'] === '今回だけの変更指示')[0].props.value, '')
  h.button(tree, '表示位置調整').props.onClick()
  tree = h.render()
  assert.equal(walkAll(tree, node => node.type?.name === 'AdjustmentRange' && node.props.name === 'scale')[0].props.value, 135)
  assert.equal(h.button(tree, '下書きを保存').props.disabled, false)
})

test('voice uploads require the transcript and send file plus matching text as multipart without altering the story', async t => {
  const h = harness(t)
  h.button(h.render(), '基準音声').props.onClick()
  let tree = h.render()
  const file = new File(['voice bytes'], 'voice.mp3', { type: 'audio/mpeg' })
  walkAll(tree, node => node.type === 'input' && node.props.type === 'file')[0].props.onChange({ target: { files: [file] } })
  tree = h.render()
  assert.equal(h.button(tree, 'ファイルを候補に保存').props.disabled, false)
  walkAll(tree, node => node.type === 'textarea' && node.props['aria-label'] === '音声に含まれる読み上げ文')[0].props.onChange({ target: { value: '' } })
  tree = h.render()
  assert.equal(h.button(tree, 'ファイルを候補に保存').props.disabled, true)
  walkAll(tree, node => node.type === 'textarea' && node.props['aria-label'] === '音声に含まれる読み上げ文')[0].props.onChange({ target: { value: 'この音声の読み上げ文です。' } })
  tree = h.render()
  assert.equal(h.button(tree, 'ファイルを候補に保存').props.disabled, false)
  h.button(tree, 'ファイルを候補に保存').props.onClick()
  await settle()
  assert.equal(h.calls[0].path, '/api/m3/projects/story/adjustments/upload')
  const form = h.calls[0].options.body
  assert.ok(form instanceof FormData)
  assert.equal(form.get('reference_text'), 'この音声の読み上げ文です。')
  assert.equal(form.get('expected_revision'), '1')
  assert.equal(form.get('character_id'), 'main')
  assert.equal(form.get('kind'), 'voice')
  assert.equal(form.get('file').name, 'voice.mp3')
  assert.equal(h.calls[0].options.headers, undefined, 'the browser must supply the multipart boundary')
})

test('revision conflicts preserve layout and require an explicit discard before reading the current server draft', async t => {
  const h = harness(t)
  walkAll(h.render(), node => node.type?.name === 'AdjustmentRange' && node.props.name === 'offset-y')[0].props.onChange(47)
  h.handle(async () => ({ ok: false, status: 409, json: async () => ({ detail: '別の画面で変更されました' }) }))
  h.button(h.render(), '下書きを保存').props.onClick()
  await settle()
  let tree = h.render()
  assert.ok(text(tree).includes('入力は保持しています'))
  assert.equal(walkAll(tree, node => node.type?.name === 'AdjustmentRange' && node.props.name === 'offset-y')[0].props.value, 47)
  h.button(tree, '入力を取り消して読み直す').props.onClick()
  tree = h.render()
  assert.equal(h.calls.length, 1)
  const latest = structuredClone(h.source); latest.draft.revision = 3; latest.draft.characters[0].offset_y = -10
  h.handle(async () => latest)
  h.button(tree, '入力を破棄して最新の下書きを読み込む').props.onClick()
  await settle()
  tree = h.render()
  assert.equal(h.calls[1].options.method, 'GET')
  assert.equal(walkAll(tree, node => node.type?.name === 'AdjustmentRange' && node.props.name === 'offset-y')[0].props.value, -10)
  assert.equal(h.button(tree, '下書きを保存').props.disabled, true)
})

test('readonly and applying drafts keep media available but disable edits and publication', t => {
  for (const mode of ['readonly', 'applying']) {
    const source = adjustmentFixture()
    if (mode === 'readonly') source.readonly = true
    else { source.draft.status = 'applying'; source.busy = true }
    const h = harness(t, source)
    h.button(h.render(), '立ち絵').props.onClick()
    const tree = h.render()
    assert.ok(walkAll(tree, node => node.type === 'fieldset').every(node => node.props.disabled))
    assert.equal(h.button(tree, '調整版を反映').props.disabled, true)
    assert.ok(walkAll(tree, node => node.type === 'img').length > 0)
  }
})

test('failed candidate generation is retried without republishing or changing the saved material selection', async t => {
  const source = adjustmentFixture()
  source.jobs = [{ id: 'failed-image', kind: 'm3_image', status: 'failed', attempt_count: 3, error: '生成を再試行してください' }]
  source.candidates.push({ id: 'failed-candidate', character_id: 'main', kind: 'image', source: 'generated', artifact_id: null, url: null, job_id: 'failed-image' })
  const h = harness(t, source)
  const tree = h.render()
  assert.equal(h.button(tree, '失敗した処理を再試行').props.disabled, false)
  h.button(tree, '失敗した処理を再試行').props.onClick()
  await settle()
  assert.equal(h.calls[0].path, '/api/m3/projects/story/adjustments/retry')
  assert.deepEqual(JSON.parse(h.calls[0].options.body), { expected_revision: 1 })
  assert.equal(h.button(h.render(), '下書きを保存').props.disabled, true)
})

test('a second click cannot submit another candidate while the first request is unresolved', async t => {
  const h = harness(t)
  h.button(h.render(), '立ち絵').props.onClick()
  walkAll(h.render(), node => node.type === 'textarea' && node.props['aria-label'] === '今回だけの変更指示')[0].props.onChange({ target: { value: '今回の候補を生成' } })
  let finish
  h.handle(() => new Promise(resolve => { finish = resolve }))
  const button = h.button(h.render(), 'この設定で候補を生成')
  button.props.onClick(); button.props.onClick()
  assert.equal(h.calls.length, 1)
  assert.equal(h.button(h.render(), 'この設定で候補を生成').props.disabled, true)
  finish(h.source)
  await settle()
  assert.equal(walkAll(h.render(), node => node.type === 'textarea' && node.props['aria-label'] === '今回だけの変更指示')[0].props.value, '')
})

test('dedicated editor uses separate tabs and preserves each character and material prompt', t => {
  const h = harness(t)
  let tree = h.render()
  assert.equal(tree.type, 'section', 'the dedicated editor is not a collapsed inline details panel')
  assert.equal(walkAll(tree, node => node.props?.role === 'tablist').length, 1)
  assert.deepEqual(walkAll(tree, node => node.props?.role === 'tab').map(node => [text(node), node.props['aria-selected']]), [['表示位置調整', true], ['立ち絵', false], ['基準音声', false]])
  assert.equal(walkAll(tree, node => node.type === 'textarea').length, 0)
  h.button(tree, '立ち絵').props.onClick()
  tree = h.render()
  assert.equal(walkAll(tree, node => node.type?.name === 'AdjustmentRange').length, 0)
  const prompt = tree => walkAll(tree, node => node.type === 'textarea' && ['外見の設定', '声・話し方の設定'].includes(node.props['aria-label']))[0]
  assert.equal(prompt(tree).props.value, '元の服装')
  prompt(tree).props.onChange({ target: { value: '編集した服装' } })
  h.button(h.render(), '基準音声').props.onClick()
  tree = h.render()
  assert.equal(prompt(tree).props.value, '元の声')
  prompt(tree).props.onChange({ target: { value: '編集した声' } })
  const characterSelector = tree => walkAll(tree, node => node.type === 'select' && node.props.value === 'main')[0]
  characterSelector(h.render()).props.onChange({ target: { value: 'guide' } })
  assert.equal(prompt(h.render()).props.value, '元の声')
  walkAll(h.render(), node => node.type === 'select' && node.props.value === 'guide')[0].props.onChange({ target: { value: 'main' } })
  assert.equal(prompt(h.render()).props.value, '編集した声')
  h.button(h.render(), '立ち絵').props.onClick()
  assert.equal(prompt(h.render()).props.value, '編集した服装')
  assert.equal(h.button(h.render(), '調整版を反映').props.disabled, true, 'unsubmitted prompt edits cannot silently be skipped when applying')
})

test('direct prompt edits can generate without an instruction, retain input, and clear pending after submission', async t => {
  const pending = []
  const h = harness(t, adjustmentFixture(), { onPendingChange: value => pending.push(value) })
  const reportPending = () => hooks.effects.find(effect => effect.dependencies.length === 3 && effect.dependencies.every(value => typeof value === 'boolean')).callback()
  h.button(h.render(), '立ち絵').props.onClick()
  let tree = h.render()
  reportPending()
  assert.equal(pending.at(-1), false)
  const prompt = tree => walkAll(tree, node => node.type === 'textarea' && ['外見の設定', '声・話し方の設定'].includes(node.props['aria-label']))[0]
  prompt(tree).props.onChange({ target: { value: '足元まで入る全身の立ち姿。青い靴。' } })
  tree = h.render()
  reportPending()
  assert.equal(pending.at(-1), true)
  assert.ok(text(tree).includes('元の服装'))
  h.button(tree, 'この設定で候補を生成').props.onClick()
  await settle()
  assert.deepEqual(JSON.parse(h.calls[0].options.body), { expected_revision: 1, character_id: 'main', kind: 'image', source_prompt: '足元まで入る全身の立ち姿。青い靴。', instruction: '' })
  tree = h.render()
  reportPending()
  assert.equal(prompt(tree).props.value, '足元まで入る全身の立ち姿。青い靴。')
  assert.equal(pending.at(-1), false)
  prompt(tree).props.onChange({ target: { value: 'さらに編集' } })
  tree = h.render(); reportPending()
  assert.equal(pending.at(-1), true)
  h.button(tree, '元の設定に戻す').props.onClick()
  tree = h.render(); reportPending()
  assert.equal(prompt(tree).props.value, '元の服装')
  assert.equal(pending.at(-1), false)
})

test('portrait candidates open an enlarged closable dialog', t => {
  const h = harness(t)
  h.button(h.render(), '立ち絵').props.onClick()
  const preview = walkAll(h.render(), node => node.type === 'button' && node.props['aria-label'] === '主人公の元の素材を拡大表示')[0]
  assert.ok(preview)
  preview.props.onClick()
  const dialog = walkAll(h.render(), node => node.type?.name === 'Dialog')[0]
  assert.equal(dialog.props.title, '主人公の元の素材')
  assert.equal(dialog.props.children.props.src, '/image/main.png')
  assert.equal(dialog.props.children.props.alt, '主人公の元の素材')
  dialog.props.onClose()
  assert.equal(walkAll(h.render(), node => node.type?.name === 'Dialog').length, 0)
})

test('clearing an edited prompt requires restoring or entering text before generation', t => {
  const h = harness(t)
  h.button(h.render(), '立ち絵').props.onClick()
  let tree = h.render()
  const prompt = tree => walkAll(tree, node => node.type === 'textarea' && ['外見の設定', '声・話し方の設定'].includes(node.props['aria-label']))[0]
  prompt(tree).props.onChange({ target: { value: '   ' } })
  walkAll(tree, node => node.type === 'textarea' && node.props['aria-label'] === '今回だけの変更指示')[0].props.onChange({ target: { value: '柔らかい表情に' } })
  tree = h.render()
  assert.equal(prompt(tree).props['aria-invalid'], true)
  assert.equal(h.button(tree, 'この設定で候補を生成').props.disabled, true)
  assert.ok(text(tree).includes('外見の設定を入力するか、元の設定に戻してください。'))
  h.button(tree, '元の設定に戻す').props.onClick()
  tree = h.render()
  assert.equal(prompt(tree).props.value, '元の服装')
  assert.equal(h.button(tree, 'この設定で候補を生成').props.disabled, false)
  assert.equal(h.calls.length, 0)
})

test('candidate details show original, submitted, instruction, and exact before and after generation prompts', t => {
  const source = adjustmentFixture()
  source.candidates.push({ id: 'generated-image', character_id: 'main', kind: 'image', source: 'generated', artifact_id: 'new-image', url: '/image/new.png', job_id: null,
    prompt_details: { source: '元の服装', input: '編集した服装', instruction: '手を下げる', baseline: 'original actual image prompt', effective: 'edited actual image prompt' } })
  const h = harness(t, source)
  h.button(h.render(), '立ち絵').props.onClick()
  h.button(h.render(), '設定と実際の生成指示').props.onClick()
  const details = walkAll(h.render(), node => node.type?.name === 'CandidatePromptDetails')[0]
  const rendered = details.type(details.props)
  assert.ok(text(rendered).includes('今回：Animaに渡した実際のプロンプト'))
  for (const value of ['元の服装', '編集した服装', '手を下げる', 'original actual image prompt', 'edited actual image prompt']) assert.ok(text(rendered).includes(value))
  const noBaseline = details.type({ kind: 'image', details: { ...details.props.details, baseline: null, effective: null } })
  assert.ok(text(noBaseline).includes('元の素材に使った実際の生成指示は記録されていません。'))
  assert.ok(text(noBaseline).includes('生成が完了すると表示します'))
  assert.ok(!text(noBaseline).includes('original actual image prompt'))
})

test('saved character settings remain visible when an older server omits source_prompts', t => {
  const source = adjustmentFixture()
  for (const person of source.cast) delete person.source_prompts
  const h = harness(t, source)
  for (const [tab, label, expected] of [['立ち絵', '外見の設定', '元の服装'], ['基準音声', '声・話し方の設定', '元の声']]) {
    h.button(h.render(), tab).props.onClick()
    const tree = h.render()
    assert.ok(text(tree).includes(`元の${label}`))
    assert.ok(text(tree).includes(expected))
    assert.equal(walkAll(tree, node => node.type === 'textarea' && node.props['aria-label'] === label)[0].props.value, expected)
    assert.ok(!text(tree).includes('元のプロンプトは記録されていません'))
    assert.equal(h.button(tree, '調整版を反映').props.disabled, false)
  }
})

test('voice transcript is prefilled with self introduction and sent explicitly', async t => {
  const source = adjustmentFixture()
  source.cast[1].result.selfIntroduction = '案内役の自己紹介です。'
  const pending = []
  const h = harness(t, source, { onPendingChange: value => pending.push(value) })
  const reportPending = () => hooks.effects.find(effect => effect.dependencies.length === 3 && effect.dependencies.every(value => typeof value === 'boolean')).callback()
  const transcript = tree => walkAll(tree, node => node.type === 'textarea' && node.props['aria-label'] === '音声に含まれる読み上げ文')[0]
  const person = tree => walkAll(tree, node => node.type === 'select' && ['main', 'guide'].includes(node.props.value))[0]
  h.button(h.render(), '基準音声').props.onClick()
  let tree = h.render(); reportPending()
  assert.equal(transcript(tree).props.value, '元の自己紹介台詞です。')
  assert.equal(pending.at(-1), false, 'prefilled text is not an unsaved edit')
  person(tree).props.onChange({ target: { value: 'guide' } })
  assert.equal(transcript(h.render()).props.value, '案内役の自己紹介です。')
  person(h.render()).props.onChange({ target: { value: 'main' } })
  h.button(h.render(), 'この設定で候補を生成').props.onClick()
  await settle()
  assert.equal(JSON.parse(h.calls[0].options.body).reference_text, '元の自己紹介台詞です。')
  tree = h.render()
  transcript(tree).props.onChange({ target: { value: '読み上げ文を編集しました。' } })
  h.button(h.render(), '立ち絵').props.onClick()
  h.button(h.render(), '基準音声').props.onClick()
  tree = h.render(); reportPending()
  assert.equal(transcript(tree).props.value, '読み上げ文を編集しました。')
  assert.equal(pending.at(-1), true)
  h.button(tree, 'この設定で候補を生成').props.onClick()
  await settle()
  tree = h.render(); reportPending()
  assert.equal(JSON.parse(h.calls[1].options.body).reference_text, '読み上げ文を編集しました。')
  assert.equal(transcript(tree).props.value, '読み上げ文を編集しました。')
  assert.equal(pending.at(-1), false, 'submitted transcript is retained without blocking navigation')
  transcript(tree).props.onChange({ target: { value: '' } })
  tree = h.render(); reportPending()
  assert.equal(transcript(tree).props.value, '', 'intentional clear must not restore default silently')
  assert.equal(pending.at(-1), true)
  assert.equal(h.button(tree, 'この設定で候補を生成').props.disabled, true)
})
