import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'
import { adjustmentMusicFixture } from './adjustmentFixture.mjs'

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
const hooks = await import(hooksUrl)
const raw = (await readFile(new URL('../src/AdjustmentMusicPreview.tsx', import.meta.url), 'utf8')).replaceAll("'react'", JSON.stringify(hooksUrl))
const { code } = await transformWithOxc(raw, 'AdjustmentMusicPreview.tsx', { target: 'es2022', jsx: { runtime: 'automatic' } })
const { AdjustmentMusicPreview, musicPreviewPosition } = await import(dataUrl(code.replace('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime')))))
const settle = () => new Promise(resolve => setImmediate(resolve))
function walk(node, predicate) {
  if (!node || typeof node !== 'object') return []
  if (Array.isArray(node)) return node.flatMap(child => walk(child, predicate))
  return [...(predicate(node) ? [node] : []), ...walk(node.props?.children, predicate)]
}
function text(node) {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(text).join('')
  return node?.props ? text(node.props.children) : ''
}

function harness(t, candidate = adjustmentMusicFixture().music_candidates[0]) {
  hooks.reset()
  const contexts = [], calls = []
  let decodedDuration = 72
  let decodedSampleRate = 48000
  const oldContext = globalThis.AudioContext, oldFetch = globalThis.fetch
  globalThis.AudioContext = class {
    currentTime = 0
    destination = {}
    sources = []
    closed = false
    constructor() { contexts.push(this) }
    createGain() { return { gain: { value: 0, setValueAtTime(value) { this.value = value } }, connect() {} } }
    createBufferSource() {
      const node = { loop: false, connect(output) { this.output = output }, disconnect() {}, start(...args) { this.started = args }, stop() { this.stopped = true } }
      this.sources.push(node); return node
    }
    async resume() {}
    async decodeAudioData() { return { duration: decodedDuration, sampleRate: decodedSampleRate } }
    async close() { this.closed = true }
  }
  let handler = async () => ({ ok: true, arrayBuffer: async () => new ArrayBuffer(8) })
  globalThis.fetch = async (url, options) => { calls.push({ url, options }); return handler(url, options) }
  let props = { candidate, volume: .35 }, previous = [], closed = false
  function render() {
    hooks.beginRender()
    const tree = AdjustmentMusicPreview(props)
    previous = hooks.effects.map((effect, index) => {
      const old = previous[index]
      if (old && effect.dependencies.length === old.dependencies.length && effect.dependencies.every((value, i) => Object.is(value, old.dependencies[i]))) return old
      old?.cleanup?.()
      return { dependencies: effect.dependencies, cleanup: effect.callback() }
    })
    return tree
  }
  function close() { if (!closed) { closed = true; previous.forEach(effect => effect.cleanup?.()) } }
  t.after(() => { close(); globalThis.AudioContext = oldContext; globalThis.fetch = oldFetch })
  render()
  const button = label => {
    const found = walk(render(), node => node.type === 'button' && text(node) === label)[0]
    assert.ok(found, `missing button ${label}`); return found
  }
  return { render, button, contexts, calls, close, update(next) { props = { ...props, ...next } }, decodeLength(value, sampleRate = 48000) { decodedDuration = value; decodedSampleRate = sampleRate }, handle(next) { handler = next } }
}

test('native Web Audio looping begins at zero and repeats A through B with visible marks', async t => {
  const h = harness(t)
  h.button('冒頭から試聴').props.onClick(); await settle()
  const context = h.contexts[0], source = context.sources[0]
  assert.deepEqual(source.started, [0, 0], 'first playback must include the introduction')
  assert.equal(source.loop, true)
  assert.equal(source.loopStart, 8)
  assert.equal(source.loopEnd, 72)
  assert.equal(h.calls[0].url, '/music/prepared.mp3')
  const tree = h.render()
  assert.ok(text(tree).includes('冒頭0秒からBまで一度再生'))
  assert.ok(text(tree).includes('A〜Bを繰り返します'))
  const marks = walk(tree, node => node.type === 'span' && node.props.className?.includes('adjustment-music-loop-mark'))
  assert.deepEqual(marks.map(text), ['A', 'B'])
  assert.equal(walk(tree, node => node.type === 'input' && node.props.type === 'range')[0].props.max, 72)
  h.button('冒頭から試聴').props.onClick(); await settle()
  assert.equal(context.sources[0].stopped, true)
  assert.deepEqual(context.sources[1].started, [0, 0], 'clicking audition again resets to the introduction')
  assert.equal(h.calls.length, 1, 'decoded audio is reused')
})

test('pause and resume keep the current point after wrapping while volume changes apply live', async t => {
  const h = harness(t)
  h.button('冒頭から試聴').props.onClick(); await settle()
  const context = h.contexts[0]
  context.currentTime = 100
  h.button('一時停止').props.onClick()
  assert.equal(walk(h.render(), node => node.type === 'input' && node.props.type === 'range')[0].props.value, 36)
  h.button('再開').props.onClick(); await settle()
  assert.deepEqual(context.sources[1].started, [0, 36])
  h.update({ volume: .17 }); h.render()
  const output = context.sources[1]
  assert.equal(output.loopStart, 8)
  assert.equal(output.loopEnd, 72)
  assert.equal(output.output.gain.value, .17)
  assert.equal(context.sources[0].stopped, true)
})

test('seeking while paused preserves the requested point for the next resume', async t => {
  const h = harness(t)
  h.button('冒頭から試聴').props.onClick(); await settle()
  h.contexts[0].currentTime = 12
  h.button('一時停止').props.onClick()
  walk(h.render(), node => node.type === 'input' && node.props.type === 'range')[0].props.onChange({ target: { value: '43.2' } })
  assert.equal(h.contexts[0].sources.length, 1, 'paused seeking does not restart playback')
  assert.equal(walk(h.render(), node => node.type === 'input' && node.props.type === 'range')[0].props.value, 43.2)
  h.button('再開').props.onClick(); await settle()
  assert.deepEqual(h.contexts[0].sources[1].started, [0, 43.2])
})

test('changing the audio URL aborts a stale trial and resets position without starting the old track', async t => {
  const h = harness(t)
  let resolve
  h.handle(() => new Promise(done => { resolve = done }))
  h.button('冒頭から試聴').props.onClick(); await settle()
  h.update({ candidate: { ...adjustmentMusicFixture().music_candidates[0], music_url: '/music/revised.mp3' } }); h.render()
  assert.equal(h.calls[0].options.signal.aborted, true)
  resolve({ ok: true, arrayBuffer: async () => new ArrayBuffer(8) }); await settle()
  assert.equal(h.contexts[0].sources.length, 0)
  h.handle(async () => ({ ok: true, arrayBuffer: async () => new ArrayBuffer(8) }))
  h.button('冒頭から試聴').props.onClick(); await settle()
  assert.equal(h.calls.at(-1).url, '/music/revised.mp3')
  assert.deepEqual(h.contexts[0].sources[0].started, [0, 0])
})

test('seeking restarts an active loop at the requested point and the original mode plays the full source', async t => {
  const h = harness(t)
  h.button('冒頭から試聴').props.onClick(); await settle()
  walk(h.render(), node => node.type === 'input' && node.props.type === 'range')[0].props.onChange({ target: { value: '24.5' } }); await settle()
  assert.deepEqual(h.contexts[0].sources[1].started, [0, 24.5])
  walk(h.render(), node => node.type === 'input' && node.props.type === 'checkbox')[0].props.onChange({ target: { checked: true } })
  h.decodeLength(120); h.render()
  h.button('冒頭から試聴').props.onClick(); await settle()
  const node = h.contexts[0].sources.at(-1)
  assert.equal(h.calls.at(-1).url, '/music/original.mp3')
  assert.equal(node.loop, false)
  assert.deepEqual(node.started, [0, 0])
  assert.equal(walk(h.render(), node => node.type === 'input' && node.props.type === 'range')[0].props.max, 120)
})

test('a source-only failed loop remains available for full original audition', async t => {
  const candidate = { ...adjustmentMusicFixture().music_candidates[0], music_url: null, artifact_id: null, loop_start_seconds: null, loop_end_seconds: null, duration_seconds: null }
  const h = harness(t, candidate); h.decodeLength(120)
  h.button('冒頭から試聴').props.onClick(); await settle()
  assert.equal(h.calls[0].url, '/music/original.mp3')
  assert.equal(h.contexts[0].sources[0].loop, false)
  assert.ok(text(h.render()).includes('元音源を冒頭から末尾まで再生'))
})

test('decode length mismatches prevent publishing a misleading trial and display an error', async t => {
  const h = harness(t); h.decodeLength(40)
  h.button('冒頭から試聴').props.onClick(); await settle()
  assert.equal(h.contexts[0].sources.length, 0)
  assert.ok(text(h.render()).includes('ループ区間と音源の長さが一致しません'))
})

test('the observed 48000 Hz endpoint rounding uses the decoded end for loops, position and seeking without changing metadata', async t => {
  const canonicalEnd = 69.47410430839003
  const decodedEnd = 69.47410416666666
  const candidate = { ...adjustmentMusicFixture().music_candidates[0], loop_end_seconds: canonicalEnd, duration_seconds: canonicalEnd }
  const saved = structuredClone(candidate)
  const h = harness(t, candidate); h.decodeLength(decodedEnd, 48000)
  h.button('冒頭から試聴').props.onClick(); await settle()
  const context = h.contexts[0]
  assert.equal(context.sources[0].loopEnd, decodedEnd)
  assert.deepEqual(context.sources[0].started, [0, 0])
  const slider = () => walk(h.render(), node => node.type === 'input' && node.props.type === 'range')[0]
  assert.equal(slider().props.max, decodedEnd)
  context.currentTime = decodedEnd
  h.button('一時停止').props.onClick()
  assert.equal(slider().props.value, candidate.loop_start_seconds, 'the displayed position must wrap at the actual loop end')
  slider().props.onChange({ target: { value: String(canonicalEnd) } })
  assert.equal(slider().props.value, candidate.loop_start_seconds, 'seeking the end uses the same actual boundary')
  h.button('再開').props.onClick(); await settle()
  assert.deepEqual(context.sources[1].started, [0, candidate.loop_start_seconds])
  assert.deepEqual(candidate, saved, 'canonical endpoints remain available for saving')
  assert.ok(!text(h.render()).includes('ループ区間と音源の長さが一致しません'))
})

for (const sampleRate of [44100, 48000, 96000]) {
  test(`an overrun of two decoded samples is accepted at ${sampleRate} Hz`, async t => {
    const h = harness(t)
    const decodedEnd = 72 - 2 / sampleRate
    h.decodeLength(decodedEnd, sampleRate)
    h.button('冒頭から試聴').props.onClick(); await settle()
    assert.equal(h.contexts[0].sources.length, 1)
    assert.equal(h.contexts[0].sources[0].loopEnd, decodedEnd)
    assert.equal(walk(h.render(), node => node.type === 'input' && node.props.type === 'range')[0].props.max, decodedEnd)
  })

  test(`an overrun greater than two decoded samples is rejected at ${sampleRate} Hz`, async t => {
    const h = harness(t); h.decodeLength(72 - 2.5 / sampleRate, sampleRate)
    h.button('冒頭から試聴').props.onClick(); await settle()
    assert.equal(h.contexts[0].sources.length, 0)
    assert.ok(text(h.render()).includes('ループ区間と音源の長さが一致しません'))
  })
}

test('unmount stops playback, closes the audio context, and aborts an in-flight fetch', async t => {
  const h = harness(t)
  let resolve
  h.handle(() => new Promise(done => { resolve = done }))
  h.button('冒頭から試聴').props.onClick(); await settle()
  assert.equal(h.calls.length, 1)
  h.close()
  assert.equal(h.calls[0].options.signal.aborted, true)
  assert.equal(h.contexts[0].closed, true)
  resolve({ ok: true, arrayBuffer: async () => new ArrayBuffer(8) }); await settle()
  assert.equal(h.contexts[0].sources.length, 0)
})

test('position calculation matches intro, repeated loops, pauses and finite original endings', () => {
  assert.equal(musicPreviewPosition(0, 7, 72, 8, 72, true), 7)
  assert.equal(musicPreviewPosition(0, 72, 72, 8, 72, true), 8)
  assert.equal(musicPreviewPosition(0, 136, 72, 8, 72, true), 8)
  assert.equal(musicPreviewPosition(36, 12, 72, 8, 72, true), 48)
  assert.equal(musicPreviewPosition(0, 121, 120, 8, 72, false), 120)
})
