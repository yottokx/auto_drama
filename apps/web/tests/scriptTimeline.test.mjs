import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement } from 'react'
import { transformWithOxc } from 'vite'

const dataUrl = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`
async function moduleUrl(file, replacements = {}) {
  let source = await readFile(new URL(`../src/${file}`, import.meta.url), 'utf8')
  for (const [name, url] of Object.entries(replacements)) source = source.replaceAll(`'${name}'`, JSON.stringify(url))
  const { code } = await transformWithOxc(source, file, { target: 'es2022', jsx: { runtime: 'automatic' } })
  return dataUrl(code.replace('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime')))
    .replaceAll('"react"', JSON.stringify(import.meta.resolve('react'))))
}
const stateUrl = await moduleUrl('scriptTimelineState.ts', { './expressionState': await moduleUrl('expressionState.ts') })
const state = await import(stateUrl)
const apiUrl = dataUrl('export const request = () => new Promise(() => {}); export const errorMessage = error => error.message;')
const iconsUrl = dataUrl('export function Icon() { return null }')
const { ScriptStagePreview } = await import(await moduleUrl('ScriptTimeline.tsx', { './api': apiUrl, './Icons': iconsUrl, './scriptTimelineState': stateUrl }))

const direction = (kind, fields = {}) => ({ id: kind, kind, timing: 'before', editable: ['expression', 'pause', 'blackout'].includes(kind), shown: true, ...fields })
const line = (id, scene, fields = {}) => ({
  id, scene_id: scene, speaker_id: null, speaker_name: '', display_text: '本文', spoken_text: '本文', voice_emotion: 'neutral', delivery: '',
  audio_url: null, remote: null, music: null, music_title: null, event_cg: null, transition: null, directions: [],
  stage: { background_url: '/bg', characters: [], lit: null, event_cg_url: null, remote_frame: null }, ...fields,
})

test('directions read as the mock named them, with the timing apart', () => {
  assert.equal(state.directionLabel(direction('expression', { character_name: 'ナギ', expression: 'anxious' })), '表情 ナギ：不安')
  assert.equal(state.directionLabel(direction('expression', { character_name: 'ナギ', expression: 'neutral' })), '表情 ナギ：通常')
  assert.equal(state.directionLabel(direction('enter', { character_name: 'ソウ', position: 'right' })), '登場 ソウ（右）')
  assert.equal(state.directionLabel(direction('position', { character_name: 'ナギ', position: 'left' })), '移動 ナギ → 左')
  assert.equal(state.directionLabel(direction('pause', { duration_ms: 600 })), '間 0.6秒')
  assert.equal(state.directionLabel(direction('blackout', { duration_ms: 800 })), '暗転 0.8秒')
  assert.equal(state.directionLabel(direction('background', { place: '灯台の観測室' })), '背景 灯台の観測室')
  assert.equal(state.directionLabel(direction('outfit', { character_name: 'ナギ', outfit_name: '制服' })), '衣装 ナギ：制服')
  assert.deepEqual(state.TIMING_NAMES, { before: '前', start: '行頭', after: '後' })
})

test('entrances and exits at one moment of a line share one chip each', () => {
  const rows = [
    direction('background', { id: 'bg', place: '観測室' }),
    direction('exit', { id: 'x1', character_name: 'ナギ' }), direction('exit', { id: 'x2', character_name: 'ソウ' }),
    direction('enter', { id: 'e1', character_name: 'ナギ', position: 'left' }), direction('enter', { id: 'e2', character_name: 'ソウ', position: 'right' }),
    direction('expression', { id: 'f', character_name: 'ナギ', expression: 'smile', shown: false }),
    direction('exit', { id: 'x3', character_name: 'ナギ', timing: 'after' }),
  ]
  assert.deepEqual(state.directionChips(rows).map(chip => [chip.label, chip.ids, chip.timing, chip.editable, chip.shown]), [
    ['背景 観測室', ['bg'], 'before', false, true],
    ['退場 ナギ・ソウ', ['x1', 'x2'], 'before', false, true],
    ['登場 ナギ（左）・ソウ（右）', ['e1', 'e2'], 'before', false, true],
    ['表情 ナギ：微笑み', ['f'], 'before', true, false],
    // Another moment of the line is a chip of its own.
    ['退場 ナギ', ['x3'], 'after', false, true],
  ])
})

test('long names are cut in chips, and kept whole for reading and the details', () => {
  assert.equal(state.shortName('ナギ'), 'ナギ')
  assert.equal(state.shortName('アラリック・フォン・レイヴンシュタイン'), 'アラリック…')
  assert.equal(state.shortName('イゾルデ・フォン・スターリング'), 'イゾルデ…')
  assert.equal(state.shortName('名もなき兵士たち'), '名もなき兵士…')
  const [chip] = state.directionChips([
    direction('enter', { id: 'e1', character_name: 'アラリック・フォン・レイヴンシュタイン', position: 'left' }),
    direction('enter', { id: 'e2', character_name: 'ガレス', position: 'right' })])
  assert.equal(chip.label, '登場 アラリック…（左）・ガレス（右）')
  assert.equal(chip.full_label, '登場 アラリック・フォン・レイヴンシュタイン（左）・ガレス（右）')
})

test('voices from elsewhere, scene changes and music are named briefly', () => {
  assert.equal(state.remoteLabel({ source: 'remote', channel: 'radio' }), '無線')
  assert.equal(state.remoteLabel({ source: 'recollection' }), '回想')
  assert.equal(state.remoteLabel(null), '')
  assert.equal(state.transitionLabel({ visual: 'fade', duration_ms: 500 }), '場面転換 暗転 0.5秒')
  assert.equal(state.transitionLabel({ visual: 'cut', duration_ms: 0 }), '場面転換 即時切り替え')
  assert.equal(state.transitionLabel({ visual: 'none', duration_ms: 0 }), '')
  assert.deepEqual(['play', 'continue', 'stop'].map(state.sceneMusicLabel), ['BGMを再生', 'BGMを継続', 'BGMなし'])
})

test('a long text is estimated to split into pages, and sooner beside a voice frame', () => {
  assert.equal(state.estimatedPages('あ'.repeat(102), false), 1)
  assert.equal(state.estimatedPages('あ'.repeat(103), false), 2)
  assert.equal(state.estimatedPages('あ'.repeat(90), true), 2)
  assert.equal(state.estimatedPages('一\n二\n三\n四', false), 2)
})

test('the search keeps each scene and the line numbers of the whole chapter', () => {
  const chapter = { production_id: 'p', chapter_number: 1, stage: { width: 960, height: 640 },
    scenes: [{ scene_id: 's1', title: 'シーン1', first_utterance_id: 'a' }, { scene_id: 's2', title: 'シーン2', first_utterance_id: 'c' }],
    lines: [line('a', 's1', { display_text: '灯台の光' }), line('b', 's1', { speaker_id: 'x', speaker_name: 'ナギ', display_text: 'え？' }), line('c', 's2')] }
  const rows = state.sceneRows(chapter, 'ナギ')
  assert.deepEqual(rows.map(row => row.lines.map(item => [item.line.id, item.number])), [[['b', 2]], []])
  assert.deepEqual(state.sceneRows(chapter, ' ').map(row => row.lines.length), [2, 1])
})

test('the preview stands portraits where the compiler put them and dims all but the lit one', () => {
  const chapter = { stage: { width: 960, height: 640 } }
  const shown = line('a', 's1', { speaker_id: 'x', speaker_name: 'ナギ', stage: { background_url: '/bg', lit: 'x', event_cg_url: null, remote_frame: null,
    characters: [{ id: 'x', image_url: '/x', left: 192, top: 64, width: 480, height: 640 }, { id: 'y', image_url: '/y', left: 480, top: 0, width: 480, height: 640 }] } })
  const markup = renderToStaticMarkup(createElement(ScriptStagePreview, { chapter, line: shown, number: 1 }))
  assert.match(markup, /src="\/x" alt="" style="left:20%;top:10%;width:50%;height:100%"/)
  assert.match(markup, /class="script-stage-portrait is-dim" src="\/y"/)
  assert.doesNotMatch(markup, /class="script-stage-portrait is-dim" src="\/x"/)
  // An event CG covers the stage: no background or portraits show through it.
  const cg = renderToStaticMarkup(createElement(ScriptStagePreview, { chapter, line: { ...shown, stage: { ...shown.stage, event_cg_url: '/cg' } }, number: 1 }))
  assert.match(cg, /src="\/cg"/)
  assert.doesNotMatch(cg, /src="\/x"|src="\/bg"/)
})
