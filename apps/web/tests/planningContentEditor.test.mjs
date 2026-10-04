import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { transformWithOxc } from 'vite'

const stateSource = await readFile(new URL('../src/planningState.ts', import.meta.url), 'utf8')
const stateOutput = await transformWithOxc(stateSource, 'planningState.ts', { target: 'es2022' })
const stateUrl = `data:text/javascript;base64,${Buffer.from(stateOutput.code).toString('base64')}`
const editorSource = await readFile(new URL('../src/PlanningContentEditor.tsx', import.meta.url), 'utf8')
const output = await transformWithOxc(editorSource.replace("'./planningState'", JSON.stringify(stateUrl)), 'PlanningContentEditor.tsx', { target: 'es2022', jsx: { runtime: 'automatic' } })
const code = output.code.replace('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime')))
const { PlanningContentEditor } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)

const main = { id: 'main', name: '凪' }
const content = {
  cast_plan: {
    supporting_characters: [{ id: 'guide', name: '航', age: '40代', gender: '男性', role: '町の案内役', settings: '港で暮らす', appearance: '青いコート', voice: '穏やかな話し方', selfIntroduction: '案内しよう', sampleLines: ['こちらだ', 'ようこそ', 'また会おう'], height_cm: 180, body_type: 'humanoid', locked: { settings: false, appearance: false, voice: false } }],
    everyday_context: [{ character_id: 'guide', personal_concern: '船の修理', contact: '港で出会う', initial_knowledge: '手紙の宛先は知らない' }],
    connections: [{ character_ids: ['main', 'guide'], relationship: '初対面から協力する' }],
  },
  plot: {
    core: { central_question: '手紙は届くか', external_resolution: '手紙を届ける', relationship_resolution: '友人になる', characters: [{ character_id: 'main', initial_behavior: '人を避ける', enduring_value: '約束', turning_experience: '案内役と出会う', final_behavior: '自分から会いに行く' }], foreshadowing: [{ setup_chapter: 1, payoff_chapter: 1, detail: '古い灯台' }] },
    chapters: [{ number: 1, title: '港の手紙', role: '手紙を受け取る', route: { start_condition: '古い計算結果', attempt: '未更新の試み', consequence: '結果', choice: '選択', next_state: '次', core_progress: '変化' }, events: [{ start_condition: '港に着く', steps: [{ character_id: 'guide', action: '手紙を渡す', result: '届け先を知る' }] }], conversation_topics: [{ character_ids: ['main', 'guide'], topic: '船の修理', exchange: '工具を借りる' }] }],
  },
}
function walk(node, predicate) {
  if (!node || typeof node !== 'object') return null
  if (Array.isArray(node)) { for (const item of node) { const found = walk(item, predicate); if (found) return found } return null }
  if (predicate(node)) return node
  return walk(node.props?.children, predicate)
}

test('the editor renders human-readable plot, supporting character and relationship fields', () => {
  const markup = renderToStaticMarkup(PlanningContentEditor({ content, mainCharacters: [main], disabled: false, onChange() {} }))
  for (const label of ['全体プロット', '章ごとのプロット', '登場予定のサブキャラ', 'サブキャラを含む関係性', '凪 と 航', '手紙を渡す', '初対面から協力する', '知っていること・知らないこと']) assert.ok(markup.includes(label), label)
  assert.ok(!markup.includes('未更新の試み'), 'event-based plans must not present their stale derived route as editable truth')
  assert.ok(!markup.includes('cast_plan'))
  assert.ok(!markup.includes('<img'))
  assert.ok(!markup.includes('<audio'))
})

test('editing a rendered event field changes only that field and leaves server content intact', () => {
  let changed
  const tree = PlanningContentEditor({ content, mainCharacters: [main], disabled: false, onChange(next) { changed = next } })
  const action = walk(tree, node => node.type === 'textarea' && node.props.value === '手紙を渡す')
  assert.ok(action)
  action.props.onChange({ target: { value: '手紙の届け先を尋ねる' } })
  assert.equal(changed.plot.chapters[0].events[0].steps[0].action, '手紙の届け先を尋ねる')
  assert.equal(content.plot.chapters[0].events[0].steps[0].action, '手紙を渡す')
  assert.deepEqual(changed.cast_plan, content.cast_plan)
})

test('legacy route plans remain editable and generation disables the full fieldset', () => {
  const legacy = structuredClone(content)
  legacy.plot.chapters[0].events = []
  const tree = PlanningContentEditor({ content: legacy, mainCharacters: [main], disabled: true, onChange() {} })
  assert.equal(tree.props.disabled, true)
  const markup = renderToStaticMarkup(tree)
  assert.ok(markup.includes('未更新の試み'))
  assert.ok(markup.includes('次の章へ残るもの'))
})

test('direct editing scopes the visible fields to the active plot or cast tab', () => {
  const props = { content, mainCharacters: [main], disabled: false, onChange() {} }
  const plot = renderToStaticMarkup(PlanningContentEditor({ ...props, tab: 'plot' }))
  assert.ok(plot.includes('章ごとのプロット'))
  assert.ok(plot.includes('手紙を渡す'))
  assert.ok(!plot.includes('登場予定のサブキャラ'))
  assert.ok(!plot.includes('サブキャラを含む関係性'))
  const cast = renderToStaticMarkup(PlanningContentEditor({ ...props, tab: 'cast' }))
  assert.ok(cast.includes('登場予定のサブキャラ'))
  assert.ok(cast.includes('凪 と 航'))
  assert.ok(cast.includes('補足設定'))
  assert.ok(!cast.includes('章ごとのプロット'))
  assert.ok(!cast.includes('手紙を渡す'))
})
