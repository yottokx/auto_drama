import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { transformWithOxc } from 'vite'

const stateSource = await readFile(new URL('../src/productionState.ts', import.meta.url), 'utf8')
const stateOutput = await transformWithOxc(stateSource, 'productionState.ts', { target: 'es2022' })
const stateUrl = `data:text/javascript;base64,${Buffer.from(stateOutput.code).toString('base64')}`
const { llmProgressItems, chapterGenerationProgress, chapterAssetProgress } = await import(stateUrl)
const componentSource = await readFile(new URL('../src/GenerationProgress.tsx', import.meta.url), 'utf8')
const output = await transformWithOxc(componentSource.replace("'./productionState'", JSON.stringify(stateUrl)).replace("import './generationProgress.css'", ''), 'GenerationProgress.tsx', { target: 'es2022', jsx: { runtime: 'automatic' } })
const code = output.code.replace('"react/jsx-runtime"', JSON.stringify(import.meta.resolve('react/jsx-runtime')))
const { GenerationProgress } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)

const items = [
  { id: 'image', label: 'サブキャラの立ち絵', current: 1, total: 3, completed: 0, status: 'running' },
  { id: 'background', label: '背景', current: 0, total: 2, completed: 0, status: 'pending' },
  { id: 'reference', label: 'サブキャラの基準音声', current: 0, total: 0, completed: 0, status: 'skipped' },
  { id: 'dialogue', label: '台詞の音声', current: 3, total: 40, completed: 3, status: 'pending' },
]

test('media counters show adopted completions once without a second count line', () => {
  const markup = renderToStaticMarkup(GenerationProgress({ items, label: '第1章の制作工程' }))
  assert.equal((markup.match(/<li\b/g) ?? []).length, 4)
  assert.ok(markup.includes('aria-label="第1章の制作工程"'))
  assert.ok(markup.includes('サブキャラの立ち絵（0/3）'))
  assert.ok(!markup.includes('完了 0/3'))
  assert.ok(!markup.includes('<small>'))
  assert.ok(markup.includes('背景（0/2）'))
  assert.ok(markup.includes('台詞の音声（3/40）'))
  assert.ok(markup.includes('対象なし'))
  assert.ok(!markup.includes('（0/0）'))
})

test('paused and frozen views retain counts without claiming running work', () => {
  const markup = renderToStaticMarkup(GenerationProgress({ items, label: '制作工程', paused: true }))
  assert.ok(markup.includes('サブキャラの立ち絵（0/3）'))
  assert.ok(markup.includes('停止中'))
  assert.ok(!markup.includes('制作中'))
  assert.ok(!markup.includes('generation-progress-running'))
})

test('sequential dialogue work keeps its active appearance during claim gaps and pauses without changing the count', () => {
  for (const [completed, runningIndex] of [[39, 39], [40, null], [40, 40]]) {
    const chapter = {
      chapter_number: 1, status: 'generating_assets', narrative_artifact_id: 'script',
      jobs: Array.from({ length: 45 }, (_, index) => ({ id: `voice-${index}`, kind: 'm3_voice_clone',
        status: index < completed ? 'completed' : index === runningIndex ? 'running' : 'pending',
        attempt_count: index < completed || index === runningIndex ? 1 : 0 })),
    }
    const row = chapterAssetProgress(chapter).find(item => item.id === 'm3_voice_clone')
    const markup = renderToStaticMarkup(GenerationProgress({ items: [row], label: '音声の制作工程' }))
    assert.ok(markup.includes(`台詞の音声（${completed}/45）`))
    assert.ok(markup.includes('generation-progress-running'))
    assert.ok(markup.includes('制作中'))
    assert.ok(!markup.includes('<small>'))
    const paused = renderToStaticMarkup(GenerationProgress({ items: [row], label: '音声の制作工程', paused: true }))
    assert.ok(paused.includes(`台詞の音声（${completed}/45）`))
    assert.ok(paused.includes('停止中'))
    assert.ok(!paused.includes('generation-progress-running'))
  }
})

test('the initial chapter view shows gray scene and validation waiting rows before its plan is known', () => {
  const chapter = { chapter_number: 1, status: 'writing', narrative_artifact_id: null, build: null,
    jobs: [{ id: 'narrative', kind: 'm3_narrative', status: 'running', attempt_count: 1 }],
  }
  const markup = renderToStaticMarkup(GenerationProgress({ items: chapterGenerationProgress(chapter), label: '第1章の制作工程' }))
  const rows = markup.match(/<li\b[^>]*>[\s\S]*?<\/li>/g) ?? []
  assert.match(rows[0], /generation-progress-running/)
  assert.ok(rows[0].includes('章とシーンの構成'))
  for (const [index, label] of [[1, 'シーンの作成'], [2, '内容・整合性の確認']]) {
    assert.ok(rows[index].includes(label))
    assert.match(rows[index], /generation-progress-pending/)
    assert.ok(rows[index].includes('待機中'))
    assert.ok(!rows[index].includes('STEP'))
  }
})

test('completed and in-progress scene requests render as one STEP row per scene', () => {
  const sceneRows = llmProgressItems({ id: 'narrative', kind: 'm3_narrative', status: 'running', attempt_count: 1, progress: {
    schema_version: 1, sequence: 4, phase: 'chapter', current_step: 'scene2-script', attempt: 1, active: true,
    steps: [
      ...['script', 'speech_extraction', 'staging'].map(stage => ({ id: `scene1-${stage}`, stage, status: 'completed', chapter_number: 1, scene_number: 1, completed: 1, total: 8 })),
      { id: 'scene2-script', stage: 'script', status: 'running', chapter_number: 1, scene_number: 2, completed: 1, total: 8 },
    ],
  } })
  const markup = renderToStaticMarkup(GenerationProgress({ items: sceneRows, label: 'シーンの制作工程' }))
  const text = markup.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ')
  assert.equal((markup.match(/<li\b/g) ?? []).length, 2)
  assert.ok(text.includes('シーン1 STEP 3/3 完了'))
  assert.ok(text.includes('シーン2 STEP 1/3 台本を生成中…'))
  assert.ok(!text.includes('STEP 2/8'), 'step counters must not leak the old chapter scene counts')
  assert.ok(!markup.includes('<small>'), 'scene rows need only the STEP count, not a second material-style count')
})

test('a completed chapter plan renders all waiting scenes and hides absent supporting material groups', () => {
  const current = {
    chapter_number: 1, status: 'writing', narrative_artifact_id: null, build: null,
    jobs: [{ id: 'narrative', kind: 'm3_narrative', status: 'running', attempt_count: 1, progress: {
      schema_version: 1, sequence: 2, phase: 'chapter', current_step: null, attempt: 1, active: true,
      chapter_plan: { chapter_number: 1, scene_count: 3, supporting_character_count: 0 },
      steps: [{ id: 'plan', stage: 'chapter_scene_plan', status: 'completed', chapter_number: 1 }],
    } }],
  }
  const markup = renderToStaticMarkup(GenerationProgress({ items: chapterGenerationProgress(current), label: '第1章の制作工程' }))
  const text = markup.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ')
  assert.equal((markup.match(/<li\b/g) ?? []).length, 8, 'plan, three scenes, validation, background, dialogue, and build occupy one row each')
  assert.match(text, /章とシーンの構成 完了.*シーン1 STEP 0\/3 待機中.*シーン2 STEP 0\/3 待機中.*シーン3 STEP 0\/3 待機中.*内容・整合性の確認 待機中.*背景.*台詞の音声/)
  assert.ok(!text.includes('シーンの作成'))
  assert.ok(!text.includes('サブキャラの立ち絵'))
  assert.ok(!text.includes('サブキャラの基準音声'))
  assert.ok(!text.includes('対象なし'))
  assert.ok(!markup.includes('generation-progress-running'), 'waiting scenes must not look active before script generation begins')

  current.jobs[0].progress.chapter_plan.supporting_character_count = 2
  const withSupporting = renderToStaticMarkup(GenerationProgress({ items: chapterGenerationProgress(current), label: '第1章の制作工程' }))
  assert.ok(withSupporting.includes('サブキャラの立ち絵'))
  assert.ok(withSupporting.includes('サブキャラの基準音声'))
  assert.ok(!withSupporting.includes('（0/0）'))
})

for (const [index, activity] of ['台本を生成中…', '台詞と話者を整理中…', '演出を設定中…'].entries()) {
  test(`scene STEP ${index + 1} activity is visible only while the view is running`, () => {
    const row = { id: 'scene-1-2', label: 'シーン2', current: index + 1, total: 3, completed: index, counter: 'step', status: 'running', statusText: activity }
    const markup = renderToStaticMarkup(GenerationProgress({ items: [row], label: 'シーンの制作工程' }))
    assert.ok(markup.includes(`シーン2 STEP ${index + 1}/3`))
    assert.ok(markup.includes(activity))
    const paused = renderToStaticMarkup(GenerationProgress({ items: [row], label: 'シーンの制作工程', paused: true }))
    assert.ok(paused.includes(`シーン2 STEP ${index + 1}/3`))
    assert.ok(paused.includes('停止中'))
    assert.ok(!paused.includes(activity))
    assert.ok(!paused.includes('generation-progress-running'))
  })
}

test('failed and stale scene steps use the normal status without an in-progress activity label', () => {
  for (const [status, expected] of [['failed', '再試行が必要'], ['pending', '待機中']]) {
    const row = { id: 'scene-1-2', label: 'シーン2', current: 2, total: 3, completed: 1, counter: 'step', status }
    const markup = renderToStaticMarkup(GenerationProgress({ items: [row], label: 'シーンの制作工程' }))
    assert.ok(markup.includes('シーン2 STEP 2/3'))
    assert.ok(markup.includes(expected))
    assert.ok(!markup.includes('生成中'))
    assert.ok(!markup.includes('整理中'))
    assert.ok(!markup.includes('generation-progress-completed'))
  }
})

test('planning and material rows use completed counts while scene STEP numbering stays distinct', () => {
  const rows = [
    { id: 'cast', label: 'サブキャラの設定・関係性', current: 0, total: null, completed: 1, status: 'completed' },
    { id: 'plot', label: '物語の核心・全体プロット', current: 3, total: 5, completed: 2, status: 'running' },
    items[0],
  ]
  const markup = renderToStaticMarkup(GenerationProgress({ items: rows, label: '生成工程' }))
  assert.ok(markup.includes('サブキャラの設定・関係性'))
  assert.ok(markup.includes('物語の核心・全体プロット（2/5）'))
  assert.ok(markup.includes('サブキャラの立ち絵（0/3）'))
  assert.ok(!markup.includes('<small>'))
  assert.ok(!markup.includes('STEP'))
})
