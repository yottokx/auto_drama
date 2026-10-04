import { editPlanningField, type PlanningContent, type PlanningTab, type PlotRoute } from './planningState'
import type { CharacterBrief } from './wizardState'

type FieldPath = (string | number)[]
const routeLabels: Record<keyof PlotRoute, string> = {
  start_condition: '開始時の状況', attempt: '試み・出来事', consequence: '結果・反応',
  choice: '選択', next_state: '次の章へ残るもの', core_progress: '人物・関係の変化',
}
const characterFields = [
  ['name', '名前'], ['age', '年齢・年齢層'], ['gender', '性別'], ['role', '物語での役割'],
  ['settings', '人物の設定'], ['appearance', '外見'], ['voice', '声・話し方'], ['selfIntroduction', '自己紹介'],
  ['freeform', '補足設定'],
] as const

export function PlanningContentEditor({ content, mainCharacters, disabled, onChange, tab }: {
  content: PlanningContent; mainCharacters: CharacterBrief[]; disabled: boolean; onChange: (content: PlanningContent) => void
  tab?: PlanningTab
}) {
  const cast = [...mainCharacters, ...content.cast_plan.supporting_characters]
  const name = (id: string) => cast.find(character => character.id === id)?.name || id
  const change = (path: FieldPath, value: unknown) => onChange(editPlanningField(content, path, value))
  function field(label: string, path: FieldPath, value: string, rows = 3, maxLength = 30000) {
    return <label className="field-label" key={path.join('.')}>{label}<textarea rows={rows} maxLength={maxLength} value={value} onChange={event => change(path, event.target.value)}/></label>
  }
  function characterSelect(label: string, path: FieldPath, value: string) {
    return <label className="field-label" key={path.join('.')}>{label}<select value={value} onChange={event => change(path, event.target.value)}>{cast.map(character => <option value={character.id} key={character.id}>{character.name}</option>)}</select></label>
  }
  return <fieldset className="planning-content" disabled={disabled}>
    <legend className="planning-sr-only">{tab === 'plot' ? 'プロットを直接編集' : tab === 'cast' ? 'サブキャラを直接編集' : '全体計画を直接編集'}</legend>
    {tab !== 'cast' && <>
    <section className="planning-card" aria-label="物語全体のプロット"><h3>全体プロット</h3>
      {field('物語の中心となる問い', ['plot', 'core', 'central_question'], content.plot.core.central_question)}
      {field('出来事の決着', ['plot', 'core', 'external_resolution'], content.plot.core.external_resolution)}
      {field('人物・関係の結末', ['plot', 'core', 'relationship_resolution'], content.plot.core.relationship_resolution)}
      <details className="planning-detail"><summary>人物の変化</summary><div className="planning-detail-content">{content.plot.core.characters.map((character, index) => <section className="planning-subcard" key={character.character_id}><h4>{name(character.character_id)}</h4>{([
        ['initial_behavior', '開始時の行動'], ['enduring_value', '守りたい価値観'], ['turning_experience', '転機となる経験'], ['final_behavior', '最後に選ぶ行動'],
      ] as const).map(([key, label]) => field(label, ['plot', 'core', 'characters', index, key], character[key]))}</section>)}</div></details>
      <details className="planning-detail"><summary>伏線（{content.plot.core.foreshadowing.length}件）</summary><div className="planning-detail-content">{content.plot.core.foreshadowing.map((item, index) => <section className="planning-subcard" key={index}><div className="planning-field-grid">{(['setup_chapter', 'payoff_chapter'] as const).map(key => <label className="field-label" key={key}>{key === 'setup_chapter' ? '提示する章' : '回収する章'}<select value={item[key]} onChange={event => change(['plot', 'core', 'foreshadowing', index, key], Number(event.target.value))}>{content.plot.chapters.map(chapter => <option value={chapter.number} key={chapter.number}>第{chapter.number}章</option>)}</select></label>)}</div>{field('伏線の内容', ['plot', 'core', 'foreshadowing', index, 'detail'], item.detail)}</section>)}{!content.plot.core.foreshadowing.length && <p>伏線は設定されていません。追加したい場合はAIへの修正指示で伝えられます。</p>}</div></details>
    </section>
    <section className="planning-card" aria-label="章ごとのプロット"><h3>章ごとのプロット</h3><p className="planning-help">章ごとの展開と、人物の行動から生まれる結果を確認できます。</p>{content.plot.chapters.map((chapter, index) => {
      const path = ['plot', 'chapters', index]
      return <details className="planning-detail" key={chapter.number} open={content.plot.chapters.length === 1 || undefined}><summary>第{chapter.number}章　{chapter.title}</summary><div className="planning-detail-content">
        {field('章のタイトル', [...path, 'title'], chapter.title, 1, 200)}
        {field('この章の役割・到達点', [...path, 'role'], chapter.role, 3, 200)}
        {chapter.events?.length ? chapter.events.map((event, eventIndex) => <section className="planning-subcard" key={eventIndex}><h4>出来事 {eventIndex + 1}</h4>{field('開始時の状況', [...path, 'events', eventIndex, 'start_condition'], event.start_condition)}{event.steps.map((step, stepIndex) => <div className="planning-event-step" key={stepIndex}>
          {characterSelect(`行動する人物 ${stepIndex + 1}`, [...path, 'events', eventIndex, 'steps', stepIndex, 'character_id'], step.character_id)}
          {field('行動・選択', [...path, 'events', eventIndex, 'steps', stepIndex, 'action'], step.action)}
          {field('その結果', [...path, 'events', eventIndex, 'steps', stepIndex, 'result'], step.result)}
        </div>)}</section>) : <div className="planning-subcard">{(Object.keys(routeLabels) as (keyof PlotRoute)[]).map(key => field(routeLabels[key], [...path, 'route', key], chapter.route[key]))}</div>}
        {Boolean(chapter.conversation_topics?.length) && <section className="planning-subcard"><h4>会話の材料</h4>{chapter.conversation_topics?.map((topic, topicIndex) => <div className="planning-event-step" key={topicIndex}><p>{topic.character_ids.map(name).join('・')}</p>{field('会話のきっかけ', [...path, 'conversation_topics', topicIndex, 'topic'], topic.topic, 2, 120)}{field('やり取りと反応', [...path, 'conversation_topics', topicIndex, 'exchange'], topic.exchange, 2, 120)}</div>)}</section>}
      </div></details>
    })}</section>
    </>}
    {tab !== 'plot' && <>
    <section className="planning-card" aria-label="登場予定のサブキャラ"><h3>登場予定のサブキャラ <span>{content.cast_plan.supporting_characters.length}人</span></h3><p className="planning-help">ここでは設定を確認します。立ち絵と声は本編の制作時に生成します。</p>{content.cast_plan.supporting_characters.map((character, index) => <details className="planning-detail" key={character.id}><summary>{character.name}　<span>{character.role}</span></summary><div className="planning-detail-content">
      {characterFields.map(([key, label]) => field(label, ['cast_plan', 'supporting_characters', index, key], character[key] ?? '', key === 'name' || key === 'age' || key === 'gender' ? 1 : 3))}
      <div className="planning-field-grid"><label className="field-label">体の形<select value={character.body_type ?? 'unknown'} onChange={event => change(['cast_plan', 'supporting_characters', index, 'body_type'], event.target.value)}><option value="humanoid">人型</option><option value="nonhumanoid">人型以外</option><option value="unknown">未指定</option></select></label><label className="field-label">身長（cm・任意）<input type="number" min={1} max={10000} step={0.1} value={character.height_cm ?? ''} onChange={event => { if (!event.target.value || Number.isFinite(event.target.valueAsNumber)) change(['cast_plan', 'supporting_characters', index, 'height_cm'], event.target.value ? event.target.valueAsNumber : null) }}/></label></div>
      {(character.sampleLines ?? []).map((line, lineIndex) => field(`代表的な台詞 ${lineIndex + 1}`, ['cast_plan', 'supporting_characters', index, 'sampleLines', lineIndex], line, 2, 1000))}
      {content.cast_plan.everyday_context.map((life, lifeIndex) => life.character_id === character.id && <section className="planning-subcard" key={life.character_id}><h4>日常と物語との接点</h4>{field('本人の関心・普段の用事', ['cast_plan', 'everyday_context', lifeIndex, 'personal_concern'], life.personal_concern)}{field('接点になる場所・状況', ['cast_plan', 'everyday_context', lifeIndex, 'contact'], life.contact)}{field('知っていること・知らないこと', ['cast_plan', 'everyday_context', lifeIndex, 'initial_knowledge'], life.initial_knowledge)}</section>)}
    </div></details>)}{!content.cast_plan.supporting_characters.length && <p>登場予定のサブキャラはいません。追加したい場合は全体へのAI修正指示で伝えられます。</p>}</section>
    <section className="planning-card" aria-label="サブキャラを含む関係性"><h3>サブキャラを含む関係性</h3>{content.cast_plan.connections.map((connection, index) => <div className="planning-subcard" key={index}>{field(connection.character_ids.map(name).join(' と '), ['cast_plan', 'connections', index, 'relationship'], connection.relationship)}</div>)}{!content.cast_plan.connections.length && <p>サブキャラとの関係性は設定されていません。</p>}</section>
    </>}
  </fieldset>
}
