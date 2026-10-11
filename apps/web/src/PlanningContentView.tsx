import { useState } from 'react'
import { Icon } from './Icons'
import { Dialog } from './PreviewComponents'
import type { CastLife, OutfitNames, PlanningAsset, PlanningAssets, PlanningContent, PlanningTab, PlotRoute } from './planningState'
import { MaterialPicker, readySheets, SharedMaterials, useMaterials } from './SharedMaterials'
import type { CharacterBrief } from './wizardState'

const routeLabels: Record<keyof PlotRoute, string> = {
  start_condition: '開始時の状況', attempt: '試み・出来事', consequence: '結果・反応',
  choice: '選択', next_state: '次の章へ残るもの', core_progress: '人物・関係の変化',
}

function ReadingBlock({ title, text }: { title: string; text: string | undefined }) {
  if (!text?.trim()) return null
  return <div className="planning-read-block"><h4>{title}</h4><p>{text}</p></div>
}

function EverydayContext({ life }: { life: CastLife }) {
  return <div className="planning-read-life">
    <ReadingBlock title="本人の関心・普段の用事" text={life.personal_concern}/>
    <ReadingBlock title="物語との接点" text={life.contact}/>
    <ReadingBlock title="知っていること・知らないこと" text={life.initial_knowledge}/>
  </div>
}

export type AssetRetake = (characterId: string, kind: 'image' | 'voice', instruction: string) => Promise<boolean>
const assetNames = { image: '立ち絵', voice: '声' } as const
const working = (asset?: PlanningAsset) => asset?.status === 'pending' || asset?.status === 'running'

/** The portrait and voice of one planned supporting character, made while the plan is reviewed. */
function CharacterAssets({ name, image, voice, disabled, onRetake, onExpand, onDress }: {
  name: string; image?: PlanningAsset; voice?: PlanningAsset; disabled: boolean
  onRetake?: (kind: 'image' | 'voice') => void; onExpand: (url: string) => void; onDress?: () => void
}) {
  const note = (kind: 'image' | 'voice', asset?: PlanningAsset) => working(asset) ? `${assetNames[kind]}を生成中…`
    : asset?.status === 'failed' ? asset.error || `${assetNames[kind]}を生成できませんでした。`
    : asset?.status === 'omitted' ? '人型ではないため、立ち絵は使いません。' : `${assetNames[kind]}はまだありません。`
  const retake = (kind: 'image' | 'voice', asset?: PlanningAsset) => onRetake && asset?.status !== 'omitted'
    && <button type="button" className="button button-light" disabled={disabled || working(asset)} onClick={() => onRetake(kind)}><Icon name="refresh" size={14}/>{asset?.status === 'ready' ? `${assetNames[kind]}を作り直す` : `${assetNames[kind]}を作る`}</button>
  return <div className="planning-read-assets">
    <div className="planning-read-asset">{image?.url
      ? <button type="button" className="planning-read-portrait" aria-label={`${name}の立ち絵を拡大`} onClick={() => onExpand(image.url!)}><img src={image.url} alt={`${name}の立ち絵`}/></button>
      : <p className="planning-read-asset-note" role="status">{note('image', image)}</p>}{image?.url && working(image) && <p className="planning-read-asset-note" role="status">立ち絵を作り直し中…</p>}{retake('image', image)}
      {onDress && image?.status === 'ready' && <button type="button" className="button button-light" disabled={disabled} onClick={onDress}><Icon name="users" size={14}/>共通の素材を着せる</button>}</div>
    <div className="planning-read-asset">{voice?.url
      ? <audio controls preload="metadata" src={voice.url} aria-label={`${name}の声`}>お使いのブラウザーは音声再生に対応していません。</audio>
      : <p className="planning-read-asset-note" role="status">{note('voice', voice)}</p>}{retake('voice', voice)}</div>
  </div>
}

export function PlanningContentView({ content, mainCharacters, tab, outfitNames = {}, assets, assetsDisabled = false, onRetakeAsset, materialsProjectId = null, onDressAsset }: {
  content: PlanningContent; mainCharacters: CharacterBrief[]; tab: PlanningTab; outfitNames?: OutfitNames
  assets?: PlanningAssets; assetsDisabled?: boolean; onRetakeAsset?: AssetRetake
  /** Shared designs of the work, and how a planned character is redrawn wearing one. */
  materialsProjectId?: string | null; onDressAsset?: (characterId: string, materialImageId: string) => Promise<boolean>
}) {
  const store = useMaterials(tab === 'cast' ? materialsProjectId : null)
  const materials = materialsProjectId === null ? undefined : store
  const [dressing, setDressing] = useState<{ id: string; name: string } | null>(null)
  const [portrait, setPortrait] = useState<{ name: string; url: string } | null>(null)
  const [retake, setRetake] = useState<{ id: string; name: string; kind: 'image' | 'voice' } | null>(null)
  const [instruction, setInstruction] = useState('')
  const { cast_plan: cast, plot } = content
  const names = new Map([...mainCharacters, ...cast.supporting_characters].map(character => [character.id, character.name]))
  const name = (id: string) => names.get(id) || id

  if (tab === 'cast') return <div className="planning-read planning-read-cast">
    <div className="planning-read-introduction"><p>登場予定のサブキャラ {cast.supporting_characters.length}人</p><span>人物像、外見、声、日常と他の人物との関わりを確認できます。{assets && '立ち絵と声はここで作り、本編でそのまま使います。'}</span></div>
    {materials && assets && <SharedMaterials store={materials} disabled={assetsDisabled} people={[
      ...mainCharacters.map(character => ({ id: character.id, name: character.name })),
      ...cast.supporting_characters.filter(character => assets[character.id]?.image?.status === 'ready').map(character => ({ id: character.id, name: character.name, planned: true })),
    ]}/>}
    {cast.supporting_characters.map((character, index) => <article className="planning-card planning-read-character" key={character.id}>
      <header className="planning-read-character-header"><span className="planning-read-number">{String(index + 1).padStart(2, '0')}</span><div><h3>{character.name}</h3><p className="planning-read-meta">{[
        character.age, character.gender,
        character.body_type === 'humanoid' ? '人型' : character.body_type === 'nonhumanoid' ? '人型以外' : '',
        character.height_cm != null ? `${character.height_cm} cm` : '',
      ].filter(Boolean).join(' · ')}</p></div></header>
      <p className="planning-read-role">{character.role}</p>
      {assets && <CharacterAssets name={character.name} image={assets[character.id]?.image} voice={assets[character.id]?.voice} disabled={assetsDisabled}
        onRetake={onRetakeAsset && (kind => { setInstruction(''); setRetake({ id: character.id, name: character.name, kind }) })} onExpand={url => setPortrait({ name: character.name, url })}
        onDress={materials && onDressAsset && materials.canEdit && readySheets(materials.materials).length ? () => setDressing({ id: character.id, name: character.name }) : undefined}/>}
      <ReadingBlock title="人物の設定" text={character.settings}/>
      <ReadingBlock title="補足設定" text={character.freeform}/>
      <div className="planning-read-columns"><ReadingBlock title="外見" text={character.appearance}/><ReadingBlock title="声・話し方" text={character.voice}/></div>
      {cast.everyday_context.filter(life => life.character_id === character.id).map(life => <section className="planning-read-section" key={life.character_id}><h4>日常と物語との接点</h4><EverydayContext life={life}/></section>)}
      {character.selfIntroduction?.trim() && <section className="planning-read-section"><h4>自己紹介</h4><blockquote className="planning-read-quote">{character.selfIntroduction}</blockquote></section>}
      {Boolean(character.sampleLines?.length) && <section className="planning-read-section"><h4>代表的な台詞</h4><ul className="planning-read-lines">{character.sampleLines?.map((line, lineIndex) => <li key={lineIndex}>「{line}」</li>)}</ul></section>}
    </article>)}
    {!cast.supporting_characters.length && <div className="planning-read-empty"><p>登場予定のサブキャラは設定されていません。</p><span>追加したい場合は、AIへの修正指示で伝えられます。</span></div>}
    <section className="planning-card planning-read-relationships"><h3>サブキャラを含む関係性</h3>{cast.connections.length ? <dl>{cast.connections.map((connection, index) => <div key={index}><dt>{connection.character_ids.map(name).join(' と ')}</dt><dd>{connection.relationship}</dd></div>)}</dl> : <p className="planning-read-muted">サブキャラとの関係性は設定されていません。</p>}</section>
    {portrait && <Dialog title={`${portrait.name}の立ち絵`} className="char-portrait-dialog" onClose={() => setPortrait(null)}><img src={portrait.url} alt={`${portrait.name}の立ち絵（拡大）`}/></Dialog>}
    {dressing && materials && onDressAsset && <MaterialPicker title={`${dressing.name}に共通の素材を着せる`} materials={materials.materials} busy={assetsDisabled} onClose={() => setDressing(null)} onPick={imageId => onDressAsset(dressing.id, imageId)}/>}
    {retake && onRetakeAsset && <Dialog title={`${retake.name}の${assetNames[retake.kind]}を作り直す`} onClose={() => setRetake(null)}>
      <p className="dialog-intro">人物の設定はそのままで、{assetNames[retake.kind]}だけを作り直します。設定から変えたい場合は、直接編集かAIへの修正指示を使ってください。</p>
      <label className="field-label">作り直しへの追加の希望<span className="char-optional">任意</span><textarea autoFocus rows={3} maxLength={2000} value={instruction} onChange={event => setInstruction(event.target.value)} placeholder={retake.kind === 'image' ? '例：もう少し柔らかい表情で。' : '例：もう少し落ち着いた話し方で。'}/></label>
      <div className="dialog-actions"><button type="button" className="button button-light" onClick={() => setRetake(null)}>キャンセル</button><button type="button" className="button button-primary" disabled={assetsDisabled} onClick={() => void onRetakeAsset(retake.id, retake.kind, instruction.trim()).then(done => { if (done) setRetake(null) })}><Icon name="refresh" size={15}/>作り直す</button></div>
    </Dialog>}
  </div>

  return <div className="planning-read planning-read-plot">
    <section className="planning-card planning-read-core"><span className="planning-read-kicker">物語全体のプロット</span><h3>{plot.core.central_question}</h3>
      <div className="planning-read-columns"><ReadingBlock title="出来事の決着" text={plot.core.external_resolution}/><ReadingBlock title="人物・関係の結末" text={plot.core.relationship_resolution}/></div>
    </section>
    <section className="planning-card"><h3>人物の変化</h3><div className="planning-read-arcs">{plot.core.characters.map(character => <article className="planning-read-arc" key={character.character_id}><h4>{name(character.character_id)}</h4>
      <div className="planning-read-change"><div><span>開始時の行動</span><p>{character.initial_behavior}</p></div><span className="planning-read-arrow" aria-hidden="true">→</span><div><span>最後に選ぶ行動</span><p>{character.final_behavior}</p></div></div>
      <ReadingBlock title="守りたい価値観" text={character.enduring_value}/><ReadingBlock title="転機となる経験" text={character.turning_experience}/>
    </article>)}</div></section>
    <section className="planning-read-chapters" aria-label="章ごとのプロット"><h3>章ごとのプロット <span>全{plot.chapters.length}章</span></h3>
      {plot.chapters.length > 1 && <nav className="planning-read-chapter-nav" aria-label="章へ移動"><ol>{plot.chapters.map(chapter => <li key={chapter.number}><a href={`#planning-chapter-${chapter.number}`}><span>第{chapter.number}章</span>{chapter.title}</a></li>)}</ol></nav>}
      {plot.chapters.map(chapter => <article className="planning-card planning-read-chapter" id={`planning-chapter-${chapter.number}`} key={chapter.number}>
        <header><span className="planning-read-kicker">第{chapter.number}章</span><h3>{chapter.title}</h3><p className="planning-read-role">{chapter.role}</p></header>
        {chapter.events?.length ? <div className="planning-read-events">{chapter.events.map((event, eventIndex) => <section className="planning-read-event" key={eventIndex}><h4>出来事 {eventIndex + 1}</h4><p className="planning-read-event-start">{event.start_condition}</p><ol className="planning-read-timeline">{event.steps.map((step, stepIndex) => <li key={stepIndex}><span className="planning-read-person">{name(step.character_id)}</span><p>{step.action}</p><div className="planning-read-result"><span aria-hidden="true">→</span><p>{step.result}</p></div></li>)}</ol></section>)}</div> : <div className="planning-read-route">{(Object.keys(routeLabels) as (keyof PlotRoute)[]).map(key => <ReadingBlock title={routeLabels[key]} text={chapter.route[key]} key={key}/>)}</div>}
        {Boolean(chapter.conversation_topics?.length) && <section className="planning-read-section"><h4>会話の材料</h4><div className="planning-read-topics">{chapter.conversation_topics?.map((topic, index) => <article key={index}><span>{topic.character_ids.map(name).join('・')}</span><h5>{topic.topic}</h5><p>{topic.exchange}</p></article>)}</div></section>}
        {Boolean(chapter.outfit_appearances?.length) && <section className="planning-read-section"><h4>この章で登場する別バージョン</h4><div className="planning-read-topics">{chapter.outfit_appearances?.map((look, index) => <article key={index}><span>{name(look.character_id)}</span><h5>{outfitNames[look.character_id]?.[look.outfit_id] || look.outfit_id}</h5><p>{look.occasion}</p></article>)}</div></section>}
      </article>)}
    </section>
    <section className="planning-card"><h3>伏線 <span>{plot.core.foreshadowing.length}件</span></h3>{plot.core.foreshadowing.length ? <ul className="planning-read-foreshadowing">{plot.core.foreshadowing.map((item, index) => <li key={index}><div><span>第{item.setup_chapter}章で提示</span><span aria-hidden="true">→</span><span>第{item.payoff_chapter}章で回収</span></div><p>{item.detail}</p></li>)}</ul> : <p className="planning-read-muted">伏線は設定されていません。</p>}</section>
  </div>
}
