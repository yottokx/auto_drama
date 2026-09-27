import { Icon } from './Icons'
import type { CharacterBrief } from './wizardState'
import type { RelationshipInput, Relationships } from './m2Api'

export function relationshipPairs(characters: CharacterBrief[], inputs: RelationshipInput[]): RelationshipInput[] {
  return characters.flatMap((first, index) => characters.slice(index + 1).map(second => {
    const characterIds = [first.id, second.id].sort() as [string, string]
    const existing = inputs.find(input => input.characterIds.includes(first.id) && input.characterIds.includes(second.id))
    return { characterIds, instruction: existing?.instruction ?? '' }
  }))
}

function nameFor(id: string, characters: CharacterBrief[]) {
  const index = characters.findIndex(character => character.id === id)
  return characters[index]?.name.trim() || `キャラクター ${String(index + 1).padStart(2, '0')}`
}

function PairInputs({ characters, values, onChange, busy = false }: {
  characters: CharacterBrief[]; values: RelationshipInput[]; onChange: (values: RelationshipInput[]) => void; busy?: boolean
}) {
  return <div className="cast-pair-inputs">{values.map((pair, index) => <label key={pair.characterIds.join(':')} className="field-label">
    <span>{nameFor(pair.characterIds[0], characters)} <span className="cast-pair-separator">↔</span> {nameFor(pair.characterIds[1], characters)}</span><span className="char-optional">任意</span>
    <textarea rows={3} disabled={busy} value={pair.instruction} placeholder="例：昔からの友人。互いに信頼しているが、旅の目的だけは話していない。空欄はAIにおまかせ。" onChange={event => onChange(values.map((value, pairIndex) => pairIndex === index ? { ...value, instruction: event.target.value } : value))}/>
  </label>)}</div>
}

export function RelationshipInputs({ characters, values, onChange }: {
  characters: CharacterBrief[]; values: RelationshipInput[]; onChange: (values: RelationshipInput[]) => void
}) {
  if (characters.length < 2) return null
  return <section className="cast-relationship-input char-input-sheet" aria-label="キャラクター同士の関係性を指示">
    <div className="char-sheet-heading"><div><span className="char-kicker">RELATIONSHIPS</span><h2>この人たちを、つなぐもの。</h2><p>人物同士の関係を自由に指定できます。指示がない組み合わせも、世界観と人物設定に合わせてAIが設定します。</p></div></div>
    <PairInputs characters={characters} values={values} onChange={onChange}/>
  </section>
}

export function RelationshipReview({ characters, values, relationships, busy, dirty, onChange, onSave, onGenerate, onDiscard }: {
  characters: CharacterBrief[]; values: RelationshipInput[]; relationships?: Relationships; busy: boolean; dirty: boolean
  onChange: (values: RelationshipInput[]) => void; onSave: () => void; onGenerate: () => void; onDiscard: () => void
}) {
  if (characters.length < 2) return null
  const results = relationships?.result?.pairs ?? []
  return <section className="cast-relationship-review char-setting-card" aria-label="キャラクター同士の関係性">
    <div className="char-detail-heading"><h3><Icon name="users" size={17}/>キャラクター同士の関係性</h3>{relationships?.version && <span className="cast-result-version">第{relationships.version}版</span>}</div>
    {results.length ? <div className="cast-relationship-results">{results.map(pair => {
      const first = nameFor(pair.characterIds[0], characters); const second = nameFor(pair.characterIds[1], characters)
      return <article key={pair.characterIds.join(':')}><h4>{first} <span>↔</span> {second}</h4><p>{pair.summary}</p><dl><div><dt>{first} → {second}</dt><dd>{pair.firstToSecond}</dd></div><div><dt>{second} → {first}</dt><dd>{pair.secondToFirst}</dd></div></dl></article>
    })}</div> : <p className="char-result-empty">{busy ? '全員の人物設定が揃うと、関係性を生成します。' : '関係性はまだ生成されていません。'}</p>}
    {relationships?.pendingChanges && results.length > 0 && <p className="cast-relation-pending" role="status">表示している関係性には、最新の人物設定・指示がまだ反映されていません。</p>}
    {(!results.length || relationships?.pendingChanges) && <button type="button" className="button button-light cast-regenerate" disabled={busy || characters.some(character => !character.settings.trim())} onClick={onGenerate}><Icon name="refresh" size={14}/>現在の設定で関係性を生成</button>}
    <details className="cast-relationship-edit">
      <summary><Icon name="edit" size={14}/><span>関係性を指示・修正する</span>{dirty && <small>未保存</small>}<Icon name="chevron" size={14}/></summary>
      <p>変更したい関係を言葉で伝えてください。人物単体の設定や立ち絵、サンプル音声は維持されます。</p>
      <PairInputs characters={characters} values={values} onChange={onChange} busy={busy}/>
      <div className="char-edit-actions"><button type="button" className="button button-light" disabled={busy || !dirty} onClick={onDiscard}>変更を取り消す</button><button type="button" className="button button-light" disabled={busy || !dirty} onClick={onSave}>指示を保存</button><button type="button" className="button button-primary" disabled={busy || characters.some(character => !character.settings.trim())} onClick={onGenerate}><Icon name="refresh" size={14}/>指示して関係性を再生成</button></div>
    </details>
  </section>
}
