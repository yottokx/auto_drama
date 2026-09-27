import { CharacterInput } from './CharacterSteps'
import { Icon } from './Icons'
import { RelationshipInputs } from './RelationshipSteps'
import { WorldInput } from './WorldSteps'
import type { RelationshipInput } from './m2Api'
import type { CharacterBrief, WorldBrief } from './wizardState'

export function CombinedBriefInput({ world, characters, relationshipInputs, onWorldChange, onCharacterChange, onRelationshipsChange, onAdd, onRemove, onNext, onSave, live = false, hasResults = false }: {
  world: WorldBrief; characters: CharacterBrief[]; relationshipInputs: RelationshipInput[]
  onWorldChange: (patch: Partial<WorldBrief>) => void
  onCharacterChange: (id: string, patch: Partial<Omit<CharacterBrief, 'id' | 'locked'>>) => void
  onRelationshipsChange: (values: RelationshipInput[]) => void
  onAdd: () => string; onRemove: (id: string) => void; onNext: () => void; onSave?: () => void
  live?: boolean; hasResults?: boolean
}) {
  return <form className="combined-brief-form" onSubmit={event => { event.preventDefault(); onNext() }}>
    <div className="combined-brief-intro"><Icon name="leaf" size={20}/><p>世界観とメインキャラクターの希望を、最初にまとめて伝えます。<small>まず世界観を生成し、確定した世界観をもとにキャラクターを生成します。</small></p></div>
    <WorldInput embedded world={world} onChange={onWorldChange}/>
    <section className="combined-character-brief" aria-label="メインキャラクターを指示">
      <div className="combined-section-heading"><span className="world-section-icon"><Icon name="users" size={20}/></span><div><h2>どんな人物の物語にしますか。</h2><p>メインキャラクターは1〜3人。決まっている希望を世界観にも反映します。</p></div></div>
      <CharacterInput embedded characters={characters} onChange={onCharacterChange} onAdd={onAdd} onRemove={onRemove} relationshipInputs={<RelationshipInputs characters={characters} values={relationshipInputs} onChange={onRelationshipsChange}/>}/>
    </section>
    {hasResults && <p className="combined-brief-change-note">世界観・人物・関係性の指示を変更すると、世界観の再生成・再確定が必要です。確定後、変更の影響を受けるキャラクターを作り直します。</p>}
    <div className="combined-brief-actions"><p>{live ? '世界観とキャラクターの指示を一緒に保存します。' : 'このプレビューでは入力を保存します。AI生成は実行されません。'}</p><div>{onSave && <button type="button" className="button button-light" onClick={onSave}>指示を保存</button>}<button type="submit" className="button button-primary">{live ? '世界観を生成する' : '世界観を確認する'}<Icon name="arrow" size={16}/></button></div></div>
  </form>
}
