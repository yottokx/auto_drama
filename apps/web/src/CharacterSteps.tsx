import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Icon, type IconName } from './Icons'
import { Dialog, LockButton } from './PreviewComponents'
import type { CharacterBrief, RevisionRequest, RevisionScope } from './wizardState'
import { CharacterVoiceTrials } from './CharacterVoiceTrials'
import { ImageGenerationDetails } from './ImageGenerationDetails'
import type { VoiceTest } from './m2Api'
import './characterSteps.css'

type CharacterPatch = Partial<Omit<CharacterBrief, 'id' | 'locked'>>
type CharacterChange = (id: string, patch: CharacterPatch) => void | boolean | Promise<boolean | void>
type DetailScope = 'settings' | 'appearance' | 'voice'
type EditScope = 'all' | DetailScope
type RetakeScope = 'image-retake' | 'voice-retake'

const scopeNames: Record<RevisionScope, string> = {
  world: '世界観', all: 'キャラクター全体', settings: '設定', appearance: '外見', voice: '声',
  'image-retake': '立ち絵だけリテイク', 'voice-retake': '音声だけリテイク',
}
const scopeDescriptions: Record<EditScope, string> = {
  all: '設定・外見・声をひとつの指示でまとめて調整します。固定中の項目は保護されます。',
  settings: '人物像や背景、価値観など、その人物自身の設定について変更を指示します。',
  appearance: '服装・髪型・表情・体格など、外見のイメージの変更を指示します。',
  voice: '声質・話す速さ・話し方など、声のイメージの変更を指示します。',
}
const scopePlaceholders: Record<EditScope, string> = {
  all: '例：もっと飄々とした人物に。服装は気取らず、声も少し力の抜けた話し方にしてください。',
  settings: '例：周囲には強気に振る舞うけれど、失敗をひとりで抱え込んでしまう人物に。',
  appearance: '例：服装を動きやすいものに。髪は短めで、表情には余裕を感じさせて。',
  voice: '例：落ち着いた低めの声で、語尾はやわらかく。親しい相手には少し早口に。',
}
const editableFields: Record<DetailScope, (keyof CharacterPatch)[]> = {
  settings: ['name', 'age', 'gender', 'role', 'settings', 'selfIntroduction', 'sampleLines'],
  appearance: ['appearance', 'height_cm', 'body_type'],
  voice: ['voice'],
}

function characterName(character: CharacterBrief, index: number) {
  return character.name.trim() || `キャラクター ${String(index + 1).padStart(2, '0')}`
}

function CharacterSelector({ characters, selected, onSelect, onAdd, status }: {
  characters: CharacterBrief[]; selected: string; onSelect: (id: string) => void; onAdd?: () => void; status?: Record<string, string>
}) {
  return <div className="char-selector-row">
    <div className="char-selector" role="group" aria-label="編集するキャラクター">
      {characters.map((character, index) => <button key={character.id} type="button" className={`char-selector-item ${selected === character.id ? 'is-selected' : ''}`} aria-pressed={selected === character.id} onClick={() => onSelect(character.id)}>
        <span className="char-selector-number">{String(index + 1).padStart(2, '0')}</span><span>{characterName(character, index)}</span>{status?.[character.id] && <small className="char-selector-status">{status[character.id]}</small>}
      </button>)}
    </div>
    {onAdd && <button type="button" className="char-add-button" disabled={characters.length >= 3} onClick={onAdd}><Icon name="plus" size={15}/>{characters.length >= 3 ? 'メインキャラは3人まで' : 'キャラクターを追加'}</button>}
  </div>
}

function CharacterBasics({ character, onChange }: { character: CharacterBrief; onChange: CharacterChange }) {
  return <div className="char-basics-grid">
    <label className="field-label">名前<span className="char-optional">任意</span><input value={character.name} disabled={character.locked.settings} placeholder="未指定はAIにおまかせ" onChange={event => onChange(character.id, { name: event.target.value })}/></label>
    <label className="field-label">年齢・年齢層<span className="char-optional">任意</span><input value={character.age} disabled={character.locked.settings} placeholder="例：30代 / 年齢不詳" onChange={event => onChange(character.id, { age: event.target.value })}/></label>
    <label className="field-label">性別<span className="char-optional">任意</span><input value={character.gender} disabled={character.locked.settings} placeholder="自由に指定 / 未指定はおまかせ" onChange={event => onChange(character.id, { gender: event.target.value })}/></label>
    <label className="field-label">物語での役割<span className="char-optional">任意</span><input value={character.role} disabled={character.locked.settings} placeholder="例：主人公 / 語り手 / 案内役" onChange={event => onChange(character.id, { role: event.target.value })}/></label>
  </div>
}

function CharacterDimensions({ character, onChange }: { character: CharacterBrief; onChange: CharacterChange }) {
  return <div className="char-dimensions">
    <div className="char-basics-grid">
      <label className="field-label">身長（cm）<span className="char-optional">任意</span><input type="number" min={1} max={10000} step="any" value={character.height_cm ?? ''} disabled={character.locked.appearance} placeholder="例：170" onChange={event => onChange(character.id, { height_cm: event.target.value === '' ? null : event.target.valueAsNumber })}/><span className="field-hint">表示用の目安です。</span></label>
      <label className="field-label">体の形<select value={character.body_type ?? 'unknown'} disabled={character.locked.appearance} onChange={event => onChange(character.id, { body_type: event.target.value as CharacterBrief['body_type'] })}><option value="unknown">自動</option><option value="humanoid">人型</option><option value="nonhumanoid">人型以外</option></select></label>
    </div>
    <p className="field-hint">人型の立ち絵は、身長に合わせて大きさを調整します。未指定の身長は自動で決まります。</p>
  </div>
}

export function CharacterInput({ characters, onChange, onAdd, onRemove, onNext, onBack, onSave, live = false, relationshipInputs, embedded = false }: {
  characters: CharacterBrief[]; onChange: CharacterChange; onAdd: () => string; onRemove: (id: string) => void; onNext?: () => void; onBack?: () => void; onSave?: () => void; live?: boolean; relationshipInputs?: ReactNode; embedded?: boolean
}) {
  const [selectedId, setSelectedId] = useState(characters[0]?.id ?? '')
  const character = characters.find(item => item.id === selectedId) ?? characters[0]
  if (!character) return null
  const characterIndex = characters.findIndex(item => item.id === character.id)

  return <div className="char-step">
    <CharacterSelector characters={characters} selected={character.id} onSelect={setSelectedId} onAdd={() => setSelectedId(onAdd())}/>
    <section className="char-input-sheet">
      <div className="char-sheet-heading">
        <div><span className="char-kicker">CHARACTER {String(characterIndex + 1).padStart(2, '0')}</span><h2>決まっていることから、少しずつ。</h2><p>すべて任意です。自由記述だけでも、すべておまかせでも進められます。</p></div>
        {characters.length > 1 && <button type="button" className="char-remove-button" onClick={() => { const next = characters.find(item => item.id !== character.id); onRemove(character.id); if (next) setSelectedId(next.id) }} aria-label={`${characterName(character, characterIndex)}を削除`}><Icon name="close" size={14}/>削除</button>}
      </div>
      {character.locked.settings && <p className="char-lock-notice"><Icon name="lock" size={13}/>設定が固定されています。変更するには、キャラクターの確認画面で固定を解除してください。</p>}
      <CharacterBasics character={character} onChange={onChange}/>
      <label className="field-label char-freeform">人物像を、自由に伝える<span className="char-optional">任意</span>
        <textarea rows={5} value={character.freeform} disabled={character.locked.settings} placeholder="性格、背景、譲れないこだわりなど。まとまっていなくても、そのまま書いてください。" onChange={event => onChange(character.id, { freeform: event.target.value })}/>
        <span className="field-hint">名前や年齢を決めずに、人物の雰囲気だけを伝えることもできます。人物同士の関係性は、2人以上の場合に専用の欄で指定できます。</span>
      </label>
      <details className="char-detail-inputs">
        <summary><span><Icon name="settings" size={16}/>設定・外見・声をもう少し指定する</span><span className="char-optional">任意</span><Icon name="chevron" size={14}/></summary>
        <div>
          <label className="field-label">詳しい設定<textarea rows={3} value={character.settings} disabled={character.locked.settings} placeholder="経歴、価値観、得意なこと、弱点など。未指定はAIにおまかせ。" onChange={event => onChange(character.id, { settings: event.target.value })}/></label>
          <label className="field-label">外見・立ち絵のイメージ<textarea rows={3} value={character.appearance} disabled={character.locked.appearance} placeholder="髪型、服装、体格、表情、絵柄など。未指定はAIにおまかせ。" onChange={event => onChange(character.id, { appearance: event.target.value })}/></label>
          <CharacterDimensions character={character} onChange={onChange}/>
          <label className="field-label">声のイメージ<textarea rows={3} value={character.voice} disabled={character.locked.voice} placeholder="声質、話す速さ、口調、演技のニュアンスなど。未指定はAIにおまかせ。" onChange={event => onChange(character.id, { voice: event.target.value })}/></label>
          {(character.locked.appearance || character.locked.voice) && <p className="field-hint">固定された項目は、キャラクターの確認画面で固定を解除すると編集できます。</p>}
        </div>
      </details>
      <div className="char-input-tip"><span><Icon name="leaf" size={20}/></span><p>まだ出会っていない人物の、輪郭だけで大丈夫。<br/><strong>{embedded ? 'まず世界観を確認・確定してから、この指示をもとに設定・外見・声をつくります。' : '次の画面で、設定・外見・声をまとめて確認し、言葉で調整できます。'}</strong></p></div>
    </section>
    {relationshipInputs}
    {!embedded && <div className="char-step-footer"><button type="button" className="button button-light" onClick={onBack}><Icon name="back" size={16}/>世界観の確認に戻る</button>{onSave && <button type="button" className="button button-light" onClick={onSave}>入力を保存</button>}<button type="button" className="button button-primary" onClick={onNext}>{live ? 'キャラクターを生成する' : 'キャラクターの確認へ'}<Icon name="arrow" size={16}/></button></div>}
  </div>
}

export function CharacterReview({ characters, requests, onChange, onToggleLock, onRequest, onBack, onApprove, onPendingChange, previewAssets, live = false, canApprove = true, generating = false, busy = false, relationshipResults, onCloneVoice, characterStatus }: {
  characters: CharacterBrief[]; requests: RevisionRequest[]; onChange: CharacterChange; onToggleLock: (id: string, scope: DetailScope) => void;
  onRequest: (scope: RevisionScope, instruction: string, characterId?: string) => boolean | Promise<boolean>; onBack: () => void; onApprove: () => void; onPendingChange?: (pending: boolean) => void;
  previewAssets?: Record<string, { image?: string | null; imageArtifactId?: string | null; voice?: string | null; voiceArtifactId?: string | null; voiceReferenceText?: string | null; voiceTests?: VoiceTest[] }>
  relationshipResults?: ReactNode; onCloneVoice?: (characterId: string, text: string) => Promise<boolean>; characterStatus?: Record<string, string>
  live?: boolean; canApprove?: boolean; generating?: boolean; busy?: boolean
}) {
  const [selectedId, setSelectedId] = useState(characters[0]?.id ?? '')
  const [scope, setScope] = useState<EditScope>('all')
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [edits, setEdits] = useState<Record<string, CharacterBrief>>({})
  const [revisionOpen, setRevisionOpen] = useState(false)
  const [clonePending, setClonePending] = useState(false)
  const [retake, setRetake] = useState<RetakeScope | null>(null)
  const [retakeInstruction, setRetakeInstruction] = useState('')
  const [savedMessage, setSavedMessage] = useState('')
  const [portrait, setPortrait] = useState<{ name: string; image: string } | null>(null)
  const revisionPanel = useRef<HTMLDetailsElement>(null)
  const instructionInput = useRef<HTMLTextAreaElement>(null)
  const pendingCallback = useRef(onPendingChange)
  const unsavedDrafts = Object.entries(drafts).filter(([key, value]) => value.trim() && characters.some(item => key.startsWith(`${item.id}:`)))
  const unsavedEdits = Object.entries(edits).filter(([key, edit]) => {
    const original = characters.find(item => item.id === edit.id)
    const editScope = key.slice(edit.id.length + 1) as DetailScope
    return original && editableFields[editScope].some(field => JSON.stringify(edit[field]) !== JSON.stringify(original[field]))
  })
  const hasPending = clonePending || unsavedDrafts.length > 0 || unsavedEdits.length > 0 || Boolean(retake && retakeInstruction.trim())
  useEffect(() => { pendingCallback.current = onPendingChange; onPendingChange?.(hasPending) }, [hasPending, onPendingChange])
  useEffect(() => () => pendingCallback.current?.(false), [])
  useEffect(() => { if (revisionOpen) instructionInput.current?.focus({ preventScroll: true }) }, [revisionOpen, scope])
  const character = characters.find(item => item.id === selectedId) ?? characters[0]
  if (!character) return null
  const characterIndex = characters.findIndex(item => item.id === character.id)
  const draftKey = `${character.id}:${scope}`
  const instruction = drafts[draftKey] ?? ''
  const protectedScopes = (['settings', 'appearance', 'voice'] as const).filter(item => character.locked[item])
  const resultUnavailable = live && !character.settings.trim()
  const scopeLocked = busy || resultUnavailable || (scope === 'all' ? protectedScopes.length === 3 : character.locked[scope])
  const characterRequests = requests.filter(item => item.characterId === character.id).slice().reverse()
  const activeScopes = scope === 'all' ? (['settings', 'appearance', 'voice'] as const).filter(item => !character.locked[item]) : character.locked[scope] ? [] : [scope]
  const settingsEdit = edits[`${character.id}:settings`]
  const appearanceEdit = edits[`${character.id}:appearance`]
  const voiceEdit = edits[`${character.id}:voice`]
  const assets = previewAssets?.[character.id]

  function chooseScope(nextScope: EditScope) {
    setScope(nextScope); setRevisionOpen(true); setSavedMessage('')
    revisionPanel.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
  function startEdit(editScope: DetailScope) {
    if (busy || character.locked[editScope]) return
    setEdits(current => ({ ...current, [`${character.id}:${editScope}`]: current[`${character.id}:${editScope}`] ?? { ...character, locked: { ...character.locked } } }))
    setSavedMessage('')
  }
  function updateEdit(editScope: DetailScope, patch: CharacterPatch) {
    const key = `${character.id}:${editScope}`
    setEdits(current => current[key] ? { ...current, [key]: { ...current[key], ...patch } } : current)
  }
  function cancelEdit(editScope: DetailScope) {
    const key = `${character.id}:${editScope}`
    setEdits(current => { const next = { ...current }; delete next[key]; return next })
  }
  async function saveEdit(editScope: DetailScope) {
    const edit = edits[`${character.id}:${editScope}`]
    if (!edit || character.locked[editScope]) return
    const patch: CharacterPatch = {}
    editableFields[editScope].forEach(field => { if (field === 'selfIntroduction' && character.locked.voice) return; if (edit[field] !== undefined) Object.assign(patch, { [field]: edit[field] }) })
    if (await onChange(character.id, patch) === false) return
    cancelEdit(editScope); setSavedMessage(`${scopeNames[editScope]}の変更を保存しました。`)
  }
  function toggleLock(editScope: DetailScope) {
    if (edits[`${character.id}:${editScope}`]) { setSavedMessage('編集中の内容を保存するかキャンセルしてから、固定を変更してください。'); return }
    onToggleLock(character.id, editScope)
  }
  async function saveInstruction() {
    if (!instruction.trim() || scopeLocked) return
    if (await onRequest(scope, instruction.trim(), character.id)) {
      setDrafts(current => ({ ...current, [draftKey]: '' })); setSavedMessage(live ? `${scopeNames[scope]}の再生成を受け付けました。` : `${scopeNames[scope]}への修正指示を保存しました。`)
    }
  }
  async function saveRetake() {
    if (!retake || character.locked[retake === 'image-retake' ? 'appearance' : 'voice']) return
    const text = retakeInstruction.trim() || (retake === 'image-retake' ? '設定・外見のイメージを維持して、立ち絵をリテイクしてください。' : '設定・声のイメージを維持して、音声をリテイクしてください。')
    if (await onRequest(retake, text, character.id)) { setSavedMessage(live ? '素材のリテイクを受け付けました。' : `${scopeNames[retake]}の指示を保存しました。`); setRetake(null); setRetakeInstruction('') }
  }
  function editActions(editScope: DetailScope) {
    return <div className="char-edit-actions"><button type="button" className="button button-light" onClick={() => cancelEdit(editScope)}>キャンセル</button><button type="button" className="button button-primary" disabled={busy || character.locked[editScope]} onClick={() => saveEdit(editScope)}><Icon name="check" size={14}/>{scopeNames[editScope]}の変更を保存</button></div>
  }

  return <div className="char-step">
    <CharacterSelector characters={characters} selected={character.id} status={characterStatus} onSelect={id => { setSelectedId(id); setSavedMessage('') }}/>
    {busy && <p className="char-generating-note" role="status"><Icon name="refresh" size={14}/>生成済みの結果は、キャラクターを切り替えて確認できます。修正は生成完了後に行えます。</p>}
    <div className="char-review-section-heading"><div><span className="char-kicker">CHARACTER {String(characterIndex + 1).padStart(2, '0')}</span><h2>{characterName(character, characterIndex)}</h2><p>人物の設定、外見、声を確認してください。変更したいところだけ修正できます。</p></div></div>
    <section className="char-setting-card">
      <DetailHeader title="人物の設定" icon="book" locked={character.locked.settings} onToggle={() => toggleLock('settings')} onEdit={() => startEdit('settings')} editing={Boolean(settingsEdit)} disabled={busy || resultUnavailable}/>
      {settingsEdit ? <fieldset disabled={busy} className="char-inline-editor" aria-label="人物の設定を編集">
        <CharacterBasics character={{ ...settingsEdit, locked: character.locked }} onChange={(_id, patch) => updateEdit('settings', patch)}/>
        <label className="field-label">人物の設定<textarea autoFocus rows={6} value={settingsEdit.settings} disabled={character.locked.settings} onChange={event => updateEdit('settings', { settings: event.target.value })}/></label>
        <label className="field-label">自己紹介の台詞<textarea rows={3} value={settingsEdit.selfIntroduction ?? ''} disabled={character.locked.settings || character.locked.voice} onChange={event => updateEdit('settings', { selfIntroduction: event.target.value })}/>{character.locked.voice && <span className="field-hint">声を固定している間は、サンプル音声の自己紹介も維持します。</span>}</label>
        <label className="field-label">代表的な台詞<span className="field-hint">1行に1つずつ、3つ入力してください。</span><textarea rows={4} value={(settingsEdit.sampleLines ?? []).join('\n')} disabled={character.locked.settings} onChange={event => updateEdit('settings', { sampleLines: event.target.value.split('\n') })}/></label>
        {editActions('settings')}
      </fieldset> : <>
        <dl className="char-result-basics">{([['名前', character.name], ['年齢・年齢層', character.age], ['性別', character.gender], ['物語での役割', character.role]] as const).map(([label, value]) => <div key={label}><dt>{label}</dt><dd className={value.trim() ? '' : 'char-result-empty'}>{value.trim() || '未生成'}</dd></div>)}</dl>
        <ResultText text={character.settings}/>
        <div className="char-speech-results"><section><h4>自己紹介</h4>{character.selfIntroduction?.trim() ? <blockquote>{character.selfIntroduction}</blockquote> : <p className="char-result-empty">未生成です。キャラクター全体を再生成すると、自己紹介と代表台詞が追加されます。</p>}</section><section><h4>代表的な台詞</h4>{character.sampleLines?.length ? <ul>{character.sampleLines.map((line, lineIndex) => <li key={lineIndex}><blockquote>{line}</blockquote></li>)}</ul> : <p className="char-result-empty">未生成</p>}</section></div>
      </>}
      {character.freeform.trim() && <details className="char-input-reference"><summary>入力した指示<Icon name="chevron" size={13}/></summary><p>{character.freeform}</p></details>}
    </section>

    <div className="char-assets-grid">
      <section className="char-asset-card">
        <DetailHeader title="外見・立ち絵" icon="users" locked={character.locked.appearance} onToggle={() => toggleLock('appearance')} onEdit={() => startEdit('appearance')} editing={Boolean(appearanceEdit)} disabled={busy || resultUnavailable}/>
        {assets?.image ? <figure className="char-result-portrait"><button type="button" className="char-portrait-expand" aria-label={`${characterName(character, characterIndex)}の立ち絵を拡大`} onClick={() => setPortrait({ name: characterName(character, characterIndex), image: assets.image! })}><img src={assets.image} alt={`${characterName(character, characterIndex)}の${live ? '立ち絵' : '立ち絵の参考素材'}`}/><span className="char-expand-hint">クリックで拡大</span></button><figcaption>{live ? '生成した立ち絵' : '立ち絵の参考素材'}</figcaption></figure> : <div className="char-image-placeholder" aria-label={generating ? '立ち絵は生成待ち・生成中' : '立ち絵は未生成'}><div className="char-portrait-outline"><Icon name="users" size={47}/></div><span>{generating ? '立ち絵は生成が完了すると表示されます' : '立ち絵はまだ生成されていません'}</span><small>{generating ? '生成待ち・生成中' : '未生成'}</small></div>}
        {assets?.image && assets.imageArtifactId && <ImageGenerationDetails key={assets.imageArtifactId} artifactId={assets.imageArtifactId}/>}
        {appearanceEdit ? <fieldset disabled={busy} className="char-inline-editor" aria-label="外見を編集"><label className="field-label">外見の設定<textarea autoFocus rows={5} value={appearanceEdit.appearance} disabled={character.locked.appearance} onChange={event => updateEdit('appearance', { appearance: event.target.value })}/></label><CharacterDimensions character={{ ...appearanceEdit, locked: character.locked }} onChange={(_id, patch) => updateEdit('appearance', patch)}/>{editActions('appearance')}</fieldset> : <><ResultText title="外見の設定" text={character.appearance}/>{character.height_cm != null && <p className="char-height-summary">身長 {character.height_cm} cm<span>表示用の目安</span></p>}</>}
        <div className="char-asset-actions"><span>素材の作り直し</span><button type="button" className="char-retake-button" disabled={busy || character.locked.appearance || resultUnavailable || unsavedEdits.length > 0} onClick={() => { setRetake('image-retake'); setRetakeInstruction('') }}><Icon name="refresh" size={14}/>立ち絵だけリテイク</button></div>
        <p className="char-retake-hint">人物の設定と外見を維持して、立ち絵だけ作り直します。</p>
      </section>
      <section className="char-asset-card">
        <DetailHeader title="声・話し方" icon="volume" locked={character.locked.voice} onToggle={() => toggleLock('voice')} onEdit={() => startEdit('voice')} editing={Boolean(voiceEdit)} disabled={busy || resultUnavailable}/>
        {assets?.voice ? <div className="char-result-audio"><div className="char-placeholder-wave" aria-hidden="true">{[12, 23, 34, 18, 42, 55, 33, 20, 43, 62, 39, 23, 48, 34, 16, 25, 12].map((height, index) => <i key={index} style={{ height }}/>)}</div><span>{live ? (assets.voiceReferenceText && assets.voiceReferenceText === character.selfIntroduction ? '自己紹介のサンプル音声' : 'サンプル音声') : '音声の参考素材'}</span><audio key={assets.voice} controls preload="metadata" src={assets.voice} aria-label={`${characterName(character, characterIndex)}の${live ? '音声' : '参考音声'}`}>お使いのブラウザーは音声再生に対応していません。</audio></div> : <div className="char-voice-placeholder" aria-label={generating ? '音声は生成待ち・生成中' : '音声は未生成'}><div className="char-placeholder-wave" aria-hidden="true">{[12, 23, 34, 18, 42, 55, 33, 20, 43, 62, 39, 23, 48, 34, 16, 25, 12].map((height, index) => <i key={index} style={{ height }}/>)}</div><span>{generating ? '音声は生成が完了すると試聴できます' : '音声はまだ生成されていません'}</span><small>{generating ? '生成待ち・生成中' : '未生成'}</small></div>}
        {assets?.voiceReferenceText && <div className="char-reference-speech"><h4>サンプル音声の台詞</h4><blockquote>{assets.voiceReferenceText}</blockquote></div>}
        {voiceEdit ? <fieldset disabled={busy} className="char-inline-editor" aria-label="声を編集"><label className="field-label">声・話し方の設定<textarea autoFocus rows={5} value={voiceEdit.voice} disabled={character.locked.voice} onChange={event => updateEdit('voice', { voice: event.target.value })}/></label>{editActions('voice')}</fieldset> : <ResultText title="声・話し方の設定" text={character.voice}/>}
        <div className="char-asset-actions"><span>素材の作り直し</span><button type="button" className="char-retake-button" disabled={busy || character.locked.voice || resultUnavailable || unsavedEdits.length > 0} onClick={() => { setRetake('voice-retake'); setRetakeInstruction('') }}><Icon name="refresh" size={14}/>音声だけリテイク</button></div>
        <p className="char-retake-hint">人物の設定と声のイメージを維持して、音声だけ作り直します。</p>
        {onCloneVoice && <CharacterVoiceTrials characterId={character.id} characterName={characterName(character, characterIndex)} characters={characters} onSelect={setSelectedId} referenceId={assets?.voiceArtifactId} tests={assets?.voiceTests ?? []} busy={busy} onGenerate={onCloneVoice} onPendingChange={setClonePending}/>}
      </section>
    </div>

    {relationshipResults}

    <details ref={revisionPanel} className="char-revision-accordion" open={revisionOpen} onToggle={event => setRevisionOpen(event.currentTarget.open)}>
      <summary><span className="char-revision-symbol"><Icon name="spark" size={20}/></span><span><strong>キャラクターを修正する</strong><small>設定・外見・声をまとめて、または個別に</small></span>{unsavedDrafts.some(([key]) => key.startsWith(`${character.id}:`)) && <span className="char-unsaved-label">未保存</span>}<Icon name="chevron" size={16}/></summary>
      {revisionOpen && <div className="char-revision-box">
        <div className="char-scope-tabs" role="group" aria-label="修正の対象">
          {(['all', 'settings', 'appearance', 'voice'] as const).map(item => <button type="button" key={item} aria-pressed={scope === item} className={scope === item ? 'is-selected' : ''} onClick={() => { setScope(item); setSavedMessage('') }}>{item === 'all' && <Icon name="spark" size={14}/>} {scopeNames[item]}{(drafts[`${character.id}:${item}`] ?? '').trim() && <span className="char-draft-dot" title="未保存の指示あり"/>}</button>)}
        </div>
        <p className="char-scope-description">{scopeDescriptions[scope]}</p>
        <label className="field-label char-instruction-label">{scopeNames[scope]}をどう修正しますか？<textarea ref={instructionInput} rows={3} value={instruction} disabled={scopeLocked} placeholder={scopePlaceholders[scope]} onChange={event => { setDrafts(current => ({ ...current, [draftKey]: event.target.value })); setSavedMessage('') }}/></label>
        <div className="char-target-row"><span>変更対象</span>{activeScopes.length ? activeScopes.map(item => <span className="char-target-tag" key={item}>{scopeNames[item]}</span>) : <span>すべて固定中</span>}{protectedScopes.length > 0 && <span className="char-protected"><Icon name="lock" size={12}/>{protectedScopes.map(item => scopeNames[item]).join('・')}を保護</span>}</div>
        {scopeLocked && <p className="char-lock-notice">{busy ? '生成が完了すると修正できます。' : resultUnavailable ? '先にキャラクターを生成してください。' : '変更するには、対象の固定を解除してください。'}</p>}
        <div className="char-save-row"><p>{live ? 'この指示をもとに、対象の設定・素材を再生成します。' : <>プレビューでは指示を保存します。<br/>AIによる修正結果の生成は未接続です。</>}</p><div className="char-instruction-actions"><button type="button" className="button button-light" disabled={!instruction} onClick={() => setDrafts(current => ({ ...current, [draftKey]: '' }))}>入力を破棄</button><button type="button" className="button button-primary" disabled={!instruction.trim() || scopeLocked || unsavedEdits.length > 0} onClick={saveInstruction}><Icon name="check" size={15}/>{live ? '指示して再生成' : '指示を保存（プレビュー）'}</button></div></div>
      </div>}
    </details>
    <p className="char-save-status" role="status" aria-live="polite">{savedMessage}</p>

    <details className="char-history">
      <summary><span><Icon name="edit" size={15}/>保存した修正指示<span className="char-history-count">{characterRequests.length}</span></span><Icon name="chevron" size={14}/></summary>
      {characterRequests.length ? <ol>{characterRequests.map(request => <li key={request.id}><div><span className="char-history-scope">{scopeNames[request.scope]}</span><span className="char-history-state">{live ? '生成・修正の履歴' : '指示保存済み・未生成'}</span></div><p>{request.instruction}</p>{request.protectedScopes.length > 0 && <small><Icon name="lock" size={11}/>保護する項目：{request.protectedScopes.map(item => scopeNames[item as RevisionScope] ?? item).join('・')}</small>}</li>)}</ol> : <p className="char-empty-history">保存した指示がここに表示されます。</p>}
    </details>
    {hasPending && <div className="char-pending-notice" role="status"><Icon name="edit" size={15}/><div><p>未保存の変更があります。保存するか、キャンセル・破棄してから進んでください。</p><div>{unsavedEdits.map(([key, edit]) => { const pendingScope = key.slice(edit.id.length + 1) as DetailScope; const index = characters.findIndex(item => item.id === edit.id); return <button type="button" key={key} onClick={() => { setSelectedId(edit.id); setSavedMessage('') }}>{characterName(edit, index)} / {scopeNames[pendingScope]}を編集中<Icon name="arrow" size={12}/></button> })}{unsavedDrafts.map(([key]) => { const draftCharacter = characters.find(item => key.startsWith(`${item.id}:`))!; const pendingScope = key.slice(draftCharacter.id.length + 1) as EditScope; return <button type="button" key={key} onClick={() => { setSelectedId(draftCharacter.id); chooseScope(pendingScope) }}>{characterName(draftCharacter, characters.indexOf(draftCharacter))} / {scopeNames[pendingScope]}への指示<Icon name="arrow" size={12}/></button> })}</div></div></div>}
    <div className="char-step-footer"><button type="button" className="button button-light" disabled={busy || hasPending} onClick={onBack}><Icon name="back" size={16}/>世界観の確認に戻る</button><button type="button" className="button button-primary" disabled={busy || hasPending || !canApprove} onClick={onApprove}><Icon name="check" size={16}/>この内容で確認を完了</button></div>
    <p className="char-complete-note">{live ? 'すべての設定と素材が揃うと確認を完了できます。承認した内容は作品の版として保存されます。' : '確認を完了すると、現在の設定と保存済みの指示をこのブラウザーに保存します。'}</p>

    {portrait && <Dialog title={`${portrait.name}の立ち絵`} className="char-portrait-dialog" onClose={() => setPortrait(null)}><img src={portrait.image} alt={`${portrait.name}の立ち絵（拡大）`}/></Dialog>}
    {retake && <Dialog title={scopeNames[retake]} onClose={() => { setRetake(null); setRetakeInstruction('') }}><p className="dialog-intro">{retake === 'image-retake' ? '人物の設定と外見のイメージを維持したまま、立ち絵の素材だけを作り直す指示です。' : '人物の設定と声のイメージを維持したまま、音声の素材だけを作り直す指示です。'}キャラクター自体を変えたい場合は「キャラクター全体」への修正指示を使えます。</p><label className="field-label">リテイクへの追加指示<span className="char-optional">任意</span><textarea autoFocus rows={4} value={retakeInstruction} onChange={event => setRetakeInstruction(event.target.value)} placeholder={retake === 'image-retake' ? '例：同じ服装で、少し顔を正面に向けて。' : '例：同じ声質で、もう少し間を置いて読んで。'}/></label><p className="dialog-info">{live ? '設定と他の素材を維持したまま、選択した素材のみを生成します。' : <>プレビューでは指示を保存します。{retake === 'image-retake' ? '画像' : '音声'}の再生成はまだ実行されません。</>}</p><div className="dialog-actions"><button type="button" className="button button-light" onClick={() => { setRetake(null); setRetakeInstruction('') }}>キャンセル</button><button type="button" className="button button-primary" disabled={busy} onClick={saveRetake}><Icon name="check" size={15}/>{live ? '指示して再生成' : '指示を保存（プレビュー）'}</button></div></Dialog>}
  </div>
}

function ResultText({ title, text }: { title?: string; text: string }) {
  return <div className="char-result-copy">{title && <h4>{title}</h4>}{text.trim() ? text.split(/\n\s*\n/).map((paragraph, index) => <p key={index}>{paragraph}</p>) : <p className="char-result-empty">未生成</p>}</div>
}

function DetailHeader({ title, icon, locked, onToggle, onEdit, editing, disabled = false }: { title: string; icon: IconName; locked: boolean; onToggle: () => void; onEdit: () => void; editing: boolean; disabled?: boolean }) {
  return <div className="char-detail-heading"><h3><Icon name={icon} size={17}/>{title}</h3><div className="char-detail-tools"><LockButton name={title} locked={locked} onClick={onToggle} disabled={disabled}/>{!editing && <button type="button" className="char-edit-button" disabled={locked || disabled} aria-label={`${title}を修正`} onClick={onEdit}><Icon name="edit" size={13}/>修正</button>}</div></div>
}
