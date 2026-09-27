import { useEffect, useState } from 'react'
import { Icon } from './Icons'
import type { RevisionRequest, WorldBrief } from './wizardState'
import './worldSteps.css'

const genreSuggestions = ['SF', '冒険', 'ミステリー', '恋愛', '時代劇', 'ホラー', 'コメディ', '日常', 'ファンタジー', 'ドキュメンタリー風']
const moodSuggestions = ['温かい', '軽やか', '切ない', '緊張感がある', 'ダーク', '幻想的', 'シュール', '壮大']

function addSuggestion(current: string, suggestion: string) {
  const entries = current.split(/[,、／/\n]/).map(entry => entry.trim())
  return entries.includes(suggestion) ? current : `${current}${current.trim() ? '、' : ''}${suggestion}`
}

function SuggestionInput({ id, label, value, placeholder, suggestions, onChange }: {
  id: string
  label: string
  value: string
  placeholder: string
  suggestions: string[]
  onChange: (value: string) => void
}) {
  return <div className="world-suggestion-field">
    <label className="field-label" htmlFor={id}>{label}<span className="world-optional">自由入力・任意</span></label>
    <input id={id} className="world-text-input" value={value} onChange={event => onChange(event.target.value)} placeholder={placeholder} />
    <div className="world-suggestions" aria-label={`${label}の候補`}>
      {suggestions.map(suggestion => <button key={suggestion} type="button" onClick={() => onChange(addSuggestion(value, suggestion))}><Icon name="plus" size={11} />{suggestion}</button>)}
    </div>
  </div>
}

export function WorldInput({ world, onChange, onNext, onSave, live = false, embedded = false }: {
  world: WorldBrief
  onChange: (patch: Partial<WorldBrief>) => void
  onNext?: () => void
  onSave?: () => void
  live?: boolean
  embedded?: boolean
}) {
  const Wrapper = embedded ? 'div' : 'form'
  return <div className={embedded ? 'world-input-embedded' : 'world-step-layout'}>
    <Wrapper className="world-form world-paper" onSubmit={event => { event.preventDefault(); onNext?.() }}>
      <div className="world-section-intro"><span className="world-section-icon"><Icon name="globe" size={20} /></span><div><h2>まず、どんな世界をつくりますか。</h2><p>決まっていることだけ。空欄はおまかせにできます。</p></div></div>
      <label className="field-label world-main-prompt" htmlFor="world-prompt">世界観・物語の方向性<span className="world-optional">自由記述</span>
        <textarea id="world-prompt" rows={6} value={world.prompt} onChange={event => onChange({ prompt: event.target.value })} placeholder={'舞台や時代、描きたいテーマ、物語が進む方向など、自由に。\n\n例：未来の都市で起きる、少しおかしな日常劇。大きな事件よりも、価値観の違う人々の会話を描きたい。'} />
        <span className="world-field-hint">ひとことでも、詳しい設定でも。ジャンルを組み合わせたり、既存の分類にない世界を指定したりできます。</span>
      </label>
      <div className="world-preference-grid">
        <SuggestionInput id="world-genre" label="ジャンル" value={world.genre} onChange={genre => onChange({ genre })} placeholder="好きなジャンルを自由に入力" suggestions={genreSuggestions} />
        <SuggestionInput id="world-mood" label="雰囲気・トーン" value={world.mood} onChange={mood => onChange({ mood })} placeholder="単語でも、文章でも" suggestions={moodSuggestions} />
      </div>
      <p className="world-suggestion-note">候補は入力のヒントです。クリックすると追記でき、複数を組み合わせられます。</p>
      <details className="world-extra-options">
        <summary><span><Icon name="settings" size={16} />作品名・章数・こだわり</span><span>必要に応じて設定<Icon name="chevron" size={14} /></span></summary>
        <div className="world-extra-fields">
          <div className="world-title-and-length">
            <label className="field-label" htmlFor="world-title">作品名<span className="world-optional">任意</span><input id="world-title" value={world.title} onChange={event => onChange({ title: event.target.value })} placeholder="あとから決めても大丈夫" /></label>
            <label className="field-label" htmlFor="world-chapters">章数<input id="world-chapters" type="number" min={1} max={99} value={world.chapterCount} onChange={event => onChange({ chapterCount: Math.max(1, Math.min(99, Number(event.target.value) || 1)) })} /></label>
          </div>
          <label className="field-label" htmlFor="world-notes">こだわり・避けたい要素<span className="world-optional">任意</span><textarea id="world-notes" rows={3} value={world.notes} onChange={event => onChange({ notes: event.target.value })} placeholder="必ず入れたい設定、表現の好み、描かないでほしい内容など" /></label>
        </div>
      </details>
      {!embedded && <div className="world-form-footer"><span><Icon name="edit" size={14} />次の画面で確認・修正できます</span>{onSave && <button type="button" className="button button-light" onClick={onSave}>入力を保存</button>}<button className="button button-primary">{live ? '世界観を生成する' : '世界観を確認する'}<Icon name="arrow" size={16} /></button></div>}
    </Wrapper>
    {!embedded && <aside className="world-companion">
      <div className="world-creative-board" aria-hidden="true"><span className="world-board-caption">YOUR STORY STARTS HERE</span><div className="world-orbit world-orbit-one" /><div className="world-orbit world-orbit-two" /><div className="world-board-card world-board-card-back"><span>MOOD</span><i /><i /><i /></div><div className="world-board-card world-board-card-front"><Icon name="spark" size={28} /><span>世界のかけら</span><p>まだ名もない<br />物語から。</p></div><span className="world-board-dot" /></div>
      <div className="world-companion-copy"><span className="world-small-eyebrow">A SPACE FOR YOUR IMAGINATION</span><h2>輪郭は、<br />あなたの言葉から。</h2><p>書きたいところから、自由に。<br />細かな設定は、確認しながら<br />少しずつ整えていきましょう。</p></div>
      <ol className="world-small-journey"><li className="is-current"><span>01</span><div>世界観を決める<small>方向性を確認して、修正・確定</small></div></li><li><span>02</span><div>メインキャラクターをつくる<small>確定した世界に、人物を迎える</small></div></li></ol>
    </aside>}
  </div>
}

function WorldResultBody({ text }: { text: string }) {
  return <div className="world-result-prose">{text.trim().split(/\n\s*\n/).map((block, index) => {
    const lines = block.split('\n')
    const heading = lines[0].match(/^#{1,6}\s+(.+)$/)
    return <div className="world-result-section" key={index}>
      {heading ? <><h3>{heading[1]}</h3>{lines.length > 1 && <p>{lines.slice(1).join('\n')}</p>}</> : <p>{block}</p>}
    </div>
  })}</div>
}

export function WorldReview({ world, requests, onChange, onRequest, onBack, onConfirm, onPendingChange, live = false, canConfirm = true, sourceWorld, generating = false, worldConfirmed = false }: {
  world: WorldBrief
  requests: RevisionRequest[]
  onChange: (patch: Partial<WorldBrief>) => void | boolean | Promise<boolean | void>
  onRequest: (instruction: string) => boolean | Promise<boolean>
  onBack: () => void
  onConfirm: () => void
  onPendingChange?: (pending: boolean) => void
  live?: boolean
  canConfirm?: boolean
  sourceWorld?: WorldBrief
  generating?: boolean
  worldConfirmed?: boolean
}) {
  const [editing, setEditing] = useState(false)
  const [editDraft, setEditDraft] = useState(world)
  const [revisionOpen, setRevisionOpen] = useState(false)
  const [instruction, setInstruction] = useState('')
  const [savedMessage, setSavedMessage] = useState('')
  const editableFields = ['title', 'genre', 'mood', 'chapterCount', 'setting'] as const
  const editChanged = editing && editableFields.some(field => editDraft[field] !== world[field])
  const pending = editChanged || Boolean(instruction.trim())
  useEffect(() => { onPendingChange?.(pending) }, [pending, onPendingChange])
  useEffect(() => () => { onPendingChange?.(false) }, [onPendingChange])
  const worldRequests = requests.filter(request => request.scope === 'world')
  const startEditing = () => {
    setEditDraft({ ...world })
    setEditing(true)
    setSavedMessage('')
  }
  const saveEdits = async () => {
    if (await onChange({ title: editDraft.title, genre: editDraft.genre, mood: editDraft.mood, chapterCount: editDraft.chapterCount, setting: editDraft.setting }) === false) return
    setEditing(false)
    setSavedMessage('世界観の変更を保存しました。')
  }
  const saveInstruction = async () => {
    if (!instruction.trim() || !await onRequest(instruction.trim())) return
    setInstruction('')
    setSavedMessage(live ? '修正を受け付けました。生成が終わると結果が更新されます。' : '修正指示を保存しました。プレビューでは本文への反映はまだ行われません。')
    setRevisionOpen(false)
  }

  return <div className="world-step-layout world-review-step">
    <div className="world-review-main">
      <article className="world-paper world-setting-preview" aria-label="世界観の結果">
        <div className="world-result-heading"><div><span className="world-small-eyebrow">WORLD BIBLE</span><h2>{world.title.trim() || 'あなたの物語の世界観'}</h2></div><span className="world-status-label">{worldConfirmed ? '確定済み' : world.setting.trim() ? '確認中' : '未生成'}</span></div>
        {editing ? <form className="world-result-editor" onSubmit={event => { event.preventDefault(); saveEdits() }}>
          <div className="world-edit-intro"><h3>世界観を修正</h3><p>保存すると、結果の表示に反映されます。</p></div>
          <label className="field-label" htmlFor="world-result-title">作品名<input id="world-result-title" autoFocus value={editDraft.title} onChange={event => setEditDraft({ ...editDraft, title: event.target.value })} /></label>
          <div className="world-preference-grid">
            <label className="field-label" htmlFor="world-result-genre">ジャンル<input id="world-result-genre" value={editDraft.genre} onChange={event => setEditDraft({ ...editDraft, genre: event.target.value })} /></label>
            <label className="field-label" htmlFor="world-result-mood">雰囲気・トーン<input id="world-result-mood" value={editDraft.mood} onChange={event => setEditDraft({ ...editDraft, mood: event.target.value })} /></label>
          </div>
          <label className="field-label world-result-chapters" htmlFor="world-result-chapters">章数<input id="world-result-chapters" type="number" min={1} max={99} value={editDraft.chapterCount} onChange={event => setEditDraft({ ...editDraft, chapterCount: Math.max(1, Math.min(99, Number(event.target.value) || 1)) })} /></label>
          <label className="field-label world-setting-editor" htmlFor="world-setting">世界観の本文<textarea id="world-setting" rows={12} value={editDraft.setting} onChange={event => setEditDraft({ ...editDraft, setting: event.target.value })} /></label>
          <div className="world-edit-actions"><button type="button" className="button button-light" onClick={() => setEditing(false)}>キャンセル</button><button className="button button-primary"><Icon name="check" size={14} />変更を保存</button></div>
        </form> : <>
          <dl className="world-brief-facts"><div><dt>ジャンル</dt><dd>{world.genre.trim() || '未定'}</dd></div><div><dt>雰囲気</dt><dd>{world.mood.trim() || '未定'}</dd></div><div><dt>構成</dt><dd>{world.chapterCount} 章</dd></div></dl>
          {world.setting.trim() ? <WorldResultBody text={world.setting} /> : <div className="world-result-empty"><Icon name="book" size={28} /><h3>{generating ? '世界観を生成しています' : '生成結果がここに表示されます'}</h3><p>舞台、世界のルール、物語の方向性を<br />ひとつの世界観として確認できます。</p><span>{generating ? '完了すると、ここに世界観が表示されます。' : live ? '入力画面から生成を開始してください。' : '現在はUIプレビューです。AI生成はまだ接続されていません。'}</span></div>}
          <div className="world-result-actions"><span>気になるところがあれば、修正できます。</span><button className="button button-light" disabled={live && !world.setting.trim()} onClick={startEditing}><Icon name="edit" size={14} />世界観を修正</button></div>
        </>}
        <details className="world-result-source"><summary>入力した指示<Icon name="chevron" size={14} /></summary><div><h3>世界観・物語の方向性</h3><p>{(sourceWorld ?? world).prompt.trim() || '未指定・おまかせ'}</p>{(sourceWorld ?? world).notes.trim() && <><h3>こだわり・避けたい要素</h3><p>{(sourceWorld ?? world).notes}</p></>}</div></details>
      </article>
      <details className="world-paper world-revision-panel" open={revisionOpen} onToggle={event => setRevisionOpen(event.currentTarget.open)}>
        <summary><span><Icon name="spark" size={17} /><span>言葉で世界観を修正<small>{instruction.trim() ? '未保存の修正指示があります' : '変えたいところがあるときに'}</small></span></span><Icon name="chevron" size={16} /></summary>
        {revisionOpen && <div className="world-revision-content">
          <label className="field-label" htmlFor="world-revision">世界観への修正指示<textarea id="world-revision" value={instruction} onChange={event => { setInstruction(event.target.value); setSavedMessage('') }} rows={3} placeholder="例：舞台を近未来に変更。温かな雰囲気は残しつつ、物語の始まりに少し謎を加えて。" /></label>
          <div className="world-revision-actions"><p>{live ? 'この指示をもとに世界観を再生成します。' : 'プレビューでは指示を保存します。AIによる本文への反映は未接続です。'}</p><div className="world-revision-buttons"><button className="button button-light" onClick={() => { setInstruction(''); setRevisionOpen(false) }}>キャンセル</button><button className="button button-primary" disabled={!instruction.trim() || editing || (live && !world.setting.trim())} onClick={saveInstruction}>{live ? '指示して再生成' : '修正指示を保存'}</button></div></div>
        </div>}
      </details>
      <p className="world-saved-message" role="status">{savedMessage}</p>
      {worldRequests.length > 0 && <details className="world-request-history world-paper"><summary>保存した修正指示 <span>{worldRequests.length}</span><Icon name="chevron" size={14} /></summary><p className="world-history-note">{live ? '生成・修正の指示履歴' : '指示を保存済み・本文には未反映'}</p><ol>{worldRequests.map((request, index) => <li key={request.id}><span>{String(index + 1).padStart(2, '0')}</span><p>{request.instruction}</p></li>)}</ol></details>}
    </div>
    <aside className="world-confirm-panel world-paper"><div className="world-confirm-mark"><Icon name="globe" size={24} /></div><span className="world-small-eyebrow">BEFORE THE CHARACTERS</span><h2>この世界から、<br />人物をつくる。</h2><p>世界観を読んで、イメージに合っていれば確定します。最初に伝えた人物の希望と、この世界観をもとにキャラクターをつくります。</p><div className="world-confirm-contents"><span>確認する内容</span><ul><li><Icon name="check" size={14} />舞台と世界のルール</li><li><Icon name="check" size={14} />物語の方向性と雰囲気</li><li><Icon name="check" size={14} />作品全体の構成</li></ul></div><button className="button button-primary world-confirm-button" disabled={editing || pending || !canConfirm} onClick={onConfirm}>{worldConfirmed ? 'キャラクターの確認へ' : live ? '世界観を確定してキャラクターを生成' : '世界観を確定してキャラクターを確認'}<Icon name="arrow" size={16} /></button><p className="world-confirm-note">{editing || pending ? '編集中の内容を保存、またはキャンセルしてから確定してください。' : !canConfirm ? '生成の完了と、最新の入力の反映が必要です。' : '世界観を変更すると、キャラクターの再生成が必要になります。'}</p><button className="world-back-button" onClick={onBack}><Icon name="back" size={15} />世界観・キャラクターの指示に戻る</button></aside>
  </div>
}
