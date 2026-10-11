import { useEffect, useState } from 'react'
import { Icon } from './Icons'
import { Dialog } from './PreviewComponents'
import type { M2Outfit } from './m2Api'

export type LookWish = { id?: string; name: string; description: string; occasion: string; appearance?: string }
const empty: LookWish = { name: '', description: '', occasion: '' }
const wishOf = (outfit?: M2Outfit): LookWish => outfit ? { name: outfit.name, description: outfit.description ?? '', occasion: outfit.occasion ?? '' } : empty
const same = (left: LookWish, right: LookWish) => left.name === right.name && left.description === right.description && left.occasion === right.occasion
export const alternateLooks = (outfits: M2Outfit[] | undefined) => (outfits ?? []).filter(outfit => outfit.id !== 'default')
export const MAX_LOOKS = 4

function LookFields({ value, onChange, disabled }: { value: LookWish; onChange: (patch: Partial<LookWish>) => void; disabled: boolean }) {
  return <>
    <label className="field-label">呼び名<input maxLength={40} value={value.name} disabled={disabled} placeholder="例：私服、魔法少女の姿" onChange={event => onChange({ name: event.target.value })}/></label>
    <label className="field-label">どんな姿にしたいか<span className="char-optional">任意</span><textarea rows={3} maxLength={400} value={value.description} disabled={disabled} placeholder="例：星屑を散りばめた青いドレスの魔法少女。書き方は自由で、空欄でも呼び名から考えます。" onChange={event => onChange({ description: event.target.value })}/></label>
    <label className="field-label">どんな時の姿か<span className="char-optional">任意</span><textarea rows={2} maxLength={400} value={value.occasion} disabled={disabled} placeholder="例：学校のない休日。変身している間だけ。" onChange={event => onChange({ occasion: event.target.value })}/></label>
  </>
}

/** STEP1: say which other looks this character has. Images are made in STEP3. */
export function BriefLooks({ looks, onChange, disabled = false }: { looks: LookWish[]; onChange: (looks: LookWish[]) => void; disabled?: boolean }) {
  return <details className="char-detail-inputs char-brief-looks" open={looks.length > 0 || undefined}>
    <summary><span><Icon name="users" size={16}/>別バージョン（服装・変身など）を指定する</span><span className="char-optional">任意{looks.length > 0 && `・${looks.length}件`}</span><Icon name="chevron" size={14}/></summary>
    <div>
      <p className="field-hint">同じ人物の別の姿です。ここではどんな姿かを伝えるだけで、立ち絵はキャラクターの生成後に自動で作ります。指定した姿は本編のどこかで必ず登場します。</p>
      {looks.map((look, index) => <fieldset className="char-inline-editor char-brief-look" key={look.id ?? `new-${index}`} disabled={disabled} aria-label={`別バージョン ${index + 1}`}>
        <LookFields value={look} disabled={disabled} onChange={patch => onChange(looks.map((item, position) => position === index ? { ...item, ...patch } : item))}/>
        <div className="char-edit-actions"><button type="button" className="button button-light" onClick={() => onChange(looks.filter((_, position) => position !== index))}>この別バージョンを削除</button></div>
      </fieldset>)}
      {looks.length < MAX_LOOKS && <button type="button" className="char-retake-button" disabled={disabled} onClick={() => onChange([...looks, { ...empty }])}><Icon name="plus" size={14}/>別バージョンを追加</button>}
    </div>
  </details>
}

/** STEP3: the base portrait and each alternate look as tabs, with "+" to add one. */
export function LookTabs({ outfits, selected, onSelect, onAdd, idPrefix }: {
  outfits: M2Outfit[]; selected: string; onSelect: (id: string) => void; onAdd: () => void; idPrefix: string
}) {
  const looks = alternateLooks(outfits)
  const base = outfits.find(outfit => outfit.id === 'default')
  const tabs = [{ id: 'default', label: base?.name || '通常の姿' }, ...looks.map(outfit => ({ id: outfit.id, label: outfit.name }))]
  return <div className="char-look-tabs" role="tablist" aria-label="姿を選ぶ">
    {tabs.map((tab, index) => <button key={tab.id} type="button" role="tab" id={`${idPrefix}-tab-${tab.id}`} aria-controls={`${idPrefix}-panel`} aria-selected={selected === tab.id} tabIndex={selected === tab.id ? 0 : -1} onClick={() => onSelect(tab.id)} onKeyDown={event => {
      const next = event.key === 'ArrowRight' ? tabs[(index + 1) % tabs.length] : event.key === 'ArrowLeft' ? tabs[(index + tabs.length - 1) % tabs.length] : null
      if (next) { event.preventDefault(); onSelect(next.id); document.getElementById(`${idPrefix}-tab-${next.id}`)?.focus() }
    }}>{tab.label}</button>)}
    {looks.length < MAX_LOOKS && <button type="button" className="char-look-add" aria-label="別バージョンを追加" title="別バージョンを追加" onClick={onAdd}><Icon name="plus" size={14}/></button>}
  </div>
}

export function LookDialog({ characterName, busy, onSave, onClose }: {
  characterName: string; busy: boolean; onSave: (look: LookWish) => Promise<boolean>; onClose: () => void
}) {
  const [value, setValue] = useState<LookWish>(empty)
  return <Dialog title={`${characterName}の別バージョンを追加`} onClose={onClose}>
    <p className="dialog-intro">同じ人物の別の姿を、いまの立ち絵をもとに作ります。作成した別バージョンは本編のどこかで必ず登場し、場面に合わせて立ち絵が切り替わります。</p>
    <form className="char-step" onSubmit={event => { event.preventDefault(); if (!busy && value.name.trim()) void onSave({ ...value, name: value.name.trim() }).then(saved => { if (saved) onClose() }) }}>
      <LookFields value={value} disabled={busy} onChange={patch => setValue(current => ({ ...current, ...patch }))}/>
      <div className="dialog-actions"><button type="button" className="button button-light" onClick={onClose}>キャンセル</button><button type="submit" className="button button-primary" disabled={busy || !value.name.trim()}>追加して立ち絵を作る</button></div>
    </form>
  </Dialog>
}

export function LookPanel({ characterId, characterName, outfit, baseImageArtifactId, baseReady, busy, editing, onEditingChange, onSave, onDelete, onGenerate, onExpand }: {
  characterId: string; characterName: string; outfit: M2Outfit; baseImageArtifactId: string | null; baseReady: boolean; busy: boolean
  editing: boolean; onEditingChange: (editing: boolean) => void
  onSave: (characterId: string, outfit: LookWish) => Promise<boolean>
  onDelete: (characterId: string, outfitId: string) => Promise<boolean>
  onGenerate: (characterId: string, outfitId: string) => Promise<boolean>
  onExpand: (name: string, image: string) => void
}) {
  // What was made is shown as text; the user's wish is only the instruction it came from.
  const appearance = outfit.generatedDescription ?? ''
  const occasion = outfit.occasion || outfit.generatedOccasion || ''
  const wish = outfit.description ?? ''
  const [draft, setDraft] = useState<{ name: string; appearance: string; occasion: string } | null>(null)
  const [retake, setRetake] = useState<string | null>(null)
  useEffect(() => { setDraft(null); setRetake(null) }, [characterId, outfit.id])
  useEffect(() => { if (!editing) setDraft(null) }, [editing])
  const value = draft ?? { name: outfit.name, appearance, occasion }
  const stale = !outfit.imageArtifactId || outfit.imagePendingChanges || outfit.baseImageArtifactId !== baseImageArtifactId
  const label = `${characterName}の${outfit.name}`
  const redraw = Boolean(value.appearance.trim()) && value.appearance.trim() !== appearance
  async function saveEdit() {
    const text = value.occasion.trim()
    const saved = await onSave(characterId, {
      id: outfit.id, name: value.name.trim(), description: wish,
      // An untouched suggestion stays a suggestion, so a later redesign may revise it.
      occasion: !outfit.occasion && text === (outfit.generatedOccasion ?? '') ? '' : text,
      ...(value.appearance.trim() ? { appearance: value.appearance.trim() } : {}),
    })
    if (!saved) return
    onEditingChange(false)
    if (redraw && baseReady) await onGenerate(characterId, outfit.id)
  }
  async function saveRetake() {
    const text = (retake ?? '').trim()
    if (text !== wish && !await onSave(characterId, { id: outfit.id, name: outfit.name, description: text, occasion: outfit.occasion ?? '' })) return
    setRetake(null)
    await onGenerate(characterId, outfit.id)
  }
  return <>
    {outfit.imageUrl
      ? <figure className="char-result-portrait"><button type="button" className="char-portrait-expand" aria-label={`${label}を拡大`} onClick={() => onExpand(label, outfit.imageUrl!)}><img src={outfit.imageUrl} alt={label}/><span className="char-expand-hint">クリックで拡大</span></button><figcaption>{stale ? '以前の立ち絵（作り直し待ち）' : '生成した立ち絵'}</figcaption></figure>
      : <div className="char-image-placeholder" aria-label={`${label}の立ち絵は未生成`}><div className="char-portrait-outline"><Icon name="users" size={47}/></div><span>{busy ? '立ち絵は生成が完了すると表示されます' : '立ち絵は未生成です'}</span></div>}
    <p className="char-outfit-status" role="status">{!outfit.imageArtifactId ? (baseReady ? '立ち絵はまだありません。' : '通常の姿の立ち絵ができると、自動で作ります。') : stale ? '内容または通常の姿の立ち絵が変わりました。作り直しが必要です。' : '立ち絵は生成済みです。'}</p>
    {editing ? <fieldset className="char-inline-editor" disabled={busy} aria-label={`${outfit.name}を編集`}>
      <label className="field-label">呼び名<input autoFocus maxLength={40} value={value.name} onChange={event => setDraft({ ...value, name: event.target.value })}/></label>
      <label className="field-label">見た目の説明<textarea rows={4} maxLength={400} value={value.appearance} onChange={event => setDraft({ ...value, appearance: event.target.value })}/>{redraw && <span className="field-hint">保存すると、この説明のとおりに立ち絵を作り直します。</span>}</label>
      <label className="field-label">どんな時の姿か<textarea rows={2} maxLength={400} value={value.occasion} onChange={event => setDraft({ ...value, occasion: event.target.value })}/></label>
      <div className="char-edit-actions"><button type="button" className="button button-light" onClick={() => onEditingChange(false)}>キャンセル</button><button type="button" className="button button-primary" disabled={busy || !value.name.trim()} onClick={() => void saveEdit()}><Icon name="check" size={14}/>変更を保存</button></div>
    </fieldset> : <>
      <div className="char-result-copy"><h4>見た目の説明</h4>{appearance ? <p>{appearance}</p> : <p className="char-result-empty">立ち絵ができると表示されます。</p>}</div>
      <div className="char-result-copy"><h4>どんな時の姿か</h4>{occasion ? <p>{occasion}</p> : <p className="char-result-empty">立ち絵を作る時に決まります。</p>}</div>
      {wish.trim() && <details className="char-input-reference"><summary>入力した指示<Icon name="chevron" size={13}/></summary><p>{wish}</p></details>}
    </>}
    <div className="char-asset-actions"><span>素材の作り直し</span><button type="button" className="char-retake-button" disabled={busy || editing || !baseReady} onClick={() => setRetake(wish)}><Icon name="refresh" size={14}/>{outfit.imageArtifactId ? 'この姿の立ち絵を作り直す' : 'この姿の立ち絵を作る'}</button></div>
    <p className="char-retake-hint">通常の姿の立ち絵をもとに、顔や髪を保ったまま服装を描き直します。</p>
    <div className="char-asset-actions"><span>この別バージョン</span><button type="button" className="char-retake-button" disabled={busy || editing} onClick={() => { if (window.confirm(`${outfit.name}を削除しますか？`)) void onDelete(characterId, outfit.id) }}><Icon name="close" size={14}/>削除</button></div>
    {retake !== null && <Dialog title={`${label}の立ち絵を作り直す`} onClose={() => setRetake(null)}>
      <p className="dialog-intro">顔や髪を保ったまま、この姿の立ち絵を作り直します。どんな姿にしたいかを書き換えると、見た目も新しく考え直します。</p>
      <label className="field-label">どんな姿にしたいか<span className="char-optional">任意</span><textarea autoFocus rows={4} maxLength={400} value={retake} placeholder="例：もっと華やかに。空欄でも呼び名から考えます。" onChange={event => setRetake(event.target.value)}/></label>
      <div className="dialog-actions"><button type="button" className="button button-light" onClick={() => setRetake(null)}>キャンセル</button><button type="button" className="button button-primary" disabled={busy} onClick={() => void saveRetake()}><Icon name="refresh" size={15}/>作り直す</button></div>
    </Dialog>}
  </>
}

/** Optional name for the base look, so the story can tell the looks apart. */
export function BaseLookName({ characterId, outfits, busy, onSave }: {
  characterId: string; outfits: M2Outfit[]; busy: boolean; onSave: (characterId: string, outfit: LookWish) => Promise<boolean>
}) {
  const base = outfits.find(outfit => outfit.id === 'default')
  const [draft, setDraft] = useState<LookWish | null>(null)
  useEffect(() => setDraft(null), [characterId])
  const value = draft ?? wishOf(base)
  if (!alternateLooks(outfits).length) return null
  return <details className="char-clone-panel char-outfits">
    <summary><Icon name="users" size={15}/><span>通常の姿に呼び名を付ける</span>{base?.name && <small className="char-unsaved-label">{base.name}</small>}<Icon name="chevron" size={13}/></summary>
    <p>「制服」のように呼び名を付けると、台本が別バージョンと区別しやすくなります（任意）。</p>
    <fieldset className="char-inline-editor" disabled={busy}>
      <label className="field-label">通常の姿の呼び名<input maxLength={40} value={value.name} placeholder="例：制服" onChange={event => setDraft({ ...value, name: event.target.value })}/></label>
      {!same(value, wishOf(base)) && <div className="char-edit-actions"><button type="button" className="button button-light" disabled={!value.name.trim()} onClick={() => void onSave(characterId, { ...value, name: value.name.trim(), id: 'default' }).then(saved => { if (saved) setDraft(null) })}>保存</button></div>}
    </fieldset>
  </details>
}
