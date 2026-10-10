import { useEffect, useState } from 'react'
import { Icon } from './Icons'
import type { M2Outfit } from './m2Api'

type OutfitFields = { name: string; description: string; occasion: string }
const empty: OutfitFields = { name: '', description: '', occasion: '' }
const fieldsOf = (outfit?: M2Outfit): OutfitFields => outfit ? { name: outfit.name, description: outfit.description ?? '', occasion: outfit.occasion ?? '' } : empty
const same = (left: OutfitFields, right: OutfitFields) => left.name === right.name && left.description === right.description && left.occasion === right.occasion

export function CharacterOutfits({ characterId, characterName, outfits, baseImageArtifactId, baseReady, busy, onSave, onDelete, onGenerate, onExpand }: {
  characterId: string; characterName: string; outfits: M2Outfit[]; baseImageArtifactId: string | null; baseReady: boolean; busy: boolean
  onSave: (characterId: string, outfit: OutfitFields & { id?: string }) => Promise<boolean>
  onDelete: (characterId: string, outfitId: string) => Promise<boolean>
  onGenerate: (characterId: string, outfitId: string) => Promise<boolean>
  onExpand: (name: string, image: string) => void
}) {
  const base = outfits.find(outfit => outfit.id === 'default')
  const looks = outfits.filter(outfit => outfit.id !== 'default')
  const [drafts, setDrafts] = useState<Record<string, OutfitFields>>({})
  // Server data replaces a draft only once the user's own text has been saved.
  useEffect(() => setDrafts({}), [characterId])
  const draft = (key: string, outfit?: M2Outfit) => drafts[key] ?? fieldsOf(outfit)
  const change = (key: string, outfit: M2Outfit | undefined, patch: Partial<OutfitFields>) => setDrafts(current => ({ ...current, [key]: { ...draft(key, outfit), ...patch } }))
  const clear = (key: string) => setDrafts(current => Object.fromEntries(Object.entries(current).filter(([name]) => name !== key)))
  async function save(key: string, id: string | undefined, outfit?: M2Outfit) {
    const value = draft(key, outfit)
    if (await onSave(characterId, { ...value, name: value.name.trim(), ...(id ? { id } : {}) })) clear(key)
  }
  const added = draft('new')
  const baseDraft = draft('default', base)
  return <details className="char-clone-panel char-outfits" open={looks.length > 0 || undefined}>
    <summary><Icon name="users" size={15}/><span>別バージョン（服装・変身など）</span>{looks.length > 0 && <small className="char-unsaved-label">{looks.length}件</small>}<Icon name="chevron" size={13}/></summary>
    <p>同じ人物の別の姿を、この立ち絵をもとに作ります。作成した別バージョンは本編のどこかで必ず登場し、場面に合わせて立ち絵が切り替わります。</p>
    <p className="char-outfit-caution">別バージョンを持たない人物は、休日や祭りの場面でも通常の姿のまま登場します。一緒に姿を変えたい人物には、それぞれ別バージョンを作ってください。</p>
    {!baseReady && <p className="char-lock-notice">現在の設定を反映した立ち絵の生成が完了すると作成できます。</p>}
    {looks.map(outfit => {
      const value = draft(outfit.id, outfit)
      const dirty = !same(value, fieldsOf(outfit))
      const stale = !outfit.imageArtifactId || outfit.imagePendingChanges || outfit.baseImageArtifactId !== baseImageArtifactId
      const label = `${characterName}の${outfit.name}`
      return <article className="char-outfit" key={outfit.id}>
        {outfit.imageUrl
          ? <button type="button" className="char-outfit-image" aria-label={`${label}を拡大`} onClick={() => onExpand(label, outfit.imageUrl!)}><img src={outfit.imageUrl} alt={label}/></button>
          : <div className="char-outfit-image char-outfit-empty" aria-label={`${label}の立ち絵は未生成`}><Icon name="users" size={26}/></div>}
        <div className="char-outfit-body">
          <label className="field-label">呼び名<input maxLength={40} value={value.name} disabled={busy} onChange={event => change(outfit.id, outfit, { name: event.target.value })}/></label>
          <label className="field-label">どんな姿にしたいか<textarea rows={3} maxLength={400} value={value.description} disabled={busy} placeholder="空欄でも、呼び名と作品の雰囲気から考えます。" onChange={event => change(outfit.id, outfit, { description: event.target.value })}/></label>
          <label className="field-label">どんな時の姿か<textarea rows={2} maxLength={400} value={value.occasion} disabled={busy} placeholder="例：学校のない休日。変身している間だけ。" onChange={event => change(outfit.id, outfit, { occasion: event.target.value })}/></label>
          {outfit.generatedDescription && !stale && <p className="char-outfit-status">この姿：{outfit.generatedDescription}</p>}
          <p className="char-outfit-status" role="status">{!outfit.imageArtifactId ? '立ち絵は未生成です。' : stale ? '内容または元の立ち絵が変わりました。立ち絵を作り直してください。' : '立ち絵は生成済みです。'}</p>
          <div className="char-edit-actions">
            <button type="button" className="button button-light" disabled={busy} onClick={() => { if (window.confirm(`${outfit.name}を削除しますか？`)) void onDelete(characterId, outfit.id) }}>削除</button>
            {dirty && <button type="button" className="button button-light" disabled={busy || !value.name.trim()} onClick={() => void save(outfit.id, outfit.id, outfit)}>変更を保存</button>}
            <button type="button" className="button button-outline" disabled={busy || dirty || !baseReady} onClick={() => void onGenerate(characterId, outfit.id)}><Icon name="refresh" size={14}/>{outfit.imageArtifactId ? '立ち絵を作り直す' : '立ち絵を生成'}</button>
          </div>
        </div>
      </article>
    })}
    {looks.length < 4 && <fieldset className="char-inline-editor char-outfit-new" disabled={busy} aria-label="別バージョンを追加">
      <label className="field-label">呼び名<input maxLength={40} value={added.name} placeholder="例：私服、魔法少女の姿" onChange={event => change('new', undefined, { name: event.target.value })}/></label>
      <label className="field-label">どんな姿にしたいか<textarea rows={3} maxLength={400} value={added.description} placeholder="例：星屑を散りばめた青いドレスの魔法少女。書き方は自由で、空欄でも呼び名から考えます。" onChange={event => change('new', undefined, { description: event.target.value })}/></label>
      <label className="field-label">どんな時の姿か<textarea rows={2} maxLength={400} value={added.occasion} placeholder="例：学校のない休日。変身している間だけ。" onChange={event => change('new', undefined, { occasion: event.target.value })}/></label>
      <div className="char-edit-actions"><button type="button" className="button button-outline" disabled={busy || !added.name.trim()} onClick={() => void save('new', undefined)}>別バージョンを追加</button></div>
    </fieldset>}
    {looks.length > 0 && <fieldset className="char-inline-editor char-outfit-new" disabled={busy} aria-label="通常の姿の呼び名">
      <p>通常の姿にも呼び名を付けると、台本が姿を区別しやすくなります（任意）。</p>
      <label className="field-label">通常の姿の呼び名<input maxLength={40} value={baseDraft.name} placeholder="例：制服" onChange={event => change('default', base, { name: event.target.value })}/></label>
      <label className="field-label">通常の姿の見た目<textarea rows={2} maxLength={400} value={baseDraft.description} onChange={event => change('default', base, { description: event.target.value })}/></label>
      {!same(baseDraft, fieldsOf(base)) && <div className="char-edit-actions"><button type="button" className="button button-light" disabled={busy || !baseDraft.name.trim()} onClick={() => void save('default', 'default', base)}>保存</button></div>}
    </fieldset>}
  </details>
}
