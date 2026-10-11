import { useCallback, useEffect, useRef, useState } from 'react'
import { errorMessage, request } from './api'
import { Icon } from './Icons'
import { Dialog } from './PreviewComponents'
import './sharedMaterials.css'

export type MaterialImage = {
  id: string; label: string; wish: string; mode: 'text' | 'derive' | 'extract'
  status: 'working' | 'failed' | 'ready' | 'missing'; url: string | null
  summary: string | null; category: 'outfit' | 'item' | null; error: string | null
}
export type Material = { id: string; name: string; description: string; images: MaterialImage[] }
type MaterialsResponse = { project_id: string; materials: Material[]; can_draw: boolean; can_edit: boolean }
/** A finished portrait a design can be taken from. */
export type MaterialPerson = { id: string; name: string; outfitId?: string; planned?: boolean }
type Source = { image_id?: string; character_id?: string; outfit_id?: string; planned?: boolean }
type Draft = { name: string; description: string; label: string; wish: string; mode: 'text' | 'derive' | 'extract'; from: string }
const emptyDraft: Draft = { name: '', description: '', label: '', wish: '', mode: 'text', from: '' }
export const imageName = (material: Material, image: MaterialImage) => image.label ? `${material.name}（${image.label}）` : material.name
export const readySheets = (materials: Material[]) => materials.flatMap(material => material.images.filter(image => image.status === 'ready').map(image => ({ material, image })))

/** The work's shared designs, refreshed while this is on screen. */
export function useMaterials(projectId: string | null) {
  const [state, setState] = useState<MaterialsResponse | null>(null)
  const [error, setError] = useState('')
  const endpoint = projectId === null ? null : `/api/materials/projects/${encodeURIComponent(projectId)}`
  const sequence = useRef(0)
  useEffect(() => {
    if (endpoint === null) return
    const url = endpoint
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    async function poll() {
      const version = sequence.current
      try {
        const next = await request<MaterialsResponse>(url, controller.signal)
        if (!stopped && version === sequence.current) { setState(next); setError('') }
      } catch (reason) {
        if (!stopped && !controller.signal.aborted) setError(errorMessage(reason))
      } finally {
        if (!stopped) timer = setTimeout(poll, 2500)
      }
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer); controller.abort() }
  }, [endpoint])
  const act = useCallback(async (body: Record<string, unknown>) => {
    if (endpoint === null) return false
    sequence.current += 1
    try {
      const next = await request<MaterialsResponse>(`${endpoint}/actions`, new AbortController().signal, body)
      sequence.current += 1; setState(next); setError('')
      return true
    } catch (reason) {
      setError(errorMessage(reason))
      return false
    }
  }, [endpoint])
  return { materials: state?.materials ?? [], canDraw: state?.can_draw ?? false, canEdit: state?.can_edit ?? false, loaded: state !== null, error, act }
}
export type Materials = ReturnType<typeof useMaterials>

/** Marks an operation that uses the image editing model, so its output is never used unknowingly. */
export const EditTag = () => <span className="edit-model-tag" title="画像編集モデルを使う操作です">画像編集</span>

function SourceFields({ draft, setDraft, material, people, canEdit, first }: {
  draft: Draft; setDraft: (draft: Draft) => void; material?: Material; people: MaterialPerson[]; canEdit: boolean; first: boolean
}) {
  const bases = (material?.images ?? []).filter(image => image.status === 'ready')
  return <>
    <fieldset className="material-modes"><legend>作り方</legend>
      <label><input type="radio" name="material-mode" checked={draft.mode === 'text'} onChange={() => setDraft({ ...draft, mode: 'text', from: '' })}/>文章から描く</label>
      {!first && <label className={!canEdit || !bases.length ? 'is-disabled' : ''}><input type="radio" name="material-mode" disabled={!canEdit || !bases.length} checked={draft.mode === 'derive'} onChange={() => setDraft({ ...draft, mode: 'derive', from: bases[0]?.id ?? '' })}/>この素材の絵をもとに変える<EditTag/></label>}
      <label className={!canEdit || !people.length ? 'is-disabled' : ''}><input type="radio" name="material-mode" disabled={!canEdit || !people.length} checked={draft.mode === 'extract'} onChange={() => setDraft({ ...draft, mode: 'extract', from: '0' })}/>人物の立ち絵から取り出す<EditTag/></label>
    </fieldset>
    {draft.mode === 'derive' && <label className="field-label">もとにする絵<select value={draft.from} onChange={event => setDraft({ ...draft, from: event.target.value })}>{bases.map(image => <option key={image.id} value={image.id}>{imageName(material!, image)}</option>)}</select></label>}
    {draft.mode === 'extract' && <label className="field-label">もとにする人物<select value={draft.from} onChange={event => setDraft({ ...draft, from: event.target.value })}>{people.map((person, index) => <option key={index} value={index}>{person.name}</option>)}</select></label>}
    {!canEdit && <p className="field-hint">絵をもとにした作り方は、画像編集の準備ができたWorkerがある時に使えます。</p>}
  </>
}

function sourceOf(draft: Draft, people: MaterialPerson[]): Source | undefined {
  if (draft.mode === 'derive') return { image_id: draft.from }
  if (draft.mode !== 'extract') return undefined
  const person = people[Number(draft.from)]
  return person && { character_id: person.id, outfit_id: person.outfitId, planned: Boolean(person.planned) }
}

/** The library of shared designs: add one, add another sheet to it, redo or remove a sheet. */
export function SharedMaterials({ store, people, disabled = false }: { store: Materials; people: MaterialPerson[]; disabled?: boolean }) {
  const { materials, canDraw, canEdit, loaded, error, act } = store
  const [adding, setAdding] = useState<{ material?: Material } | null>(null)
  const [draft, setDraft] = useState<Draft>(emptyDraft)
  const [shown, setShown] = useState<{ name: string; url: string } | null>(null)
  const [sending, setSending] = useState(false)
  const open = (material?: Material) => { setDraft(emptyDraft); setAdding({ material }) }
  async function send(body: Record<string, unknown>) {
    setSending(true)
    try { return await act(body) } finally { setSending(false) }
  }
  async function submit() {
    if (!adding) return
    const common = { label: draft.label.trim(), wish: draft.wish.trim(), mode: draft.mode, source: sourceOf(draft, people) }
    const done = adding.material
      ? await send({ action: 'add-image', material_id: adding.material.id, ...common })
      : await send({ action: 'add-material', name: draft.name.trim(), description: draft.description.trim(), ...common, wish: '' })
    if (done) setAdding(null)
  }
  const blocked = disabled || sending
  return <details className="shared-materials" open={materials.length > 0 || undefined}>
    <summary><span><Icon name="users" size={16}/>共通の素材（制服・おそろいの小物など）</span><span className="char-optional">任意{materials.length > 0 && `・${materials.length}件`}</span><Icon name="chevron" size={14}/></summary>
    <div className="shared-materials-body">
      <p className="field-hint">複数の人物が同じデザインで身につける服や小物です。ここで絵を用意し、着せる人物はそれぞれの立ち絵の欄で選びます。</p>
      {error && <p className="m2-job-error" role="alert">{error}</p>}
      {materials.map(material => <section className="material-card" key={material.id} aria-label={material.name}>
        <header><div><h4>{material.name}</h4>{material.description && <p>{material.description}</p>}</div>
          <button type="button" className="char-retake-button" disabled={blocked} onClick={() => { if (window.confirm(`${material.name}を削除しますか？`)) void send({ action: 'delete-material', material_id: material.id }) }}><Icon name="close" size={14}/>削除</button></header>
        <ul className="material-images">{material.images.map(image => <li key={image.id}>
          {image.url ? <button type="button" className="material-thumb" aria-label={`${imageName(material, image)}を拡大`} onClick={() => setShown({ name: imageName(material, image), url: image.url! })}><img src={image.url} alt={imageName(material, image)}/></button>
            : <div className="material-thumb material-thumb-empty" role="status">{image.status === 'working' ? '作成中…' : '未作成'}</div>}
          <div className="material-image-text"><strong>{image.label || '基本'}{image.mode !== 'text' && <EditTag/>}</strong>
            {image.status === 'working' && image.url && <span role="status">作り直し中…</span>}
            {image.status === 'failed' && <span className="m2-job-error" role="alert">{image.error || '作成できませんでした。'}</span>}
            {image.summary && <p>{image.summary}</p>}
            <div className="material-image-actions"><button type="button" className="char-retake-button" disabled={blocked || image.status === 'working'} onClick={() => void send({ action: 'retake-image', image_id: image.id })}><Icon name="refresh" size={13}/>作り直す</button>
              <button type="button" className="char-retake-button" disabled={blocked} onClick={() => { if (window.confirm('この絵を削除しますか？')) void send({ action: 'delete-image', image_id: image.id }) }}><Icon name="close" size={13}/>削除</button></div>
          </div></li>)}</ul>
        <button type="button" className="char-retake-button" disabled={blocked || material.images.length >= 6} onClick={() => open(material)}><Icon name="plus" size={14}/>別の絵を追加（男子用・色違いなど）</button>
      </section>)}
      <button type="button" className="button button-light" disabled={blocked || !loaded || (!canDraw && !canEdit)} onClick={() => open()}><Icon name="plus" size={15}/>共通の素材を追加</button>
      {loaded && !canDraw && !canEdit && <p className="field-hint">素材の絵を作るには、立ち絵を作れるWorkerを起動してください。</p>}
    </div>
    {adding && <Dialog title={adding.material ? `${adding.material.name}に別の絵を追加` : '共通の素材を追加'} onClose={() => setAdding(null)}>
      <p className="dialog-intro">{adding.material ? '同じ素材の、形や色が違う絵を追加します。もとの絵と違うところだけを伝えてください。' : '人物なしで、服や小物だけの絵を作ります。書き方は自由で、空欄のところは世界観から考えます。'}</p>
      <form className="char-step" onSubmit={event => { event.preventDefault(); if (!blocked) void submit() }}>
        {adding.material
          ? <><label className="field-label">この絵の呼び名<input autoFocus maxLength={40} value={draft.label} placeholder="例：男子用、夏服、赤" onChange={event => setDraft({ ...draft, label: event.target.value })}/></label>
            <label className="field-label">どう違うか<span className="char-optional">任意</span><textarea rows={3} maxLength={400} value={draft.wish} placeholder="例：下はズボンにする。" onChange={event => setDraft({ ...draft, wish: event.target.value })}/></label></>
          : <><label className="field-label">呼び名<input autoFocus maxLength={40} value={draft.name} placeholder="例：制服、部隊の戦闘服、おそろいの腕章" onChange={event => setDraft({ ...draft, name: event.target.value })}/></label>
            <label className="field-label">どんなものか<span className="char-optional">任意</span><textarea rows={3} maxLength={400} value={draft.description} placeholder="例：紺のブレザーに赤いリボン。空欄でも呼び名と世界観から考えます。" onChange={event => setDraft({ ...draft, description: event.target.value })}/></label></>}
        <SourceFields draft={draft} setDraft={setDraft} material={adding.material} people={people} canEdit={canEdit} first={!adding.material}/>
        <div className="dialog-actions"><button type="button" className="button button-light" onClick={() => setAdding(null)}>キャンセル</button><button type="submit" className="button button-primary" disabled={blocked || (adding.material ? !draft.label.trim() : !draft.name.trim()) || (draft.mode === 'text' && !canDraw)}>追加して絵を作る</button></div>
      </form>
    </Dialog>}
    {shown && <Dialog title={shown.name} className="char-portrait-dialog" onClose={() => setShown(null)}><img src={shown.url} alt={`${shown.name}（拡大）`}/></Dialog>}
  </details>
}

/** Choose which finished sheet a look or portrait wears. */
export function MaterialPicker({ title, materials, busy = false, onPick, onClose }: {
  title: string; materials: Material[]; busy?: boolean; onPick: (imageId: string) => Promise<boolean>; onClose: () => void
}) {
  const sheets = readySheets(materials)
  const [selected, setSelected] = useState(sheets[0]?.image.id ?? '')
  return <Dialog title={title} onClose={onClose}>
    <p className="dialog-intro">選んだ絵の服や小物を、顔や髪を保ったまま身につけた立ち絵に描き直します。<EditTag/></p>
    {sheets.length ? <div className="material-picker" role="radiogroup" aria-label="着せる素材">{sheets.map(({ material, image }) => <label key={image.id} className={selected === image.id ? 'is-selected' : ''}>
      <input type="radio" name="material-pick" checked={selected === image.id} onChange={() => setSelected(image.id)}/><img src={image.url!} alt=""/><span>{imageName(material, image)}</span></label>)}</div>
      : <p className="field-hint">使える絵がまだありません。「共通の素材」で絵を用意してください。</p>}
    <div className="dialog-actions"><button type="button" className="button button-light" onClick={onClose}>キャンセル</button><button type="button" className="button button-primary" disabled={busy || !selected} onClick={() => void onPick(selected).then(done => { if (done) onClose() })}><Icon name="check" size={15}/>これを着せる</button></div>
  </Dialog>
}
