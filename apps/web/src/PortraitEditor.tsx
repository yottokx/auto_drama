import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { ApiError, errorMessage, request } from './api'
import { Icon } from './Icons'
import type { ProductionResponse } from './ProductionPanel'
import {
  changedEdge, drawnBounds, editedBounds, fullBounds, movedBounds, portraitPayload,
  portraitsChanged, portraitSourceMatches,
  type BodyBounds, type BoundsEdge, type PortraitCharacter, type PortraitEdits, type PortraitResponse,
} from './portraitEditorState'
import './portrait-editor.css'

const conflictMessage = '公開されている章が更新されました。入力した範囲は保持しています。最新の設定を読み直してから調整してください。'

async function savePortraits(path: string, signal: AbortSignal, body: ReturnType<typeof portraitPayload>) {
  const response = await fetch(path, {
    method: 'PUT', signal: AbortSignal.any([signal, AbortSignal.timeout(120000)]),
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  if (!response.ok) {
    const reason = await response.json().catch(() => null)
    throw new ApiError(typeof reason?.detail === 'string' ? reason.detail : '表示設定を保存できませんでした。', response.status)
  }
  return response.json() as Promise<ProductionResponse>
}

export function PortraitEditor({ projectId, productionId, buildId, onSavingChange, onSaved, onPendingChange }: {
  projectId: string; productionId: string; buildId: string
  onSavingChange: (saving: boolean) => void; onSaved: (response: ProductionResponse) => void
  onPendingChange?: (pending: boolean) => void
}) {
  const [open, setOpen] = useState(false)
  const [source, setSource] = useState<PortraitResponse | null>(null)
  const [edits, setEdits] = useState<PortraitEdits>({})
  const [selectedId, setSelectedId] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [savedMessage, setSavedMessage] = useState('')
  const [conflicted, setConflicted] = useState(false)
  const [reload, setReload] = useState(0)
  const [confirmReload, setConfirmReload] = useState(false)
  const identity = `${projectId}:${productionId}`
  const latest = useRef({ identity, buildId })
  latest.current = { identity, buildId }
  const controllers = useRef(new Set<AbortController>())
  const savingRef = useRef(false)
  const callbacks = useRef({ onSavingChange, onSaved, onPendingChange })
  callbacks.current = { onSavingChange, onSaved, onPendingChange }
  const sourceRef = useRef(source)
  sourceRef.current = source
  const requestVersion = useRef(0)
  const endpoint = `/api/m3/projects/${encodeURIComponent(projectId)}/portraits`

  useEffect(() => () => {
    requestVersion.current += 1
    controllers.current.forEach(controller => controller.abort())
    if (savingRef.current) callbacks.current.onSavingChange(false)
    callbacks.current.onPendingChange?.(false)
  }, [identity])

  useEffect(() => {
    if (!open || sourceRef.current) return
    const controller = new AbortController()
    controllers.current.add(controller)
    const version = ++requestVersion.current
    const requestedBuild = buildId
    setLoading(true); setError('')
    async function load() {
      try {
        const next = await request<PortraitResponse>(endpoint, controller.signal)
        if (controller.signal.aborted || latest.current.identity !== identity || version !== requestVersion.current) return
        if (!portraitSourceMatches(next, projectId, productionId, requestedBuild) || latest.current.buildId !== requestedBuild) {
          throw new Error('表示設定が更新されています。少し待ってから再読み込みしてください。')
        }
        setSource(next); setEdits({}); setSelectedId(next.characters[0]?.character_id ?? '')
        setConflicted(false); setSavedMessage(''); setConfirmReload(false)
      } catch (reason) {
        if (!controller.signal.aborted && latest.current.identity === identity && version === requestVersion.current) setError(errorMessage(reason))
      } finally {
        controllers.current.delete(controller)
        if (!controller.signal.aborted && latest.current.identity === identity && version === requestVersion.current) setLoading(false)
      }
    }
    void load()
    return () => controller.abort()
  }, [open, endpoint, identity, projectId, productionId, buildId, reload])

  const currentSource = source?.project_id === projectId && source.production_id === productionId ? source : null
  const conflict = conflicted || Boolean(currentSource && currentSource.build_id !== buildId)
  const dirty = Boolean(currentSource && portraitsChanged(currentSource, edits))
  const selected = currentSource?.characters.find(character => character.character_id === selectedId) ?? currentSource?.characters[0]
  useEffect(() => { onPendingChange?.(dirty || saving) }, [dirty, saving, onPendingChange])

  function change(bounds: BodyBounds | null) {
    if (!selected || savingRef.current) return
    setEdits(current => ({ ...current, [selected.character_id]: bounds })); setSavedMessage('')
  }

  function reloadSettings() {
    if (savingRef.current) return
    if (dirty && !confirmReload) { setConfirmReload(true); return }
    setSource(null); sourceRef.current = null; setEdits({}); setConfirmReload(false)
    setConflicted(false); setError(''); setSavedMessage(''); setReload(value => value + 1)
  }

  async function save() {
    if (!currentSource || !dirty || conflict || savingRef.current) return
    const payload = portraitPayload(currentSource, edits)
    const controller = new AbortController()
    controllers.current.add(controller)
    savingRef.current = true; setSaving(true); setError(''); setSavedMessage('')
    callbacks.current.onSavingChange(true)
    try {
      const response = await savePortraits(endpoint, controller.signal, payload)
      if (controller.signal.aborted || latest.current.identity !== identity) return
      if (response.project_id !== projectId || response.production?.id !== productionId || response.production.build?.status !== 'published') {
        throw new Error('保存結果を確認できませんでした。入力は保持しています。最新の設定を読み直してください。')
      }
      const updated: PortraitResponse = {
        ...currentSource, build_id: response.production.build.id,
        characters: currentSource.characters.map(character => ({ ...character, body_bounds: editedBounds(character, edits) })),
      }
      setSource(updated); setEdits({}); setConflicted(false); setConfirmReload(false)
      setSavedMessage('表示を更新しました。「第1章を鑑賞する」から確認できます。')
      callbacks.current.onSaved(response)
    } catch (reason) {
      if (!controller.signal.aborted && latest.current.identity === identity) {
        if (reason instanceof ApiError && reason.status === 409) setConflicted(true)
        setError(reason instanceof ApiError && reason.status === 409 ? '' : `${errorMessage(reason)} 入力した範囲は保持しています。`)
      }
    } finally {
      controllers.current.delete(controller)
      if (latest.current.identity === identity && savingRef.current) {
        savingRef.current = false; setSaving(false); callbacks.current.onSavingChange(false)
      }
    }
  }

  return <details className="portrait-editor" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary><Icon name="users" size={17}/>立ち絵の表示を調整</summary>
    {open && <div className="portrait-editor-content" aria-busy={loading || saving}>
      <p className="portrait-editor-intro">頭頂から足元、体の左右を枠で囲んでください。傘や帽子など、体からはみ出す部分は含めません。画像全体を表示したまま、人物の大きさと位置を調整します。</p>
      {loading && <p role="status">立ち絵を読み込み中…</p>}
      {error && <p className="m2-job-error" role="alert">{error}</p>}
      {conflict && <p className="portrait-conflict" role="alert">{conflictMessage}</p>}
      {!loading && (!currentSource || conflict) && <div className="portrait-reload"><button type="button" className="button button-light" disabled={saving} onClick={reloadSettings}>{confirmReload ? '入力を破棄して最新の設定を読み込む' : '最新の設定を読み込む'}</button>{confirmReload && <button type="button" className="button button-light" onClick={() => setConfirmReload(false)}>入力を残す</button>}</div>}
      {currentSource && <>
        {currentSource.characters.length > 0 ? <>
          <div className="portrait-tabs" role="group" aria-label="調整する人物">{currentSource.characters.map(character => <button type="button" key={character.character_id} disabled={saving} aria-pressed={selected?.character_id === character.character_id} onClick={() => setSelectedId(character.character_id)}>{character.name}</button>)}</div>
          {selected && <PortraitCanvas key={`${selected.character_id}:${selected.image_artifact_id}`} character={selected} bounds={editedBounds(selected, edits)} disabled={saving} onChange={change}/>}
          <div className="portrait-save"><p>保存すると章の表示を更新します。画像は再生成しません。</p><button type="button" className="button button-light" disabled={!dirty || saving} onClick={() => { setEdits({}); setSavedMessage(''); setError(''); setConfirmReload(false) }}>変更を取り消す</button><button type="button" className="button button-primary" disabled={!dirty || saving || conflict || loading} onClick={() => void save()}><Icon name={saving ? 'refresh' : 'check'} size={15}/>{saving ? '表示を更新中…' : '表示を保存する'}</button></div>
        </> : <p>調整できる立ち絵はありません。</p>}
      </>}
      {savedMessage && <p className="portrait-saved" role="status">{savedMessage}</p>}
    </div>}
  </details>
}

type Drag = {
  pointerId: number; start: { x: number; y: number }; previous: BodyBounds | null
  moving: BodyBounds | null
}

export function PortraitCanvas({ character, bounds, disabled, onChange }: {
  character: PortraitCharacter; bounds: BodyBounds | null; disabled: boolean; onChange: (bounds: BodyBounds | null) => void
}) {
  const drag = useRef<Drag | null>(null)
  const [imageReady, setImageReady] = useState(false)
  const [imageError, setImageError] = useState(false)
  const [imageAttempt, setImageAttempt] = useState(0)
  const unavailable = disabled || !imageReady
  function point(event: PointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    return { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height }
  }
  function start(event: PointerEvent<HTMLDivElement>) {
    if (unavailable || event.button !== 0) return
    const position = point(event)
    const moving = (event.target as HTMLElement).closest('[data-portrait-move]') ? bounds : null
    drag.current = { pointerId: event.pointerId, start: position, previous: bounds, moving }
    event.currentTarget.setPointerCapture(event.pointerId); event.currentTarget.focus(); event.preventDefault()
    if (!moving) onChange(drawnBounds(position, position))
  }
  function move(event: PointerEvent<HTMLDivElement>) {
    if (unavailable || drag.current?.pointerId !== event.pointerId) return
    const position = point(event)
    const current = drag.current
    onChange(current.moving ? movedBounds(current.moving, position.x - current.start.x, position.y - current.start.y) : drawnBounds(current.start, position))
  }
  function keyboard(event: KeyboardEvent<HTMLDivElement>) {
    if (unavailable) return
    const amount = event.shiftKey ? 0.05 : 0.01
    const offsets: Record<string, [number, number]> = { ArrowLeft: [-amount, 0], ArrowRight: [amount, 0], ArrowUp: [0, -amount], ArrowDown: [0, amount] }
    if (offsets[event.key]) {
      event.preventDefault()
      onChange(movedBounds(bounds ?? { left: 0.2, top: 0.1, right: 0.8, bottom: 0.9 }, ...offsets[event.key]))
    }
  }
  const labels: Record<BoundsEdge, string> = { left: '体の左', top: '頭頂', right: '体の右', bottom: '足元' }
  return <div className="portrait-adjustment">
    <div className="portrait-image-area">
      {!imageReady && !imageError && <p role="status">画像を読み込み中…</p>}
      {imageError && <div><p className="m2-job-error" role="alert">立ち絵を読み込めませんでした。</p><button type="button" className="button button-light" disabled={disabled} onClick={() => { setImageError(false); setImageAttempt(value => value + 1) }}>画像を再読み込み</button></div>}
      <div className={`portrait-canvas ${unavailable ? 'is-disabled' : ''}`} tabIndex={unavailable ? -1 : 0} role="group" aria-label={`${character.name}の範囲を指定。画像上をドラッグ、枠の内側をドラッグして移動。矢印キーでも移動できます。`} onKeyDown={keyboard} onPointerDown={start} onPointerMove={move} onPointerUp={event => { if (drag.current?.pointerId === event.pointerId) { move(event); drag.current = null; event.currentTarget.releasePointerCapture(event.pointerId) } }} onPointerCancel={() => { if (drag.current) onChange(drag.current.previous); drag.current = null }} onLostPointerCapture={() => { drag.current = null }}>
        <img key={imageAttempt} src={character.image_url} alt={`${character.name}の立ち絵全体`} draggable={false} onLoad={() => { setImageReady(true); setImageError(false) }} onError={() => { setImageReady(false); setImageError(true) }}/>
        {bounds && imageReady && <div data-portrait-move className="portrait-body-bounds" style={{ left: `${bounds.left * 100}%`, top: `${bounds.top * 100}%`, width: `${(bounds.right - bounds.left) * 100}%`, height: `${(bounds.bottom - bounds.top) * 100}%` }}><span>人物本体</span><i/><i/><i/><i/></div>}
      </div>
    </div>
    <div className="portrait-adjustment-controls">
      <div className="portrait-mode"><strong>{character.name}</strong><span>{bounds ? '範囲を指定' : '自動'}</span></div>
      <p>画像上をドラッグして枠を描き、内側をドラッグして移動できます。数値でも調整できます。</p>
      <fieldset className="portrait-edges" disabled={disabled}><legend>画像の端からの位置（%）</legend>{(Object.keys(labels) as BoundsEdge[]).map(edge => <label key={edge}>{labels[edge]}<input type="number" inputMode="decimal" min={0} max={100} step={0.1} value={bounds ? Math.round(bounds[edge] * 10000) / 100 : ''} placeholder={String(fullBounds[edge] * 100)} onChange={event => { if (event.target.value !== '' && Number.isFinite(event.target.valueAsNumber)) onChange(changedEdge(bounds ?? fullBounds, edge, event.target.valueAsNumber / 100)) }}/></label>)}</fieldset>
      <p className="portrait-keyboard-help">枠にフォーカスして矢印キーで1%、Shift＋矢印キーで5%ずつ移動できます。枠の幅と高さは5%以上必要です。</p>
      <button type="button" className="button button-light" disabled={disabled || bounds === null} onClick={() => onChange(null)}><Icon name="refresh" size={14}/>自動に戻す</button>
    </div>
  </div>
}
