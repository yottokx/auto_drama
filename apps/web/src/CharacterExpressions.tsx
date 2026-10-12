import { useEffect, useRef, useState, type PointerEvent } from 'react'
import { errorMessage, request } from './api'
import { Icon } from './Icons'
import { EXPRESSIONS, EXPRESSION_SETS, expressionPicture, expressionsToDraw, faceBoxFrom, portraitExpressionView, usedExpressions, type ExpressionId, type ExpressionPicture, type ExpressionRequest, type ExpressionUsage, type FaceBox, type PortraitExpressions } from './expressionState'
import './expressions.css'

/** Marks the face on a portrait by dragging, for the portraits whose face is not found on its own. */
function FaceBoxPicker({ image, name, value, disabled, onChange }: {
  image: string; name: string; value: FaceBox | null; disabled: boolean; onChange: (value: FaceBox | null) => void
}) {
  const origin = useRef<{ x: number; y: number } | null>(null)
  function point(event: PointerEvent<HTMLDivElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    return { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height }
  }
  return <div className="char-face-picker" role="img" aria-label={`${name}の顔の範囲`} aria-disabled={disabled || undefined}
    onPointerDown={event => { if (disabled || event.button !== 0) return; event.currentTarget.setPointerCapture(event.pointerId); origin.current = point(event) }}
    onPointerMove={event => { if (origin.current) onChange(faceBoxFrom(origin.current, point(event))) }}
    onPointerUp={() => { origin.current = null }} onPointerCancel={() => { origin.current = null }}>
    <img src={image} alt="" draggable={false}/>
    {value && <span className="char-face-box" style={{ left: `${value[0] * 100}%`, top: `${value[1] * 100}%`, width: `${(value[2] - value[0]) * 100}%`, height: `${(value[3] - value[1]) * 100}%` }}/>}
  </div>
}

/** STEP3: the expression pictures of the look being shown, and making more of them. */
export function ExpressionPanel({ characterId, characterName, look, looks, pictures, busy, notes = {}, usage, hint, portraitUrl = null, onGenerate, onDelete, onExpand }: {
  characterId: string; characterName: string
  /** The look whose tab is open, and whether it has a portrait that reflects its settings. */
  look: { id: string; name: string; ready: boolean }
  /** Every look that can be drawn from now, the open one included. */
  looks: { id: string; name: string }[]
  pictures: ExpressionPicture[]; busy: boolean
  /** What is being made, or could not be made, said in place of the usual status. */
  notes?: Partial<Record<ExpressionId, string>>
  /** How often the finished work asks this character for each expression, once it is known. */
  usage?: Partial<Record<ExpressionId, number>>
  hint?: string
  /** The portrait the open look's faces are edited from, to mark its face by hand. */
  portraitUrl?: string | null
  onGenerate: (characterId: string, request: ExpressionRequest) => Promise<boolean>
  onDelete: (characterId: string, outfitId: string, expression: ExpressionId) => Promise<boolean>
  onExpand: (name: string, image: string) => void
}) {
  const [chosen, setChosen] = useState<ExpressionId[]>([])
  const [everyLook, setEveryLook] = useState(false)
  const [faceBox, setFaceBox] = useState<FaceBox | null>(null)
  useEffect(() => { setChosen([]); setEveryLook(false); setFaceBox(null) }, [characterId, look.id, portraitUrl])
  // A marked face belongs to one portrait, so it is never used for the other looks.
  const marked = faceBox ? { faceBox } : {}
  const mine = pictures.filter(item => item.outfitId === look.id)
  const targets = everyLook && !faceBox ? looks.map(item => item.id) : look.ready ? [look.id] : []
  const drawing = expressionsToDraw(pictures, targets, chosen)
  const label = `${characterName}の${look.name}`
  const used = usedExpressions(usage)
  function toggle(id: ExpressionId) {
    setChosen(current => current.includes(id) ? current.filter(item => item !== id) : [...current, id])
  }
  return <details className="char-clone-panel char-expressions" open={mine.length > 0 || undefined}>
    <summary><Icon name="users" size={15}/><span>表情</span><span className="char-optional">任意{mine.length > 0 && `・${mine.length}枚`}</span><Icon name="chevron" size={13}/></summary>
    <p><span className="edit-model-tag">画像編集</span>この姿の立ち絵をもとに、顔だけを描き変えた絵を作ります。台本が指定した表情の絵があれば、その場面で立ち絵が切り替わります。作っていない表情の場面では、いまの立ち絵をそのまま表示します。</p>
    <ul className="char-expression-grid" aria-label={`${label}の表情`}>
      {EXPRESSIONS.map(item => {
        const picture = expressionPicture(pictures, look.id, item.id)
        const name = `${label}・${item.name}`
        return <li key={item.id} className={picture ? picture.stale ? 'is-stale' : 'is-made' : undefined}>
          {picture
            ? <button type="button" className="char-expression-thumb" aria-label={`${name}を拡大`} onClick={() => onExpand(name, picture.imageUrl)}><img src={picture.imageUrl} alt={name}/></button>
            : <span className="char-expression-thumb is-empty" aria-hidden="true"><Icon name="users" size={22}/></span>}
          <label><input type="checkbox" checked={chosen.includes(item.id)} disabled={busy} onChange={() => toggle(item.id)}/><strong>{item.name}</strong></label>
          <small>{notes[item.id] ?? (picture ? picture.stale ? '以前の立ち絵のもの（作り直し待ち）' : '作成済み' : '未作成')}</small>
          {usage?.[item.id] && <small className="char-expression-used">本編で{usage[item.id]}回</small>}
          {picture && <span className="char-expression-actions">
            <button type="button" disabled={busy || !look.ready} onClick={() => void onGenerate(characterId, { outfitIds: [look.id], expressions: [item.id], redraw: true, ...marked })}>作り直す</button>
            <button type="button" disabled={busy} onClick={() => { if (window.confirm(`${name}を削除しますか？`)) void onDelete(characterId, look.id, item.id) }}>削除</button>
          </span>}
        </li>
      })}
    </ul>
    <div className="char-expression-sets" role="group" aria-label="まとめて選ぶ">
      {used.length > 0 && <button type="button" className="button button-light" disabled={busy} onClick={() => setChosen(used)}>本編で使われている{used.length}種</button>}
      {EXPRESSION_SETS.map(set => <button key={set.id} type="button" className="button button-light" disabled={busy} onClick={() => setChosen(set.expressions)}>{set.label}</button>)}
      <button type="button" className="button button-light" disabled={busy || chosen.length === 0} onClick={() => setChosen([])}>選択を外す</button>
    </div>
    {looks.length > 1 && <label className="char-expression-scope"><input type="checkbox" checked={everyLook && !faceBox} disabled={busy || Boolean(faceBox)} onChange={event => setEveryLook(event.target.checked)}/>すべての姿（{looks.map(item => item.name).join('・')}）に同じ表情を作る</label>}
    {!look.ready && !everyLook && <p className="char-retake-hint">この姿の立ち絵ができると、表情を作れます。</p>}
    {look.ready && portraitUrl && <details className="char-face-range">
      <summary>顔の範囲を自分で指定する{faceBox && '（指定中）'}</summary>
      <p>顔を見つけられずに作成できなかったときや、描き変わる範囲がずれるときに使います。立ち絵の上で、眉の上からあごの下までをドラッグで囲んでください。指定している間は、この姿の表情だけを作ります。</p>
      <FaceBoxPicker image={portraitUrl} name={label} value={faceBox} disabled={busy} onChange={setFaceBox}/>
      <button type="button" className="button button-light" disabled={busy || !faceBox} onClick={() => setFaceBox(null)}>指定を外して自動で見つける</button>
    </details>}
    <div className="char-edit-actions">
      <button type="button" className="button button-primary" disabled={busy || drawing.length === 0} onClick={() => void onGenerate(characterId, { outfitIds: targets, expressions: chosen, ...marked }).then(started => { if (started) setChosen([]) })}><Icon name="refresh" size={14}/>選んだ表情を作る{drawing.length > 0 && `（${drawing.length}枚）`}</button>
    </div>
    {chosen.length > 0 && drawing.length === 0 && targets.length > 0 && <p className="char-retake-hint">選んだ表情はすべて作成済みです。描き直す場合は、それぞれの「作り直す」を使ってください。</p>}
    {usage && used.length === 0 && <p className="char-retake-hint">本編はこの人物に通常以外の表情を指定していません。表情を作っても、いまの本編では表示されません。</p>}
    {hint && <p className="char-retake-hint">{hint}</p>}
    <p className="char-retake-hint">1枚ごとに、別バージョンの立ち絵1枚と同じくらいの時間がかかります。立ち絵を作り直すと、その姿の表情は使われなくなるので、必要な分を作り直してください。</p>
  </details>
}

/**
 * The expressions of one portrait that is not a main character's STEP3 look: a planned supporting
 * character's, or one chosen while adjusting a finished work. They are kept by the portrait itself,
 * so this reads and makes them on its own and can be placed wherever that portrait is shown.
 */
export function PortraitExpressionPanel({ projectId, characterId, characterName, baseArtifactId, disabled, onExpand, onChange, finished = false }: {
  projectId: string; characterId: string; characterName: string
  /** The portrait the pictures are edited from; none while it is still being made. */
  baseArtifactId: string | null; disabled: boolean
  onExpand: (name: string, image: string) => void
  /** Told when pictures were made or removed, so a preview that shows them can refresh. */
  onChange?: () => void
  /** The work is finished: say which expressions its story uses, and how new pictures reach it. */
  finished?: boolean
}) {
  const [usage, setUsage] = useState<ExpressionUsage | null>(null)
  const [data, setData] = useState<PortraitExpressions | null>(null)
  const [error, setError] = useState('')
  const [sending, setSending] = useState(false)
  const [tick, setTick] = useState(0)
  const wasWorking = useRef(false)
  const endpoint = `/api/expressions/projects/${encodeURIComponent(projectId)}`
  useEffect(() => {
    setData(null); setError('')
    if (!baseArtifactId) return
    const controller = new AbortController()
    void request<PortraitExpressions>(`${endpoint}?base=${encodeURIComponent(baseArtifactId)}`, controller.signal)
      .then(value => { if (!controller.signal.aborted) setData(value) })
      .catch(reason => { if (!controller.signal.aborted) setError(errorMessage(reason)) })
    return () => controller.abort()
  }, [endpoint, baseArtifactId, tick])
  useEffect(() => {
    setUsage(null)
    if (!finished) return
    const controller = new AbortController()
    // Without the counts the panel still works; it only cannot say what the story uses.
    void request<ExpressionUsage>(`${endpoint}/usage`, controller.signal).then(value => { if (!controller.signal.aborted) setUsage(value) }).catch(() => undefined)
    return () => controller.abort()
  }, [endpoint, finished])
  const view = portraitExpressionView((baseArtifactId && data?.portraits[baseArtifactId]) || [])
  useEffect(() => {
    // While something is being made, look again; once it settles, tell whoever shows the pictures.
    if (view.working) { wasWorking.current = true; const timer = window.setTimeout(() => setTick(value => value + 1), 3000); return () => window.clearTimeout(timer) }
    if (wasWorking.current) { wasWorking.current = false; onChange?.() }
  }, [view.working, tick, onChange])
  async function send(path: string, body: unknown): Promise<boolean> {
    if (sending) return false
    setSending(true); setError('')
    try { setData(await request<PortraitExpressions>(endpoint + path, new AbortController().signal, body)); return true }
    catch (reason) { setError(errorMessage(reason)); return false }
    finally { setSending(false) }
  }
  // A work that forgoes the image editing model has no expressions to make.
  if (!baseArtifactId || data?.enabled === false) return null
  return <>
    <ExpressionPanel characterId={characterId} characterName={characterName} look={{ id: 'default', name: '立ち絵', ready: true }} looks={[{ id: 'default', name: '立ち絵' }]}
      pictures={view.pictures} notes={view.notes} busy={disabled || sending || view.working}
      usage={usage ? usage.characters[characterId] ?? {} : undefined}
      portraitUrl={`/api/artifacts/${encodeURIComponent(baseArtifactId)}/content`}
      hint={finished ? '作った表情は、「全章へ反映」を行うと本編に入ります。' : undefined}
      onGenerate={(character_id, wanted) => send('', { base_artifact_id: baseArtifactId, character_id, expressions: wanted.expressions, ...(wanted.redraw ? { redraw: true } : {}), ...(wanted.faceBox ? { face_box: wanted.faceBox } : {}) })}
      onDelete={async (_characterId, _outfitId, expression) => { const removed = await send('/remove', { base_artifact_id: baseArtifactId, expression }); if (removed) onChange?.(); return removed }}
      onExpand={onExpand}/>
    {error && <p className="tts-error" role="alert">{error}</p>}
  </>
}
