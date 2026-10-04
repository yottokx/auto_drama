import { useEffect, useRef, useState } from 'react'
import { ApiError, errorMessage, request } from './api'
import {
  EVENT_CG_MAX, EVENT_CG_VARIANTS_MAX, eventCgMaximum, eventCgPolicyError, eventCgReadinessReason, putEventCg,
  type EventCgEditorStatus, type EventCgPolicy, type EventCgSettings,
} from './eventCgState'
import './event-cg.css'

export function EventCgPolicyEditor({ projectId, disabled, readonly, onStatusChange }: {
  projectId: string; disabled: boolean; readonly: boolean; onStatusChange: (status: EventCgEditorStatus) => void
}) {
  const [policy, setPolicy] = useState<EventCgPolicy | null>(null)
  const [maxCgs, setMaxCgs] = useState('0')
  const [maxVariants, setMaxVariants] = useState('0')
  const [settings, setSettings] = useState<EventCgSettings | null>(null)
  const [error, setError] = useState('')
  const [readinessError, setReadinessError] = useState('')
  const [notice, setNotice] = useState('')
  const [saving, setSaving] = useState(false)
  const [conflict, setConflict] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [readinessRefresh, setReadinessRefresh] = useState(0)
  const savingRef = useRef(false)
  const mounted = useRef(false)
  const controllers = useRef(new Set<AbortController>())
  const endpoint = `/api/event-cg/projects/${encodeURIComponent(projectId)}/policy`
  const count = maxCgs.trim() ? Number(maxCgs) : NaN
  const variants = maxVariants.trim() ? Number(maxVariants) : NaN
  const invalid = eventCgPolicyError(count, variants)
  const dirty = Boolean(policy && (count !== policy.max_cgs || variants !== policy.max_variants_per_cg))
  const readinessReason = invalid ? null : eventCgReadinessReason(count, settings)
  const blocked = count > 0 ? readinessError || readinessReason : null

  useEffect(() => {
    mounted.current = true
    const controller = new AbortController()
    controllers.current.add(controller)
    setError('')
    void request<EventCgPolicy>(endpoint, controller.signal).then(value => {
      if (controller.signal.aborted) return
      setPolicy(value); setMaxCgs(String(value.max_cgs)); setMaxVariants(String(value.max_variants_per_cg)); setConflict(false)
    }).catch(reason => { if (!controller.signal.aborted) setError(errorMessage(reason)) })
    return () => { mounted.current = false; controllers.current.forEach(item => item.abort()); controllers.current.clear() }
  }, [endpoint, refresh])

  useEffect(() => {
    // Disabled works never probe the Qwen environment or require a ready worker.
    if (!policy || invalid || count === 0 || readonly) { setSettings(null); setReadinessError(''); return }
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const value = await request<EventCgSettings>('/api/event-cg/settings', controller.signal)
        if (!controller.signal.aborted) { setSettings(value); setReadinessError('') }
      } catch (reason) { if (!controller.signal.aborted) { setSettings(null); setReadinessError(errorMessage(reason)) } }
      if (!controller.signal.aborted) timer = setTimeout(poll, 5000)
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [Boolean(policy), count > 0, Boolean(invalid), readonly, readinessRefresh])

  useEffect(() => {
    onStatusChange({ pending: dirty || saving, canApprove: Boolean(policy && !error && !invalid && !blocked && !dirty && !saving && !conflict), policy })
  }, [dirty, saving, policy, error, invalid, blocked, conflict, onStatusChange])

  async function save() {
    if (!policy || invalid || savingRef.current || disabled || readonly || conflict) return
    savingRef.current = true; setSaving(true); setError(''); setNotice('')
    const controller = new AbortController(); controllers.current.add(controller)
    try {
      const value = await putEventCg<EventCgPolicy>(endpoint, {
        max_cgs: count, max_variants_per_cg: variants, expected_revision: policy.revision,
      }, controller.signal)
      if (!mounted.current || controller.signal.aborted) return
      setPolicy(value); setMaxCgs(String(value.max_cgs)); setMaxVariants(String(value.max_variants_per_cg))
      setNotice('イベントCGの設定を保存しました。構成の承認時に、この制作版へ固定します。')
    } catch (reason) {
      if (!mounted.current || controller.signal.aborted) return
      if (reason instanceof ApiError && reason.status === 409) setConflict(true)
      setError(`${errorMessage(reason)} 入力は保持しています。`)
    } finally {
      controllers.current.delete(controller); savingRef.current = false
      if (mounted.current) setSaving(false)
    }
  }

  return <section className="event-cg-policy" aria-label="イベントCGの自動生成">
    <div className="event-cg-heading"><h3>イベントCG</h3><span>任意</span></div>
    <p>AIが印象的な場面を選び、人物の立ち絵を参照した一枚絵を生成します。生成にはQwen-Image-2.1の準備と、追加の時間・メモリが必要です。</p>
    {!policy ? <p role="status">CG設定を読み込み中…</p> : <>
      <fieldset className="event-cg-fields" disabled={disabled || readonly || saving || conflict}>
        <label>作品全体の基本CG上限<input type="number" min={0} max={EVENT_CG_MAX} step={1} value={maxCgs} onChange={event => { setMaxCgs(event.target.value); setNotice('') }}/><small>0件で自動生成を無効にします。</small></label>
        <label>CG1件あたりの追加差分上限<input type="number" min={0} max={EVENT_CG_VARIANTS_MAX} step={1} value={maxVariants} onChange={event => { setMaxVariants(event.target.value); setNotice('') }}/><small>基本画像に加えて作る表情・動作などの差分です。</small></label>
      </fieldset>
      {!invalid && <p className="event-cg-total">{count === 0 ? '自動生成なし。通常の背景と立ち絵で制作します。' : `基本CG最大${count}件、差分込みで最大${eventCgMaximum({ max_cgs: count, max_variants_per_cg: variants })}枚（${count} × ${1 + variants}）。必要な場面だけ生成します。`}</p>}
      {invalid && <p className="m2-job-error" role="alert">{invalid}</p>}
      {!readonly && blocked && <p className="event-cg-warning" role="status">{blocked} CGを有効にして承認するには、実行環境の準備が必要です。</p>}
      {!readonly && count > 0 && !blocked && <p className="event-cg-ready" role="status">Qwenの実行環境は準備できています。</p>}
      {readonly ? <p>この制作版の設定は確定しています。</p> : <div className="event-cg-actions">
        {count > 0 && <button type="button" className="button button-light" disabled={saving} onClick={() => setReadinessRefresh(value => value + 1)}>準備状態を再確認</button>}
        {dirty && <button type="button" className="button button-light" disabled={saving || disabled} onClick={() => { setMaxCgs(String(policy.max_cgs)); setMaxVariants(String(policy.max_variants_per_cg)); setNotice('') }}>変更を取り消す</button>}
        <button type="button" className="button button-light" disabled={!dirty || Boolean(invalid) || disabled || saving || conflict} onClick={() => void save()}>{saving ? '保存中…' : 'CG設定を保存'}</button>
      </div>}
    </>}
    {error && <div className="m2-job-error" role="alert"><p>{error}</p><button type="button" className="button button-light" disabled={saving} onClick={() => { setPolicy(null); setNotice(''); setRefresh(value => value + 1) }}>{policy ? '入力を取り消して保存済みの設定を読み込む' : 'CG設定を再読み込み'}</button></div>}
    {notice && <p role="status">{notice}</p>}
  </section>
}
