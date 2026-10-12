import { useEffect, useRef, useState } from 'react'
import { ApiError, errorMessage, request } from './api'
import { applyEventCgConfiguration, applyEventCgModel, eventCgConfigurationId, eventCgModelChoiceId, eventCgModelChoices, eventCgTiers, putEventCg, type EventCgProfile, type EventCgSettings as Settings } from './eventCgState'
import './event-cg.css'

export function EventCgSettings({ visible, onBusyChange }: { visible: boolean; onBusyChange: (busy: boolean) => void }) {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [profile, setProfile] = useState<EventCgProfile | null>(null)
  const [steps, setSteps] = useState('40')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [conflict, setConflict] = useState(false)
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const mounted = useRef(false)
  const inFlight = useRef(false)
  const requests = useRef(new Set<AbortController>())
  const stepCount = steps.trim() ? Number(steps) : NaN
  const invalid = !Number.isInteger(stepCount) || stepCount < 1 || stepCount > 100
  const dirty = Boolean(settings && profile && (JSON.stringify(profile) !== JSON.stringify(settings.profile) || stepCount !== settings.profile.steps))
  const configurations = settings?.configurations ?? []
  const sizes = settings?.sizes ?? (profile ? [{ width: profile.width, height: profile.height }] : [])
  const selected = profile ? eventCgConfigurationId(profile, configurations) : null
  const choices = settings?.models && profile ? eventCgModelChoices(settings.models, profile) : []
  const choice = profile ? choices.find(item => item.id === eventCgModelChoiceId(profile)) : undefined
  // Only what the chosen model can run with is offered.
  const tiers = profile ? eventCgTiers(configurations, profile.transformer_storage ?? 'native') : []
  const turbo = profile?.model === 'turbo'

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; requests.current.forEach(controller => controller.abort()) }
  }, [])
  useEffect(() => { onBusyChange(saving) }, [saving, onBusyChange])
  useEffect(() => () => onBusyChange(false), [onBusyChange])
  useEffect(() => {
    if (!visible) return
    // Preserve unsaved settings when returning from another settings tab.
    if (settings) return
    const controller = new AbortController(); requests.current.add(controller)
    setLoading(true); setError('')
    void request<Settings>('/api/event-cg/settings', controller.signal).then(value => {
      if (controller.signal.aborted) return
      setSettings(value); setProfile(value.profile); setSteps(String(value.profile.steps)); setConflict(false)
    }).catch(reason => { if (!controller.signal.aborted) setError(errorMessage(reason)) })
      .finally(() => { requests.current.delete(controller); if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [visible, refresh])

  async function save() {
    if (!settings || !profile || invalid || inFlight.current || conflict) return
    inFlight.current = true; setSaving(true); setError(''); setNotice('')
    const controller = new AbortController(); requests.current.add(controller)
    try {
      const value = await putEventCg<Settings>('/api/event-cg/settings', {
        expected_revision: settings.revision, profile: { ...profile, steps: stepCount },
      }, controller.signal)
      if (!mounted.current || controller.signal.aborted) return
      setSettings(value); setProfile(value.profile); setSteps(String(value.profile.steps))
      setNotice('画像設定を保存しました。次に開始する制作版へ適用されます。')
    } catch (reason) {
      if (!mounted.current || controller.signal.aborted) return
      if (reason instanceof ApiError && reason.status === 409) setConflict(true)
      setError(`${errorMessage(reason)} 入力は保持しています。`)
    } finally {
      requests.current.delete(controller); inFlight.current = false
      if (mounted.current) setSaving(false)
    }
  }

  return <div className="event-cg-settings tts-settings">
    <div className="tts-settings-notice"><strong>画像編集モデル</strong><p>立ち絵を参照して描くモデルです。別バージョンの立ち絵とイベントCGに使います。イベントCGの枚数は、作品の構成確認画面で設定できます。</p></div>
    {loading && <p role="status">画像設定と実行環境を確認中…</p>}
    {settings && profile && <>
      <form onSubmit={event => { event.preventDefault(); void save() }}>
        {choices.length > 0 && <fieldset className="event-cg-configurations event-cg-models" disabled={saving || conflict}>
          <legend>モデル</legend>
          <p>接続中のワーカーに入っているモデルから選びます。8ビットは少ないVRAMで動きます。</p>
          <ul>{choices.map(item => <li key={item.id}><label>
            <input type="radio" name="event-cg-model" checked={choice?.id === item.id} onChange={() => { const next = applyEventCgModel(profile, item, configurations); setProfile(next); setSteps(String(next.steps)); setNotice('') }}/>
            <strong>{item.label}</strong>
            {!item.model.available && <small>接続中のワーカーに、このモデルがありません。</small>}
          </label></li>)}</ul>
        </fieldset>}
        <fieldset className="event-cg-fields" disabled={saving || conflict}>
          <label>画像サイズ<select value={`${profile.width}x${profile.height}`} onChange={event => { const [width, height] = event.target.value.split('x').map(Number); setProfile({ ...profile, width, height }); setNotice('') }}>{sizes.map(size => <option key={`${size.width}x${size.height}`} value={`${size.width}x${size.height}`}>{size.width} × {size.height}{size.width === 960 ? '（標準）' : '（高精細）'}</option>)}</select><small>鑑賞画面は960×640です。1536×1024は高解像度の画面で細部まで表示され、1枚の生成時間は約1.6倍になります。</small></label>
          {!turbo && <label>推論ステップ数<input type="number" min={1} max={100} step={1} value={steps} onChange={event => { setSteps(event.target.value); setNotice('') }}/><small>初期値40。変更すると画質・所要時間に影響します。</small></label>}
        </fieldset>
        {configurations.length > 0 && <fieldset className="event-cg-configurations" disabled={saving || conflict}>
          <legend>ワーカーのVRAM</legend>
          <p>ワーカーのGPUに合わせて選びます。</p>
          <ul>{tiers.map(item => <li key={item.id}><label>
            <input type="radio" name="event-cg-configuration" checked={selected === item.id} onChange={() => { setProfile(applyEventCgConfiguration(profile, item)); setNotice('') }}/>
            <strong>{item.label}</strong>
          </label></li>)}</ul>
          {!selected && <p className="event-cg-warning" role="status">現在の設定は一覧のどれとも一致しません。選ぶと、その内容に置き換わります。</p>}
        </fieldset>}
        {invalid && <p className="tts-error" role="alert">推論ステップ数は1〜100の整数で指定してください。</p>}
        <p>保存した設定は、次に開始する制作版で固定されます。</p>
        <div className="tts-settings-actions"><button type="submit" disabled={saving || !dirty || invalid || conflict}>{saving ? '保存中…' : '画像設定を保存'}</button></div>
      </form>
      <h3>接続中のワーカー</h3>
      <p role="status" className={settings.ready ? 'event-cg-ready' : 'event-cg-warning'}>{settings.workers.length === 0
        ? '接続中のワーカーがありません。ワーカーを起動してから再読み込みしてください。'
        : settings.ready
          ? '接続中のワーカーで、このモデルを使えます。'
          : 'ワーカーは接続していますが、このモデルを使う準備ができていません。'}</p>
      {settings.reasons?.map(reason => <p key={reason}>{reason}</p>)}
      {settings.workers.length > 0 && <ul className="tts-inventory-list" aria-label="接続中のワーカー">{settings.workers.map(worker => <li key={worker.id}><span>{worker.name}{worker.reason && <small>{worker.reason}</small>}</span><strong>{worker.ready ? '使用可能' : '未準備'}</strong></li>)}</ul>}
      <p>モデルの取得手順は <code>docs/setup/qwen-image-edit.md</code> を参照してください。画像編集を使わない作品では準備不要です。</p>
    </>}
    {error && <p className="tts-error" role="alert">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    <div className="tts-settings-actions"><button type="button" disabled={saving || loading} onClick={() => { setSettings(null); setNotice(''); setRefresh(value => value + 1) }}>{dirty || conflict ? '入力を取り消して設定を読み込む' : '設定と準備状態を再読み込み'}</button></div>
  </div>
}
