import { useEffect, useRef, useState } from 'react'
import { ApiError, errorMessage, request } from './api'
import { applyEventCgConfiguration, eventCgConfigurationId, putEventCg, type EventCgProfile, type EventCgSettings as Settings } from './eventCgState'
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
    <div className="tts-settings-notice"><strong>イベントCGの画像生成</strong><p>Qwen-Image-2.1で人物の立ち絵を参照した一枚絵を生成します。作品の構成確認画面で、基本CGと追加差分の上限を設定できます。</p></div>
    {loading && <p role="status">画像設定と実行環境を確認中…</p>}
    {settings && profile && <>
      <h3>Qwen-Image-2.1</h3>
      <dl><dt>精度</dt><dd>{profile.transformer_storage === 'fp8' ? '8ビット（fp8）で保持し、BF16で計算' : 'BF16（量子化なし）'}</dd><dt>モデルの版</dt><dd><code>{profile.model_revision}</code></dd></dl>
      <form onSubmit={event => { event.preventDefault(); void save() }}>
        <fieldset className="event-cg-fields" disabled={saving || conflict}>
          <label>画像サイズ<select value={`${profile.width}x${profile.height}`} onChange={event => { const [width, height] = event.target.value.split('x').map(Number); setProfile({ ...profile, width, height }); setNotice('') }}>{sizes.map(size => <option key={`${size.width}x${size.height}`} value={`${size.width}x${size.height}`}>{size.width} × {size.height}{size.width === 960 ? '（標準）' : '（高精細）'}</option>)}</select><small>鑑賞画面は960×640です。1536×1024は高解像度の画面で細部まで表示され、1枚の生成時間は約1.6倍になります。</small></label>
          <label>推論ステップ数<input type="number" min={1} max={100} step={1} value={steps} onChange={event => { setSteps(event.target.value); setNotice('') }}/><small>初期値40。変更すると画質・所要時間に影響します。</small></label>
        </fieldset>
        {configurations.length > 0 && <fieldset className="event-cg-configurations" disabled={saving || conflict}>
          <legend>メモリーの構成</legend>
          <p>ワーカーのGPUのVRAMに合う構成を選びます。上にあるものほど基準の画質に近く、下にあるものほど少ないVRAMで動きます。</p>
          <ul>{configurations.map(item => <li key={item.id}><label>
            <input type="radio" name="event-cg-configuration" checked={selected === item.id} onChange={() => { setProfile(applyEventCgConfiguration(profile, item)); setNotice('') }}/>
            <strong>{item.label}</strong>
            <span>VRAM 約{item.peak_vram_gib}GB</span>
            <span>{item.time_ratio === 1 ? '時間 基準' : `時間 基準の約${item.time_ratio}倍`}</span>
            <small>{item.quality}</small>
          </label></li>)}</ul>
          {!selected && <p className="event-cg-warning" role="status">現在の設定は一覧のどの構成とも一致しません。構成を選ぶと、その内容に置き換わります。</p>}
          <p><small>VRAMは、3人を参照した1536×1024の画像を生成したときの最大使用量の実測です。このほかに3〜4GBの空きが必要です。参照する人物が2人以下なら1〜2GBずつ軽くなります。時間は「32GB：標準」を基準にした比で、実際の秒数はGPUなどの構成によって変わります。画質の説明は少数の画像を見比べた目安です。</small></p>
        </fieldset>}
        {invalid && <p className="tts-error" role="alert">推論ステップ数は1〜100の整数で指定してください。</p>}
        <p>保存した設定は、次に開始する制作版で固定されます。</p>
        <div className="tts-settings-actions"><button type="submit" disabled={saving || !dirty || invalid || conflict}>{saving ? '保存中…' : '画像設定を保存'}</button></div>
      </form>
      <h3>接続中のワーカー</h3>
      <p role="status" className={settings.ready ? 'event-cg-ready' : 'event-cg-warning'}>{settings.workers.length === 0
        ? '接続中のワーカーがありません。ワーカーを起動してから再読み込みしてください。'
        : settings.ready
          ? '接続中のワーカーでイベントCGを生成できます。'
          : 'ワーカーは接続していますが、イベントCGを生成する準備ができていません。'}</p>
      {settings.reasons?.map(reason => <p key={reason}>{reason}</p>)}
      {settings.workers.length > 0 && <ul className="tts-inventory-list" aria-label="接続中のワーカー">{settings.workers.map(worker => <li key={worker.id}><span>{worker.name}{worker.reason && <small>{worker.reason}</small>}</span><strong>{worker.ready ? 'CG生成可能' : 'CG未準備'}</strong></li>)}</ul>}
      <p>専用のDiffusers環境とモデルは、既存のQwen画像編集テストと共通です。準備手順は <code>docs/setup/qwen-image-edit.md</code> を参照してください。作品のCG上限が0件の場合は準備不要です。</p>
    </>}
    {error && <p className="tts-error" role="alert">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    <div className="tts-settings-actions"><button type="button" disabled={saving || loading} onClick={() => { setSettings(null); setNotice(''); setRefresh(value => value + 1) }}>{dirty || conflict ? '入力を取り消して設定を読み込む' : '設定と準備状態を再読み込み'}</button></div>
  </div>
}
