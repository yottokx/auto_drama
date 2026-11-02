import { useEffect, useRef, useState } from 'react'
import { ApiError, errorMessage, request } from './api'
import { putEventCg } from './eventCgState'
import './event-cg.css'

type Precision = 'float32' | 'float16'
type MusicProfile = { precision: Precision }
type MusicSettingsSnapshot = {
  profile: MusicProfile; revision: number; ready: boolean
  workers: { id: string; name: string; ready: boolean }[]
  configurations: { id: string; label: string; precision: Precision }[]
}

export function MusicSettings({ visible, onBusyChange }: { visible: boolean; onBusyChange: (busy: boolean) => void }) {
  const [settings, setSettings] = useState<MusicSettingsSnapshot | null>(null)
  const [profile, setProfile] = useState<MusicProfile | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [conflict, setConflict] = useState(false)
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const mounted = useRef(false)
  const inFlight = useRef(false)
  const requests = useRef(new Set<AbortController>())
  const dirty = Boolean(settings && profile && profile.precision !== settings.profile.precision)

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
    void request<MusicSettingsSnapshot>('/api/music/settings', controller.signal).then(value => {
      if (controller.signal.aborted) return
      setSettings(value); setProfile(value.profile); setConflict(false)
    }).catch(reason => { if (!controller.signal.aborted) setError(errorMessage(reason)) })
      .finally(() => { requests.current.delete(controller); if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [visible, refresh])

  async function save() {
    if (!settings || !profile || inFlight.current || conflict) return
    inFlight.current = true; setSaving(true); setError(''); setNotice('')
    const controller = new AbortController(); requests.current.add(controller)
    try {
      const value = await putEventCg<MusicSettingsSnapshot>('/api/music/settings', {
        expected_revision: settings.revision, profile,
      }, controller.signal)
      if (!mounted.current || controller.signal.aborted) return
      setSettings(value); setProfile(value.profile)
      setNotice('BGM設定を保存しました。まだBGMのない作品へ適用されます。')
    } catch (reason) {
      if (!mounted.current || controller.signal.aborted) return
      if (reason instanceof ApiError && reason.status === 409) setConflict(true)
      setError(`${errorMessage(reason)} 入力は保持しています。`)
    } finally {
      requests.current.delete(controller); inFlight.current = false
      if (mounted.current) setSaving(false)
    }
  }

  return <div className="music-settings tts-settings">
    <div className="tts-settings-notice"><strong>BGM生成モデル</strong><p>場面のBGMを作るモデルです（Stable Audio 3.0 Medium）。</p></div>
    {loading && <p role="status">BGM設定と実行環境を確認中…</p>}
    {settings && profile && <>
      <form onSubmit={event => { event.preventDefault(); void save() }}>
        <fieldset className="event-cg-configurations" disabled={saving || conflict}>
          <legend>ワーカーのVRAM</legend>
          <p>ワーカーのGPUに合わせて選びます。12GBは同じ指定でも、16GB以上とは曲の展開が変わることがあります。</p>
          <ul>{settings.configurations.map(item => <li key={item.id}><label>
            <input type="radio" name="music-configuration" checked={profile.precision === item.precision} onChange={() => { setProfile({ precision: item.precision }); setNotice('') }}/>
            <strong>{item.label}</strong>
          </label></li>)}</ul>
        </fieldset>
        <p>作品のBGMは、最初の1曲を作ったときの設定で作り続けます。設定を変えても、すでにBGMがある作品には影響しません。</p>
        <div className="tts-settings-actions"><button type="submit" disabled={saving || !dirty || conflict}>{saving ? '保存中…' : 'BGM設定を保存'}</button></div>
      </form>
      <h3>接続中のワーカー</h3>
      <p role="status" className={settings.ready ? 'event-cg-ready' : 'event-cg-warning'}>{settings.workers.length === 0
        ? '接続中のワーカーがありません。ワーカーを起動してから再読み込みしてください。'
        : settings.ready
          ? '接続中のワーカーで、BGMを作れます。'
          : 'ワーカーは接続していますが、BGMを作る準備ができていません。'}</p>
      {settings.workers.length > 0 && <ul className="tts-inventory-list" aria-label="接続中のワーカー">{settings.workers.map(worker => <li key={worker.id}><span>{worker.name}</span><strong>{worker.ready ? '使用可能' : '未準備'}</strong></li>)}</ul>}
      <p>モデルの準備手順は <code>docs/setup/stable-audio3.md</code> を参照してください。</p>
    </>}
    {error && <p className="tts-error" role="alert">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    <div className="tts-settings-actions"><button type="button" disabled={saving || loading} onClick={() => { setSettings(null); setNotice(''); setRefresh(value => value + 1) }}>{dirty || conflict ? '入力を取り消して設定を読み込む' : '設定と準備状態を再読み込み'}</button></div>
  </div>
}
