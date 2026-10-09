import { useEffect, useState } from 'react'
import { errorMessage, request } from './api'
import { Dialog } from './PreviewComponents'
import './llm-settings.css'

type Settings = { model: string; temperature: number; top_p: number; reasoning_effort: string; ctx_size: number }
type Model = { model: string; reasoning_efforts: string[]; worker_name: string }
type Response = { settings: Settings; models: Model[] }

export function LLMSettings({ onClose }: { onClose: () => void }) {
  const [busy, setBusy] = useState(false)
  return <Dialog title="LLM共通設定" onClose={() => { if (!busy) onClose() }}><LLMSettingsContent onBusyChange={setBusy}/></Dialog>
}

export function LLMSettingsContent({ onBusyChange }: { onBusyChange?: (busy: boolean) => void }) {
  const [value, setValue] = useState<Settings | null>(null)
  const [models, setModels] = useState<Model[]>([])
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const [busy, setBusy] = useState(false)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => { onBusyChange?.(busy) }, [busy, onBusyChange])
  useEffect(() => {
    const controller = new AbortController()
    setError('')
    request<Response>('/api/settings/llm', controller.signal).then(result => {
      if (controller.signal.aborted) return
      setValue(current => current ?? result.settings); setModels(result.models)
    }).catch(reason => { if (!controller.signal.aborted) setError(errorMessage(reason)) })
    return () => controller.abort()
  }, [attempt])
  const available = models.filter(model => model.model === value?.model)
  const supported = value && available.length > 0 && value.ctx_size >= 16384
  function change(patch: Partial<Settings>) { setValue(current => current && { ...current, ...patch }); setSaved(false) }
  async function save() {
    if (!value) return
    setBusy(true); setError(''); setSaved(false)
    try {
      const result = await request<Response>('/api/settings/llm', new AbortController().signal, value)
      setValue(result.settings); setModels(result.models); setSaved(true)
    } catch (reason) { setError(errorMessage(reason)) }
    finally { setBusy(false) }
  }
  return <>
    <p>世界観・人物設定・台本などの生成に使う共通設定です。新しい生成や「再試行」では現在の設定を使います。制作中の物語も、次の章から変更が反映されます。実行中の処理とその自動再試行は、処理開始時の設定で続けます。</p>
    {error && <p role="alert" className="m2-error">{error}</p>}
    {!value ? <p>設定を読み込んでいます。</p> : <form onSubmit={event => { event.preventDefault(); void save() }}>
      <fieldset className="llm-settings-fields" disabled={busy}>
        <label>モデル<select value={value.model} onChange={event => change({ model: event.target.value })}>
          {!available.length && <option value={value.model}>{value.model}（対応Worker未接続）</option>}
          {[...new Set(models.map(model => model.model))].map(model => <option key={model} value={model}>{model}</option>)}
        </select></label>
        <label>Temperature<input type="number" min="0" max="2" step="any" required value={value.temperature} onChange={event => change({ temperature: event.target.valueAsNumber })}/></label>
        <label>Top P<input type="number" min="0.000001" max="1" step="any" required value={value.top_p} onChange={event => change({ top_p: event.target.valueAsNumber })}/></label>
        <label>Reasoning effort<input type="text" value={value.reasoning_effort} onChange={event => change({ reasoning_effort: event.target.value })}/></label>
        <label>コンテキスト長<input type="number" min="16384" step="1" required value={value.ctx_size} onChange={event => change({ ctx_size: event.target.valueAsNumber })}/><small>最低16,384。指定した値でモデルを起動します。</small></label>
      </fieldset>
      <p>出力トークン上限は、各処理に必要な値をアプリが設定します。</p>
      {!supported && <p role="status">この設定に対応するWorkerが接続されていません。Workerのモデル設定と起動状態を確認してください。</p>}
      {saved && <p role="status">共通設定を保存しました。</p>}
      <div className="llm-settings-actions"><button type="button" disabled={busy} onClick={() => setAttempt(attempt + 1)}>Worker情報を更新</button><button type="submit" disabled={busy || !supported}>{busy ? '保存中…' : '保存'}</button></div>
    </form>}
    {!value && <button onClick={() => setAttempt(attempt + 1)}>再読み込み</button>}
  </>
}
