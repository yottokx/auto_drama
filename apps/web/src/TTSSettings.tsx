import { useEffect, useState } from 'react'
import { errorMessage, request } from './api'
import { activeDownload, availableTTSChoices, chooseTTSModel, downloadBlockReason, eligibleDownloadWorker, formatBytes, precisionOptions, resumableDownload, sameTTSChoice, ttsSettingsBlockReason, validTTSChoice, type DownloadOperation, type DownloadSnapshot, type Precision, type TTSChoice, type TTSConfiguration, type TTSModel } from './ttsSettingsState'

type SettingsResponse = { settings: TTSConfiguration; generation_active: boolean }
const statusLabels: Record<string, string> = { queued: '待機中', downloading: 'ダウンロード中', verifying: 'ファイルを検証中', converting: '変換中', cancelling: '中断処理中', completed: '取得完了', cancelled: '中断済み', interrupted: '接続が切れて中断', failed: '失敗' }
const providerLabel = (id: string) => id === 'irodori' ? 'Irodori-TTS' : id

export function TTSSettings({ visible, onBusyChange }: { visible: boolean; onBusyChange: (busy: boolean) => void }) {
  const [settings, setSettings] = useState<TTSConfiguration | null>(null)
  const [models, setModels] = useState<TTSModel[]>([])
  const [loadError, setLoadError] = useState('')
  const [saveError, setSaveError] = useState('')
  const [settingsBusy, setSettingsBusy] = useState(false)
  const [saved, setSaved] = useState(false)
  const [loadAttempt, setLoadAttempt] = useState(0)
  const [snapshot, setSnapshot] = useState<DownloadSnapshot | null>(null)
  const [downloadError, setDownloadError] = useState('')
  const [downloadAttempt, setDownloadAttempt] = useState(0)
  const [workerId, setWorkerId] = useState('')
  const [downloadChoice, setDownloadChoice] = useState<TTSChoice>({ provider_id: 'irodori', model_id: 'irodori-v4-large', precision: 'bf16' })
  const [actionId, setActionId] = useState('')
  const [actionNotice, setActionNotice] = useState('')
  const [actionError, setActionError] = useState('')
  const hasActiveDownloads = Boolean(snapshot?.operations.some(activeDownload))
  useEffect(() => { onBusyChange(settingsBusy || Boolean(actionId)) }, [settingsBusy, actionId, onBusyChange])

  useEffect(() => {
    const controller = new AbortController()
    setLoadError('')
    void Promise.allSettled([
      request<SettingsResponse>('/api/settings/tts', controller.signal).then(result => { if (!controller.signal.aborted) setSettings(current => current ?? result.settings) }),
      request<{ models: TTSModel[] }>('/api/tts/models', controller.signal).then(result => { if (!controller.signal.aborted) setModels(result.models) }),
    ]).then(results => { if (!controller.signal.aborted) setLoadError(results.filter(result => result.status === 'rejected').map(result => errorMessage((result as PromiseRejectedResult).reason)).join(' ')) })
    return () => controller.abort()
  }, [loadAttempt])

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    async function poll() {
      try {
        const result = await request<DownloadSnapshot>('/api/tts-downloads', controller.signal)
        if (controller.signal.aborted) return
        setSnapshot(result); setDownloadError('')
        setWorkerId(current => result.workers.some(worker => worker.id === current && eligibleDownloadWorker(worker)) ? current : result.workers.find(eligibleDownloadWorker)?.id ?? '')
      } catch (reason) { if (!controller.signal.aborted) setDownloadError(errorMessage(reason)) }
      if (!controller.signal.aborted && (visible || hasActiveDownloads)) timer = setTimeout(() => void poll(), 2000)
    }
    void poll()
    return () => { controller.abort(); if (timer) clearTimeout(timer) }
  }, [visible, hasActiveDownloads, downloadAttempt])

  function changeSettings(key: 'voice_design' | 'voice_clone', value: TTSChoice) {
    setSettings(current => current && { ...current, [key]: value }); setSaved(false); setSaveError('')
  }
  async function saveSettings() {
    if (!settings) return
    setSettingsBusy(true); setSaveError(''); setSaved(false)
    try {
      const signal = new AbortController().signal
      const [currentSnapshot, currentCatalog] = await Promise.all([
        request<DownloadSnapshot>('/api/tts-downloads', signal),
        request<{ models: TTSModel[] }>('/api/tts/models', signal),
      ])
      setSnapshot(currentSnapshot); setModels(currentCatalog.models); setDownloadError('')
      const blocked = ttsSettingsBlockReason(settings, currentSnapshot, currentCatalog.models)
      if (blocked) { setSaveError(blocked); return }
      const result = await request<SettingsResponse>('/api/settings/tts', new AbortController().signal, settings)
      setSettings(result.settings); setSaved(true)
    } catch (reason) { setSaveError(errorMessage(reason)) }
    finally { setSettingsBusy(false) }
  }
  async function downloadAction(path: string, body: unknown, key: string, notice: string) {
    setActionId(key); setActionError(''); setActionNotice('')
    try {
      await request<DownloadOperation>(path, new AbortController().signal, body)
      setActionNotice(notice); setDownloadAttempt(current => current + 1)
    } catch (reason) { setActionError(errorMessage(reason)); setDownloadAttempt(current => current + 1) }
    finally { setActionId('') }
  }
  const modelLabel = (id: string) => models.find(model => model.model_id === id)?.label ?? id
  const downloadModel = models.find(model => model.model_id === downloadChoice.model_id && model.provider_id === downloadChoice.provider_id)
  const downloadBytes = downloadModel?.variants?.find(variant => variant.precision === downloadChoice.precision)?.download_bytes
  const downloadBlocked = downloadBlockReason(snapshot, workerId, downloadChoice, models)
  const inventoryWorkers = snapshot?.workers.filter(worker => worker.id === workerId) ?? []
  const settingsBlocked = settings && ttsSettingsBlockReason(settings, snapshot, models)
  const canSave = Boolean(settings && !settingsBlocked && !downloadError && !loadError)
  const availableDesignChoices = availableTTSChoices(snapshot, models, 'voice_design')
  const availableCloneChoices = availableTTSChoices(snapshot, models, 'voice_clone')

  return <div className="tts-settings">
    <div className="tts-settings-notice"><strong>TTS設定</strong><p>ボイスデザインとボイスクローンで使用するモデルを個別に設定できます。保存した設定は、次に開始する音声生成に適用されます。</p></div>
    {loadError && <div className="tts-error" role="alert"><p>{loadError}</p><button type="button" onClick={() => setLoadAttempt(current => current + 1)}>設定とモデル一覧を再読み込み</button></div>}
    <section aria-labelledby="tts-voice-settings-heading">
      <h3 id="tts-voice-settings-heading">用途ごとのモデル</h3>
      <p>取得済みで、接続中の実行環境があるモデルと精度を選べます。未取得のモデルは、下の「モデルのダウンロード」から取得してください。</p>
      {!settings ? <p role="status">TTS設定を読み込んでいます。</p> : <form onSubmit={event => { event.preventDefault(); void saveSettings() }}>
        <fieldset className="tts-settings-fields" disabled={settingsBusy}>
          <div className="tts-choice-grid"><ChoiceFields title="ボイスデザイン" choice={settings.voice_design} models={models} availableChoices={availableDesignChoices} onChange={value => changeSettings('voice_design', value)}/><ChoiceFields title="ボイスクローン" choice={settings.voice_clone} models={models} availableChoices={availableCloneChoices} onChange={value => changeSettings('voice_clone', value)}/></div>
        </fieldset>
        {settingsBlocked && <p role="status">{settingsBlocked}</p>}
        {saveError && <p className="tts-error" role="alert">{saveError}</p>}
        {saved && <p role="status">TTS設定を保存しました。次に開始する音声生成に適用されます。</p>}
        <div className="tts-settings-actions"><button type="submit" disabled={settingsBusy || !canSave}>{settingsBusy ? '保存中…' : 'TTS設定を保存'}</button></div>
      </form>}
    </section>
    <section className="tts-download-section" aria-labelledby="tts-download-heading">
      <div className="tts-section-heading"><h3 id="tts-download-heading">モデルのダウンロード</h3><button type="button" disabled={Boolean(actionId)} onClick={() => setDownloadAttempt(current => current + 1)}>保存先の情報を更新</button></div>
      <p>モデルファイルは選択した保存先に保存します。画面を閉じた後も、ダウンロード処理は続きます。</p>
      <form onSubmit={event => { event.preventDefault(); if (!downloadBlocked && validTTSChoice(downloadChoice, models)) void downloadAction(`/api/workers/${encodeURIComponent(workerId)}/tts-downloads`, { model_id: downloadChoice.model_id, precision: downloadChoice.precision }, 'start', 'ダウンロードを受け付けました。') }}>
        <fieldset className="tts-settings-fields tts-download-fields" disabled={Boolean(actionId)}>
          <label>保存先<select value={workerId} onChange={event => setWorkerId(event.target.value)}><option value="">保存先を選択</option>{snapshot?.workers.filter(eligibleDownloadWorker).map(worker => <option key={worker.id} value={worker.id}>{worker.name}</option>)}</select></label>
          <ChoiceFields title="ダウンロードするモデル" choice={downloadChoice} models={models} onChange={setDownloadChoice}/>
        </fieldset>
        <p className="tts-file-note">fp32 と bf16 は同じ公式ウェイトを取得します。int8 と int4 はそれぞれの公式量子化ウェイトを取得します。</p>
        {downloadModel && <p className="tts-source">配布元：<a href={downloadModel.source_url} target="_blank" rel="noreferrer">{downloadModel.label}</a> · ライセンス：{downloadModel.license}{downloadBytes != null && <> · 取得容量：{formatBytes(downloadBytes)}</>}</p>}
        {downloadBlocked && <p role="status">{downloadBlocked}</p>}
        <div className="tts-settings-actions"><button type="submit" disabled={Boolean(actionId) || Boolean(downloadBlocked) || !validTTSChoice(downloadChoice, models)}>{actionId === 'start' ? '受付中…' : 'ダウンロードを開始'}</button></div>
      </form>
      {downloadError && <p className="tts-error" role="alert">{downloadError}</p>}
      {actionError && <p className="tts-error" role="alert">{actionError}</p>}
      {actionNotice && <p role="status">{actionNotice}</p>}
      {!snapshot && !downloadError && <p role="status">保存先とダウンロード状況を読み込んでいます。</p>}
      {snapshot && <>
        <h4>ダウンロード状況</h4>
        {!snapshot.operations.length ? <p>ダウンロードの履歴はありません。</p> : <ul className="tts-download-list">{[...snapshot.operations].sort((a, b) => b.created_at.localeCompare(a.created_at)).map(operation => {
          const worker = snapshot.workers.find(worker => worker.id === operation.worker_id)
          const percent = operation.total_bytes && operation.total_bytes > 0 ? Math.min(100, Math.max(0, operation.done_bytes / operation.total_bytes * 100)) : null
          const active = activeDownload(operation)
          return <li key={operation.id}><div className="tts-download-title"><strong>{modelLabel(operation.model_id)} · {operation.precision}</strong><span role="status">{statusLabels[operation.status] ?? operation.status}</span></div>
            <p className="tts-download-worker">{worker?.name ?? operation.worker_id}{worker && !worker.online ? '（未接続）' : ''}</p>
            {(active || operation.status !== 'completed') && <><progress aria-label={`${modelLabel(operation.model_id)} ${operation.precision}のダウンロード進捗`} {...(percent === null ? {} : { value: percent, max: 100 })}/><p className="tts-download-progress">{formatBytes(operation.done_bytes)} / {formatBytes(operation.total_bytes)}{percent !== null && `（${Math.floor(percent)}%）`}</p></>}
            {operation.current_file && <p className="tts-download-file" title={operation.current_file}>{operation.current_file}</p>}
            {operation.error && <p className="tts-error">{operation.error}</p>}
            {operation.status === 'completed' && <p className="tts-download-complete">ファイル取得済み · 実行できるモデルは、用途ごとの設定から選択できます。</p>}
            {active && <div className="tts-operation-actions"><button type="button" disabled={Boolean(actionId) || operation.cancel_requested || operation.status === 'cancelling'} onClick={() => void downloadAction(`/api/tts-downloads/${encodeURIComponent(operation.id)}/cancel`, {}, operation.id, 'ダウンロードの中断を受け付けました。')}>{operation.cancel_requested || operation.status === 'cancelling' ? '中断処理中…' : '中断'}</button></div>}
            {resumableDownload(operation) && <div className="tts-operation-actions"><button type="button" disabled={Boolean(actionId) || !snapshot.workers.some(worker => worker.id === workerId && eligibleDownloadWorker(worker))} onClick={() => void downloadAction(`/api/tts-downloads/${encodeURIComponent(operation.id)}/resume`, { worker_id: workerId }, operation.id, '選択した保存先でダウンロードを再開します。')}>選択した保存先で再開</button></div>}
          </li>
        })}</ul>}
        <h4>保存先の取得済みモデルファイル</h4>
        {!workerId ? <p>保存先を選択してください。</p> : !inventoryWorkers.some(worker => worker.inventory.some(item => item.file_download_ready)) ? <p>この保存先に取得済みのモデルはありません。</p> : <ul className="tts-inventory-list">{inventoryWorkers.map(worker => worker.inventory.filter(item => item.file_download_ready).map(item => <li key={`${worker.id}-${item.manifest_id}-${item.precision}`}><span>{modelLabel(item.model_id)} · {item.precision}<small>{worker.name}{!worker.online ? '（未接続）' : ''}</small></span><strong>取得済み</strong></li>))}</ul>}
      </>}
    </section>
  </div>
}

function ChoiceFields({ title, choice, models, availableChoices, onChange }: { title: string; choice: TTSChoice; models: TTSModel[]; availableChoices?: TTSChoice[]; onChange: (choice: TTSChoice) => void }) {
  const providers = [...new Set(models.map(model => model.provider_id))]
  const availableModels = models.filter(model => model.provider_id === choice.provider_id)
  const selectedModel = availableModels.find(model => model.model_id === choice.model_id)
  const precisions = selectedModel?.precisions ?? precisionOptions
  const selectableModel = (model: TTSModel) => !availableChoices || availableChoices.some(item => item.provider_id === model.provider_id && item.model_id === model.model_id)
  const selectablePrecision = (precision: Precision) => !availableChoices || availableChoices.some(item => sameTTSChoice(item, { ...choice, precision }))
  const changeModel = (model: TTSModel) => {
    if (!availableChoices) { onChange(chooseTTSModel(choice, model)); return }
    const choices = availableChoices.filter(item => item.provider_id === model.provider_id && item.model_id === model.model_id)
    const next = choices.find(item => item.precision === choice.precision) ?? choices[0]
    if (next) onChange(next)
  }
  const unavailable = availableChoices?.length === 0
  return <fieldset className="tts-model-choice"><legend>{title}</legend>
    <label>TTSエンジン<select value={choice.provider_id} disabled={unavailable} onChange={event => { const model = models.find(model => model.provider_id === event.target.value && selectableModel(model)); if (model) changeModel(model) }}>
      {!providers.includes(choice.provider_id) && <option value={choice.provider_id} disabled>{providerLabel(choice.provider_id)}（未対応）</option>}
      {providers.map(provider => <option value={provider} key={provider} disabled={Boolean(availableChoices && !availableChoices.some(item => item.provider_id === provider))}>{providerLabel(provider)}</option>)}
    </select></label>
    <label>モデル<select value={choice.model_id} disabled={unavailable} onChange={event => { const model = availableModels.find(model => model.model_id === event.target.value && selectableModel(model)); if (model) changeModel(model) }}>
      {!selectedModel && <option value={choice.model_id} disabled>{choice.model_id}（未対応）</option>}
      {availableModels.map(model => <option value={model.model_id} key={model.model_id} disabled={!selectableModel(model)}>{model.label}{!selectableModel(model) && '（利用不可）'}</option>)}
    </select></label>
    <label>精度<select value={choice.precision} disabled={unavailable} onChange={event => { const precision = event.target.value as Precision; if (selectablePrecision(precision)) onChange({ ...choice, precision }) }}>
      {!precisions.includes(choice.precision) && <option value={choice.precision} disabled>{choice.precision}（未対応）</option>}
      {precisions.map(precision => <option value={precision} key={precision} disabled={!selectablePrecision(precision)}>{precision}{!selectablePrecision(precision) && '（利用不可）'}</option>)}
    </select></label>
    {unavailable && <p className="tts-file-note" role="status">選択できる取得済みモデルがありません。</p>}
  </fieldset>
}
