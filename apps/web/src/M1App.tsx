import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ApiError, errorMessage, request, type Artifact, type Job, type Project, type ProjectDetail, type Worker } from './api'

const jobLabels: Record<Job['status'], string> = { pending: '実行待ち', running: '変換中', completed: '完了', failed: '失敗' }
const artifactLabels: Record<Artifact['kind'], string> = { script: '中間脚本 JSON', background: '背景', character: '立ち絵', audio: '音声', tyrano_export: 'ティラノ出力 ZIP' }
const dateFormatter = new Intl.DateTimeFormat('ja-JP', { dateStyle: 'short', timeStyle: 'short' })
function formatDate(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : dateFormatter.format(date)
}
function formatBytes(bytes: number) {
  return bytes < 1024 ? `${bytes} B` : bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>([])
  const [projectsLoaded, setProjectsLoaded] = useState(false)
  const [projectListError, setProjectListError] = useState('')
  const [workers, setWorkers] = useState<Worker[]>([])
  const [workersLoaded, setWorkersLoaded] = useState(false)
  const [workerError, setWorkerError] = useState('')
  const [health, setHealth] = useState<'checking' | 'ok' | 'failed'>('checking')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<{ id: string; data: ProjectDetail | null; error: string } | null>(null)
  const [title, setTitle] = useState('灯台の約束')
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState('')
  const [retry, setRetry] = useState<{ id: string; pending: boolean; error: string } | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const mutationControllers = useRef(new Set<AbortController>())

  useEffect(() => () => {
    for (const controller of mutationControllers.current) controller.abort()
    mutationControllers.current.clear()
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    async function poll() {
      const results = await Promise.allSettled([
        request<{ status: string }>('/api/health', controller.signal),
        request<{ projects: Project[] }>('/api/projects', controller.signal),
        request<{ workers: Worker[] }>('/api/workers', controller.signal),
      ])
      if (controller.signal.aborted) return
      const [healthResult, projectResult, workerResult] = results
      setHealth(healthResult.status === 'fulfilled' && healthResult.value.status === 'ok' ? 'ok' : 'failed')
      if (projectResult.status === 'fulfilled') {
        setProjects(projectResult.value.projects)
        setProjectsLoaded(true)
        setProjectListError('')
        setSelectedId(current => current ?? projectResult.value.projects[0]?.id ?? null)
      } else setProjectListError(errorMessage(projectResult.reason))
      if (workerResult.status === 'fulfilled') {
        setWorkers(workerResult.value.workers)
        setWorkersLoaded(true)
        setWorkerError('')
      } else setWorkerError(errorMessage(workerResult.reason))
      timer = setTimeout(poll, 2000)
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [refreshKey])

  useEffect(() => {
    if (!selectedId) return
    const projectId = selectedId
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    setDetail(current => current?.id === projectId ? current : { id: projectId, data: null, error: '' })
    async function poll() {
      try {
        const data = await request<ProjectDetail>(`/api/projects/${encodeURIComponent(projectId)}`, controller.signal)
        if (!controller.signal.aborted) setDetail({ id: projectId, data, error: '' })
      } catch (error) {
        if (!controller.signal.aborted) setDetail(current => ({ id: projectId, data: current?.id === projectId ? current.data : null, error: errorMessage(error) }))
      }
      if (!controller.signal.aborted) timer = setTimeout(poll, 2000)
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [selectedId, refreshKey])

  async function createDemo(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!title.trim() || creating) return
    const controller = new AbortController()
    mutationControllers.current.add(controller)
    setCreating(true)
    setCreateError('')
    try {
      const result = await request<{ project: Project; job: Job }>('/api/projects/demo', controller.signal, { title: title.trim() })
      if (controller.signal.aborted) return
      setProjects(current => [result.project, ...current.filter(project => project.id !== result.project.id)])
      setProjectsLoaded(true)
      setSelectedId(result.project.id)
      setRefreshKey(current => current + 1)
    } catch (error) {
      if (!controller.signal.aborted) {
        const uncertain = !(error instanceof ApiError) || error.status >= 500
        setCreateError(errorMessage(error) + (uncertain ? ' 作成済みの可能性があるため、作品一覧を確認してから再操作してください。' : ''))
      }
    } finally {
      mutationControllers.current.delete(controller)
      if (!controller.signal.aborted) setCreating(false)
    }
  }

  async function retryJob(job: Job) {
    if (retry?.pending) return
    const controller = new AbortController()
    mutationControllers.current.add(controller)
    setRetry({ id: job.id, pending: true, error: '' })
    try {
      await request<Job>(`/api/jobs/${encodeURIComponent(job.id)}/retry`, controller.signal, {})
      if (controller.signal.aborted) return
      setRetry(null)
      setRefreshKey(current => current + 1)
    } catch (error) {
      if (!controller.signal.aborted) setRetry({ id: job.id, pending: false, error: errorMessage(error) })
    } finally { mutationControllers.current.delete(controller) }
  }

  const selected = detail?.id === selectedId ? detail : null
  const project = selected?.data?.project
  const artifacts = selected?.data?.artifacts ?? []
  const completedExportIds = new Set(selected?.data?.jobs.filter(job => job.status === 'completed').map(job => job.result_artifact_id))
  const exports = artifacts.filter(artifact => artifact.kind === 'tyrano_export' && completedExportIds.has(artifact.id))
  const scripts = artifacts.filter(artifact => artifact.kind === 'script')
  const sourceAssets = artifacts.filter(artifact => !['script', 'tyrano_export'].includes(artifact.kind))

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand"><span className="brand-mark" aria-hidden="true">灯</span><div><p className="eyebrow">AI AUTO DRAMA</p><h1>AIオートドラマ</h1></div></div>
        <div className={`connection connection-${health}`} role="status"><span className="status-dot" aria-hidden="true" />{health === 'ok' ? '制御サーバー接続中' : health === 'checking' ? '接続を確認中…' : '制御サーバーに接続できません'}</div>
      </header>
      <section className="intro">
        <div><p className="eyebrow">MILESTONE 01 · 基盤と出力</p><h2>小さな物語を、<br className="mobile-break" />作品のかたちに。</h2><p>固定サンプルから、保存・ジョブ実行・ティラノへの書き出しを確認できます。</p></div>
        <span className="stage-badge">固定サンプル / 全1章</span>
      </section>
      <div className="workspace">
        <aside className="sidebar" aria-label="作品の作成と選択">
          <section className="panel create-panel">
            <p className="section-kicker">NEW PROJECT</p><h2>サンプル作品を作成</h2>
            <p className="muted">同じ短い脚本と素材で、新しい作品を保存します。AIによる生成は後続の実装です。</p>
            <form onSubmit={createDemo}>
              <label htmlFor="project-title">作品名</label>
              <input id="project-title" name="title" value={title} onChange={event => setTitle(event.target.value)} maxLength={120} required disabled={creating} />
              <button className="button primary full-width" type="submit" disabled={creating || !title.trim()}>{creating ? '作品を保存中…' : 'サンプルを作成して変換'}</button>
            </form>
            {createError && <p className="error-message" role="alert">{createError}</p>}
          </section>
          <section className="panel library-panel">
            <div className="section-heading"><h2>保存した作品</h2><span className="count">{projects.length}</span></div>
            {projectListError && <p className="error-message" role="alert">{projectListError}{projectsLoaded ? ' 前回取得した一覧を表示しています。' : ''}</p>}
            {!projectsLoaded && !projectListError && <p className="muted" role="status">作品を読み込み中…</p>}
            {projectsLoaded && projects.length === 0 && <p className="empty-copy">まだ作品がありません。<br />最初のサンプルを作成してください。</p>}
            <ul className="project-list">{projects.map(item => <li key={item.id}><button type="button" className={`project-button ${item.id === selectedId ? 'selected' : ''}`} aria-pressed={item.id === selectedId} onClick={() => setSelectedId(item.id)}><span className="project-title">{item.title}</span><span className="project-meta">{formatDate(item.created_at)} · {item.chapter_count}章</span></button></li>)}</ul>
          </section>
          <section className="panel worker-panel">
            <div className="section-heading"><h2>登録ワーカー</h2><span className="count">{workers.length}</span></div>
            {workerError && <p className="error-message" role="alert">ワーカー情報を取得できません。{workersLoaded ? ' 前回取得した情報を表示しています。' : ''}</p>}
            {!workersLoaded && !workerError && <p className="muted" role="status">ワーカーを確認中…</p>}
            {workersLoaded && workers.length === 0 && <p className="muted">ワーカーが未登録です。起動すると待機中のジョブを実行します。</p>}
            <ul className="worker-list">{workers.map(worker => <li key={worker.id}><strong>{worker.name}</strong><span>最終通信 {formatDate(worker.last_seen_at)}</span><span className="capabilities">{worker.capabilities.join(' / ') || '能力の登録なし'}</span></li>)}</ul>
            <details className="worker-help" open={workersLoaded && workers.length === 0 ? true : undefined}><summary>ローカルワーカーの起動方法</summary><p>リポジトリのルートで、別のPowerShellを開いて実行してください。</p><code>.\scripts\start-worker.ps1</code><p>登録履歴は再起動後も残ります。実行されない場合は最終通信とワーカーの起動状態を確認してください。</p></details>
          </section>
        </aside>
        <section className="project-workspace" aria-label="選択した作品の制作状況" aria-busy={Boolean(selectedId && !selected?.data && !selected?.error)}>
          {!selectedId && <div className="panel welcome-panel"><span className="welcome-symbol" aria-hidden="true">01</span><h2>最初の出力を、ここから。</h2><p>サンプル作品を作ると、ジョブの進行と<br />保存された素材・出力をここで確認できます。</p><ol className="steps"><li>作品を保存</li><li>ワーカーで変換</li><li>ZIPをダウンロード</li></ol></div>}
          {selectedId && !selected?.data && !selected?.error && <div className="panel loading-panel" role="status">作品の状態を読み込み中…</div>}
          {selected?.error && <p className="error-message detail-error" role="alert">{selected.error}{selected.data ? ' 前回取得した状態を表示しています。自動で再接続します。' : ' 自動で再接続します。'}</p>}
          {project && selected?.data && <>
            <section className="panel project-overview">
              <div className="section-heading"><p className="section-kicker">PROJECT WORKSPACE</p><span className="subtle-badge">固定サンプル</span></div>
              <h2>{project.title}</h2><p className="muted">{project.instructions}</p>
              <dl className="project-facts"><div><dt>章数</dt><dd>{project.chapter_count}章</dd></div><div><dt>設定の版</dt><dd>v{project.settings_version}</dd></div><div><dt>作成日時</dt><dd>{formatDate(project.created_at)}</dd></div></dl>
            </section>
            <section className="panel jobs-panel">
              <div className="section-heading"><div><p className="section-kicker">PROCESS</p><h2>変換ジョブ</h2></div><span className="live-note">約2秒ごとに更新</span></div>
              <p className="muted">画面を閉じても、起動中のワーカーが処理を続けます。ジョブと成果物は保存されます。</p>
              {selected.data.jobs.length === 0 && <p className="empty-copy">この作品にはまだジョブがありません。</p>}
              <ul className="job-list">{selected.data.jobs.map(job => <li className="job-item" key={job.id}>
                <div className="job-heading"><div><h3>中間JSON → ティラノ変換</h3><p className="muted">試行 {job.attempt_count} / {job.max_attempts} · 更新 {formatDate(job.updated_at)}</p></div><span className={`job-status ${job.status}`} role="status">{jobLabels[job.status]}</span></div>
                {job.status === 'pending' && <p className="job-note">ローカルワーカーによる実行を待っています。</p>}
                {job.status === 'running' && <p className="job-note">脚本と素材を検証し、ティラノソースを組み立てています。</p>}
                {job.status === 'completed' && <p className="job-note">変換が完了しました。確定した出力を下からダウンロードできます。</p>}
                {job.error && <p className="error-message">{job.error}</p>}
                {job.status === 'failed' && <button className="button secondary" type="button" disabled={Boolean(retry?.pending)} onClick={() => void retryJob(job)}>{retry?.id === job.id && retry.pending ? '再試行を登録中…' : 'このジョブを再試行'}</button>}
                {retry?.id === job.id && retry.error && <p className="error-message" role="alert">{retry.error}</p>}
              </li>)}</ul>
            </section>
            <section className="panel output-panel">
              <div className="section-heading"><div><p className="section-kicker">EXPORT</p><h2>保存済みの出力</h2></div><span className="subtle-badge">確定済みファイルのみ</span></div>
              <p className="muted">ZIPにはシナリオ・素材・中間JSONを含みます。ティラノ本体は含みません。</p>
              <div className="download-grid"><div className={`download-card ${exports.length ? 'ready' : ''}`}><span className="file-type">ZIP</span><h3>ティラノ用ソースと素材</h3><p>{exports.length ? 'ご自身のティラノ環境で利用できます。' : '変換完了後にダウンロードできます。'}</p>{exports.map(artifact => <Download key={artifact.id} artifact={artifact} prominent />)}{exports.length === 0 && <span className="waiting-label">出力を待っています</span>}</div><div className="download-card"><span className="file-type">JSON</span><h3>中間脚本</h3><p>表示本文・話者・演出・素材参照を保存した、変換元のデータです。</p>{scripts.map(artifact => <Download key={artifact.id} artifact={artifact} />)}{scripts.length === 0 && <span className="waiting-label">脚本はまだありません</span>}</div></div>
              {sourceAssets.length > 0 && <details className="asset-details"><summary>固定サンプルの素材（{sourceAssets.length}件）</summary><ul className="asset-list">{sourceAssets.map(artifact => <li key={artifact.id}><div><span className="asset-kind">{artifactLabels[artifact.kind]}</span><strong>{artifact.filename}</strong><span className="muted">v{artifact.version} · {formatBytes(artifact.size_bytes)}</span></div><a href={artifact.download_url} download={artifact.filename} aria-label={`${artifact.filename}をダウンロード`}>保存 <span aria-hidden="true">↓</span></a></li>)}</ul></details>}
            </section>
          </>}
        </section>
      </div>
      <footer className="app-footer"><span>AIオートドラマ · M1</span><span>作品・ジョブ・成果物は制御サーバーに保存されます</span></footer>
    </main>
  )
}

function Download({ artifact, prominent = false }: { artifact: Artifact; prominent?: boolean }) {
  return <a className={`button download-button ${prominent ? 'primary' : 'secondary'}`} href={artifact.download_url} download={artifact.filename}><span>{prominent ? 'ZIPをダウンロード' : 'JSONをダウンロード'}<small>v{artifact.version} · {formatBytes(artifact.size_bytes)}</small></span><span aria-hidden="true">↓</span></a>
}
