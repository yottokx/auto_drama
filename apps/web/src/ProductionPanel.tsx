import { useEffect, useRef, useState } from 'react'
import { errorMessage, request } from './api'
import { Icon } from './Icons'
import { PlotReview } from './PlotReview'
import { GenerationProgress } from './GenerationProgress'
import { eventCgBudgetCompleted, eventCgImagePlanText, type EventCgSummary } from './eventCgState'
import { chapterGenerationProgress, chapterStatusNames, coordinatorNeedsRestart, jobNames, productionCurrentProgress, productionView, serverRestartMessage, type CoordinatorHealth, type Production, type ProductionResponse } from './productionState'
import './production.css'
import './event-cg.css'

export type { ProductionResponse } from './productionState'

type ProductionPanelProps = { projectId: string; approved: boolean; chapterCount: number; onPendingChange?: (pending: boolean) => void; onOpenAdjustments: () => void }

export function EventCgStatus({ value, chapter = false }: { value: EventCgSummary; chapter?: boolean }) {
  if (value.max_cgs === 0) return chapter ? null : <p className="event-cg-summary">この制作版：イベントCGの自動生成なし</p>
  return <div className="event-cg-summary">
    {!chapter && <strong>この制作版のイベントCG：作品全体で基本CG最大{value.max_cgs}件、1件あたり追加差分最大{value.max_variants_per_cg}枚</strong>}
    {!chapter && (value.budget_omission_reason ? <p>作品全体のCG配分：作成できず省略</p> : eventCgBudgetCompleted(value)
      ? <div className="event-cg-budget"><p>作品全体のCG配分：確定</p><ul aria-label="各章の基本CG枠">{value.chapter_budgets?.map(item => <li key={item.chapter_number}>第{item.chapter_number}章：{item.limit === 0 ? '対象なし' : `最大${item.limit}件`}</li>)}</ul><p>各章の枠内で、台本完成後にCGを入れる場面と差分を選びます。</p></div>
      : <p>作品全体のCG配分：第1章の本文生成前に、全章の基本CG枠を決定します。</p>)}
    {chapter && value.chapter_budget !== undefined && <p>この章の基本CG枠：{value.budget_omission_reason ? 'CG配分を作成できず省略' : value.chapter_budget === null ? '配分待ち' : `最大${value.chapter_budget}件`}</p>}
    {value.budget_omission_reason && <p className="event-cg-warning" role="status">理由：{value.budget_omission_reason}</p>}
    <p>{eventCgImagePlanText(value, chapter)}</p>
    <p>生成済み {value.generated}枚 · 省略 {value.omitted}枚（差分を含む）</p>
    {Boolean(value.planning_omitted) && <p>CG指示の作成を{value.planning_omitted}件省略しました。制作の詳細で理由を確認できます。</p>}
  </div>
}

export function ProductionPanel(props: ProductionPanelProps) {
  return <ProductionPanelContent key={props.projectId} {...props}/>
}

function ProductionPanelContent({ projectId, approved, chapterCount, onPendingChange, onOpenAdjustments }: ProductionPanelProps) {
  const [production, setProduction] = useState<Production | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [needsServerRestart, setNeedsServerRestart] = useState(false)
  const [error, setError] = useState('')
  const [mutating, setMutating] = useState(false)
  const mutation = useRef(false)
  const sequence = useRef(0)
  const controllers = useRef(new Set<AbortController>())
  const mounted = useRef(false)
  const endpoint = `/api/m3/projects/${encodeURIComponent(projectId)}`
  useEffect(() => { onPendingChange?.(mutating) }, [mutating, onPendingChange])
  useEffect(() => () => onPendingChange?.(false), [onPendingChange])

  useEffect(() => {
    mounted.current = true
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      const controller = new AbortController()
      controllers.current.add(controller)
      const current = sequence.current
      try {
        const [health, next] = await Promise.all([
          request<CoordinatorHealth>('/api/health', controller.signal),
          request<ProductionResponse>(endpoint, controller.signal),
        ])
        if (!stopped && !mutation.current && current === sequence.current && next.project_id === projectId) {
          setNeedsServerRestart(coordinatorNeedsRestart(health))
          setProduction(next.production); setLoaded(true); setError('')
        }
      } catch (reason) {
        if (!stopped && !controller.signal.aborted && current === sequence.current) setError(errorMessage(reason))
      } finally {
        controllers.current.delete(controller)
        if (!stopped) timer = setTimeout(poll, 2000)
      }
    }
    void poll()
    return () => {
      stopped = true; mounted.current = false; clearTimeout(timer)
      controllers.current.forEach(controller => controller.abort())
    }
  }, [endpoint, approved])

  async function mutate(path: string, body: Record<string, unknown> = {}) {
    if (mutation.current || needsServerRestart) return
    mutation.current = true; setMutating(true); setError(''); sequence.current += 1
    const controller = new AbortController()
    controllers.current.add(controller)
    try {
      const health = await request<CoordinatorHealth>('/api/health', controller.signal)
      if (coordinatorNeedsRestart(health)) {
        if (mounted.current && !controller.signal.aborted) setNeedsServerRestart(true)
        return
      }
      await request(path, controller.signal, body)
      const next = await request<ProductionResponse>(endpoint, controller.signal)
      if (mounted.current && !controller.signal.aborted && next.project_id === projectId) {
        setProduction(next.production); setLoaded(true)
      }
    } catch (reason) {
      if (mounted.current && !controller.signal.aborted) setError(errorMessage(reason))
    } finally {
      controllers.current.delete(controller); mutation.current = false; sequence.current += 1
      if (mounted.current) setMutating(false)
    }
  }

  const view = productionView(production, chapterCount, needsServerRestart)
  const sequentialChapters = Boolean(production?.plan_approval_id)
  const failed = production?.jobs.filter(job => job.status === 'failed') ?? []
  const queued = production?.jobs.some(job => job.status === 'pending')
  const currentProgress = productionCurrentProgress(view.chapters)
  const jobChapters = new Map(view.chapters.flatMap(chapter => chapter.jobs.map(job => [job.id, chapter.chapter_number] as const)))
  return <section className={`production-panel ${view.publishedCount ? 'production-ready' : ''}`} aria-labelledby="production-heading">
    <div className="production-heading"><span className="production-symbol"><Icon name={view.publishedCount ? 'play' : 'film'} size={25}/></span><div><p className="eyebrow">YOUR STORY · {view.chapters.length || chapterCount} CHAPTERS</p><h2 id="production-heading">{view.title}</h2></div></div>
    <p className="production-description">{needsServerRestart ? '再起動後、この画面で続きの章の制作を再開できます。完成済みの章は保持されます。' : view.historyFrozen ? '履歴から復元した状態です。自動では再開しません。完成した版を使う場合は、ページ下部の「変更履歴・元に戻す」から選んでください。' : view.complete ? 'すべての章の台本・素材・組み立てが完成しました。各章を別のタブで鑑賞・書き出しできます。' : '完成した章から鑑賞でき、鑑賞中も後続の章を制作します。プロットと前章までの本文を引き継いで進めるため、追加の承認は必要ありません。'}</p>
    {needsServerRestart && <p className="m2-job-error" role="alert">{serverRestartMessage}</p>}
    {error && <p className="m2-job-error" role="alert">{error}</p>}
    {!loaded && !error && <p role="status">制作状況を読み込み中…</p>}
    {production?.event_cg && <EventCgStatus value={production.event_cg}/>}
    {production && !view.complete && <div className="production-progress" role="status">
      <span>鑑賞可能 {view.publishedCount} / 全{view.chapters.length}章</span>
      <strong>{needsServerRestart ? '制御サーバーの再起動待ち' : view.historyFrozen ? '自動処理は停止しています' : view.paused ? '再開すると、保存済みの成果物から制作を続けます' : view.stopping ? '新しい工程を開始せず、実行中の工程の終了を待っています' : currentProgress ?? (failed.length ? '失敗した工程の再試行が必要です' : queued ? 'ワーカーの実行待ち' : '章の検査・組み立て')}</strong>
      {!needsServerRestart && !view.historyFrozen && !view.paused && !view.stopping && <p>画面を閉じても、制御サーバーとワーカーが起動していれば制作は続きます。</p>}
    </div>}
    <div className="production-actions">
      {view.publishedCount > 0 && production?.chapters_export_url && <a className="button button-light" href={production.chapters_export_url} download>公開済みの章をまとめて書き出す</a>}
      {loaded && !production && approved && <button className="button button-primary" disabled={needsServerRestart || mutating} onClick={() => void mutate(`${endpoint}/start`)}><Icon name="film" size={17}/>{mutating ? '制作を開始中…' : '承認済みの設定で本編を制作'}</button>}
      {view.canStop && <button className="button button-light" disabled={mutating} onClick={() => void mutate(`${endpoint}/stop`, { mode: 'graceful' })}><Icon name="pause" size={15}/>工程を保存して停止</button>}
      {view.canInterrupt && <button className="button button-light" disabled={mutating} onClick={() => void mutate(`${endpoint}/stop`, { mode: 'immediate' })}>今すぐ中断</button>}
      {view.canResume && <button className="button button-primary" disabled={mutating} onClick={() => void mutate(`${endpoint}/resume`)}><Icon name="play" size={15}/>制作を再開</button>}
      {view.canRetry && failed.map(job => <button key={job.id} className="button button-light" disabled={mutating} onClick={() => void mutate(`/api/jobs/${encodeURIComponent(job.id)}/retry`)}><Icon name="refresh" size={15}/>{jobChapters.has(job.id) ? `第${jobChapters.get(job.id)}章 ` : ''}{jobNames[job.kind] ?? '失敗した工程'}を再試行</button>)}
      {view.canRetry && production?.status === 'failed' && !failed.length && <button className="button button-light" disabled={mutating} onClick={() => void mutate(`${endpoint}/start`)}><Icon name="refresh" size={15}/>章の検査と組み立てを再試行</button>}
    </div>
    {(view.canStop || view.canInterrupt || view.canResume) && <p className="production-control-help">「工程を保存して停止」は、実行中の工程が終わってから停止します。「今すぐ中断」は、実行中の工程を再開時にやり直します。完成済みの成果物と鑑賞中の章は保持します。</p>}
    {view.chapters.length > 0 && <ol className="production-chapters" aria-label="章ごとの制作状況">{view.chapters.map(chapter => {
      const published = chapter.build?.status === 'published'
      const withMusic = chapter.music_enabled === true || chapter.jobs.some(job => job.kind === 'm3_music_plan' || job.kind === 'm3_music')
      const allocatingCg = chapter.chapter_number === 1 && Boolean(chapter.event_cg?.max_cgs && !eventCgBudgetCompleted(chapter.event_cg))
      const allocationStatus = view.paused || view.historyFrozen ? '作品全体のCG配分を停止中'
        : chapter.status === 'failed' ? 'CG配分の再試行が必要'
          : chapter.jobs.some(job => job.kind === 'm3_event_cg_budget' && job.status === 'running') ? '作品全体のCGを配分中' : '作品全体のCG配分待ち'
      return <li key={chapter.chapter_number} className={`production-chapter ${published ? 'chapter-ready' : ''}`}>
        <div className="production-chapter-heading"><h3>第{chapter.chapter_number}章</h3><span className={`chapter-status chapter-status-${chapter.status}`}>{published ? '鑑賞できます' : allocatingCg ? allocationStatus : needsServerRestart && chapter.status === 'waiting' ? '制御サーバーの再起動待ち' : sequentialChapters && chapter.status === 'waiting' ? '前章の完成を待機' : chapterStatusNames[chapter.status]}</span></div>
        <p>{published ? '台本と素材がそろい、鑑賞の準備ができました。' : allocatingCg ? 'まず承認済みの全体プロットから各章の基本CG枠を決めます。配分が確定すると、第1章の本文生成を進めます。' : needsServerRestart && chapter.status === 'waiting' ? '制御サーバーを再起動してから、制作を再開してください。' : chapter.narrative_artifact_id ? withMusic ? '本文は完成しています。BGM・場面転換を設計し、立ち絵・背景・音声と必要な曲をそろえて章を組み立てます。' : '本文は完成しています。サブキャラの立ち絵・背景・基準音声・台詞の音声をそろえて章を組み立てます。' : chapter.status === 'waiting' ? sequentialChapters ? '前の章の台本・素材・組み立てが完成し、鑑賞できるようになってから制作を始めます。' : '前の章の台本が完成すると、制作を始めます。' : '承認したプロットに沿ってシーン・台本・演出を作り、前の章から物語を引き継ぎます。'}</p>
        {!published && chapter.status !== 'waiting' && <GenerationProgress items={chapterGenerationProgress(chapter)} label={`第${chapter.chapter_number}章の制作工程`} paused={view.paused || view.historyFrozen}/>}
        {chapter.event_cg && <EventCgStatus value={chapter.event_cg} chapter/>}
        {published && <div className="production-actions">
          {chapter.player_url && <a className="button button-primary" href={chapter.player_url} target="_blank" rel="noreferrer"><Icon name="play" size={17}/>第{chapter.chapter_number}章を鑑賞する</a>}
          {chapter.export_url && <a className="button button-light" href={chapter.export_url} download>第{chapter.chapter_number}章を書き出す</a>}
        </div>}
      </li>
    })}</ol>}
    {production && view.complete && <section className="production-adjustment-entry" aria-labelledby="adjustment-entry-heading"><div><h3 id="adjustment-entry-heading">完成後調整</h3><p>専用画面で、人物の表示位置・立ち絵・基準音声・場面のBGMを調整できます。</p></div><button className="button button-light" disabled={needsServerRestart || mutating} onClick={onOpenAdjustments}><Icon name="settings" size={16}/>完成後調整を開く<Icon name="arrow" size={16}/></button></section>}
    {production && <PlotReview key={production.id} projectId={projectId} productionId={production.id} narrativeArtifactId={production.narrative_artifact_id}/>}
    {production && <details className="production-details"><summary>制作の詳細を表示</summary><p>{view.historyFrozen && '復元した時点の記録です。現在は自動処理を停止しています。'}物語の内容に触れるエラーが表示される場合があります。</p>{production.error && <p className="m2-job-error">{production.error}</p>}{view.chapters.map(chapter => <div key={chapter.chapter_number}><h3>第{chapter.chapter_number}章</h3>{chapter.error && <p className="m2-job-error">{chapter.error}</p>}{Boolean(chapter.event_cg?.omissions?.length) && <div><h4>イベントCGの省略理由</h4><ul>{chapter.event_cg!.omissions!.map((item, index) => <li key={`${item.cg_id ?? 'plan'}-${item.variant_id ?? 'base'}-${index}`}><span>{item.variant_id ? 'CGの差分' : item.cg_id ? '基本CG' : 'CG計画'}{item.cg_id ? `（${item.cg_id}）` : ''}</span><p>{item.reason}</p></li>)}</ul></div>}{chapter.build && <GenerationProgress items={chapterGenerationProgress(chapter)} label={`第${chapter.chapter_number}章の完了した制作工程`}/>}<ul>{chapter.jobs.filter(job => job.status === 'failed').map(job => <li key={job.id}><span>{jobNames[job.kind] ?? '章の素材'}</span><strong>再試行が必要</strong>{job.error && <p className="m2-job-error">{job.error}</p>}</li>)}</ul>{chapter.status === 'waiting' && <p>前の章の制作を待っています。</p>}</div>)}</details>}
    {!needsServerRestart && <p className="production-scope">全{view.chapters.length || chapterCount}章を順に制作・公開します。鑑賞中の章の内容は変わりません。次章が未完成の場合も、プレイヤーで閲覧位置を保存できます。書き出しには各章の中間脚本・素材・manifestを含みます。</p>}
  </section>
}
