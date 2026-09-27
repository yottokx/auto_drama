import { useEffect, useState } from 'react'
import { errorMessage, request } from './api'

type Plot = {
  outline: {
    chapters: { number: number; title: string; role: string; summary: string }[]
    ending: string
    character_arcs: { character_id: string; change: string }[]
    foreshadowing: { setup_chapter: number; payoff_chapter: number; detail: string }[]
  }
  characters: { id: string; name: string }[]
}
type PlotResponse = {
  project_id: string
  production_id: string | null
  narrative_artifact_id: string | null
  plot: Plot | null
}
type PlotState =
  | { status: 'loading'; artifactId: string }
  | { status: 'ready'; artifactId: string; plot: Plot }
  | { status: 'error'; artifactId: string; message: string }

export function PlotReview({ projectId, productionId, narrativeArtifactId }: {
  projectId: string; productionId: string; narrativeArtifactId: string | null
}) {
  const [open, setOpen] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const [state, setState] = useState<PlotState | null>(null)

  useEffect(() => {
    if (!open || !narrativeArtifactId) return
    const controller = new AbortController()
    setState({ status: 'loading', artifactId: narrativeArtifactId })
    async function load() {
      try {
        const result = await request<PlotResponse>(
          `/api/m3/projects/${encodeURIComponent(projectId)}/plot`, controller.signal,
        )
        if (controller.signal.aborted) return
        if (result.project_id !== projectId || result.production_id !== productionId ||
          result.narrative_artifact_id !== narrativeArtifactId || !result.plot) {
          throw new Error('プロットが更新されています。少し待ってから再読み込みしてください。')
        }
        setState({ status: 'ready', artifactId: narrativeArtifactId!, plot: result.plot })
      } catch (reason) {
        if (!controller.signal.aborted) {
          setState({ status: 'error', artifactId: narrativeArtifactId!, message: errorMessage(reason) })
        }
      }
    }
    void load()
    return () => controller.abort()
  }, [open, projectId, productionId, narrativeArtifactId, attempt])

  const current = state?.artifactId === narrativeArtifactId ? state : null
  const plot = current?.status === 'ready' ? current.plot : null
  return <details className="production-plot" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>プロットを確認<small>各章のあらすじ・結末・人物の変化・伏線</small></summary>
    {open && <div className="plot-content">
      {!narrativeArtifactId ? <p className="plot-status" role="status">プロットと本文の生成・検査が終わると、ここで確認できます。</p>
        : !current || current.status === 'loading' ? <p className="plot-status" role="status">プロットを読み込み中…</p>
          : current.status === 'error' ? <div className="plot-status"><p className="m2-job-error" role="alert">{current.message}</p><button className="button button-light" onClick={() => setAttempt(value => value + 1)}>プロットを再読み込み</button></div> : null}
      {plot && <>
        <section aria-label="章ごとのあらすじ">
          <h3>章ごとのあらすじ</h3>
          <ol className="plot-chapters">{plot.outline.chapters.map(chapter => <li key={chapter.number}>
            <span className="plot-chapter-number">第{chapter.number}章</span>
            <div><h4>{chapter.title}</h4><p className="plot-chapter-role">{chapter.role}</p><p>{chapter.summary}</p></div>
          </li>)}</ol>
        </section>
        <section aria-label="物語の結末"><h3>物語の結末</h3><p>{plot.outline.ending}</p></section>
        <section aria-label="人物の変化"><h3>人物の変化</h3><dl className="plot-character-arcs">
          {plot.outline.character_arcs.map(arc => <div key={arc.character_id}>
            <dt>{plot.characters.find(character => character.id === arc.character_id)?.name || arc.character_id}</dt>
            <dd>{arc.change}</dd>
          </div>)}
        </dl></section>
        <section aria-label="伏線"><h3>伏線</h3>
          {plot.outline.foreshadowing.length ? <ul className="plot-foreshadowing">
            {plot.outline.foreshadowing.map((item, index) => <li key={index}>
              <span>第{item.setup_chapter}章で提示 → 第{item.payoff_chapter}章で回収</span><p>{item.detail}</p>
            </li>)}
          </ul> : <p>伏線の設定はありません。</p>}
        </section>
      </>}
    </div>}
  </details>
}
