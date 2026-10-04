import { useEffect, useRef, useState } from 'react'
import { request } from './api'
import { Icon } from './Icons'
import { AdjustmentEditor } from './AdjustmentEditor'
import { coordinatorNeedsRestart, serverRestartMessage, type CoordinatorHealth } from './productionState'

export function AdjustmentScreen({ projectId, title, onPendingChange, onBack }: {
  projectId: string; title: string; onPendingChange: (pending: boolean) => void; onBack: () => void
}) {
  const heading = useRef<HTMLHeadingElement>(null)
  const [needsServerRestart, setNeedsServerRestart] = useState(false)
  useEffect(() => {
    heading.current?.focus({ preventScroll: true })
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function pollHealth() {
      try {
        const health = await request<CoordinatorHealth>('/api/health', controller.signal)
        if (!controller.signal.aborted) setNeedsServerRestart(coordinatorNeedsRestart(health))
      } catch { /* The editor presents connection errors and preserves unsaved input. */ }
      if (!controller.signal.aborted) timer = setTimeout(pollHealth, 5000)
    }
    void pollHealth()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [projectId])
  return <section aria-labelledby="adjustment-screen-heading">
    <div className="page-heading"><div><div className="eyebrow">AFTER PRODUCTION</div><h1 id="adjustment-screen-heading" ref={heading} tabIndex={-1}>完成後調整</h1><p>{title}</p></div><button className="button button-light" onClick={onBack}><Icon name="back" size={16}/>制作・鑑賞へ戻る</button></div>
    {needsServerRestart && <p className="m2-job-error" role="alert">{serverRestartMessage}</p>}
    <AdjustmentEditor key={projectId} projectId={projectId} disabled={needsServerRestart} onPendingChange={onPendingChange}/>
  </section>
}
