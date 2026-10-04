import { progressItemLabel, type GenerationProgressItem } from './productionState'
import './generationProgress.css'

export function GenerationProgress({ items, label, paused = false }: {
  items: GenerationProgressItem[]; label: string; paused?: boolean
}) {
  const statuses = { pending: '待機中', running: '制作中', completed: '完了', failed: '再試行が必要', skipped: '対象なし' }
  return <ol className="generation-progress-list" aria-label={label}>{items.map(item => <li key={item.id} className={`generation-progress-item generation-progress-${paused && (item.status === 'running' || item.status === 'pending') ? 'paused' : item.status}`}>
    <span className="generation-progress-marker" aria-hidden="true">{item.status === 'completed' ? '✓' : item.status === 'failed' ? '!' : '·'}</span>
    <span className="generation-progress-label">{progressItemLabel(item)}</span>
    <span className="generation-progress-status">{paused && (item.status === 'running' || item.status === 'pending') ? '停止中' : item.statusText ?? statuses[item.status]}</span>
  </li>)}</ol>
}
