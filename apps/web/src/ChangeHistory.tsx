import { useEffect, useRef, useState } from 'react'
import { errorMessage, request } from './api'
import type { M2Detail } from './m2Api'
import { Icon } from './Icons'
import { acceptHistory, restoreBlockReason, type HistoryEntry, type ProjectHistory, type RestoreConfirmation } from './historyState'
import './change-history.css'

const operationStatuses = { pending: '実行待ち', running: '実行中', failed: '失敗' }
const dateLabel = (value: string) => new Date(value).toLocaleString('ja-JP', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })

export function ChangeHistory({ projectId, revision, busy, pending, actionError, restoreEpoch, onRestore, onNotice }: {
  projectId: string; revision: number; busy: boolean; pending: boolean; actionError: string; restoreEpoch: number
  onRestore: (revisionId: string, expectedVersion: number, expectedRevision: number) => Promise<M2Detail | null>
  onNotice: (message: string) => void
}) {
  const [open, setOpen] = useState(false)
  const [history, setHistory] = useState<ProjectHistory | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [confirmation, setConfirmation] = useState<RestoreConfirmation | null>(null)
  const [restoring, setRestoring] = useState(false)
  const [restoreFailed, setRestoreFailed] = useState(false)
  const historyRef = useRef(history); historyRef.current = history
  const restoringRef = useRef(false)
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])

  useEffect(() => {
    if (!open || restoringRef.current) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    const controller = new AbortController()
    async function poll() {
      if (!historyRef.current) setLoading(true)
      try {
        const next = await request<ProjectHistory>(`/api/m2/projects/${encodeURIComponent(projectId)}/history`, controller.signal)
        if (stopped || controller.signal.aborted) return
        if (next.project_id !== projectId) throw new Error('作品の履歴を確認できませんでした。もう一度読み込んでください。')
        if (acceptHistory(next, projectId, historyRef.current)) { setHistory(next); setError('') }
      } catch (reason) {
        if (!stopped && !controller.signal.aborted) setError(errorMessage(reason))
      } finally {
        if (!stopped) { setLoading(false); timer = setTimeout(poll, 2500) }
      }
    }
    void poll()
    return () => { stopped = true; controller.abort(); clearTimeout(timer) }
  }, [open, projectId, revision, attempt, restoreEpoch, restoring])

  const current = history?.project_id === projectId ? history : null
  const blocked = busy || Boolean(current?.busy) || pending || restoring || loading || Boolean(error)
  const confirmationBlocked = confirmation ? restoreBlockReason(current, confirmation, revision, busy || restoring, pending) : null
  const selectedEntry = confirmation && current?.entries.find(entry => entry.id === confirmation.entry.id)

  function select(entry: HistoryEntry) {
    if (!current || blocked || entry.current || current.current_revision_id === entry.id) return
    setConfirmation({ entry, version: current.version, revision }); setRestoreFailed(false)
  }

  async function restore() {
    if (!confirmation || blocked || confirmationBlocked || restoringRef.current) return
    restoringRef.current = true; setRestoring(true); setRestoreFailed(false)
    try {
      const result = await onRestore(confirmation.entry.id, confirmation.version, confirmation.revision)
      if (!mounted.current) return
      if (result && result.project.id === projectId) {
        setConfirmation(null)
        onNotice(`「${confirmation.entry.label}」の状態に戻しました。変更履歴は残っています。`)
      } else setRestoreFailed(true)
    } finally {
      restoringRef.current = false
      if (mounted.current) { setRestoring(false); setAttempt(value => value + 1) }
    }
  }

  return <details className="change-history" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary><Icon name="refresh" size={15}/>変更履歴・元に戻す</summary>
    {open && <div className="change-history-content" aria-busy={loading || restoring}>
      <p className="change-history-intro">操作ごとに保存した状態へ、作品全体を戻せます。過去の履歴も引き続き残ります。</p>
      {loading && <p role="status">変更履歴を読み込み中…</p>}
      {error && <div className="change-history-error"><p className="m2-job-error" role="alert">{error}</p><button type="button" className="button button-light" disabled={restoring} onClick={() => setAttempt(value => value + 1)}>履歴を再読み込み</button></div>}
      {current?.pending_operation && <p className="change-history-operation" role="status">{current.pending_operation.label}：{operationStatuses[current.pending_operation.status]}{current.pending_operation.status === 'failed' ? '。完了した状態は下の履歴から確認できます。' : '。完了すると履歴に追加されます。'}</p>}
      {(busy || current?.busy) && !restoring && <p className="change-history-hint">実行中の処理が終わると、過去の状態に戻せます。</p>}
      {pending && <p className="change-history-hint">編集中の内容を保存するか、取り消してから元に戻してください。</p>}
      {current && !current.entries.length && <p>保存された変更履歴はまだありません。</p>}
      {current && <ol className="change-history-list">{[...current.entries].sort((left, right) => right.number - left.number).map(entry => <li key={entry.id}>
        <div className="change-history-entry"><span className="change-history-number">{String(entry.number).padStart(2, '0')}</span><div><strong>{entry.label}</strong><time dateTime={entry.created_at}>{dateLabel(entry.created_at)}</time>{entry.restored_from_id && <small>過去の状態を復元</small>}</div></div>
        {entry.current || current.current_revision_id === entry.id ? <span className="change-history-current">現在の状態</span> : <button type="button" className="button button-light" disabled={blocked} aria-label={`${entry.label}（履歴${entry.number}）に戻す内容を確認`} onClick={() => select(entry)}>この状態に戻す</button>}
      </li>)}</ol>}
      {confirmation && <section className="change-history-confirm" aria-label="元に戻す内容を確認">
        <h3>「{confirmation.entry.label}」の状態に戻します</h3>
        <p>{dateLabel(confirmation.entry.created_at)}の履歴です。世界観・キャラクター・立ち絵・音声・本編と表示設定を、この時点の保存済みの状態にまとめて戻します。</p>
        <p>新しく生成する処理はありません。この後の変更履歴も残り、再び選んで戻せます。</p>
        {confirmationBlocked === 'stale' && <p className="change-history-hint" role="alert">確認中に作品または履歴が更新されました。最新の状態でもう一度、戻す内容を確認してください。</p>}
        {confirmationBlocked === 'current' && <p className="change-history-hint" role="status">選んだ履歴は、現在の状態です。</p>}
        {restoreFailed && <p className="m2-job-error" role="alert">{actionError || '元に戻せませんでした。選んだ履歴は保持しています。状態を確認して再操作してください。'}</p>}
        <div className="change-history-confirm-actions"><button type="button" className="button button-light" disabled={restoring} onClick={() => { setConfirmation(null); setRestoreFailed(false) }}>キャンセル</button>
          {confirmationBlocked === 'stale' && selectedEntry ? <button type="button" className="button button-primary" disabled={blocked || selectedEntry.current} onClick={() => select(selectedEntry)}>最新の状態で確認し直す</button> : <button type="button" className="button button-primary" disabled={blocked || Boolean(confirmationBlocked)} onClick={() => void restore()}>{restoring ? '元に戻しています…' : 'この履歴の状態に戻す'}</button>}
        </div>
      </section>}
    </div>}
  </details>
}
