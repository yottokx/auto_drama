export type HistoryEntry = {
  id: string; number: number; label: string; created_at: string
  restored_from_id: string | null; current: boolean
}
export type ProjectHistory = {
  project_id: string; version: number; current_revision_id: string | null; busy: boolean
  entries: HistoryEntry[]
  pending_operation: null | { label: string; status: 'pending' | 'running' | 'failed' }
}
export type RestoreConfirmation = { entry: HistoryEntry; version: number; revision: number }

export function acceptHistory(next: ProjectHistory, projectId: string, current: ProjectHistory | null) {
  return next.project_id === projectId && (!current || current.project_id !== projectId || next.version >= current.version)
}

export function restoreBlockReason(history: ProjectHistory | null, confirmation: RestoreConfirmation, revision: number, busy: boolean, pending: boolean) {
  if (busy || history?.busy) return 'busy'
  if (pending) return 'pending'
  if (!history || history.version !== confirmation.version || confirmation.revision !== revision) return 'stale'
  const entry = history.entries.find(item => item.id === confirmation.entry.id)
  if (!entry) return 'stale'
  if (entry.current || history.current_revision_id === entry.id) return 'current'
  return null
}
