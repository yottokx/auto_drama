import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, errorMessage, request, type Job, type Project } from './api'
import { createInitialWizardDraft, type CharacterBrief, type RevisionRequest, type WizardStep, type WorldBrief } from './wizardState'

export type M2Character = {
  id: string; input: CharacterBrief; result: CharacterBrief | null
  locked: CharacterBrief['locked']; pendingChanges?: boolean
  imagePendingChanges?: boolean; voicePendingChanges?: boolean
  imageArtifactId: string | null; voiceArtifactId: string | null
  imageUrl: string | null; voiceUrl: string | null
  voiceReferenceText?: string | null; voiceTests?: VoiceTest[]
}
export type VoiceTest = { id: string; text: string; artifactId: string; url: string; sourceVoiceArtifactId: string; createdAt: string }
export type RelationshipInput = { characterIds: [string, string]; instruction: string }
export type RelationshipPair = { characterIds: [string, string]; summary: string; firstToSecond: string; secondToFirst: string }
export type Relationships = { result: { pairs: RelationshipPair[] } | null; artifactId: string | null; version: number | null; pendingChanges: boolean }
export type M2Draft = {
  revision: number; step: WizardStep; worldInput: WorldBrief; worldResult: WorldBrief | null
  worldConfirmed: boolean; worldPendingChanges?: boolean; characters: M2Character[]
  approved: boolean; hasProduction?: boolean; planningRequired?: boolean; planApproved?: boolean
  approval: Record<string, unknown> | null; requests: RevisionRequest[]
  activeJobId?: string | null; remainingJobCount?: number
  relationshipInputs?: RelationshipInput[]; relationships?: Relationships
}
export type M2Detail = { project: Project; draft: M2Draft; jobs: Job[] }
export const activeJob = (job: Job) => job.status === 'pending' || job.status === 'running'
const selectedKey = 'auto-drama:m2-selected-project:v1'

export function useM2Api() {
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(() => { try { return localStorage.getItem(selectedKey) } catch { return null } })
  const [detail, setDetail] = useState<M2Detail | null>(null)
  const [loading, setLoading] = useState(true)
  const [connectionError, setConnectionError] = useState('')
  const [actionError, setActionError] = useState('')
  const [mutating, setMutating] = useState(false)
  const [restoreEpoch, setRestoreEpoch] = useState(0)
  const selectedRef = useRef(selectedId)
  const detailRef = useRef(detail)
  const mutationRef = useRef(false)
  const controllers = useRef(new Set<AbortController>())
  const generation = useRef(0)
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; controllers.current.forEach(controller => controller.abort()) } }, [])
  const commit = useCallback((next: M2Detail) => {
    if (selectedRef.current !== next.project.id) return
    // A server upgraded from the four-step flow can still return a saved old step.
    if ((next.draft.step as string) === 'character-input') next = { ...next, draft: { ...next.draft, step: 'world-input' } }
    if (next.draft.step === 'production' && next.draft.planningRequired && !next.draft.planApproved && !next.draft.hasProduction) next = { ...next, draft: { ...next.draft, step: 'planning-review' } }
    // A poll sent before a mutation may return late. Never replace a newer revision.
    if (detailRef.current?.project.id === next.project.id && detailRef.current.draft.revision > next.draft.revision) return
    detailRef.current = next; setDetail(next)
    setProjects(current => [next.project, ...current.filter(project => project.id !== next.project.id)])
  }, [])
  const select = useCallback((id: string | null) => {
    generation.current += 1; selectedRef.current = id; detailRef.current = null
    setSelectedId(id); setDetail(null); setActionError(''); setLoading(Boolean(id))
    try { if (id) localStorage.setItem(selectedKey, id); else localStorage.removeItem(selectedKey) } catch { /* Project data stays on the server. */ }
  }, [])
  const refresh = useCallback(async () => {
    const controller = new AbortController(); controllers.current.add(controller)
    const selection = selectedRef.current; const startGeneration = generation.current
    try {
      const listing = await request<{ projects: Project[] }>('/api/m2/projects', controller.signal)
      if (!mounted.current || controller.signal.aborted) return
      setProjects(listing.projects)
      if (selection && selectedRef.current === selection) {
        const next = await request<M2Detail>(`/api/m2/projects/${encodeURIComponent(selection)}`, controller.signal)
        if (mounted.current && !controller.signal.aborted && !mutationRef.current && startGeneration === generation.current) commit(next)
      }
      if (mounted.current && !controller.signal.aborted) { setConnectionError(''); setLoading(false) }
    } catch (error) {
      if (mounted.current && !controller.signal.aborted) { setConnectionError(errorMessage(error)); setLoading(false) }
    } finally { controllers.current.delete(controller) }
  }, [commit])
  useEffect(() => {
    let stopped = false; let timer: ReturnType<typeof setTimeout>
    async function poll() { await refresh(); if (!stopped) timer = setTimeout(poll, 2000) }
    void poll(); return () => { stopped = true; clearTimeout(timer) }
  }, [refresh, selectedId])
  const mutate = useCallback(async (operation: (signal: AbortSignal) => Promise<M2Detail | null>) => {
    if (mutationRef.current) return null
    mutationRef.current = true; setMutating(true); setActionError(''); generation.current += 1
    const controller = new AbortController(); controllers.current.add(controller)
    try {
      const next = await operation(controller.signal)
      if (next && mounted.current && !controller.signal.aborted) { commit(next); setConnectionError('') }
      return next
    } catch (error) {
      if (mounted.current && !controller.signal.aborted) {
        setActionError(error instanceof ApiError && error.status === 409
          ? `保存済みの状態が更新されています。入力は保持しています。最新の状態を確認して再操作してください。 ${error.message}`
          : `${errorMessage(error)}${!(error instanceof ApiError) || error.status >= 500 ? ' 処理が受理済みの可能性があります。状態を再取得してから再操作してください。' : ''}`)
        void refresh()
      }
      return null
    } finally {
      // Navigation intentionally keeps the data revision. Invalidate any poll
      // started during this request as well, so it cannot restore the old step.
      generation.current += 1
      controllers.current.delete(controller); mutationRef.current = false
      if (mounted.current) setMutating(false)
    }
  }, [commit, refresh])
  const action = useCallback((actionName: string, fields: Record<string, unknown> = {}) => mutate(async signal => {
    const current = detailRef.current
    if (!current) throw new Error('作品の読み込みが完了していません。')
    return request<M2Detail>(`/api/m2/projects/${encodeURIComponent(current.project.id)}/actions`, signal, { expected_revision: current.draft.revision, action: actionName, ...fields })
  }), [mutate])
  const create = useCallback(() => mutate(async signal => {
    const initial = createInitialWizardDraft()
    const next = await request<M2Detail>('/api/m2/projects', signal, { world: initial.world, characters: initial.characters })
    select(next.project.id)
    return next
  }), [mutate, select])
  const retry = useCallback((jobId: string) => mutate(async signal => {
    const current = detailRef.current
    if (!current) return null
    await request(`/api/jobs/${encodeURIComponent(jobId)}/retry`, signal, {})
    return request<M2Detail>(`/api/m2/projects/${encodeURIComponent(current.project.id)}`, signal)
  }), [mutate])
  const restore = useCallback(async (revisionId: string, expectedVersion: number, expectedRevision: number) => {
    const projectId = selectedRef.current
    const restored = await mutate(async signal => {
      const current = detailRef.current
      if (!current || current.project.id !== projectId) throw new Error('作品の読み込みが完了していません。')
      if (current.draft.revision !== expectedRevision) throw new ApiError('確認中に作品が更新されました。復元内容をもう一度確認してください。', 409)
      const next = await request<M2Detail>(`/api/m2/projects/${encodeURIComponent(projectId!)}/history/${encodeURIComponent(revisionId)}/restore`, signal, {
        expected_version: expectedVersion, expected_revision: expectedRevision,
      })
      if (next.project.id !== projectId) throw new Error('復元した作品を確認できませんでした。状態を再取得してください。')
      return next
    })
    if (!restored || !mounted.current || selectedRef.current !== projectId) return null
    setRestoreEpoch(value => value + 1)
    return restored
  }, [mutate])
  return { projects, selectedId, detail, loading, connectionError, actionError, mutating, restoreEpoch, select, refresh, action, create, retry, restore }
}
