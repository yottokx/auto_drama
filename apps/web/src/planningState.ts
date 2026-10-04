import type { Job } from './api'
import type { CharacterBrief, WizardStep } from './wizardState'

export type CastLife = { character_id: string; personal_concern: string; contact: string; initial_knowledge: string }
export type CastConnection = { character_ids: string[]; relationship: string }
export type PlotEvent = { start_condition: string; steps: { character_id: string; action: string; result: string }[] }
export type PlotRoute = { start_condition: string; attempt: string; consequence: string; choice: string; next_state: string; core_progress: string }
export type PlotChapter = {
  number: number; title: string; role: string; route: PlotRoute; events?: PlotEvent[]
  conversation_topics?: { character_ids: string[]; topic: string; exchange: string }[]
}
export type PlanningContent = {
  cast_plan: { supporting_characters: CharacterBrief[]; everyday_context: CastLife[]; connections: CastConnection[] }
  plot: {
    core: {
      central_question: string; external_resolution: string; relationship_resolution: string
      characters: { character_id: string; initial_behavior: string; enduring_value: string; turning_experience: string; final_behavior: string }[]
      foreshadowing: { setup_chapter: number; payoff_chapter: number; detail: string }[]
    }
    chapters: PlotChapter[]
  }
}
export type Planning = {
  id: string; revision: number; status: 'draft' | 'generating' | 'ready' | 'planned' | 'approved' | 'failed'
  content: PlanningContent | null; approval_id: string | null; active_job_id: string | null
  jobs: Job[]; error: string | null
}
export type PlanningResponse = { project_id: string; planning: Planning | null; legacy_production: boolean }
export type RevisionTarget = 'all' | 'plot' | 'character' | 'relationships'
export type PlanningTab = 'plot' | 'cast'
export type PlanningEditorState = {
  latest: Planning | null; base: Planning | null; content: PlanningContent | null; instruction: string; conflict: boolean
}
export const initialPlanningEditor: PlanningEditorState = { latest: null, base: null, content: null, instruction: '', conflict: false }
export const planningDirty = (state: PlanningEditorState) => JSON.stringify(state.content) !== JSON.stringify(state.base?.content ?? null)

export function planningTabDirty(state: PlanningEditorState, tab: PlanningTab): boolean {
  const key = tab === 'plot' ? 'plot' : 'cast_plan'
  return JSON.stringify(state.content?.[key]) !== JSON.stringify(state.base?.content?.[key])
}

/** Cancelling one tab must keep the other tab's edits, AI instructions, and conflict context. */
export function cancelPlanningTab(state: PlanningEditorState, tab: PlanningTab, baseline: PlanningContent): PlanningEditorState {
  if (!state.content) return state
  const key = tab === 'plot' ? 'plot' : 'cast_plan'
  return { ...state, content: { ...state.content, [key]: baseline[key] } }
}

export function planningDisplayStep(current: WizardStep, server: WizardStep, pending: boolean): WizardStep {
  return (current === 'planning-review' || current === 'production') && pending ? current : server
}

export function planningActivity(planning: Planning | null) {
  const active = planning?.jobs.find(job => job.id === planning.active_job_id)
  return {
    generating: planning?.status === 'generating' || active?.status === 'pending' || active?.status === 'running',
    failed: active?.status === 'failed' ? [active] : [],
  }
}

export function canApprovePlanning(state: PlanningEditorState, blocked = false): boolean {
  return Boolean(state.content && !blocked && !state.conflict && !planningDirty(state)
    && !state.instruction.trim() && !planningActivity(state.latest).generating
    && state.latest && ['ready', 'planned', 'failed'].includes(state.latest.status))
}

/** Polls can report newer content, but must never overwrite local edits or instructions. */
export function receivePlanning(state: PlanningEditorState, next: Planning | null): PlanningEditorState {
  if (next && state.latest?.id === next.id && next.revision < state.latest.revision) return state
  const changed = next?.id !== state.base?.id || next?.revision !== state.base?.revision
  if (changed && (planningDirty(state) || state.instruction.trim())) return { ...state, latest: next, conflict: true }
  if (changed) return { ...state, latest: next, base: next, content: next?.content ?? null, conflict: false }
  return { ...state, latest: next, base: next }
}

export function resetPlanning(state: PlanningEditorState): PlanningEditorState {
  return { latest: state.latest, base: state.latest, content: state.latest?.content ?? null, instruction: '', conflict: false }
}

/** Update a known field while retaining unedited plan data and generation metadata. */
export function editPlanningField(content: PlanningContent, path: (string | number)[], value: unknown): PlanningContent {
  const next = structuredClone(content)
  let parent: any = next
  for (const part of path.slice(0, -1)) parent = parent[part]
  parent[path[path.length - 1]] = value
  return next
}

export function planningAction(state: PlanningEditorState, action: 'generate' | 'save' | 'revise' | 'approve', options: {
  target?: RevisionTarget; characterId?: string; chapterNumber?: number
} = {}) {
  if (state.conflict) throw new Error('最新の全体計画を確認してから操作してください。')
  if (action !== 'generate' && !state.content) throw new Error('全体計画の生成が完了してから操作してください。')
  if (action === 'approve' && (planningDirty(state) || state.instruction.trim())) throw new Error('未保存の変更や修正指示があります。保存するか取り消してから承認してください。')
  const request: Record<string, unknown> = { action, expected_revision: state.base?.revision ?? 0 }
  if (action === 'save') request.content = state.content
  if (action === 'revise') {
    if (planningDirty(state)) throw new Error('直接編集した内容を保存してからAIに修正を指示してください。')
    if (!state.instruction.trim()) throw new Error('修正指示を入力してください。')
    request.target = options.target ?? 'all'
    request.instruction = state.instruction.trim()
    if (request.target === 'character') {
      if (!state.content?.cast_plan.supporting_characters.some(character => character.id === options.characterId)) throw new Error('修正するサブキャラを選択してください。')
      request.character_id = options.characterId
    }
    if (request.target === 'plot' && options.chapterNumber !== undefined) {
      if (!state.content?.plot.chapters.some(chapter => chapter.number === options.chapterNumber)) throw new Error('修正する章を選択してください。')
      request.chapter_number = options.chapterNumber
    }
  }
  return request
}
