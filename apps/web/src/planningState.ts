import type { Job } from './api'
import type { CharacterBrief, WizardStep } from './wizardState'

export type CastLife = { character_id: string; personal_concern: string; contact: string; initial_knowledge: string }
export type CastConnection = { character_ids: string[]; relationship: string }
export type PlotEvent = { start_condition: string; steps: { character_id: string; action: string; result: string }[] }
export type PlotRoute = { start_condition: string; attempt: string; consequence: string; choice: string; next_state: string; core_progress: string }
export type PlotChapter = {
  number: number; title: string; role: string; route: PlotRoute; events?: PlotEvent[]
  conversation_topics?: { character_ids: string[]; topic: string; exchange: string }[]
  outfit_appearances?: { character_id: string; outfit_id: string; occasion: string }[]
}
export type OutfitNames = Record<string, Record<string, string>>
/** How many chapters the cast and setting were judged to need. */
export type ChapterJudgment = { endurance: string; structure: string; distinct_characters: string[]; chapters: number; reason: string }
export type PlanningContent = {
  cast_plan: { supporting_characters: CharacterBrief[]; everyday_context: CastLife[]; connections: CastConnection[] }
  /** A long-form plan first holds its cast and the judged chapter count; its plot comes at the count chosen. */
  chapter_judgment?: ChapterJudgment
  chapter_count?: number
  plot?: {
    core: {
      central_question: string; external_resolution: string; relationship_resolution: string
      characters: { character_id: string; initial_behavior: string; enduring_value: string; turning_experience: string; final_behavior: string }[]
      foreshadowing: { setup_chapter: number; payoff_chapter: number; detail: string }[]
    }
    chapters: PlotChapter[]
  }
}
export type PlanningAsset = { status: 'missing' | 'pending' | 'running' | 'failed' | 'cancelled' | 'ready' | 'omitted'; url: string | null; job_id: string | null; error: string | null; artifact_id?: string }
export type PlanningAssets = Record<string, { image?: PlanningAsset; voice?: PlanningAsset }>
export type Planning = {
  id: string; revision: number; status: 'draft' | 'generating' | 'ready' | 'planned' | 'approved' | 'failed'
  content: PlanningContent | null; approval_id: string | null; active_job_id: string | null
  jobs: Job[]; error: string | null; assets?: PlanningAssets
  /** How many chapters the current settings allow a long-form plan. */
  chapter_limit?: number
}
/** Supporting portraits and voices are made during the review; approval waits for them. */
export const planningAssetsPending = (planning: Planning | null) => Object.values(planning?.assets ?? {})
  .some(row => [row.image, row.voice].some(asset => asset?.status === 'pending' || asset?.status === 'running'))
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

/** A long-form plan before its chapter count is chosen: only its cast and the judged count. */
export const castStage = (content: PlanningContent | null | undefined) => Boolean(content?.chapter_judgment && !content.plot)

/** What a revision may change: the cast before a long-form plan's count is chosen, then only its plot. */
export function revisionTargets(content: PlanningContent | null | undefined): RevisionTarget[] {
  if (!content?.chapter_judgment) return ['all', 'plot', 'character', 'relationships']
  return content.plot ? ['plot'] : ['character', 'relationships']
}

/** The judged count, held to what the settings allow; the default is the smaller of the two. */
export function chapterChoice(judgment: ChapterJudgment, limit: number) {
  return { suggested: judgment.chapters, limit, initial: Math.min(judgment.chapters, limit), beyondLimit: judgment.chapters > limit }
}

export function canApprovePlanning(state: PlanningEditorState, blocked = false): boolean {
  return Boolean(state.content?.plot && !blocked && !state.conflict && !planningDirty(state)
    && !state.instruction.trim() && !planningActivity(state.latest).generating && !planningAssetsPending(state.latest)
    && state.latest && ['ready', 'planned', 'failed', 'approved'].includes(state.latest.status))
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

export function planningAction(state: PlanningEditorState, action: 'generate' | 'save' | 'revise' | 'approve' | 'plot', options: {
  target?: RevisionTarget; characterId?: string; chapterNumber?: number; chapterCount?: number
} = {}) {
  if (state.conflict) throw new Error('最新の全体計画を確認してから操作してください。')
  if (action !== 'generate' && !state.content) throw new Error('全体計画の生成が完了してから操作してください。')
  if (action === 'approve' && (planningDirty(state) || state.instruction.trim())) throw new Error('未保存の変更や修正指示があります。保存するか取り消してから承認してください。')
  const request: Record<string, unknown> = { action, expected_revision: state.base?.revision ?? 0 }
  if (action === 'save') request.content = state.content
  if (action === 'plot') {
    if (planningDirty(state) || state.instruction.trim()) throw new Error('未保存の変更や修正指示があります。保存するか取り消してから全体プロットを作ってください。')
    if (!Number.isInteger(options.chapterCount) || options.chapterCount! < 1) throw new Error('章数を選んでください。')
    request.chapter_count = options.chapterCount
  }
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
      if (!state.content?.plot?.chapters.some(chapter => chapter.number === options.chapterNumber)) throw new Error('修正する章を選択してください。')
      request.chapter_number = options.chapterNumber
    }
  }
  return request
}
