import type { Job, JobProgressStep } from './api'
import type { EventCgSummary } from './eventCgState'

export type PublishedBuild = { id: string; chapter_number: number; status: 'published' }
export type AssetRequirement = {
  id?: string; kind: string; target_id?: string; artifact_id: string | null; job_id?: string | null
  descriptor?: string | Record<string, unknown>
}
export type GenerationProgressItem = {
  id: string; label: string; status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  current: number; total: number | null; completed: number
  counter?: 'step' | 'none'; statusText?: string; stepIds?: string[]
}
export type ProductionChapter = {
  chapter_number: number; production_id: string | null; narrative_artifact_id: string | null
  status: 'waiting' | 'writing' | 'generating_assets' | 'published' | 'failed'
  jobs: Job[]; requirements?: AssetRequirement[]; build: PublishedBuild | null; error: string | null
  player_url: string | null; export_url: string | null
  music_enabled?: boolean
  event_cg?: EventCgSummary
}
export type Production = {
  id: string; approval_id: string; plan_approval_id?: string | null; chapter_number: number; chapter_count?: number
  narrative_artifact_id: string | null; history_frozen?: boolean
  music_enabled?: boolean
  event_cg?: EventCgSummary
  status: 'pending' | 'running' | 'failed' | 'published'
  stage: 'narrative' | 'assets' | 'building' | 'published'
  control_state?: 'running' | 'stopping' | 'paused' | 'interrupted'
  jobs: Job[]; completed_jobs: number; total_jobs: number; error: string | null
  build: PublishedBuild | null; player_url: string | null; export_url: string | null
  chapters_export_url?: string | null
  chapters?: ProductionChapter[]
}
export type ProductionResponse = { project_id: string; production: Production | null }
export type CoordinatorHealth = { status: string; stage?: string; schema_version?: number }

export const serverRestartMessage = '制御サーバーを再起動してください。現在は旧バージョンのサーバーが動作しているため、続きの章の制作と停止・再開を利用できません。'

export function coordinatorNeedsRestart(health: CoordinatorHealth): boolean {
  return (health.schema_version ?? 0) < 4
}

export const jobNames: Record<string, string> = {
  m3_narrative: 'シーン・台本の生成', m3_background: '背景',
  m3_image: 'サブキャラの立ち絵', m3_voice: 'サブキャラの基準音声',
  m3_voice_clone: '台詞の音声', m3_dialogue: '台詞の音声',
  m3_music_plan: 'BGM・場面転換の設計', m3_music: '場面のBGM', music: '場面のBGM',
  m3_event_cg_budget: '作品全体のCG配分', m3_event_cg_plan: 'イベントCGの選定・指示作成',
  m3_event_cg: 'イベントCGの画像',
}
export const assetPhaseOrder = ['m3_image', 'm3_background', 'm3_voice', 'm3_voice_clone'] as const
export const llmStageNames: Record<JobProgressStep['stage'], string> = {
  supporting_characters: 'サブキャラの設定', relationships: '人物の関係性', story_core: '物語の核心',
  plot: '全体プロット', chapter_plan: '章の構成', scene_plan: 'シーンの構成', script: '台本',
  speech_extraction: '台詞と話者の整理', staging: '演出の設定', validation: '内容・整合性の確認',
  memory: '次章へ引き継ぐ情報の整理', revision: '修正指示の反映',
  cast_plan: 'サブキャラの設定・関係性', plot_plan: '物語の核心・全体プロット', chapter_scene_plan: '章とシーンの構成',
}

export function progressItemLabel(item: GenerationProgressItem): string {
  if (item.counter === 'none') return item.label
  if (item.counter === 'step') return `${item.label} STEP ${item.current}/${item.total}`
  if (item.id === 'm3_music' && item.total === 0) return `${item.label}（0曲）`
  return `${item.label}${item.total === null || item.total === 0 ? '' : `（${item.completed}/${item.total}）`}`
}

const sceneStages = ['script', 'speech_extraction', 'staging'] as const
const sceneStageActivity = ['台本を生成中…', '台詞と話者を整理中…', '演出を設定中…']

/** A scene has three stages; only reported completions count towards completion. */
function sceneProgressItem(steps: { step: JobProgressStep; item: GenerationProgressItem }[], currentStep: string | null, chapterNumber?: number, jobActive = false): GenerationProgressItem {
  const states = sceneStages.map(stage => {
    const matching = steps.filter(({ step }) => step.stage === stage)
    return matching.some(({ item }) => item.status === 'running') ? 'running'
      : matching.some(({ item }) => item.status === 'failed') ? 'failed'
        : matching.length && matching.every(({ item }) => item.status === 'completed') ? 'completed' : 'pending'
  })
  const status = states.includes('running') ? 'running' : states.includes('failed') ? 'failed'
    : states.every(value => value === 'completed') ? 'completed' : 'pending'
  const preparing = status === 'pending' && jobActive && states.includes('completed')
  const active = steps.find(({ step, item }) => step.id === currentStep && item.status === 'running')
  const index = active ? sceneStages.findIndex(stage => stage === active.step.stage)
    : status === 'completed' ? sceneStages.length - 1
      : status === 'running' || status === 'failed' ? states.indexOf(status) : states.findIndex(value => value !== 'completed')
  const first = steps[0].step
  return {
    id: `scene-${first.chapter_number ?? chapterNumber ?? 0}-${first.scene_number}`, label: `シーン${first.scene_number}`,
    status: preparing ? 'running' : status, counter: 'step', current: index + 1, total: sceneStages.length,
    completed: states.filter(value => value === 'completed').length,
    ...(preparing ? { statusText: '次の工程を準備中…' } : status === 'running' ? { statusText: sceneStageActivity[index] } : {}),
    stepIds: steps.map(({ step }) => step.id),
  }
}

/** Scene requests share one row; chapter and planning stages remain separate. */
export function llmProgressItems(job: Job): GenerationProgressItem[] {
  const progress = job.progress
  if (!progress || progress.schema_version !== 1 || !progress.steps.length) return [{
    id: job.id, label: job.kind === 'm3_plan' ? '全体プロット・サブキャラの生成' : jobNames[job.kind] ?? '物語の生成',
    status: job.status, current: 0, total: null, completed: job.status === 'completed' ? 1 : 0,
  }]
  const active = progress.active && job.status === 'running' && progress.attempt === job.attempt_count
  const reported = progress.steps.map(step => {
    let status: GenerationProgressItem['status'] = step.status
    const sceneStage = progress.phase === 'chapter' && step.scene_number && sceneStages.some(stage => stage === step.stage)
    if (status === 'running' && !active) status = job.status === 'failed' ? 'failed' : job.status === 'completed' && !sceneStage ? 'completed' : 'pending'
    const total = Number.isInteger(step.total) && step.total! > 0 ? step.total! : null
    const completed = Math.min(total ?? Infinity, Math.max(0, step.completed ?? (status === 'completed' ? step.scene_number ?? 1 : 0)))
    const current = total === null ? 0 : Math.min(total, completed + (status === 'running' ? 1 : 0))
    return { step, item: { id: step.id, label: `${step.scene_number ? `シーン${step.scene_number}・` : ''}${llmStageNames[step.stage] ?? '物語の生成'}`,
      status, current, total, completed } }
  })
  const completedPlans = progress.steps.filter(step => step.status === 'completed' && ['chapter_scene_plan', 'scene_plan'].includes(step.stage))
  const inferredChapter = progress.chapter_plan?.chapter_number ?? (completedPlans.length === 1 ? completedPlans[0].chapter_number : undefined)
  const sceneKey = (step: JobProgressStep) => progress.phase === 'chapter' && step.scene_number
    && sceneStages.some(stage => stage === step.stage) ? `scene-${step.chapter_number ?? inferredChapter ?? 0}-${step.scene_number}` : null
  const scenes = new Map<string, typeof reported>()
  for (const row of reported) {
    const key = sceneKey(row.step)
    if (key) scenes.set(key, [...(scenes.get(key) ?? []), row])
  }
  const plannedScenes = new Map<string, GenerationProgressItem[]>()
  const plannedIds = new Set<string>()
  if (progress.phase === 'chapter') {
    for (const { step } of reported) {
      if (step.status !== 'completed' || !['chapter_scene_plan', 'scene_plan'].includes(step.stage)) continue
      const number = step.chapter_number ?? progress.chapter_plan?.chapter_number ?? 0
      const sceneReports = reported.filter(row => sceneKey(row.step) && (row.step.chapter_number ?? number) === number)
      const plan = progress.chapter_plan?.chapter_number === number ? progress.chapter_plan : undefined
      // Older workers already report the planned scene count alongside each scene.
      const count = plan?.scene_count ?? Math.max(0, ...sceneReports.map(row => row.step.total ?? 0))
      if (!Number.isInteger(count) || count < 1 || count > 100) continue
      const rows = Array.from({ length: count }, (_, index): GenerationProgressItem => {
        const sceneNumber = index + 1
        const report = sceneReports.find(row => row.step.scene_number === sceneNumber)
        const id = report ? sceneKey(report.step)! : `scene-${number}-${sceneNumber}`
        plannedIds.add(id)
        return scenes.has(id) ? sceneProgressItem(scenes.get(id)!, progress.current_step, number, active) : {
          id, label: `シーン${sceneNumber}`, status: 'pending', counter: 'step', current: 0, total: 3, completed: 0,
        }
      })
      plannedScenes.set(step.id, rows)
    }
  }
  const shown = new Set<string>()
  return reported.flatMap(({ step, item }) => {
    const key = sceneKey(step)
    if (!key) return [item, ...(plannedScenes.get(step.id) ?? [])]
    if (shown.has(key) || plannedIds.has(key)) return []
    shown.add(key)
    return [sceneProgressItem(scenes.get(key)!, progress.current_step, inferredChapter, active)]
  })
}

function supportingRequirement(requirement: AssetRequirement): boolean {
  if (requirement.kind !== 'm3_image' && requirement.kind !== 'm3_voice') return true
  try {
    const descriptor = typeof requirement.descriptor === 'string' ? JSON.parse(requirement.descriptor) : requirement.descriptor
    if (descriptor && typeof descriptor === 'object') return Boolean(descriptor.character_result)
  } catch { /* Older snapshots can omit readable descriptors; their jobs identify generated assets. */ }
  return Boolean(requirement.job_id)
}

function cgDescriptor(requirement: AssetRequirement): Record<string, unknown> {
  try {
    const value = typeof requirement.descriptor === 'string' ? JSON.parse(requirement.descriptor) : requirement.descriptor
    return value && typeof value === 'object' ? value : {}
  } catch { return {} }
}

/** Omission artifacts finish the work but must not be presented as generated images. */
function chapterCgProgress(chapter: ProductionChapter): GenerationProgressItem[] {
  const requirements = chapter.requirements ?? []
  const byId = new Map(chapter.jobs.map(job => [job.id, job]))
  const has = (kind: string) => requirements.some(item => item.kind === kind) || chapter.jobs.some(job => job.kind === kind)
  const enabled = (chapter.event_cg?.max_cgs ?? 0) > 0
  const budgetDone = chapter.event_cg?.budget_completed ?? (chapter.event_cg?.chapter_budget != null)
  const noBudget = budgetDone && chapter.event_cg?.chapter_budget === 0
  const hasPlan = has('m3_event_cg_plan') || enabled && !noBudget
  const hasBudget = has('m3_event_cg_budget') || enabled && chapter.chapter_number === 1
  if (!hasBudget && !hasPlan && !has('m3_event_cg')) return []
  const planReqs = requirements.filter(item => item.kind === 'm3_event_cg_plan')
  const planJobs = chapter.jobs.filter(job => job.kind === 'm3_event_cg_plan')
  const planned = chapter.event_cg?.plan_completed ?? (planReqs.length ? planReqs.every(item => Boolean(item.artifact_id))
    : planJobs.length ? planJobs.every(job => job.status === 'completed') : has('m3_event_cg'))
  const definitions = [
    ...(hasBudget ? [{ id: 'm3_event_cg_budget', kind: 'm3_event_cg_budget', label: jobNames.m3_event_cg_budget }] : []),
    ...(hasPlan ? [{ id: 'm3_event_cg_plan', kind: 'm3_event_cg_plan', label: jobNames.m3_event_cg_plan }] : []),
    ...(hasPlan || has('m3_event_cg') ? [
      { id: 'm3_event_cg_base', kind: 'm3_event_cg', label: '基本CG', variant: false },
      { id: 'm3_event_cg_variant', kind: 'm3_event_cg', label: 'CGの差分', variant: true },
    ] : []),
  ]
  return definitions.map(definition => {
    const images = definition.kind === 'm3_event_cg'
    const variant = 'variant' in definition && definition.variant
    const matching = requirements.filter(item => item.kind === definition.kind && (!images || Boolean(cgDescriptor(item).variant_id) === variant))
    // Without per-image descriptors, old status snapshots can only count the combined image jobs.
    const jobs = chapter.jobs.filter(job => job.kind === definition.kind && (!images || !variant))
    const items = chapter.requirements !== undefined ? matching.map(item => ({ done: Boolean(item.artifact_id), job: byId.get(item.job_id ?? '') }))
      : jobs.map(job => ({ done: job.status === 'completed', job }))
    const acceptedPlan = definition.kind === 'm3_event_cg_budget' ? budgetDone : planned
    const total = images ? variant && chapter.event_cg?.max_variants_per_cg === 0 ? 0 : planned ? items.length : null : Math.max(1, items.length)
    const completed = !images && acceptedPlan ? total ?? 1 : items.filter(item => item.done).length
    const failed = items.some(item => !item.done && item.job?.status === 'failed')
    const started = items.some(item => item.job && (item.job.status === 'running' || item.job.status === 'completed' || item.job.attempt_count > 0))
    const omitted = images ? chapter.event_cg?.omissions?.filter(item => item.cg_id && matching.some(requirement => {
      const descriptor = cgDescriptor(requirement)
      return descriptor.cg_id === item.cg_id && (descriptor.variant_id ?? null) === (item.variant_id ?? null)
    })).length ?? 0 : 0
    const status: GenerationProgressItem['status'] = total === null ? 'pending' : total === 0 ? 'skipped'
      : completed === total ? omitted === total ? 'skipped' : 'completed' : failed ? 'failed' : started ? 'running' : 'pending'
    return { id: definition.id, label: definition.label, current: completed, completed, total, status,
      ...(!images ? { counter: 'none' as const } : {}),
      ...(omitted > 0 ? { statusText: `${completed === total ? '処理済み・' : ''}${omitted}枚省略` } : {}) }
  })
}

/** Requirements include reused assets and work not queued yet; jobs alone do not. */
export function chapterAssetProgress(chapter: ProductionChapter): GenerationProgressItem[] {
  const byId = new Map(chapter.jobs.map(job => [job.id, job]))
  const requirements = chapter.requirements
  const hasMusic = chapter.music_enabled === true || chapter.jobs.some(job => job.kind === 'm3_music_plan' || job.kind === 'm3_music')
    || requirements?.some(item => item.kind === 'm3_music_plan' || item.kind === 'm3_music')
  const phaseOrder: readonly string[] = hasMusic ? [...assetPhaseOrder, 'm3_music_plan', 'm3_music'] : assetPhaseOrder
  const hasMediaJobs = chapter.jobs.some(job => job.kind === 'm3_dialogue' || phaseOrder.includes(job.kind))
  const useRequirements = requirements !== undefined && (requirements.length > 0 || !hasMediaJobs)
  const known = Boolean(chapter.narrative_artifact_id || requirements?.some(item => assetPhaseOrder.some(kind => kind === item.kind) || item.kind === 'm3_dialogue')
    || hasMediaJobs)
  const planRequirements = requirements?.filter(item => item.kind === 'm3_music_plan') ?? []
  const planJobs = chapter.jobs.filter(job => job.kind === 'm3_music_plan')
  const hasTrackWork = Boolean(requirements?.some(item => item.kind === 'm3_music') || chapter.jobs.some(job => job.kind === 'm3_music'))
  // An accepted plan fixes the number of new tracks, including a valid zero.
  // Older snapshots without requirements can still identify a completed plan
  // or existing track jobs; an enabled flag alone never fixes their count.
  const tracksKnown = planRequirements.length > 0 ? planRequirements.every(item => Boolean(item.artifact_id))
    : planJobs.length > 0 ? planJobs.every(job => job.status === 'completed') : hasTrackWork
  const media: GenerationProgressItem[] = phaseOrder.map(kind => {
    const matching = useRequirements ? requirements?.filter(item => item.kind === kind && supportingRequirement(item)) : undefined
    const jobs = chapter.jobs.filter(job => job.kind === kind || (kind === 'm3_voice_clone' && job.kind === 'm3_dialogue'))
    const items = matching !== undefined ? matching.map(item => ({ completed: Boolean(item.artifact_id), job: item.job_id ? byId.get(item.job_id) : undefined }))
      : jobs.map(job => ({ completed: job.status === 'completed', job }))
    const completed = items.filter(item => item.completed).length
    // A media phase remains in progress between completion and the next claim.
    // Reused artifacts alone do not mean this chapter started generating it.
    const started = items.some(item => item.job?.status === 'completed'
      || !item.completed && (item.job?.status === 'running' || (item.job?.attempt_count ?? 0) > 0))
    const failed = items.some(item => !item.completed && item.job?.status === 'failed')
    const total = kind === 'm3_music' ? tracksKnown ? items.length : null
      : kind === 'm3_music_plan' && !items.length ? chapter.music_enabled === true || !hasTrackWork ? 1 : 0
        : known ? items.length : null
    const status = total === null ? 'pending' : total === 0 ? 'skipped' : completed === total ? 'completed'
      : failed ? 'failed' : started ? 'running' : 'pending'
    return { id: kind, label: jobNames[kind], status, current: completed, total, completed,
      ...(kind === 'm3_music_plan' ? { counter: 'none' as const } : {}) }
  })
  const cg = chapterCgProgress(chapter)
  const voiceIndex = media.findIndex(item => item.id === 'm3_voice')
  media.splice(voiceIndex < 0 ? media.length : voiceIndex, 0, ...cg)
  return media
}

function chapterNarrativeProgress(chapter: ProductionChapter, narrative?: Job): GenerationProgressItem[] {
  const progress = narrative?.progress
  const detailed = progress?.phase === 'chapter' && progress.steps.length > 0
  const items: GenerationProgressItem[] = narrative ? llmProgressItems(narrative) : [{
    id: 'narrative', label: chapter.narrative_artifact_id ? 'シーン・台本の生成' : '章とシーンの構成',
    status: chapter.narrative_artifact_id ? 'completed' : 'pending',
    current: 0, total: null, completed: chapter.narrative_artifact_id ? 1 : 0,
  }]
  // Completed legacy work has no detailed reports from which to reconstruct rows.
  if (!detailed && (chapter.narrative_artifact_id || narrative?.status === 'completed')) return items
  if (!detailed && narrative) items[0] = { ...items[0], label: '章とシーンの構成' }
  if (!items.some(item => item.counter === 'step')) {
    const planIds = new Set(progress?.steps.filter(step => ['chapter_scene_plan', 'scene_plan'].includes(step.stage)).map(step => step.id))
    const planIndex = items.reduce((last, item, index) => planIds.has(item.id) ? index : last, items.length - 1)
    items.splice(planIndex + 1, 0, { id: 'scene-creation', label: 'シーンの作成', status: 'pending', current: 0, total: null, completed: 0 })
  }
  const validationIds = new Set(progress?.steps.filter(step => step.stage === 'validation' && !step.scene_number).map(step => step.id))
  const validation = items.find(item => validationIds.has(item.id))
  const rows = items.filter(item => !validationIds.has(item.id))
  const sceneIndex = rows.reduce((last, item, index) => item.counter === 'step' || item.id === 'scene-creation' ? index : last, rows.length - 1)
  rows.splice(sceneIndex + 1, 0, validation ? {
    ...validation, id: 'chapter-validation', stepIds: [...validationIds],
  } : { id: 'chapter-validation', label: '内容・整合性の確認', status: 'pending', current: 0, total: null, completed: 0 })
  return rows
}

export function chapterGenerationProgress(chapter: ProductionChapter): GenerationProgressItem[] {
  if (chapter.status === 'waiting') return []
  const narrative = chapter.jobs.find(job => job.kind === 'm3_narrative')
  const textItems = chapterNarrativeProgress(chapter, narrative)
  const assets = chapterAssetProgress(chapter)
  const assetsReady = chapter.narrative_artifact_id && assets.every(item => item.status === 'completed' || item.status === 'skipped')
  const plan = narrative?.progress?.chapter_plan
  const supportingCount = plan?.chapter_number === chapter.chapter_number ? plan.supporting_character_count : 0
  const visibleAssets = assets.filter(item => !['m3_image', 'm3_voice'].includes(item.id)
    || (item.total === null ? supportingCount > 0 : item.total > 0))
  const mediaPlans = visibleAssets.filter(item => ['m3_music_plan', 'm3_event_cg_budget', 'm3_event_cg_plan'].includes(item.id))
  const budget = mediaPlans.find(item => item.id === 'm3_event_cg_budget')
  const chapterPlans = mediaPlans.filter(item => item !== budget)
  if (chapterPlans.length) {
    const validationIndex = textItems.findIndex(item => item.id === 'chapter-validation')
    textItems.splice(validationIndex < 0 ? textItems.length : validationIndex + 1, 0, ...chapterPlans)
  }
  return [...(budget ? [budget] : []), ...textItems, ...visibleAssets.filter(item => !mediaPlans.includes(item)), { id: 'build', label: '章の検査・組み立て', current: 0, total: null,
    completed: chapter.build ? 1 : 0, status: chapter.build ? 'completed' : assetsReady ? chapter.error ? 'failed' : 'running' : 'pending' }]
}

export function productionCurrentProgress(chapters: ProductionChapter[]): string | null {
  const chapter = chapters.find(item => item.jobs.some(job => job.status === 'running'))
    ?? chapters.find(item => item.status === 'failed')
    ?? chapters.find(item => item.status !== 'waiting' && item.status !== 'published')
  if (!chapter) return null
  const items = chapterGenerationProgress(chapter)
  const activeJob = chapter.jobs.find(job => job.status === 'running')
  const currentStep = activeJob?.progress?.current_step
  const current = activeJob?.progress?.active ? items.find(item => (item.id === currentStep || item.stepIds?.includes(currentStep ?? '')) && item.status === 'running') : null
  const item = current ?? items.find(value => value.status === 'running') ?? items.find(value => value.status === 'failed')
    ?? items.find(value => value.status === 'pending')
  if (!item) return null
  return `${item.id === 'm3_event_cg_budget' ? '' : `第${chapter.chapter_number}章・`}${progressItemLabel(item)}${item.status === 'running' ? item.statusText ? ` ${item.statusText}` : 'を制作中' : item.status === 'failed' ? 'の再試行待ち' : 'の実行待ち'}`
}
export const chapterStatusNames: Record<ProductionChapter['status'], string> = {
  waiting: '前章の本文を待機', writing: '本文を制作中',
  generating_assets: '本文完成・素材を制作中', published: '鑑賞できます', failed: '再試行が必要',
}

export function productionChapters(production: Production | null, chapterCount: number): ProductionChapter[] {
  if (!production) return []
  // Saved one-chapter productions remain viewable after upgrading the server.
  const existing: ProductionChapter[] = (production.chapters ?? [{
    chapter_number: production.chapter_number, production_id: production.id,
    narrative_artifact_id: production.narrative_artifact_id, jobs: production.jobs,
    status: production.build ? 'published' : production.status === 'failed' ? 'failed'
      : production.narrative_artifact_id ? 'generating_assets' : 'writing',
    build: production.build, player_url: production.player_url, export_url: production.export_url,
    error: production.error,
  }]).map(chapter => ({ ...chapter, ...(chapter.music_enabled === undefined && production.music_enabled !== undefined ? { music_enabled: production.music_enabled } : {}) }))
  const count = Math.max(production.chapter_count ?? chapterCount, ...existing.map(chapter => chapter.chapter_number), 1)
  return Array.from({ length: count }, (_, index) => existing.find(chapter => chapter.chapter_number === index + 1) ?? {
    chapter_number: index + 1, production_id: null, narrative_artifact_id: null, status: 'waiting',
    jobs: [], build: null, error: null, player_url: null, export_url: null,
  })
}

export function productionView(production: Production | null, chapterCount: number, needsServerRestart = false) {
  const chapters = productionChapters(production, chapterCount)
  const publishedCount = chapters.filter(chapter => chapter.build?.status === 'published').length
  const complete = chapters.length > 0 && publishedCount === chapters.length
  const historyFrozen = Boolean(production?.history_frozen)
  const controlState = production?.control_state ?? 'running'
  const paused = controlState === 'paused' || controlState === 'interrupted'
  const stopping = controlState === 'stopping'
  const canControl = Boolean(production && !complete && !historyFrozen && !needsServerRestart)
  const canStop = canControl && controlState === 'running'
  const canInterrupt = canControl && (canStop || stopping)
  const canResume = canControl && paused
  const canRetry = canControl && !stopping && !paused
  const title = needsServerRestart ? '制御サーバーの再起動が必要です。'
    : !production ? '承認した世界から、物語を。'
    : complete ? 'すべての章を鑑賞できます。'
      : historyFrozen ? '制作途中の状態を表示しています。'
        : stopping ? '実行中の工程を保存して、停止します。'
          : paused ? (controlState === 'interrupted' ? '制作を中断しています。' : '制作を一時停止しています。')
            : production.status === 'failed' ? '制作の再試行を待っています。'
              : publishedCount ? `完成した${publishedCount}章から鑑賞できます。` : '物語を、制作しています。'
  return { chapters, publishedCount, complete, historyFrozen, controlState, paused, stopping, canStop, canInterrupt, canResume, canRetry, title }
}
