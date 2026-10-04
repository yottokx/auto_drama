import type { Job } from './api'
import type { BodyBounds } from './portraitEditorState'
import type { AdjustmentGeometry, PreviewSlot } from './AdjustmentPreview'
import type { GenerationProgressItem } from './productionState'

export type AdjustmentKind = 'image' | 'voice'
export type SceneMusicTransition = {
  visual: 'none' | 'dissolve' | 'fade' | 'cut'; duration_ms: number
  music_fade_out_ms: number; music_fade_in_ms: number
}
export type AdjustmentMusic = {
  production_id: string; scene_id: string; candidate_id: string | null
  action: 'play' | 'continue' | 'stop'; volume: number
  transition?: SceneMusicTransition; reason?: string
}
export type AdjustmentMusicCandidate = {
  id: string; production_id: string; scene_id: string
  source: 'original' | 'generated' | 'upload' | 'reused'; artifact_id: string | null
  music_url: string | null; source_url: string | null; prompt: string | null
  loop_start_seconds: number | null; loop_end_seconds: number | null
  duration_seconds: number | null; source_duration_seconds: number | null
  job_id?: string | null; status?: string; error?: string | null
  quality?: { needs_review?: boolean; near_silence_seconds?: number
    longest_near_silence_seconds?: number; quiet_intervals?: unknown[]
    silence?: { total_seconds?: number; longest_seconds?: number; intervals?: unknown[] }
    flags?: string[]; [key: string]: unknown } | null
}
export type AdjustmentCharacter = {
  character_id: string; image_candidate_id: string | null; voice_candidate_id: string | null
  framing: 'auto' | 'upper_body' | 'full_body'; height_cm: number | null; body_bounds: BodyBounds | null
  offset_y: number; scale: number
}
export type AdjustmentCandidate = {
  id: string; character_id: string; kind: AdjustmentKind; artifact_id: string | null; url: string | null
  reference_text: string | null; source: 'original' | 'generated' | 'upload'; prompt: string | null
  job_id: string | null; sample_url?: string | null; status?: string
  prompt_details?: { source: string; input: string; instruction: string; effective: string | null; baseline: string | null } | null
}
export type AdjustmentDraft = {
  id: string; revision: number; status: 'draft' | 'applying' | 'failed' | 'applied'; error: string | null
  base_edition_id: string | null; characters: AdjustmentCharacter[]; geometry: AdjustmentGeometry
  scene_music?: AdjustmentMusic[]
}
export type AdjustmentScene = {
  chapter_number: number; scene_id: string; title: string; background_url: string | null
  character_ids: string[]; positions?: Record<string, PreviewSlot>
  production_id?: string; music_prompt?: string | null
  music_plan?: { action: AdjustmentMusic['action']; candidate_id: string | null; source_scene_id?: string | null
    transition: SceneMusicTransition; reason: string; prompt?: string } | null
}
export type AdjustmentResponse = {
  project_id: string; complete: boolean; readonly: boolean; busy: boolean; draft: AdjustmentDraft | null
  cast: { character_id: string; name: string; role: 'main' | 'supporting'; chapter_numbers: number[]
    result?: { appearance?: string; voice?: string; selfIntroduction?: string }
    source_prompts?: { image: string; voice: string }
    image_candidate_id: string | null; voice_candidate_id: string | null; image_url: string | null; voice_url: string | null; reference_text: string | null }[]
  scenes: AdjustmentScene[]; candidates: AdjustmentCandidate[]; jobs: Job[]
  music_candidates?: AdjustmentMusicCandidate[]
  edition: { id: string } | null
  limits: { upload_bytes: number; image_max_side: number; audio_min_seconds: number; audio_max_seconds: number
    music_upload_bytes?: number; music_min_seconds?: number; music_max_seconds?: number }
}
export type AdjustmentEditorState = {
  source: AdjustmentResponse | null; base: AdjustmentDraft | null
  characters: AdjustmentCharacter[]; conflict: boolean
  scene_music?: AdjustmentMusic[]
}
export const initialAdjustmentEditor: AdjustmentEditorState = { source: null, base: null, characters: [], conflict: false }
export const adjustmentDirty = (state: AdjustmentEditorState) => (
  JSON.stringify(state.characters) !== JSON.stringify(state.base?.characters ?? [])
  || JSON.stringify(state.scene_music ?? []) !== JSON.stringify(state.base?.scene_music ?? [])
)

export function adjustmentSetting(character: AdjustmentResponse['cast'][number] | undefined, kind: AdjustmentKind): string {
  const setting = character?.result?.[kind === 'image' ? 'appearance' : 'voice']
  return setting?.trim() ? setting : character?.source_prompts?.[kind] ?? ''
}

export function adjustmentSelfIntroduction(character: AdjustmentResponse['cast'][number] | undefined): string {
  const introduction = character?.result?.selfIntroduction
  return introduction?.trim() ? introduction : character?.reference_text ?? ''
}

/** Polls refresh jobs/candidates but preserve local changes until explicitly reloaded. */
export function receiveAdjustment(state: AdjustmentEditorState, source: AdjustmentResponse, mode: 'poll' | 'adopt' | 'candidate' = 'poll'): AdjustmentEditorState {
  if (mode === 'adopt' || !state.base || (!adjustmentDirty(state) && !state.conflict)) {
    return { source, base: source.draft, characters: structuredClone(source.draft?.characters ?? []),
      scene_music: structuredClone(source.draft?.scene_music ?? []), conflict: false }
  }
  if (mode === 'candidate') return { ...state, source, base: source.draft }
  const changed = state.base.id !== source.draft?.id || state.base.revision !== source.draft.revision
  return { ...state, source, conflict: state.conflict || changed }
}

export function changeAdjustment(state: AdjustmentEditorState, characterId: string, patch: Partial<AdjustmentCharacter>): AdjustmentEditorState {
  return { ...state, characters: state.characters.map(character => character.character_id === characterId ? { ...character, ...patch, character_id: characterId } : character) }
}

export function selectAdjustmentCandidate(state: AdjustmentEditorState, candidate: AdjustmentCandidate): AdjustmentEditorState {
  if (!candidate.url || !candidate.artifact_id) return state
  const character = state.characters.find(value => value.character_id === candidate.character_id)
  if (!character) return state
  if (candidate.kind === 'image') {
    if (character.image_candidate_id === candidate.id) return state
    return changeAdjustment(state, candidate.character_id, { image_candidate_id: candidate.id, body_bounds: null, offset_y: 0, scale: 1 })
  }
  return changeAdjustment(state, candidate.character_id, { voice_candidate_id: candidate.id })
}

export function adjustmentPayload(state: AdjustmentEditorState) {
  return { expected_revision: state.base?.revision ?? 0, characters: state.characters,
    ...(state.base?.scene_music !== undefined || state.scene_music?.length
      ? { scene_music: state.scene_music ?? [] } : {}) }
}

export function adjustmentMusicSetting(state: AdjustmentEditorState, scene: AdjustmentScene | undefined): AdjustmentMusic | undefined {
  if (!scene?.production_id) return undefined
  return state.scene_music?.find(value => value.production_id === scene.production_id && value.scene_id === scene.scene_id)
    ?? { production_id: scene.production_id, scene_id: scene.scene_id, candidate_id: null, action: 'stop', volume: 0.35 }
}

export function changeAdjustmentMusic(state: AdjustmentEditorState, scene: AdjustmentScene, patch: Partial<AdjustmentMusic>): AdjustmentEditorState {
  const previous = adjustmentMusicSetting(state, scene)
  if (!previous) return state
  const automaticContinuation = previous.action === 'continue' && scene.music_plan?.action === 'continue'
    && previous.reason === scene.music_plan.reason
    && JSON.stringify(previous.transition) === JSON.stringify(scene.music_plan.transition)
  const fades = automaticContinuation && patch.action && patch.action !== 'continue' && !patch.transition
    ? { transition: { ...sceneMusicTransition(previous), music_fade_out_ms: 1000, music_fade_in_ms: patch.action === 'play' ? 1000 : 0 } } : {}
  const value = { ...previous, ...fades, ...patch, production_id: previous.production_id, scene_id: previous.scene_id }
  const rows = state.scene_music ?? []
  const exists = rows.some(row => row.production_id === value.production_id && row.scene_id === value.scene_id)
  return { ...state, scene_music: exists ? rows.map(row => row.production_id === value.production_id && row.scene_id === value.scene_id ? value : row) : [...rows, value] }
}

export function selectAdjustmentMusicCandidate(state: AdjustmentEditorState, scene: AdjustmentScene, candidate: AdjustmentMusicCandidate): AdjustmentEditorState {
  if (!candidate.music_url || !candidate.artifact_id || candidate.production_id !== scene.production_id || candidate.scene_id !== scene.scene_id) return state
  return changeAdjustmentMusic(state, scene, { candidate_id: candidate.id, action: 'play' })
}

export const defaultSceneMusicTransition: SceneMusicTransition = { visual: 'fade', duration_ms: 500, music_fade_out_ms: 1000, music_fade_in_ms: 1000 }

export function sceneMusicTransition(setting: AdjustmentMusic | undefined): SceneMusicTransition {
  return { ...defaultSceneMusicTransition, ...setting?.transition }
}

export type SceneMusicContinuity = {
  scene: AdjustmentScene; setting: AdjustmentMusic; candidate?: AdjustmentMusicCandidate
  source_scene?: AdjustmentScene; range?: { start: AdjustmentScene; end: AdjustmentScene }; track_volume?: number; error?: string
}

/** Recompute contiguous ranges from the local draft, including unsaved overrides. */
export function adjustmentMusicContinuity(state: AdjustmentEditorState): SceneMusicContinuity[] {
  const result: SceneMusicContinuity[] = []
  let active: Pick<SceneMusicContinuity, 'candidate' | 'source_scene' | 'range' | 'track_volume'> | undefined
  let production = ''
  for (const scene of state.source?.scenes ?? []) {
    const setting = adjustmentMusicSetting(state, scene)
    if (!setting) continue
    if (production !== scene.production_id) active = undefined
    production = scene.production_id ?? ''
    const row: SceneMusicContinuity = { scene, setting }
    if (setting.action === 'play') {
      const candidate = state.source?.music_candidates?.find(value => value.id === setting.candidate_id
        && value.production_id === scene.production_id && value.scene_id === scene.scene_id && value.artifact_id && value.music_url)
      active = candidate ? { candidate, source_scene: scene, range: { start: scene, end: scene }, track_volume: setting.volume } : undefined
      if (!active) row.error = 'この場面で再生する完成済みのBGM候補を選んでください。'
    } else if (setting.action === 'stop') active = undefined
    else if (!active) row.error = '継続するBGMがありません。この章の前の場面で曲を選ぶか、BGMなしにしてください。'
    if (active) {
      active.range!.end = scene
      Object.assign(row, active)
    }
    result.push(row)
  }
  return result
}

export function resetAutomaticSceneMusic(state: AdjustmentEditorState, scene: AdjustmentScene): AdjustmentEditorState {
  const plan = scene.music_plan
  if (!plan) return state
  if (plan.action === 'play' && !state.source?.music_candidates?.some(value => value.id === plan.candidate_id
    && value.production_id === scene.production_id && value.scene_id === scene.scene_id && value.artifact_id && value.music_url)) return state
  return changeAdjustmentMusic(state, scene, { action: plan.action, candidate_id: plan.action === 'play' ? plan.candidate_id : null,
    transition: structuredClone(plan.transition), reason: plan.reason })
}

export function previewCast(scene: AdjustmentScene | undefined, allowed: string[]): Record<string, PreviewSlot> {
  const slots: PreviewSlot[] = ['left', 'center', 'right']
  const ids = (scene?.character_ids ?? []).filter((id, index, all) => allowed.includes(id) && all.indexOf(id) === index).slice(0, 3)
  const selected: Record<string, PreviewSlot> = {}
  for (const id of ids) {
    const preferred = scene?.positions?.[id] ?? (ids.length === 1 ? 'center' : ids.length === 2 && !Object.keys(selected).length ? 'left' : ids.length === 2 ? 'right' : slots[Object.keys(selected).length])
    selected[id] = Object.values(selected).includes(preferred) ? slots.find(slot => !Object.values(selected).includes(slot))! : preferred
  }
  return selected
}

export function togglePreviewCharacter(selected: Record<string, PreviewSlot>, id: string): Record<string, PreviewSlot> {
  if (selected[id]) { const next = { ...selected }; delete next[id]; return next }
  if (Object.keys(selected).length >= 3) return selected
  const slot = (['center', 'left', 'right'] as PreviewSlot[]).find(value => !Object.values(selected).includes(value))!
  return { ...selected, [id]: slot }
}

export function movePreviewCharacter(selected: Record<string, PreviewSlot>, id: string, slot: PreviewSlot): Record<string, PreviewSlot> {
  if (!selected[id]) return selected
  const other = Object.keys(selected).find(key => key !== id && selected[key] === slot)
  return { ...selected, ...(other ? { [other]: selected[id] } : {}), [id]: slot }
}

export function adjustmentUploadError(file: Pick<File, 'size' | 'name' | 'type'>, kind: AdjustmentKind, limit: number): string | null {
  if (file.size <= 0) return '空のファイルは取り込めません。'
  if (file.size > limit) return `ファイルは${Math.floor(limit / 1024 / 1024)}MiB以下にしてください。`
  const extension = file.name.split('.').at(-1)?.toLowerCase() ?? ''
  if (!(kind === 'image' ? ['png', 'webp', 'jpg', 'jpeg'] : ['wav', 'mp3']).includes(extension)) return kind === 'image' ? 'PNG・WebP・JPEGを選んでください。' : 'WAV・MP3を選んでください。'
  return null
}

export function adjustmentJobProgress(source: AdjustmentResponse): GenerationProgressItem[] {
  const groups = [
    { id: 'm3_image', label: '立ち絵の候補', kinds: ['m3_image'] },
    { id: 'm3_voice', label: '基準音声の候補', kinds: ['m3_voice'] },
    { id: 'm3_voice_clone', label: '台詞・試聴の音声', kinds: ['m3_voice_clone', 'm3_dialogue'] },
    { id: 'm3_music_plan', label: 'BGM・場面転換の設計', kinds: ['m3_music_plan'] },
    { id: 'music', label: '場面のBGM候補', kinds: ['music', 'm3_music'] },
    { id: 'music_replan', label: '章のBGM・場面転換の見直し', kinds: [] },
  ]
  return groups.flatMap(group => {
    const jobs = source.jobs.filter(job => group.id === 'music_replan' ? job.purpose === 'music_replan'
      : job.purpose !== 'music_replan' && group.kinds.includes(job.generation_kind ?? job.kind))
    if (!jobs.length) return []
    const completed = jobs.filter(job => job.status === 'completed').length
    const started = jobs.some(job => job.status === 'running' || job.status === 'completed' || job.attempt_count > 0)
    return [{ id: group.id, label: group.label, completed, current: completed, total: jobs.length,
      status: jobs.some(job => job.status === 'failed') ? 'failed' : completed === jobs.length ? 'completed' : started ? 'running' : 'pending' }]
  })
}
