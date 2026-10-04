import type { Job } from './api'
import type { BodyBounds } from './portraitEditorState'
import type { AdjustmentGeometry, PreviewSlot } from './AdjustmentPreview'
import type { GenerationProgressItem } from './productionState'

export type AdjustmentKind = 'image' | 'voice'
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
}
export type AdjustmentScene = {
  chapter_number: number; scene_id: string; title: string; background_url: string | null
  character_ids: string[]; positions?: Record<string, PreviewSlot>
}
export type AdjustmentResponse = {
  project_id: string; complete: boolean; readonly: boolean; busy: boolean; draft: AdjustmentDraft | null
  cast: { character_id: string; name: string; role: 'main' | 'supporting'; chapter_numbers: number[]
    result?: { appearance?: string; voice?: string; selfIntroduction?: string }
    source_prompts?: { image: string; voice: string }
    image_candidate_id: string | null; voice_candidate_id: string | null; image_url: string | null; voice_url: string | null; reference_text: string | null }[]
  scenes: AdjustmentScene[]; candidates: AdjustmentCandidate[]; jobs: Job[]
  edition: { id: string } | null
  limits: { upload_bytes: number; image_max_side: number; audio_min_seconds: number; audio_max_seconds: number }
}
export type AdjustmentEditorState = {
  source: AdjustmentResponse | null; base: AdjustmentDraft | null
  characters: AdjustmentCharacter[]; conflict: boolean
}
export const initialAdjustmentEditor: AdjustmentEditorState = { source: null, base: null, characters: [], conflict: false }
export const adjustmentDirty = (state: AdjustmentEditorState) => JSON.stringify(state.characters) !== JSON.stringify(state.base?.characters ?? [])

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
    return { source, base: source.draft, characters: structuredClone(source.draft?.characters ?? []), conflict: false }
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
  return { expected_revision: state.base?.revision ?? 0, characters: state.characters }
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
  ]
  return groups.flatMap(group => {
    const jobs = source.jobs.filter(job => group.kinds.includes(job.kind))
    if (!jobs.length) return []
    const completed = jobs.filter(job => job.status === 'completed').length
    const started = jobs.some(job => job.status === 'running' || job.status === 'completed' || job.attempt_count > 0)
    return [{ id: group.id, label: group.label, completed, current: completed, total: jobs.length,
      status: jobs.some(job => job.status === 'failed') ? 'failed' : completed === jobs.length ? 'completed' : started ? 'running' : 'pending' }]
  })
}
