import { useCallback, useEffect, useRef, useState } from 'react'
import type { RelationshipInput } from './m2Api'

export type WizardStep = 'world-input' | 'world-review' | 'character-review' | 'planning-review' | 'production'
export type RevisionScope = 'world' | 'all' | 'settings' | 'appearance' | 'voice' | 'image-retake' | 'voice-retake'
export type CharacterScope = 'settings' | 'appearance' | 'voice'

export interface CharacterBrief {
  id: string
  name: string
  age: string
  gender: string
  role: string
  freeform: string
  settings: string
  appearance: string
  height_cm?: number | null
  body_type?: 'humanoid' | 'nonhumanoid' | 'unknown'
  voice: string
  selfIntroduction?: string
  sampleLines?: string[]
  locked: Record<CharacterScope, boolean>
}

export interface WorldBrief {
  title: string
  prompt: string
  genre: string
  mood: string
  notes: string
  chapterCount: number
  setting: string
}

export interface RevisionRequest {
  id: string
  characterId?: string
  scope: RevisionScope
  instruction: string
  createdAt: string
  protectedScopes: string[]
}

export interface WizardDraft {
  world: WorldBrief
  characters: CharacterBrief[]
  relationshipInputs?: RelationshipInput[]
  worldConfirmed: boolean
  approved: boolean
  planApproved?: boolean
  step: WizardStep
  requests: RevisionRequest[]
  revision: number
}

export const WIZARD_STORAGE_KEY = 'auto-drama:m2-wizard:v1'
const LEGACY_STORAGE_KEY = 'auto-drama:m2-ui-preview:v1'
const STORAGE_VERSION = 1
const characterScopes: CharacterScope[] = ['settings', 'appearance', 'voice']
const steps: WizardStep[] = ['world-input', 'world-review', 'character-review', 'planning-review', 'production']
const revisionScopes: RevisionScope[] = ['world', 'all', 'settings', 'appearance', 'voice', 'image-retake', 'voice-retake']
const worldTextFields = ['title', 'prompt', 'genre', 'mood', 'notes', 'setting'] as const
const characterTextFields = ['name', 'age', 'gender', 'role', 'freeform', 'settings', 'appearance', 'voice'] as const
const bodyTypes = ['humanoid', 'nonhumanoid', 'unknown'] as const
type CharacterPatch = Partial<Omit<CharacterBrief, 'id' | 'locked'>>

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function positiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
}

function isCharacterHeight(value: unknown): value is number | null {
  return value === null || (typeof value === 'number' && Number.isFinite(value) && value >= 1 && value <= 10000)
}

function isBodyType(value: unknown): value is NonNullable<CharacterBrief['body_type']> {
  return bodyTypes.some(type => value === type)
}

function isStep(value: unknown): value is WizardStep {
  return typeof value === 'string' && steps.includes(value as WizardStep)
}

function accessibleStep(step: unknown, worldConfirmed: boolean, approved: boolean, planApproved = false): WizardStep {
  if (step === 'planning-review' || step === 'production') {
    if (!worldConfirmed) return 'world-review'
    if (!approved) return 'character-review'
    return step === 'production' && planApproved ? 'production' : 'planning-review'
  }
  if (step === 'character-review' && !worldConfirmed) return 'world-review'
  return isStep(step) ? step : 'world-input'
}

export function wizardStepLocked(step: WizardStep, draft: {
  worldConfirmed: boolean; approved: boolean; planApproved?: boolean; planningRequired?: boolean; hasProduction?: boolean
}): boolean {
  if (step === 'character-review') return !draft.worldConfirmed
  if (step === 'planning-review') return !draft.approved
  if (step === 'production') return !draft.hasProduction && (!draft.approved || (draft.planningRequired !== false && !draft.planApproved))
  return false
}

export function wizardSetupBusy(jobs: { kind: string; status: string }[]): boolean {
  return jobs.some(job => job.kind.startsWith('m2_') && (job.status === 'pending' || job.status === 'running'))
}

function isScope(value: unknown): value is RevisionScope {
  return typeof value === 'string' && revisionScopes.includes(value as RevisionScope)
}

export function createCharacterBrief(id: string): CharacterBrief {
  return {
    id, name: '', age: '', gender: '', role: '', freeform: '', settings: '', appearance: '', height_cm: null, body_type: 'unknown', voice: '', selfIntroduction: '', sampleLines: [],
    locked: { settings: false, appearance: false, voice: false },
  }
}

export function createInitialWizardDraft(): WizardDraft {
  return {
    world: { title: '', prompt: '', genre: '', mood: '', notes: '', chapterCount: 3, setting: '' },
    characters: [createCharacterBrief('character-1')],
    relationshipInputs: [],
    worldConfirmed: false,
    approved: false,
    planApproved: false,
    step: 'world-input',
    requests: [],
    revision: 1,
  }
}

/** Validate persisted data and retain only the fields the wizard understands. */
export function readWizardDraft(value: unknown): WizardDraft | null {
  if (!isRecord(value) || value.version !== STORAGE_VERSION || !isRecord(value.draft)) return null
  const draft = value.draft
  if (!isRecord(draft.world) || !Array.isArray(draft.characters) || draft.characters.length === 0) return null
  const world = draft.world
  if (!worldTextFields.every(key => typeof world[key] === 'string') || !positiveInteger(world.chapterCount)) return null
  if (typeof draft.worldConfirmed !== 'boolean' || typeof draft.approved !== 'boolean') return null
  if (!positiveInteger(draft.revision) || !Array.isArray(draft.requests)) return null

  const characters: CharacterBrief[] = []
  const characterIds = new Set<string>()
  for (const value of draft.characters) {
    if (!isRecord(value) || typeof value.id !== 'string' || !value.id.trim() || characterIds.has(value.id)) return null
    if (!characterTextFields.every(key => typeof value[key] === 'string') || !isRecord(value.locked)) return null
    if (value.height_cm !== undefined && !isCharacterHeight(value.height_cm)) return null
    if (value.body_type !== undefined && !isBodyType(value.body_type)) return null
    const locked = value.locked
    if (!characterScopes.every(key => typeof locked[key] === 'boolean')) return null
    characterIds.add(value.id)
    characters.push({
      id: value.id,
      name: value.name as string,
      age: value.age as string,
      gender: value.gender as string,
      role: value.role as string,
      freeform: value.freeform as string,
      settings: value.settings as string,
      appearance: value.appearance as string,
      height_cm: value.height_cm === undefined ? null : value.height_cm as number | null,
      body_type: value.body_type === undefined ? 'unknown' : value.body_type as CharacterBrief['body_type'],
      voice: value.voice as string,
      ...(typeof value.selfIntroduction === 'string' ? { selfIntroduction: value.selfIntroduction } : {}),
      ...(Array.isArray(value.sampleLines) && value.sampleLines.every(line => typeof line === 'string') ? { sampleLines: value.sampleLines as string[] } : {}),
      locked: { settings: locked.settings as boolean, appearance: locked.appearance as boolean, voice: locked.voice as boolean },
    })
  }

  const requests: RevisionRequest[] = []
  const requestIds = new Set<string>()
  for (const value of draft.requests) {
    if (!isRecord(value) || typeof value.id !== 'string' || !value.id.trim() || requestIds.has(value.id)) return null
    if (!isScope(value.scope) || typeof value.instruction !== 'string' || !value.instruction.trim()) return null
    if (typeof value.createdAt !== 'string' || !Number.isFinite(Date.parse(value.createdAt))) return null
    if (!Array.isArray(value.protectedScopes) || !value.protectedScopes.every(scope => characterScopes.includes(scope as CharacterScope))) return null
    if (value.scope !== 'world' && (typeof value.characterId !== 'string' || !characterIds.has(value.characterId))) return null
    requestIds.add(value.id)
    requests.push({
      id: value.id,
      ...(value.scope !== 'world' ? { characterId: value.characterId as string } : {}),
      scope: value.scope,
      instruction: value.instruction,
      createdAt: value.createdAt,
      protectedScopes: [...new Set(value.protectedScopes)] as string[],
    })
  }

  const relationshipInputs: RelationshipInput[] = []
  if (draft.relationshipInputs !== undefined) {
    if (!Array.isArray(draft.relationshipInputs)) return null
    for (const input of draft.relationshipInputs) {
      if (!isRecord(input) || !Array.isArray(input.characterIds) || input.characterIds.length !== 2 || input.characterIds[0] === input.characterIds[1] || !input.characterIds.every(id => typeof id === 'string' && characterIds.has(id)) || typeof input.instruction !== 'string') return null
      relationshipInputs.push({ characterIds: [...input.characterIds] as [string, string], instruction: input.instruction })
    }
  }
  return {
    world: {
      title: world.title as string, prompt: world.prompt as string, genre: world.genre as string,
      mood: world.mood as string, notes: world.notes as string, setting: world.setting as string,
      chapterCount: world.chapterCount,
    },
    characters,
    relationshipInputs,
    worldConfirmed: draft.worldConfirmed,
    approved: draft.worldConfirmed && draft.approved,
    planApproved: draft.worldConfirmed && draft.approved && draft.planApproved === true,
    step: accessibleStep(draft.step, draft.worldConfirmed, draft.approved, draft.planApproved === true),
    requests,
    revision: draft.revision,
  }
}

/** Old preview values are carried forward without interpreting or completing them. */
export function migrateLegacyDraft(value: unknown): WizardDraft | null {
  if (!isRecord(value) || value.version !== 1 || !isRecord(value.draft)) return null
  const legacy = value.draft
  const legacyStrings = ['title', 'prompt', 'genre', 'mood', 'world', 'characterName', 'characterRole', 'characterDescription', 'voiceDescription'] as const
  if (!legacyStrings.every(key => typeof legacy[key] === 'string') || !positiveInteger(legacy.chapterCount)) return null
  const initial = createInitialWizardDraft()
  const oldLocks = isRecord(legacy.locked) ? legacy.locked : {}
  return {
    ...initial,
    world: {
      ...initial.world,
      title: legacy.title as string,
      prompt: legacy.prompt as string,
      genre: legacy.genre as string,
      mood: legacy.mood as string,
      setting: legacy.world as string,
      chapterCount: legacy.chapterCount,
    },
    characters: [{
      ...initial.characters[0],
      name: legacy.characterName as string,
      role: legacy.characterRole as string,
      settings: legacy.characterDescription as string,
      voice: legacy.voiceDescription as string,
      locked: { settings: oldLocks.character === true, appearance: oldLocks.appearance === true, voice: oldLocks.voice === true },
    }],
    revision: positiveInteger(legacy.revision) ? legacy.revision : 1,
  }
}

export function updateWorldDraft(draft: WizardDraft, patch: Partial<WorldBrief>): WizardDraft {
  const world = { ...draft.world }
  for (const key of worldTextFields) if (typeof patch[key] === 'string') world[key] = patch[key]
  if (positiveInteger(patch.chapterCount)) world.chapterCount = patch.chapterCount
  if (worldTextFields.every(key => world[key] === draft.world[key]) && world.chapterCount === draft.world.chapterCount) return draft
  return {
    ...draft, world, worldConfirmed: false, approved: false, planApproved: false, revision: draft.revision + 1,
    step: draft.step === 'character-review' || draft.step === 'planning-review' || draft.step === 'production' ? 'world-input' : draft.step,
  }
}

export function confirmWorldDraft(draft: WizardDraft): WizardDraft {
  return { ...draft, worldConfirmed: true, step: 'character-review' }
}

export function goToWizardStep(draft: WizardDraft, step: WizardStep): WizardDraft {
  const nextStep = accessibleStep(step, draft.worldConfirmed, draft.approved, draft.planApproved)
  return draft.step === nextStep ? draft : { ...draft, step: nextStep }
}

export function addCharacterDraft(draft: WizardDraft, id: string): WizardDraft {
  if (!id.trim() || draft.characters.length >= 3 || draft.characters.some(character => character.id === id)) return draft
  return { ...draft, characters: [...draft.characters, createCharacterBrief(id)], worldConfirmed: false, approved: false, planApproved: false, step: accessibleStep(draft.step, false, false), revision: draft.revision + 1 }
}

export function updateCharacterDraft(draft: WizardDraft, id: string, patch: CharacterPatch): WizardDraft {
  const index = draft.characters.findIndex(character => character.id === id)
  if (index === -1) return draft
  const current = draft.characters[index]
  const next = { ...current }
  for (const key of characterTextFields) if (typeof patch[key] === 'string') next[key] = patch[key]
  if (!current.locked.appearance && isCharacterHeight(patch.height_cm)) next.height_cm = patch.height_cm
  if (!current.locked.appearance && isBodyType(patch.body_type)) next.body_type = patch.body_type
  if (!current.locked.settings && !current.locked.voice && typeof patch.selfIntroduction === 'string') next.selfIntroduction = patch.selfIntroduction
  if (!current.locked.settings && patch.sampleLines?.every(line => typeof line === 'string')) next.sampleLines = patch.sampleLines
  if (characterTextFields.every(key => next[key] === current[key]) && next.height_cm === current.height_cm && next.body_type === current.body_type && next.selfIntroduction === current.selfIntroduction && JSON.stringify(next.sampleLines) === JSON.stringify(current.sampleLines)) return draft
  const characters = [...draft.characters]
  characters[index] = next
  const worldConfirmed = draft.step === 'world-input' ? false : draft.worldConfirmed
  return { ...draft, characters, worldConfirmed, approved: false, planApproved: false, step: accessibleStep(draft.step, worldConfirmed, false), revision: draft.revision + 1 }
}

export function removeCharacterDraft(draft: WizardDraft, id: string): WizardDraft {
  if (draft.characters.length <= 1 || !draft.characters.some(character => character.id === id)) return draft
  return {
    ...draft,
    characters: draft.characters.filter(character => character.id !== id),
    relationshipInputs: draft.relationshipInputs?.filter(input => !input.characterIds.includes(id)),
    requests: draft.requests.filter(request => request.characterId !== id),
    worldConfirmed: false,
    approved: false,
    planApproved: false,
    step: accessibleStep(draft.step, false, false),
    revision: draft.revision + 1,
  }
}

export function toggleCharacterLock(draft: WizardDraft, id: string, scope: CharacterScope): WizardDraft {
  if (!draft.characters.some(character => character.id === id)) return draft
  return {
    ...draft,
    characters: draft.characters.map(character => character.id === id
      ? { ...character, locked: { ...character.locked, [scope]: !character.locked[scope] } }
      : character),
    approved: false,
    planApproved: false,
    step: accessibleStep(draft.step, draft.worldConfirmed, false),
    revision: draft.revision + 1,
  }
}

/** Store the requested edit. Generation and interpretation belong to the future backend. */
export function requestDraftRevision(
  draft: WizardDraft,
  scope: RevisionScope,
  instruction: string,
  id: string,
  createdAt: string,
  characterId?: string,
): WizardDraft | null {
  if (!instruction.trim()) return null
  let protectedScopes: CharacterScope[] = []
  if (scope !== 'world') {
    const character = draft.characters.find(candidate => candidate.id === characterId)
    if (!character) return null
    protectedScopes = characterScopes.filter(key => character.locked[key])
    if (scope === 'all' && protectedScopes.length === characterScopes.length) return null
    const target = scope === 'image-retake' ? 'appearance' : scope === 'voice-retake' ? 'voice' : scope
    if (target !== 'all' && character.locked[target]) return null
  }
  return {
    ...draft,
    requests: [...draft.requests, {
      id, scope, instruction: instruction.trim(), createdAt, protectedScopes,
      ...(scope !== 'world' ? { characterId } : {}),
    }],
    approved: false,
    planApproved: false,
    revision: draft.revision + 1,
    ...(scope === 'world' ? { worldConfirmed: false, step: 'world-review' as const } : { step: accessibleStep(draft.step, draft.worldConfirmed, false) }),
  }
}

export function approveDraft(draft: WizardDraft): WizardDraft | null {
  return draft.worldConfirmed && draft.characters.length > 0 ? { ...draft, approved: true, planApproved: false, step: 'planning-review' } : null
}

export function approvePlanDraft(draft: WizardDraft): WizardDraft | null {
  return draft.worldConfirmed && draft.approved ? { ...draft, planApproved: true, step: 'production' } : null
}

function loadDraft(sample?: WizardDraft): { draft: WizardDraft; storageAvailable: boolean } {
  const initial = sample ?? createInitialWizardDraft()
  if (typeof window === 'undefined') return { draft: initial, storageAvailable: false }
  let stored: string | null
  let legacy: string | null = null
  try {
    stored = window.localStorage.getItem(sample ? 'auto-drama:m2-result-sample:v1' : WIZARD_STORAGE_KEY)
    if (stored === null && !sample) legacy = window.localStorage.getItem(LEGACY_STORAGE_KEY)
  } catch {
    return { draft: initial, storageAvailable: false }
  }
  try {
    const draft = stored !== null ? readWizardDraft(JSON.parse(stored)) : legacy !== null ? migrateLegacyDraft(JSON.parse(legacy)) : null
    return { draft: draft ?? initial, storageAvailable: true }
  } catch {
    return { draft: initial, storageAvailable: true }
  }
}

let fallbackId = 0
function newId(prefix: string): string {
  return `${prefix}-${typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : `${Date.now()}-${++fallbackId}`}`
}

/** Local UI preview only: save input and revision instructions in this browser. */
export function useWizardState(sample?: WizardDraft) {
  const [loaded] = useState(() => loadDraft(sample))
  const [draft, setDraft] = useState(loaded.draft)
  const [storageAvailable, setStorageAvailable] = useState(loaded.storageAvailable)
  const currentDraft = useRef(draft)
  const commit = useCallback((next: WizardDraft) => {
    if (next === currentDraft.current) return
    currentDraft.current = next
    setDraft(next)
  }, [])

  useEffect(() => {
    try {
      window.localStorage.setItem(sample ? 'auto-drama:m2-result-sample:v1' : WIZARD_STORAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, draft }))
      setStorageAvailable(true)
    } catch {
      setStorageAvailable(false)
    }
  }, [draft, sample])

  const updateWorld = useCallback((patch: Partial<WorldBrief>) => commit(updateWorldDraft(currentDraft.current, patch)), [commit])
  const updateRelationships = useCallback((relationshipInputs: RelationshipInput[]) => {
    const current = currentDraft.current
    if (JSON.stringify(current.relationshipInputs ?? []) === JSON.stringify(relationshipInputs)) return
    commit({ ...current, relationshipInputs, worldConfirmed: false, approved: false, planApproved: false, step: accessibleStep(current.step, false, false), revision: current.revision + 1 })
  }, [commit])
  const confirmWorld = useCallback(() => commit(confirmWorldDraft(currentDraft.current)), [commit])
  const goTo = useCallback((step: WizardStep) => commit(goToWizardStep(currentDraft.current, step)), [commit])
  const addCharacter = useCallback(() => {
    const id = newId('character')
    commit(addCharacterDraft(currentDraft.current, id))
    return id
  }, [commit])
  const updateCharacter = useCallback((id: string, patch: CharacterPatch) => commit(updateCharacterDraft(currentDraft.current, id, patch)), [commit])
  const removeCharacter = useCallback((id: string) => commit(removeCharacterDraft(currentDraft.current, id)), [commit])
  const toggleLock = useCallback((id: string, scope: CharacterScope) => commit(toggleCharacterLock(currentDraft.current, id, scope)), [commit])
  const requestRevision = useCallback((scope: RevisionScope, instruction: string, characterId?: string): boolean => {
    const next = requestDraftRevision(currentDraft.current, scope, instruction, newId('request'), new Date().toISOString(), characterId)
    if (!next) return false
    commit(next)
    return true
  }, [commit])
  const approve = useCallback((): boolean => {
    const next = approveDraft(currentDraft.current)
    if (!next) return false
    commit(next)
    return true
  }, [commit])

  const approvePlan = useCallback((): boolean => {
    const next = approvePlanDraft(currentDraft.current)
    if (!next) return false
    commit(next)
    return true
  }, [commit])

  return { draft, storageAvailable, updateWorld, updateRelationships, confirmWorld, goTo, addCharacter, updateCharacter, removeCharacter, toggleLock, requestRevision, approve, approvePlan }
}
