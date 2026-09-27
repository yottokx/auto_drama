export type BodyBounds = { left: number; top: number; right: number; bottom: number }
export type BoundsEdge = keyof BodyBounds
export type PortraitCharacter = {
  character_id: string; name: string; image_artifact_id: string; image_url: string
  framing: 'auto' | 'upper_body' | 'full_body'; height_cm: number | null
  body_bounds: BodyBounds | null
}
export type PortraitResponse = {
  project_id: string; production_id: string; build_id: string; characters: PortraitCharacter[]
}
export type PortraitEdits = Record<string, BodyBounds | null>

const minimum = 0.05
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))
const rounded = (value: number) => Math.round(value * 10000) / 10000
export const fullBounds: BodyBounds = { left: 0, top: 0, right: 1, bottom: 1 }

export function drawnBounds(start: { x: number; y: number }, end: { x: number; y: number }): BodyBounds {
  let left = clamp(Math.min(start.x, end.x), 0, 1)
  let top = clamp(Math.min(start.y, end.y), 0, 1)
  const right = clamp(Math.max(start.x, end.x, left + minimum), minimum, 1)
  const bottom = clamp(Math.max(start.y, end.y, top + minimum), minimum, 1)
  left = Math.min(left, right - minimum)
  top = Math.min(top, bottom - minimum)
  return { left: rounded(left), top: rounded(top), right: rounded(right), bottom: rounded(bottom) }
}

export function movedBounds(bounds: BodyBounds, x: number, y: number): BodyBounds {
  const dx = clamp(x, -bounds.left, 1 - bounds.right)
  const dy = clamp(y, -bounds.top, 1 - bounds.bottom)
  return {
    left: rounded(bounds.left + dx), top: rounded(bounds.top + dy),
    right: rounded(bounds.right + dx), bottom: rounded(bounds.bottom + dy),
  }
}

export function changedEdge(bounds: BodyBounds, edge: BoundsEdge, value: number): BodyBounds {
  if (!Number.isFinite(value)) return bounds
  const limits: Record<BoundsEdge, [number, number]> = {
    left: [0, bounds.right - minimum], top: [0, bounds.bottom - minimum],
    right: [bounds.left + minimum, 1], bottom: [bounds.top + minimum, 1],
  }
  return { ...bounds, [edge]: rounded(clamp(value, ...limits[edge])) }
}

export function editedBounds(character: PortraitCharacter, edits: PortraitEdits): BodyBounds | null {
  return Object.hasOwn(edits, character.character_id) ? edits[character.character_id] : character.body_bounds
}

export function portraitsChanged(source: PortraitResponse, edits: PortraitEdits): boolean {
  return source.characters.some(character => {
    const before = character.body_bounds
    const after = editedBounds(character, edits)
    return before === null || after === null ? before !== after
      : (Object.keys(fullBounds) as BoundsEdge[]).some(edge => before[edge] !== after[edge])
  })
}

export function portraitPayload(source: PortraitResponse, edits: PortraitEdits) {
  return {
    expected_build_id: source.build_id,
    characters: source.characters.map(character => ({
      character_id: character.character_id, framing: character.framing,
      height_cm: character.height_cm, image_artifact_id: character.image_artifact_id,
      body_bounds: editedBounds(character, edits),
    })),
  }
}

export function portraitSourceMatches(source: PortraitResponse, projectId: string, productionId: string, buildId: string) {
  return source.project_id === projectId && source.production_id === productionId && source.build_id === buildId
}
