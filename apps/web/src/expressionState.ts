/** Expression pictures of a character's looks. None is required: a face without a picture shows the ordinary portrait. */

export type ExpressionId = 'smile' | 'laugh' | 'angry' | 'sad' | 'cry' | 'surprised' | 'confused' | 'anxious' | 'serious' | 'fear' | 'shy'
export type ExpressionPicture = {
  outfitId: string; expression: ExpressionId; imageArtifactId: string; baseImageArtifactId: string
  imageUrl: string
  /** Edited from a portrait that has since been replaced; kept, but no longer shown in the story. */
  stale: boolean
}
/** A face marked by hand: left, top, right and bottom as fractions of the portrait. */
export type FaceBox = [number, number, number, number]
export type ExpressionRequest = { outfitIds?: string[]; expressions: ExpressionId[]; redraw?: boolean; faceBox?: FaceBox }
/** An expression of a portrait kept by the portrait itself: a supporting character's, or an adjusted one. */
export type PortraitExpression = {
  expression: ExpressionId; base_artifact_id: string; status: 'ready' | 'pending' | 'running' | 'failed' | 'cancelled' | 'missing'
  url: string | null; artifact_id: string | null; job_id: string | null; error: string | null
}
/** How often a finished work asks each character for each expression. */
export type ExpressionUsage = { project_id: string; characters: Record<string, Partial<Record<ExpressionId, number>>> }
export type PortraitExpressions = { project_id: string; enabled?: boolean; portraits: Record<string, PortraitExpression[]> }

/** The fixed kinds, in the order they are offered. The ordinary face is the portrait itself. */
export const EXPRESSIONS: { id: ExpressionId; name: string }[] = [
  { id: 'smile', name: '微笑み' }, { id: 'angry', name: '怒り' }, { id: 'sad', name: '悲しみ' },
  { id: 'surprised', name: '驚き' }, { id: 'confused', name: '困惑' }, { id: 'anxious', name: '不安' }, { id: 'serious', name: '真剣' },
  { id: 'laugh', name: '笑顔' }, { id: 'cry', name: '泣き' }, { id: 'fear', name: '恐怖' }, { id: 'shy', name: '照れ' },
]
/** Each set contains the one before it, so a larger choice only adds pictures. */
export const EXPRESSION_SETS: { id: string; label: string; expressions: ExpressionId[] }[] = [
  { id: 'light', label: 'よく使う3種', expressions: EXPRESSIONS.slice(0, 3).map(item => item.id) },
  { id: 'standard', label: '標準の7種', expressions: EXPRESSIONS.slice(0, 7).map(item => item.id) },
  { id: 'all', label: '全11種', expressions: EXPRESSIONS.map(item => item.id) },
]

/** The pictures to show for one portrait, and what is still being made or failed for it. */
export function portraitExpressionView(rows: PortraitExpression[]): { pictures: ExpressionPicture[]; notes: Partial<Record<ExpressionId, string>>; working: boolean } {
  const notes: Partial<Record<ExpressionId, string>> = {}
  for (const row of rows) {
    if (row.status === 'pending' || row.status === 'running') notes[row.expression] = '作成中…'
    else if (row.status === 'failed' || row.status === 'cancelled') notes[row.expression] = row.error ? `作成できませんでした：${row.error}` : '作成できませんでした'
  }
  return {
    pictures: rows.filter(row => row.url && row.artifact_id).map(row => ({ outfitId: 'default', expression: row.expression, imageArtifactId: row.artifact_id!, baseImageArtifactId: row.base_artifact_id, imageUrl: row.url!, stale: false })),
    notes, working: rows.some(row => row.status === 'pending' || row.status === 'running'),
  }
}

/** The expressions a work asks a character for, most used first; ties keep the offered order. */
export function usedExpressions(counts: Partial<Record<ExpressionId, number>> | undefined): ExpressionId[] {
  return EXPRESSIONS.map(item => item.id).filter(id => (counts?.[id] ?? 0) > 0)
    .sort((left, right) => (counts![right] ?? 0) - (counts![left] ?? 0))
}

/** The face a drag marks on a portrait, whichever way it was dragged; nothing if it is too small to be one. */
export function faceBoxFrom(from: { x: number; y: number }, to: { x: number; y: number }): FaceBox | null {
  const clamp = (value: number) => Math.round(Math.min(1, Math.max(0, value)) * 1000) / 1000
  const box: FaceBox = [clamp(Math.min(from.x, to.x)), clamp(Math.min(from.y, to.y)), clamp(Math.max(from.x, to.x)), clamp(Math.max(from.y, to.y))]
  return box[2] - box[0] < 0.02 || box[3] - box[1] < 0.02 ? null : box
}

export function expressionName(id: string): string {
  return EXPRESSIONS.find(item => item.id === id)?.name ?? id
}

/** The current picture of an expression of a look, if one exists. */
export function expressionPicture(pictures: ExpressionPicture[], outfitId: string, expression: string): ExpressionPicture | undefined {
  return pictures.find(item => item.outfitId === outfitId && item.expression === expression)
}

/**
 * What a request would actually draw: the chosen expressions that a look does not have yet, or
 * has only from a replaced portrait. Looks without a current portrait cannot be drawn from.
 */
export function expressionsToDraw(pictures: ExpressionPicture[], outfitIds: string[], chosen: ExpressionId[]): { outfitId: string; expression: ExpressionId }[] {
  return outfitIds.flatMap(outfitId => chosen
    .filter(expression => { const picture = expressionPicture(pictures, outfitId, expression); return !picture || picture.stale })
    .map(expression => ({ outfitId, expression })))
}
