import { EXPRESSIONS } from './expressionState'

/** A published chapter's script as the script and staging tab reads it, line by line. */
export type ScriptDirection = {
  id: string; kind: 'background' | 'enter' | 'exit' | 'position' | 'outfit' | 'expression' | 'blackout' | 'pause'
  timing: 'before' | 'start' | 'after'
  /** The tab may change this kind; the others follow from the cast and the scene plan. */
  editable: boolean
  /** False where the direction changes nothing on screen. */
  shown: boolean
  /** Left out by an earlier edition; listed so that it can be turned on again. */
  disabled: boolean
  /** Set on the draft's view of a direction: it differs from the published one. */
  edited?: boolean
  character_id?: string; character_name?: string; place?: string; position?: 'left' | 'center' | 'right'
  outfit_name?: string; expression?: string; duration_ms?: number
}
export type ScriptRemote = { source: 'remote' | 'recollection' | 'offstage'; channel?: string; expression?: string }
export type ScriptStage = {
  background_url: string | null
  /** In stage pixels, the order they stand in. */
  characters: { id: string; image_url: string; left: number; top: number; width: number; height: number }[]
  /** The one person left bright; everyone else on stage is dimmed. */
  lit: string | null
  event_cg_url: string | null
  /** The bust beside the text for a voice from elsewhere: crop is left, top, width, height in the picture's pixels. */
  remote_frame: { image_url: string; crop: [number, number, number, number]; width: number; height: number } | null
}
export type ScriptLine = {
  id: string; scene_id: string; speaker_id: string | null; speaker_name: string
  display_text: string; spoken_text: string; voice_emotion: string; delivery: string; audio_url: string | null
  remote: ScriptRemote | null
  /** Set where music starts or stops at this line. */
  music: { action: 'play'; title: string } | { action: 'stop' } | null
  /** The track playing while this line shows. */
  music_title: string | null
  event_cg: { start: boolean; image_url: string } | null
  transition: SceneTransition | null
  directions: ScriptDirection[]
  stage: ScriptStage
}
export type SceneTransition = { visual: 'none' | 'dissolve' | 'fade' | 'cut'; duration_ms: number }
export type ScriptScene = {
  scene_id: string; title: string; first_utterance_id: string; background_url: string | null
  music: 'play' | 'continue' | 'stop'; transition: SceneTransition | null
}
export type ScriptChapter = {
  production_id: string; chapter_number: number; stage: { width: number; height: number }
  scenes: ScriptScene[]; lines: ScriptLine[]
}

export const TIMING_NAMES: Record<ScriptDirection['timing'], string> = { before: '前', start: '行頭', after: '後' }
const POSITION_NAMES = { left: '左', center: '中央', right: '右' }
const EXPRESSION_NAMES: Record<string, string> = Object.fromEntries([['neutral', '通常'], ...EXPRESSIONS.map(item => [item.id, item.name])])
export const EMOTION_NAMES: Record<string, string> = {
  neutral: '平常', happy: '喜び', sad: '悲しみ', angry: '怒り', surprised: '驚き', afraid: '恐れ', calm: '落ち着き',
}
const CHANNEL_NAMES: Record<string, string> = {
  phone: '電話', voicemail: '留守電', radio: '無線', speaker: 'スピーカー', telepathy: 'テレパシー', message: 'メッセージ',
}
const TRANSITION_NAMES: Record<SceneTransition['visual'], string> = { none: '', dissolve: 'ディゾルブ', fade: '暗転', cut: '即時切り替え' }

const seconds = (milliseconds = 0) => `${(milliseconds / 1000).toFixed(1)}秒`

/** A long name cut to fit a chip: its first part where it has parts, else its first six characters. */
export function shortName(name: string): string {
  const characters = Array.from(name)
  if (characters.length <= 6) return name
  const cut = characters.findIndex((character, index) => index >= 2 && index <= 6 && /[・＝=\s　]/.test(character))
  return (cut > 0 ? characters.slice(0, cut) : characters.slice(0, 6)).join('') + '…'
}

export function directionLabel(direction: ScriptDirection, short = false): string {
  const name = short ? shortName(direction.character_name ?? '') : direction.character_name ?? ''
  switch (direction.kind) {
    case 'background': return `背景 ${direction.place ?? ''}`.trim()
    case 'enter': return `登場 ${name}（${POSITION_NAMES[direction.position ?? 'center']}）`
    case 'position': return `移動 ${name} → ${POSITION_NAMES[direction.position ?? 'center']}`
    case 'exit': return `退場 ${name}`
    case 'outfit': return `衣装 ${name}：${direction.outfit_name ?? ''}`
    case 'expression': return `表情 ${name}：${EXPRESSION_NAMES[direction.expression ?? 'neutral'] ?? direction.expression}`
    case 'pause': return `間 ${seconds(direction.duration_ms)}`
    case 'blackout': return `暗転 ${seconds(direction.duration_ms)}`
  }
}

/** One chip: a direction, or every entrance or every exit at one moment of a line. ``label`` shortens long names; ``full_label`` keeps them. */
export type DirectionChip = {
  id: string; ids: string[]; timing: ScriptDirection['timing']; editable: boolean; shown: boolean; label: string; full_label: string
  disabled: boolean; edited: boolean
  /** The direction itself, for a chip of one direction. */
  direction?: ScriptDirection
}

export function directionChips(directions: ScriptDirection[]): DirectionChip[] {
  const chips: DirectionChip[] = []
  const groups = new Map<string, { chip: DirectionChip; rows: ScriptDirection[] }>()
  for (const direction of directions) {
    if (direction.kind === 'enter' || direction.kind === 'exit') {
      const key = `${direction.kind}:${direction.timing}`
      const group = groups.get(key)
      if (group) {
        group.rows.push(direction); group.chip.ids.push(direction.id)
        Object.assign(group.chip, { label: groupLabel(direction.kind, group.rows, true), full_label: groupLabel(direction.kind, group.rows, false) })
        continue
      }
      const chip = { id: direction.id, ids: [direction.id], timing: direction.timing, editable: false, shown: true,
        label: directionLabel(direction, true), full_label: directionLabel(direction), disabled: false, edited: false }
      groups.set(key, { chip, rows: [direction] }); chips.push(chip)
      continue
    }
    chips.push({ id: direction.id, ids: [direction.id], timing: direction.timing, editable: direction.editable, shown: direction.shown,
      label: directionLabel(direction, true), full_label: directionLabel(direction), disabled: direction.disabled,
      edited: Boolean(direction.edited), direction })
  }
  return chips
}

function groupLabel(kind: 'enter' | 'exit', rows: ScriptDirection[], short: boolean) {
  const name = (row: ScriptDirection) => short ? shortName(row.character_name ?? '') : row.character_name ?? ''
  return kind === 'enter'
    ? `登場 ${rows.map(row => `${name(row)}（${POSITION_NAMES[row.position ?? 'center']}）`).join('・')}`
    : `退場 ${rows.map(name).join('・')}`
}

export function expressionName(expression: string) { return EXPRESSION_NAMES[expression] ?? expression }
export const EXPRESSION_OPTIONS = Object.entries(EXPRESSION_NAMES).map(([id, name]) => ({ id, name }))

/** What a voice from elsewhere comes through; empty for a voice in the room. */
export function remoteLabel(remote: ScriptRemote | null): string {
  if (!remote) return ''
  if (remote.source === 'recollection') return '回想'
  if (remote.source === 'offstage') return '画面外'
  return CHANNEL_NAMES[remote.channel ?? ''] ?? ''
}

export function transitionLabel(transition: SceneTransition | null): string {
  if (!transition || transition.visual === 'none') return ''
  const name = `場面転換 ${TRANSITION_NAMES[transition.visual]}`
  return transition.visual === 'cut' || !transition.duration_ms ? name : `${name} ${seconds(transition.duration_ms)}`
}

export function sceneMusicLabel(action: ScriptScene['music']): string {
  return action === 'play' ? 'BGMを再生' : action === 'continue' ? 'BGMを継続' : 'BGMなし'
}

/** About how many pages the player splits a text into: three lines a page, a narrower column beside a voice frame. */
export function estimatedPages(text: string, framed: boolean): number {
  const perLine = framed ? 29 : 34
  const lines = text.split('\n').reduce((total, part) => total + Math.max(1, Math.ceil(part.length / perLine)), 0)
  return Math.max(1, Math.ceil(lines / 3))
}

export function lineMatches(line: ScriptLine, query: string): boolean {
  const words = query.trim()
  return !words || line.display_text.includes(words) || line.speaker_name.includes(words)
}

/** The lines of each scene, in playing order, keeping only those that match. */
export function sceneRows(chapter: ScriptChapter, query: string, keep: (line: ScriptLine) => boolean = () => true): { scene: ScriptScene; lines: { line: ScriptLine; number: number }[] }[] {
  const numbered = chapter.lines.map((line, index) => ({ line, number: index + 1 }))
  return chapter.scenes.map(scene => ({ scene, lines: numbered.filter(row => row.line.scene_id === scene.scene_id && lineMatches(row.line, query) && keep(row.line)) }))
}

/* ── Edits ──
 * The draft keeps what a line or direction should be. An entry stays once made: every apply
 * builds the script again from its narrative, so an edit from an earlier edition lives only
 * in its entry. Undoing a change sets the entry back to the published values instead. */

export type ScriptLineEdit = {
  production_id: string; utterance_id: string; display_text: string; spoken_text: string; voice_emotion: string; delivery: string
  /** The voice of changed words: keep the old recording, or use a candidate made for exactly these words. */
  audio?: 'old' | null; voice_job_id?: string | null
}
export type ScriptDirectionEdit = {
  production_id: string; direction_id: string; disabled: boolean; expression?: string | null; duration_ms?: number | null
  /** Kept by the server: the direction as it was, so a disabled one can come back. */
  original?: unknown
}
export type ScriptEdits = { lines: ScriptLineEdit[]; directions: ScriptDirectionEdit[] }
export type ScriptVoice = {
  job_id: string; production_id: string; utterance_id: string; spoken_text: string; voice_emotion: string; delivery: string
  status: string; error: string | null; url: string | null
}
export const noScriptEdits: ScriptEdits = { lines: [], directions: [] }
export const PAUSE_BUDGET_MS = 1500

export function scriptPayload(edits: ScriptEdits): ScriptEdits {
  return { lines: edits.lines, directions: edits.directions.map(({ original: _original, ...row }) => row) }
}

export type LineValue = Pick<ScriptLineEdit, 'display_text' | 'spoken_text' | 'voice_emotion' | 'delivery'> & { audio: 'old' | null; voice_job_id: string | null }

const lineEdit = (edits: ScriptEdits, productionId: string, id: string) => edits.lines.find(row => row.production_id === productionId && row.utterance_id === id)
const directionEdit = (edits: ScriptEdits, productionId: string, id: string) => edits.directions.find(row => row.production_id === productionId && row.direction_id === id)
const voiceOf = (value: { spoken_text: string; voice_emotion: string; delivery?: string | null }) => JSON.stringify([value.spoken_text, value.voice_emotion, value.delivery ?? ''])

export function lineValue(line: ScriptLine, productionId: string, edits: ScriptEdits): LineValue {
  const edit = lineEdit(edits, productionId, line.id)
  return { display_text: edit?.display_text ?? line.display_text, spoken_text: edit?.spoken_text ?? line.spoken_text,
    voice_emotion: edit?.voice_emotion ?? line.voice_emotion, delivery: edit?.delivery ?? line.delivery,
    audio: edit?.audio ?? null, voice_job_id: edit?.voice_job_id ?? null }
}

/** Whether the words a line is spoken with differ from its published recording. */
export function voiceChanged(line: ScriptLine, value: LineValue): boolean {
  return Boolean(line.speaker_id) && voiceOf(value) !== voiceOf(line)
}

export function editLine(edits: ScriptEdits, productionId: string, line: ScriptLine, patch: Partial<LineValue>): ScriptEdits {
  const previous = lineValue(line, productionId, edits)
  let next: LineValue = { ...previous, ...patch }
  // Narration has no voice: what it reads is what it shows.
  if (!line.speaker_id) next = { ...next, spoken_text: next.display_text }
  // A recording belongs to the words it was made for; new words need a voice chosen again.
  if (voiceOf(next) !== voiceOf(previous) && !('audio' in patch) && !('voice_job_id' in patch)) next = { ...next, audio: null, voice_job_id: null }
  const row: ScriptLineEdit = { production_id: productionId, utterance_id: line.id, ...next }
  const exists = Boolean(lineEdit(edits, productionId, line.id))
  return { ...edits, lines: exists ? edits.lines.map(item => item.production_id === productionId && item.utterance_id === line.id ? row : item) : [...edits.lines, row] }
}

export function directionValue(direction: ScriptDirection, productionId: string, edits: ScriptEdits): ScriptDirection {
  const edit = directionEdit(edits, productionId, direction.id)
  if (!edit) return direction
  const value = { ...direction, disabled: edit.disabled, expression: edit.expression ?? direction.expression, duration_ms: edit.duration_ms ?? direction.duration_ms }
  return { ...value, edited: value.disabled !== direction.disabled || value.expression !== direction.expression || value.duration_ms !== direction.duration_ms }
}

export function editDirection(edits: ScriptEdits, productionId: string, direction: ScriptDirection, patch: { disabled?: boolean; expression?: string; duration_ms?: number }): ScriptEdits {
  const previous = directionEdit(edits, productionId, direction.id)
  const current = directionValue(direction, productionId, edits)
  const merged = { disabled: current.disabled, expression: current.expression, duration_ms: current.duration_ms, ...patch }
  const row: ScriptDirectionEdit = { ...previous, production_id: productionId, direction_id: direction.id, disabled: merged.disabled,
    // The published value needs no entry of its own.
    expression: merged.expression === direction.expression ? null : merged.expression ?? null,
    duration_ms: merged.duration_ms === direction.duration_ms ? null : merged.duration_ms ?? null }
  return { ...edits, directions: previous ? edits.directions.map(item => item === previous ? row : item) : [...edits.directions, row] }
}

/** The line and its directions as published; entries stay, set back to the published values. */
export function revertLine(edits: ScriptEdits, productionId: string, line: ScriptLine): ScriptEdits {
  let next = lineEdit(edits, productionId, line.id)
    ? editLine(edits, productionId, line, { display_text: line.display_text, spoken_text: line.spoken_text, voice_emotion: line.voice_emotion, delivery: line.delivery, audio: null, voice_job_id: null })
    : edits
  for (const direction of line.directions) {
    if (directionEdit(next, productionId, direction.id)) next = editDirection(next, productionId, direction, { disabled: direction.disabled, expression: direction.expression, duration_ms: direction.duration_ms })
  }
  return next
}

export type LineVoice = {
  state: 'none' | 'stale' | 'working' | 'candidate' | 'adopted' | 'kept'
  /** The newest recording made for this line in the draft, or the one in use. */
  candidate?: ScriptVoice
  /** It still waits for a voice, which holds the apply back. */
  pending: boolean
}

export function lineVoice(line: ScriptLine, productionId: string, edits: ScriptEdits, voices: ScriptVoice[]): LineVoice {
  const value = lineValue(line, productionId, edits)
  const changed = voiceChanged(line, value)
  const mine = voices.filter(row => row.production_id === productionId && row.utterance_id === line.id)
  const candidate = (value.voice_job_id ? mine.find(row => row.job_id === value.voice_job_id) : undefined) ?? mine.at(-1)
  if (value.voice_job_id) return { state: 'adopted', candidate, pending: false }
  if (value.audio === 'old' && changed) return { state: 'kept', candidate, pending: false }
  if (candidate && (candidate.status === 'pending' || candidate.status === 'running')) return { state: 'working', candidate, pending: changed }
  if (candidate?.status === 'completed' && candidate.url && voiceOf(candidate) === voiceOf(value)) return { state: 'candidate', candidate, pending: changed }
  return { state: changed ? 'stale' : 'none', candidate, pending: changed }
}

export function lineChanged(line: ScriptLine, productionId: string, edits: ScriptEdits): boolean {
  const value = lineValue(line, productionId, edits)
  return value.display_text !== line.display_text || voiceChanged(line, value) || Boolean(value.voice_job_id)
    || line.directions.some(direction => directionValue(direction, productionId, edits).edited)
}

/** How long a pause may be, given the other pauses of its line: a line waits 1.5 seconds at most. */
export function pauseLimit(line: ScriptLine, direction: ScriptDirection, productionId: string, edits: ScriptEdits): number {
  const others = line.directions.filter(row => row.kind === 'pause' && row.id !== direction.id)
    .map(row => directionValue(row, productionId, edits)).filter(row => !row.disabled)
    .reduce((total, row) => total + (row.duration_ms ?? 0), 0)
  return Math.max(100, PAUSE_BUDGET_MS - others)
}
