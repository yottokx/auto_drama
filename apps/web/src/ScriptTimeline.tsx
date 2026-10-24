import { useEffect, useRef, useState } from 'react'
import { errorMessage, request } from './api'
import { Icon } from './Icons'
import {
  directionChips, directionValue, editDirection, editLine, EMOTION_NAMES, estimatedPages, EXPRESSION_OPTIONS, lineChanged, lineValue, lineVoice,
  pauseLimit, remoteLabel, revertLine, sceneMusicLabel, sceneRows, scriptPayload, TIMING_NAMES, transitionLabel,
  type DirectionChip as Chip, type ScriptChapter, type ScriptDirection, type ScriptEdits, type ScriptLine, type ScriptStage, type ScriptVoice,
} from './scriptTimelineState'

export type ScriptVoiceRequest = { production_id: string; utterance_id: string; spoken_text: string; voice_emotion: string; delivery: string }
type Chapters = { production_id: string; chapter_number: number }[]
/** The stage and the reach of each direction, staged again by the server for the draft's directions. */
type Preview = { key: string; stages: Record<string, ScriptStage>; shown: Record<string, boolean> }

/** A chapter's script line by line, with the stage each line shows, edited into the draft. */
export function ScriptTimeline({ projectId, chapters, edits, voices, pending, disabled, sending, problem = '', onChange, onRegenerate }: {
  projectId: string; chapters: Chapters; edits: ScriptEdits; voices: ScriptVoice[]
  /** Lines of the saved draft still waiting for a voice, in chapters this tab may not have read. */
  pending: { production_id: string; utterance_id: string }[]
  disabled: boolean; sending: boolean
  /** Why the draft cannot be saved as it stands, from the server's check of the edits. */
  problem?: string
  onChange: (edits: ScriptEdits) => void; onRegenerate: (request: ScriptVoiceRequest) => void
}) {
  const [productionId, setProductionId] = useState(chapters[0]?.production_id ?? '')
  const [loaded, setLoaded] = useState<Record<string, ScriptChapter>>({})
  const [error, setError] = useState('')
  const [selectedId, setSelectedId] = useState('')
  const [directionId, setDirectionId] = useState<string | null>(null)
  const [panel, setPanel] = useState<'text' | 'directions'>('text')
  const [query, setQuery] = useState('')
  const [onlyChanged, setOnlyChanged] = useState(false)
  const [preview, setPreview] = useState<Preview | null>(null)
  const rows = useRef(new Map<string, HTMLElement>())
  const endpoint = `/api/m3/projects/${encodeURIComponent(projectId)}/adjustments/script/${encodeURIComponent(productionId)}`
  const chapter = loaded[productionId]

  useEffect(() => {
    if (!productionId || loaded[productionId]) return
    const controller = new AbortController()
    setError('')
    request<ScriptChapter>(endpoint, controller.signal)
      .then(next => { if (!controller.signal.aborted) { setLoaded(previous => ({ ...previous, [productionId]: next })); setSelectedId(next.lines[0]?.id ?? ''); setDirectionId(null); setPanel('text') } })
      .catch(reason => { if (!controller.signal.aborted) setError(errorMessage(reason)) })
    return () => controller.abort()
  }, [endpoint, productionId, loaded])

  // Changed directions are staged again by the server, through the same pass as the player.
  const chapterEdits = scriptPayload({ lines: edits.lines.filter(row => row.production_id === productionId),
    directions: edits.directions.filter(row => row.production_id === productionId) })
  const staged = Boolean(chapter?.lines.some(line => line.directions.some(row => directionValue(row, productionId, edits).edited)))
  const previewKey = staged ? JSON.stringify(chapterEdits.directions) : ''
  useEffect(() => {
    if (!previewKey) { setPreview(null); return }
    const controller = new AbortController()
    const timer = setTimeout(() => {
      request<ScriptChapter>(`${endpoint}/preview`, controller.signal, { lines: [], directions: JSON.parse(previewKey) })
        .then(next => { if (!controller.signal.aborted) setPreview({ key: previewKey,
          stages: Object.fromEntries(next.lines.map(row => [row.id, row.stage])),
          shown: Object.fromEntries(next.lines.flatMap(row => row.directions.map(direction => [direction.id, direction.shown]))) }) })
        .catch(reason => { if (!controller.signal.aborted) setError(errorMessage(reason)) })
    }, 250)
    return () => { clearTimeout(timer); controller.abort() }
  }, [endpoint, previewKey])

  /** The line as the draft has it: its words, its directions, and the stage they make. */
  function drafted(line: ScriptLine): ScriptLine {
    const value = lineValue(line, productionId, edits)
    const current = preview?.key === previewKey ? preview : null
    return { ...line, display_text: value.display_text, spoken_text: value.spoken_text, voice_emotion: value.voice_emotion, delivery: value.delivery,
      stage: current?.stages[line.id] ?? line.stage,
      directions: line.directions.map(row => {
        const direction = directionValue(row, productionId, edits)
        return { ...direction, shown: !direction.disabled && (current?.shown[row.id] ?? direction.shown) }
      }) }
  }

  const index = chapter ? Math.max(0, chapter.lines.findIndex(line => line.id === selectedId)) : 0
  const published = chapter?.lines[index]
  const line = published && drafted(published)
  function select(id: string, options: { direction?: string; scroll?: boolean } = {}) {
    setSelectedId(id); setDirectionId(options.direction ?? null); setPanel(options.direction ? 'directions' : 'text')
    if (options.scroll) rows.current.get(id)?.scrollIntoView({ block: 'nearest' })
  }
  const groups = chapter ? sceneRows(chapter, query, row => !onlyChanged || lineChanged(row, productionId, edits)) : []
  const visible = groups.reduce((total, group) => total + group.lines.length, 0)
  const changed = Object.values(loaded).reduce((total, item) => total + item.lines.filter(row => lineChanged(row, item.production_id, edits)).length, 0)
  const waiting = Object.values(loaded).reduce((total, item) => total + item.lines.filter(row => lineVoice(row, item.production_id, edits, voices).pending).length, 0)
    + pending.filter(row => !loaded[row.production_id]).length
  const change = (next: ScriptEdits) => { if (!disabled) onChange(next) }

  return <section id="adjustment-panel-script" role="tabpanel" aria-labelledby="adjustment-tab-script" tabIndex={0} className="script-timeline">
    <div className="adjustment-preview-tools">
      <label>章<select aria-label="台本の章" value={productionId} onChange={event => { setProductionId(event.target.value); const next = loaded[event.target.value]; if (next) select(next.lines[0]?.id ?? '') }}>{chapters.map(row => <option key={row.production_id} value={row.production_id}>第{row.chapter_number}章</option>)}</select></label>
      <label>検索<input type="search" aria-label="台本の検索" value={query} onChange={event => setQuery(event.target.value)}/></label>
      <label className="script-only-changed"><input type="checkbox" checked={onlyChanged} onChange={event => setOnlyChanged(event.target.checked)}/>変更のみ</label>
      <span className="script-counts">変更 <b>{changed}</b> 行{waiting > 0 && <span className="script-waiting">再生成待ち {waiting}</span>}</span>
    </div>
    {(error || problem) && <p className="m2-job-error" role="alert">{error || problem}</p>}
    {!chapter && !error && <p role="status">台本を読み込み中…</p>}
    {chapter && published && line && <div className="script-workspace">
      <div className="script-preview-column">
        <ScriptStagePreview chapter={chapter} line={line} number={index + 1}/>
        <div className="script-preview-tools">
          <button type="button" className="button button-light" disabled={index === 0} onClick={() => select(chapter.lines[index - 1].id, { scroll: true })}>前の行</button>
          <button type="button" className="button button-light" disabled={index === chapter.lines.length - 1} onClick={() => select(chapter.lines[index + 1].id, { scroll: true })}>次の行</button>
          <span>{index + 1} / {chapter.lines.length} 行</span>
          <span className="script-music-now">{line.music_title ? `♪ ${line.music_title}` : 'BGMなし'}</span>
          {line.directions.some(row => row.timing === 'after') && <span className="script-after-now"><span>後：</span>{directionChips(line.directions).filter(row => row.timing === 'after').map(row => <DirectionChip key={row.id} chip={row} selected={directionId !== null && row.ids.includes(directionId)} onOpen={() => select(line.id, { direction: row.id })}/>)}</span>}
        </div>
        <LineEditor key={line.id} published={published} line={line} productionId={productionId} number={index + 1} edits={edits} voices={voices}
          panel={panel} directionId={directionId} disabled={disabled} sending={sending} onPanel={setPanel} onChange={change} onRegenerate={onRegenerate}/>
      </div>
      <div className="script-lines" aria-label="台本の行">
        {groups.map(({ scene, lines }) => <section key={scene.scene_id} aria-label={scene.title}>
          <header className="script-scene">
            {scene.background_url ? <img src={scene.background_url} alt="" loading="lazy"/> : <span className="script-scene-blank"/>}
            <div><h3>{scene.title}</h3><p><span>{sceneMusicLabel(scene.music)}</span>{transitionLabel(scene.transition) && <span>{transitionLabel(scene.transition)}</span>}</p></div>
          </header>
          {lines.map(({ line: row, number }) => <ScriptRow key={row.id} line={drafted(row)} number={number} current={row.id === line.id}
            changed={lineChanged(row, productionId, edits)} voice={lineVoice(row, productionId, edits, voices)} directionId={row.id === line.id ? directionId : null}
            rowRef={element => { if (element) rows.current.set(row.id, element); else rows.current.delete(row.id) }}
            onSelect={direction => select(row.id, { direction })}/>)}
        </section>)}
        {!visible && <p className="script-empty">該当なし</p>}
      </div>
    </div>}
  </section>
}

function ScriptRow({ line, number, current, changed, voice, directionId, rowRef, onSelect }: {
  line: ScriptLine; number: number; current: boolean; changed: boolean; voice: ReturnType<typeof lineVoice>; directionId: string | null
  rowRef: (element: HTMLElement | null) => void; onSelect: (direction?: string) => void
}) {
  const chips = directionChips(line.directions)
  const before = chips.filter(row => row.timing !== 'after')
  const after = chips.filter(row => row.timing === 'after')
  const selected = (chip: Chip) => directionId !== null && chip.ids.includes(directionId)
  const heard = remoteLabel(line.remote)
  return <div ref={rowRef} className={`script-row${current ? ' is-current' : ''}`}>
    <span className={`script-band script-band-music${line.music_title ? ' is-on' : ''}`} aria-hidden="true"/>
    <span className={`script-band script-band-cg${line.event_cg ? ' is-on' : ''}`} aria-hidden="true"/>
    <div className="script-row-body">
      {(line.music || line.event_cg?.start) && <div className="script-marks">
        {line.music && <span className="script-mark-music">{line.music.action === 'play' ? `♪ ${line.music.title}` : 'BGMなし'}</span>}
        {line.event_cg?.start && <span className="script-mark-cg"><img src={line.event_cg.image_url} alt="" loading="lazy"/>イベントCG</span>}
      </div>}
      {before.length > 0 && <div className="script-chips">{before.map(row => <DirectionChip key={row.id} chip={row} timing selected={selected(row)} onOpen={() => onSelect(row.id)}/>)}</div>}
      <button type="button" className="script-line" aria-current={current || undefined} onClick={() => onSelect()}>
        <span className="script-line-number">{number}</span>
        <span className={line.speaker_id ? 'script-speaker' : 'script-speaker is-narration'} title={line.speaker_name || undefined}>{line.speaker_id ? line.speaker_name : '地の文'}</span>
        <span className="script-text">{line.display_text}
          {heard && <span className="script-badge script-badge-voice">{heard}</span>}
          {changed && <span className="script-badge script-badge-changed">変更あり</span>}
          {voice.pending && <span className="script-badge script-badge-stale">{voice.state === 'candidate' ? '候補の選択待ち' : '音声が古い'}</span>}
          {line.speaker_id && line.spoken_text !== line.display_text && <span className="script-badge">読みあり</span>}
        </span>
      </button>
      {after.length > 0 && <div className="script-chips">{after.map(row => <DirectionChip key={row.id} chip={row} timing selected={selected(row)} onOpen={() => onSelect(row.id)}/>)}</div>}
    </div>
  </div>
}

function DirectionChip({ chip, timing = false, selected, onOpen }: { chip: Chip; timing?: boolean; selected: boolean; onOpen: () => void }) {
  const kind = chip.editable ? 'is-editable' : 'is-locked'
  return <button type="button" className={`script-chip ${kind}${chip.shown ? '' : ' is-unseen'}${chip.disabled ? ' is-disabled' : ''}${chip.edited ? ' is-edited' : ''}${selected ? ' is-selected' : ''}`}
    title={chip.label === chip.full_label ? undefined : chip.full_label}
    aria-label={`${TIMING_NAMES[chip.timing]}の演出 ${chip.full_label}${chip.editable ? '' : '（変更不可）'}${chip.disabled ? '（無効）' : chip.shown ? '' : '（画面に反映されない）'}`} onClick={onOpen}>
    {!chip.editable && <Icon name="lock" size={11}/>}
    {!chip.shown && !chip.disabled && <Icon name="unseen" size={12}/>}
    {timing && <span className="script-chip-timing">{TIMING_NAMES[chip.timing]}</span>}{chip.label}
  </button>
}

function LineEditor({ published, line, productionId, number, edits, voices, panel, directionId, disabled, sending, onPanel, onChange, onRegenerate }: {
  published: ScriptLine; line: ScriptLine; productionId: string; number: number; edits: ScriptEdits; voices: ScriptVoice[]
  panel: 'text' | 'directions'; directionId: string | null; disabled: boolean; sending: boolean
  onPanel: (panel: 'text' | 'directions') => void; onChange: (edits: ScriptEdits) => void; onRegenerate: (request: ScriptVoiceRequest) => void
}) {
  const value = lineValue(published, productionId, edits)
  const [separate, setSeparate] = useState(value.spoken_text !== value.display_text)
  const heard = remoteLabel(line.remote)
  const chips = directionChips(line.directions)
  const pages = estimatedPages(value.display_text, Boolean(line.stage.remote_frame))
  const voice = lineVoice(published, productionId, edits, voices)
  const edit = (patch: Parameters<typeof editLine>[3]) => onChange(editLine(edits, productionId, published, patch))
  const regenerate = () => onRegenerate({ production_id: productionId, utterance_id: published.id, spoken_text: value.spoken_text,
    voice_emotion: value.voice_emotion, delivery: value.delivery })
  const editable = (chip: Chip) => chip.editable && chip.direction ? published.directions.find(row => row.id === chip.direction!.id) : undefined
  return <div className="script-details">
    <div className="script-details-heading"><span>{number}行目</span><strong>{line.speaker_id ? line.speaker_name : '地の文'}</strong>{heard && <span className="script-badge script-badge-voice">{heard}</span>}
      <button type="button" className="button button-light script-revert" disabled={disabled || !lineChanged(published, productionId, edits)} onClick={() => { onChange(revertLine(edits, productionId, published)); setSeparate(published.spoken_text !== published.display_text) }}><Icon name="back" size={13}/>元に戻す</button>
    </div>
    <div className="script-details-tabs" role="tablist" aria-label="行の内容">
      <button type="button" role="tab" aria-selected={panel === 'text'} onClick={() => onPanel('text')}>本文・音声</button>
      <button type="button" role="tab" aria-selected={panel === 'directions'} onClick={() => onPanel('directions')}>演出（{chips.length}）</button>
    </div>
    {panel === 'text' && <fieldset className="script-fields" disabled={disabled}>
      <label>表示テキスト<textarea rows={3} value={value.display_text} onChange={event => edit(separate ? { display_text: event.target.value } : { display_text: event.target.value, spoken_text: event.target.value })}/></label>
      {pages > 1 && <p className="script-pages">{pages} ページに分かれます</p>}
      {line.speaker_id && <>
        <label className="script-check"><input type="checkbox" checked={!separate} onChange={event => { setSeparate(!event.target.checked); if (event.target.checked) edit({ spoken_text: value.display_text }) }}/>読みは表示と同じ</label>
        {separate && <label>読み<textarea rows={2} value={value.spoken_text} onChange={event => edit({ spoken_text: event.target.value })}/></label>}
        <div className="script-field-row">
          <label>声の感情<select value={value.voice_emotion} onChange={event => edit({ voice_emotion: event.target.value })}>{Object.entries(EMOTION_NAMES).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>
          <label>話し方<input type="text" maxLength={200} value={value.delivery} onChange={event => edit({ delivery: event.target.value })}/></label>
        </div>
        <div className="script-voice">
          <span className="script-voice-title">台詞の音声</span>
          {(voice.state === 'stale' || voice.state === 'working') && <p className="script-voice-stale"><Icon name="refresh" size={14}/>音声が古い</p>}
          <div className="script-voice-grid">
            <span>旧音声</span>{published.audio_url ? <audio controls preload="none" src={published.audio_url}/> : <span/>}
            {voice.candidate?.url && voice.state !== 'none' && voice.state !== 'stale' && <><span>{voice.state === 'adopted' ? '使用中' : '新しい候補'}</span><audio controls preload="none" src={voice.candidate.url}/></>}
          </div>
          {voice.candidate?.status === 'failed' && <p className="m2-job-error">{voice.candidate.error || '作成できませんでした'}</p>}
          <div className="script-voice-actions">
            {voice.state === 'candidate' && <button type="button" className="button button-primary" onClick={() => edit({ voice_job_id: voice.candidate!.job_id, audio: null })}>この候補を使う</button>}
            {voice.state === 'adopted' && <><span className="script-voice-note">変更済み</span><button type="button" className="button button-light" onClick={() => edit({ voice_job_id: null })}>候補を表示</button></>}
            {voice.state === 'kept' && <span className="script-voice-note">旧音声を使用</span>}
            {voice.state === 'working' ? <span className="script-voice-note" role="status">作成中…</span>
              : voice.state !== 'adopted' && <button type="button" className={`button ${voice.state === 'stale' ? 'button-primary' : 'button-light'}`} disabled={sending} onClick={regenerate}>{voice.state === 'candidate' ? 'もう一度生成' : '再生成'}</button>}
            {voice.state === 'stale' && <button type="button" className="button button-light" onClick={() => edit({ audio: 'old' })}>旧音声を使う</button>}
          </div>
        </div>
      </>}
    </fieldset>}
    {panel === 'directions' && <div className="script-direction-list">
      {!chips.length && <p>なし</p>}
      {chips.map(chip => {
        const original = editable(chip)
        const current = chip.direction
        return <div key={chip.id} className={`script-direction${directionId !== null && chip.ids.includes(directionId) ? ' is-selected' : ''}`}>
          <span className="script-chip-timing">{TIMING_NAMES[chip.timing]}</span>
          <strong className={chip.disabled ? 'is-disabled' : chip.shown ? '' : 'is-unseen'}>{!chip.shown && !chip.disabled && <Icon name="unseen" size={13}/>}{chip.full_label}</strong>
          {chip.edited && <span className="script-edited">変更あり</span>}
          {!chip.editable && <p><Icon name="lock" size={13}/>変更できません</p>}
          {original && current && <DirectionControls original={original} current={current} limit={original.kind === 'pause' ? pauseLimit(published, original, productionId, edits) : 1500}
            disabled={disabled} onChange={patch => onChange(editDirection(edits, productionId, original, patch))}/>}
        </div>
      })}
    </div>}
  </div>
}

function DirectionControls({ original, current, limit, disabled, onChange }: {
  original: ScriptDirection; current: ScriptDirection; limit: number; disabled: boolean
  onChange: (patch: { disabled?: boolean; expression?: string; duration_ms?: number }) => void
}) {
  const off = <label className="script-check"><input type="checkbox" checked={current.disabled} disabled={disabled} onChange={event => onChange({ disabled: event.target.checked })}/>無効</label>
  if (original.kind === 'expression') return <div className="script-direction-controls">
    <label>表情<select value={current.expression} disabled={disabled || current.disabled} onChange={event => onChange({ expression: event.target.value })}>{EXPRESSION_OPTIONS.map(row => <option key={row.id} value={row.id}>{row.name}</option>)}</select></label>{off}
  </div>
  const milliseconds = current.duration_ms ?? 100
  return <div className="script-direction-controls">
    <label className="script-duration">長さ<input type="range" min={100} max={limit} step={50} value={Math.min(milliseconds, limit)} disabled={disabled || current.disabled} onChange={event => onChange({ duration_ms: Number(event.target.value) })}/><output>{(milliseconds / 1000).toFixed(2)}秒</output></label>{off}
  </div>
}

/** The line's moment on stage: background or event CG, portraits, the voice frame and the message window. */
export function ScriptStagePreview({ chapter, line, number }: { chapter: ScriptChapter; line: ScriptLine; number: number }) {
  const { width, height } = chapter.stage
  const { stage } = line
  const frame = stage.remote_frame
  const pages = estimatedPages(line.display_text, Boolean(frame))
  const box = (left: number, top: number, w: number, h: number) => ({ left: `${left / width * 100}%`, top: `${top / height * 100}%`, width: `${w / width * 100}%`, height: `${h / height * 100}%` })
  return <div className="script-stage" role="img" aria-label={`${number}行目の表示プレビュー：${line.speaker_id ? line.speaker_name : '地の文'}「${line.display_text}」`}>
    {stage.event_cg_url ? <img className="script-stage-fill" src={stage.event_cg_url} alt=""/> : <>
      {stage.background_url && <img className="script-stage-fill" src={stage.background_url} alt=""/>}
      {stage.characters.map(row => <img key={row.id} className={`script-stage-portrait${stage.lit === row.id ? '' : ' is-dim'}`} src={row.image_url} alt="" style={box(row.left, row.top, row.width, row.height)}/>)}
    </>}
    <div className="script-window">
      {frame && <div className="script-voice-frame"><img src={frame.image_url} alt="" style={{ width: `${frame.width / frame.crop[2] * 100}%`, left: `${-frame.crop[0] / frame.crop[2] * 100}%`, top: `${-frame.crop[1] / frame.crop[3] * 100}%` }}/></div>}
      <div className={`script-window-column${frame ? ' is-framed' : ''}`}>
        <div className="script-window-name"><span>{line.speaker_name}</span><i/>{pages > 1 && <small>1 / {pages}</small>}</div>
        <div className="script-window-text">{line.display_text}</div>
      </div>
    </div>
  </div>
}
