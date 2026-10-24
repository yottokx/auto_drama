import { useEffect, useRef, useState } from 'react'
import { errorMessage, request } from './api'
import { Icon } from './Icons'
import {
  directionChips, EMOTION_NAMES, estimatedPages, remoteLabel, sceneMusicLabel, sceneRows, TIMING_NAMES, transitionLabel,
  type DirectionChip as Chip, type ScriptChapter, type ScriptLine,
} from './scriptTimelineState'

/** The published script of one chapter, line by line, with the stage each line shows. */
export function ScriptTimeline({ projectId, chapters }: { projectId: string; chapters: { production_id: string; chapter_number: number }[] }) {
  const [productionId, setProductionId] = useState(chapters[0]?.production_id ?? '')
  const [chapter, setChapter] = useState<ScriptChapter | null>(null)
  const [error, setError] = useState('')
  const [selectedId, setSelectedId] = useState('')
  const [directionId, setDirectionId] = useState<string | null>(null)
  const [panel, setPanel] = useState<'text' | 'directions'>('text')
  const [query, setQuery] = useState('')
  const rows = useRef(new Map<string, HTMLElement>())

  useEffect(() => {
    if (!productionId) return
    const controller = new AbortController()
    setChapter(null); setError('')
    request<ScriptChapter>(`/api/m3/projects/${encodeURIComponent(projectId)}/adjustments/script/${encodeURIComponent(productionId)}`, controller.signal)
      .then(next => { if (!controller.signal.aborted) { setChapter(next); setSelectedId(next.lines[0]?.id ?? ''); setDirectionId(null); setPanel('text') } })
      .catch(reason => { if (!controller.signal.aborted) setError(errorMessage(reason)) })
    return () => controller.abort()
  }, [projectId, productionId])

  const index = chapter ? Math.max(0, chapter.lines.findIndex(line => line.id === selectedId)) : 0
  const line = chapter?.lines[index]
  const chips = line ? directionChips(line.directions) : []
  function select(id: string, options: { direction?: string; scroll?: boolean } = {}) {
    setSelectedId(id); setDirectionId(options.direction ?? null); setPanel(options.direction ? 'directions' : 'text')
    if (options.scroll) rows.current.get(id)?.scrollIntoView({ block: 'nearest' })
  }
  const groups = chapter ? sceneRows(chapter, query) : []
  const visible = groups.reduce((total, group) => total + group.lines.length, 0)

  return <section id="adjustment-panel-script" role="tabpanel" aria-labelledby="adjustment-tab-script" tabIndex={0} className="script-timeline">
    <div className="adjustment-preview-tools">
      <label>章<select aria-label="台本の章" value={productionId} onChange={event => setProductionId(event.target.value)}>{chapters.map(row => <option key={row.production_id} value={row.production_id}>第{row.chapter_number}章</option>)}</select></label>
      <label>検索<input type="search" aria-label="台本の検索" value={query} onChange={event => setQuery(event.target.value)}/></label>
    </div>
    {error && <p className="m2-job-error" role="alert">{error}</p>}
    {!chapter && !error && <p role="status">台本を読み込み中…</p>}
    {chapter && line && <div className="script-workspace">
      <div className="script-preview-column">
        <ScriptStagePreview chapter={chapter} line={line} number={index + 1}/>
        <div className="script-preview-tools">
          <button type="button" className="button button-light" disabled={index === 0} onClick={() => select(chapter.lines[index - 1].id, { scroll: true })}>前の行</button>
          <button type="button" className="button button-light" disabled={index === chapter.lines.length - 1} onClick={() => select(chapter.lines[index + 1].id, { scroll: true })}>次の行</button>
          <span>{index + 1} / {chapter.lines.length} 行</span>
          <span className="script-music-now">{line.music_title ? `♪ ${line.music_title}` : 'BGMなし'}</span>
          {chips.some(row => row.timing === 'after') && <span className="script-after-now"><span>後：</span>{chips.filter(row => row.timing === 'after').map(row => <DirectionChip key={row.id} chip={row} selected={directionId !== null && row.ids.includes(directionId)} onOpen={() => select(line.id, { direction: row.id })}/>)}</span>}
        </div>
        <LineDetails line={line} number={index + 1} panel={panel} directionId={directionId} onPanel={setPanel}/>
      </div>
      <div className="script-lines" aria-label="台本の行">
        {groups.map(({ scene, lines }) => <section key={scene.scene_id} aria-label={scene.title}>
          <header className="script-scene">
            {scene.background_url ? <img src={scene.background_url} alt="" loading="lazy"/> : <span className="script-scene-blank"/>}
            <div><h3>{scene.title}</h3><p><span>{sceneMusicLabel(scene.music)}</span>{transitionLabel(scene.transition) && <span>{transitionLabel(scene.transition)}</span>}</p></div>
          </header>
          {lines.map(({ line: row, number }) => <ScriptRow key={row.id} line={row} number={number} current={row.id === line.id} directionId={row.id === line.id ? directionId : null}
            rowRef={element => { if (element) rows.current.set(row.id, element); else rows.current.delete(row.id) }}
            onSelect={direction => select(row.id, { direction })}/>)}
        </section>)}
        {!visible && <p className="script-empty">該当なし</p>}
      </div>
    </div>}
  </section>
}

function ScriptRow({ line, number, current, directionId, rowRef, onSelect }: {
  line: ScriptLine; number: number; current: boolean; directionId: string | null
  rowRef: (element: HTMLElement | null) => void; onSelect: (direction?: string) => void
}) {
  const chips = directionChips(line.directions)
  const before = chips.filter(row => row.timing !== 'after')
  const after = chips.filter(row => row.timing === 'after')
  const selected = (chip: Chip) => directionId !== null && chip.ids.includes(directionId)
  const voice = remoteLabel(line.remote)
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
          {voice && <span className="script-badge script-badge-voice">{voice}</span>}
          {line.speaker_id && line.spoken_text !== line.display_text && <span className="script-badge">読みあり</span>}
        </span>
      </button>
      {after.length > 0 && <div className="script-chips">{after.map(row => <DirectionChip key={row.id} chip={row} timing selected={selected(row)} onOpen={() => onSelect(row.id)}/>)}</div>}
    </div>
  </div>
}

function DirectionChip({ chip, timing = false, selected, onOpen }: { chip: Chip; timing?: boolean; selected: boolean; onOpen: () => void }) {
  const kind = chip.editable ? 'is-editable' : 'is-locked'
  return <button type="button" className={`script-chip ${kind}${chip.shown ? '' : ' is-unseen'}${selected ? ' is-selected' : ''}`}
    title={chip.label === chip.full_label ? undefined : chip.full_label}
    aria-label={`${TIMING_NAMES[chip.timing]}の演出 ${chip.full_label}${chip.editable ? '' : '（変更不可）'}${chip.shown ? '' : '（画面に反映されない）'}`} onClick={onOpen}>
    {!chip.editable && <Icon name="lock" size={11}/>}
    {!chip.shown && <Icon name="unseen" size={12}/>}
    {timing && <span className="script-chip-timing">{TIMING_NAMES[chip.timing]}</span>}{chip.label}
  </button>
}

function LineDetails({ line, number, panel, directionId, onPanel }: {
  line: ScriptLine; number: number; panel: 'text' | 'directions'; directionId: string | null; onPanel: (panel: 'text' | 'directions') => void
}) {
  const voice = remoteLabel(line.remote)
  const chips = directionChips(line.directions)
  return <div className="script-details">
    <div className="script-details-heading"><span>{number}行目</span><strong>{line.speaker_id ? line.speaker_name : '地の文'}</strong>{voice && <span className="script-badge script-badge-voice">{voice}</span>}</div>
    <div className="script-details-tabs" role="tablist" aria-label="行の内容">
      <button type="button" role="tab" aria-selected={panel === 'text'} onClick={() => onPanel('text')}>本文・音声</button>
      <button type="button" role="tab" aria-selected={panel === 'directions'} onClick={() => onPanel('directions')}>演出（{chips.length}）</button>
    </div>
    {panel === 'text' && <dl className="script-fields">
      <dt>表示テキスト</dt><dd>{line.display_text}</dd>
      {line.speaker_id && <>
        {line.spoken_text !== line.display_text && <><dt>読み</dt><dd>{line.spoken_text}</dd></>}
        <dt>声の感情</dt><dd>{EMOTION_NAMES[line.voice_emotion] ?? line.voice_emotion}</dd>
        {line.delivery && <><dt>話し方</dt><dd>{line.delivery}</dd></>}
        {line.audio_url && <><dt>台詞の音声</dt><dd><audio key={line.id} controls preload="none" src={line.audio_url}/></dd></>}
      </>}
    </dl>}
    {panel === 'directions' && <div className="script-direction-list">
      {!chips.length && <p>なし</p>}
      {chips.map(row => <div key={row.id} className={`script-direction${directionId !== null && row.ids.includes(directionId) ? ' is-selected' : ''}`}>
        <span className="script-chip-timing">{TIMING_NAMES[row.timing]}</span>
        <strong className={row.shown ? '' : 'is-unseen'}>{!row.shown && <Icon name="unseen" size={13}/>}{row.full_label}</strong>
        {!row.editable && <p><Icon name="lock" size={13}/>変更できません</p>}
      </div>)}
    </div>}
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
