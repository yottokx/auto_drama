import { useEffect, useRef, useState } from 'react'
import type { AdjustmentMusicCandidate } from './adjustmentState'

export function musicPreviewPosition(offset: number, elapsed: number, duration: number, start: number, end: number, loop: boolean): number {
  const position = Math.max(0, offset + elapsed)
  return loop && end > start && position >= end ? start + (position - end) % (end - start) : Math.min(position, duration)
}

function timestamp(seconds: number) {
  const value = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  return `${Math.floor(value / 60)}:${(value % 60).toFixed(1).padStart(4, '0')}`
}

/** Web Audio loops sample-accurately, including the one-time introduction. */
export function AdjustmentMusicPreview({ candidate, volume, autoPlayRequest = 0 }: { candidate: AdjustmentMusicCandidate; volume: number; autoPlayRequest?: number }) {
  const [playing, setPlaying] = useState(false)
  const [loading, setLoading] = useState(false)
  const [original, setOriginal] = useState(!candidate.music_url && Boolean(candidate.source_url))
  const [position, setPosition] = useState(0)
  const [error, setError] = useState('')
  const context = useRef<AudioContext | null>(null)
  const gain = useRef<GainNode | null>(null)
  const source = useRef<AudioBufferSourceNode | null>(null)
  const buffer = useRef<{ url: string; value: AudioBuffer } | null>(null)
  const pending = useRef<AbortController | null>(null)
  const operation = useRef(0)
  const offset = useRef(0)
  const startedAt = useRef(0)
  const active = useRef(false)
  const metadataDuration = original ? candidate.source_duration_seconds ?? candidate.duration_seconds ?? 0 : candidate.duration_seconds ?? candidate.loop_end_seconds ?? 0
  const start = candidate.loop_start_seconds ?? 0
  const canonicalEnd = candidate.loop_end_seconds ?? metadataDuration
  const url = original ? candidate.source_url ?? candidate.music_url : candidate.music_url
  const decoded = buffer.current?.url === url ? buffer.current.value : null
  const duration = decoded ? Math.min(metadataDuration || decoded.duration, decoded.duration) : metadataDuration
  const end = decoded ? Math.min(canonicalEnd, decoded.duration) : canonicalEnd
  const validLoop = Number.isFinite(start) && Number.isFinite(canonicalEnd) && start >= 0 && end > start
    && (!decoded || canonicalEnd <= decoded.duration + 2 / decoded.sampleRate)

  function currentPosition() {
    const node = source.current
    return active.current && context.current && node?.buffer
      ? musicPreviewPosition(offset.current, context.current.currentTime - startedAt.current,
        Math.min(metadataDuration || node.buffer.duration, node.buffer.duration), node.loopStart, node.loopEnd, node.loop)
      : offset.current
  }
  function stopNode() {
    active.current = false
    const node = source.current; source.current = null
    if (node) { node.onended = null; node.stop(); node.disconnect() }
  }
  function pause() {
    const next = currentPosition()
    operation.current += 1; pending.current?.abort(); pending.current = null
    stopNode(); offset.current = next; setPosition(next); setPlaying(false); setLoading(false)
  }

  async function play(from: number) {
    if (!url || (!original && !validLoop)) return
    const version = ++operation.current
    pending.current?.abort(); stopNode(); setPlaying(false); setLoading(true); setError('')
    const controller = new AbortController(); pending.current = controller
    try {
      if (!context.current) {
        context.current = new AudioContext()
        gain.current = context.current.createGain(); gain.current.connect(context.current.destination)
      }
      const audioContext = context.current
      await audioContext.resume()
      if (buffer.current?.url !== url) {
        const response = await fetch(url, { signal: controller.signal })
        if (!response.ok) throw new Error('音源を読み込めませんでした。')
        const decoded = await audioContext.decodeAudioData(await response.arrayBuffer())
        if (version !== operation.current || controller.signal.aborted) return
        buffer.current = { url, value: decoded }
      }
      if (version !== operation.current || controller.signal.aborted) return
      const decoded = buffer.current!.value
      const effectiveEnd = Math.min(canonicalEnd, decoded.duration)
      if (!original && (canonicalEnd > decoded.duration + 2 / decoded.sampleRate || effectiveEnd <= start)) throw new Error('ループ区間と音源の長さが一致しません。')
      const playbackDuration = Math.min(metadataDuration || decoded.duration, decoded.duration)
      const node = audioContext.createBufferSource()
      node.buffer = decoded; node.loop = !original && validLoop
      if (node.loop) { node.loopStart = start; node.loopEnd = effectiveEnd }
      node.connect(gain.current!)
      gain.current!.gain.value = Math.max(0, Math.min(1, volume))
      const next = node.loop ? musicPreviewPosition(from, 0, playbackDuration, start, effectiveEnd, true)
        : Math.max(0, Math.min(from, Math.max(0, decoded.duration - 1 / decoded.sampleRate)))
      source.current = node; offset.current = next; startedAt.current = audioContext.currentTime; active.current = true
      node.onended = () => {
        if (source.current !== node) return
        source.current = null; active.current = false; offset.current = playbackDuration
        node.disconnect(); setPosition(playbackDuration); setPlaying(false)
      }
      node.start(0, next); setPosition(next); setPlaying(true)
    } catch (reason) {
      if (version === operation.current && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : '音源を再生できませんでした。')
    } finally {
      if (version === operation.current) { pending.current = null; setLoading(false) }
    }
  }

  useEffect(() => {
    if (gain.current && context.current) gain.current.gain.setValueAtTime(Math.max(0, Math.min(1, volume)), context.current.currentTime)
  }, [volume])
  useEffect(() => {
    const timer = setInterval(() => { if (active.current) setPosition(currentPosition()) }, 100)
    return () => clearInterval(timer)
  }, [duration, start, end, original, validLoop])
  useEffect(() => {
    operation.current += 1; pending.current?.abort(); stopNode(); offset.current = 0
    setPosition(0); setPlaying(false); setLoading(false); setError('')
  }, [url])
  useEffect(() => { if (autoPlayRequest > 0) void play(0) }, [autoPlayRequest])
  useEffect(() => () => {
    operation.current += 1; pending.current?.abort(); stopNode(); buffer.current = null
    const previous = context.current; context.current = null; gain.current = null
    if (previous) void previous.close().catch(() => {})
  }, [])

  function seek(value: number) {
    const wasPlaying = active.current
    const next = musicPreviewPosition(Math.max(0, Math.min(value, duration)), 0, duration, start, end, !original && validLoop)
    pause(); offset.current = next; setPosition(next)
    if (wasPlaying) void play(next)
  }
  const startPercent = duration > 0 ? Math.min(100, start / duration * 100) : 0
  const endPercent = duration > 0 ? Math.min(100, end / duration * 100) : 100
  return <div className="adjustment-music-preview">
    <div className="adjustment-music-playback-actions">
      <button type="button" className="button button-light" disabled={!url || loading || (!original && !validLoop)} onClick={() => void play(0)}>{loading ? '音源を読み込み中…' : '冒頭から試聴'}</button>
      <button type="button" className="button button-light" disabled={!url || loading || (!playing && position === 0)} onClick={() => playing ? pause() : void play(position)}>{playing ? '一時停止' : '再開'}</button>
      {candidate.source_url && <label><input type="checkbox" checked={original} onChange={event => { pause(); setOriginal(event.target.checked) }}/>元の音源を最後まで聴く</label>}
    </div>
    <div className="adjustment-music-timeline">
      <div className="adjustment-music-loop-band" aria-hidden="true" style={{ left: `${startPercent}%`, width: `${Math.max(0, endPercent - startPercent)}%` }}/>
      <span className="adjustment-music-loop-mark" style={{ left: `${startPercent}%` }} aria-hidden="true">A</span>
      <span className="adjustment-music-loop-mark adjustment-music-loop-end" style={{ left: `${endPercent}%` }} aria-hidden="true">B</span>
      <input type="range" min={0} max={Math.max(.1, duration)} step={.1} value={Math.min(position, duration)} disabled={!url || loading || duration <= 0} aria-label="BGMの再生位置" aria-valuetext={`${timestamp(position)} / ${timestamp(duration)}`} onChange={event => seek(Number(event.target.value))}/>
    </div>
    <p className="adjustment-music-loop-description">{timestamp(position)} / {timestamp(duration)} · A {timestamp(start)} · B {timestamp(end)}<br/>{original ? '元音源を冒頭から末尾まで再生します。' : validLoop ? '冒頭0秒からBまで一度再生し、その後はA〜Bを繰り返します。' : '有効なループ区間がありません。元音源と品質情報を確認してください。'}</p>
    {error && <p className="m2-job-error" role="alert">{error}</p>}
  </div>
}
