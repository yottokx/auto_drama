import { useEffect, useRef, useState } from 'react'
import { sceneMusicTransition, type SceneMusicContinuity } from './adjustmentState'
import { chapterMusicRuntime, type MusicController, type MusicScript, type TransitionController } from './musicRuntime'

export function musicSeamScript(previous: SceneMusicContinuity | undefined, current: SceneMusicContinuity) {
  const urls = new Map<string, string>()
  const script: MusicScript = { assets: [], utterances: [{ id: 'before' }, { id: 'after' }], music_cues: [] }
  for (const [id, row, action] of [
    ['before', previous, previous?.candidate ? 'play' : 'stop'],
    ['after', current, current.setting.action],
  ] as const) {
    const candidate = action === 'play' ? row?.candidate : undefined
    if (candidate?.music_url) {
      script.assets.push({ id, kind: 'music', filename: `${id}.mp3` })
      urls.set(`./data/bgm/${id}.mp3`, candidate.music_url)
    }
    script.music_cues.push({ id, utterance_id: id, action,
      ...(candidate ? { asset_id: id, loop_start_seconds: candidate.loop_start_seconds, loop_end_seconds: candidate.loop_end_seconds } : {}),
      volume: id === 'before' ? row?.track_volume ?? row?.setting.volume ?? .35 : row?.setting.volume ?? .35 })
  }
  return { script, urls, previousPosition: Math.max(0, (previous?.candidate?.loop_end_seconds ?? previous?.candidate?.duration_seconds ?? 4) - 4) }
}

export function AdjustmentMusicSeamPreview({ previous, current, active, onStart, onStop }: {
  previous?: SceneMusicContinuity; current: SceneMusicContinuity; active: boolean
  onStart: () => void; onStop: () => void
}) {
  const [phase, setPhase] = useState<'idle' | 'loading' | 'before' | 'transition' | 'after'>('idle')
  const [error, setError] = useState('')
  const [screen, setScreen] = useState<'before' | 'after'>('before')
  const overlay = useRef<HTMLDivElement | null>(null)
  const animations = useRef(new Set<Animation>())
  const owner = useRef<{ controller: AbortController; context: AudioContext; music?: MusicController; transitions?: TransitionController; timer?: ReturnType<typeof setTimeout> } | null>(null)
  const callbacks = useRef({ onStart, onStop })
  callbacks.current = { onStart, onStop }
  const transition = sceneMusicTransition(current.setting)
  const signature = JSON.stringify({ before: previous && [previous.candidate, previous.track_volume ?? previous.setting.volume], after: current.candidate, action: current.setting.action, volume: current.setting.volume, transition })
  const running = phase !== 'idle'

  function resetOverlay() {
    for (const animation of animations.current) animation.cancel()
    animations.current.clear()
    if (overlay.current) { overlay.current.style.opacity = '0'; overlay.current.style.visibility = 'hidden' }
  }
  function stop(notify = true) {
    const value = owner.current; owner.current = null
    if (value) {
      value.controller.abort(); clearTimeout(value.timer)
      value.transitions?.dispose(); value.music?.dispose()
      void value.context.close().catch(() => {})
    }
    resetOverlay(); setPhase('idle')
    if (notify && value) callbacks.current.onStop()
  }
  useEffect(() => { if (!active) stop(false) }, [active])
  useEffect(() => { stop(false); if (active) callbacks.current.onStop(); setError(''); setScreen('before'); return () => stop(false) }, [signature])

  async function animate(from: number, to: number, duration: number, signal: AbortSignal) {
    const element = overlay.current
    if (signal.aborted) throw new Error('試聴を中止しました。')
    if (!element) return
    element.style.visibility = 'visible'; element.style.opacity = String(from)
    if (!duration || !element.animate) { element.style.opacity = String(to); return }
    const animation = element.animate([{ opacity: from }, { opacity: to }], { duration, easing: 'linear', fill: 'forwards' })
    animations.current.add(animation)
    const cancel = () => animation.cancel()
    signal.addEventListener('abort', cancel, { once: true })
    try { await animation.finished; if (signal.aborted) throw new Error('試聴を中止しました。'); element.style.opacity = String(to) }
    finally { signal.removeEventListener('abort', cancel); animations.current.delete(animation) }
  }

  async function start() {
    if (current.error || previous?.error) return
    stop(false); callbacks.current.onStart(); setPhase('loading'); setScreen('before'); setError('')
    let value: NonNullable<typeof owner.current> | undefined
    try {
      const Context = globalThis.AudioContext ?? (globalThis as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
      if (!Context) throw new Error('このブラウザーでは音源の試聴を利用できません。')
      value = { controller: new AbortController(), context: new Context() }; owner.current = value
      const { controller, context } = value
      // Unlock on the click before awaiting network access.
      await context.resume()
      if (owner.current !== value || controller.signal.aborted) return
      const { script, urls, previousPosition } = musicSeamScript(previous, current)
      const runtime = chapterMusicRuntime()
      const report = (message: string) => { if (owner.current === value && !controller.signal.aborted) setError(message) }
      const music = runtime.AutoDramaMusic.create(script, { context, registerOutput: false, report,
        loadAudio: async (relative, { signal }) => {
          const url = urls.get(relative)
          if (!url) throw new Error('試聴する音源を確認できませんでした。')
          const response = await fetch(url, { signal: AbortSignal.any([signal, controller.signal]) })
          if (!response.ok) throw new Error('試聴する音源を取得できませんでした。')
          return context.decodeAudioData(await response.arrayBuffer())
        },
      }); value.music = music
      const transitions = runtime.AutoDramaTransitions.create({ scenes: [{ id: 'after', music_cue_id: 'after', ...transition }] }, { music, report, adapter: {
        prepare: async () => true, same: () => false,
        cover: async (visual, duration, signal) => {
          const element = overlay.current
          if (element) { element.style.backgroundImage = visual === 'dissolve' && previous?.scene.background_url ? `url(${JSON.stringify(previous.scene.background_url)})` : 'none'; element.style.backgroundColor = visual === 'fade' ? '#000' : 'transparent' }
          if (visual === 'fade') await animate(0, 1, duration, signal)
          else if (visual === 'dissolve') await animate(1, 1, 0, signal)
        },
        apply: () => { setScreen('after'); setPhase('after') },
        reveal: async (duration, signal) => { if (transition.visual === 'fade' || transition.visual === 'dissolve') await animate(1, 0, duration, signal) },
        reset: resetOverlay, finish: () => { for (const animation of animations.current) animation.finish() },
      } }); value.transitions = transitions
      if (!await music.prepareCue('before', { signal: controller.signal }) || !await music.prepareCue('after', { signal: controller.signal })) throw new Error('場面のBGMを準備できませんでした。')
      if (owner.current !== value || controller.signal.aborted) return
      const before = music.getCue('before')
      if (before?.action === 'play' && !await music.cue(before, { position: previousPosition })) throw new Error('前の場面のBGMを再生できませんでした。')
      if (owner.current !== value || controller.signal.aborted) return
      setPhase('before')
      value.timer = setTimeout(async () => {
        if (owner.current !== value || controller.signal.aborted) return
        setPhase('transition')
        await transitions.run('after')
        if (owner.current !== value || controller.signal.aborted) return
        setPhase('after'); value!.timer = setTimeout(() => { if (owner.current === value) stop() }, 6000)
      }, 4000)
    } catch (reason) {
      if (value && (owner.current !== value || value.controller.signal.aborted)) return
      stop(); setError(reason instanceof Error ? reason.message : '場面のつなぎ目を試聴できませんでした。')
      if (!value) callbacks.current.onStop()
    }
  }

  const displayedScene = screen === 'after' ? current.scene : previous?.scene
  return <section className="adjustment-music-seam" aria-label="場面のつなぎ目の試聴">
    <div className="adjustment-music-playback-actions"><h3>場面のつなぎ目</h3><button type="button" className="button button-light" disabled={running || Boolean(current.error || previous?.error)} onClick={() => void start()}>場面のつなぎ目を試聴</button>{running && <button type="button" className="button button-light" onClick={() => stop()}>つなぎ目の試聴を停止</button>}</div>
    <p>{previous?.scene.title ?? '章の開始'} → {current.scene.title}。前の曲をB付近から4秒再生し、切り替え後を6秒試聴します。継続は同じ再生位置、曲の変更は冒頭から再生します。</p>
    {running && <div className="adjustment-music-seam-stage" style={{ backgroundImage: displayedScene?.background_url ? `url(${JSON.stringify(displayedScene.background_url)})` : undefined }}><div ref={overlay} className="adjustment-music-seam-overlay"/><strong>{displayedScene?.title ?? '章の開始'}</strong><span role="status">{phase === 'loading' ? '音源を準備中…' : phase === 'before' ? '前の場面 · 4秒' : phase === 'transition' ? '場面を切り替え中…' : '次の場面 · 6秒'}</span></div>}
    {error && <p className="m2-job-error" role="alert">{error}</p>}
  </section>
}
