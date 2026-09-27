import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Icon } from './Icons'

export function Dialog({ title, children, onClose, wide = false, className = '' }: { title: string; children: ReactNode; onClose: () => void; wide?: boolean; className?: string }) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const dialog = ref.current!
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null
    dialog.showModal()
    return () => { dialog.close(); if (opener?.isConnected) opener.focus({ preventScroll: true }) }
  }, [])
  return <dialog className={`dialog ${wide ? 'dialog-wide' : ''} ${className}`} ref={ref} aria-labelledby="dialog-title" onCancel={event => { event.preventDefault(); onClose() }} onClick={event => { if (event.target === event.currentTarget) { const box = event.currentTarget.getBoundingClientRect(); if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) onClose() } }}>
    <div className="dialog-heading"><h2 id="dialog-title">{title}</h2><button className="icon-button" onClick={onClose} aria-label="閉じる"><Icon name="close"/></button></div>{children}
  </dialog>
}

export function LockButton({ name, locked, onClick, disabled = false }: { name: string; locked: boolean; onClick: () => void; disabled?: boolean }) {
  return <button className={`lock-button ${locked ? 'is-locked' : ''}`} disabled={disabled} onClick={onClick} aria-pressed={locked} aria-label={`${name}の固定${locked ? 'を解除' : ''}`} title={locked ? '再生成から保護されています' : 'この項目を再生成から保護'}><Icon name={locked ? 'lock' : 'unlock'} size={13}/><span>{locked ? '固定中' : '固定する'}</span></button>
}

export function VoicePlayer() {
  const audio = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [error, setError] = useState(false)
  const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
  async function toggle() {
    if (!audio.current) return
    if (playing) audio.current.pause()
    else { try { await audio.current.play(); setError(false) } catch { setError(true) } }
  }
  return <div className="voice-player">
    <audio ref={audio} src="/demo/nagi-voice.wav" preload="metadata" onLoadedMetadata={() => setDuration(audio.current?.duration ?? 0)} onTimeUpdate={() => setTime(audio.current?.currentTime ?? 0)} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} onError={() => setError(true)}/>
    <div className="wave-player"><button className="play-button" onClick={() => void toggle()} aria-label={playing ? '参考音声を一時停止' : '参考音声を試聴'}><Icon name={playing ? 'pause' : 'play'} size={15}/></button><div className={`waveform ${playing ? 'playing' : ''}`} aria-hidden="true">{Array.from({ length: 35 }, (_, i) => <span key={i} style={{ height: `${7 + ((i * 17 + 11) % 26)}px`, opacity: duration && i / 35 < time / duration ? 1 : 0.45 }}/>)}</div><span className="audio-time">{clock(playing ? time : duration)}</span></div>
    <input className="audio-seek" type="range" min="0" max={duration || 1} step="0.1" value={time} aria-label="参考音声の再生位置" onChange={event => { if (audio.current) audio.current.currentTime = Number(event.target.value); setTime(Number(event.target.value)) }}/>
    {error && <p className="form-error" role="alert">音声を読み込めませんでした。もう一度お試しください。</p>}
  </div>
}


