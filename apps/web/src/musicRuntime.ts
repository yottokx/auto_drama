// Bundle the same package-owned controllers used by exported chapters.
import '../../../packages/tyrano_export/music.js'
import '../../../packages/tyrano_export/transitions.js'

export type MusicCue = {
  id: string; utterance_id: string; action: 'play' | 'continue' | 'stop'
  asset_id?: string | null; loop_start_seconds?: number | null
  loop_end_seconds?: number | null; volume: number
}
export type MusicScript = { assets: { id: string; kind: 'music'; filename: string }[]; utterances: { id: string }[]; music_cues: MusicCue[] }
export type MusicController = {
  getCue(id: string): MusicCue | null
  prepareCue(id: string, options?: { signal?: AbortSignal; timeoutMs?: number }): Promise<boolean>
  cue(cue: MusicCue, options?: { position?: number }): Promise<boolean>
  snapshot(): { position_seconds: number; phase: string; cue_id: string } | null
  pause(): void; dispose(): void
}
export type PreviewTransition = { id: string; music_cue_id: string; visual: string; duration_ms: number; music_fade_out_ms: number; music_fade_in_ms: number }
export type TransitionAdapter = {
  prepare(scene: PreviewTransition, signal: AbortSignal): Promise<boolean>
  same(scene: PreviewTransition): boolean
  cover(visual: string, duration: number, signal: AbortSignal): Promise<void>
  apply(scene: PreviewTransition): void
  reveal(duration: number, signal: AbortSignal): Promise<void>
  reset(): void; finish(): void
}
export type TransitionController = { run(id: string): Promise<boolean>; dispose(): void }
type Runtime = {
  AutoDramaMusic: { create(script: MusicScript, options: { context: AudioContext; loadAudio: (url: string, options: { signal: AbortSignal }) => Promise<AudioBuffer>; report: (message: string) => void; registerOutput: false }): MusicController }
  AutoDramaTransitions: { create(data: { scenes: PreviewTransition[] }, options: { music: MusicController; adapter: TransitionAdapter; report: (message: string) => void }): TransitionController }
}
export const chapterMusicRuntime = () => (globalThis as unknown as Runtime)
