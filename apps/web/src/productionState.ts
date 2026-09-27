import type { Job } from './api'

export type PublishedBuild = { id: string; chapter_number: number; status: 'published' }
export type ProductionChapter = {
  chapter_number: number; production_id: string | null; narrative_artifact_id: string | null
  status: 'waiting' | 'writing' | 'generating_assets' | 'published' | 'failed'
  jobs: Job[]; requirements?: unknown[]; build: PublishedBuild | null; error: string | null
  player_url: string | null; export_url: string | null
}
export type Production = {
  id: string; approval_id: string; chapter_number: number; chapter_count?: number
  narrative_artifact_id: string | null; history_frozen?: boolean
  status: 'pending' | 'running' | 'failed' | 'published'
  stage: 'narrative' | 'assets' | 'building' | 'published'
  control_state?: 'running' | 'stopping' | 'paused' | 'interrupted'
  jobs: Job[]; completed_jobs: number; total_jobs: number; error: string | null
  build: PublishedBuild | null; player_url: string | null; export_url: string | null
  chapters_export_url?: string | null
  chapters?: ProductionChapter[]
}
export type ProductionResponse = { project_id: string; production: Production | null }
export type CoordinatorHealth = { status: string; stage?: string; schema_version?: number }

export const serverRestartMessage = '制御サーバーを再起動してください。現在は旧バージョンのサーバーが動作しているため、続きの章の制作と停止・再開を利用できません。'

export function coordinatorNeedsRestart(health: CoordinatorHealth): boolean {
  return (health.schema_version ?? 0) < 4
}

export const jobNames: Record<string, string> = {
  m3_narrative: 'プロット・シーン・台本の生成', m3_background: '背景',
  m3_image: 'サブキャラの立ち絵', m3_voice: 'サブキャラの基準音声',
  m3_voice_clone: '台詞の音声', m3_dialogue: '台詞の音声',
}
export const chapterStatusNames: Record<ProductionChapter['status'], string> = {
  waiting: '前章の本文を待機', writing: '本文を制作中',
  generating_assets: '本文完成・素材を制作中', published: '鑑賞できます', failed: '再試行が必要',
}

export function productionChapters(production: Production | null, chapterCount: number): ProductionChapter[] {
  if (!production) return []
  // Saved one-chapter productions remain viewable after upgrading the server.
  const existing: ProductionChapter[] = production.chapters ?? [{
    chapter_number: production.chapter_number, production_id: production.id,
    narrative_artifact_id: production.narrative_artifact_id, jobs: production.jobs,
    status: production.build ? 'published' : production.status === 'failed' ? 'failed'
      : production.narrative_artifact_id ? 'generating_assets' : 'writing',
    build: production.build, player_url: production.player_url, export_url: production.export_url,
    error: production.error,
  }]
  const count = Math.max(production.chapter_count ?? chapterCount, ...existing.map(chapter => chapter.chapter_number), 1)
  return Array.from({ length: count }, (_, index) => existing.find(chapter => chapter.chapter_number === index + 1) ?? {
    chapter_number: index + 1, production_id: null, narrative_artifact_id: null, status: 'waiting',
    jobs: [], build: null, error: null, player_url: null, export_url: null,
  })
}

export function productionView(production: Production | null, chapterCount: number, needsServerRestart = false) {
  const chapters = productionChapters(production, chapterCount)
  const publishedCount = chapters.filter(chapter => chapter.build?.status === 'published').length
  const complete = chapters.length > 0 && publishedCount === chapters.length
  const historyFrozen = Boolean(production?.history_frozen)
  const controlState = production?.control_state ?? 'running'
  const paused = controlState === 'paused' || controlState === 'interrupted'
  const stopping = controlState === 'stopping'
  const canControl = Boolean(production && !complete && !historyFrozen && !needsServerRestart)
  const canStop = canControl && controlState === 'running'
  const canInterrupt = canControl && (canStop || stopping)
  const canResume = canControl && paused
  const canRetry = canControl && !stopping && !paused
  const title = needsServerRestart ? '制御サーバーの再起動が必要です。'
    : !production ? '承認した世界から、物語を。'
    : complete ? 'すべての章を鑑賞できます。'
      : historyFrozen ? '制作途中の状態を表示しています。'
        : stopping ? '実行中の工程を保存して、停止します。'
          : paused ? (controlState === 'interrupted' ? '制作を中断しています。' : '制作を一時停止しています。')
            : production.status === 'failed' ? '制作の再試行を待っています。'
              : publishedCount ? `完成した${publishedCount}章から鑑賞できます。` : '物語を、制作しています。'
  return { chapters, publishedCount, complete, historyFrozen, controlState, paused, stopping, canStop, canInterrupt, canResume, canRetry, title }
}
