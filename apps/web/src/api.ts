export type Project = {
  id: string; title: string; instructions: string; chapter_count: number
  settings_version: number; schema_version: number; created_at: string
}
export type Job = {
  id: string; project_id: string; kind: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  priority: number; attempt_count: number; max_attempts: number; error: string | null
  created_at: string; updated_at: string; result_artifact_id: string | null
  progress?: JobProgress | null
}
export type JobProgressStep = {
  id: string
  stage: 'supporting_characters' | 'relationships' | 'story_core' | 'plot' | 'chapter_plan' | 'scene_plan'
    | 'script' | 'speech_extraction' | 'staging' | 'validation' | 'memory' | 'revision'
    | 'cast_plan' | 'plot_plan' | 'chapter_scene_plan'
  status: 'running' | 'completed' | 'failed'
  chapter_number?: number; scene_number?: number; completed?: number; total?: number
}
export type JobProgress = {
  schema_version: 1; sequence: number; phase: 'planning' | 'chapter'; current_step: string | null
  steps: JobProgressStep[]; attempt: number; updated_at: string; active: boolean
  chapter_plan?: { chapter_number: number; scene_count: number; supporting_character_count: number }
}
export type Artifact = {
  id: string; project_id: string; logical_id: string; version: number
  kind: 'script' | 'background' | 'character' | 'audio' | 'tyrano_export'
  filename: string; media_type: string; sha256: string; size_bytes: number
  schema_version: number; created_at: string; download_url: string
}
export type Worker = { id: string; name: string; capabilities: string[]; last_seen_at: string }
export type ProjectDetail = { project: Project; jobs: Job[]; artifacts: Artifact[] }

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message) }
}

export async function request<T>(path: string, signal: AbortSignal, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(10000)]),
    ...(body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    const fallback = [502, 503, 504].includes(response.status)
      ? '制御サーバーに接続できません。起動状態を確認してください。'
      : `リクエストに失敗しました（${response.status}）`
    throw new ApiError(typeof error?.detail === 'string' ? error.detail : fallback, response.status)
  }
  return response.json() as Promise<T>
}

export function errorMessage(error: unknown): string {
  if (error instanceof Error && error.name === 'TimeoutError') return '応答がありません。制御サーバーの起動状態を確認してください。'
  if (error instanceof Error && error.name !== 'TypeError') return error.message
  return '制御サーバーに接続できません。起動状態を確認してください。'
}
