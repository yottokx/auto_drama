import { ApiError } from './api'

export type EventCgPolicy = { max_cgs: number; max_variants_per_cg: number; revision: number }
export type EventCgProfile = {
  backend: 'qwen_image21'; model_revision: string; dtype: 'bfloat16'; cpu_offload: boolean
  use_kv_cache: boolean; steps: number; width: number; height: number
}
export type EventCgSettings = {
  revision: number; profile: EventCgProfile; ready: boolean
  workers: { id: string; name: string; ready: boolean; reason?: string }[]
  reasons?: string[]
}
export type EventCgSummary = {
  max_cgs: number; max_variants_per_cg: number; planned: number; generated: number; omitted: number
  chapter_budget?: number | null; budget_completed?: boolean; plan_completed?: boolean
  chapter_budgets?: { chapter_number: number; limit: number }[]
  budget_omission_reason?: string
  planning_omitted?: number
  omissions?: { cg_id?: string | null; variant_id?: string | null; reason: string }[]
}
export type EventCgEditorStatus = { pending: boolean; canApprove: boolean; policy: EventCgPolicy | null }

export const EVENT_CG_MAX = 100
export const EVENT_CG_VARIANTS_MAX = 10

export function eventCgBudgetCompleted(value: EventCgSummary): boolean {
  return value.budget_completed ?? (value.chapter_budget !== null && value.chapter_budget !== undefined)
}

export function eventCgImagePlanText(value: EventCgSummary, chapter = false): string {
  if (value.budget_omission_reason) return '画像の計画：CG配分を作成できなかったため省略'
  if (value.plan_completed || chapter && value.chapter_budget === 0) return `画像の計画 ${value.planned}枚（確定・差分を含む）`
  if (value.planned > 0) return `画像の計画 現時点で${value.planned}枚（ほかの${chapter ? 'CG' : '章'}は未確定・差分を含む）`
  if (!eventCgBudgetCompleted(value)) return '画像の計画：作品全体のCG配分待ち（枚数未確定）'
  return chapter ? '画像の計画：この章の台本完成後に確定' : '画像の計画：各章の台本完成後に順次確定'
}

export function eventCgPolicyError(maxCgs: number, maxVariants: number): string | null {
  if (!Number.isInteger(maxCgs) || maxCgs < 0 || maxCgs > EVENT_CG_MAX) return `基本CGの上限は0〜${EVENT_CG_MAX}件の整数で指定してください。`
  if (!Number.isInteger(maxVariants) || maxVariants < 0 || maxVariants > EVENT_CG_VARIANTS_MAX) return `追加差分は0〜${EVENT_CG_VARIANTS_MAX}枚の整数で指定してください。`
  return null
}

export function eventCgMaximum(policy: Pick<EventCgPolicy, 'max_cgs' | 'max_variants_per_cg'>): number {
  return policy.max_cgs * (1 + policy.max_variants_per_cg)
}

export function eventCgReadinessReason(maxCgs: number, settings: EventCgSettings | null): string | null {
  if (maxCgs === 0) return null
  if (!settings) return 'Qwenの実行環境を確認しています。'
  if (settings.ready && settings.profile.backend === 'qwen_image21' && settings.workers.some(worker => worker.ready)) return null
  return settings.reasons?.join(' ') || 'Qwen-Image-2.1を実行できるワーカーが準備できていません。設定の「画像」で準備状態を確認してください。'
}

/** The shared request helper uses POST; these versioned settings endpoints use PUT. */
export async function putEventCg<T>(path: string, body: unknown, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method: 'PUT', signal: AbortSignal.any([signal, AbortSignal.timeout(10000)]),
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  if (!response.ok) {
    const value = await response.json().catch(() => null)
    throw new ApiError(typeof value?.detail === 'string' ? value.detail : `設定を保存できませんでした（${response.status}）。`, response.status)
  }
  return response.json() as Promise<T>
}
