export type Precision = 'fp32' | 'bf16' | 'int8' | 'int4'
export type TTSPurpose = 'voice_design' | 'voice_clone'
export type TTSChoice = { provider_id: string; model_id: string; precision: Precision }
export type TTSConfiguration = { schema_version: 1; voice_design: TTSChoice; voice_clone: TTSChoice }
export type TTSModel = { model_id: string; label: string; provider_id: string; precisions: Precision[]; license: string; source_url: string; variants?: { precision: Precision; manifest_id: string; download_bytes: number }[] }
export type TTSInventory = { model_id: string; precision: Precision; manifest_id: string; file_download_ready: boolean; generation_ready?: boolean; generation_purposes?: TTSPurpose[] }
export type DownloadWorker = { id: string; name: string; online: boolean; download_capable?: boolean; inventory: TTSInventory[] }
export type DownloadOperation = {
  id: string; worker_id: string; model_id: string; precision: Precision; manifest_id?: string; status: string
  done_bytes: number; total_bytes: number | null; phase?: string; current_file?: string | null; error?: string | null
  file_download_ready?: boolean; cancel_requested?: boolean; created_at: string; updated_at: string
}
export type DownloadSnapshot = { workers: DownloadWorker[]; operations: DownloadOperation[] }

export const precisionOptions: Precision[] = ['fp32', 'bf16', 'int8', 'int4']
export const activeDownload = (operation: DownloadOperation) => ['queued', 'downloading', 'verifying', 'converting', 'cancelling'].includes(operation.status)
export const resumableDownload = (operation: DownloadOperation) => ['cancelled', 'failed', 'interrupted'].includes(operation.status)
export const eligibleDownloadWorker = (worker: DownloadWorker) => worker.online && worker.download_capable !== false
export const validTTSChoice = (choice: TTSChoice, models: TTSModel[]) => models.some(model => model.provider_id === choice.provider_id && model.model_id === choice.model_id && model.precisions.includes(choice.precision))
export const sameTTSChoice = (left: TTSChoice, right: TTSChoice) => left.provider_id === right.provider_id && left.model_id === right.model_id && left.precision === right.precision
export function hasTTSFiles(worker: DownloadWorker, choice: TTSChoice, models: TTSModel[]): boolean {
  const manifestId = models.find(model => model.provider_id === choice.provider_id && model.model_id === choice.model_id)?.variants?.find(variant => variant.precision === choice.precision)?.manifest_id
  return Boolean(manifestId && worker.inventory.some(item => item.model_id === choice.model_id && item.precision === choice.precision && item.manifest_id === manifestId && item.file_download_ready))
}
export function readyTTSChoice(worker: DownloadWorker, choice: TTSChoice, models: TTSModel[], purpose: TTSPurpose): boolean {
  const manifestId = models.find(model => model.provider_id === choice.provider_id && model.model_id === choice.model_id)?.variants?.find(variant => variant.precision === choice.precision)?.manifest_id
  return Boolean(worker.online && manifestId && worker.inventory.some(item => item.model_id === choice.model_id && item.precision === choice.precision && item.manifest_id === manifestId && item.file_download_ready && item.generation_ready && item.generation_purposes?.includes(purpose)))
}
export function availableTTSChoices(snapshot: DownloadSnapshot | null, models: TTSModel[], purpose: TTSPurpose): TTSChoice[] {
  return models.flatMap(model => model.precisions.map(precision => ({ provider_id: model.provider_id, model_id: model.model_id, precision })))
    .filter(choice => snapshot?.workers.some(worker => readyTTSChoice(worker, choice, models, purpose)))
}
export function ttsSettingsBlockReason(settings: TTSConfiguration, snapshot: DownloadSnapshot | null, models: TTSModel[]): string | null {
  if (!snapshot || !models.length) return '取得済みモデルと実行環境を確認しています。'
  for (const purpose of ['voice_design', 'voice_clone'] as const) {
    const label = purpose === 'voice_design' ? 'ボイスデザイン' : 'ボイスクローン'
    const choice = settings[purpose]
    if (!validTTSChoice(choice, models)) return `${label}のモデルまたは精度が対応していません。`
    if (!snapshot.workers.some(worker => hasTTSFiles(worker, choice, models))) return `${label}のモデルファイルを先にダウンロードしてください。`
    if (!snapshot.workers.some(worker => readyTTSChoice(worker, choice, models, purpose))) return `${label}の取得済みモデルを実行できる環境が接続されていません。`
  }
  return null
}
export function chooseTTSModel(choice: TTSChoice, model: TTSModel): TTSChoice {
  return { provider_id: model.provider_id, model_id: model.model_id, precision: model.precisions.includes(choice.precision) ? choice.precision : model.precisions[0] }
}
export function downloadBlockReason(snapshot: DownloadSnapshot | null, workerId: string, choice: TTSChoice, models: TTSModel[] = []): string | null {
  const worker = snapshot?.workers.find(worker => worker.id === workerId)
  if (!worker || !eligibleDownloadWorker(worker)) return 'ダウンロードに対応した接続中の保存先を選択してください。'
  const manifestId = models.find(model => model.model_id === choice.model_id && model.provider_id === choice.provider_id)?.variants?.find(variant => variant.precision === choice.precision)?.manifest_id
  const sameFiles = (item: { model_id: string; precision: Precision; manifest_id?: string }) => manifestId ? item.manifest_id === manifestId : item.model_id === choice.model_id && item.precision === choice.precision
  if (worker.inventory.some(item => sameFiles(item) && item.file_download_ready)) return 'このモデルと精度のファイルは取得済みです。'
  if (snapshot?.operations.some(operation => operation.worker_id === workerId && sameFiles(operation) && activeDownload(operation))) return 'このモデルのファイルはダウンロード処理中です。'
  return null
}
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return '容量確認中'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let value = Math.max(0, bytes), unit = 0
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1 }
  return `${value.toLocaleString('ja-JP', { maximumFractionDigits: unit ? 2 : 0 })} ${units[unit]}`
}
