import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const source = await readFile(new URL('../src/ttsSettingsState.ts', import.meta.url), 'utf8')
const { code } = await transformWithOxc(source, 'ttsSettingsState.ts', { target: 'es2022' })
const { availableTTSChoices, chooseTTSModel, downloadBlockReason, eligibleDownloadWorker, formatBytes, hasTTSFiles, precisionOptions, readyTTSChoice, sameTTSChoice, ttsSettingsBlockReason, validTTSChoice } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)
const small = { provider_id: 'irodori', model_id: 'irodori-v4.1-small', precisions: precisionOptions }
const large = { provider_id: 'irodori', model_id: 'irodori-v4-large', precisions: precisionOptions }
const choice = { provider_id: 'irodori', model_id: large.model_id, precision: 'bf16' }
const worker = { id: 'worker-1', name: 'GPU', online: true, download_capable: true, inventory: [] }

test('each model offers the four requested precisions and can switch independently', () => {
  assert.deepEqual(precisionOptions, ['fp32', 'bf16', 'int8', 'int4'])
  for (const model of [small, large]) for (const precision of precisionOptions) {
    assert.equal(validTTSChoice({ ...choice, model_id: model.model_id, precision }, [small, large]), true)
  }
  const cloneChoice = { ...choice, precision: 'int4' }
  assert.deepEqual(chooseTTSModel(cloneChoice, small), { ...cloneChoice, model_id: small.model_id })
  assert.deepEqual(cloneChoice, { ...choice, precision: 'int4' })
  assert.equal(validTTSChoice({ ...choice, precision: 'fp16' }, [small, large]), false)
})

test('download requires a connected capable worker and prevents duplicates', () => {
  const snapshot = { workers: [worker], operations: [] }
  assert.equal(downloadBlockReason(snapshot, worker.id, choice), null)
  assert.equal(downloadBlockReason(snapshot, 'missing', choice), 'ダウンロードに対応した接続中の保存先を選択してください。')
  assert.equal(eligibleDownloadWorker({ ...worker, online: false }), false)
  assert.equal(eligibleDownloadWorker({ ...worker, download_capable: false }), false)
  assert.match(downloadBlockReason({ ...snapshot, workers: [{ ...worker, inventory: [{ ...choice, file_download_ready: true }] }] }, worker.id, choice), /取得済み/)
  assert.match(downloadBlockReason({ ...snapshot, operations: [{ ...choice, worker_id: worker.id, status: 'downloading' }] }, worker.id, choice), /処理中/)
  assert.equal(downloadBlockReason({ ...snapshot, operations: [{ ...choice, worker_id: worker.id, status: 'interrupted' }] }, worker.id, choice), null)
})

test('a provider change chooses an available precision and formats unknown progress safely', () => {
  assert.deepEqual(chooseTTSModel(choice, { provider_id: 'future-tts', model_id: 'other', precisions: ['int8'] }), { provider_id: 'future-tts', model_id: 'other', precision: 'int8' })
  assert.equal(formatBytes(null), '容量確認中')
  assert.equal(formatBytes(0), '0 B')
  assert.equal(formatBytes(1024 ** 3), '1 GiB')
})

test('shared fp32/bf16 files block duplicates while stale manifests do not block new versions', () => {
  const models = [{ ...large, variants: [{ precision: 'fp32', manifest_id: 'base-current' }, { precision: 'bf16', manifest_id: 'base-current' }] }]
  const snapshot = { workers: [worker], operations: [{ ...choice, precision: 'fp32', manifest_id: 'base-current', worker_id: worker.id, status: 'downloading' }] }
  assert.match(downloadBlockReason(snapshot, worker.id, choice, models), /処理中/)
  const withInventory = { ...snapshot, operations: [], workers: [{ ...worker, inventory: [{ ...choice, precision: 'fp32', manifest_id: 'base-current', file_download_ready: true }] }] }
  assert.match(downloadBlockReason(withInventory, worker.id, choice, models), /取得済み/)
  assert.equal(downloadBlockReason({ ...withInventory, workers: [{ ...worker, inventory: [{ ...choice, manifest_id: 'base-old', file_download_ready: true }] }] }, worker.id, choice, models), null)
})

const installedModels = [small, large].map(model => ({ ...model, variants: precisionOptions.map(precision => ({ precision, manifest_id: `${model.model_id}-${['fp32', 'bf16'].includes(precision) ? 'base' : precision}` })) }))
const inventoryEntry = (choice, purposes = ['voice_design', 'voice_clone']) => ({ ...choice, manifest_id: installedModels.find(model => model.model_id === choice.model_id).variants.find(variant => variant.precision === choice.precision).manifest_id, file_download_ready: true, generation_ready: true, generation_purposes: purposes })

test('settings only offer exact current, installed, runtime-ready selections while all downloads stay available', () => {
  const inventory = [small, large].flatMap(model => ['fp32', 'bf16'].map(precision => inventoryEntry({ provider_id: model.provider_id, model_id: model.model_id, precision })))
  const snapshot = { workers: [{ ...worker, inventory }], operations: [] }
  for (const purpose of ['voice_design', 'voice_clone']) {
    assert.equal(availableTTSChoices(snapshot, installedModels, purpose).length, 4)
    for (const model of [small, large]) for (const precision of precisionOptions) {
      const candidate = { provider_id: model.provider_id, model_id: model.model_id, precision }
      assert.equal(availableTTSChoices(snapshot, installedModels, purpose).some(item => sameTTSChoice(item, candidate)), ['fp32', 'bf16'].includes(precision))
      assert.equal(validTTSChoice(candidate, installedModels), true)
    }
  }
  assert.equal(availableTTSChoices({ ...snapshot, workers: [{ ...worker, inventory, online: false }] }, installedModels, 'voice_design').length, 0)
  assert.equal(availableTTSChoices({ ...snapshot, workers: [{ ...worker, inventory: inventory.map(item => ({ ...item, manifest_id: 'stale' })) }] }, installedModels, 'voice_design').length, 0)
})

test('inventory must explicitly report runtime readiness and the requested purpose', () => {
  const filesOnly = { ...inventoryEntry(choice), generation_ready: false }
  assert.equal(hasTTSFiles({ ...worker, inventory: [filesOnly] }, choice, installedModels), true)
  assert.equal(readyTTSChoice({ ...worker, inventory: [filesOnly] }, choice, installedModels, 'voice_design'), false)
  assert.equal(readyTTSChoice({ ...worker, inventory: [{ ...filesOnly, generation_ready: undefined }] }, choice, installedModels, 'voice_design'), false)
  const designWorker = { ...worker, inventory: [inventoryEntry(choice, ['voice_design'])] }
  assert.equal(readyTTSChoice(designWorker, choice, installedModels, 'voice_design'), true)
  assert.equal(readyTTSChoice(designWorker, choice, installedModels, 'voice_clone'), false)
  assert.equal(hasTTSFiles({ ...worker, inventory: [{ ...inventoryEntry(choice), precision: 'fp32' }] }, choice, installedModels), false)
})

test('independent purposes may use different connected workers, but stale or disconnected selections cannot be saved', () => {
  const clone = { provider_id: 'irodori', model_id: small.model_id, precision: 'fp32' }
  const settings = { schema_version: 1, voice_design: choice, voice_clone: clone }
  const snapshot = { workers: [{ ...worker, inventory: [inventoryEntry(choice, ['voice_design'])] }, { ...worker, id: 'worker-2', inventory: [inventoryEntry(clone, ['voice_clone'])] }], operations: [] }
  assert.equal(ttsSettingsBlockReason(settings, snapshot, installedModels), null)
  assert.match(ttsSettingsBlockReason(settings, { ...snapshot, workers: [snapshot.workers[0], { ...snapshot.workers[1], online: false }] }, installedModels), /ボイスクローン.*接続/)
  assert.match(ttsSettingsBlockReason({ ...settings, voice_clone: { ...clone, precision: 'int4' } }, snapshot, installedModels), /ボイスクローン.*ダウンロード/)
  assert.match(ttsSettingsBlockReason(settings, null, installedModels), /確認/)
  assert.match(ttsSettingsBlockReason({ ...settings, voice_clone: { ...clone, precision: 'fp16' } }, snapshot, installedModels), /対応/)
})
