import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { transformWithOxc } from 'vite'

const source = await readFile(new URL('../src/llmSettingsState.ts', import.meta.url), 'utf8')
const { code } = await transformWithOxc(source, 'llmSettingsState.ts', { target: 'es2022' })
const { chooseModel, chooseProvider, effortChoices, effortLabel, modelChoices, modelLabel, modelProvider, providerChoices, providerLabel } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`)

const all = ['none', 'low', 'medium', 'high', 'xhigh', 'max']
const models = [
  { model: 'gemma-4-31B-it-UD-Q4_K_XL', reasoning_efforts: ['none'], worker_name: 'GPU' },
  { model: 'Qwen3.8-27B-UD-Q4_K_XL', reasoning_efforts: ['none', 'low', 'medium'], worker_name: 'GPU' },
  { model: 'openrouter/google/gemma-4-31b-it', reasoning_efforts: ['none'], worker_name: 'GPU' },
  { model: 'anthropic/claude-haiku-5-5', reasoning_efforts: all, sampling_efforts: [], worker_name: 'GPU' },
  { model: 'anthropic/claude-haiku-5-5', reasoning_efforts: all, sampling_efforts: [], worker_name: 'Laptop' },
]
const settings = { model: 'gemma-4-31B-it-UD-Q4_K_XL', temperature: 0.65, top_p: 0.95, reasoning_effort: 'none', ctx_size: 32768 }

test('models are grouped by where they run and shown by their own names', () => {
  assert.equal(modelProvider('gemma-4-31B-it-UD-Q4_K_XL'), 'local')
  assert.equal(modelProvider('openrouter/google/gemma-4-31b-it'), 'openrouter')
  assert.deepEqual(providerChoices(models), ['local', 'anthropic', 'openrouter'])
  assert.deepEqual(providerChoices(models).map(providerLabel), ['ローカル', 'anthropic', 'openrouter'])
  assert.deepEqual(modelChoices(models, 'anthropic'), ['anthropic/claude-haiku-5-5'])
  assert.deepEqual(modelChoices(models, 'local'), ['gemma-4-31B-it-UD-Q4_K_XL', 'Qwen3.8-27B-UD-Q4_K_XL'])
  assert.equal(modelLabel('openrouter/google/gemma-4-31b-it'), 'google/gemma-4-31b-it')
  assert.equal(modelLabel('gemma-4-31B-it-UD-Q4_K_XL'), 'gemma-4-31B-it-UD-Q4_K_XL')
  // A saved model whose worker is gone keeps its provider listed.
  assert.deepEqual(providerChoices(models, 'openai/gpt-6-luna'), ['local', 'anthropic', 'openai', 'openrouter'])
  assert.equal(providerLabel('my-vllm'), 'my-vllm')
})

test('reasoning offers what the model takes, and a switch drops what the new one does not', () => {
  assert.deepEqual(effortChoices(models, 'gemma-4-31B-it-UD-Q4_K_XL'), ['none'])
  assert.deepEqual(effortChoices(models, 'anthropic/claude-haiku-5-5'), all)
  assert.deepEqual(effortChoices(models, 'anthropic/claude-haiku-5-5').map(effortLabel), ['なし', '低', '中', '高', 'とても高い', '最大'])
  const haiku = chooseModel(settings, models, 'anthropic/claude-haiku-5-5')
  assert.deepEqual(chooseModel({ ...haiku, reasoning_effort: 'medium' }, models, 'Qwen3.8-27B-UD-Q4_K_XL').reasoning_effort, 'medium')
  assert.equal(chooseModel({ ...haiku, reasoning_effort: 'high' }, models, 'Qwen3.8-27B-UD-Q4_K_XL').reasoning_effort, 'none')
  assert.deepEqual(settings.model, 'gemma-4-31B-it-UD-Q4_K_XL')
})

test('a provider switch picks its first model and keeps the current one when already there', () => {
  assert.equal(chooseProvider(settings, models, 'openrouter').model, 'openrouter/google/gemma-4-31b-it')
  assert.equal(chooseProvider({ ...settings, model: 'Qwen3.8-27B-UD-Q4_K_XL' }, models, 'local').model, 'Qwen3.8-27B-UD-Q4_K_XL')
  assert.equal(chooseProvider(settings, models, 'openai'), settings)
  assert.equal(chooseProvider(settings, models, 'anthropic').temperature, 0.65)
})
