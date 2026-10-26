// Which models the LLM settings offer, grouped by where they run. Workers list the models
// in their config: local GGUF files by name, API models as provider/model after the
// provider IDs their config defines (a file name cannot contain '/').
// sampling_efforts: the reasoning efforts with which the model takes temperature and top_p (absent: all).
export type LLMModel = { model: string; reasoning_efforts: string[]; sampling_efforts?: string[]; worker_name: string }

const EFFORT_LABELS: Record<string, string> = { none: 'なし', low: '低', medium: '中', high: '高', xhigh: 'とても高い', max: '最大' }

export function modelProvider(model: string): string {
  const slash = model.indexOf('/')
  return slash > 0 ? model.slice(0, slash) : 'local'
}

/** API providers are shown by the ID the worker config gives them. */
export function providerLabel(provider: string): string {
  return provider === 'local' ? 'ローカル' : provider
}

/** The model's own name, without the provider it is listed under. */
export function modelLabel(model: string): string {
  return modelProvider(model) === 'local' ? model : model.slice(model.indexOf('/') + 1)
}

export function effortLabel(effort: string): string {
  return EFFORT_LABELS[effort] ?? effort
}

/** Providers with at least one connected model, local first; the current one stays listed. */
export function providerChoices(models: LLMModel[], current?: string): string[] {
  const providers = new Set(models.map(model => modelProvider(model.model)))
  if (current) providers.add(modelProvider(current))
  return [...providers].sort((a, b) => Number(b === 'local') - Number(a === 'local') || a.localeCompare(b))
}

export function modelChoices(models: LLMModel[], provider: string): string[] {
  return [...new Set(models.filter(model => modelProvider(model.model) === provider).map(model => model.model))]
}

/** Every effort some connected worker accepts for the model; 'none' always. */
export function effortChoices(models: LLMModel[], model: string): string[] {
  const efforts = new Set(['none'])
  for (const row of models) if (row.model === model) for (const effort of row.reasoning_efforts) efforts.add(effort)
  const order = Object.keys(EFFORT_LABELS)
  return [...efforts].sort((a, b) => (order.indexOf(a) + 1 || order.length + 1) - (order.indexOf(b) + 1 || order.length + 1))
}

/** Switch to a model, dropping a reasoning effort it does not take. */
export function chooseModel<T extends { model: string; reasoning_effort: string }>(settings: T, models: LLMModel[], model: string): T {
  const efforts = effortChoices(models, model)
  return { ...settings, model, reasoning_effort: efforts.includes(settings.reasoning_effort) ? settings.reasoning_effort : 'none' }
}

/** Switch to a provider's first model, unless the current model is already one of its. */
export function chooseProvider<T extends { model: string; reasoning_effort: string }>(settings: T, models: LLMModel[], provider: string): T {
  if (modelProvider(settings.model) === provider) return settings
  const first = modelChoices(models, provider)[0]
  return first ? chooseModel(settings, models, first) : settings
}
