import { useEffect, useState } from 'react'
import { errorMessage, request } from './api'
import { Icon } from './Icons'

type ImageMetadata = {
  prompt?: string; negative_prompt?: string; seed?: number
  model?: string; width?: number; height?: number; steps?: number; guidance_scale?: number
}

function imageMetadata(value: unknown): ImageMetadata {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const record = value as Record<string, unknown>
  const result: ImageMetadata = {}
  for (const key of ['prompt', 'negative_prompt', 'model'] as const) {
    if (typeof record[key] === 'string') result[key] = record[key]
  }
  for (const key of ['seed', 'width', 'height', 'steps', 'guidance_scale'] as const) {
    if (typeof record[key] === 'number' && Number.isFinite(record[key])) result[key] = record[key]
  }
  return result
}

export function ImageGenerationDetails({ artifactId }: { artifactId: string }) {
  const [open, setOpen] = useState(false)
  const [metadata, setMetadata] = useState<ImageMetadata | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (!open || metadata) return
    const controller = new AbortController()
    setError('')
    void request<{ id: string; provenance?: { image?: unknown } }>(`/api/artifacts/${encodeURIComponent(artifactId)}`, controller.signal)
      .then(artifact => {
        if (controller.signal.aborted) return
        if (artifact.id !== artifactId) throw new Error('表示中の立ち絵の生成情報を取得できませんでした。')
        setMetadata(imageMetadata(artifact.provenance?.image))
      })
      .catch(reason => { if (!controller.signal.aborted) setError(errorMessage(reason)) })
    return () => controller.abort()
  }, [artifactId, open, metadata, attempt])

  return <details className="char-generation-details" open={open} onToggle={event => setOpen(event.currentTarget.open)}>
    <summary><Icon name="settings" size={14}/><span>生成の詳細</span><Icon name="chevron" size={14}/></summary>
    {open && <div className="char-generation-content">
      <p>この立ち絵を生成したときの記録です。</p>
      {error ? <div className="char-generation-error" role="alert"><p>{error}</p><button type="button" className="button button-light" onClick={() => setAttempt(value => value + 1)}>再読み込み</button></div> : !metadata ? <p role="status">生成情報を読み込んでいます…</p> : <>
        <dl className="char-generation-values">
          <div><dt>使用したプロンプト</dt><dd>{metadata.prompt === undefined ? '記録されていません' : metadata.prompt || '指定なし'}</dd></div>
          <div><dt>ネガティブプロンプト</dt><dd>{metadata.negative_prompt === undefined ? '記録されていません' : metadata.negative_prompt || '指定なし'}</dd></div>
          <div><dt>Seed</dt><dd>{metadata.seed ?? '記録されていません'}</dd></div>
          {metadata.model !== undefined && <div><dt>モデル</dt><dd>{metadata.model}</dd></div>}
          {metadata.width !== undefined && metadata.height !== undefined && <div><dt>生成サイズ</dt><dd>{metadata.width} × {metadata.height}</dd></div>}
          {metadata.steps !== undefined && <div><dt>ステップ数</dt><dd>{metadata.steps}</dd></div>}
          {metadata.guidance_scale !== undefined && <div><dt>CFG</dt><dd>{metadata.guidance_scale}</dd></div>}
        </dl>
        {(metadata.prompt === undefined || metadata.negative_prompt === undefined || metadata.seed === undefined) && <p>古い生成結果には、一部の情報が保存されていない場合があります。</p>}
      </>}
    </div>}
  </details>
}
