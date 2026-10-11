import { useEffect, useState } from 'react'
import { request } from './api'

type Usage = { enabled: boolean; model: string; caution: boolean; portraits: { name: string; look: string }[]; event_cgs: number }

/** Says which pictures of the work came out of the image editing model, when that needs care. */
export function ImageEditingNotice({ projectId, refreshKey = 0 }: { projectId: string; refreshKey?: number | string }) {
  const [usage, setUsage] = useState<Usage | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    request<Usage>(`/api/image-editing/projects/${encodeURIComponent(projectId)}`, controller.signal)
      .then(next => {
        setUsage(next)
        // Labels on editing operations follow the model in use.
        document.documentElement.dataset.editCaution = next.caution ? 'on' : 'off'
      })
      .catch(() => { /* The list is advisory; a failed fetch must not block the page. */ })
    return () => controller.abort()
  }, [projectId, refreshKey])
  if (!usage || !usage.caution || (!usage.portraits.length && !usage.event_cgs)) return null
  const counts = [usage.portraits.length ? `立ち絵${usage.portraits.length}点` : '', usage.event_cgs ? `イベントCG${usage.event_cgs}点` : ''].filter(Boolean).join('、')
  return <details className="image-editing-notice">
    <summary><span className="edit-model-tag">画像編集</span>この作品には、画像編集モデル（{usage.model}）で作った素材が含まれます：{counts}</summary>
    {usage.portraits.length > 0 && <ul>{usage.portraits.map((item, index) => <li key={index}>{item.name}（{item.look}）</li>)}</ul>}
    {usage.event_cgs > 0 && <p>イベントCGはすべて画像編集モデルで作ります。</p>}
    <p>公開や配布の前に、このモデルの生成物の利用条件を確認してください。{usage.enabled ? '避けたい場合は、世界観・メインキャラの指示の画面で「画像編集を使わない」を選べます。' : 'この作品は画像編集を使わない設定ですが、上の素材は設定の前に作られたものです。'}</p>
  </details>
}
