import { useState } from 'react'
import { errorMessage, request } from './api'
import { Icon } from './Icons'
import { Dialog } from './PreviewComponents'

type Level = 'brief' | 'world' | 'cast' | 'plan'
const levels: { id: Level; title: string; carries: string; starts: string }[] = [
  { id: 'brief', title: 'STEP1の指示まで', carries: '世界観とメインキャラの指示、別バージョンと共通の素材の指定', starts: '世界観の生成から始めます。' },
  { id: 'world', title: 'STEP2の世界観まで', carries: '上に加えて、世界観と共通の素材の絵', starts: 'メインキャラの生成から始めます。' },
  { id: 'cast', title: 'STEP3のメインキャラまで', carries: '上に加えて、メインキャラの設定・立ち絵・声・別バージョン・関係性', starts: '全体計画の生成から始めます。' },
  { id: 'plan', title: 'STEP4の全体計画まで', carries: '上に加えて、全体計画とサブキャラの立ち絵・声', starts: '全体計画の確認から始め、承認すると本編を制作します。' },
]

/** Start another work from a chosen step of this one; this one stays exactly as it is. */
export function ForkWork({ projectId, disabled = false, onForked }: { projectId: string; disabled?: boolean; onForked: (projectId: string) => void }) {
  const [available, setAvailable] = useState<Level[] | null>(null)
  const [chosen, setChosen] = useState<Level>('brief')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const endpoint = `/api/forks/projects/${encodeURIComponent(projectId)}`
  async function open() {
    setError('')
    try {
      const result = await request<{ levels: Level[] }>(endpoint, new AbortController().signal)
      setAvailable(result.levels); setChosen(result.levels[result.levels.length - 1] ?? 'brief')
    } catch (reason) { setAvailable([]); setError(errorMessage(reason)) }
  }
  async function fork() {
    setSending(true); setError('')
    try {
      const result = await request<{ project: { id: string } }>(endpoint, new AbortController().signal, { through: chosen })
      setAvailable(null); onForked(result.project.id)
    } catch (reason) { setError(errorMessage(reason)) } finally { setSending(false) }
  }
  return <>
    <div className="fork-work"><button type="button" className="button button-light" disabled={disabled} onClick={() => void open()}><Icon name="plus" size={15}/>この作品から別の作品を作る</button><p>途中まで同じ内容を引き継いだ、新しい作品を作ります。別の案を試したい時に使います。この作品は変わりません。</p></div>
    {available && <Dialog title="この作品から別の作品を作る" onClose={() => { if (!sending) setAvailable(null) }}>
      <p className="dialog-intro">どこまでを引き継ぐかを選んでください。それより後のステップは、新しい作品で最初から作ります。もとの作品はそのまま残ります。</p>
      <div className="fork-levels" role="radiogroup" aria-label="引き継ぐ範囲">{levels.map(level => {
        const ready = available.includes(level.id)
        return <label key={level.id} className={`${chosen === level.id ? 'is-selected' : ''} ${ready ? '' : 'is-disabled'}`}>
          <input type="radio" name="fork-level" disabled={!ready || sending} checked={chosen === level.id} onChange={() => setChosen(level.id)}/>
          <span><strong>{level.title}</strong><small>{level.carries}</small><small>{ready ? level.starts : 'このステップはまだ出来上がっていません。'}</small></span></label>
      })}</div>
      {error && <p className="m2-job-error" role="alert">{error}</p>}
      <div className="dialog-actions"><button type="button" className="button button-light" disabled={sending} onClick={() => setAvailable(null)}>キャンセル</button><button type="button" className="button button-primary" disabled={sending || !available.includes(chosen)} onClick={() => void fork()}><Icon name="check" size={15}/>別の作品を作る</button></div>
    </Dialog>}
  </>
}
