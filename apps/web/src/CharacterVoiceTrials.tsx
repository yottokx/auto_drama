import { useEffect, useState } from 'react'
import { Icon } from './Icons'
import type { VoiceTest } from './m2Api'
import type { CharacterBrief } from './wizardState'

export function CharacterVoiceTrials({ characterId, characterName, characters, onSelect, referenceId, tests, busy, onGenerate, onPendingChange }: {
  characterId: string; characterName: string; referenceId?: string | null; tests: VoiceTest[]; busy: boolean
  characters: CharacterBrief[]; onSelect: (id: string) => void
  onGenerate: (characterId: string, text: string) => Promise<boolean>; onPendingChange: (pending: boolean) => void
}) {
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [message, setMessage] = useState('')
  const text = drafts[characterId] ?? ''
  const hasPending = Object.values(drafts).some(value => value.trim())
  useEffect(() => onPendingChange(hasPending), [hasPending, onPendingChange])
  useEffect(() => setMessage(''), [characterId])
  async function generate() {
    if (busy || !referenceId || !text.trim() || text.length > 1000) return
    if (await onGenerate(characterId, text.trim())) {
      setDrafts(current => ({ ...current, [characterId]: '' }))
      setMessage('台詞の音声生成を受け付けました。完了すると試聴履歴に追加されます。')
    }
  }
  return <div className="char-voice-trials">
    {tests.length > 0 && <section className="char-trial-results" aria-label={`${characterName}の台詞の試聴履歴`}>
      <h4>この声で生成した台詞</h4>
      {[...tests].reverse().map(test => <article key={test.id}>
        <blockquote>{test.text}</blockquote>
        <audio controls preload="metadata" src={test.url} aria-label={`${characterName}の台詞：${test.text}`}/>
        <div className="char-trial-meta"><time dateTime={test.createdAt}>{new Date(test.createdAt).toLocaleString('ja-JP')}</time>{test.sourceVoiceArtifactId !== referenceId && <span>以前のサンプル音声で生成</span>}</div>
      </article>)}
    </section>}
    <details className="char-clone-panel">
      <summary><Icon name="volume" size={15}/><span>この声で、好きな台詞を試す</span>{text.trim() && <small className="char-unsaved-label">入力中</small>}<Icon name="chevron" size={13}/></summary>
      <p>サンプル音声をもとに声を再現します。試した音声はサンプル音声を置き換えず、履歴に残ります。</p>
      {!referenceId && <p className="char-lock-notice">サンプル音声の生成が完了すると利用できます。</p>}
      <label className="field-label">読み上げる台詞<textarea rows={4} maxLength={1000} value={text} disabled={busy || !referenceId} placeholder="このキャラクターに話してほしい台詞を自由に入力してください。" onChange={event => { setDrafts(current => ({ ...current, [characterId]: event.target.value })); setMessage('') }}/><span className="field-hint">{text.length} / 1,000文字</span></label>
      <div className="char-edit-actions"><button type="button" className="button button-light" disabled={busy || !text} onClick={() => setDrafts(current => ({ ...current, [characterId]: '' }))}>入力を破棄</button><button type="button" className="button button-primary" disabled={busy || !referenceId || !text.trim()} onClick={() => void generate()}><Icon name="volume" size={14}/>この台詞の音声を生成</button></div>
      {message && <p className="char-save-status" role="status">{message}</p>}
    </details>
    {characters.some(character => character.id !== characterId && drafts[character.id]?.trim()) && <div className="char-trial-drafts"><p>ほかのキャラクターに入力中の台詞があります。</p>{characters.map((character, index) => character.id !== characterId && drafts[character.id]?.trim() ? <button key={character.id} type="button" onClick={() => onSelect(character.id)}>{character.name || `キャラクター ${index + 1}`}の台詞を確認<Icon name="arrow" size={12}/></button> : null)}</div>}
  </div>
}
