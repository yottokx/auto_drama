import { useEffect, useRef, useState } from 'react'
import { ApiError, errorMessage } from './api'
import { Icon } from './Icons'
import { PortraitCanvas } from './PortraitEditor'
import { GenerationProgress } from './GenerationProgress'
import { Dialog } from './PreviewComponents'
import { AdjustmentPreview, type AdjustmentGeometry, type PreviewSlot } from './AdjustmentPreview'
import { AdjustmentMusicPreview } from './AdjustmentMusicPreview'
import { AdjustmentMusicSeamPreview } from './AdjustmentMusicSeamPreview'
import {
  adjustmentDirty, adjustmentJobProgress, adjustmentPayload, adjustmentSelfIntroduction, adjustmentSetting, adjustmentUploadError, changeAdjustment, initialAdjustmentEditor,
  adjustmentMusicSetting, changeAdjustmentMusic, selectAdjustmentMusicCandidate, adjustmentMusicContinuity,
  sceneMusicTransition, resetAutomaticSceneMusic,
  movePreviewCharacter, previewCast, receiveAdjustment, selectAdjustmentCandidate, togglePreviewCharacter,
  type AdjustmentCandidate, type AdjustmentEditorState, type AdjustmentKind, type AdjustmentResponse,
} from './adjustmentState'
import './adjustment.css'

export async function adjustmentRequest(path: string, signal: AbortSignal, method = 'GET', body?: object | FormData): Promise<AdjustmentResponse> {
  const multipart = body instanceof FormData
  const response = await fetch(path, {
    method, signal: AbortSignal.any([signal, AbortSignal.timeout(path.endsWith('/music/upload') ? 240000 : 120000)]),
    ...(body === undefined ? {} : multipart ? { body } : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  })
  if (!response.ok) {
    const reason = await response.json().catch(() => null)
    throw new ApiError(typeof reason?.detail === 'string' ? reason.detail : `調整内容を取得・保存できませんでした（${response.status}）。`, response.status)
  }
  return response.json()
}

export function AdjustmentEditor({ projectId, disabled = false, onPendingChange }: {
  projectId: string; disabled?: boolean; onPendingChange?: (pending: boolean) => void
}) {
  const [state, setState] = useState(initialAdjustmentEditor)
  const [characterId, setCharacterId] = useState('')
  const [sceneKey, setSceneKey] = useState('')
  const [selected, setSelected] = useState<Record<string, PreviewSlot>>({})
  const [tab, setTab] = useState<'layout' | AdjustmentKind | 'music'>('layout')
  const [musicPreviewId, setMusicPreviewId] = useState('')
  const [musicPlayRequest, setMusicPlayRequest] = useState(0)
  const [musicAuditionMode, setMusicAuditionMode] = useState<'candidate' | 'seam'>('candidate')
  const [prompts, setPrompts] = useState<Record<string, string>>({})
  const [sourcePrompts, setSourcePrompts] = useState<Record<string, string>>({})
  const [submittedPrompts, setSubmittedPrompts] = useState<Record<string, string>>({})
  const [enlarged, setEnlarged] = useState<{ url: string; name: string } | null>(null)
  const [promptCandidateId, setPromptCandidateId] = useState<string | null>(null)
  const [transcripts, setTranscripts] = useState<Record<string, string>>({})
  const [submittedTranscripts, setSubmittedTranscripts] = useState<Record<string, string>>({})
  const [files, setFiles] = useState<Record<string, File>>({})
  const [error, setError] = useState('')
  const [connectionError, setConnectionError] = useState('')
  const [notice, setNotice] = useState('')
  const [mutating, setMutating] = useState(false)
  const [confirmReload, setConfirmReload] = useState(false)
  const [geometry, setGeometry] = useState<AdjustmentGeometry | null>(null)
  const [previewing, setPreviewing] = useState(false)
  const [previewError, setPreviewError] = useState('')
  const current = useRef(state)
  const mounted = useRef(false)
  const mutation = useRef(false)
  const sequence = useRef(0)
  const previewSequence = useRef(0)
  const controllers = useRef(new Set<AbortController>())
  const pendingCallback = useRef(onPendingChange)
  pendingCallback.current = onPendingChange
  const endpoint = `/api/m3/projects/${encodeURIComponent(projectId)}/adjustments`
  const source = state.source
  const draft = state.base
  const dirty = adjustmentDirty(state)
  const sourcePromptPending = Object.entries(sourcePrompts).some(([key, value]) => {
    if (key.startsWith('music:')) {
      const target = source?.scenes.find(row => `music:${row.production_id}:${row.scene_id}` === key)
      return value !== (submittedPrompts[key] ?? target?.music_prompt ?? '')
    }
    const separator = key.lastIndexOf(':')
    const cast = source?.cast.find(person => person.character_id === key.slice(0, separator))
    const original = adjustmentSetting(cast, key.slice(separator + 1) as AdjustmentKind)
    return value !== (submittedPrompts[key] ?? original)
  })
  const transcriptPending = Object.entries(transcripts).some(([id, value]) => value !== (submittedTranscripts[id] ?? adjustmentSelfIntroduction(source?.cast.find(person => person.character_id === id))))
  const formPending = sourcePromptPending || Object.values(prompts).some(value => value.trim()) || Object.keys(files).length > 0 || transcriptPending
  const blocked = disabled || Boolean(source?.readonly) || !source?.complete || state.conflict || source?.draft?.status === 'applying' || source?.draft?.status === 'applied'
  const sendingBlocked = blocked || mutating || Boolean(source?.busy)
  const character = source?.cast.find(value => value.character_id === characterId) ?? source?.cast[0]
  const edit = state.characters.find(value => value.character_id === character?.character_id)
  const kind: AdjustmentKind = tab === 'voice' ? 'voice' : 'image'
  const inputKey = `${character?.character_id ?? ''}:${kind}`
  const instruction = prompts[inputKey] ?? ''
  const settingLabel = kind === 'image' ? '外見の設定' : '声・話し方の設定'
  const originalPrompt = adjustmentSetting(character, kind)
  const sourcePrompt = sourcePrompts[inputKey] ?? originalPrompt
  const blankEditedPrompt = inputKey in sourcePrompts && !sourcePrompt.trim()
  const transcript = transcripts[character?.character_id ?? ''] ?? adjustmentSelfIntroduction(character)
  const file = files[inputKey]
  const scene = source?.scenes.find(value => `${value.chapter_number}:${value.scene_id}` === sceneKey) ?? source?.scenes[0]
  const musicInputKey = `music:${scene?.production_id ?? ''}:${scene?.scene_id ?? ''}`
  const musicSetting = adjustmentMusicSetting(state, scene)
  const musicInstruction = prompts[musicInputKey] ?? ''
  const musicPrompt = sourcePrompts[musicInputKey] ?? scene?.music_prompt ?? ''
  const musicFile = files[musicInputKey]
  const musicContinuity = adjustmentMusicContinuity(state)
  const musicErrors = musicContinuity.filter(row => row.error)
  const musicActive = musicContinuity.find(row => row.scene.production_id === scene?.production_id && row.scene.scene_id === scene?.scene_id)
  const activeIndex = musicActive ? musicContinuity.indexOf(musicActive) : -1
  const musicPrevious = activeIndex > 0 && musicContinuity[activeIndex - 1].scene.production_id === scene?.production_id ? musicContinuity[activeIndex - 1] : undefined
  const musicTransition = sceneMusicTransition(musicSetting)
  const automaticMusicAvailable = scene?.music_plan?.action !== 'play' || Boolean(source?.music_candidates?.some(candidate => candidate.id === scene.music_plan?.candidate_id && candidate.production_id === scene.production_id && candidate.scene_id === scene.scene_id && candidate.artifact_id && candidate.music_url))
  const musicCandidates = source?.music_candidates?.filter(value => value.production_id === scene?.production_id && value.scene_id === scene?.scene_id) ?? []
  const musicPreview = musicCandidates.find(value => value.id === musicPreviewId)
    ?? musicCandidates.find(value => value.id === musicSetting?.candidate_id)
    ?? musicActive?.candidate
    ?? musicCandidates.find(value => value.music_url || value.source_url)
  const chapterNumbers = [...new Set(source?.scenes.map(value => value.chapter_number) ?? [])]
  const candidates = source?.candidates.filter(value => value.character_id === character?.character_id && value.kind === kind) ?? []
  const chosenImage = source?.candidates.find(value => value.id === edit?.image_candidate_id)
  const promptCandidate = source?.candidates.find(value => value.id === promptCandidateId)
  const limits = source?.limits
  const working = source?.jobs.filter(job => job.status === 'running' || job.status === 'pending') ?? []
  const failed = source?.jobs.filter(job => job.status === 'failed') ?? []
  const latestMusicReplan = source?.jobs.filter(job => job.purpose === 'music_replan').at(-1)

  function commit(next: AdjustmentEditorState) { current.current = next; setState(next) }
  function accept(next: AdjustmentResponse, mode: 'poll' | 'adopt' | 'candidate' = 'poll') {
    if (next.project_id !== projectId) throw new Error('作品の調整内容を確認できませんでした。')
    const updated = receiveAdjustment(current.current, next, mode)
    commit(updated)
    if (!adjustmentDirty(updated) && !updated.conflict) setGeometry(updated.base?.geometry ?? null)
  }

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; sequence.current += 1; controllers.current.forEach(controller => controller.abort()); pendingCallback.current?.(false) }
  }, [projectId])
  useEffect(() => { pendingCallback.current?.(dirty || formPending || mutating) }, [dirty, formPending, mutating])

  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      const controller = new AbortController(); controllers.current.add(controller)
      const version = sequence.current
      try {
        const next = await adjustmentRequest(endpoint, controller.signal)
        if (!stopped && !mutation.current && version === sequence.current) { accept(next); setConnectionError('') }
      } catch (reason) {
        if (!stopped && !controller.signal.aborted) setConnectionError(errorMessage(reason))
      } finally { controllers.current.delete(controller); if (!stopped) timer = setTimeout(poll, 2000) }
    }
    void poll()
    return () => { stopped = true; clearTimeout(timer) }
  }, [endpoint])

  const previewPayload = JSON.stringify(adjustmentPayload(state))
  useEffect(() => {
    if (!draft || !dirty || state.conflict || blocked || musicErrors.length) { setPreviewing(false); return }
    const controller = new AbortController(); controllers.current.add(controller)
    const version = ++previewSequence.current
    setPreviewing(true); setPreviewError('')
    const timer = setTimeout(async () => {
      try {
        const next = await adjustmentRequest(`${endpoint}/preview`, controller.signal, 'POST', JSON.parse(previewPayload))
        if (!controller.signal.aborted && mounted.current && version === previewSequence.current) {
          if (next.project_id !== projectId || next.draft?.id !== draft.id) throw new Error('プレビューの版を確認できませんでした。')
          setGeometry(next.draft.geometry)
        }
      } catch (reason) {
        if (!controller.signal.aborted && mounted.current && version === previewSequence.current) setPreviewError(errorMessage(reason))
      } finally { controllers.current.delete(controller); if (!controller.signal.aborted && mounted.current && version === previewSequence.current) setPreviewing(false) }
    }, 250)
    return () => { clearTimeout(timer); controller.abort(); controllers.current.delete(controller) }
  }, [previewPayload, dirty, blocked, endpoint, projectId, draft?.id, state.conflict, musicErrors.length])

  useEffect(() => {
    if (!source || sceneKey || !source.scenes.length) return
    const first = source.scenes[0]
    setSceneKey(`${first.chapter_number}:${first.scene_id}`)
    setSelected(previewCast(first, source.cast.map(value => value.character_id)))
  }, [source, sceneKey])

  function changeScene(key: string) {
    const next = source?.scenes.find(value => `${value.chapter_number}:${value.scene_id}` === key)
    setSceneKey(key); setSelected(previewCast(next, source?.cast.map(value => value.character_id) ?? []))
    setMusicPreviewId(''); setMusicPlayRequest(0)
    setMusicAuditionMode('candidate')
  }
  function change(patch: Partial<NonNullable<typeof edit>>) {
    if (!edit || blocked || mutation.current) return
    commit(changeAdjustment(current.current, edit.character_id, patch)); setNotice('')
  }

  async function action(name: 'start' | 'save' | 'generate' | 'upload' | 'apply' | 'retry' | 'sample' | 'music/generate' | 'music/upload' | 'music/replan', candidate?: AdjustmentCandidate) {
    if (mutation.current || disabled || source?.readonly || !source?.complete || state.conflict) return
    if (name !== 'start' && (!draft || (source?.draft?.status === 'applying' && name !== 'retry'))) return
    if (name === 'apply' && (dirty || formPending)) return
    if ((name === 'save' || name === 'apply') && musicErrors.length) return
    if (name === 'music/replan' && dirty) return
    const controller = new AbortController(); controllers.current.add(controller)
    mutation.current = true; sequence.current += 1; setMutating(true); setError(''); setNotice('')
    try {
      let body: object | FormData = { expected_revision: draft?.revision }
      if (name === 'start') body = source?.edition ? { expected_edition_id: source.edition.id } : {}
      if (name === 'save') body = adjustmentPayload(current.current)
      if (name === 'generate') {
        if (blankEditedPrompt) throw new Error(`${settingLabel}を入力するか、元の設定に戻してください。`)
        if (!character || (!sourcePrompt.trim() && !instruction.trim())) throw new Error(`${settingLabel}か変更指示を入力してください。`)
        if (kind === 'voice' && !transcript.trim()) throw new Error('音声に含まれる読み上げ文を入力してください。')
        body = { expected_revision: draft?.revision, character_id: character.character_id, kind, ...(sourcePrompt.trim() ? { source_prompt: sourcePrompt } : {}), instruction: instruction.trim(), ...(kind === 'voice' ? { reference_text: transcript.trim() } : {}) }
      }
      if (name === 'upload') {
        if (!character || !file || !limits) throw new Error('取り込むファイルを選んでください。')
        const problem = adjustmentUploadError(file, kind, limits.upload_bytes)
        if (problem) throw new Error(problem)
        if (kind === 'voice' && !transcript.trim()) throw new Error('音声に含まれる読み上げ文を入力してください。')
        const form = new FormData()
        form.set('expected_revision', String(draft?.revision)); form.set('character_id', character.character_id); form.set('kind', kind); form.set('file', file)
        if (kind === 'voice') form.set('reference_text', transcript.trim())
        body = form
      }
      if (name === 'sample') body = { expected_revision: draft?.revision, candidate_id: candidate?.id }
      if (name === 'music/generate') {
        if (!scene?.production_id) throw new Error('BGMを生成する場面を選んでください。')
        body = { expected_revision: draft?.revision, production_id: scene.production_id, scene_id: scene.scene_id,
          instruction: musicInstruction.trim(), ...(musicPrompt.trim() ? { source_prompt: musicPrompt.trim() } : {}) }
      }
      if (name === 'music/replan') {
        if (!scene?.production_id) throw new Error('場面のつながりを見直す章を選んでください。')
        body = { expected_revision: draft?.revision, production_id: scene.production_id }
      }
      if (name === 'music/upload') {
        if (!scene?.production_id || !musicFile || !limits) throw new Error('場面と取り込む音源を選んでください。')
        const problem = adjustmentUploadError(musicFile, 'voice', limits.music_upload_bytes ?? limits.upload_bytes)
        if (problem) throw new Error(problem)
        const form = new FormData()
        form.set('expected_revision', String(draft?.revision)); form.set('production_id', scene.production_id)
        form.set('scene_id', scene.scene_id); form.set('file', musicFile); body = form
      }
      const next = await adjustmentRequest(name === 'save' ? endpoint : `${endpoint}/${name}`, controller.signal, name === 'save' ? 'PUT' : 'POST', body)
      if (!mounted.current || controller.signal.aborted) return
      accept(next, ['generate', 'upload', 'sample', 'retry', 'music/generate', 'music/upload', 'music/replan'].includes(name) ? 'candidate' : 'adopt')
      if (name === 'generate') {
        setPrompts(previous => ({ ...previous, [inputKey]: '' }))
        setSubmittedPrompts(previous => ({ ...previous, [inputKey]: sourcePrompt }))
      }
      if (name === 'upload') setFiles(previous => { const updated = { ...previous }; delete updated[inputKey]; return updated })
      if (name === 'music/generate') {
        setPrompts(previous => ({ ...previous, [musicInputKey]: '' }))
        setSubmittedPrompts(previous => ({ ...previous, [musicInputKey]: musicPrompt }))
      }
      if (name === 'music/upload') setFiles(previous => { const updated = { ...previous }; delete updated[musicInputKey]; return updated })
      if ((name === 'upload' || name === 'generate') && kind === 'voice') setSubmittedTranscripts(previous => ({ ...previous, [character!.character_id]: transcript }))
      setNotice(name === 'save' ? '調整内容を下書きに保存しました。本編に反映するには「調整版を反映」を押してください。' : name === 'apply' ? '調整版の反映を開始しました。完成後、すべての対象章をまとめて切り替えます。' : name === 'music/generate' ? '場面のBGM候補を生成します。試聴してから下書きで選択してください。' : name === 'music/upload' ? 'BGM候補を取り込みました。ループ区間と音源を確認してから選択できます。' : name === 'upload' ? '候補を保存しました。プレビュー・試聴してから選択できます。' : name === 'generate' ? '今回の指示で候補の生成を受け付けました。人物設定は変更されません。' : name === 'retry' ? '失敗した処理の再試行を受け付けました。' : name === 'sample' ? '既存の台詞を使う短い試聴を受け付けました。' : '')
      if (name === 'music/replan') setNotice('この章の場面のつながりを見直します。既存の音源を使い、物語や音源の生成は行いません。')
    } catch (reason) {
      if (mounted.current && !controller.signal.aborted) {
        if (reason instanceof ApiError && reason.status === 409) commit({ ...current.current, conflict: true })
        setError(errorMessage(reason))
      }
    } finally {
      controllers.current.delete(controller); mutation.current = false; sequence.current += 1
      if (mounted.current) setMutating(false)
    }
  }

  async function reload() {
    if (mutating) return
    if ((dirty || formPending) && !confirmReload) { setConfirmReload(true); return }
    const controller = new AbortController(); controllers.current.add(controller)
    mutation.current = true; sequence.current += 1; setMutating(true); setError('')
    try {
      const next = await adjustmentRequest(endpoint, controller.signal)
      if (!mounted.current || controller.signal.aborted) return
      accept(next, 'adopt'); setPrompts({}); setSourcePrompts({}); setSubmittedPrompts({}); setTranscripts({}); setSubmittedTranscripts({}); setFiles({}); setConfirmReload(false); setNotice('')
    } catch (reason) { if (mounted.current && !controller.signal.aborted) setError(errorMessage(reason)) }
    finally { controllers.current.delete(controller); mutation.current = false; sequence.current += 1; if (mounted.current) setMutating(false) }
  }
  const sourceNames = { original: '元の素材', generated: '生成した候補', upload: '取り込んだ候補', reused: '既存音源を再利用' }
  const statusNames = { pending: '待機中', running: '処理中', completed: '完了', failed: '再試行が必要' }

  return <section className="adjustment-editor" aria-label="完成後の調整">
    <div className="adjustment-content" aria-busy={mutating}>
      <p className="adjustment-intro">実際の背景で立ち絵の位置と大きさを確認し、画像・声・場面のBGM候補を選べます。変更は下書きに保存し、調整版の完成後に全対象章へまとめて反映します。</p>
      {connectionError && <p className="m2-job-error" role="alert">{connectionError}</p>}
      {error && <p className="m2-job-error" role="alert">{error}</p>}
      {!source && !connectionError && <p role="status">調整内容を読み込み中…</p>}
      {source && <>
        {(disabled || source.readonly || !source.complete) && <p className="adjustment-notice">編集できるのは、全章完成後の現在の作品です。履歴から復元した制作途中の状態では編集できません。</p>}
        {state.conflict && <p className="adjustment-notice" role="alert">調整版が別の画面で更新されました。入力は保持しています。最新の下書きを読み直してから続けてください。</p>}
        {(state.conflict || dirty || formPending) && <div className="adjustment-actions"><button type="button" className="button button-light" disabled={mutating} onClick={() => void reload()}>{confirmReload ? '入力を破棄して最新の下書きを読み込む' : '入力を取り消して読み直す'}</button>{confirmReload && <button type="button" className="button button-light" onClick={() => setConfirmReload(false)}>入力を残す</button>}</div>}
        {(!draft || draft.status === 'applied') && <div className="adjustment-actions"><button type="button" className="button button-primary" disabled={disabled || source.readonly || !source.complete || source.busy || mutating} onClick={() => void action('start')}>{draft?.status === 'applied' ? 'この版から次の調整を始める' : '調整を始める'}</button>{draft?.status === 'applied' && <span role="status">調整版を反映しました。制作画面の鑑賞・書き出しから確認できます。</span>}</div>}
        {draft && <>
          {tab !== 'music' && <div className="adjustment-character-heading"><label>編集対象の人物<select value={character?.character_id ?? ''} onChange={event => setCharacterId(event.target.value)}>{source.cast.map(cast => <option key={cast.character_id} value={cast.character_id}>{cast.name}（{cast.role === 'main' ? 'メイン' : 'サブ'}）</option>)}</select></label>{character && <span>登場章：{character.chapter_numbers.map(number => `第${number}章`).join('・')}</span>}</div>}
          <div className="adjustment-kind-tabs" role="tablist" aria-label="編集対象">
            {([{ id: 'layout', label: '表示位置調整' }, { id: 'image', label: '立ち絵' }, { id: 'voice', label: '基準音声' }, { id: 'music', label: 'BGM' }] as const).map((item, index, tabs) => <button key={item.id} type="button" role="tab" id={`adjustment-tab-${item.id}`} aria-controls={`adjustment-panel-${item.id}`} aria-selected={tab === item.id} tabIndex={tab === item.id ? 0 : -1} onClick={() => { setTab(item.id); setMusicPlayRequest(0); setMusicAuditionMode('candidate') }} onKeyDown={event => {
              const next = event.key === 'ArrowRight' ? tabs[(index + 1) % tabs.length] : event.key === 'ArrowLeft' ? tabs[(index + tabs.length - 1) % tabs.length] : event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs[tabs.length - 1] : null
              if (next) { event.preventDefault(); setTab(next.id); setMusicPlayRequest(0); setMusicAuditionMode('candidate'); document.getElementById(`adjustment-tab-${next.id}`)?.focus() }
            }}>{item.label}</button>)}
          </div>
          {tab === 'layout' && <section id="adjustment-panel-layout" role="tabpanel" aria-labelledby="adjustment-tab-layout" tabIndex={0}>
          <div className="adjustment-workspace"><div className="adjustment-preview-column">
          <div className="adjustment-preview-tools">
            <label>章<select value={scene?.chapter_number ?? ''} onChange={event => { const first = source.scenes.find(value => value.chapter_number === Number(event.target.value)); if (first) changeScene(`${first.chapter_number}:${first.scene_id}`) }}>{chapterNumbers.map(number => <option key={number} value={number}>第{number}章</option>)}</select></label>
            <label>場面<select value={scene ? `${scene.chapter_number}:${scene.scene_id}` : ''} onChange={event => changeScene(event.target.value)}>{source.scenes.filter(value => value.chapter_number === scene?.chapter_number).map((value, index) => <option key={value.scene_id} value={`${value.chapter_number}:${value.scene_id}`}>{value.title || `場面${index + 1}`}</option>)}</select></label>
            <span>本編と同じ比率 · 960 × 640</span>
          </div>
          {geometry && <AdjustmentPreview geometry={geometry} backgroundUrl={scene?.background_url ?? null} pending={previewing} characters={Object.entries(selected).flatMap(([id, slot]) => {
            const cast = source.cast.find(value => value.character_id === id)
            const settings = state.characters.find(value => value.character_id === id)
            const image = source.candidates.find(value => value.id === settings?.image_candidate_id)?.url ?? cast?.image_url
            return cast && image ? [{ id, name: cast.name, imageUrl: image, slot }] : []
          })}/>}
          {previewError && <p className="m2-job-error" role="alert">プレビューを更新できませんでした。{previewError}</p>}
          </div><div className="adjustment-controls-column">
          <div className="adjustment-selection"><div><h3>プレビューに表示する人物</h3><p>最大3人。人物や左右の位置の選択は、本編の出演・演出を変更しません。</p></div><span>{Object.keys(selected).length} / 3人</span></div>
          <div className="adjustment-cast" role="group" aria-label="プレビューに表示する人物">{source.cast.map(cast => <div key={cast.character_id} className={selected[cast.character_id] ? 'is-selected' : ''}>
            <label><input type="checkbox" checked={Boolean(selected[cast.character_id])} disabled={!selected[cast.character_id] && Object.keys(selected).length >= 3} onChange={() => setSelected(previous => togglePreviewCharacter(previous, cast.character_id))}/><span>{cast.name}<small>{cast.role === 'main' ? 'メイン' : 'サブ'}</small></span></label>
            {selected[cast.character_id] && <select aria-label={`${cast.name}のプレビュー表示位置`} value={selected[cast.character_id]} onChange={event => setSelected(previous => movePreviewCharacter(previous, cast.character_id, event.target.value as PreviewSlot))}><option value="left">左</option><option value="center">中央</option><option value="right">右</option></select>}
          </div>)}</div>
          {character && edit && <fieldset className="adjustment-layout-controls" disabled={blocked || mutating}><legend>{character.name}の表示位置と倍率</legend>
              {!selected[character.character_id] && <p>この人物はプレビューに表示されていません。上の一覧でチェックすると、調整を確認できます。</p>}
              <AdjustmentRange label="表示高さ" name="offset-y" value={edit.offset_y} min={-640} max={640} step={1} suffix="px" onChange={value => change({ offset_y: value })}/>
              <AdjustmentRange label="倍率" name="scale" value={Math.round(edit.scale * 100)} min={10} max={300} step={1} suffix="%" onChange={value => change({ scale: value / 100 })}/>
              <p>高さは自動配置からの上下移動です。マイナスで上、プラスで下へ移動します。</p><button type="button" className="button button-light" onClick={() => change({ offset_y: 0, scale: 1, body_bounds: null })}>この人物を自動配置に戻す</button>
            </fieldset>}
          </div></div>
            {character && edit && chosenImage?.url && <details className="adjustment-bounds"><summary>詳細設定：人物本体の範囲</summary><p>頭頂から足元、体の左右を囲みます。設定を変えると、この画像の自動配置を計算し直します。</p><PortraitCanvas key={`${character.character_id}:${chosenImage.id}`} character={{ character_id: character.character_id, name: character.name, image_artifact_id: chosenImage.artifact_id ?? '', image_url: chosenImage.url, framing: edit.framing, height_cm: edit.height_cm, body_bounds: edit.body_bounds }} bounds={edit.body_bounds} disabled={blocked || mutating} onChange={body_bounds => change({ body_bounds })}/></details>}
          </section>}
          {tab === 'music' && <section id="adjustment-panel-music" role="tabpanel" aria-labelledby="adjustment-tab-music" tabIndex={0}>
            <div className="adjustment-preview-tools">
              <label>章<select aria-label="BGMの章" value={scene?.chapter_number ?? ''} onChange={event => { const first = source.scenes.find(value => value.chapter_number === Number(event.target.value)); if (first) changeScene(`${first.chapter_number}:${first.scene_id}`) }}>{chapterNumbers.map(number => <option key={number} value={number}>第{number}章</option>)}</select></label>
              <label>場面<select aria-label="BGMの場面" value={scene ? `${scene.chapter_number}:${scene.scene_id}` : ''} onChange={event => changeScene(event.target.value)}>{source.scenes.filter(value => value.chapter_number === scene?.chapter_number).map((value, index) => <option key={`${value.chapter_number}:${value.scene_id}`} value={`${value.chapter_number}:${value.scene_id}`}>{value.title || `場面${index + 1}`}</option>)}</select></label>
            </div>
            <p className="adjustment-help">場面ごとのBGMを試聴して選びます。生成は通常120秒。ループ区間は音源から自動探索し、冒頭を一度再生した後にA〜Bを繰り返します。</p>
            {!scene?.production_id && <p className="adjustment-notice">この場面のBGM情報を取得できませんでした。制御サーバーを更新してから読み直してください。</p>}
            {scene?.production_id && <section className="adjustment-music-plan" aria-label="自動の場面設計">
              <div><h3>自動の場面設計</h3>{scene.music_plan ? <><strong>{musicActionLabel(scene.music_plan.action)}</strong><p>{scene.music_plan.reason || 'この章の場面の流れに合わせた設定です。'}</p></> : <p>現在の素材と場面順を使っています。章全体のつながりを見直すこともできます。</p>}</div>
              <div className="adjustment-actions">{scene.music_plan && <button type="button" className="button button-light" disabled={blocked || mutating || !automaticMusicAvailable} onClick={() => { commit(resetAutomaticSceneMusic(current.current, scene)); setNotice('この場面を自動の設計に戻しました。音量は保持しています。') }}>この場面を自動設定に戻す</button>}
                <button type="button" className="button button-light" disabled={sendingBlocked || dirty} onClick={() => void action('music/replan')}>この章のつながりを見直す</button></div>
              {dirty && <p>未保存の選択・切り替え設定を保存してから、章全体の見直しを実行できます。</p>}
            </section>}
            {scene && <section className="adjustment-music-continuity" aria-label="この章のBGMのつながり"><h3>この章のBGMのつながり</h3>
              <ol>{musicContinuity.filter(row => row.scene.production_id === scene.production_id).map(row => <li key={row.scene.scene_id} className={row.scene.scene_id === scene.scene_id ? 'is-current' : ''}>
                <button type="button" aria-current={row.scene.scene_id === scene.scene_id ? 'true' : undefined} onClick={() => changeScene(`${row.scene.chapter_number}:${row.scene.scene_id}`)}>{row.scene.title}</button><span>{musicActionLabel(row.setting.action)}{row.setting.action === 'continue' && row.source_scene && ` · ${row.source_scene.title}の曲`}</span>
                {row.setting.action === 'play' && row.range && <small>{row.range.start.title}〜{row.range.end.title}で同じ曲を再生</small>}{row.error && <p role="alert">{row.error}</p>}
              </li>)}</ol>
              <p>{musicActive?.range ? `現在の曲の範囲：${musicActive.range.start.title}〜${musicActive.range.end.title}。${musicSetting?.action === 'continue' ? '前の場面から再生位置を引き継ぎます。' : 'この場面の冒頭から曲を再生します。'}` : musicActive?.error ? 'この場面のBGM設定を確認してください。' : 'この場面はBGMなしです。'}</p>
              {musicSetting?.reason && musicSetting.reason !== scene.music_plan?.reason && <p>現在の設定：{musicSetting.reason}</p>}
            </section>}
            {musicSetting && scene && <fieldset className="adjustment-music-settings" disabled={blocked || mutating}>
              <legend>この場面での再生</legend>
              <label>BGMの扱い<select aria-label="場面のBGMの扱い" value={musicSetting.action} onChange={event => {
                const action = event.target.value as 'play' | 'continue' | 'stop'
                commit(changeAdjustmentMusic(current.current, scene, { action, reason: action === 'continue' ? '手動でBGMを継続' : action === 'stop' ? '手動でBGMなしに設定' : '手動でBGMを再生', ...(action !== 'play' ? { candidate_id: null } : {}) })); setNotice('')
              }}><option value="stop">BGMなし</option><option value="continue">前の場面のBGMを続ける</option>{musicSetting.candidate_id && <option value="play">選択した候補を再生</option>}</select></label>
              <label>BGM音量 <output>{Math.round((musicSetting.action === 'continue' ? musicActive?.track_volume ?? musicSetting.volume : musicSetting.volume) * 100)}%</output><input type="range" aria-label="場面のBGM音量" disabled={musicSetting.action === 'continue'} min={0} max={100} step={1} value={Math.round((musicSetting.action === 'continue' ? musicActive?.track_volume ?? musicSetting.volume : musicSetting.volume) * 100)} onChange={event => { commit(changeAdjustmentMusic(current.current, scene, { volume: Number(event.target.value) / 100 })); setNotice('') }}/>{musicSetting.action === 'continue' && <span>曲を開始した場面の音量を引き継ぎます。</span>}</label>
              <label>画面の切り替え<select aria-label="場面の画面切り替え" value={musicTransition.visual} onChange={event => { commit(changeAdjustmentMusic(current.current, scene, { transition: { ...musicTransition, visual: event.target.value as typeof musicTransition.visual }, reason: '画面遷移を手動調整' })); setNotice('') }}><option value="none">切り替え演出なし</option><option value="dissolve">ディゾルブ</option><option value="fade">暗転して切り替え</option><option value="cut">すぐ切り替え</option></select></label>
              <details className="adjustment-music-transition-details"><summary>詳細：切り替え・音量変化の時間</summary>
                {([{ key: 'duration_ms', label: '画面の切り替え時間', max: 3000 }, { key: 'music_fade_out_ms', label: '前のBGMのフェードアウト', max: 5000 }, { key: 'music_fade_in_ms', label: '次のBGMのフェードイン', max: 5000 }] as const).map(field => <label key={field.key}>{field.label}<span><input type="number" aria-label={field.label} min={0} max={field.max} step={100} value={musicTransition[field.key]} onChange={event => { const value = Number(event.target.value); if (Number.isFinite(value)) commit(changeAdjustmentMusic(current.current, scene, { transition: { ...musicTransition, [field.key]: Math.round(Math.max(0, Math.min(field.max, value))) }, reason: '場面の切り替え時間を手動調整' })) }}/> ms</span></label>)}
                <p>継続するBGMは再スタートせず、再生位置を保ちます。切り替えとBGMの音量変化は本編と同じ設定を使います。</p>
              </details>
            </fieldset>}
            {musicActive && <AdjustmentMusicSeamPreview key={`${scene?.production_id}:${scene?.scene_id}`} previous={musicPrevious} current={musicActive} active={musicAuditionMode === 'seam'} onStart={() => { setMusicPlayRequest(0); setMusicAuditionMode('seam') }} onStop={() => setMusicAuditionMode('candidate')}/>}
            <div className="adjustment-material-workspace adjustment-music-workspace"><div className="adjustment-material-candidates">
              {musicPreview && musicAuditionMode !== 'seam' && <div className="adjustment-music-audition"><h3>候補の試聴</h3><AdjustmentMusicPreview key={musicPreview.id} candidate={musicPreview} volume={musicSetting?.action === 'continue' ? musicActive?.track_volume ?? musicSetting.volume : musicSetting?.volume ?? .35} autoPlayRequest={musicPreview.id === musicPreviewId ? musicPlayRequest : 0}/></div>}
              <div className="adjustment-candidates adjustment-candidates-music">{musicCandidates.map(candidate => {
                const selectedCandidate = musicSetting?.action === 'play' && musicSetting.candidate_id === candidate.id
                const candidateJob = source.jobs.find(job => job.id === candidate.job_id)
                const status = candidateJob?.status ?? candidate.status
                const nearSilence = candidate.quality?.near_silence_seconds ?? candidate.quality?.silence?.total_seconds
                const longestSilence = candidate.quality?.longest_near_silence_seconds ?? candidate.quality?.silence?.longest_seconds
                return <article key={candidate.id} className={selectedCandidate ? 'is-selected' : ''}><div className="adjustment-candidate-info">
                  <strong>{sourceNames[candidate.source]}</strong>
                  {status && status !== 'completed' && <span>{statusNames[status as keyof typeof statusNames] ?? status}</span>}
                  {candidate.duration_seconds != null && <p>再生範囲 {candidate.duration_seconds.toFixed(1)}秒{candidate.source_duration_seconds != null && ` / 元音源 ${candidate.source_duration_seconds.toFixed(1)}秒`}</p>}
                  {candidate.quality?.needs_review && <p className="adjustment-music-warning" role="status">品質の確認が必要です。{nearSilence != null && `無音に近い箇所：合計${nearSilence.toFixed(1)}秒。`}{longestSilence != null && `最長${longestSilence.toFixed(1)}秒。`}元音源とエラーを確認し、必要なら別の候補を作成してください。</p>}
                  {(candidate.error || candidateJob?.error) && <p className="m2-job-error" role="alert">{candidate.error || candidateJob?.error}</p>}
                  {candidate.prompt && <details><summary>この候補の生成プロンプト</summary><p>{candidate.prompt}</p></details>}
                  <button type="button" className="button button-light" disabled={!candidate.music_url && !candidate.source_url} onClick={() => { setMusicAuditionMode('candidate'); setMusicPreviewId(candidate.id); setMusicPlayRequest(previous => previous + 1) }}>この候補を試聴</button>
                  <button type="button" className={`button ${selectedCandidate ? 'button-light' : 'button-primary'}`} disabled={blocked || mutating || !scene || !candidate.music_url || !candidate.artifact_id || selectedCandidate} onClick={() => { if (scene) commit(changeAdjustmentMusic(selectAdjustmentMusicCandidate(current.current, scene, candidate), scene, { reason: '手動でBGM候補を選択' })); setNotice('') }}>{selectedCandidate ? '下書きで選択中' : 'このBGM候補を選ぶ'}</button>
                </div></article>
              })}</div>
              {!musicCandidates.length && <p>この場面のBGM候補はまだありません。</p>}
            </div><fieldset className="adjustment-new-material" disabled={sendingBlocked || !scene?.production_id}>
              <legend>新しいBGM候補</legend>
              <label>英語の音楽プロンプト（空欄で自動）<textarea aria-label="BGMの英語プロンプト" rows={5} maxLength={10000} value={musicPrompt} onChange={event => setSourcePrompts(previous => ({ ...previous, [musicInputKey]: event.target.value }))}/><span>作品全体と選択した場面から作成します。必要なら具体的な楽器・曲調を直接指定できます。</span></label>
              <label>今回だけの音楽指示（任意）<textarea aria-label="今回だけのBGM指示" rows={3} maxLength={10000} value={musicInstruction} placeholder="例：緊迫した疾走感。明確な主旋律をシンセで奏でる" onChange={event => setPrompts(previous => ({ ...previous, [musicInputKey]: event.target.value }))}/></label>
              <button type="button" className="button button-primary" disabled={sendingBlocked || !scene?.production_id} onClick={() => void action('music/generate')}><Icon name="spark" size={15}/>120秒のBGM候補を生成</button>
              <p>Stable Audio 3 Mediumでインストを生成し、ループ区間を自動探索します。候補の生成だけでは公開版を変更しません。</p>
              <div className="adjustment-upload"><label>音源から候補を取り込む<input key={`${musicInputKey}:${Boolean(musicFile)}`} type="file" accept=".mp3,.wav,audio/mpeg,audio/wav" aria-label="BGM音源の取り込み" onChange={event => {
                const selectedFile = event.target.files?.[0]
                if (selectedFile) { const problem = adjustmentUploadError(selectedFile, 'voice', limits?.music_upload_bytes ?? limits?.upload_bytes ?? 33554432); if (problem) { setError(problem); return }; setError(''); setFiles(previous => ({ ...previous, [musicInputKey]: selectedFile })) }
              }}/></label><p>MP3・WAV。{Math.floor((limits?.music_upload_bytes ?? limits?.upload_bytes ?? 33554432) / 1024 / 1024)}MiB以下。取り込み後にループ区間を自動探索します。</p>{musicFile && <p>{musicFile.name}</p>}
                <button type="button" className="button button-light" disabled={sendingBlocked || !musicFile || !scene?.production_id} onClick={() => void action('music/upload')}>BGM候補に取り込む</button>
              </div>
            </fieldset></div>
          </section>}
          {(tab === 'image' || tab === 'voice') && character && edit && <section id={`adjustment-panel-${kind}`} role="tabpanel" aria-labelledby={`adjustment-tab-${kind}`} tabIndex={0}>
            <div className="adjustment-material-heading"><h3>{character.name}の{kind === 'image' ? '立ち絵' : '基準音声'}</h3></div>
            <p className="adjustment-help">候補を選ぶと下書きのプレビューに使用します。人物設定・プロット・台本は変更されません。{kind === 'voice' && '反映すると、この人物の全台詞音声を作り直します。'}</p>
            <div className="adjustment-material-workspace"><div className="adjustment-material-candidates">
            <div className={`adjustment-candidates adjustment-candidates-${kind}`}>{candidates.map(candidate => {
              const selectedCandidate = kind === 'image' ? edit.image_candidate_id === candidate.id : edit.voice_candidate_id === candidate.id
              const candidateJob = source.jobs.find(job => job.id === candidate.job_id)
              return <article key={candidate.id} className={selectedCandidate ? 'is-selected' : ''}>
                {candidate.url && (kind === 'image' ? <button type="button" className="adjustment-image-preview" aria-label={`${character.name}の${sourceNames[candidate.source]}を拡大表示`} onClick={() => setEnlarged({ url: candidate.url!, name: `${character.name}の${sourceNames[candidate.source]}` })}><img src={candidate.url} alt={`${character.name}の立ち絵候補`} loading="lazy"/><span>クリックで拡大</span></button> : <audio controls preload="none" src={candidate.url} aria-label={`${character.name}の基準音声候補`}/>)}
                <div className="adjustment-candidate-info"><strong>{sourceNames[candidate.source]}</strong>{candidateJob && candidateJob.status !== 'completed' && <span>{statusNames[candidateJob.status]}</span>}
                  {candidate.reference_text && <p className="adjustment-transcript">{candidate.reference_text}</p>}
                  {candidate.prompt_details ? <button type="button" className="adjustment-prompt-button" onClick={() => setPromptCandidateId(candidate.id)}>設定と実際の生成指示</button> : candidate.prompt && <details><summary>この候補の生成指示</summary><p>{candidate.prompt}</p></details>}
                  {candidate.sample_url && <label>既存の台詞で試聴<audio controls preload="none" src={candidate.sample_url}/></label>}
                  {kind === 'voice' && candidate.url && !candidate.sample_url && <button type="button" className="button button-light" disabled={sendingBlocked} onClick={() => void action('sample', candidate)}>既存の台詞で試聴を作る</button>}
                  <button type="button" className={`button ${selectedCandidate ? 'button-light' : 'button-primary'}`} disabled={blocked || mutating || !candidate.url || !candidate.artifact_id || selectedCandidate} onClick={() => { commit(selectAdjustmentCandidate(current.current, candidate)); setNotice('') }}>{selectedCandidate ? '下書きで選択中' : 'この候補を選ぶ'}</button>
                </div>
              </article>
            })}</div>
            {!candidates.length && <p>保存済みの候補はありません。</p>}
            </div>
            <fieldset className="adjustment-new-material" disabled={sendingBlocked}>
              <legend>{kind === 'image' ? '新しい立ち絵の候補' : '新しい基準音声の候補'}</legend>
              <div className="adjustment-original-prompt"><h4>元の{settingLabel}</h4><p>{originalPrompt || `${settingLabel}を取得できませんでした。`}</p></div>
              <label>{settingLabel}<textarea aria-label={settingLabel} aria-invalid={blankEditedPrompt || undefined} aria-describedby={blankEditedPrompt ? 'adjustment-prompt-error' : undefined} rows={5} maxLength={10000} value={sourcePrompt} onChange={event => setSourcePrompts(previous => ({ ...previous, [inputKey]: event.target.value }))}/><span>直接編集できます。変更指示を空欄にすると、この設定で生成します。</span></label>
              {blankEditedPrompt && <p id="adjustment-prompt-error" className="m2-job-error">{settingLabel}を入力するか、元の設定に戻してください。</p>}
              <button type="button" className="button button-light adjustment-restore-prompt" disabled={sendingBlocked || !(inputKey in sourcePrompts)} onClick={() => {
                setSourcePrompts(previous => { const next = { ...previous }; delete next[inputKey]; return next })
                setSubmittedPrompts(previous => { const next = { ...previous }; delete next[inputKey]; return next })
              }}>元の設定に戻す</button>
              {sourcePrompt !== originalPrompt && <div className="adjustment-prompt-change" role="status">{settingLabel}を編集しています。元の設定は上で比較できます。</div>}
              <label>{kind === 'image' ? '今回だけの変更指示（構図・ポーズなど／任意）' : '今回だけの変更指示（任意）'}<textarea aria-label="今回だけの変更指示" rows={3} maxLength={10000} value={instruction} placeholder={kind === 'image' ? '例：今の服装で、手を自然に下ろした立ち姿にする' : '例：落ち着いた声で、少しゆっくり話す'} onChange={event => setPrompts(previous => ({ ...previous, [inputKey]: event.target.value }))}/></label>
              <p>{kind === 'image' ? '外見の設定をSTEP3と同じ処理でAnima用プロンプトに変換します。生成後、候補から実際に使ったプロンプトを確認できます。' : '生成後、候補から実際に使った声・話し方の指示を確認できます。'}編集した設定はこの画面に保持し、変更指示は生成ごとに空欄へ戻します。</p>
              {kind === 'voice' && <label>音声に含まれる読み上げ文<textarea aria-label="音声に含まれる読み上げ文" rows={3} maxLength={2000} value={transcript} placeholder="読み上げる文章を入力してください" onChange={event => setTranscripts(previous => ({ ...previous, [character.character_id]: event.target.value }))}/><span>自己紹介台詞を初期入力しています。必要に応じて編集してください。アップロード時は、音声の内容と一致する文章にしてください。</span></label>}
              <button type="button" className="button button-primary" disabled={sendingBlocked || blankEditedPrompt || (!sourcePrompt.trim() && !instruction.trim()) || (kind === 'voice' && !transcript.trim())} onClick={() => void action('generate')}><Icon name="spark" size={15}/>この設定で候補を生成</button>
              <div className="adjustment-upload"><label>ファイルから候補を取り込む<input key={`${inputKey}:${Boolean(file)}`} type="file" accept={kind === 'image' ? '.png,.webp,.jpg,.jpeg,image/png,image/webp,image/jpeg' : '.wav,.mp3,audio/wav,audio/mpeg'} onChange={event => { const selectedFile = event.target.files?.[0]; if (selectedFile) { const problem = adjustmentUploadError(selectedFile, kind, limits!.upload_bytes); if (problem) { setError(problem); return }; setError(''); setFiles(previous => ({ ...previous, [inputKey]: selectedFile })) } }}/></label>
                <p>{kind === 'image' ? `PNG・WebP・JPEG。長辺${limits?.image_max_side ?? 4096}pxまで。透過がない画像は背景も表示されます。` : `WAV・MP3。${limits?.audio_min_seconds ?? 0.25}〜${limits?.audio_max_seconds ?? 30}秒。`}{Math.floor((limits?.upload_bytes ?? 33554432) / 1024 / 1024)}MiB以下。</p>
                {file && <p>{file.name}</p>}<button type="button" className="button button-light" disabled={sendingBlocked || !file || (kind === 'voice' && !transcript.trim())} onClick={() => void action('upload')}>ファイルを候補に保存</button>
              </div>
            </fieldset>
            </div>
          </section>}
          <div className="adjustment-save"><div><strong>{dirty ? '未保存の調整があります' : '下書きの選択・配置は保存済みです'}</strong>{formPending && <p>未生成・未取り込みの入力があります。候補を生成・保存するか、入力を取り消してから反映してください。</p>}{musicErrors.length > 0 && <p role="alert">BGMの継続元と再生する候補を確認してください。{musicErrors.length}場面の設定が未完成です。</p>}<p>表示と画像は登場する全章へ反映します。声を変更した場合は、その人物の台詞音声も更新します。BGM・場面の切り替えは編集した章へ反映します。</p></div><button type="button" className="button button-light" disabled={sendingBlocked || !dirty || previewing || Boolean(previewError) || musicErrors.length > 0} onClick={() => void action('save')}>下書きを保存</button><button type="button" className="button button-primary" disabled={sendingBlocked || dirty || formPending || draft.status === 'failed' || musicErrors.length > 0} onClick={() => void action('apply')}>調整版を反映</button></div>
        </>}
        {(working.length > 0 || failed.length > 0 || latestMusicReplan?.status === 'completed' || draft?.status === 'applying' || draft?.error) && <section className="adjustment-job-status" aria-label="調整の処理状況"><h3>{draft?.status === 'applying' ? '調整版を反映中' : '素材・調整の処理状況'}</h3><p>処理中も元の公開版を鑑賞できます。</p>{draft?.error && <p className="m2-job-error" role="alert">{draft.error}</p>}<GenerationProgress items={adjustmentJobProgress(source)} label="素材調整の進捗" paused={disabled || source.readonly}/>{latestMusicReplan?.status === 'completed' && <p role="status">{latestMusicReplan.adoption_status === 'stale' ? `章の見直し結果は下書きへ反映していません。${latestMusicReplan.adoption_reason || '見直し中に設定が変更されたため、現在の手動設定を保持しました。'}` : '章全体の見直しが完了しました。下書きの設定を確認してから調整版へ反映してください。'}</p>}{draft?.status === 'applying' && !working.length && <p role="status">対象章を組み立てて公開版を切り替えています…</p>}<ul>{failed.map(job => <li key={job.id}><span>{job.purpose === 'music_replan' ? '章のBGM・場面転換の見直し' : source.candidates.find(candidate => candidate.job_id === job.id)?.kind === 'image' ? '立ち絵候補' : source.candidates.find(candidate => candidate.job_id === job.id)?.kind === 'voice' ? '基準音声候補' : source.music_candidates?.some(candidate => candidate.job_id === job.id) || ['m3_music', 'music', 'm3_music_plan'].includes(job.kind) ? 'BGM候補' : '台詞・試聴音声'}</span><strong>再試行が必要</strong>{job.error && <p className="m2-job-error">{job.error}</p>}</li>)}</ul>{(failed.length > 0 || draft?.status === 'failed') && <button type="button" className="button button-light" disabled={mutating || source.busy || disabled || source.readonly || state.conflict || dirty} onClick={() => void action('retry')}><Icon name="refresh" size={15}/>失敗した処理を再試行</button>}</section>}
      </>}
      {notice && <p className="adjustment-saved" role="status">{notice}</p>}
    </div>
    {enlarged && <Dialog title={enlarged.name} onClose={() => setEnlarged(null)} wide className="adjustment-image-dialog"><img src={enlarged.url} alt={enlarged.name}/></Dialog>}
    {promptCandidate?.prompt_details && <Dialog title="設定と実際の生成指示" onClose={() => setPromptCandidateId(null)} wide className="adjustment-prompt-dialog"><CandidatePromptDetails kind={promptCandidate.kind} details={promptCandidate.prompt_details}/></Dialog>}
  </section>
}

function musicActionLabel(action: 'play' | 'continue' | 'stop') {
  return action === 'play' ? 'この場面から曲を再生' : action === 'continue' ? '前の場面から曲を継続' : 'BGMなし'
}

function CandidatePromptDetails({ details, kind }: { details: NonNullable<AdjustmentCandidate['prompt_details']>; kind: AdjustmentKind }) {
  const settingLabel = kind === 'image' ? '外見の設定' : '声・話し方の設定'
  const actualLabel = kind === 'image' ? 'Animaに渡した実際のプロンプト' : '音声生成に渡した実際の指示'
  return <div className="adjustment-prompt-details">
    <div className="adjustment-prompt-comparison"><div><h4>元の{settingLabel}</h4><p>{details.source || '記録なし'}</p></div><div><h4>今回の{settingLabel}</h4><p>{details.input || '記録なし'}</p></div></div>
    {details.instruction && <div className="adjustment-prompt-instruction"><h4>今回の変更指示</h4><p>{details.instruction}</p></div>}
    <div className="adjustment-prompt-comparison">
      {details.baseline && <div><h4>元の素材：{actualLabel}</h4><p>{details.baseline}</p></div>}
      <div><h4>今回：{actualLabel}</h4><p>{details.effective ?? '生成が完了すると表示します。'}</p></div>
    </div>
    {!details.baseline && <p>元の素材に使った実際の生成指示は記録されていません。保存済みの{settingLabel}は上に表示しています。</p>}
  </div>
}

function AdjustmentRange({ label, name, value, min, max, step, suffix, onChange }: {
  label: string; name: string; value: number; min: number; max: number; step: number; suffix: string; onChange: (value: number) => void
}) {
  function change(value: number) { if (Number.isFinite(value)) onChange(Math.min(max, Math.max(min, value))) }
  return <div className="adjustment-range"><label htmlFor={`adjustment-${name}`}>{label}</label><input id={`adjustment-${name}`} type="range" min={min} max={max} step={step} value={value} onChange={event => change(event.target.valueAsNumber)}/><label className="adjustment-number"><span className="sr-only">{label}の数値</span><input type="number" min={min} max={max} step={step} value={value} onChange={event => { if (event.target.value !== '') change(event.target.valueAsNumber) }}/><span>{suffix}</span></label></div>
}
