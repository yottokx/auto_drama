import { useEffect, useId, useRef, useState } from 'react'
import { ApiError, errorMessage, request } from './api'
import { Icon } from './Icons'
import { Dialog } from './PreviewComponents'
import {
  canApprovePlanning, cancelPlanningTab, initialPlanningEditor, planningAction, planningActivity, planningDirty, planningTabDirty, receivePlanning,
  type PlanningContent, type PlanningEditorState, type PlanningResponse, type PlanningTab, type RevisionTarget,
} from './planningState'
import { PlanningContentEditor } from './PlanningContentEditor'
import { PlanningContentView } from './PlanningContentView'
import { GenerationProgress } from './GenerationProgress'
import { llmProgressItems } from './productionState'
import type { CharacterBrief, WizardStep } from './wizardState'
import './planning.css'

export function PlanningReviewPanel({ projectId, mainCharacters, mainApprovalId, setupApproved, serverStep, onRefresh, onReloadSetup, onPendingChange, onApproved, onBack, onProduction }: {
  projectId: string; mainCharacters: CharacterBrief[]; onPendingChange: (pending: boolean) => void
  mainApprovalId: string | null; setupApproved: boolean; serverStep: WizardStep
  onRefresh: () => Promise<void>; onReloadSetup: () => void
  onApproved: () => Promise<void>; onBack: () => void; onProduction: () => void
}) {
  const [state, setState] = useState(initialPlanningEditor)
  const [loaded, setLoaded] = useState(false)
  const [legacy, setLegacy] = useState(false)
  const [error, setError] = useState('')
  const [connectionError, setConnectionError] = useState('')
  const [notice, setNotice] = useState('')
  const [mutating, setMutating] = useState(false)
  const [target, setTarget] = useState<RevisionTarget>('all')
  const [characterId, setCharacterId] = useState('')
  const [chapter, setChapter] = useState('')
  const [approveOpen, setApproveOpen] = useState(false)
  const [discardOpen, setDiscardOpen] = useState(false)
  const [discardSetup, setDiscardSetup] = useState(false)
  const [tab, setTab] = useState<PlanningTab>('plot')
  const [editBaselines, setEditBaselines] = useState<Partial<Record<PlanningTab, PlanningContent>>>({})
  const [revisionOpen, setRevisionOpen] = useState(false)
  const tabButtons = useRef<Partial<Record<PlanningTab, HTMLButtonElement | null>>>({})
  const tabsAnchor = useRef<HTMLDivElement | null>(null)
  const tabScrollRequested = useRef(false)
  const panelId = useId()
  const setupSource = useRef({ approvalId: mainApprovalId, characters: mainCharacters })
  const current = useRef(state)
  const mounted = useRef(true)
  const mutation = useRef(false)
  const sequence = useRef(0)
  const controllers = useRef(new Set<AbortController>())
  const pendingCallback = useRef(onPendingChange)
  pendingCallback.current = onPendingChange
  const endpoint = `/api/planning/projects/${encodeURIComponent(projectId)}`

  function commit(update: (previous: PlanningEditorState) => PlanningEditorState) {
    const next = update(current.current)
    current.current = next
    setState(next)
    pendingCallback.current(planningDirty(next) || Boolean(next.instruction.trim()) || mutation.current)
  }
  function accept(response: PlanningResponse, adopted = false, clearInstruction = false) {
    if (response.project_id !== projectId) throw new Error('作品の全体計画を確認できませんでした。再読み込みしてください。')
    setLoaded(true); setLegacy(response.legacy_production)
    const previous = current.current
    const next = adopted ? {
      latest: response.planning, base: response.planning, content: response.planning?.content ?? null,
      instruction: clearInstruction ? '' : previous.instruction, conflict: false,
    } : receivePlanning(previous, response.planning)
    if (adopted || next.base?.id !== previous.base?.id || next.base?.revision !== previous.base?.revision) setEditBaselines({})
    commit(() => next)
  }

  useEffect(() => {
    mounted.current = true
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      const controller = new AbortController()
      controllers.current.add(controller)
      const version = sequence.current
      try {
        const response = await request<PlanningResponse>(endpoint, controller.signal)
        if (!stopped && !controller.signal.aborted && !mutation.current && version === sequence.current) { accept(response); setConnectionError('') }
      } catch (reason) {
        if (!stopped && !controller.signal.aborted && version === sequence.current) setConnectionError(errorMessage(reason))
      } finally {
        controllers.current.delete(controller)
        if (!stopped) timer = setTimeout(poll, 2000)
      }
    }
    void poll()
    return () => { stopped = true; mounted.current = false; clearTimeout(timer); controllers.current.forEach(controller => controller.abort()); pendingCallback.current(false) }
  }, [endpoint])

  const dirty = planningDirty(state)
  const pending = dirty || Boolean(state.instruction.trim()) || mutating
  useEffect(() => { onPendingChange(pending) }, [pending, onPendingChange])
  const planning = state.latest
  const setupInvalidated = !setupApproved || setupSource.current.approvalId !== mainApprovalId
  const remoteStepChanged = serverStep !== 'planning-review'
  const { generating, failed } = planningActivity(planning)
  const activePlanningJob = planning?.jobs.find(job => job.id === planning.active_job_id)
  const busy = mutating || Boolean(generating)
  const canApprove = canApprovePlanning(state, mutating || setupInvalidated)
  const selectedCharacter = state.content?.cast_plan.supporting_characters.find(character => character.id === characterId)?.id
    ?? state.content?.cast_plan.supporting_characters[0]?.id
  const editing = Boolean(editBaselines[tab])

  useEffect(() => {
    if (!tabScrollRequested.current) return
    tabScrollRequested.current = false
    tabsAnchor.current?.scrollIntoView({ block: 'start', behavior: 'instant' })
  }, [tab])

  function selectTab(next: PlanningTab) {
    if (next === tab) return
    tabScrollRequested.current = true
    setTab(next)
  }

  function beginEditing() {
    if (busy || state.conflict || setupInvalidated || !state.content) return
    setEditBaselines(previous => ({ ...previous, [tab]: state.content! }))
    setNotice('')
  }
  function cancelEditing() {
    const baseline = editBaselines[tab]
    if (busy || !baseline) return
    commit(previous => cancelPlanningTab(previous, tab, baseline))
    setEditBaselines(previous => { const next = { ...previous }; delete next[tab]; return next })
    setNotice(`${tab === 'plot' ? 'プロット' : 'サブキャラ'}の直接編集を取り消しました。`)
  }

  async function mutate(action: 'generate' | 'save' | 'revise' | 'approve', retryId?: string) {
    if (mutation.current || generating || state.conflict || setupInvalidated) return
    const controller = new AbortController()
    controllers.current.add(controller)
    mutation.current = true; sequence.current += 1; setMutating(true); setError(''); setNotice('')
    try {
      let result: PlanningResponse
      if (retryId) {
        await request(`/api/jobs/${encodeURIComponent(retryId)}/retry`, controller.signal, {})
        result = await request<PlanningResponse>(endpoint, controller.signal)
      } else {
        const body = planningAction(current.current, action, { target, characterId: selectedCharacter, ...(chapter ? { chapterNumber: Number(chapter) } : {}) })
        result = await request<PlanningResponse>(`${endpoint}/actions`, controller.signal, body)
      }
      if (!mounted.current || controller.signal.aborted) return
      accept(result, true, action === 'revise')
      if (action === 'revise' && !retryId) setRevisionOpen(false)
      setNotice(retryId ? '全体計画の生成を再試行します。' : action === 'save' ? '全体計画の変更を保存しました。内容を確認して承認してください。'
        : action === 'approve' ? '全体計画を承認しました。第1章から本編を制作します。' : '全体計画の生成を受け付けました。')
      if (action === 'approve' && !retryId) { setApproveOpen(false); await onApproved() }
      else await onRefresh()
    } catch (reason) {
      if (mounted.current && !controller.signal.aborted) {
        if (reason instanceof ApiError && reason.status === 409) commit(previous => ({ ...previous, conflict: true }))
        setError(`${errorMessage(reason)} 入力は保持しています。`)
      }
    } finally {
      controllers.current.delete(controller); mutation.current = false; sequence.current += 1
      if (mounted.current) setMutating(false)
    }
  }

  async function reload() {
    if (mutation.current) return
    if (discardSetup) { onReloadSetup(); return }
    const controller = new AbortController()
    controllers.current.add(controller)
    mutation.current = true; sequence.current += 1; setMutating(true)
    try {
      const response = await request<PlanningResponse>(endpoint, controller.signal)
      if (!mounted.current || controller.signal.aborted) return
      accept(response, true, true); setDiscardOpen(false); setError(''); setConnectionError('')
      setNotice('最新の保存済み計画を表示しました。')
    } catch (reason) {
      if (mounted.current && !controller.signal.aborted) { setDiscardOpen(false); setError(`${errorMessage(reason)} 入力は保持しています。`) }
    } finally {
      controllers.current.delete(controller); mutation.current = false; sequence.current += 1
      if (mounted.current) setMutating(false)
    }
  }

  return <section className="planning-panel" aria-label="全体計画の確認" aria-busy={busy}>
    <div className="planning-intro"><Icon name="book" size={20}/><div><p>全体プロットとサブキャラを確認し、必要に応じて修正してください。承認後は第1章から本編を制作します。</p><small>サブキャラの立ち絵・基準音声は本編制作で生成します。物語の進行に応じて登場人物が追加されることもあります。</small></div></div>
    {!loaded && <p role="status">全体計画を読み込み中…</p>}
    {connectionError && <p className="m2-job-error" role="alert">{connectionError} 入力を保持して再接続を試みています。</p>}
    {error && <p className="m2-job-error" role="alert">{error}</p>}
    {(setupInvalidated || (remoteStepChanged && !mutating)) && <div className="planning-conflict" role="alert"><p>{setupInvalidated ? '別の画面で世界観・メインキャラの承認版が変更されました。編集中の全体計画と指示を保持しています。この計画への保存・承認を停止しました。必要な入力を控えてから、最新の設定を開いてください。' : '別の画面で制作ステップが変更されました。未保存の入力を保持するため、全体計画の編集画面を表示しています。'}</p><button className="button button-light" disabled={mutating} onClick={() => { setDiscardSetup(true); setDiscardOpen(true) }}>入力を取り消して最新の画面へ</button></div>}
    {legacy && !planning && <div className="planning-status"><p>この作品は以前の手順で本編の制作を開始しています。保存済みの章は本編の画面から確認できます。</p><button className="button button-primary" onClick={onProduction}>本編の制作・鑑賞へ<Icon name="arrow" size={15}/></button></div>}
    {loaded && !planning && !legacy && <button className="button button-primary" disabled={busy} onClick={() => void mutate('generate')}><Icon name="spark" size={16}/>全体計画を生成する</button>}
    {planning && <>
      {planning.status === 'draft' && !state.content && <button className="button button-primary" disabled={busy || pending || state.conflict || setupInvalidated} onClick={() => void mutate('generate')}>全体計画を生成する</button>}
      <div className="planning-status" role="status"><strong>{generating ? '全体プロットとサブキャラを生成・修正中…' : planning.status === 'failed' ? '全体計画の生成に失敗しました。' : planning.status === 'approved' ? 'この全体計画は承認済みです。' : !state.content ? '全体計画を生成してください。' : '全体計画の準備ができました。'}</strong><span>第{planning.revision}版</span>{generating && <p>サーバーとワーカーが起動していれば、画面を閉じても生成は続きます。</p>}</div>
      {activePlanningJob && (generating || planning.status === 'failed') && <GenerationProgress items={llmProgressItems(activePlanningJob)} label="全体計画の生成工程"/>}
      {planning.error && <p className="m2-job-error" role="alert">{planning.error}</p>}
      {planning.status === 'failed' && state.content && <p className="planning-notice" role="status">修正前の保存済み計画を表示しています。内容を確認してこの計画を承認するか、生成を再試行できます。</p>}
      {failed.map(job => <div className="planning-job-error" key={job.id}><p>{job.error || '生成処理を完了できませんでした。'}</p><button className="button button-light" disabled={busy || pending || state.conflict || setupInvalidated} onClick={() => void mutate('generate', job.id)}>この生成を再試行</button></div>)}
      {state.conflict && !setupInvalidated && <div className="planning-conflict" role="alert"><p>保存済みの全体計画が更新されています。編集中の内容と指示を保持しています。最新の計画を読み直してから、変更を反映してください。</p><button className="button button-light" disabled={busy} onClick={() => { setDiscardSetup(false); setDiscardOpen(true) }}>最新の計画を読み込む</button></div>}
      {state.content && <>
        <div className="planning-tabs-anchor" ref={tabsAnchor} aria-hidden="true"/>
        <div className="planning-tabs" role="tablist" aria-label="全体計画の表示内容">{(['plot', 'cast'] as const).map(item => <button key={item} ref={element => { tabButtons.current[item] = element }} type="button" className="planning-tab" role="tab" id={`${panelId}-tab-${item}`} aria-selected={tab === item} aria-controls={`${panelId}-panel-${item}`} tabIndex={tab === item ? 0 : -1} onClick={() => selectTab(item)} onKeyDown={event => {
          let next: PlanningTab | undefined
          if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') next = item === 'plot' ? 'cast' : 'plot'
          if (event.key === 'Home') next = 'plot'
          if (event.key === 'End') next = 'cast'
          if (next) { event.preventDefault(); selectTab(next); tabButtons.current[next]?.focus({ preventScroll: true }) }
        }}>{item === 'plot' ? 'プロット' : 'サブキャラ'}{planningTabDirty(state, item) && <span className="planning-tab-unsaved">（未保存）</span>}</button>)}</div>
        {(['plot', 'cast'] as const).map(item => <section key={item} className="planning-tabpanel" role="tabpanel" id={`${panelId}-panel-${item}`} aria-labelledby={`${panelId}-tab-${item}`} hidden={tab !== item} tabIndex={0}>{tab === item && <>
          <div className="planning-toolbar"><p>{editing ? `${item === 'plot' ? 'プロット' : 'サブキャラ'}を直接編集中です。タブを切り替えても入力は保持されます。` : '内容を確認し、変更したいときは直接編集できます。'}</p>{!editing && <button className="button button-light" disabled={busy || state.conflict || setupInvalidated} onClick={beginEditing}>直接編集</button>}</div>
          {editing ? <PlanningContentEditor content={state.content!} mainCharacters={setupSource.current.characters} tab={item} disabled={busy || state.conflict || setupInvalidated} onChange={content => { commit(previous => ({ ...previous, content })); setNotice('') }}/>
            : <PlanningContentView content={state.content!} mainCharacters={setupSource.current.characters} tab={item}/>}
          {editing && <div className="planning-save"><p>{dirty ? '未保存の変更があります。保存すると、両方のタブで編集した内容をまとめて記録します。' : '保存したあとは、内容を読みやすい表示に戻します。'}</p><div><button className="button button-light" disabled={busy} onClick={cancelEditing}>編集を取り消す</button><button className="button button-primary" disabled={!dirty || busy || state.conflict || setupInvalidated} onClick={() => void mutate('save')}><Icon name="check" size={15}/>変更を保存する</button></div></div>}
        </>}</section>)}
        <section className="planning-revision" aria-label="AIへの修正指示"><div className="planning-revision-toggle"><button className="button button-light" aria-expanded={revisionOpen} aria-controls={`${panelId}-revision`} onClick={() => setRevisionOpen(previous => !previous)}><Icon name="spark" size={15}/>{revisionOpen ? '修正指示を閉じる' : 'AIに修正を指示'}</button>{!revisionOpen && state.instruction.trim() && <span>送信前の指示を保持しています。</span>}</div><div id={`${panelId}-revision`} hidden={!revisionOpen}>{revisionOpen && <><h3>AIに修正を指示する</h3><fieldset disabled={busy || state.conflict || setupInvalidated}>
          <div className="planning-targets"><label>修正する範囲<select value={target} onChange={event => setTarget(event.target.value as RevisionTarget)}><option value="all">全体プロットとサブキャラ</option><option value="plot">プロット</option><option value="character">サブキャラ1人</option><option value="relationships">人物の関係性</option></select></label>
            {target === 'character' && <label>サブキャラ<select value={selectedCharacter ?? ''} onChange={event => setCharacterId(event.target.value)}>{!state.content.cast_plan.supporting_characters.length && <option value="">サブキャラがいません</option>}{state.content.cast_plan.supporting_characters.map(character => <option key={character.id} value={character.id}>{character.name}</option>)}</select></label>}
            {target === 'plot' && <label>対象の章<select value={chapter} onChange={event => setChapter(event.target.value)}><option value="">全章</option>{state.content.plot.chapters.map(item => <option key={item.number} value={item.number}>第{item.number}章</option>)}</select></label>}
          </div>
          <label className="field-label">修正指示<textarea rows={4} maxLength={30000} value={state.instruction} onChange={event => commit(previous => ({ ...previous, instruction: event.target.value }))} placeholder="例：第2章で二人が協力するきっかけを、もう少し自然な出来事にしてください。"/></label>
          <div className="planning-actions"><button className="button button-light" disabled={!state.instruction} onClick={() => commit(previous => ({ ...previous, instruction: '' }))}>指示を取り消す</button><button className="button button-primary" disabled={!state.instruction.trim() || dirty || (target === 'character' && !selectedCharacter)} onClick={() => void mutate('revise')}><Icon name="spark" size={15}/>指示して修正する</button></div>
        </fieldset></>}</div></section>
      </>}
      {notice && <p className="planning-notice" role="status">{notice}</p>}
      <div className="planning-footer"><button className="button button-light" disabled={pending} onClick={onBack}><Icon name="back" size={16}/>メインキャラの確認に戻る</button>{planning.status === 'approved' ? <button className="button button-primary" disabled={pending} onClick={onProduction}>本編の制作・鑑賞へ<Icon name="arrow" size={16}/></button> : <button className="button button-primary" disabled={!canApprove} onClick={() => setApproveOpen(true)}><Icon name="check" size={16}/>全体計画を承認して本編を制作</button>}</div>
    </>}
    {approveOpen && <Dialog title="全体計画を承認して、本編の制作へ。" onClose={() => { if (!mutating) setApproveOpen(false) }}><p className="dialog-intro">表示中の全体プロットとサブキャラの設定・関係性を、物語全体の計画として保存します。第1章から順番に本文・背景・立ち絵・音声を制作し、完成した章から鑑賞できます。</p><p className="dialog-info">本編制作では、計画と前章の内容を踏まえて場面を具体化します。追加の章ごとの承認はありません。</p><div className="dialog-actions"><button className="button button-light" disabled={mutating} onClick={() => setApproveOpen(false)}>確認に戻る</button><button className="button button-primary" disabled={!canApprove} onClick={() => void mutate('approve')}>{mutating ? '制作を開始中…' : '承認して第1章から制作する'}</button></div></Dialog>}
    {discardOpen && <Dialog title="編集中の内容を取り消しますか。" onClose={() => { if (!mutating) setDiscardOpen(false) }}><p className="dialog-intro">未保存の直接編集とAIへの修正指示を取り消し、{discardSetup ? '最新の世界観・メインキャラの承認状態に対応した画面を開きます。' : '最新の保存済み計画を表示します。'}</p><div className="dialog-actions"><button className="button button-light" disabled={mutating} onClick={() => setDiscardOpen(false)}>入力を残す</button><button className="button button-primary" disabled={mutating} onClick={() => void reload()}>{mutating ? '読み込み中…' : '取り消して読み込む'}</button></div></Dialog>}
  </section>
}
