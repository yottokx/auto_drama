import { useEffect, useRef, useState } from 'react'
import { Icon, type IconName } from './Icons'
import { Dialog } from './PreviewComponents'
import { WorldReview } from './WorldSteps'
import { CharacterReview } from './CharacterSteps'
import { CombinedBriefInput } from './CombinedBriefInput'
import { activeJob, useM2Api, type M2Detail } from './m2Api'
import { createCharacterBrief, createInitialWizardDraft, type CharacterBrief, type WizardStep } from './wizardState'
import { RelationshipReview, relationshipPairs } from './RelationshipSteps'
import { ProductionPanel } from './ProductionPanel'
import { ChangeHistory } from './ChangeHistory'
import './wizard.css'
import './m2.css'

const steps: { id: WizardStep; title: string; heading: string; description: string; icon: IconName }[] = [
  { id: 'world-input', title: '世界観・メインキャラを指示', heading: 'どんな世界で、誰の物語をはじめますか。', description: '世界観とメインキャラクターの希望をまとめて伝え、世界観から順番に確定します。', icon: 'globe' },
  { id: 'world-review', title: '世界観を確認・確定', heading: 'この物語の世界観。', description: '生成された世界観を読んで、イメージに合っていれば確定します。', icon: 'book' },
  { id: 'character-review', title: 'メインキャラを確認・確定', heading: 'この物語を生きる、登場人物。', description: '確定した世界観に合わせて生成した設定・立ち絵・声を確認します。気になるところだけ、結果の下から修正してください。', icon: 'spark' },
  { id: 'production', title: '本編を制作・鑑賞', heading: 'あなたの物語を、楽しむ。', description: '本編の制作状況を確認し、完成した章を鑑賞・書き出しできます。', icon: 'film' },
]
type Api = ReturnType<typeof useM2Api>
const jobNames: Record<string, string> = { m2_world: '世界観', m2_character: '人物設定', m2_image: '立ち絵', m2_voice: 'サンプル音声', m2_voice_clone: '台詞の音声', m2_relationships: '関係性' }
const statusNames = { pending: '実行待ち', running: '生成中', completed: '完了', failed: '失敗' }
const stepDescriptions = ['まとめて伝える', '世界観を確定する', 'キャストを確定する', '完成した章を楽しむ']
const isStepLocked = (step: WizardStep, draft: M2Detail['draft']) =>
  step === 'production' ? !draft.approved && !draft.hasProduction : step === 'character-review' && !draft.worldConfirmed

export default function LiveWizardApp() {
  const api = useM2Api()
  const [library, setLibrary] = useState(!api.selectedId)
  const [pending, setPending] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [notice, setNotice] = useState('')
  const draft = api.detail?.draft
  const busy = api.mutating || Boolean(api.detail?.jobs.some(job => job.kind.startsWith('m2_') && activeJob(job)))
  const current = steps.find(step => step.id === draft?.step) ?? steps[0]
  const title = draft?.worldResult?.title || draft?.worldInput.title || 'タイトル未定の物語'
  useEffect(() => {
    if (!pending) return
    const beforeUnload = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', beforeUnload)
    return () => window.removeEventListener('beforeunload', beforeUnload)
  }, [pending])
  useEffect(() => { if (!notice) return; const timer = setTimeout(() => setNotice(''), 6500); return () => clearTimeout(timer) }, [notice])
  function canLeave() {
    if (pending) { setNotice('未保存の変更があります。保存するか、キャンセルしてから移動してください。'); return false }
    return true
  }
  async function navigate(step: WizardStep) {
    if (!draft || (step !== draft.step && !canLeave())) return
    if (step !== draft.step && !await api.action('go-to', { step })) return
    setLibrary(false); setMenuOpen(false); window.scrollTo({ top: 0 })
  }
  function showLibrary() { if (canLeave()) { setLibrary(true); setMenuOpen(false); void api.refresh() } }
  async function createProject() {
    if (!canLeave()) return
    if (await api.create()) { setLibrary(false); setMenuOpen(false); window.scrollTo({ top: 0 }) }
  }
  function selectProject(id: string) { if (canLeave()) { api.select(id); setLibrary(false); setMenuOpen(false); window.scrollTo({ top: 0 }) } }
  return <div className="studio wizard-studio">
    <a className="skip-link" href="#main">メインコンテンツへ</a>
    {menuOpen && <button className="nav-scrim" aria-label="ナビゲーションを閉じる" onClick={() => setMenuOpen(false)}/>}
    <aside id="studio-navigation" className={`studio-sidebar ${menuOpen ? 'sidebar-open' : ''}`} aria-label="メインナビゲーション">
      <button className="studio-brand" onClick={showLibrary}><span className="brand-symbol"><Icon name="spark" size={26}/></span><span>AIオートドラマ<small>YOUR STORY, YOUR WAY.</small></span></button>
      <button className="new-project" disabled={api.mutating} onClick={() => void createProject()}><Icon name="plus" size={17}/>新しい物語をつくる</button>
      <nav className="primary-nav"><button className={library ? 'nav-active' : ''} onClick={showLibrary}><Icon name="grid"/>作品ライブラリ</button></nav>
      <div className="nav-divider"/><p className="nav-caption">物語をつくる · 4 STEPS</p>
      <div className="current-project"><span className="project-initial"><Icon name="book" size={16}/></span><span>{api.selectedId ? title : '作品を選択してください'}<small>{draft?.approved ? '本編制作・鑑賞' : '下書き'}</small></span></div>
      <nav className="workspace-nav" aria-label="制作ステップ">{steps.map((step, index) => <button key={step.id} className={!library && current.id === step.id ? 'nav-active' : ''} disabled={!draft || busy || isStepLocked(step.id, draft)} onClick={() => void navigate(step.id)} aria-current={!library && current.id === step.id ? 'step' : undefined}><Icon name={step.icon} size={17}/>{step.title}<span className="nav-step">{index + 1}</span></button>)}</nav>
      <div className="sidebar-bottom"><div className="sidebar-note"><span className="tiny-spark">✧</span><p>決めたいところは、自由に。<br/>まだ決めないところは、おまかせに。</p></div><a className="m2-sidebar-link" href="/?demo=results" onClick={event => { if (!canLeave()) event.preventDefault() }}><Icon name="help" size={16}/>表示サンプルを見る</a><a className="m2-sidebar-link" href="/?view=m1" onClick={event => { if (!canLeave()) event.preventDefault() }}><Icon name="settings" size={16}/>M1の保存・書き出し</a><div className="local-profile"><span>AD</span><div>ローカルワークスペース<small><i/>M4 · 物語の制作</small></div></div></div>
    </aside>
    <div className="studio-body">
      <header className="topbar"><div className="breadcrumbs"><button className="mobile-menu icon-button" aria-label="ナビゲーションを開く" aria-controls="studio-navigation" aria-expanded={menuOpen} onClick={() => setMenuOpen(true)}><Icon name="grid"/></button><button onClick={showLibrary}>ワークスペース</button><Icon name="chevron" size={12}/><span>{library ? '作品ライブラリ' : title}</span></div><div className="topbar-right"><span className="preview-pill">M4</span><span className="save-status"><Icon name="check" size={13}/>{api.connectionError ? 'サーバー接続を確認中' : pending ? '未保存の変更あり' : api.mutating ? '保存中…' : 'サーバーに保存'}</span></div></header>
      <main id="main" className="main-content">
        {(api.connectionError || api.actionError) && <div className="m2-error" role="alert"><div><strong>{api.connectionError ? '制御サーバーとの接続を確認してください' : '操作を完了できませんでした'}</strong><p>{api.actionError || api.connectionError}</p>{api.connectionError && <small>画面上の編集中の内容は保持しています。サーバー復旧後、自動で再接続します。</small>}</div><button className="button button-light" onClick={() => void api.refresh()}>状態を再取得</button></div>}
        {library || !api.selectedId ? <>
          <div className="page-heading"><div><div className="eyebrow">YOUR COLLECTION</div><h1>あなたの物語。</h1><p>保存した世界観とキャラクターから、つづきを。</p></div><button className="button button-primary" disabled={api.mutating} onClick={() => void createProject()}><Icon name="plus" size={16}/>新しい物語</button></div>
          {api.loading && <p className="m2-loading" role="status">作品を読み込み中…</p>}
          {!api.loading && !api.projects.length && <div className="world-paper m2-welcome"><Icon name="book" size={36}/><h2>最初の物語を、ここから。</h2><p>世界観と登場人物の希望を伝え、世界観から順番に確定していきましょう。</p><button className="button button-primary" disabled={api.mutating} onClick={() => void createProject()}>物語をつくる<Icon name="arrow" size={16}/></button></div>}
          <div className="m2-library">{api.projects.map(project => <button key={project.id} className="wizard-library-card" onClick={() => selectProject(project.id)}><span className="wizard-library-art"><Icon name="book" size={34}/></span><span><small>全{project.chapter_count}章 · {new Date(project.created_at).toLocaleDateString('ja-JP')}</small><strong>{project.title || 'タイトル未定の物語'}</strong><span>制作を再開<Icon name="arrow" size={16}/></span></span></button>)}</div>
        </> : api.detail ? <ProjectWorkspace key={api.selectedId} detail={api.detail} api={api} onPendingChange={setPending} onNavigate={navigate} onNotice={setNotice}/> : <div className="world-paper m2-loading" role="status">{api.loading ? '作品を読み込み中…' : '作品を取得できません。接続を確認して再取得するか、作品ライブラリから選び直してください。'}</div>}
        <footer className="studio-footer"><span>AI AUTO DRAMA <i/> YOUR STORY, YOUR WAY.</span><span>M4 · STORY PRODUCTION</span></footer>
      </main>
    </div>
    {notice && <div className="toast" role="status"><Icon name="check" size={16}/>{notice}<button className="icon-button" aria-label="通知を閉じる" onClick={() => setNotice('')}><Icon name="close" size={14}/></button></div>}
  </div>
}

function useInputDraft<T>(source: T) {
  const [value, setValue] = useState(source)
  const baseline = useRef(JSON.stringify(source))
  const latestValue = useRef(value); latestValue.current = value
  const sourceText = JSON.stringify(source)
  useEffect(() => {
    if (JSON.stringify(latestValue.current) === baseline.current) setValue(source)
    baseline.current = sourceText
  }, [sourceText]) // A poll never replaces locally edited input.
  return { value, setValue, dirty: JSON.stringify(value) !== sourceText, reset: () => setValue(source) }
}

function ProjectWorkspace({ detail, api, onPendingChange, onNavigate, onNotice }: { detail: M2Detail; api: Api; onPendingChange: (pending: boolean) => void; onNavigate: (step: WizardStep) => Promise<void>; onNotice: (text: string) => void }) {
  const { draft, jobs } = detail
  const worldForm = useInputDraft(draft.worldInput)
  const characterForm = useInputDraft(draft.characters.map(character => ({ ...character.input, locked: character.locked })))
  const relationshipForm = useInputDraft(relationshipPairs(draft.characters.map(character => character.input), draft.relationshipInputs ?? []))
  const relationshipValues = relationshipPairs(characterForm.value, relationshipForm.value)
  const [reviewPending, setReviewPending] = useState(false)
  const [productionPending, setProductionPending] = useState(false)
  const [approveOpen, setApproveOpen] = useState(false)
  const index = steps.findIndex(step => step.id === draft.step)
  const current = steps[index]
  const setupJobs = jobs.filter(job => job.kind.startsWith('m2_'))
  const generating = setupJobs.some(activeJob)
  const busy = api.mutating || jobs.some(activeJob)
  const navigationBusy = api.mutating || generating
  const pending = reviewPending || productionPending || worldForm.dirty || characterForm.dirty || relationshipForm.dirty
  useEffect(() => { onPendingChange(pending) }, [pending, onPendingChange])
  useEffect(() => () => onPendingChange(false), [onPendingChange])
  const world = draft.worldResult ?? { ...createInitialWizardDraft().world, prompt: draft.worldInput.prompt, notes: draft.worldInput.notes }
  const characters = draft.characters.map(character => ({ ...(character.result ?? { ...createCharacterBrief(character.id), name: character.input.name }), id: character.id, freeform: character.input.freeform, locked: character.locked }))
  const assets = Object.fromEntries(draft.characters.map(character => [character.id, { image: character.imageUrl, imageArtifactId: character.imageArtifactId, voice: character.voiceUrl, voiceArtifactId: character.voiceArtifactId, voiceReferenceText: character.voiceReferenceText, voiceTests: character.voiceTests }]))
  const characterStatus = Object.fromEntries(draft.characters.map(character => [character.id, !character.result ? '生成待ち' : character.imageUrl && character.voiceUrl ? '生成済み' : '設定生成済み']))
  const relationshipsComplete = characters.length < 2 || Boolean(draft.relationships?.artifactId && draft.relationships.result?.pairs.length === characters.length * (characters.length - 1) / 2 && !draft.relationships.pendingChanges)
  const complete = relationshipsComplete && draft.worldConfirmed && !draft.worldPendingChanges && draft.characters.every(character => character.result && character.imageUrl && character.voiceUrl && !character.pendingChanges && !character.imagePendingChanges && !character.voicePendingChanges) && !draft.activeJobId && !busy && !pending

  async function saveBrief() {
    const result = await api.action('save-brief', { world: worldForm.value, characters: characterForm.value, relationshipInputs: relationshipValues })
    if (!result) return false
    worldForm.setValue(result.draft.worldInput)
    characterForm.setValue(result.draft.characters.map(character => ({ ...character.input, locked: character.locked })))
    relationshipForm.setValue(relationshipPairs(result.draft.characters.map(character => character.input), result.draft.relationshipInputs ?? []))
    return true
  }
  async function saveRelationships() {
    if (!relationshipForm.dirty) return true
    const result = await api.action('save-relationships', { relationshipInputs: relationshipValues })
    if (!result) return false
    relationshipForm.setValue(relationshipPairs(result.draft.characters.map(character => character.input), result.draft.relationshipInputs ?? []))
    return true
  }
  async function generateRelationships() {
    if (await saveRelationships() && await api.action('generate-relationships')) onNotice('関係性の生成を受け付けました。')
  }
  async function generateWorld() {
    if (await saveBrief() && await api.action('generate-world')) { window.scrollTo({ top: 0 }); onNotice('世界観とキャラクターの指示を保存し、世界観の生成を受け付けました。') }
  }
  async function generateCharacters() {
    if (await api.action('generate-characters')) { window.scrollTo({ top: 0 }); onNotice('キャラクターの設定・立ち絵・音声を順番に生成します。') }
  }
  const changeCharacter = (id: string, patch: Partial<CharacterBrief>) => characterForm.setValue(current => current.map(character => character.id === id ? { ...character, ...patch } : character))
  return <>
    <div className="page-heading"><div><div className="eyebrow">CREATE YOUR STORY <span>/</span> 0{index + 1}</div><h1>{current.heading}</h1><p>{current.description}</p></div>{draft.step !== 'production' && <span className="wizard-draft-label">DRAFT {String(draft.revision).padStart(2, '0')}</span>}</div>
    <ol className="wizard-stepper" aria-label="制作のステップ">{steps.map((step, stepIndex) => <li key={step.id} className={step.id === draft.step ? 'current' : (stepIndex < 2 && draft.worldConfirmed) || (stepIndex === 2 && draft.approved) ? 'complete' : ''}><button disabled={navigationBusy || isStepLocked(step.id, draft)} onClick={() => void onNavigate(step.id)} aria-current={step.id === draft.step ? 'step' : undefined}><span>{String(stepIndex + 1).padStart(2, '0')}</span><div>{step.title}<small>{stepDescriptions[stepIndex]}</small></div></button></li>)}</ol>
    {draft.step === 'production' ? <ProductionPanel key={api.restoreEpoch} projectId={detail.project.id} approved={draft.approved} chapterCount={world.chapterCount} onPendingChange={setProductionPending}/> : <>
    {draft.worldPendingChanges && draft.worldResult && <div className="m2-notice"><p>世界観の入力に未反映の変更があります。最新の指示から世界観を生成してください。</p><button className="button button-light" disabled={busy || pending} onClick={() => void generateWorld()}>最新の入力で世界観を生成</button></div>}
    {draft.step === 'character-review' && !generating && draft.characters.some(character => character.pendingChanges) && <div className="m2-notice"><p>人物の入力に未反映の変更があります。生成してから結果を確認してください。</p><button className="button button-light" disabled={busy || pending} onClick={() => void generateCharacters()}>最新の入力でキャラクターを生成</button></div>}
    <div className={`wizard-workspace ${index < 2 ? 'world-stage-layout' : ''}`}>
      <section className="wizard-stage" aria-label={current.title}>
        <GenerationActivity jobs={setupJobs} api={api} activeJobId={draft.activeJobId} remainingJobCount={draft.remainingJobCount} pending={pending}/>
        <fieldset className="m2-stage-fieldset" disabled={busy && draft.step !== 'character-review'} aria-busy={busy}>
          {draft.step === 'world-input' && <CombinedBriefInput live world={worldForm.value} characters={characterForm.value} relationshipInputs={relationshipValues} onWorldChange={patch => worldForm.setValue(current => ({ ...current, ...patch }))} onCharacterChange={changeCharacter} onRelationshipsChange={relationshipForm.setValue} onAdd={() => { if (characterForm.value.length >= 3) return characterForm.value[2].id; const id = `character-${crypto.randomUUID()}`; characterForm.setValue(current => [...current, createCharacterBrief(id)]); return id }} onRemove={id => characterForm.setValue(current => current.filter(character => character.id !== id))} onSave={() => { void saveBrief().then(saved => { if (saved) onNotice('世界観とキャラクターの指示を保存しました。') }) }} onNext={() => void generateWorld()} hasResults={Boolean(draft.worldResult)}/>}
          {draft.step === 'world-review' && <WorldReview live worldConfirmed={draft.worldConfirmed} generating={generating} world={world} sourceWorld={draft.worldInput} requests={draft.requests} onChange={async patch => Boolean(await api.action('edit-world', { world: { ...world, ...patch } }))} onRequest={async instruction => Boolean(await api.action('generate-world', { instruction }))} onBack={() => void onNavigate('world-input')} onConfirm={() => { void api.action('confirm-world').then(result => { if (result) { window.scrollTo({ top: 0 }); onNotice(draft.worldConfirmed ? 'キャラクターの確認画面を開きました。' : '世界観を確定しました。キャラクターの設定・立ち絵・音声を順番に生成します。') } }) }} onPendingChange={setReviewPending} canConfirm={Boolean(draft.worldResult) && !draft.worldPendingChanges && !busy}/>}

          {draft.step === 'character-review' && <CharacterReview live generating={generating} busy={busy} characterStatus={characterStatus} relationshipResults={<RelationshipReview characters={characters} values={relationshipValues} relationships={draft.relationships} busy={busy || reviewPending} dirty={relationshipForm.dirty} onChange={relationshipForm.setValue} onDiscard={relationshipForm.reset} onSave={() => void saveRelationships()} onGenerate={() => void generateRelationships()}/>} onCloneVoice={async (character_id, text) => Boolean(await api.action('clone-voice', { character_id, text }))} characters={characters} previewAssets={assets} requests={draft.requests} onChange={async (character_id, patch) => Boolean(await api.action('edit-character', { character_id, patch }))} onToggleLock={(character_id, scope) => { void api.action('toggle-lock', { character_id, scope }) }} onRequest={async (scope, instruction, character_id) => Boolean(await api.action(scope === 'image-retake' || scope === 'voice-retake' ? 'retake' : 'revise-character', { scope, instruction, character_id }))} onBack={() => void onNavigate('world-review')} onApprove={() => setApproveOpen(true)} onPendingChange={setReviewPending} canApprove={complete}/>}
        </fieldset>
        {draft.step === 'world-input' && (worldForm.dirty || characterForm.dirty || relationshipForm.dirty) && <div className="m2-unsaved"><p>入力を編集中です。「指示を保存」または世界観の生成で、すべての指示を一緒に保存します。</p><button className="button button-light" disabled={busy} onClick={() => { worldForm.reset(); characterForm.reset(); relationshipForm.reset() }}>変更を取り消す</button></div>}
      </section>
      <aside className="wizard-summary" aria-label="作品の概要"><p className="eyebrow">STORY NOTE</p><h2>{world.title || draft.worldInput.title || 'タイトル未定の物語'}</h2><dl><div><dt>世界観</dt><dd className={draft.worldConfirmed ? 'confirmed-text' : ''}>{draft.worldConfirmed ? '確定済み' : '調整中'}</dd></div><div><dt>ジャンル</dt><dd>{world.genre || '生成前'}</dd></div><div><dt>雰囲気</dt><dd>{world.mood || '生成前'}</dd></div><div><dt>構成</dt><dd>全{world.chapterCount}章</dd></div></dl><div className="wizard-summary-divider"/><h3>メインキャラクター <span>{characters.length}</span></h3><ul>{characters.map((character, i) => <li key={character.id}><span>{String(i + 1).padStart(2, '0')}</span><div>{character.name || `キャラクター ${i + 1}`}<small>{character.role || '生成前'}</small></div></li>)}</ul><div className="wizard-summary-tip"><Icon name="leaf" size={17}/><p>気になるところだけ、自由に。<small>固定した設定・素材は再生成から保護されます。</small></p></div><span className="wizard-request-count">修正・リテイク指示：{draft.requests.length}件</span>{index >= 2 && <button className="wizard-world-back" disabled={busy} onClick={() => void onNavigate('world-review')}><Icon name="back" size={14}/>確定した世界観を見直す</button>}</aside>
    </div>
    </>}
    {(setupJobs.length > 0 || draft.approved) && <details className="wizard-approval-record" key={draft.step}>
      <summary>生成履歴・承認の記録</summary>
      {draft.approved && <div className="wizard-approved"><Icon name="check" size={18}/><div><strong>世界観とキャストを承認しました</strong><p>承認した設定・素材の版を保存しました。本編では、この承認版を使います。</p></div></div>}
      <JobHistory jobs={setupJobs}/>
    </details>}
    <ChangeHistory key={detail.project.id} projectId={detail.project.id} revision={draft.revision} busy={busy} pending={pending} actionError={api.actionError} restoreEpoch={api.restoreEpoch} onRestore={api.restore} onNotice={onNotice}/>
    {approveOpen && <Dialog title="世界観とキャストを、制作の出発点に。" onClose={() => { if (!api.mutating) setApproveOpen(false) }}><p className="dialog-intro">確認した世界観と全キャラクターの設定・立ち絵・音声を承認します。現在の版を、後から参照できる形で保存します。</p><div className="wizard-approval-title"><Icon name="book" size={23}/><div><h3>{world.title}</h3><p>全{world.chapterCount}章 / メインキャラ {characters.length}人</p></div></div><p className="dialog-info">承認すると、全章のプロット・本文・背景・音声の制作が自動で始まります。完成した章から鑑賞でき、後続の章も引き続き制作します。追加の承認はありません。</p>{api.actionError && <p className="m2-job-error" role="alert">{api.actionError}</p>}<div className="dialog-actions"><button className="button button-light" disabled={api.mutating} onClick={() => setApproveOpen(false)}>確認に戻る</button><button className="button button-primary" disabled={!complete} onClick={() => { void api.action('approve').then(result => { if (result) { setApproveOpen(false); onNotice('世界観とキャストを承認し、本編の制作を開始しました。'); window.scrollTo({ top: 0 }) } }) }}><Icon name="check" size={15}/>承認して本編を制作</button></div></Dialog>}
  </>
}

type JobPanelProps = { jobs: M2Detail['jobs']; api: Api; activeJobId?: string | null; remainingJobCount?: number; pending: boolean }

function GenerationActivity({ jobs, api, activeJobId, remainingJobCount, pending }: JobPanelProps) {
  const active = jobs.filter(activeJob)
  const failed = jobs.find(job => job.status === 'failed' && job.id === activeJobId)
  if (!active.length && !failed) return null
  return <section className={`m2-jobs ${active.length ? 'm2-jobs-active' : ''}`} aria-label="生成の状況">
    <div className="m2-job-heading"><Icon name={active.length ? 'refresh' : 'help'} size={18}/><div>
      <strong>{active.length ? `物語のかたちを、整えています。${remainingJobCount ? ` 残り${remainingJobCount}件` : ''}` : '生成に失敗しました。再試行できます。'}</strong>
      {active.length > 0 && <p>生成済みのキャラクターは、切り替えて確認できます。ページを閉じても、サーバーとワーカーが起動していれば処理は続きます。</p>}
    </div></div>
    {active.map(job => <div className="m2-job-line" key={job.id}><span>{jobNames[job.kind] || job.kind}</span><strong>{statusNames[job.status]}</strong>{job.status === 'pending' && <small>ワーカーが受け付けるまでお待ちください</small>}</div>)}
    {failed && <div><p className="m2-job-error">{failed.error}</p><button className="button button-light" disabled={api.mutating || active.length > 0 || pending} onClick={() => void api.retry(failed.id)}>この生成を再試行</button></div>}
  </section>
}

function JobHistory({ jobs }: { jobs: M2Detail['jobs'] }) {
  if (!jobs.length) return null
  return <section className="m2-jobs m2-job-history" aria-label="生成履歴">
    <p>生成履歴 {jobs.length}件</p>
    <ul>{[...jobs].reverse().map(job => <li key={job.id}><div><strong>{jobNames[job.kind] || job.kind}</strong><span className={job.status === 'failed' ? 'm2-job-failed' : ''}>{statusNames[job.status]}</span><small>試行 {job.attempt_count}</small></div>{job.error && <p className="m2-job-error">{job.error}</p>}</li>)}</ul>
  </section>
}
