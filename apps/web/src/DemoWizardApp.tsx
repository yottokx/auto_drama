import { useEffect, useState } from 'react'
import { Icon, type IconName } from './Icons'
import { Dialog } from './PreviewComponents'
import { useWizardState, type WizardStep } from './wizardState'
import { WorldReview } from './WorldSteps'
import { CombinedBriefInput } from './CombinedBriefInput'
import { relationshipPairs } from './RelationshipSteps'
import { CharacterReview } from './CharacterSteps'
import { createReviewSample, reviewSampleAssets } from './reviewSample'
import './wizard.css'
import './worldSteps.css'
import './characterSteps.css'

const steps: { id: WizardStep; title: string; short: string; icon: IconName; heading: string; description: string }[] = [
  { id: 'world-input', title: '世界観・メインキャラを指示', short: '世界観・人物の指示', icon: 'globe', heading: 'どんな世界で、誰の物語をはじめますか。', description: '世界観とメインキャラクターの希望をまとめて伝え、世界観から順番に確定します。' },
  { id: 'world-review', title: '世界観を確認・確定', short: '世界観の確定', icon: 'book', heading: 'この物語の世界観。', description: '舞台設定と物語の方向性を確認します。気になるところがあれば修正し、内容を確定してください。' },
  { id: 'character-review', title: 'メインキャラを確認・確定', short: 'キャラの確認', icon: 'spark', heading: 'この物語を生きる、登場人物。', description: '人物の設定・立ち絵・声を確認します。修正したいときは、結果の下にある修正欄を開いてください。' },
  { id: 'planning-review', title: '全体計画を確認・確定', short: '全体計画の確認', icon: 'book', heading: '物語全体の道筋を、整える。', description: '全体プロットと登場予定のサブキャラを確認・修正してから、本編の制作へ進みます。' },
  { id: 'production', title: '本編の制作・鑑賞', short: '本編の制作・鑑賞', icon: 'film', heading: '本編の制作・鑑賞', description: '承認した設定から、本編の制作へ進みます。完成した章の鑑賞と書き出しも、この画面で行います。' },
]

export default function WizardApp() {
  const [sample] = useState(() => new URLSearchParams(window.location.search).get('demo') === 'results' ? createReviewSample() : undefined)
  const state = useWizardState(sample)
  const { draft } = state
  const [library, setLibrary] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [pending, setPending] = useState(false)
  const [toast, setToast] = useState('')
  const [dialog, setDialog] = useState<'approve' | 'settings' | 'guide' | null>(null)
  const index = steps.findIndex(step => step.id === draft.step)
  const current = steps[index]
  const title = draft.world.title.trim() || 'タイトル未定の物語'
  const completedStep = (stepIndex: number) => stepIndex < 2 ? draft.worldConfirmed : stepIndex === 2 ? draft.approved : stepIndex === 3 && draft.planApproved

  function stepGate(step: WizardStep) {
    if ((step === 'character-review' || step === 'planning-review' || step === 'production') && !draft.worldConfirmed) return '先に世界観を確定してください'
    if ((step === 'planning-review' || step === 'production') && !draft.approved) return '先にメインキャラを確認・承認してください'
    if (step === 'production' && !draft.planApproved) return '先に全体計画を確認・承認してください'
    return undefined
  }

  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(() => setToast(''), 5500)
    return () => clearTimeout(timer)
  }, [toast])

  useEffect(() => {
    if (!menuOpen) return
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setMenuOpen(false) }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [menuOpen])

  useEffect(() => {
    if (!pending) return
    const warnBeforeLeaving = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', warnBeforeLeaving)
    return () => window.removeEventListener('beforeunload', warnBeforeLeaving)
  }, [pending])

  function canLeave() {
    if (!pending) return true
    setToast('未保存の変更があります。変更を保存するか、キャンセルしてから移動してください。')
    setMenuOpen(false)
    return false
  }

  function navigate(step: WizardStep) {
    if (step !== draft.step && !canLeave()) return
    state.goTo(step)
    setLibrary(false)
    setMenuOpen(false)
    window.scrollTo({ top: 0 })
  }

  function openLibrary() {
    if (!canLeave()) return
    setLibrary(true)
    setMenuOpen(false)
    window.scrollTo({ top: 0 })
  }

  function confirmWorld() {
    state.confirmWorld()
    setPending(false)
    setToast('世界観を確定し、キャラクターの確認画面へ進みました。プレビューではAI生成は実行されません。')
    window.scrollTo({ top: 0 })
  }

  function request(scope: Parameters<typeof state.requestRevision>[0], instruction: string, characterId?: string) {
    const accepted = state.requestRevision(scope, instruction, characterId)
    setToast(accepted ? '修正指示を保存しました。AIによる書き換え・素材生成はまだ実行されません。' : '対象が固定されているか、指示が未入力です。対象と内容を確認してください。')
    return accepted
  }

  return <div className="studio wizard-studio">
    <a className="skip-link" href="#main">メインコンテンツへ</a>
    {menuOpen && <button className="nav-scrim" aria-label="ナビゲーションを閉じる" onClick={() => setMenuOpen(false)}/>}
    <aside id="studio-navigation" className={`studio-sidebar ${menuOpen ? 'sidebar-open' : ''}`} aria-label="メインナビゲーション">
      <button className="studio-brand" onClick={() => navigate('world-input')}><span className="brand-symbol"><Icon name="spark" size={26}/></span><span>AIオートドラマ<small>YOUR STORY, YOUR WAY.</small></span></button>
      <button className="new-project" onClick={() => navigate(draft.step)}><Icon name="edit" size={17}/>制作ウィザード</button>
      <nav className="primary-nav"><button className={library ? 'nav-active' : ''} onClick={openLibrary}><Icon name="grid"/>作品ライブラリ</button></nav>
      <div className="nav-divider"/>
      <p className="nav-caption">物語をつくる · 5 STEPS</p>
      <div className="current-project"><span className="project-initial"><Icon name="book" size={16}/></span><span>{title}<small>全{draft.world.chapterCount}章 · {draft.approved ? '承認プレビュー済み' : '下書き'}</small></span></div>
      <nav className="workspace-nav" aria-label="制作ステップ">
        {steps.map((step, i) => <button key={step.id} className={!library && draft.step === step.id ? 'nav-active' : ''} disabled={Boolean(stepGate(step.id))} onClick={() => navigate(step.id)} aria-current={!library && draft.step === step.id ? 'step' : undefined} title={stepGate(step.id)}>
          <Icon name={step.icon} size={17}/>{step.title}<span className="nav-step">{i + 1}</span>
        </button>)}
      </nav>
      <div className="sidebar-bottom"><div className="sidebar-note"><span className="tiny-spark">✧</span><p>決めたいところは、自由に。<br/>まだ決めないところは、おまかせに。</p></div><button onClick={() => setDialog('settings')}><Icon name="settings"/>環境設定</button><button onClick={() => setDialog('guide')}><Icon name="help"/>使い方ガイド</button><div className="local-profile"><span>AD</span><div>ローカルワークスペース<small><i/>UIプレビュー</small></div></div></div>
    </aside>
    <div className="studio-body">
      <header className="topbar"><div className="breadcrumbs"><button className="mobile-menu icon-button" aria-label="ナビゲーションを開く" aria-controls="studio-navigation" aria-expanded={menuOpen} onClick={() => setMenuOpen(true)}><Icon name="grid"/></button><button onClick={openLibrary}>ワークスペース</button><Icon name="chevron" size={12}/><span>{title}</span></div><div className="topbar-right"><span className="preview-pill">UIプレビュー</span><span className="save-status"><Icon name="check" size={13}/>{pending ? '未保存の変更あり' : state.storageAvailable ? 'ブラウザーに保存' : '保存できません・一時編集中'}</span></div></header>
      <main id="main" className="main-content">
        {library ? <>
          <div className="page-heading"><div><div className="eyebrow">YOUR COLLECTION</div><h1>あなたの物語。</h1><p>制作中の下書きから、つづきを。</p></div></div>
          <button className="wizard-library-card" onClick={() => navigate(draft.step)}><span className="wizard-library-art"><Icon name="book" size={38}/></span><span><small>全{draft.world.chapterCount}章 · {draft.world.genre || 'ジャンルはおまかせ'}</small><strong>{title}</strong><span>{current.title}から再開<Icon name="arrow" size={16}/></span></span></button>
          <p className="sample-note">このプレビューでは1つの下書きを編集できます。M1の保存済み作品は環境設定から開けます。</p>
        </> : <>
          <div className="page-heading"><div><div className="eyebrow">CREATE YOUR STORY <span>/</span> 0{index + 1}</div><h1>{current.heading}</h1><p>{current.description}</p></div><span className="wizard-draft-label">DRAFT {String(draft.revision).padStart(2, '0')}</span></div>
          <ol className="wizard-stepper" aria-label="制作のステップ">{steps.map((step, i) => <li key={step.id} className={i === index ? 'current' : completedStep(i) ? 'complete' : ''} aria-current={i === index ? 'step' : undefined}><button disabled={Boolean(stepGate(step.id))} title={stepGate(step.id)} onClick={() => navigate(step.id)}><span>{completedStep(i) ? <Icon name="check" size={14}/> : `0${i + 1}`}</span><div>{step.short}<small>{['まとめて伝える', '世界観を確定する', 'メインキャラを確定', 'プロットとサブキャラ', '制作して楽しむ'][i]}</small></div></button></li>)}</ol>
          <div className="wizard-preview-note"><Icon name="edit" size={14}/><p>{sample ? '生成結果の表示サンプルです。設定と参考素材は外観確認用で、入力に基づくAI生成ではありません。' : 'UIプレビューです。AIによる設定・素材の生成はまだ接続していません。'}</p><a href={sample ? '/' : '/?demo=results'} onClick={event => { if (!canLeave()) event.preventDefault() }}>{sample ? '自分の下書きへ戻る' : '生成結果の表示サンプルを見る'}<Icon name="arrow" size={12}/></a></div>
          <div className={`wizard-workspace ${index < 2 ? 'world-stage-layout' : ''}`}>
            <section className="wizard-stage" aria-label={current.title}>
              {draft.step === 'world-input' && <CombinedBriefInput world={draft.world} characters={draft.characters} relationshipInputs={relationshipPairs(draft.characters, draft.relationshipInputs ?? [])} onWorldChange={state.updateWorld} onCharacterChange={state.updateCharacter} onRelationshipsChange={state.updateRelationships} onAdd={state.addCharacter} onRemove={state.removeCharacter} onNext={() => navigate('world-review')} hasResults={Boolean(draft.world.setting)}/>}
              {draft.step === 'world-review' && <WorldReview worldConfirmed={draft.worldConfirmed} world={draft.world} requests={draft.requests} onChange={state.updateWorld} onRequest={instruction => request('world', instruction)} onBack={() => navigate('world-input')} onConfirm={confirmWorld} onPendingChange={setPending}/>}
              {draft.step === 'character-review' && <CharacterReview previewAssets={sample ? reviewSampleAssets : undefined} characters={draft.characters} requests={draft.requests} onChange={state.updateCharacter} onToggleLock={state.toggleLock} onRequest={request} onBack={() => navigate('world-review')} onApprove={() => setDialog('approve')} onPendingChange={setPending}/>}
              {draft.step === 'planning-review' && <section className="char-input-sheet"><div className="char-sheet-heading"><div><span className="char-kicker">STEP 04</span><h2>全体計画を確認・確定</h2><p>実際の制作では、全体プロットと登場予定のサブキャラの設定・関係性を確認します。直接編集やAIへの修正指示で整え、承認後に第1章から本編を制作します。</p><p>サブキャラの立ち絵と基準音声はSTEP5で生成します。この表示サンプルでは全体計画の生成・AI修正は実行しません。</p></div></div><div className="char-step-footer"><button className="button button-light" onClick={() => navigate('character-review')}><Icon name="back" size={16}/>メインキャラの確認に戻る</button><button className="button button-primary" onClick={() => { if (state.approvePlan()) { setToast('全体計画の承認後の画面を表示しています。本編の生成は実行されません。'); window.scrollTo({ top: 0 }) } }}>全体計画を承認（プレビュー）<Icon name="arrow" size={16}/></button></div></section>}
              {draft.step === 'production' && <section className="char-input-sheet"><div className="char-sheet-heading"><div><span className="char-kicker">STEP 05</span><h2>本編の制作・鑑賞</h2><p>第1章から順番に制作し、完成した章を確認する画面です。<br/>このプレビューでは、本編の生成・鑑賞・書き出しは実行しません。</p></div></div><button type="button" className="button button-light" onClick={() => navigate('planning-review')}><Icon name="back" size={16}/>全体計画の確認に戻る</button></section>}
            </section>
            <aside className="wizard-summary" aria-label="下書きの概要">
              <p className="eyebrow">STORY NOTE</p><h2>{title}</h2>
              <dl><div><dt>世界観</dt><dd className={draft.worldConfirmed ? 'confirmed-text' : ''}>{draft.worldConfirmed ? '確定済み' : '調整中'}</dd></div><div><dt>ジャンル</dt><dd>{draft.world.genre || 'おまかせ'}</dd></div><div><dt>雰囲気</dt><dd>{draft.world.mood || 'おまかせ'}</dd></div><div><dt>構成</dt><dd>全{draft.world.chapterCount}章</dd></div></dl>
              <div className="wizard-summary-divider"/>
              <h3>メインキャラクター <span>{draft.characters.length}</span></h3>
              <ul>{draft.characters.map((character, i) => <li key={character.id}><span>{String(i + 1).padStart(2, '0')}</span><div>{character.name || `キャラクター ${i + 1}`}<small>{character.role || '役割はおまかせ'}</small></div></li>)}</ul>
              <div className="wizard-summary-tip"><Icon name="leaf" size={17}/>{index >= 3 ? <p>物語の準備が整いました。<small>世界観やキャストは、前のステップから見直せます。</small></p> : index === 2 ? <p>まずは、人物像をじっくり。<small>気になる箇所だけ修正できます。全体を変えたいときは、結果の下にある修正欄へ。</small></p> : <p>決まっていることだけで大丈夫。<small>ジャンルの組み合わせや、枠に収まらない人物像も自由に指定できます。</small></p>}</div>
              <span className="wizard-request-count">保存した修正・リテイク指示：{draft.requests.length}件</span>
              {index >= 2 && <button className="wizard-world-back" onClick={() => navigate('world-review')}><Icon name="back" size={14}/>確定した世界観を見直す</button>}
            </aside>
          </div>
          {draft.approved && <details className="wizard-approval-record"><summary><Icon name="check" size={15}/>承認の記録</summary><div className="wizard-approved"><Icon name="check" size={18}/><div><strong>承認後の表示をプレビュー中</strong><p>入力した設定と修正指示をまとめて保存しました。本編の制作は開始されません。</p></div></div></details>}
        </>}
        <footer className="studio-footer"><span>AI AUTO DRAMA <i/> YOUR STORY, YOUR WAY.</span><span>M2 · UI PREVIEW</span></footer>
      </main>
    </div>
    {toast && <div className="toast" role="status"><Icon name="check" size={16}/>{toast}<button className="icon-button" aria-label="通知を閉じる" onClick={() => setToast('')}><Icon name="close" size={14}/></button></div>}
    {dialog === 'approve' && <Dialog title="世界観とキャストを、制作の出発点に。" onClose={() => setDialog(null)}><p className="dialog-intro">表示中の世界観と、メインキャラの設定・外見・声を全体計画の出発点として確定します。</p><div className="wizard-approval-title"><Icon name="book" size={23}/><div><h3>{title}</h3><p>全{draft.world.chapterCount}章 / メインキャラ {draft.characters.length}人</p></div></div><ul className="approval-details"><li><Icon name="check" size={15}/>世界観・方向性：{draft.worldConfirmed ? '確定済み' : '未確定'}</li><li><Icon name="check" size={15}/>人物設定・外見・声：表示中の内容を確認</li><li><Icon name="edit" size={15}/>保存した修正指示：{draft.requests.length}件（AIによる反映は未実行）</li></ul><p className="dialog-info">UIプレビューでは、承認状態をブラウザーに保存します。AIの修正結果や新しい立ち絵・音声は未生成です。本編の制作は開始されません。</p><div className="dialog-actions"><button className="button button-light" onClick={() => setDialog(null)}>確認に戻る</button><button className="button button-primary" disabled={!draft.worldConfirmed} onClick={() => { if (state.approve()) { setDialog(null); setToast('全体計画の確認画面に進みました。設定と指示はブラウザーに保存されています。'); window.scrollTo({ top: 0 }) } }}><Icon name="check" size={15}/>承認して全体計画へ（プレビュー）</button></div></Dialog>}
    {dialog === 'settings' && <Dialog title="ワークスペースの環境" onClose={() => setDialog(null)}><div className="settings-row"><span>現在のモード</span><strong>UIプレビュー</strong></div><div className="settings-row"><span>設定と指示の保存先</span><strong>{state.storageAvailable ? 'このブラウザー' : '一時編集中・保存不可'}</strong></div><p className="dialog-intro">AIによる生成と修正は今後接続します。M1の保存・ジョブ・書き出し機能は下から開けます。</p><a className="button button-light" href="/?view=m1" onClick={event => { if (!canLeave()) { event.preventDefault(); setDialog(null) } }}>M1の実装済み画面を開く<Icon name="arrow" size={16}/></a></Dialog>}
    {dialog === 'guide' && <Dialog title="自由に伝えて、順番に整える。" onClose={() => setDialog(null)}><ol className="guide-steps">{steps.map((step, i) => <li key={step.id}><span>0{i + 1}</span><div><h3>{step.title}</h3><p>{step.description}</p></div></li>)}</ol><p className="dialog-info">世界観の変更後は再確定してから、キャラクターへ進みます。全体修正では人物設定・外見・声を横断して調整し、固定した項目を保護します。立ち絵や音声のリテイクは、設定を保ったまま素材だけを作り直す操作です。</p></Dialog>}
  </div>
}
