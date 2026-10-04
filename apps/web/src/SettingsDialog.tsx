import { useRef, useState, type KeyboardEvent } from 'react'
import { Dialog } from './PreviewComponents'
import { LLMSettingsContent } from './LLMSettings'
import { TTSSettings } from './TTSSettings'
import { EventCgSettings } from './EventCgSettings'
import './tts-settings.css'

type SettingsTab = 'llm' | 'tts' | 'image'
const tabs: SettingsTab[] = ['llm', 'tts', 'image']

export function SettingsDialog({ onClose }: { onClose: () => void }) {
  const [tab, setTab] = useState<SettingsTab>('llm')
  const [llmBusy, setLLMBusy] = useState(false)
  const [ttsBusy, setTTSBusy] = useState(false)
  const [imageBusy, setImageBusy] = useState(false)
  const tabButtons = useRef<(HTMLButtonElement | null)[]>([])
  function navigateTab(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : null
    if (next === null) return
    event.preventDefault(); setTab(tabs[next]); tabButtons.current[next]?.focus()
  }
  return <Dialog title="設定" wide className="settings-dialog" onClose={() => { if (!llmBusy && !ttsBusy && !imageBusy) onClose() }}>
    <div className="settings-tabs" role="tablist" aria-label="設定の種類">{tabs.map((id, index) => <button key={id} ref={button => { tabButtons.current[index] = button }} id={`settings-tab-${id}`} type="button" role="tab" aria-selected={tab === id} aria-controls={`settings-panel-${id}`} tabIndex={tab === id ? 0 : -1} onClick={() => setTab(id)} onKeyDown={event => navigateTab(event, index)}>{id === 'image' ? '画像' : id.toUpperCase()}</button>)}</div>
    <div id="settings-panel-llm" role="tabpanel" aria-labelledby="settings-tab-llm" hidden={tab !== 'llm'} tabIndex={0}><LLMSettingsContent onBusyChange={setLLMBusy}/></div>
    <div id="settings-panel-tts" role="tabpanel" aria-labelledby="settings-tab-tts" hidden={tab !== 'tts'} tabIndex={0}><TTSSettings visible={tab === 'tts'} onBusyChange={setTTSBusy}/></div>
    <div id="settings-panel-image" role="tabpanel" aria-labelledby="settings-tab-image" hidden={tab !== 'image'} tabIndex={0}><EventCgSettings visible={tab === 'image'} onBusyChange={setImageBusy}/></div>
  </Dialog>
}
