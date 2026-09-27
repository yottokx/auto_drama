import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

async function start() {
  const legacy = new URLSearchParams(window.location.search).get('view') === 'm1'
  const { default: App } = legacy ? await import('./M1App') : await import('./App')
  if (legacy) await import('./m1.css')
  else {
    await import('./style.css')
    await import('./previewExtras.css')
  }
  createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)
}

void start()
