import DemoWizardApp from './DemoWizardApp'
import LiveWizardApp from './LiveWizardApp'

export default function WizardApp() {
  return new URLSearchParams(window.location.search).get('demo') === 'results' ? <DemoWizardApp/> : <LiveWizardApp/>
}
