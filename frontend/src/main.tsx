import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import OfficerDashboard from './officer/OfficerDashboard.tsx'
import SelfAssessmentPage from './consumer/SelfAssessmentPage.tsx'

function Router() {
  const path = window.location.pathname
  if (path === '/officer') return <OfficerDashboard />
  if (path === '/advisor') return <SelfAssessmentPage />
  return <App />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Router />
  </StrictMode>,
)
