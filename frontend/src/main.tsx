import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import OfficerDashboard from './officer/OfficerDashboard.tsx'
import SelfAssessmentPage from './consumer/SelfAssessmentPage.tsx'
import { AuthProvider } from './auth/AuthContext.tsx'
import RequireAuth from './auth/RequireAuth.tsx'

function Router() {
  const path = window.location.pathname
  if (path === '/officer') {
    return (
      <AuthProvider>
        <RequireAuth productName="loan officer">
          <OfficerDashboard />
        </RequireAuth>
      </AuthProvider>
    )
  }
  if (path === '/advisor') {
    return (
      <AuthProvider>
        <RequireAuth productName="advisor">
          <SelfAssessmentPage />
        </RequireAuth>
      </AuthProvider>
    )
  }
  return <App />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Router />
  </StrictMode>,
)
