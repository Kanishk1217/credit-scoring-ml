import { useAuth } from './AuthContext'
import LoginPage from './LoginPage'

export default function RequireAuth({ children, productName }: { children: React.ReactNode; productName?: string }) {
  const { session, loading } = useAuth()

  if (loading) {
    return <div className="flex min-h-screen items-center justify-center bg-bg text-ink/55">Loading…</div>
  }
  if (!session) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-bg px-6">
        <LoginPage productName={productName} />
      </div>
    )
  }
  return <>{children}</>
}
