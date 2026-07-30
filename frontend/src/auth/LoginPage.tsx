import { useState } from 'react'
import { supabase, supabaseConfigured } from '../lib/supabaseClient'

export default function LoginPage({ productName = 'creditscore' }: { productName?: string }) {
  const [mode, setMode] = useState<'signin' | 'signup'>('signin')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!supabase) return
    setBusy(true); setError(null); setInfo(null)
    try {
      if (mode === 'signin') {
        const { error: err } = await supabase.auth.signInWithPassword({ email, password })
        if (err) throw err
      } else {
        const { error: err } = await supabase.auth.signUp({ email, password })
        if (err) throw err
        setInfo('Check your email to confirm your account.')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setBusy(false)
    }
  }

  async function handleGoogle() {
    if (!supabase) return
    setError(null)
    const { error: err } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: window.location.href },
    })
    if (err) setError(err.message)
  }

  if (!supabaseConfigured) {
    return (
      <div className="mx-auto max-w-sm border border-line bg-paper p-8 text-center">
        <p className="font-display text-lg text-ink">Sign-in isn't configured yet.</p>
        <p className="mt-2 text-sm text-ink/55">
          Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to enable accounts.
        </p>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-sm border border-line bg-paper p-8">
      <p className="font-display text-lg text-ink">{mode === 'signin' ? 'Sign in' : 'Create an account'}</p>
      <p className="mt-1 text-sm text-ink/55">to access the {productName} dashboard</p>

      <button
        onClick={handleGoogle}
        className="mt-6 flex w-full items-center justify-center gap-2 border border-line py-2.5 text-sm text-ink transition-colors hover:border-ink"
      >
        Continue with Google
      </button>

      <div className="my-6 flex items-center gap-3 text-xs uppercase tracking-[0.1em] text-ink/40">
        <div className="h-px flex-1 bg-line" /> or <div className="h-px flex-1 bg-line" />
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <label className="flex flex-col gap-1.5">
          <span className="font-label text-[11px] font-500 uppercase tracking-[0.06em] text-ink/55">Email</span>
          <input
            type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
            className="border-b border-line bg-transparent py-1.5 text-ink outline-none focus:border-ink"
          />
        </label>
        <label className="flex flex-col gap-1.5">
          <span className="font-label text-[11px] font-500 uppercase tracking-[0.06em] text-ink/55">Password</span>
          <input
            type="password" required minLength={6} value={password} onChange={(e) => setPassword(e.target.value)}
            className="border-b border-line bg-transparent py-1.5 text-ink outline-none focus:border-ink"
          />
        </label>

        {error && <p className="text-sm text-ink">{error}</p>}
        {info && <p className="text-sm text-ink/70">{info}</p>}

        <button
          type="submit" disabled={busy}
          className="mt-2 bg-accent px-6 py-2.5 font-display text-sm font-500 text-paper transition-opacity disabled:opacity-40"
        >
          {mode === 'signin' ? 'Sign in' : 'Create account'}
        </button>
      </form>

      <button
        onClick={() => setMode(mode === 'signin' ? 'signup' : 'signin')}
        className="mt-4 text-sm text-ink/55 transition-colors hover:text-ink"
      >
        {mode === 'signin' ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
      </button>
    </div>
  )
}
