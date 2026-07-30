import type { Session, User } from '@supabase/supabase-js'
import { createContext, useContext, useEffect, useState } from 'react'
import { supabase, supabaseConfigured } from '../lib/supabaseClient'

interface AuthState {
  session: Session | null
  user: User | null
  loading: boolean
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthState>({
  session: null, user: null, loading: false, signOut: async () => {},
})

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(supabaseConfigured)

  useEffect(() => {
    if (!supabase) return
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session)
      setLoading(false)
    })
    const { data: listener } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next)
    })
    return () => listener.subscription.unsubscribe()
  }, [])

  const signOut = async () => {
    if (supabase) await supabase.auth.signOut()
    setSession(null)
  }

  return (
    <AuthContext.Provider value={{ session, user: session?.user ?? null, loading, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
