import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL as string | undefined
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined

// Anon key is safe to ship client-side (Supabase's Row Level Security enforces access, not this
// key) -- unlike CREDIT_API_KEYS, this is not a value that needs to stay server-side.
export const supabaseConfigured = Boolean(url && anonKey)

export const supabase = supabaseConfigured
  ? createClient(url!, anonKey!)
  : null
