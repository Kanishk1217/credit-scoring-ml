import { useState } from 'react'
import type { AssessResponse, PageState, Profile } from './types'
import { EMPTY_PROFILE } from './types'
import { assess, ApiError } from './api'
import AssessmentForm from './AssessmentForm'
import ResultsView from './ResultsView'
import { COPY } from './copy'

function isValid(p: Profile): boolean {
  return p.monthly_income > 0 && p.age >= 18 && p.age <= 80
    && p.employment_years >= 0 && p.num_existing_loans >= 0 && p.existing_debt >= 0
    && p.credit_limit >= 0 && p.payment_history.length === 12
}

function ResultsSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <div className="h-32 animate-pulse border border-line bg-line/20" />
      <div className="h-24 animate-pulse border border-line bg-line/20" />
      <div className="h-40 animate-pulse border border-line bg-line/20" />
    </div>
  )
}

function ErrorBanner({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex items-center justify-between border border-line bg-paper p-4">
      <p className="text-sm text-ink">{message}</p>
      <button onClick={onRetry} className="border border-line px-4 py-1.5 text-sm text-ink hover:border-ink">Retry</button>
    </div>
  )
}

export default function SelfAssessmentPage() {
  const [state, setState] = useState<PageState>('idle')
  const [profile, setProfile] = useState<Profile>(EMPTY_PROFILE)
  const [data, setData] = useState<AssessResponse | null>(null)
  const [err, setErr] = useState<string | null>(null)

  async function submit() {
    setState('submitting')
    try {
      const result = await assess(profile)
      setData(result)
      setState('results')
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : "Couldn't reach the network — check your connection.")
      setState('error')
    }
  }

  return (
    <div className="min-h-screen bg-bg">
      <header className="border-b border-line">
        <div className="mx-auto max-w-2xl px-6 py-5">
          <span className="font-display text-base font-600 tracking-tight text-ink">Agility</span>
        </div>
      </header>
      <main className="mx-auto max-w-2xl px-6 py-16">
        <h1 className="font-display text-4xl font-600 leading-tight text-ink md:text-5xl">
          {COPY.heading.split(' ').map((w, i) => (
            <span key={i}>{w.toLowerCase() === COPY.headingAccent ? <em className="italic text-accent">{w}</em> : w}{' '}</span>
          ))}
        </h1>
        <p className="mt-4 max-w-md text-base text-ink/70">{COPY.sub}</p>

        {state !== 'results' && (
          <div className="mt-10 flex flex-col gap-4">
            <AssessmentForm
              value={profile} onChange={(p) => { setProfile(p); if (state === 'idle') setState('editing') }}
              onSubmit={submit} submitting={state === 'submitting'} isValid={isValid(profile)}
            />
            {state === 'error' && <ErrorBanner message={err ?? COPY.errorFallback} onRetry={submit} />}
          </div>
        )}

        {state === 'submitting' && <div className="mt-10"><ResultsSkeleton /></div>}

        {state === 'results' && data && (
          <div className="mt-10">
            <ResultsView
              data={data}
              onEdit={() => setState('editing')}
              onStartOver={() => { setProfile(EMPTY_PROFILE); setData(null); setState('idle') }}
            />
          </div>
        )}
      </main>
    </div>
  )
}
