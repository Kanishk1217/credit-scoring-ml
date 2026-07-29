import { useState } from 'react'
import type { ScoreResult, Verdict } from './types'
import { APPROVE_MAX, DECLINE_MIN, T_STAR } from './types'
import { fmtCurrency, fmtPct } from './format'

type PanelState = 'empty' | 'loading' | 'error' | 'result'

interface Props {
  state: PanelState
  result: ScoreResult | null
  errorMessage: string | null
  onRetry: () => void
  onCopyJson: () => void
}

function Label({ children }: { children: React.ReactNode }) {
  return <span className="font-label text-[11px] font-500 uppercase tracking-[0.06em] text-ink/55">{children}</span>
}

const VERDICT_LABEL: Record<Verdict, string> = { approve: 'Approve', review: 'Review', decline: 'Decline' }

function VerdictHeader({ result }: { result: ScoreResult }) {
  return (
    <div className="flex items-start justify-between">
      <div>
        <Label>Probability of default</Label>
        <div className="mt-1 flex items-baseline gap-3">
          <span className="font-display text-6xl font-600 tabular-nums text-ink">{fmtPct(result.pd, 1)}</span>
          <span className="border border-line px-2 py-0.5 text-xs font-500 uppercase tracking-wide text-ink/70">
            Band {result.band}
          </span>
        </div>
        <div className={`mt-1 font-display text-3xl italic ${result.verdict === 'decline' ? 'text-ink line-through decoration-1' : result.verdict === 'review' ? 'text-ink' : 'text-accent'}`}>
          {VERDICT_LABEL[result.verdict]}
        </div>
        {result.override_reason && (
          <p className="mt-2 max-w-sm border-l-2 border-accent pl-3 text-xs text-ink/70">
            Policy override: {result.override_reason}.
          </p>
        )}
      </div>
    </div>
  )
}

function ThresholdGauge({ pd, dimmed = false }: { pd: number; dimmed?: boolean }) {
  const clamp = (x: number) => Math.max(0, Math.min(1, x / 0.40))
  return (
    <div className={`mt-10 ${dimmed ? 'opacity-40' : ''}`}>
      <div className="relative h-px bg-line">
        <Tick at={APPROVE_MAX} label="10%" />
        <Tick at={DECLINE_MIN} label="25%" />
        <Tick at={T_STAR} label="breakeven 16.7%" dotted />
        {!dimmed && (
          <div
            className="absolute top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent"
            style={{ left: `${clamp(pd) * 100}%` }}
          />
        )}
      </div>
      <div className="mt-3 flex justify-between text-[10px] uppercase tracking-[0.06em] text-ink/45">
        <span>Approve</span><span>Review</span><span>Decline</span>
      </div>
    </div>
  )

  // stacked in normal flow inside a bottom-anchored wrapper, so label + hash both sit
  // strictly above the line -- can never collide with the zone labels rendered below it
  function Tick({ at, label, dotted }: { at: number; label: string; dotted?: boolean }) {
    return (
      <div className="absolute bottom-0 flex -translate-x-1/2 flex-col items-center gap-1"
           style={{ left: `${clamp(at) * 100}%` }}>
        <span className="whitespace-nowrap text-[9px] text-ink/45">{label}</span>
        <div className={`h-2 w-px ${dotted ? 'border-l border-dotted border-ink/40' : 'bg-ink/40'}`} />
      </div>
    )
  }
}

function WhyList({ factors }: { factors: ScoreResult['factors'] }) {
  return (
    <div className="mt-10">
      <Label>Why</Label>
      <div className="mt-3 divide-y divide-line border-t border-line">
        {factors.slice(0, 5).map((f) => (
          <div key={f.feature} className="flex items-center gap-3 py-2.5">
            <span className="w-44 shrink-0 text-sm text-ink">
              {f.label}<em className="ml-2 not-italic text-xs text-ink/55">{f.value}</em>
            </span>
            <div className="relative h-px flex-1 bg-line">
              <div
                className="absolute right-0 top-1/2 h-[3px] -translate-y-1/2"
                style={{ width: `${f.weightPct * 100}%`, background: f.direction === 'raises' ? 'var(--color-accent)' : 'rgba(23,20,15,.3)' }}
              />
            </div>
            <span className={f.direction === 'raises' ? 'text-accent' : 'text-ink/55'}>
              {f.direction === 'raises' ? '↑' : '↓'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function PricingCard({ result }: { result: ScoreResult }) {
  if (result.verdict === 'decline') return null
  return (
    <div className="mt-10 border border-line p-6">
      <Label>{result.verdict === 'review' ? 'Provisional offer' : 'Offer'}</Label>
      <div className="mt-3 grid grid-cols-3 gap-4">
        <div>
          <div className="text-[10px] uppercase tracking-[0.06em] text-ink/45">Offered</div>
          <div className="mt-1 font-display text-xl tabular-nums text-ink">{fmtCurrency(result.pricing.offered_amount)}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.06em] text-ink/45">APR</div>
          <div className="mt-1 font-display text-xl tabular-nums text-ink">{fmtPct(result.pricing.apr, 1)}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.06em] text-ink/45">EMI</div>
          <div className="mt-1 font-display text-xl tabular-nums text-ink">
            {fmtCurrency(result.pricing.emi)} <span className="text-xs text-ink/45">/ {result.pricing.tenor_months} mo</span>
          </div>
        </div>
      </div>
      <p className="mt-4 text-xs text-ink/45">Amount capped by requested limit and 50% DTI ceiling.</p>
    </div>
  )
}

const DECLINE_REASONS = ['DTI too high', 'Payment history', 'Insufficient tenure', 'Other']

function ActionBar({ result, onCopyJson }: { result: ScoreResult; onCopyJson: () => void }) {
  const [reason, setReason] = useState(DECLINE_REASONS[0])
  return (
    <div className="mt-10 flex flex-wrap items-center gap-4 border-t border-line pt-6">
      {result.verdict === 'approve' && (
        <button className="bg-accent px-5 py-2.5 font-display text-sm font-500 text-paper">
          Approve &amp; generate offer
        </button>
      )}
      {result.verdict === 'review' && (
        <button className="border border-ink px-5 py-2.5 font-display text-sm font-500 text-ink">
          Send to review queue
        </button>
      )}
      {result.verdict === 'decline' && (
        <>
          <button className="border border-ink px-5 py-2.5 font-display text-sm font-500 text-ink">Decline</button>
          <select value={reason} onChange={(e) => setReason(e.target.value)}
            className="border-b border-line bg-transparent py-1 text-sm text-ink outline-none">
            {DECLINE_REASONS.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </>
      )}
      <span className="flex-1" />
      <button onClick={onCopyJson} className="text-sm text-ink/55 transition-colors hover:text-ink">Copy result JSON</button>
      <span className="text-sm text-ink/25">·</span>
      <button className="text-sm text-ink/55 transition-colors hover:text-ink">Export PDF</button>
    </div>
  )
}

export default function DecisionPanel({ state, result, errorMessage, onRetry, onCopyJson }: Props) {
  if (state === 'empty') {
    return (
      <div className="border border-line p-8 md:p-10">
        <p className="font-display text-lg italic text-ink/55">Enter applicant details and score.</p>
        <ThresholdGauge pd={0} dimmed />
      </div>
    )
  }
  if (state === 'loading') {
    return (
      <div className="border border-line p-8 md:p-10">
        <Label>Probability of default</Label>
        <div className="mt-2 h-14 w-40 animate-pulse bg-line" />
        <div className="mt-3 font-display text-2xl italic text-ink/45">Scoring…</div>
      </div>
    )
  }
  if (state === 'error') {
    return (
      <div className="border border-line p-8 md:p-10">
        <p className="text-ink">Could not score this applicant.</p>
        <p className="mt-1 text-sm text-ink/55">{errorMessage}</p>
        <button onClick={onRetry} className="mt-4 border border-line px-4 py-2 text-sm text-ink transition-colors hover:border-ink">
          Retry
        </button>
      </div>
    )
  }
  if (!result) return null
  return (
    <div className="border border-line p-8 md:p-10">
      <VerdictHeader result={result} />
      <ThresholdGauge pd={result.pd} />
      <WhyList factors={result.factors} />
      <PricingCard result={result} />
      <ActionBar result={result} onCopyJson={onCopyJson} />
    </div>
  )
}
