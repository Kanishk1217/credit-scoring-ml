import type { AssessResponse, Offer } from './types'
import { COPY, EFFORT_LABEL } from './copy'
import { fmtCurrency, fmtPct } from '../officer/format'

interface Props {
  data: AssessResponse
  onEdit: () => void
  onStartOver: () => void
}

function Label({ children }: { children: React.ReactNode }) {
  return <span className="font-label text-[11px] font-500 uppercase tracking-[0.06em] text-ink/55">{children}</span>
}

function ReadinessRing({ readiness }: { readiness: number }) {
  const r = 42
  const c = 2 * Math.PI * r
  const offset = c * (1 - readiness / 100)
  return (
    <svg width="104" height="104" viewBox="0 0 104 104" className="shrink-0">
      <circle cx="52" cy="52" r={r} fill="none" stroke="var(--color-line)" strokeWidth="3" />
      <circle
        cx="52" cy="52" r={r} fill="none" stroke="var(--color-accent)" strokeWidth="3"
        strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round"
        transform="rotate(-90 52 52)"
      />
      <text x="52" y="58" textAnchor="middle" className="font-display" style={{ fontSize: 26, fill: 'var(--color-ink)' }}>
        {readiness}
      </text>
    </svg>
  )
}

function RiskReadCard({ data }: { data: AssessResponse }) {
  const readiness = Math.round(100 - data.pd * 100)
  return (
    <div className="flex items-center gap-8 border border-line p-8">
      <ReadinessRing readiness={readiness} />
      <div>
        <Label>{COPY.readinessLabel}</Label>
        <h2 className="mt-1 font-display text-3xl italic text-accent">{data.band_headline}</h2>
        <p className="mt-2 max-w-md text-sm text-ink/70">
          {data.band === 'thriving' || data.band === 'steady'
            ? "You're well positioned for most loans on offer."
            : data.band === 'almost'
              ? "You're close. A small change or two could move you over the line."
              : "There's a clear path forward from here — see your plan below."}
        </p>
      </div>
    </div>
  )
}

function OfferCard({ label, offer, highlight }: { label: string; offer: Offer; highlight?: boolean }) {
  return (
    <div className="flex-1 border border-line p-6">
      <Label>{label}</Label>
      <div className={`mt-2 font-display text-3xl tabular-nums ${highlight ? 'text-accent' : 'text-ink'}`}>
        {fmtCurrency(offer.max_amount)}
      </div>
      <div className="mt-3 flex gap-6 text-sm text-ink/70">
        <span>{offer.apr.toFixed(1)}% APR</span>
        <span>{fmtCurrency(offer.monthly_emi)} / mo</span>
      </div>
      {offer.secured && <p className="mt-2 text-xs text-ink/45">Available as a secured offer.</p>}
    </div>
  )
}

function OfferCompare({ data }: { data: AssessResponse }) {
  const after = data.goal.steps.length > 0 ? data.goal.projected_offer : null
  return (
    <div className="flex flex-col gap-4 sm:flex-row">
      <OfferCard label={COPY.offerNowLabel} offer={data.offer_now} />
      {after && (
        <OfferCard label={COPY.offerAfterLabel} offer={after} highlight />
      )}
    </div>
  )
}

function WhyFactors({ data }: { data: AssessResponse }) {
  const maxImpact = Math.max(...data.why.map((w) => w.impact), 0.0001)
  return (
    <div>
      <Label>{COPY.whyLabel}</Label>
      <div className="mt-3 divide-y divide-line border-t border-line">
        {data.why.map((w) => (
          <div key={w.feature} className="flex items-center gap-3 py-2.5">
            <span className="w-44 shrink-0 text-sm text-ink">{w.label}</span>
            <div className="relative h-px flex-1 bg-line">
              <div
                className="absolute right-0 top-1/2 h-[3px] -translate-y-1/2"
                style={{ width: `${(w.impact / maxImpact) * 100}%`, background: w.direction === 'raises' ? 'var(--color-accent)' : 'rgba(23,20,15,.3)' }}
              />
            </div>
            <span className="w-40 shrink-0 text-right text-xs text-ink/55">{w.detail}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function AdviceCard({ item }: { item: AssessResponse['advice'][number] }) {
  const beforePct = Math.round(item.pd_before * 100)
  const afterPct = Math.round(item.pd_after * 100)
  return (
    <div className="border border-line p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="border border-line px-2 py-0.5 text-[10px] uppercase tracking-wide text-ink/55">
            {EFFORT_LABEL[item.effort]}
          </span>
          <span className="font-display text-lg text-ink">{item.title}</span>
        </div>
        <span className="font-display text-lg text-accent">−{Math.round(item.delta * 100)}%</span>
      </div>
      <div className="mt-4 flex items-center gap-3">
        <div className="h-2 flex-1 bg-line">
          <div className="h-2 bg-ink/60" style={{ width: `${beforePct}%` }} />
        </div>
        <span className="text-xs text-ink/55">{beforePct}%</span>
        <span className="text-ink/30">→</span>
        <div className="h-2 flex-1 bg-line">
          <div className="h-2 bg-accent" style={{ width: `${afterPct}%` }} />
        </div>
        <span className="text-xs text-ink/55">{afterPct}%</span>
      </div>
      <p className="mt-3 text-sm text-ink/70">
        Unlocks up to {fmtCurrency(item.unlocks.max_amount)} at {item.unlocks.apr.toFixed(1)}%
        {item.cost_inr ? ` · costs ₹${item.cost_inr.toLocaleString('en-IN')}` : ''}
        {item.horizon_months > 0 ? ` · ~${item.horizon_months} months` : ' · do this now'}
      </p>
    </div>
  )
}

function AdviceList({ data }: { data: AssessResponse }) {
  if (data.advice.length === 0) return null
  return (
    <div>
      <Label>{COPY.adviceLabel}</Label>
      <div className="mt-3 flex flex-col gap-3">
        {data.advice.map((item) => <AdviceCard key={item.id} item={item} />)}
      </div>
    </div>
  )
}

function GoalPlanner({ data }: { data: AssessResponse }) {
  const titleById = Object.fromEntries(data.advice.map((a) => [a.id, a.title]))
  return (
    <div className="border border-line p-6">
      <Label>{COPY.goalLabel}</Label>
      <p className="mt-2 font-display text-xl text-ink">
        {data.goal.steps.length === 0
          ? COPY.goalReached
          : data.goal.reachable
            ? COPY.goalReachable(data.goal.steps.length)
            : COPY.goalUnreachable}
      </p>
      {data.goal.steps.length > 0 && (
        <ol className="mt-4 flex flex-col gap-2">
          {data.goal.steps.map((id, i) => (
            <li key={id} className="flex items-center gap-3 text-sm text-ink">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center border border-line font-display text-xs">{i + 1}</span>
              {titleById[id] ?? id}
            </li>
          ))}
        </ol>
      )}
      <p className="mt-4 text-sm text-ink/70">
        Projected result: {fmtPct(data.goal.projected_pd, 0)} risk, up to {fmtCurrency(data.goal.projected_offer.max_amount)} available.
      </p>
    </div>
  )
}

export default function ResultsView({ data, onEdit, onStartOver }: Props) {
  return (
    <div className="flex flex-col gap-8">
      <RiskReadCard data={data} />
      <OfferCompare data={data} />
      <WhyFactors data={data} />
      <AdviceList data={data} />
      <GoalPlanner data={data} />
      <div className="flex gap-6 border-t border-line pt-6 text-sm">
        <button onClick={onEdit} className="text-ink/55 transition-colors hover:text-ink">{COPY.editInputs}</button>
        <button onClick={onStartOver} className="text-ink/55 transition-colors hover:text-ink">{COPY.startOver}</button>
      </div>
    </div>
  )
}
