import { useCallback, useEffect, useState } from 'react'

const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
const STATES = [
  { v: 0, label: 'on time', color: 'var(--color-success)' },
  { v: 1, label: '1mo late', color: 'var(--color-warning)' },
  { v: 2, label: '2mo late', color: 'var(--color-danger)' },
]
type Pricing = { max_loan_amount: number; interest_rate_pct: number }
type Result = { probability_of_default: number; recommendation: string; pricing: Pricing }
const REC = {
  approve: { color: 'var(--color-success)', label: 'approve' },
  review: { color: 'var(--color-warning)', label: 'review' },
  decline: { color: 'var(--color-danger)', label: 'decline' },
} as const

export default function LiveDemo() {
  const [seq, setSeq] = useState<number[]>([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
  const [limit, setLimit] = useState(300000)
  const [collateral, setCollateral] = useState(false)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState(false)
  const [loading, setLoading] = useState(false)

  const cycle = (i: number) => setSeq((s) => s.map((v, j) => (j === i ? (v >= 2 ? 0 : v + 1) : v)))

  const score = useCallback(async () => {
    setLoading(true); setError(false)
    try {
      const res = await fetch('/api/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          age: 34, monthly_income: 45000, credit_limit: limit, existing_debt: 80000,
          employment_years: 4, num_existing_loans: 2, pay_status: seq, has_collateral: collateral,
        }),
      })
      if (!res.ok) throw new Error(String(res.status))
      setResult(await res.json())
    } catch { setError(true); setResult(null) } finally { setLoading(false) }
  }, [seq, limit, collateral])

  useEffect(() => { const t = setTimeout(score, 300); return () => clearTimeout(t) }, [score])

  const pd = result?.probability_of_default ?? 0
  const rec = result ? REC[result.recommendation as keyof typeof REC] : null

  return (
    <div className="border border-ink bg-paper">
      <div className="flex items-center justify-between border-b border-line px-6 py-4">
        <span className="font-label text-xs font-500 uppercase tracking-[0.2em] text-faint">live model</span>
        <span className="flex items-center gap-2 text-xs text-muted">
          <span className="h-1.5 w-1.5 rounded-full animate-pulse-dot" style={{ background: 'var(--color-success)' }} />
          real time
        </span>
      </div>

      <div className="px-6 py-6">
        <p className="mb-4 text-sm text-muted">12 months of payment history — tap a month to change it.</p>
        <div className="grid grid-cols-6 border border-line">
          {seq.map((v, i) => {
            const st = STATES.find((s) => s.v === v) ?? STATES[0]
            return (
              <button key={i} onClick={() => cycle(i)}
                className="group flex cursor-pointer flex-col items-center gap-2 border-b border-r border-line py-3 transition-colors last:border-r-0 hover:bg-bg [&:nth-child(6)]:border-r-0"
                aria-label={`${MONTHS[i]}: ${st.label}`}>
                <span className="font-label text-[10px] uppercase tracking-wider text-faint">{MONTHS[i]}</span>
                <span className="h-2 w-2 rounded-full" style={{ background: st.color }} />
              </button>
            )
          })}
        </div>

        <div className="mt-6 mb-4">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="text-muted">Credit limit</span>
            <span className="font-display font-500 tabular-nums text-ink">₹{limit.toLocaleString('en-IN')}</span>
          </div>
          <input type="range" min={50000} max={800000} step={10000} value={limit}
                 onChange={(e) => setLimit(Number(e.target.value))}
                 className="w-full accent-[var(--color-accent)]" />
        </div>

        <label className="mb-6 flex cursor-pointer items-center justify-between text-sm">
          <span className="text-muted">Secured with collateral</span>
          <input type="checkbox" checked={collateral} onChange={(e) => setCollateral(e.target.checked)}
                 className="h-4 w-4 accent-[var(--color-accent)]" />
        </label>

        <div className="border-t border-line pt-6">
          {error ? (
            <p className="text-sm text-danger">Start the scoring API on :8077.</p>
          ) : (
            <>
              <div className="flex items-end justify-between">
                <div>
                  <span className="font-label text-xs font-500 uppercase tracking-[0.2em] text-faint">probability of default</span>
                  <div className="mt-1 font-display font-700 tabular-nums leading-none transition-colors" style={{ fontSize: '3.25rem', color: rec?.color }}>
                    {(pd * 100).toFixed(1)}<span className="text-2xl">%</span>
                  </div>
                </div>
                {rec && <span className="font-label text-sm font-600 uppercase tracking-wider" style={{ color: rec.color }}>{rec.label}</span>}
              </div>
              <div className="mt-4 h-px w-full bg-line">
                <div className="h-px transition-all duration-500" style={{ width: `${Math.min(pd * 100, 100)}%`, background: rec?.color ?? 'var(--color-ink)', opacity: loading ? 0.4 : 1 }} />
              </div>
              {result && result.pricing.max_loan_amount > 0 && (
                <div className="mt-5 flex items-center justify-between text-sm">
                  <span className="text-muted">Offer: ₹{result.pricing.max_loan_amount.toLocaleString('en-IN')} at</span>
                  <span className="font-display font-600 text-ink">{result.pricing.interest_rate_pct}%</span>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
