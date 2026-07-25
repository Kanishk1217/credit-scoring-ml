import { useCallback, useEffect, useState } from 'react'

const MONTHS = ['apr', 'may', 'jun', 'jul', 'aug', 'sep']
const STATES = [
  { v: 0, label: 'on time', color: 'var(--color-success)' },
  { v: 1, label: '1mo late', color: 'var(--color-warning)' },
  { v: 2, label: '2mo late', color: 'var(--color-danger)' },
]
type Result = { probability_of_default: number; recommendation: string }
const REC = {
  approve: { color: 'var(--color-success)', label: 'approve' },
  review: { color: 'var(--color-warning)', label: 'review' },
  decline: { color: 'var(--color-danger)', label: 'decline' },
} as const

export default function LiveDemo() {
  const [seq, setSeq] = useState<number[]>([0, 0, 0, 0, 2, 2])
  const [limit, setLimit] = useState(120000)
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
          limit_bal: limit, sex: 2, education: 2, marriage: 1, age: 34,
          bill_amt: [80000, 80000, 80000, 80000, 80000, 80000],
          pay_amt: [3000, 3000, 3000, 3000, 3000, 3000], pay_status: seq,
        }),
      })
      if (!res.ok) throw new Error(String(res.status))
      setResult(await res.json())
    } catch { setError(true); setResult(null) } finally { setLoading(false) }
  }, [seq, limit])

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
        <p className="mb-4 text-sm text-muted">Tap a month to change its payment status.</p>
        <div className="grid grid-cols-6 border border-line">
          {seq.map((v, i) => {
            const st = STATES.find((s) => s.v === v) ?? STATES[0]
            return (
              <button key={i} onClick={() => cycle(i)}
                className="group flex cursor-pointer flex-col items-center gap-2.5 border-r border-line py-4 transition-colors last:border-r-0 hover:bg-bg"
                aria-label={`${MONTHS[i]}: ${st.label}`}>
                <span className="font-label text-[11px] uppercase tracking-wider text-faint">{MONTHS[i]}</span>
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: st.color }} />
                <span className="text-[10px] text-muted">{st.label}</span>
              </button>
            )
          })}
        </div>

        <div className="mt-6 mb-6">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="text-muted">Credit limit</span>
            <span className="font-display font-500 tabular-nums text-ink">₹{limit.toLocaleString('en-IN')}</span>
          </div>
          <input type="range" min={20000} max={500000} step={10000} value={limit}
                 onChange={(e) => setLimit(Number(e.target.value))}
                 className="w-full accent-[var(--color-accent)]" />
        </div>

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
            </>
          )}
        </div>
      </div>
    </div>
  )
}
