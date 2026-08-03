import { Fragment, useMemo, useState } from 'react'
import Papa from 'papaparse'
import type { ApplicantInput, Band, BatchRow, BatchSummary } from './types'
import { scoreBatch } from './api'
import { fmtCurrency, fmtCurrencyCompact, fmtPct } from './format'

const REQUIRED_HEADER = [
  'applicant_id', 'age', 'monthly_income', 'credit_limit', 'existing_debt',
  'employment_years', 'num_existing_loans',
  ...Array.from({ length: 12 }, (_, i) => `pay_${i + 1}`),
]
const DROPPED_COLS = ['gender', 'region']

type BatchState = 'idle' | 'dragging' | 'parsing' | 'scoring' | 'error' | 'done'

interface ParsedRow { applicant_id: string; input: ApplicantInput }
interface RowError { row: number; reason: string }

function downloadCsv(filename: string, rows: string[][]) {
  const csv = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

function parseCsvRows(rows: Record<string, string>[]): { parsed: ParsedRow[]; errors: RowError[]; droppedNotice: string | null } {
  const errors: RowError[] = []
  const parsed: ParsedRow[] = []
  const header = rows.length ? Object.keys(rows[0]) : []
  const hadDropped = DROPPED_COLS.filter((c) => header.includes(c))
  const missing = REQUIRED_HEADER.filter((c) => !header.includes(c))
  if (missing.length) {
    errors.push({ row: 0, reason: `missing columns: ${missing.join(', ')}` })
    return { parsed, errors, droppedNotice: null }
  }

  rows.forEach((r, i) => {
    const rowNum = i + 2 // 1-indexed + header row
    try {
      const num = (k: string) => {
        const v = Number(r[k])
        if (!Number.isFinite(v)) throw new Error(`"${k}" is not numeric`)
        return v
      }
      const payment_history = Array.from({ length: 12 }, (_, j) => num(`pay_${j + 1}`))
      if (payment_history.some((p) => p < -2 || p > 9)) throw new Error('pay_* out of range (-2..9)')
      const id = r.applicant_id?.trim()
      if (!id) throw new Error('missing applicant_id')
      parsed.push({
        applicant_id: id,
        input: {
          age: num('age'), monthly_income: num('monthly_income'), credit_limit: num('credit_limit'),
          existing_debt: num('existing_debt'), employment_years: num('employment_years'),
          num_existing_loans: num('num_existing_loans'), payment_history,
        },
      })
    } catch (e) {
      errors.push({ row: rowNum, reason: e instanceof Error ? e.message : 'invalid row' })
    }
  })

  return {
    parsed, errors,
    droppedNotice: hadDropped.length ? `fairness-only fields removed before scoring: ${hadDropped.join(', ')}` : null,
  }
}

const BAND_ORDER: Band[] = ['A', 'B', 'C', 'D', 'E']

function BatchSummaryTiles({ summary }: { summary: BatchSummary }) {
  const maxBand = Math.max(1, ...BAND_ORDER.map((b) => summary.band_dist[b] || 0))
  return (
    <div className="grid grid-cols-2 gap-6 border border-line p-6 md:grid-cols-5">
      <Tile label="Scored" value={`${summary.count}`} />
      <Tile label="Approve / Review / Decline" value={`${summary.approve} / ${summary.review} / ${summary.decline}`} />
      <Tile label="Avg PD" value={fmtPct(summary.avg_pd)} />
      <Tile label="Exposure" value={fmtCurrencyCompact(summary.total_offered_exposure)} />
      <div>
        <div className="text-[10px] uppercase tracking-[0.06em] text-ink/45">Band A–E</div>
        <div className="mt-2 flex h-10 items-end gap-1">
          {BAND_ORDER.map((b) => {
            const count = summary.band_dist[b] || 0
            const h = (count / maxBand) * 100
            return <div key={b} className="w-4" style={{ height: `${Math.max(4, h)}%`, background: count === maxBand ? 'var(--color-accent)' : 'var(--color-ink)' }} />
          })}
        </div>
      </div>
    </div>
  )

  function Tile({ label, value }: { label: string; value: string }) {
    return (
      <div>
        <div className="text-[10px] uppercase tracking-[0.06em] text-ink/45">{label}</div>
        <div className="mt-1 font-display text-2xl tabular-nums text-ink">{value}</div>
      </div>
    )
  }
}

function BatchTable({ rows }: { rows: BatchRow[] }) {
  const [sortDesc, setSortDesc] = useState(true)
  const [expanded, setExpanded] = useState<string | null>(null)
  const sorted = useMemo(() => [...rows].sort((a, b) => sortDesc ? b.pd - a.pd : a.pd - b.pd), [rows, sortDesc])

  return (
    <div className="mt-6 overflow-x-auto border border-line">
      <table className="w-full min-w-[720px] text-sm">
        <thead>
          <tr className="border-b border-line text-left text-ink/55">
            <th className="p-3 font-normal">ID</th>
            <th className="cursor-pointer p-3 text-right font-normal" onClick={() => setSortDesc((s) => !s)}>
              PD {sortDesc ? '↓' : '↑'}
            </th>
            <th className="p-3 font-normal">Band</th>
            <th className="p-3 font-normal">Recommendation</th>
            <th className="p-3 font-normal">Top factor</th>
            <th className="p-3 text-right font-normal">Offered</th>
            <th className="p-3 text-right font-normal">APR</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <Fragment key={r.applicant_id}>
              <tr className="cursor-pointer border-b border-line hover:bg-line/20"
                onClick={() => setExpanded(expanded === r.applicant_id ? null : r.applicant_id)}>
                <td className="p-3 text-ink">{r.applicant_id}</td>
                <td className="p-3 text-right tabular-nums text-ink">{fmtPct(r.pd)}</td>
                <td className="p-3">
                  <span className="border border-line px-1.5 py-0.5 text-xs text-ink/70">{r.band}</span>
                </td>
                <td className={`p-3 font-display ${r.verdict === 'decline' ? 'text-accent' : 'text-ink'}`}>{r.verdict}</td>
                <td className="p-3 text-ink/70">
                  {r.factors[0]?.label} {r.factors[0]?.direction === 'raises' ? '↑' : '↓'}
                </td>
                <td className="p-3 text-right tabular-nums text-ink">
                  {r.verdict === 'decline' ? '—' : fmtCurrency(r.pricing.offered_amount)}
                </td>
                <td className="p-3 text-right tabular-nums text-ink">{fmtPct(r.pricing.apr)}</td>
              </tr>
              {expanded === r.applicant_id && (
                <tr className="border-b border-line bg-line/10">
                  <td colSpan={7} className="p-4">
                    <div className="grid gap-1 text-xs text-ink/70">
                      {r.factors.map((f) => (
                        <div key={f.feature} className="flex justify-between">
                          <span>{f.label} <em className="not-italic text-ink/45">{f.value}</em></span>
                          <span className={f.direction === 'raises' ? 'text-accent' : 'text-ink/55'}>
                            {f.direction === 'raises' ? '↑' : '↓'} {(f.weightPct * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function BatchView() {
  const [state, setState] = useState<BatchState>('idle')
  const [errors, setErrors] = useState<RowError[]>([])
  const [notice, setNotice] = useState<string | null>(null)
  const [results, setResults] = useState<{ rows: BatchRow[]; summary: BatchSummary } | null>(null)
  const [parsedCount, setParsedCount] = useState(0)

  async function handleFile(file: File) {
    setState('parsing')
    setErrors([]); setNotice(null); setResults(null)
    Papa.parse<Record<string, string>>(file, {
      header: true, skipEmptyLines: true,
      complete: async (res) => {
        const { parsed, errors: rowErrors, droppedNotice } = parseCsvRows(res.data)
        setNotice(droppedNotice)
        setParsedCount(parsed.length)
        if (rowErrors.length && parsed.length === 0) {
          setErrors(rowErrors); setState('error'); return
        }
        if (parsed.length > 1000) {
          setErrors([{ row: 0, reason: 'more than 1,000 rows — split the file' }]); setState('error'); return
        }
        setErrors(rowErrors)
        await scoreRows(parsed)
      },
      error: (err) => {
        setErrors([{ row: 0, reason: err.message }]); setState('error')
      },
    })
  }

  async function scoreRows(rows: ParsedRow[]) {
    setState('scoring')
    try {
      const applicants = Object.fromEntries(rows.map((r) => [r.applicant_id, r.input]))
      const { results: batchResults, summary } = await scoreBatch(applicants)
      setResults({ rows: batchResults, summary })
      setState('done')
    } catch (e) {
      setErrors([{ row: 0, reason: e instanceof Error ? e.message : 'scoring failed' }])
      setState('error')
    }
  }

  function exportResults() {
    if (!results) return
    const header = ['applicant_id', 'pd', 'band', 'recommendation', 'top_factor', 'top_factor_direction', 'offered_amount', 'apr', 'emi']
    const rows = results.rows.map((r) => [
      r.applicant_id, r.pd, r.band, r.verdict, r.factors[0]?.label ?? '', r.factors[0]?.direction ?? '',
      r.pricing.offered_amount, r.pricing.apr, r.pricing.emi,
    ])
    downloadCsv('scoring-results.csv', [header, ...rows.map((r) => r.map(String))])
  }

  function downloadTemplate() {
    downloadCsv('template.csv', [REQUIRED_HEADER])
  }

  function downloadErrorReport() {
    downloadCsv('errors.csv', [['row', 'reason'], ...errors.map((e) => [String(e.row), e.reason])])
  }

  return (
    <div className="flex flex-col gap-6">
      <label
        onDragOver={(e) => { e.preventDefault(); setState('dragging') }}
        onDragLeave={() => setState('idle')}
        onDrop={(e) => {
          e.preventDefault()
          const file = e.dataTransfer.files[0]
          if (file) void handleFile(file)
        }}
        className={`flex cursor-pointer flex-col items-center gap-3 border-2 border-dashed p-16 text-center transition-colors ${state === 'dragging' ? 'border-accent' : 'border-line'}`}
      >
        <input type="file" accept=".csv" className="hidden" onChange={(e) => {
          const file = e.target.files?.[0]; if (file) void handleFile(file)
        }} />
        <span className="font-display text-lg text-ink">
          {state === 'dragging' ? 'Release to upload' : 'Drop CSV or browse'}
        </span>
        {state === 'parsing' && <span className="text-sm text-ink/55">Reading rows…</span>}
        {state === 'scoring' && <span className="text-sm text-ink/55">Scoring {parsedCount} applicants…</span>}
        <div className="mt-2 flex items-center gap-4 text-sm">
          <span onClick={(e) => { e.preventDefault(); downloadTemplate() }} className="text-ink/55 underline decoration-line hover:text-ink">
            Download template
          </span>
        </div>
      </label>

      {notice && <p className="text-xs text-ink/55">{notice}</p>}

      {state === 'error' && (
        <div className="border border-line p-6">
          <p className="text-ink">{errors.length} row{errors.length === 1 ? '' : 's'} could not be read.</p>
          <div className="mt-4 flex gap-4">
            <button onClick={downloadErrorReport} className="border border-line px-4 py-2 text-sm text-ink hover:border-ink">
              Download error report
            </button>
          </div>
        </div>
      )}

      {results && (
        <>
          {errors.length > 0 && (
            <p className="text-xs text-ink/55">
              {errors.length} row{errors.length === 1 ? '' : 's'} skipped — <span onClick={downloadErrorReport} className="cursor-pointer underline">download error report</span>
            </p>
          )}
          <BatchSummaryTiles summary={results.summary} />
          <BatchTable rows={results.rows} />
          <div className="flex justify-end">
            <button onClick={exportResults} className="border border-line px-4 py-2 text-sm text-ink hover:border-ink">
              Export results CSV
            </button>
          </div>
        </>
      )}
    </div>
  )
}
