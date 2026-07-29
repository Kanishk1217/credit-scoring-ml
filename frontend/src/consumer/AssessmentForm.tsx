import { useState } from 'react'
import type { Profile } from './types'
import { COPY } from './copy'
import { fmtPct } from '../officer/format'

interface Props {
  value: Profile
  onChange: (next: Profile) => void
  onSubmit: () => void
  submitting: boolean
  isValid: boolean
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <span className="font-label text-[11px] font-500 uppercase tracking-[0.06em] text-ink/55">{children}</span>
}

function CurrencyField({ label, value, onChange }: { label: string; value: number; onChange: (n: number) => void }) {
  return (
    <label className="flex flex-col gap-1.5">
      <FieldLabel>{label}</FieldLabel>
      <div className="flex items-baseline gap-1.5 border-b border-line pb-1.5 focus-within:border-ink">
        <span className="font-display text-lg text-ink/55">₹</span>
        <input
          type="number" value={Number.isFinite(value) && value ? value : ''} min={0}
          onChange={(e) => onChange(Math.max(0, e.target.valueAsNumber || 0))}
          className="w-full bg-transparent font-display text-lg tabular-nums text-ink outline-none"
        />
      </div>
    </label>
  )
}

function Stepper({ label, value, onChange, min = 0, max = 50, step = 1, suffix }: {
  label: string; value: number; onChange: (n: number) => void
  min?: number; max?: number; step?: number; suffix?: string
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <FieldLabel>{label}</FieldLabel>
      <div className="flex items-center gap-3 border-b border-line pb-1.5">
        <button type="button" aria-label={`decrease ${label}`}
          onClick={() => onChange(Math.max(min, +(value - step).toFixed(1)))}
          className="h-7 w-7 border border-line text-ink transition-colors hover:border-ink">−</button>
        <span className="min-w-10 text-center font-display text-lg tabular-nums text-ink">{value}</span>
        <button type="button" aria-label={`increase ${label}`}
          onClick={() => onChange(Math.min(max, +(value + step).toFixed(1)))}
          className="h-7 w-7 border border-line text-ink transition-colors hover:border-ink">+</button>
        {suffix && <span className="text-sm text-ink/55">{suffix}</span>}
      </div>
    </div>
  )
}

function PaymentHistoryGrid({ value, onChange }: { value: number[]; onChange: (v: number[]) => void }) {
  const cycle = (i: number, back: boolean) => {
    const v = [...value]
    v[i] = back ? (v[i] + 3) % 4 : (v[i] + 1) % 4
    onChange(v)
  }
  return (
    <div className="flex flex-col gap-3">
      <FieldLabel>Your last 12 months of payments</FieldLabel>
      <div className="grid grid-cols-12 gap-1.5">
        {value.map((s, i) => (
          <button
            key={i} type="button"
            onClick={(e) => cycle(i, e.shiftKey)}
            aria-label={`Month ${i + 1}: ${s <= 0 ? 'on time' : `${s} late`}`}
            className="aspect-square rounded-sm border border-ink/30 text-[10px] text-ink/60 transition-colors"
            style={{ background: s > 0 ? `rgba(193,90,52,${0.25 + 0.75 * Math.min(s, 9) / 9})` : 'transparent' }}
          >
            {s > 0 ? s : ''}
          </button>
        ))}
      </div>
      <p className="text-xs text-ink/45">Tap a month if a payment was late — tap again for more, shift-tap to go back. Oldest on the left.</p>
    </div>
  )
}

export default function AssessmentForm({ value, onChange, onSubmit, submitting, isValid }: Props) {
  const [mode, setMode] = useState<'manual' | 'csv'>('manual')
  const dti = value.existing_debt / (value.monthly_income * 12 + 1)
  const set = (patch: Partial<Profile>) => onChange({ ...value, ...patch })

  return (
    <form
      onSubmit={(e) => { e.preventDefault(); onSubmit() }}
      className="flex flex-col gap-8 border border-line bg-paper p-6 md:p-8"
    >
      <div className="flex gap-6 border-b border-line pb-4">
        {(['manual', 'csv'] as const).map((m) => (
          <button key={m} type="button" onClick={() => setMode(m)}
            className={`font-label text-xs font-500 uppercase tracking-[0.1em] ${mode === m ? 'text-ink' : 'text-ink/40'}`}>
            {m === 'manual' ? 'Enter manually' : 'Upload CSV'}
          </button>
        ))}
      </div>

      {mode === 'manual' ? (
        <>
          <div className="grid grid-cols-2 gap-x-6 gap-y-6">
            <CurrencyField label="Monthly income" value={value.monthly_income}
              onChange={(monthly_income) => set({ monthly_income })} />
            <CurrencyField label="Existing debt" value={value.existing_debt}
              onChange={(existing_debt) => set({ existing_debt })} />
            <CurrencyField label="Credit limit" value={value.credit_limit}
              onChange={(credit_limit) => set({ credit_limit })} />
            <Stepper label="Employment" value={value.employment_years} max={50} step={0.5} suffix="yrs"
              onChange={(employment_years) => set({ employment_years })} />
            <Stepper label="Existing loans" value={value.num_existing_loans} max={20}
              onChange={(num_existing_loans) => set({ num_existing_loans })} />
            <Stepper label="Age" value={value.age} min={18} max={80}
              onChange={(age) => set({ age })} />
          </div>
          <div>
            <FieldLabel>Debt vs income</FieldLabel>
            <div className="mt-1.5 inline-block border border-line px-2 py-1 font-display text-sm text-ink">
              {fmtPct(dti)}
            </div>
          </div>
          <PaymentHistoryGrid value={value.payment_history} onChange={(payment_history) => set({ payment_history })} />
        </>
      ) : (
        <CsvUpload onParsed={(p) => onChange(p)} />
      )}

      <button
        type="submit" disabled={submitting || !isValid}
        className="bg-accent px-6 py-3 font-display text-sm font-500 text-paper transition-opacity disabled:opacity-40"
      >
        {submitting ? COPY.submitting : COPY.submit}
      </button>
    </form>
  )
}

const CSV_HEADER = ['monthly_income', 'existing_debt', 'credit_limit', 'employment_years',
  'num_existing_loans', 'age', ...Array.from({ length: 12 }, (_, i) => `m${i + 1}`)]

function CsvUpload({ onParsed }: { onParsed: (p: Profile) => void }) {
  const [error, setError] = useState<string | null>(null)
  const [preview, setPreview] = useState<Profile | null>(null)

  function handleFile(file: File) {
    setError(null)
    const reader = new FileReader()
    reader.onload = () => {
      const text = String(reader.result ?? '')
      const lines = text.trim().split(/\r?\n/).filter(Boolean)
      if (lines.length < 2) { setError('Add a header row and one data row.'); return }
      const header = lines[0].split(',').map((h) => h.trim())
      const missing = CSV_HEADER.filter((c) => !header.includes(c))
      if (missing.length) { setError(`Missing columns: ${missing.join(', ')}`); return }
      const row = lines[1].split(',').map((v) => v.trim())
      const get = (col: string) => Number(row[header.indexOf(col)])
      const payment_history = Array.from({ length: 12 }, (_, i) => get(`m${i + 1}`))
      const income = get('monthly_income')
      if (!(income > 0) || payment_history.some((v) => !Number.isFinite(v))) {
        setError('Check that income is positive and all 12 months are numbers.')
        return
      }
      const profile: Profile = {
        age: get('age'), monthly_income: income, credit_limit: get('credit_limit'),
        existing_debt: get('existing_debt'), employment_years: get('employment_years'),
        num_existing_loans: get('num_existing_loans'), payment_history,
      }
      setPreview(profile)
      onParsed(profile)
    }
    reader.readAsText(file)
  }

  return (
    <div className="flex flex-col gap-4">
      <label className="flex cursor-pointer flex-col items-center gap-2 border-2 border-dashed border-line p-10 text-center">
        <input type="file" accept=".csv" className="hidden" onChange={(e) => {
          const file = e.target.files?.[0]; if (file) handleFile(file)
        }} />
        <span className="font-display text-base text-ink">Drop your CSV or browse</span>
        <span className="text-xs text-ink/45">{CSV_HEADER.join(',')}</span>
      </label>
      {error && <p className="text-sm text-ink">{error}</p>}
      {preview && (
        <div className="border border-line p-4 text-sm text-ink/70">
          <p className="mb-2 font-display text-ink">We read this from your file:</p>
          <p>Income ₹{preview.monthly_income.toLocaleString('en-IN')} · Debt ₹{preview.existing_debt.toLocaleString('en-IN')} ·
            {' '}{preview.payment_history.filter((v) => v > 0).length} late month(s) in the last 12</p>
        </div>
      )}
    </div>
  )
}
