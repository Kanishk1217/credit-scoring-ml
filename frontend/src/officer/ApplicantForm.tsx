import type { ApplicantInput, Pay } from './types'
import { fmtPct } from './format'

interface Props {
  value: ApplicantInput
  onChange: (next: ApplicantInput) => void
  onSubmit: () => void
  onReset: () => void
  disabled: boolean
  isValid: boolean
}

function Label({ children }: { children: React.ReactNode }) {
  return <span className="font-label text-[11px] font-500 uppercase tracking-[0.06em] text-ink/55">{children}</span>
}

function NumberField({
  label, value, onChange, min, max, step = 1, suffix,
}: {
  label: string; value: number; onChange: (n: number) => void
  min?: number; max?: number; step?: number; suffix?: string
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <Label>{label}</Label>
      <div className="flex items-baseline gap-1 border-b border-line pb-1.5 focus-within:border-ink">
        <input
          type="number" value={Number.isFinite(value) ? value : ''} min={min} max={max} step={step}
          onChange={(e) => onChange(e.target.valueAsNumber || 0)}
          className="w-full bg-transparent font-display text-lg tabular-nums text-ink outline-none"
        />
        {suffix && <span className="text-sm text-ink/55">{suffix}</span>}
      </div>
    </label>
  )
}

function CurrencyField({
  label, value, onChange, hint,
}: { label: string; value: number; onChange: (n: number) => void; hint?: string }) {
  return (
    <label className="flex flex-col gap-1.5">
      <Label>{label}</Label>
      <div className="flex items-baseline gap-1.5 border-b border-line pb-1.5 focus-within:border-ink">
        <span className="font-display text-lg text-ink/55">₹</span>
        <input
          type="number" value={Number.isFinite(value) ? value : ''} min={0}
          onChange={(e) => onChange(Math.max(0, e.target.valueAsNumber || 0))}
          className="w-full bg-transparent font-display text-lg tabular-nums text-ink outline-none"
        />
      </div>
      {hint && <span className="text-xs text-ink/45">{hint}</span>}
    </label>
  )
}

function Stepper({ label, value, onChange, min = 0, max = 50 }: {
  label: string; value: number; onChange: (n: number) => void; min?: number; max?: number
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <Label>{label}</Label>
      <div className="flex items-center gap-3 border-b border-line pb-1.5">
        <button type="button" aria-label="decrease"
          onClick={() => onChange(Math.max(min, value - 1))}
          className="h-7 w-7 border border-line text-ink transition-colors hover:border-ink">−</button>
        <span className="min-w-6 text-center font-display text-lg tabular-nums text-ink">{value}</span>
        <button type="button" aria-label="increase"
          onClick={() => onChange(Math.min(max, value + 1))}
          className="h-7 w-7 border border-line text-ink transition-colors hover:border-ink">+</button>
      </div>
    </div>
  )
}

function MonthCell({ v, i, onChange }: { v: Pay; i: number; onChange: (n: number) => void }) {
  const late = Math.max(0, v)
  const t = late / 9
  const col = `color-mix(in oklab, var(--color-ink) ${100 - t * 100}%, var(--color-accent) ${t * 100}%)`
  return (
    <button
      type="button"
      onClick={(e) => onChange(e.shiftKey ? (v + 9) % 10 : (v + 1) % 10)}
      aria-label={`Month -${12 - i}, ${v <= 0 ? 'on time' : `${v} months late`}`}
      style={{ color: col, borderColor: col }}
      className="h-12 w-10 border font-display text-sm tabular-nums transition-colors"
    >
      {v <= 0 ? 'OK' : v}
    </button>
  )
}

function PaymentHistory({ value, onChange }: { value: Pay[]; onChange: (v: Pay[]) => void }) {
  return (
    <div className="flex flex-col gap-3">
      <Label>Payment history — oldest to newest</Label>
      <div className="flex flex-wrap gap-1.5">
        {value.map((v, i) => (
          <MonthCell key={i} v={v} i={i} onChange={(n) => {
            const next = [...value]; next[i] = n; onChange(next)
          }} />
        ))}
      </div>
      <p className="text-xs text-ink/45">0 on time · 1–9 months past due · click to advance, shift-click to go back</p>
    </div>
  )
}

export default function ApplicantForm({ value, onChange, onSubmit, onReset, disabled, isValid }: Props) {
  const dti = value.existing_debt / (value.monthly_income * 12 + 1)
  const set = (patch: Partial<ApplicantInput>) => onChange({ ...value, ...patch })

  return (
    <form
      onSubmit={(e) => { e.preventDefault(); onSubmit() }}
      className="flex flex-col gap-8 border border-line bg-paper p-6 md:p-8"
    >
      <div>
        <Label>Applicant</Label>
        <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-6">
          <NumberField label="Age" value={value.age} min={18} max={100}
            onChange={(age) => set({ age })} suffix="yrs" />
          <NumberField label="Employment" value={value.employment_years} min={0} max={50} step={0.5}
            onChange={(employment_years) => set({ employment_years })} suffix="yrs" />
          <CurrencyField label="Monthly income" value={value.monthly_income}
            onChange={(monthly_income) => set({ monthly_income })} />
          <CurrencyField label="Credit limit" value={value.credit_limit}
            onChange={(credit_limit) => set({ credit_limit })} hint="requested / sanctioned limit" />
          <CurrencyField label="Existing debt" value={value.existing_debt}
            onChange={(existing_debt) => set({ existing_debt })} />
          <div className="flex flex-col gap-1.5">
            <Label>Debt-to-income</Label>
            <div className={`border-b pb-1.5 font-display text-lg tabular-nums text-ink ${dti > 0.5 ? 'border-accent' : 'border-line'}`}>
              {fmtPct(dti)}
            </div>
            <span className="text-xs text-ink/45">derived — the model scores it either way</span>
          </div>
          <Stepper label="Existing loans" value={value.num_existing_loans} max={20}
            onChange={(num_existing_loans) => set({ num_existing_loans })} />
        </div>
      </div>

      <PaymentHistory value={value.payment_history} onChange={(payment_history) => set({ payment_history })} />

      <div className="flex items-center gap-4">
        <button
          type="submit" disabled={disabled || !isValid}
          className="bg-accent px-6 py-2.5 font-display text-sm font-500 text-paper transition-opacity disabled:opacity-40"
        >
          Score applicant
        </button>
        <button type="button" onClick={onReset} className="text-sm text-ink/55 transition-colors hover:text-ink">
          Reset
        </button>
      </div>
    </form>
  )
}
