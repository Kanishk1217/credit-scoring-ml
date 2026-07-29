# Loan-Officer Dashboard — build spec

Editorial styling throughout: `--paper #F4F1EA`, `--ink #17140F`, `--accent #C15A34`, `--line rgba(23,20,15,.14)` hairlines, Fraunces for the PD number / verdict / section display, Inter for body and inputs, uppercase letterspaced labels (`.06em`, 11px, ink at 55%). No shadows, no gradients. Generous whitespace, a single accent color used only for the verdict word, risk direction "up", and the primary action.

## Routes / shell

Single route `/officer` with an in-page tab switch (state, not URL) so a scored applicant isn't lost when peeking at batch. Optionally `/officer?tab=batch`.

```
<OfficerDashboard>              // holds tab state, API key from context
  <TopBar/>                     // wordmark "Agility", officer name, key status dot
  <TabBar/>                     // SINGLE APPLICANT | CSV BATCH
  {tab==='single' && <SingleView/>}
  {tab==='batch'  && <BatchView/>}
```

## Single view — component tree

```
<SingleView>                    // 2-col grid, 1.1fr / .9fr; stacks <900px
  <ApplicantForm>               // left column, sticky top on desktop
    <FieldGroup label="APPLICANT">
      <NumberField  id=age/>
      <CurrencyField id=monthly_income/>
      <CurrencyField id=credit_limit/>
      <CurrencyField id=existing_debt/>
      <DtiReadout/>             // derived, read-only
      <NumberField  id=employment_years step=0.5/>
      <Stepper      id=num_existing_loans/>
    </FieldGroup>
    <PaymentHistory>            // the 12-month row
      <MonthCell/> x12
      <PayLegend/>
    </PaymentHistory>
    <FormActions/>              // [Score applicant] primary, [Reset] ghost
  </ApplicantForm>

  <DecisionPanel state=empty|loading|error|result>
    <VerdictHeader/>            // PD %, band chip, verdict word
    <ThresholdGauge/>           // PD on a track with approve/decline markers
    <WhyList><WhyRow/> xN</WhyList>
    <PricingCard/>              // hidden on decline
    <ActionBar/>               // primary action + Copy JSON / Export PDF
  </DecisionPanel>
</SingleView>
```

## Exact fields and controls

| id | control | unit | validation | notes |
|---|---|---|---|---|
| `age` | number | years | int 18-100 | |
| `monthly_income` | currency | INR | int > 0 | `₹` prefix, grouped `1,20,000` |
| `credit_limit` | currency | INR | int ≥ 0 | requested/sanctioned limit; caps offered amount |
| `existing_debt` | currency | INR | int ≥ 0 | |
| `debt_to_income` | read-only | ratio | derived | `existing_debt / (monthly_income*12)`, shown as `%` |
| `employment_years` | number | years | 0-50, step 0.5 | |
| `num_existing_loans` | stepper | count | int 0-20 | `−`/`+` buttons + inline number |

`DtiReadout` recomputes live and shows a hairline warning underline when > 0.50 (does not block; the model scores it).

### Payment history control

A single row of 12 `MonthCell`s, oldest (M-12) on the left to newest (M-1) on the right, each 40x48px, hairline border, value centered in Fraunces. State machine per cell:

- click / tap: increment `0 → 1 → … → 9 → 0`
- shift-click (or long-press on touch): decrement
- `0` renders as `OK` in muted ink; `1..9` render the number, cell border and glyph shift toward accent as lateness rises (interpolate ink→accent by `v/9`, no fill).

Model contract: `<=0` = on time, `1..9` = months late. Store as `number[12]`. Default all `0`. A small `PayLegend` reads: `0 on time · 1–9 months past due · oldest → newest`.

```tsx
function MonthCell({v, i, onChange}:{v:number;i:number;onChange:(n:number)=>void}) {
  const late = Math.max(0, v);
  const t = late / 9;                         // 0..1
  const col = `color-mix(in oklab, var(--ink) ${100-t*100}%, var(--accent) ${t*100}%)`;
  return (
    <button type="button"
      onClick={e => onChange(e.shiftKey ? (v+9)%10 : (v+1)%10)}
      aria-label={`Month -${12-i}, ${v<=0?'on time':`${v} months late`}`}
      style={{color: col, borderColor: col}}
      className="h-12 w-10 border font-display tabular-nums">
      {v<=0 ? 'OK' : v}
    </button>
  );
}
```

## Types

```ts
type Pay = number; // -inf..9, <=0 on time
interface ApplicantInput {
  age: number; monthly_income: number; credit_limit: number;
  existing_debt: number; employment_years: number; num_existing_loans: number;
  payment_history: Pay[]; // length 12, oldest..newest
}
type Verdict = 'approve' | 'review' | 'decline';
type Band = 'A' | 'B' | 'C' | 'D' | 'E';
interface Factor {
  feature: string;        // "payment_history" | "debt_to_income" | ...
  label: string;          // "Payment history", "Debt-to-income"
  value: string;          // formatted display value, e.g. "0.62", "₹1.2L"
  contribution: number;   // signed, log-odds (margin) space
  direction: 'raises' | 'lowers';
  weightPct: number;      // |contribution| / Σ|contribution|, 0..1
}
interface Pricing {
  offered_amount: number; // INR, 0 on decline
  apr: number;            // annual, 0..1
  emi: number;            // INR/month at default tenor
  tenor_months: number;   // 24
}
interface ScoreResult {
  pd: number;             // calibrated 0..1
  band: Band;
  verdict: Verdict;
  factors: Factor[];      // ranked desc by |contribution|, static SHAP + seq counterfactual fused
  pricing: Pricing;
}
```

## API shapes

`POST /api/score` (single). Body = `ApplicantInput`. Response = `ScoreResult`.

```jsonc
// request
{ "age":34,"monthly_income":85000,"credit_limit":300000,"existing_debt":420000,
  "employment_years":6,"num_existing_loans":2,
  "payment_history":[0,0,0,1,0,0,2,0,0,0,1,0] }
// response
{ "pd":0.084,"band":"C","verdict":"approve",
  "factors":[
    {"feature":"debt_to_income","label":"Debt-to-income","value":"0.41","contribution":0.71,"direction":"raises","weightPct":0.34},
    {"feature":"payment_history","label":"Payment history","value":"3 late months","contribution":0.52,"direction":"raises","weightPct":0.25},
    {"feature":"monthly_income","label":"Monthly income","value":"₹85,000","contribution":-0.44,"direction":"lowers","weightPct":0.21},
    {"feature":"employment_years","label":"Employment","value":"6 yrs","contribution":-0.28,"direction":"lowers","weightPct":0.13},
    {"feature":"num_existing_loans","label":"Existing loans","value":"2","contribution":0.14,"direction":"raises","weightPct":0.07}
  ],
  "pricing":{"offered_amount":300000,"apr":0.166,"emi":14760,"tenor_months":24} }
```

`POST /api/score/batch` (multipart CSV) → streamed or full JSON array of `{ applicant_id, ...ScoreResult }` plus a `summary`. See CSV section.

Backend fuses the WHY list: static features come from `booster.predict(pred_contribs=True)` (already log-odds margin, exact); `payment_history` contribution = `logit(pd_actual) − logit(pd_all_on_time)` from the sequence counterfactual, appended into the same list so ranking by `|contribution|` is apples-to-apples. `gender`/`region` are never sent and never scored.

## Decision + pricing logic (frozen constants, cost-based)

```ts
// Cost of approving a defaulter = 5x cost of declining a good customer.
// Breakeven PD: approve if p·5G < (1−p)·G  ⇒  p < 1/6 ≈ 0.167.
export const T_STAR = 1/6;          // 0.167 cost breakeven (marker only)
export const APPROVE_MAX = 0.10;    // clear approve
export const DECLINE_MIN = 0.25;    // clear decline
// (0.10, 0.25] straddles the 0.167 breakeven → manual review absorbs uncertainty

export function verdict(pd:number):Verdict {
  if (pd <= APPROVE_MAX) return 'approve';
  if (pd >  DECLINE_MIN) return 'decline';
  return 'review';
}
// Risk grade, independent of the action bands:
export function band(pd:number):Band {
  return pd<=0.03?'A':pd<=0.07?'B':pd<=0.15?'C':pd<=0.30?'D':'E';
}

const DTI_CEILING = 0.50, LGD = 0.60, BASE_APR = 0.11, APR_CAP = 0.36, TENOR = 24;
export function price(pd:number, a:ApplicantInput):Pricing {
  if (verdict(pd)==='decline') return {offered_amount:0,apr:0,emi:0,tenor_months:TENOR};
  const annual = a.monthly_income*12;
  const capacity = Math.max(0, annual*DTI_CEILING - a.existing_debt);
  const riskFactor = Math.max(0, 1 - pd/DECLINE_MIN);          // shrinks sanction as PD rises
  let offered = Math.min(a.credit_limit, capacity*riskFactor);
  if (verdict(pd)==='review') offered *= 0.5;                  // provisional on review
  offered = Math.max(0, Math.round(offered/10000)*10000);      // round to ₹10k
  const apr = clamp(BASE_APR + LGD*pd/Math.max(1-pd,0.01), BASE_APR, APR_CAP);
  const r = apr/12;
  const emi = offered>0 ? Math.round(offered*r*Math.pow(1+r,TENOR)/(Math.pow(1+r,TENOR)-1)) : 0;
  return {offered_amount:offered, apr, emi, tenor_months:TENOR};
}
```

Client mirrors these so the panel can render instantly from `pd`, but backend is source of truth and returns `verdict`/`band`/`pricing` in the response.

## Decision panel design

`VerdictHeader`: PD as a large Fraunces number (`8.4%`, tabular-nums, 64px), a small band chip to its right (`BAND C`, hairline outline), and beneath it the verdict word in Fraunces italic accent, 32px: *Approve* / *Review* / *Decline*. Review renders ink, not accent; decline renders ink with a hairline strike-through rule under it.

`ThresholdGauge`: one horizontal hairline track 0→0.40 (clamp display). Two tick marks at `APPROVE_MAX` and `DECLINE_MIN` labeled tiny (`10%`, `25%`), a faint dotted tick at `T_STAR` labeled `breakeven 16.7%`, and a single terracotta dot at `pd`. Zones are not filled; only labels `APPROVE · REVIEW · DECLINE` sit under their segments in muted uppercase.

`WhyList`: ranked rows, top 5. Each `WhyRow`:
- left: `label` (Inter medium) + `value` (muted).
- a thin magnitude bar, width = `weightPct`, right-aligned, terracotta if `raises`, ink-30% if `lowers`.
- a direction glyph: `↑` accent for raises risk, `↓` muted for lowers.

```tsx
<div className="flex items-center gap-3 py-2 border-b border-[var(--line)]">
  <span className="w-40">{f.label}<em className="ml-2 opacity-55">{f.value}</em></span>
  <div className="flex-1 h-px bg-[var(--line)] relative">
    <div className="absolute right-0 top-1/2 -translate-y-1/2 h-[3px]"
         style={{width:`${f.weightPct*100}%`,
                 background:f.direction==='raises'?'var(--accent)':'rgba(23,20,15,.3)'}}/>
  </div>
  <span className={f.direction==='raises'?'text-[var(--accent)]':'opacity-55'}>
    {f.direction==='raises'?'↑':'↓'}
  </span>
</div>
```

`PricingCard` (hidden on decline; header "PROVISIONAL OFFER" on review): three columns, each a tiny uppercase label over a Fraunces figure: `OFFERED ₹3,00,000` · `APR 16.6%` · `EMI ₹14,760 / 24 mo`. Footnote: `Amount capped by requested limit and 50% DTI ceiling.`

`ActionBar`, primary changes by verdict:
- approve → `Approve & generate offer` (accent fill, paper text)
- review → `Send to review queue`
- decline → `Decline` + a required reason select (`DTI too high`, `Payment history`, `Insufficient tenure`, `Other`)

Secondary always: `Copy result JSON`, `Export PDF`, `Rescore`.

### Single-view states

- empty (no score yet): panel shows a hairline-framed placeholder, Fraunces italic `Enter applicant details and score.` plus the gauge in a dimmed, unmarked state.
- loading: replace PD number with a shimmer bar (no spinner glow), verdict word `Scoring…`, disable form Score button, keep inputs editable-disabled.
- error: hairline panel, ink text `Could not score this applicant.` + machine reason (`network`, `422 validation`, `401 key`), a `Retry` ghost button. 422 also highlights the offending field inline.

## CSV batch view

```
<BatchView state=idle|dragging|parsing|scoring|error|done>
  <CsvDropzone/>                 // + "Download template" link, + column help
  {done && <>
    <BatchSummary/>              // stat tiles row
    <BatchToolbar/>              // filter by verdict, search id, sort
    <BatchTable/>                // virtualized if >200 rows
    <BatchActions/>              // Export results CSV, Clear
  </>}
</BatchView>
```

### CSV input columns

Required header (exact): `applicant_id,age,monthly_income,credit_limit,existing_debt,employment_years,num_existing_loans,pay_1,pay_2,pay_3,pay_4,pay_5,pay_6,pay_7,pay_8,pay_9,pay_10,pay_11,pay_12`

`pay_1` = oldest (M-12), `pay_12` = newest (M-1), each an integer `<=0..9`. `applicant_id` is any string, unique. `gender`/`region` columns, if present, are dropped client-side before upload with a notice (`fairness-only fields removed before scoring`). Validation before send: header match, row count > 0 and ≤ 5,000, every numeric parses, `pay_*` in range. Bad rows collected into an error report, not silently skipped.

### Results table columns

`id` · `PD` (right-aligned %, tabular) · `Band` (chip) · `Recommendation` (verdict word, accent only for decline to draw the eye) · `Top factor` (label + `↑/↓`) · `Offered` (₹, blank on decline) · `APR`. Row click expands an inline drawer showing the full ranked WHY list and pricing for that applicant (reuses `WhyList`/`PricingCard`). Sort by any column; default sort PD desc so risky files surface first.

```ts
interface BatchRow extends ScoreResult { applicant_id: string; }
interface BatchSummaryStats {
  count: number;
  approve: number; review: number; decline: number;
  avg_pd: number; median_pd: number;
  band_dist: Record<Band, number>;
  total_offered_exposure: number; // Σ offered_amount over approve+review
}
```

### BatchSummary tiles

Row of hairline-separated tiles, Fraunces figures: `142 SCORED` · `88 APPROVE / 31 REVIEW / 23 DECLINE` · `AVG PD 11.3%` · `EXPOSURE ₹2.4Cr` · a tiny 5-cell A-E band histogram (bars in ink, tallest gets a terracotta cap). No pie charts.

### Batch states

- idle: dropzone with dashed hairline (only place a dashed border is allowed), `Drop CSV or browse`, template + column list beneath.
- dragging: border switches to solid accent, label `Release to upload`.
- parsing: `Reading 142 rows…` progress as a determinate hairline bar.
- scoring: same bar advancing as batch responses stream; partial table can render as rows arrive.
- error: red-free, ink message `12 rows could not be read.` with a `Download error report` (CSV of `row,reason`) and a `Score valid rows anyway` button.
- done: summary + table + `Export results CSV`.

### Export results CSV columns

`applicant_id,pd,band,recommendation,top_factor,top_factor_direction,offered_amount,apr,emi`

## Data flow notes

Form state in `useReducer` (`{fields, payment_history, status, result, error}`). `Score` disabled until all seven statics valid. Debounce nothing on submit (explicit action). Batch uses `papaparse` in a worker for parse, then posts the cleaned rows; render table incrementally from a streamed NDJSON response if the host supports it, else one JSON payload. All currency formatting via `Intl.NumberFormat('en-IN',{style:'currency',currency:'INR',maximumFractionDigits:0})`; large figures abbreviate to `L`/`Cr` in tiles only, exact in table and offer.