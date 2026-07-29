# Consumer Self-Assessment Page — Build Spec

## 1. State machine

```
idle ──edit──▶ editing ──submit──▶ submitting ──200──▶ results
                  ▲                     │
                  └────── error ◀───────┘ (4xx/5xx/parse)
results ──"start over"──▶ idle
```

`type PageState = 'idle' | 'editing' | 'submitting' | 'results' | 'error'`

- `idle`: empty form, mode toggle visible.
- `editing`: user has touched a field; live client-side validation, submit enabled only when valid.
- `submitting`: form locked, skeleton results mount below (optimistic layout, no data).
- `results`: full results view, form collapses to a summary strip with an "Edit inputs" affordance.
- `error`: inline banner above form, inputs preserved, retry.

CSV upload and manual entry converge on the identical `Profile` payload, so results rendering is mode-agnostic.

## 2. Component tree

```
<SelfAssessmentPage>            // owns PageState + AssessResponse
  <PageHead/>                   // Fraunces H1 "Where do you stand?", italic accent on "you"
  <ModeToggle mode/onChange/>   // MANUAL | CSV  (uppercase letterspaced labels)

  {idle|editing|error}
  <AssessmentForm>
    <ManualEntry>               // mode === 'manual'
      <CurrencyField  name=monthly_income/>
      <CurrencyField  name=existing_debt/>
      <CurrencyField  name=credit_limit/>
      <StepperField   name=employment_years/>
      <StepperField   name=num_existing_loans/>
      <StepperField   name=age/>
      <PaymentHistoryGrid value=number[12] onChange/>   // 12 tap cells
      <DerivedReadout dti/>     // shows debt_to_income live, hairline chip
    </ManualEntry>
    <CsvDropzone>               // mode === 'csv'
      <FilePicker/> <RowPicker rows/> <ParsedPreview profile/>
    </CsvDropzone>
    <SubmitBar disabled=!valid/>
  </AssessmentForm>

  {error} <ErrorBanner message onRetry/>

  {submitting} <ResultsSkeleton/>

  {results}
  <ResultsView data=AssessResponse>
    <RiskReadCard/>             // band name + PD ring + encouraging line
    <OfferCompare/>            // NOW  ▸  AFTER YOUR PLAN  (two hairline cards)
    <WhyFactors items/>        // top 3-4 drivers, sign + magnitude bars
    <AdviceList items/>        // ranked improvement cards, before→after PD
    <GoalPlanner goal/>        // stacked path to approval
    <InputSummary onEdit/>     // collapsed inputs + Start over
  </ResultsView>
</SelfAssessmentPage>
```

## 3. TypeScript types

```ts
// ---- input ----
interface Profile {
  age: number;                 // 18..80
  monthly_income: number;      // INR, > 0
  credit_limit: number;        // INR, >= 0
  existing_debt: number;       // INR outstanding, >= 0
  employment_years: number;    // 0..50
  num_existing_loans: number;  // 0..20
  payment_history: number[];   // length 12, oldest→newest, each -? clamped 0..9
                               // <=0 on time, 1..9 = months late
}
// debt_to_income is derived server-side, never sent.

// ---- output ----
type Band = 'thriving' | 'steady' | 'almost' | 'building' | 'starting';

interface Offer {
  qualifies: boolean;          // pd <= threshold
  secured: boolean;            // true when offered below the approval line
  max_amount: number;          // INR, rounded to 5000
  apr: number;                 // annual %, 1 dp
  tenure_months: number;       // 36 default
  monthly_emi: number;         // INR
}

interface WhyFactor {
  feature: keyof Profile | 'debt_to_income';
  label: string;               // human copy, e.g. "Recent late payments"
  impact: number;              // absolute PD contribution, [0..1]
  direction: 'raises' | 'lowers';
  detail: string;              // "2 late months in the last 6"
}

type Effort = 'time' | 'money' | 'habit';

interface Advice {
  id: string;                  // 'ontime6' | 'paydown_50k' | ...
  title: string;               // imperative, warm: "Make the next 6 payments on time"
  pd_before: number;
  pd_after: number;
  delta: number;               // pd_before - pd_after, > 0
  effort: Effort;
  horizon_months: number;      // 0 = do now, 6 = takes 6 months
  cost_inr?: number;           // for money-effort items
  unlocks: Offer;              // offer if this single change is applied
}

interface Goal {
  target_pd: number;           // = threshold
  reachable: boolean;
  steps: string[];             // ordered advice ids that clear the line
  projected_pd: number;
  projected_offer: Offer;
}

interface AssessResponse {
  pd: number;                  // calibrated PD [0..1]
  band: Band;
  threshold: number;           // 0.1667
  offer_now: Offer;
  why: WhyFactor[];            // desc by impact, top 4
  advice: Advice[];            // desc by delta
  goal: Goal;
}
```

## 4. API contract

`POST /api/self-assessment` (Cloudflare Pages Function proxies to FastAPI, injects API key)

Request:
```json
{ "profile": {
    "age": 34, "monthly_income": 65000, "credit_limit": 200000,
    "existing_debt": 180000, "employment_years": 4, "num_existing_loans": 2,
    "payment_history": [0,0,1,0,0,2,0,0,0,0,1,0]
}}
```

Response 200:
```json
{
  "pd": 0.42, "band": "building", "threshold": 0.1667,
  "offer_now": { "qualifies": false, "secured": true, "max_amount": 120000,
                 "apr": 18.5, "tenure_months": 36, "monthly_emi": 4360 },
  "why": [
    {"feature":"payment_history","label":"Recent late payments","impact":0.14,"direction":"raises","detail":"2 late months in the last 6"},
    {"feature":"debt_to_income","label":"Debt vs income","impact":0.09,"direction":"raises","detail":"23% of yearly income"},
    {"feature":"employment_years","label":"Steady employment","impact":0.04,"direction":"lowers","detail":"4 years"}
  ],
  "advice": [
    {"id":"ontime6","title":"Make the next 6 payments on time","pd_before":0.42,"pd_after":0.28,"delta":0.14,"effort":"habit","horizon_months":6,
     "unlocks":{"qualifies":false,"secured":true,"max_amount":210000,"apr":16.2,"tenure_months":36,"monthly_emi":7420}},
    {"id":"paydown_50k","title":"Bring your debt down by ₹50,000","pd_before":0.42,"pd_after":0.33,"delta":0.09,"effort":"money","horizon_months":0,"cost_inr":50000,
     "unlocks":{"qualifies":false,"secured":true,"max_amount":180000,"apr":17.4,"tenure_months":36,"monthly_emi":6360}}
  ],
  "goal": { "target_pd":0.1667, "reachable":true, "steps":["ontime6","paydown_50k"],
            "projected_pd":0.15, "projected_offer":{"qualifies":true,"secured":false,"max_amount":520000,"apr":14.2,"tenure_months":36,"monthly_emi":17820} }
}
```

Errors: `422` validation (field-level `{loc, msg}`), `429` rate limit, `500`. Client maps to `ErrorState`.

## 5. Formulas (all tunable constants named)

Decision threshold from the cost ratio (approving a defaulter = 5x declining a good customer):
```
approve if PD*5 < (1-PD)*1  →  PD < 1/6
THRESHOLD = 0.1667
```

Offer pricing (buildable, runs identically client and server so previews are instant):
```ts
const THRESHOLD = 1/6;
const clamp = (x,a,b) => Math.min(b, Math.max(a, x));

function priceOffer(pd:number, income:number, debt:number, tenure=36): Offer {
  const apr = clamp(11 + 60*pd, 11, 26);            // % : base 11, +60 slope
  const r   = apr/100/12;                            // monthly rate
  // affordability: ceiling on new EMI
  const currentObligation = debt / 36;               // proxy monthly service
  const maxEMI = Math.max(0, 0.45*income - currentObligation);
  const afford = r > 0 ? maxEMI*(1 - Math.pow(1+r,-tenure))/r : maxEMI*tenure;
  // risk cap as multiple of MONTHLY income
  const mult = pd<=0.05?24 : pd<=0.10?18 : pd<=THRESHOLD?12 : pd<=0.30?6 : 3;
  const riskCap = mult*income;
  const amount = Math.round(Math.min(afford, riskCap)/5000)*5000;
  const emi = Math.round(amount*r/(1-Math.pow(1+r,-tenure)));
  return { qualifies: pd<=THRESHOLD, secured: pd>THRESHOLD,
           max_amount: amount, apr:+apr.toFixed(1), tenure_months:tenure, monthly_emi:emi };
}
```

Band mapping (warm, never a rejection label):
```
pd <= 0.05        thriving   "You're in great shape"
0.05 < pd <=0.10  steady     "Solid footing"
0.10 < pd <=0.167 almost     "Almost at the line"
0.167< pd <=0.30  building   "You're building momentum"
pd > 0.30         starting   "A clear place to start"
```

WHY factors: XGBoost `pred_contribs=True` gives signed per-static-feature SHAP in log-odds; sequence contribution is the counterfactual `PD(actual) - PD(all-on-time)`. Convert each static SHAP to a comparable PD-space magnitude by `impact = |PD(actual) - PD(feature at cohort baseline)|`, take top 4 by `impact`, tag `direction` by sign.

## 6. Advice engine (server)

Enumerate candidate interventions, re-score PD for each, keep positive-delta ones, sort by `delta`, then run a greedy stack for the goal until `projected_pd <= THRESHOLD`.

| id | mutation | horizon | effort |
|----|----------|---------|--------|
| `ontime3/6/12` | slide window: drop N oldest, append N zeros | N months | habit |
| `paydown_25k/50k/clear` | `existing_debt -= X` (recompute DTI) | 0 | money |
| `raise_limit` | `credit_limit *= 1.5` (lower utilization) | 1-2 | habit |
| `close_one_loan` | `num_existing_loans -= 1` | 0-1 | money |
| `tenure_plus_year` | `employment_years += 1` | 12 | time |

Each item carries `unlocks = priceOffer(pd_after, income, debt')`. Sequence items are the strongest lever here, matching the "6 on-time months: 42%→28%" example.

Goal: greedily add the highest-delta, lowest-horizon items until `PD <= THRESHOLD`; if unreachable within the candidate set, set `reachable:false` and still return the closest projected offer (secured product), never a dead end.

## 7. Results layout (editorial)

Single column, generous whitespace, hairline `#17140F` borders at 1px, off-white `#F4F1EA` ground.

`RiskReadCard`: left is a terracotta `#C15A34` PD ring (SVG stroke-dasharray, thin 3px), center label shows `100 - round(PD*100)` as a "readiness" number so higher is better (never show raw default probability as a scary figure). Right is the Fraunces band headline with one italic accent word, plus a plain-prose reassurance line.

`OfferCompare`: two hairline cards, NOW and AFTER YOUR PLAN, each showing amount (Fraunces, large), APR, EMI. The AFTER card uses the terracotta accent on the amount and a small "-{delta} APR" chip. Uppercase letterspaced captions.

`WhyFactors`: top 4, each a row with the label, a horizontal bar (terracotta = raises, near-black hairline = lowers), and the `detail` string in Inter. No jargon, no "SHAP".

`AdviceList`: ranked cards. Each card:
```
[effort tag]  Make the next 6 payments on time      −14%
42% ████████████░░░░  →  28% ████████░░░░░░░░
Unlocks up to ₹2,10,000 at 16.2%      ~6 months
```
Before/after rendered as two mini bars, the after bar visibly shorter, terracotta delta badge. Verbs are imperative and kind ("Make", "Bring down", "Keep"). Money items show `cost_inr`, time items show `horizon_months` as "~N months".

`GoalPlanner`: stacked timeline of `goal.steps` with a running PD line dropping past the threshold marker, ending in `projected_offer`. Header copy frames it as a plan, e.g. "Two moves put you over the line."

## 8. Payment history grid

12 cells, oldest to newest, left to right, month captions below. Tap cycles 0→1→2→3→0 (cap the tap-cycle at 3, long-press or a stepper for 4-9). Color: on-time is hairline outline only, lateness ramps terracotta opacity `0.25 + 0.75*min(status,9)/9`. Keyboard: arrow keys move, digit keys set. Value is `number[12]`.

```tsx
function PaymentHistoryGrid({value,onChange}:{value:number[];onChange:(v:number[])=>void}){
  const cycle=(i:number)=>{const v=[...value]; v[i]=(v[i]+1)%4; onChange(v);};
  return <div className="grid grid-cols-12 gap-2">
    {value.map((s,i)=>(
      <button key={i} onClick={()=>cycle(i)}
        className="aspect-square rounded-sm border border-[#17140F]/30"
        style={{background:s>0?`rgba(193,90,52,${0.25+0.75*Math.min(s,9)/9})`:'transparent'}}
        aria-label={`Month ${i+1}: ${s<=0?'on time':s+' late'}`}/>
    ))}
  </div>;
}
```

## 9. CSV path

Expected header (single data row; if multiple, show `RowPicker`, use selected):
```
monthly_income,existing_debt,credit_limit,employment_years,num_existing_loans,age,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12
```
Parse client-side (no lib needed), map to `Profile`, show `ParsedPreview` before submit so the user confirms. `m1..m12` map directly to `payment_history` (oldest to newest). Reject files where `income<=0` or history length != 12 with a friendly field-level message.

## 10. Validation rules

- `monthly_income > 0`, `existing_debt >= 0`, `credit_limit >= 0`, `18 <= age <= 80`, `0 <= employment_years <= 50`, `0 <= num_existing_loans <= 20`, `payment_history.length === 12`, each entry `-∞..9` clamped to `0..9` on send.
- Derived `debt_to_income` shown live in `DerivedReadout` as `existing_debt/(monthly_income*12)`, formatted as a percent, no scoring done client-side.
- Submit disabled until all required fields valid; inline messages are encouraging, not red-alert ("Add your monthly income to continue").

## 11. Short code sketch — page container

```tsx
function SelfAssessmentPage(){
  const [state,setState]=useState<PageState>('idle');
  const [profile,setProfile]=useState<Profile>(EMPTY);
  const [data,setData]=useState<AssessResponse|null>(null);
  const [err,setErr]=useState<string|null>(null);

  async function submit(){
    setState('submitting');
    try{
      const r=await fetch('/api/self-assessment',{method:'POST',
        headers:{'content-type':'application/json'},body:JSON.stringify({profile})});
      if(!r.ok) throw new Error((await r.json()).message ?? 'Something went wrong');
      setData(await r.json()); setState('results');
    }catch(e:any){ setErr(e.message); setState('error'); }
  }

  return <main className="mx-auto max-w-2xl px-6 py-16 bg-[#F4F1EA] text-[#17140F]">
    <PageHead/>
    {state!=='results' && <>
      <ModeToggle/>
      <AssessmentForm profile={profile} onChange={setProfile}
        onSubmit={submit} submitting={state==='submitting'}/>
      {state==='error' && <ErrorBanner message={err!} onRetry={submit}/>}
    </>}
    {state==='submitting' && <ResultsSkeleton/>}
    {state==='results' && data &&
      <ResultsView data={data} onEdit={()=>setState('editing')}/>}
  </main>;
}
```

## 12. Tone rules baked into copy constants

Store all user-facing strings in one `copy.ts`. Never use "denied", "rejected", "high risk", "bad". Use "readiness", "building", "your plan", "unlocks". Every results state ends with a forward action, including the `starting` band, which routes to a secured/small-ticket offer plus the first advice step rather than a wall.

Key files if you scaffold under the existing frontend: `src/pages/SelfAssessment.tsx`, `src/components/assessment/*`, `src/lib/pricing.ts` (the `priceOffer` above, shared with previews), `src/lib/copy.ts`, and the Pages Function at `functions/api/self-assessment.ts`.