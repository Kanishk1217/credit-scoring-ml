# Advice Engine Design

## 1. Pipeline contract

The engine never re-implements scoring. It reuses one function that takes a full feature dict and returns the calibrated PD, so perturbations flow through the exact same fusion + isotonic path as live scoring.

```python
# scoring/pipeline.py  (already exists, this is the contract the engine relies on)
def calibrated_pd(feat: Features) -> float:
    xgb_logit = booster_margin(static_vec(feat))          # XGBoost margin
    lstm_logit = lstm_numpy(seq_vec(feat.payment_sequence)) # NumPy LSTM
    raw = fuse(xgb_logit, lstm_logit)                      # fused logit -> sigmoid
    return float(np.interp(raw, KNOTS_X, KNOTS_Y))         # isotonic knots, clamped 0..1
```

`Features` is the scored contract only. gender/region are never passed in.

```python
@dataclass
class Features:
    age: int
    monthly_income: float        # INR
    credit_limit: float
    existing_debt: float
    employment_years: float
    num_existing_loans: int
    payment_sequence: list[int]  # len 12, index 0 = oldest, 11 = most recent, values clamped 0..9
    # derived, NOT stored: debt_to_income = existing_debt / (monthly_income * 12)
```

Derived feature recompute (call after any perturbation that touches income or debt):

```python
def dti(f: Features) -> float:
    return f.existing_debt / (f.monthly_income * 12.0)
```

`num_existing_loans`, `age`, `employment_years` are never perturbed by any scenario (a consumer cannot truthfully change age/tenure on demand, and closing loans is not in the requested set).

## 2. Decision threshold (cost-based)

Approve when expected cost of approving < expected cost of declining. With loss of approving a defaulter `L=5` and loss of declining a good customer `G=1`:

```
t_base = G / (G + L) = 1 / 6 = 0.16667
decision = "approve" if pd < t_base else "decline"
```

This threshold is the reference for every scenario's `flips_decision`.

## 3. Scenario catalog (exact perturbations)

| id | field(s) touched | exact op | guard (skip if) | difficulty 1..5 | effort type |
|----|------|----------|------|------|------|
| `ontime_6m` | payment_sequence | `seq[6:12] = [0]*6` (last 6 months set on-time) | last 6 already all `<=0` | 2 | time/behavior |
| `reduce_debt_25` | existing_debt (+dti) | `debt *= 0.75` | existing_debt == 0 | 3 | money |
| `reduce_debt_50` | existing_debt (+dti) | `debt *= 0.50` | existing_debt == 0 | 4 | money |
| `income_up_20` | monthly_income (+dti) | `income *= 1.20` | never skipped | 4 | money/time |
| `util_raise_limit` | credit_limit | `limit = max(limit, debt/0.30)` (target 30% utilization) | debt/limit <= 0.30 or limit==0 | 2 | request (lender-granted) |
| `util_lower_debt` | existing_debt (+dti) | `debt = min(debt, 0.30*limit)` (target 30% utilization) | debt/limit <= 0.30 | 3 | money |
| `collateral` | none (business rule) | see below | borrower has no pledgeable asset flag | 2 | structural |

Utilization (`existing_debt/credit_limit`) is a consumer-facing concept, not a scored feature. Scenarios (d) act on it through the two real features `credit_limit` and `existing_debt`. `util_lower_debt` and `reduce_debt_*` can collide (both cut debt); the engine dedups by keeping the one with the larger `delta_pd` when the resulting `existing_debt` values are within 1%.

## 4. Collateral as a business rule

Collateral is not a model feature, so PD is unchanged. A secured facility recovers a fraction `recovery` of exposure, cutting effective loss-given-default. That raises the approval threshold:

```
L_secured = L * (1 - recovery)          # recovery default 0.60 -> L_secured = 2.0
t_secured = G / (G + L_secured) = 1/3 = 0.33333
```

For `collateral`: `projected.pd == baseline.pd`, and `flips_decision = (baseline.pd >= t_base) and (baseline.pd < t_secured)`. `recovery` is configurable per product/asset class.

## 5. Ranking formula

Per scenario:

```
delta_pd      = pd_base - pd_scn                 # positive = improvement
relative_drop = delta_pd / pd_base               # 0..1
ease_factor   = 1 / difficulty                   # 0.2 .. 1.0
impact_norm   = relative_drop                    # collateral uses 0 (no PD change)
flip_bonus    = 1.0 if flips_decision else 0.0
rank_score    = 0.6*impact_norm + 0.4*ease_factor + flip_bonus
```

Sort descending by `rank_score`. Ties broken by `delta_pd` desc, then `difficulty` asc. Non-applicable scenarios are dropped before ranking. `rank` is assigned 1..n after sort. This puts decision-flipping actions on top, then trades raw PD drop against how hard the action is.

## 6. Backend module

```python
# advice/engine.py
from dataclasses import replace
from copy import deepcopy

L, G, RECOVERY = 5.0, 1.0, 0.60
T_BASE = G / (G + L)
T_SECURED = G / (G + L * (1 - RECOVERY))
W_IMPACT, W_EASE, FLIP_BONUS = 0.6, 0.4, 1.0

DIFFICULTY = {
    "ontime_6m": 2, "reduce_debt_25": 3, "reduce_debt_50": 4,
    "income_up_20": 4, "util_raise_limit": 2, "util_lower_debt": 3, "collateral": 2,
}

def _clamp_seq(seq): return [max(0, min(9, s)) for s in seq]

# each returns (perturbed Features | None, perturbation_meta). None = not applicable.
def p_ontime_6m(f):
    if all(s <= 0 for s in f.payment_sequence[6:]): return None, None
    seq = f.payment_sequence[:6] + [0]*6
    return replace(f, payment_sequence=seq), {"field": "payment_sequence", "op": "set_recent_ontime", "months": 6}

def p_reduce_debt(f, factor, oid):
    if f.existing_debt <= 0: return None, None
    return replace(f, existing_debt=f.existing_debt*factor), {"field": "existing_debt", "op": "scale", "factor": factor}

def p_income_up(f):
    return replace(f, monthly_income=f.monthly_income*1.20), {"field": "monthly_income", "op": "scale", "factor": 1.20}

def p_util_raise_limit(f):
    if f.credit_limit <= 0 or f.existing_debt/f.credit_limit <= 0.30: return None, None
    new = max(f.credit_limit, f.existing_debt/0.30)
    return replace(f, credit_limit=new), {"field": "credit_limit", "op": "set", "value": round(new)}

def p_util_lower_debt(f):
    if f.credit_limit <= 0 or f.existing_debt/f.credit_limit <= 0.30: return None, None
    new = min(f.existing_debt, 0.30*f.credit_limit)
    return replace(f, existing_debt=new), {"field": "existing_debt", "op": "set", "value": round(new)}

MODEL_SCENARIOS = [
    ("ontime_6m", "behavior", p_ontime_6m),
    ("reduce_debt_25", "money", lambda f: p_reduce_debt(f, 0.75, "reduce_debt_25")),
    ("reduce_debt_50", "money", lambda f: p_reduce_debt(f, 0.50, "reduce_debt_50")),
    ("income_up_20", "money", p_income_up),
    ("util_raise_limit", "request", p_util_raise_limit),
    ("util_lower_debt", "money", p_util_lower_debt),
]

def build_advice(f, score=calibrated_pd):
    pd_base = score(f)
    dec_base = "approve" if pd_base < T_BASE else "decline"
    out = []

    for sid, cat, fn in MODEL_SCENARIOS:
        pf, meta = fn(f)
        if pf is None: continue
        pd_scn = score(pf)                       # dti recomputed inside score()
        delta = pd_base - pd_scn
        if delta <= 1e-4: continue               # drop no-ops / regressions
        flips = pd_base >= T_BASE and pd_scn < T_BASE
        out.append(_row(sid, cat, meta, pd_base, pd_scn, delta, flips,
                        new_dec="approve" if pd_scn < T_BASE else dec_base, threshold=T_BASE))

    # collateral: business rule, PD unchanged
    flips_c = pd_base >= T_BASE and pd_base < T_SECURED
    out.append(_row("collateral", "structural",
                    {"field": None, "op": "secured_facility", "recovery": RECOVERY},
                    pd_base, pd_base, 0.0, flips_c,
                    new_dec="approve" if flips_c else dec_base, threshold=T_SECURED))

    out.sort(key=lambda r: (r["rank_score"], r["delta_pd"], -r["effort"]["difficulty"]), reverse=True)
    for i, r in enumerate(out, 1): r["rank"] = i
    return {"baseline": {"pd": pd_base, "pd_pct": round(pd_base*100,1),
                         "decision": dec_base, "threshold": T_BASE, "cost_ratio": L/G},
            "scenarios": out}

def _row(sid, cat, meta, pd_base, pd_scn, delta, flips, new_dec, threshold):
    diff = DIFFICULTY[sid]
    impact = (delta/pd_base) if pd_base > 0 else 0.0
    ease = 1.0/diff
    score = W_IMPACT*impact + W_EASE*ease + (FLIP_BONUS if flips else 0.0)
    return {
        "id": sid, "category": cat, "mechanism": "business_rule" if sid=="collateral" else "model",
        "perturbation": meta,
        "projected": {"pd": pd_scn, "pd_pct": round(pd_scn*100,1)},
        "delta_pd": round(delta,4), "delta_pp": round(delta*100,1),
        "relative_drop": round(impact,3),
        "flips_decision": flips, "new_decision": new_dec, "effective_threshold": threshold,
        "effort": {"type": _EFFORT_TYPE[sid], "difficulty": diff,
                   "months_to_effect": 6 if sid=="ontime_6m" else 0,
                   "rupees_required": 0 if sid in ("ontime_6m","util_raise_limit","collateral") else None},
        "rank_score": round(score,4),
        "phrasing": phrase(sid, pd_base, pd_scn, flips, meta),
    }
```

FastAPI endpoint:

```python
# advice/routes.py
@router.post("/api/advice", dependencies=[Depends(require_api_key)])
def advice(req: ScoreRequest) -> AdviceResponse:
    return build_advice(req.to_features())
```

`ScoreRequest` reuses the existing scoring request body, so the officer/consumer clients send the same payload they already send to `/api/score`.

## 7. JSON output shape

```json
{
  "baseline": {
    "pd": 0.2841,
    "pd_pct": 28.4,
    "decision": "decline",
    "threshold": 0.16667,
    "cost_ratio": 5.0
  },
  "scenarios": [
    {
      "id": "ontime_6m",
      "category": "behavior",
      "mechanism": "model",
      "perturbation": { "field": "payment_sequence", "op": "set_recent_ontime", "months": 6 },
      "projected": { "pd": 0.1179, "pd_pct": 11.8 },
      "delta_pd": 0.1662,
      "delta_pp": 16.6,
      "relative_drop": 0.585,
      "flips_decision": true,
      "new_decision": "approve",
      "effective_threshold": 0.16667,
      "effort": { "type": "behavior", "difficulty": 2, "months_to_effect": 6, "rupees_required": 0 },
      "rank_score": 1.551,
      "phrasing": {
        "headline": "6 on-time months lowers your risk from 28% to 12%.",
        "detail": "Make the next 6 monthly payments on time and your projected default risk drops from 28% to 12%.",
        "flips": "This alone would move you from likely declined to likely approved."
      },
      "rank": 1
    },
    {
      "id": "util_raise_limit",
      "category": "request",
      "mechanism": "model",
      "perturbation": { "field": "credit_limit", "op": "set", "value": 600000 },
      "projected": { "pd": 0.2210, "pd_pct": 22.1 },
      "delta_pd": 0.0631,
      "delta_pp": 6.3,
      "relative_drop": 0.222,
      "flips_decision": false,
      "new_decision": "decline",
      "effective_threshold": 0.16667,
      "effort": { "type": "request", "difficulty": 2, "months_to_effect": 0, "rupees_required": 0 },
      "rank_score": 0.333,
      "phrasing": {
        "headline": "A higher credit limit lowers your risk from 28% to 22%.",
        "detail": "Raising your credit limit to about 6,00,000 (or paying debt down to under 30% of your limit) cuts projected risk from 28% to 22%.",
        "flips": null
      },
      "rank": 2
    },
    {
      "id": "collateral",
      "category": "structural",
      "mechanism": "business_rule",
      "perturbation": { "field": null, "op": "secured_facility", "recovery": 0.6 },
      "projected": { "pd": 0.2841, "pd_pct": 28.4 },
      "delta_pd": 0.0,
      "delta_pp": 0.0,
      "relative_drop": 0.0,
      "flips_decision": true,
      "new_decision": "approve",
      "effective_threshold": 0.33333,
      "effort": { "type": "structural", "difficulty": 2, "months_to_effect": 0, "rupees_required": null },
      "rank_score": 1.2,
      "phrasing": {
        "headline": "Pledging collateral makes approval possible at your current risk.",
        "detail": "Your risk stays at 28%, but a secured loan lets us approve up to 33% risk instead of 17%, so your application clears.",
        "flips": "Collateral does not lower your risk, it changes the approval bar."
      },
      "rank": 3
    }
  ]
}
```

Note ranking: `ontime_6m` (flips + high impact + easy) tops, `collateral` (flips + easy but zero PD change) beats non-flipping `util_raise_limit`.

## 8. Phrasing engine

All math uses full-precision PD; display rounds `pd_pct` to nearest integer. INR formats with Indian grouping (`6,00,000`).

```python
def _pct(p): return round(p*100)

_TMPL = {
  "ontime_6m":       "6 on-time months lowers your risk from {b}% to {s}%.",
  "reduce_debt_25":  "Paying down 25% of your debt lowers your risk from {b}% to {s}%.",
  "reduce_debt_50":  "Paying down half your debt lowers your risk from {b}% to {s}%.",
  "income_up_20":    "A 20% higher monthly income lowers your risk from {b}% to {s}%.",
  "util_raise_limit":"A higher credit limit lowers your risk from {b}% to {s}%.",
  "util_lower_debt": "Keeping your balance under 30% of your limit lowers your risk from {b}% to {s}%.",
  "collateral":      "Pledging collateral makes approval possible at your current risk.",
}

def phrase(sid, pb, ps, flips, meta):
    head = _TMPL[sid].format(b=_pct(pb), s=_pct(ps))
    flip = None
    if flips and sid != "collateral":
        flip = "This alone would move you from likely declined to likely approved."
    elif flips and sid == "collateral":
        flip = "Collateral does not lower your risk, it changes the approval bar."
    return {"headline": head, "detail": _detail(sid, pb, ps, meta), "flips": flip}
```

More phrasing examples the templates emit:

- reduce_debt_50: "Paying down half your debt lowers your risk from 28% to 19%."
- income_up_20: "A 20% higher monthly income lowers your risk from 28% to 24%."
- util_lower_debt: "Keeping your balance under 30% of your limit lowers your risk from 28% to 22%."

## 9. TypeScript types

```ts
export type ScenarioId =
  | "ontime_6m" | "reduce_debt_25" | "reduce_debt_50"
  | "income_up_20" | "util_raise_limit" | "util_lower_debt" | "collateral";

export type Mechanism = "model" | "business_rule";
export type EffortType = "behavior" | "money" | "time" | "request" | "structural";

export interface Baseline {
  pd: number; pd_pct: number;
  decision: "approve" | "decline";
  threshold: number; cost_ratio: number;
}

export interface Effort {
  type: EffortType; difficulty: 1|2|3|4|5;
  months_to_effect: number; rupees_required: number | null;
}

export interface Phrasing { headline: string; detail: string; flips: string | null; }

export interface Scenario {
  id: ScenarioId; category: string; mechanism: Mechanism;
  perturbation: Record<string, unknown>;
  projected: { pd: number; pd_pct: number };
  delta_pd: number; delta_pp: number; relative_drop: number;
  flips_decision: boolean; new_decision: "approve" | "decline";
  effective_threshold: number; effort: Effort;
  rank_score: number; rank: number; phrasing: Phrasing;
}

export interface AdviceResponse { baseline: Baseline; scenarios: Scenario[]; }
```

## 10. Frontend component tree

```
<AdvicePanel data={AdviceResponse}>
  <BaselineHeader baseline />          // PD figure, decision chip, threshold caption
  <ScenarioList>
    <ScenarioCard scenario />          // one per ranked scenario
      <RankBadge rank />
      <Headline text />                // Fraunces, italic on the "from X% to Y%" clause
      <PdDeltaBar from to threshold /> // hairline track, terracotta fill, threshold tick
      <EffortChips effort />           // uppercase letterspaced labels
      <FlipBanner text />              // shown only when flips_decision
  </ScenarioList>
</AdvicePanel>
```

`ScenarioCard` sketch (editorial styling per the frontend contract):

```tsx
function ScenarioCard({ s }: { s: Scenario }) {
  return (
    <article className="border-t border-[#17140F]/15 py-8">
      <div className="flex items-baseline gap-4">
        <span className="text-xs tracking-[0.2em] uppercase text-[#C15A34]">#{s.rank}</span>
        <h3 className="font-[Fraunces] text-2xl text-[#17140F]">
          {renderHeadlineWithItalicClause(s.phrasing.headline)}
        </h3>
      </div>
      <PdDeltaBar from={s.baseline_pct ?? undefined}
                  fromPct={s.projected.pd_pct + s.delta_pp}
                  toPct={s.projected.pd_pct}
                  thresholdPct={s.effective_threshold * 100} />
      <div className="mt-4 flex gap-3">
        <Chip label="EFFORT" value={s.effort.type} />
        <Chip label="DIFFICULTY" value={"●".repeat(s.effort.difficulty)} />
        {s.effort.months_to_effect > 0 && <Chip label="TIME" value={`${s.effort.months_to_effect} MO`} />}
      </div>
      {s.flips_decision && (
        <p className="mt-4 text-[#C15A34] font-[Fraunces] italic">{s.phrasing.flips}</p>
      )}
    </article>
  );
}
```

`PdDeltaBar` renders a single hairline track (0..100%), a terracotta segment from `to` to `from` showing the drop, and a thin tick at the threshold so the consumer sees exactly where the approval line sits. No gradients or glows.

## Key formulas, one place

```
dti            = existing_debt / (monthly_income * 12)
t_base         = G / (G + L)                    = 1/6  = 0.16667
t_secured      = G / (G + L*(1 - recovery))     = 1/3  = 0.33333  (recovery=0.6)
delta_pd       = pd_base - pd_scenario
relative_drop  = delta_pd / pd_base
ease_factor    = 1 / difficulty
rank_score     = 0.6*relative_drop + 0.4*ease_factor + (1.0 if flips else 0)
ontime_6m      : seq[6:12] = [0,0,0,0,0,0]
reduce_debt_k  : existing_debt *= (1 - k)        k in {0.25, 0.50}
income_up_20   : monthly_income *= 1.20
util_raise     : credit_limit  = max(limit, existing_debt / 0.30)
util_lower     : existing_debt = min(debt, 0.30 * credit_limit)
```

Every model scenario re-enters `calibrated_pd`, so all projected PDs are isotonic-calibrated and directly comparable to the live score and the threshold. Collateral is the only non-model path and moves the threshold, not the PD.