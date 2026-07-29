# Risk-Based Pricing Engine

Torch-free, pure-Python. Sits after calibration, consumes calibrated PD, returns a full offer. No external deps beyond stdlib `math`.

## Constants (policy config, tune without code change)

```python
COST_RATIO              = 5.0    # cost(approve defaulter) / cost(decline good)
RECOVERY_RATE           = 0.5    # fraction of principal recovered on secured default (LGD reduction)
REVIEW_BAND_FACTOR      = 1.6    # decline cutoff = approve cutoff * this

BASE_RATE               = 11.0   # % annual, unsecured base
RISK_PREMIUM            = 40.0   # percentage points per unit PD
SECURED_BASE_DISCOUNT   = 3.0    # pp knocked off base when collateral present
RATE_FLOOR              = 10.5   # % annual
RATE_CAP                = 24.0   # % annual

DTI_EMI_CAP             = 0.40   # new EMI <= 40% of disposable income
EXISTING_DEBT_SERVICE_RATE = 0.03  # monthly obligation ~= 3% of outstanding balance
TENURE_MONTHS           = 36     # default amortization horizon
```

## Formulas

Cost-based decision thresholds. Approve when expected cost of approving beats expected cost of declining: `PD*C_eff < (1-PD)*1`, giving cutoff `1/(1+C_eff)`. Collateral cuts effective loss to `C*(1-recovery)`, so it raises the cutoff.

```
C_eff          = COST_RATIO * (1 - RECOVERY_RATE)   if has_collateral else COST_RATIO
t_approve      = 1 / (1 + C_eff)
t_decline      = min(t_approve * REVIEW_BAND_FACTOR, 1.0)

decision = "approve"  if PD <= t_approve
         = "review"   if t_approve < PD <= t_decline
         = "decline"  if PD > t_decline
```

Unsecured: t_approve = 1/6 = 0.1667, t_decline = 0.2667.
Secured:   t_approve = 1/3.5 = 0.2857, t_decline = 0.4571.

Interest rate. Collateral both discounts the base and shrinks the risk premium (recovery softens loss):

```
premium_mult = (1 - RECOVERY_RATE) if has_collateral else 1.0
base         = BASE_RATE - (SECURED_BASE_DISCOUNT if has_collateral else 0.0)
rate_pct     = clamp(base + RISK_PREMIUM * PD * premium_mult, RATE_FLOOR, RATE_CAP)
```

Affordability → max principal (standard reverse annuity):

```
existing_service = existing_debt * EXISTING_DEBT_SERVICE_RATE
disposable       = monthly_income - existing_service
max_emi          = DTI_EMI_CAP * disposable            # <= 0  => decline, max_loan = 0
r                = rate_pct / 100 / 12                  # monthly rate
n                = TENURE_MONTHS
principal        = max_emi * (1 - (1 + r)**-n) / r
max_loan_amount  = floor_to(min(principal, credit_limit), 1000)   # credit_limit is hard ceiling
```

`requested_amount` never raises the cap; it only flags shortfall in the note.

## Signature

```python
def price_loan(
    calibrated_pd: float,
    monthly_income: float,
    existing_debt: float,
    credit_limit: float,
    has_collateral: bool = False,
    requested_amount: float | None = None,
    tenure_months: int = TENURE_MONTHS,
) -> "PricingResult": ...
```

## Code sketch

```python
import math
from dataclasses import dataclass, asdict

@dataclass
class PricingResult:
    decision: str            # "approve" | "review" | "decline"
    calibrated_pd: float
    max_loan_amount: int     # INR, rounded to 1000
    interest_rate_pct: float # annual %, 2dp
    tenure_months: int
    monthly_emi: int         # EMI on max_loan_amount at offered rate
    secured: bool
    approve_cutoff_pd: float
    note: str

def _annuity_principal(emi, r, n):
    if r <= 0: return emi * n
    return emi * (1 - (1 + r) ** -n) / r

def _emi(principal, r, n):
    if r <= 0: return principal / n
    return principal * r / (1 - (1 + r) ** -n)

def price_loan(calibrated_pd, monthly_income, existing_debt, credit_limit,
               has_collateral=False, requested_amount=None, tenure_months=TENURE_MONTHS):
    pd = max(0.0, min(1.0, calibrated_pd))

    c_eff     = COST_RATIO * (1 - RECOVERY_RATE) if has_collateral else COST_RATIO
    t_approve = 1 / (1 + c_eff)
    t_decline = min(t_approve * REVIEW_BAND_FACTOR, 1.0)
    decision  = "approve" if pd <= t_approve else "review" if pd <= t_decline else "decline"

    premium_mult = (1 - RECOVERY_RATE) if has_collateral else 1.0
    base = BASE_RATE - (SECURED_BASE_DISCOUNT if has_collateral else 0.0)
    rate = max(RATE_FLOOR, min(RATE_CAP, base + RISK_PREMIUM * pd * premium_mult))

    disposable = monthly_income - existing_debt * EXISTING_DEBT_SERVICE_RATE
    max_emi    = DTI_EMI_CAP * disposable
    r, n       = rate / 1200, tenure_months

    if decision == "decline" or max_emi <= 0:
        max_loan = 0
    else:
        principal = _annuity_principal(max_emi, r, n)
        max_loan  = int(min(principal, credit_limit) // 1000 * 1000)

    emi_on_offer = int(round(_emi(max_loan, r, n))) if max_loan > 0 else 0

    parts = []
    if decision == "decline":
        parts.append(f"PD {pd:.1%} above decline cutoff {t_decline:.1%}.")
    elif decision == "review":
        parts.append(f"PD {pd:.1%} in manual-review band ({t_approve:.1%}-{t_decline:.1%}).")
    else:
        parts.append(f"Approved. PD {pd:.1%} within cutoff {t_approve:.1%}.")
    if has_collateral:
        parts.append("Collateral raised the approve cutoff and cut the rate.")
    if max_loan > 0 and credit_limit < _annuity_principal(max_emi, r, n):
        parts.append("Capped by credit limit, not affordability.")
    if requested_amount and max_loan and requested_amount > max_loan:
        parts.append(f"Requested {int(requested_amount):,} exceeds max {max_loan:,}; offer a top-up or longer tenure.")

    return PricingResult(decision, round(pd, 4), max_loan, round(rate, 2),
                         tenure_months, emi_on_offer, has_collateral,
                         round(t_approve, 4), " ".join(parts))
```

## Worked examples

**1. Low risk, unsecured.** PD 0.04, income 90000, existing_debt 100000, credit_limit 800000, requested 500000.
- decision: 0.04 <= 0.1667 → **approve**
- rate: 11 + 40*0.04 = **12.60%**
- disposable = 90000 - 3000 = 87000; max_emi = 34800; r = 0.0105, n = 36; annuity factor 29.86
- affordable principal ≈ 1,039,000 → capped by credit_limit → **max_loan 800,000**
- EMI on 800000 ≈ 26,800; requested 500000 fully covered.

**2. Medium risk, unsecured.** PD 0.14, income 55000, existing_debt 300000, credit_limit 400000, requested 350000.
- decision: 0.14 <= 0.1667 → **approve** (near cutoff)
- rate: 11 + 40*0.14 = **16.60%**
- disposable = 55000 - 9000 = 46000; max_emi = 18400; r = 0.013833; factor 28.20
- affordable ≈ 518,900 → capped by credit_limit → **max_loan 400,000**
- requested 350000 covered.

**3. High risk, rescued by collateral.** PD 0.24, income 40000, existing_debt 150000, credit_limit 300000, requested 250000, has_collateral True.
- secured cutoff = 0.2857, so 0.24 → **approve**. Unsecured, the same PD sits in the 0.1667-0.2667 review band, so collateral flips review to approve.
- rate: base 11-3=8, premium 40*0.24*0.5=4.8 → **12.80%**. Unsecured it would be 11+40*0.24 = 20.60%, so collateral saves ~7.8pp.
- disposable = 40000 - 4500 = 35500; max_emi = 14200; r = 0.010667; factor 29.77
- affordable ≈ 422,700 → capped by credit_limit → **max_loan 300,000**; requested 250000 covered.

## API response JSON

```json
{
  "decision": "approve",
  "calibrated_pd": 0.04,
  "max_loan_amount": 800000,
  "interest_rate_pct": 12.60,
  "tenure_months": 36,
  "monthly_emi": 26800,
  "secured": false,
  "approve_cutoff_pd": 0.1667,
  "note": "Approved. PD 4.0% within cutoff 16.7%. Capped by credit limit, not affordability."
}
```

## Frontend type

```ts
type Decision = "approve" | "review" | "decline";

interface PricingResult {
  decision: Decision;
  calibrated_pd: number;      // 0..1
  max_loan_amount: number;    // INR, multiple of 1000
  interest_rate_pct: number;  // annual %, 2dp
  tenure_months: number;
  monthly_emi: number;        // INR on max_loan_amount
  secured: boolean;
  approve_cutoff_pd: number;
  note: string;
}
```

Decision → accent mapping for the editorial UI: `approve` uses near-black `#17140F` text on off-white, `review` uses terracotta `#C15A34`, `decline` uses a hairline-bordered muted state. Render `note` as the italic Fraunces accent line under the offer figures.

## Notes for the implementer

- `existing_debt` is a balance, not a monthly figure, consistent with the `debt_to_income` scoring feature. The 3% service-rate assumption converts it to a monthly obligation. If real EMI data becomes available, pass it directly and drop `EXISTING_DEBT_SERVICE_RATE`.
- Thresholds are derived from `COST_RATIO`, so retuning the 5x cost assumption automatically shifts approve/decline boundaries. Keep the cost-based cutoff as the single source of truth, do not hardcode 0.1667.
- `RECOVERY_RATE` is the one lever that couples collateral to both decision and rate. Set per collateral class if you later distinguish property from vehicle.