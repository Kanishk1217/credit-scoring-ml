"""Consumer self-assessment: warm framing over the same model and decision logic as the
officer dashboard -- band names instead of raw risk, an offer preview, ranked WHY factors,
and a "what would change my outcome" advice + goal engine. No protected attributes here either.
"""
from __future__ import annotations

from api import scoring

# --- offer pricing (mirrors docs/specs/consumer-ux.md's priceOffer, run server-side only) ---
APR_BASE = 11.0
APR_SLOPE = 60.0
APR_FLOOR, APR_CAP = 11.0, 26.0
AFFORD_DTI_CEILING = 0.45
DEBT_SERVICE_MONTHS = 36  # proxy: existing_debt / 36 approximates a current monthly obligation
TENURE_MONTHS = 36


def price_offer(pd: float, threshold: float, monthly_income: float, existing_debt: float,
                tenure_months: int = TENURE_MONTHS) -> dict:
    apr = min(APR_CAP, max(APR_FLOOR, APR_BASE + APR_SLOPE * pd))
    r = apr / 100 / 12
    current_obligation = existing_debt / DEBT_SERVICE_MONTHS
    max_emi = max(0.0, AFFORD_DTI_CEILING * monthly_income - current_obligation)
    afford = max_emi * (1 - (1 + r) ** -tenure_months) / r if r > 0 else max_emi * tenure_months

    if pd <= 0.05:
        mult = 24
    elif pd <= 0.10:
        mult = 18
    elif pd <= threshold:
        mult = 12
    elif pd <= 0.30:
        mult = 6
    else:
        mult = 3
    risk_cap = mult * monthly_income

    amount = round(min(afford, risk_cap) / 5000) * 5000
    amount = max(0, int(amount))
    emi = int(round(amount * r / (1 - (1 + r) ** -tenure_months))) if amount > 0 and r > 0 else 0
    return {
        "qualifies": pd <= threshold, "secured": pd > threshold,
        "max_amount": amount, "apr": round(apr, 1),
        "tenure_months": tenure_months, "monthly_emi": emi,
    }


def band(pd: float, threshold: float) -> tuple[str, str]:
    if pd <= 0.05:
        return "thriving", "You're in great shape"
    if pd <= 0.10:
        return "steady", "Solid footing"
    if pd <= threshold:
        return "almost", "Almost at the line"
    if pd <= 0.30:
        return "building", "You're building momentum"
    return "starting", "A clear place to start"


CONSUMER_LABELS = {
    "age": "Age", "monthly_income": "Monthly income", "credit_limit": "Credit limit",
    "existing_debt": "Existing debt", "debt_to_income": "Debt vs income",
    "employment_years": "Steady employment", "num_existing_loans": "Number of loans",
    "payment_history": "Recent payment history",
}


def _consumer_detail(col: str, profile: dict) -> str:
    if col == "payment_history":
        late = sum(1 for v in profile["payment_history"][-6:] if v > 0)
        return f"{late} late month{'s' if late != 1 else ''} in the last 6"
    if col == "debt_to_income":
        dti = profile["existing_debt"] / (profile["monthly_income"] * 12 + 1)
        return f"{dti:.0%} of yearly income"
    if col == "employment_years":
        return f"{profile['employment_years']:g} years"
    if col == "num_existing_loans":
        return f"{profile['num_existing_loans']}"
    if col in ("monthly_income", "existing_debt", "credit_limit"):
        return f"₹{profile[col]:,.0f}"
    if col == "age":
        return f"{profile['age']}"
    return ""


def score_profile(cfg: dict, xgb, net, profile: dict) -> float:
    static_cols = cfg["static_cols"]
    static_row = scoring.build_static_row(
        static_cols, profile["age"], profile["monthly_income"], profile["credit_limit"],
        profile["existing_debt"], profile["employment_years"], profile["num_existing_loans"],
    )
    xgb_score = float(xgb.predict_proba(static_row)[0, 1])
    seq_std = [(float(v) - cfg["seq_mean"]) / cfg["seq_std"] for v in profile["payment_history"]]
    raw_pd = net.predict(seq_std, xgb_score)
    return scoring.calibrate(cfg, raw_pd)


def why_factors(cfg: dict, xgb, net, profile: dict, top_n: int = 4) -> list[dict]:
    static_cols = cfg["static_cols"]
    static_row = scoring.build_static_row(
        static_cols, profile["age"], profile["monthly_income"], profile["credit_limit"],
        profile["existing_debt"], profile["employment_years"], profile["num_existing_loans"],
    )
    xgb_score = float(xgb.predict_proba(static_row)[0, 1])
    seq_std = [(float(v) - cfg["seq_mean"]) / cfg["seq_std"] for v in profile["payment_history"]]
    raw_factors = scoring.explain(cfg, xgb, static_cols, static_row, xgb_score, net, seq_std, top_n=top_n)
    return [
        {
            "feature": f["col"],
            "label": CONSUMER_LABELS.get(f["col"], f["factor"]),
            "impact": round(abs(f["impact"]), 4),
            "direction": "raises" if f["direction"] == "increases_risk" else "lowers",
            "detail": _consumer_detail(f["col"], profile),
        }
        for f in raw_factors
    ]


# --- advice candidates: each mutates a copy of the profile, re-scored through the real model ---

def _ontime(profile: dict, n: int) -> dict:
    hist = profile["payment_history"]
    return {**profile, "payment_history": hist[n:] + [0] * n}


def _paydown(profile: dict, amount: float | None) -> dict:
    debt = 0.0 if amount is None else max(0.0, profile["existing_debt"] - amount)
    return {**profile, "existing_debt": debt}


CANDIDATES = [
    ("ontime3", "Make the next 3 payments on time", "habit", 3, lambda p: _ontime(p, 3)),
    ("ontime6", "Make the next 6 payments on time", "habit", 6, lambda p: _ontime(p, 6)),
    ("ontime12", "Make the next 12 payments on time", "habit", 12, lambda p: _ontime(p, 12)),
    ("paydown_25k", "Bring your debt down by ₹25,000", "money", 0, lambda p: _paydown(p, 25_000)),
    ("paydown_50k", "Bring your debt down by ₹50,000", "money", 0, lambda p: _paydown(p, 50_000)),
    ("paydown_clear", "Clear your existing debt", "money", 0, lambda p: _paydown(p, None)),
    ("raise_limit", "Ask for a higher credit limit", "habit", 2,
     lambda p: {**p, "credit_limit": p["credit_limit"] * 1.5}),
    ("close_one_loan", "Close or consolidate one existing loan", "money", 1,
     lambda p: {**p, "num_existing_loans": max(0, p["num_existing_loans"] - 1)}),
    ("tenure_plus_year", "Stay in your current job another year", "time", 12,
     lambda p: {**p, "employment_years": p["employment_years"] + 1}),
]


def _candidate_cost(cid: str, profile: dict) -> float | None:
    if cid == "paydown_25k":
        return 25_000.0
    if cid == "paydown_50k":
        return 50_000.0
    if cid == "paydown_clear":
        return profile["existing_debt"]
    return None


def build_advice(cfg: dict, xgb, net, profile: dict, pd_now: float, threshold: float,
                 top_n: int = 5) -> list[dict]:
    items = []
    for cid, title, effort, horizon, mutate in CANDIDATES:
        mutated = mutate(profile)
        if mutated == profile:
            continue
        pd_after = score_profile(cfg, xgb, net, mutated)
        delta = pd_now - pd_after
        if delta <= 1e-4:
            continue
        items.append({
            "id": cid, "title": title, "pd_before": round(pd_now, 4), "pd_after": round(pd_after, 4),
            "delta": round(delta, 4), "effort": effort, "horizon_months": horizon,
            "cost_inr": _candidate_cost(cid, profile),
            "unlocks": price_offer(pd_after, threshold, mutated["monthly_income"], mutated["existing_debt"]),
            "_mutate": mutate,
        })
    items.sort(key=lambda a: a["delta"], reverse=True)
    return items[:top_n]


def build_goal(cfg: dict, xgb, net, profile: dict, advice_items: list[dict], pd_now: float,
              threshold: float) -> dict:
    """Greedily stack the highest-delta advice items (cumulatively re-scored, not just summed)
    until the combined profile clears the threshold."""
    ranked = sorted(advice_items, key=lambda a: (-a["delta"], a["horizon_months"]))
    current = profile
    current_pd = pd_now
    steps: list[str] = []
    for item in ranked:
        if current_pd <= threshold:
            break
        candidate = item["_mutate"](current)
        candidate_pd = score_profile(cfg, xgb, net, candidate)
        if candidate_pd < current_pd:
            current, current_pd = candidate, candidate_pd
            steps.append(item["id"])
    reachable = current_pd <= threshold
    return {
        "target_pd": round(threshold, 4), "reachable": reachable, "steps": steps,
        "projected_pd": round(current_pd, 4),
        "projected_offer": price_offer(
            current_pd, threshold, current["monthly_income"], current["existing_debt"]),
    }
