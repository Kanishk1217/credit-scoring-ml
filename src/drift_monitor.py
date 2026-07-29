"""Population Stability Index (PSI) drift monitoring.

Directly motivated by the balanced-vs-natural experiment: the SAME model looks miscalibrated the
moment the population's distribution shifts (e.g. real deployment traffic drifting away from the
population it was calibrated on). PSI is the standard way to detect that shift automatically,
before it silently degrades decisions.

Usage:
    # once, after training: save the reference distribution to compare future traffic against
    uv run python src/drift_monitor.py --save-reference

    # periodically, on a batch of recent live applicants (a DataFrame with the same columns):
    from src.drift_monitor import check_drift
    report = check_drift(recent_applicants_df)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synth_data import STATIC_COLS, generate

REFERENCE_PATH = ROOT / "models" / "reference_distribution.json"

# Standard industry PSI bands (finance/credit-risk convention).
PSI_NO_SHIFT = 0.10
PSI_MODERATE_SHIFT = 0.25


def _psi_from_pct(ref_pct: np.ndarray, current: np.ndarray, bin_edges: np.ndarray) -> float:
    """PSI = sum((cur% - ref%) * ln(cur% / ref%)) over the same bins for both distributions.
    Takes the reference as PRE-COMPUTED bin percentages (not raw values) — smaller to store and
    doesn't require shipping raw historical data in the reference artifact."""
    eps = 1e-4  # avoids log(0) / divide-by-zero on empty bins
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    cur_pct = cur_counts / max(1, cur_counts.sum()) + eps
    ref_pct = ref_pct + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _verdict(psi: float) -> str:
    if psi < PSI_NO_SHIFT:
        return "stable"
    if psi < PSI_MODERATE_SHIFT:
        return "moderate_shift"
    return "major_shift_retrain"


def save_reference(n: int = 100_000, seed: int = 42) -> None:
    """Save the training population's feature distributions (bin edges + the reference
    percentage in each bin) as the baseline all future traffic gets compared against. Stores
    only the summary histogram, not raw values, so the artifact stays small and doesn't ship
    historical data."""
    df = generate(n, seed=seed)
    ref = {}
    for col in STATIC_COLS:
        values = df[col].values.astype(float)
        edges = np.quantile(values, np.linspace(0, 1, 11))   # 10 bins
        edges[0], edges[-1] = -np.inf, np.inf                # catch out-of-range future values
        counts, _ = np.histogram(values, bins=edges)
        pct = (counts / counts.sum()).tolist()
        ref[col] = {"bin_edges": edges.tolist(), "reference_pct": pct}
    REFERENCE_PATH.write_text(json.dumps(ref, indent=2))
    print(f"saved reference distribution ({n} rows, {len(STATIC_COLS)} features) -> {REFERENCE_PATH}")


def check_drift(current: pd.DataFrame) -> dict:
    """Compare a batch of recent applicants (DataFrame with STATIC_COLS) against the saved
    reference distribution. Returns a per-feature PSI report and an overall verdict."""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError("No reference distribution saved. Run save_reference() first.")
    ref = json.loads(REFERENCE_PATH.read_text())

    report = {}
    max_psi = 0.0
    for col in STATIC_COLS:
        if col not in current.columns:
            continue
        edges = np.array(ref[col]["bin_edges"])
        ref_pct = np.array(ref[col]["reference_pct"])
        psi = _psi_from_pct(ref_pct, current[col].values.astype(float), edges)
        report[col] = {"psi": round(psi, 4), "verdict": _verdict(psi)}
        max_psi = max(max_psi, psi)

    return {
        "n_current": len(current),
        "per_feature": report,
        "max_psi": round(max_psi, 4),
        "overall_verdict": _verdict(max_psi),
        "action": ("none" if max_psi < PSI_NO_SHIFT else
                  "monitor closely, consider recalibration" if max_psi < PSI_MODERATE_SHIFT else
                  "retrain/recalibrate before continuing to serve"),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-reference", action="store_true")
    args = parser.parse_args()

    if args.save_reference:
        save_reference()
    else:
        # self-test: no drift vs itself, real drift vs a shifted population
        same = generate(5_000, seed=123)
        shifted = generate(5_000, seed=456)
        shifted["monthly_income"] = shifted["monthly_income"] * 0.5    # simulate a recession
        shifted["existing_debt"] = shifted["existing_debt"] * 1.8      # simulate rising debt

        if not REFERENCE_PATH.exists():
            save_reference()

        print("\n--- drift check: a fresh, undrifted sample (should be stable) ---")
        print(json.dumps(check_drift(same), indent=2))

        print("\n--- drift check: a simulated 'recession' population (income halved, debt up) ---")
        print(json.dumps(check_drift(shifted), indent=2))
