# Model Report Card — as of 2026-07-29

One consolidated snapshot of everything verified about the served hybrid model
(XGBoost + LSTM + fusion, isotonic-calibrated). Sources: `models/hybrid_config.json`,
`reports/cross_validation/results.json`, `reports/demo_100_applicants/`, and the monotonicity
check run this session.

## 1. Performance (held-out test set)
| Metric | Value |
|---|---|
| AUC | 0.892 |
| Brier score (calibrated) | 0.101 |
| Actual default rate | 22.4% |
| Mean predicted PD | 22.0% (tracks actual) |

## 2. Cross-validation stability (5 folds, full pipeline retrained per fold)
| | mean | std | range |
|---|---|---|---|
| AUC | 0.8872 | 0.0024 | 0.8842 – 0.8905 |
| Brier | 0.1030 | 0.0012 | — |
| Default rate | 22.36% | 0.0016% | — |

**Conclusion:** performance is stable, not a lucky split. Class imbalance (22.4% vs 77.6%) is a
real, consistent property of the data.

## 3. Balanced vs natural evaluation (same model, same threshold, fresh 50k sample)
| | Natural (22.3%) | Balanced (50/50) |
|---|---|---|
| AUC | 0.886 | 0.885 |
| Recall | 86.2% | 86.2% (identical) |
| Precision | 48.0% | 75.9% |
| Accuracy | 76.1% (below the 77.7% naive baseline) | 79.5% (vs a 50% coin flip) |

**Conclusion:** AUC and recall are prevalence-invariant (real model skill). Precision and
calibration are prevalence-*dependent* — the same model looks "differently calibrated" purely
because the population mix changed. This is the central reason a real deployment must calibrate
to the real population's own default rate, not reuse ours.

## 4. Fairness audit (gender/region held out of scoring)
| Attribute | Demographic parity gap | Equalized odds gap |
|---|---|---|
| Gender (2 groups) | 0.000 | 0.009 |
| Region (5 groups) | 0.015 | 0.026 |

**Caveat:** near-zero by construction on synthetic data (gender/region don't causally drive risk
in the generator). Proves the audit *methodology* works; says nothing about a real population.

## 5. Confusion matrix on a live 100-applicant batch (fresh, unseen, seed=7777)
|  | actual: repaid | actual: defaulted |
|---|---|---|
| predicted repaid | 46 | 3 (missed) |
| predicted decline | 31 | 20 (caught) |

Recall 87%, precision 39%, accuracy 66% (below the 77% naive baseline — the accuracy trap, live).

## 6. Monotonicity / sanity check (does risk move the way it should?)
| Factor | Direction tested | Result |
|---|---|---|
| Existing debt ↑ | risk should ↑ | 3.5% → 6.3% → 11.0% → 25.3% ✓ |
| Monthly income ↑ | risk should ↓ | 30.8% → 9.6% → 3.5% → 1.4% ✓ |
| Employment years ↑ | risk should ↓ | 10.7% → 9.4% → 6.3% → 3.5% → 1.4% ✓ |
| Number of existing loans ↑ | risk should ↑ | 5.1% → 6.3% → 12.7% → 25.3% ✓ |
| Recent late payments ↑ | risk should ↑ | 6.3% → 13.4% → 29.5% → 70.2% (saturates) ✓ |

Every single-feature slice moves in the economically sensible direction, with no reversals.

## Bottom line
The model is internally consistent, stable, well-calibrated for its own population, and behaves
sensibly. **None of this validates it against real people** — that gap is the subject of the
world-readiness roadmap (see `docs/model_card.md` limitations section and the next learning-log
entry).
