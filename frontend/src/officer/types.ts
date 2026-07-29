export type Pay = number // -2..9, <=0 on time

export interface ApplicantInput {
  age: number
  monthly_income: number
  credit_limit: number
  existing_debt: number
  employment_years: number
  num_existing_loans: number
  payment_history: Pay[] // length 12, oldest..newest
}

export type Verdict = 'approve' | 'review' | 'decline'
export type Band = 'A' | 'B' | 'C' | 'D' | 'E'

export interface Factor {
  feature: string
  label: string
  value: string
  contribution: number
  direction: 'raises' | 'lowers'
  weightPct: number
}

export interface Pricing {
  offered_amount: number
  apr: number
  emi: number
  tenor_months: number
}

export interface ScoreResult {
  pd: number
  band: Band
  verdict: Verdict
  factors: Factor[]
  pricing: Pricing
}

export interface BatchRow extends ScoreResult {
  applicant_id: string
}

export interface BatchSummary {
  count: number
  approve: number
  review: number
  decline: number
  avg_pd: number
  median_pd: number
  band_dist: Record<Band, number>
  total_offered_exposure: number
}

export const EMPTY_APPLICANT: ApplicantInput = {
  age: 30,
  monthly_income: 0,
  credit_limit: 0,
  existing_debt: 0,
  employment_years: 0,
  num_existing_loans: 0,
  payment_history: Array(12).fill(0),
}

export const APPROVE_MAX = 0.10
export const DECLINE_MIN = 0.25
export const T_STAR = 1 / 6
