export interface Profile {
  age: number
  monthly_income: number
  credit_limit: number
  existing_debt: number
  employment_years: number
  num_existing_loans: number
  payment_history: number[] // length 12, oldest -> newest, <=0 on time, 1..9 months late
}

export const EMPTY_PROFILE: Profile = {
  age: 30,
  monthly_income: 0,
  credit_limit: 0,
  existing_debt: 0,
  employment_years: 0,
  num_existing_loans: 0,
  payment_history: Array(12).fill(0),
}

export type Band = 'thriving' | 'steady' | 'almost' | 'building' | 'starting'

export interface Offer {
  qualifies: boolean
  secured: boolean
  max_amount: number
  apr: number
  tenure_months: number
  monthly_emi: number
}

export interface WhyFactor {
  feature: string
  label: string
  impact: number
  direction: 'raises' | 'lowers'
  detail: string
}

export type Effort = 'time' | 'money' | 'habit'

export interface Advice {
  id: string
  title: string
  pd_before: number
  pd_after: number
  delta: number
  effort: Effort
  horizon_months: number
  cost_inr: number | null
  unlocks: Offer
}

export interface Goal {
  target_pd: number
  reachable: boolean
  steps: string[]
  projected_pd: number
  projected_offer: Offer
}

export interface AssessResponse {
  pd: number
  band: Band
  band_headline: string
  threshold: number
  offer_now: Offer
  why: WhyFactor[]
  advice: Advice[]
  goal: Goal
  note: string | null
}

export type PageState = 'idle' | 'editing' | 'submitting' | 'results' | 'error'
