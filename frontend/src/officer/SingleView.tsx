import { useReducer } from 'react'
import ApplicantForm from './ApplicantForm'
import DecisionPanel from './DecisionPanel'
import { scoreApplicant, ApiError } from './api'
import { EMPTY_APPLICANT, type ApplicantInput, type ScoreResult } from './types'

interface State {
  fields: ApplicantInput
  status: 'empty' | 'loading' | 'error' | 'result'
  result: ScoreResult | null
  error: string | null
}

type Action =
  | { type: 'edit'; value: ApplicantInput }
  | { type: 'reset' }
  | { type: 'submit' }
  | { type: 'success'; result: ScoreResult }
  | { type: 'failure'; error: string }

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'edit': return { ...state, fields: action.value }
    case 'reset': return { fields: EMPTY_APPLICANT, status: 'empty', result: null, error: null }
    case 'submit': return { ...state, status: 'loading', error: null }
    case 'success': return { ...state, status: 'result', result: action.result }
    case 'failure': return { ...state, status: 'error', error: action.error }
  }
}

function isValid(a: ApplicantInput): boolean {
  return a.age >= 18 && a.age <= 100 && a.monthly_income > 0 && a.credit_limit >= 0
    && a.existing_debt >= 0 && a.employment_years >= 0 && a.num_existing_loans >= 0
    && a.payment_history.length === 12
}

export default function SingleView() {
  const [state, dispatch] = useReducer(reducer, {
    fields: EMPTY_APPLICANT, status: 'empty', result: null, error: null,
  })

  async function submit() {
    dispatch({ type: 'submit' })
    try {
      const result = await scoreApplicant(state.fields)
      dispatch({ type: 'success', result })
    } catch (e) {
      const msg = e instanceof ApiError
        ? (e.status === 401 ? 'unauthorized (401 key)' : e.status === 422 ? `validation error (422): ${e.message}` : `error ${e.status}`)
        : 'network error'
      dispatch({ type: 'failure', error: msg })
    }
  }

  function copyJson() {
    if (state.result) void navigator.clipboard.writeText(JSON.stringify(state.result, null, 2))
  }

  return (
    <div className="grid gap-8 md:grid-cols-[1.1fr_0.9fr] md:items-start">
      <div className="md:sticky md:top-24">
        <ApplicantForm
          value={state.fields}
          onChange={(value) => dispatch({ type: 'edit', value })}
          onSubmit={submit}
          onReset={() => dispatch({ type: 'reset' })}
          disabled={state.status === 'loading'}
          isValid={isValid(state.fields)}
        />
      </div>
      <DecisionPanel
        state={state.status}
        result={state.result}
        errorMessage={state.error}
        onRetry={submit}
        onCopyJson={copyJson}
      />
    </div>
  )
}
