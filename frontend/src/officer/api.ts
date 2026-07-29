import type { ApplicantInput, BatchRow, BatchSummary, ScoreResult } from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = `${res.status}`
    try {
      const j = await res.json()
      detail = j.detail ? JSON.stringify(j.detail) : detail
    } catch {
      // ignore body-parse failure, fall back to status code
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export function scoreApplicant(a: ApplicantInput): Promise<ScoreResult> {
  return post<ScoreResult>('/score', a)
}

export function scoreBatch(
  applicants: Record<string, ApplicantInput>,
): Promise<{ results: BatchRow[]; summary: BatchSummary }> {
  return post('/score/batch', { applicants })
}
