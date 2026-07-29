import type { AssessResponse, Profile } from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function assess(profile: Profile): Promise<AssessResponse> {
  const res = await fetch('/api/self-assessment', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profile }),
  })
  if (!res.ok) {
    let detail = 'Something went wrong. Please try again.'
    try {
      const j = await res.json()
      if (res.status === 422) detail = 'A few details need a second look.'
      else if (res.status === 429) detail = "You're going a little fast — try again in a moment."
      else if (j.detail) detail = typeof j.detail === 'string' ? j.detail : detail
    } catch {
      // ignore body-parse failure, use the generic message
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<AssessResponse>
}
